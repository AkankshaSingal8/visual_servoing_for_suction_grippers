#!/usr/bin/env python3
import time
import sys
import threading
import os
import tempfile


import cv2


import numpy as np

THIRD_PARTY_ROOT = os.path.join(os.path.dirname(__file__), "third-party")

PROMPT = "yellow box"

GDINO_CONFIG  = os.path.join(THIRD_PARTY_ROOT, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GDINO_WEIGHTS = os.path.join(THIRD_PARTY_ROOT, "GroundingDINO/weights/groundingdino_swint_ogc.pth")
SAM2_CONFIG   = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT     = os.path.join(THIRD_PARTY_ROOT, "sam2/checkpoints/sam2.1_hiera_large.pt")
DA_CKPT_DIR   = os.path.join(THIRD_PARTY_ROOT, "Depth-Anything-V2/checkpoints")
DA_ENCODER    = "vits"

ROBOT_IP     = "192.168.1.241"
VS_GAIN      = 0.08
VS_APPROACH  = 3.0
VS_DEAD_ZONE = 15
MAX_YZ_STEP  = 12.0
MAX_JUMP_PX  = 80
VS_SPEED     = 150
VS_MVACC     = 500
VS_RATE      = 0.3
CTRL_GAIN    = 0.5

CAL_DELTA    = 8.0
CAL_WAIT     = 1.5

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch  = None
    DEVICE = "cpu"


def compute_quality_score(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    score = np.zeros(depth.shape[:2], dtype=np.float32)
    m = mask > 0
    if not m.any():
        return score

    d = depth.astype(np.float32)
    d_min   = float(np.nanmin(d[m]))
    d_range = float(np.nanmax(d[m])) - d_min + 1e-6
    d_norm  = (d - d_min) / d_range

    gx       = cv2.Sobel(d_norm, cv2.CV_32F, 1, 0, ksize=5)
    gy       = cv2.Sobel(d_norm, cv2.CV_32F, 0, 1, ksize=5)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    k         = np.ones((11, 11), np.float32) / 121.0
    mean_d    = cv2.filter2D(d_norm,      -1, k)
    mean_d2   = cv2.filter2D(d_norm ** 2, -1, k)
    local_std = np.sqrt(np.maximum(mean_d2 - mean_d ** 2, 0.0))

    g_scale = float(np.percentile(grad_mag[m], 75)) + 1e-6
    s_scale = float(np.percentile(local_std[m], 75)) + 1e-6

    combined = (np.exp(-2.0 * grad_mag  / g_scale) *
                np.exp(-2.0 * local_std / s_scale)).astype(np.float32)
    combined[~m] = 0.0
    return combined


def find_best_region(score_map: np.ndarray,
                     threshold: float = 0.45,
                     min_area:  int   = 300):
    binary = (score_map > threshold).astype(np.uint8)
    kern   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kern)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    best_lbl, best_area = -1, min_area
    for lbl in range(1, n):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area > best_area:
            best_area, best_lbl = area, lbl

    if best_lbl == -1:
        return None, None, None

    region_mask = (labels == best_lbl).astype(np.uint8) * 255
    cxy  = (int(centroids[best_lbl][0]), int(centroids[best_lbl][1]))
    cnts, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cxy, (cnts[0] if cnts else None), region_mask


_gdino_model  = None
_gdino_failed = False
_sam2_pred    = None
_depth_model  = None


def _patch_bert_compat():
    try:
        import torch as _t
        from transformers import BertModel
        from transformers.modeling_utils import PreTrainedModel

        if not hasattr(PreTrainedModel, "get_head_mask"):
            def _ghm(self, head_mask, n, chunked=False):
                if head_mask is not None:
                    hm = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    hm = hm.expand(n, -1, -1, -1, -1)
                    if chunked:
                        hm = hm.unsqueeze(-1)
                    return hm
                return [None] * n
            PreTrainedModel.get_head_mask = _ghm

        _orig_geam = getattr(PreTrainedModel, "get_extended_attention_mask")
        if not getattr(_orig_geam, "_gdino_patched", False):
            def _geam(self, attention_mask, input_shape, dev_or_dtype=None, dtype=None):
                if isinstance(dev_or_dtype, _t.device):
                    dev_or_dtype = None
                try:
                    return _orig_geam(self, attention_mask, input_shape, dev_or_dtype)
                except TypeError:
                    return _orig_geam(self, attention_mask, input_shape)
            _geam._gdino_patched = True
            PreTrainedModel.get_extended_attention_mask = _geam
            BertModel.get_extended_attention_mask = _geam

        print("BERT compat patch applied")
    except Exception as e:
        print("BERT compat patch failed:", e)
        import traceback; traceback.print_exc()


def _get_gdino():
    global _gdino_model, _gdino_failed
    if _gdino_failed:
        return None
    if _gdino_model is None:
        try:
            _patch_bert_compat()
            sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, "GroundingDINO"))
            from groundingdino.util.inference import load_model
            _gdino_model = load_model(GDINO_CONFIG, GDINO_WEIGHTS, device=DEVICE)
        except Exception as e:
            print("GroundingDINO load failed (won't retry):", e)
            _gdino_failed = True
    return _gdino_model


def _get_sam2():
    global _sam2_pred
    if _sam2_pred is None:
        sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, "sam2"))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model      = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=DEVICE)
        _sam2_pred = SAM2ImagePredictor(model)
    return _sam2_pred


def _get_depth_model():
    global _depth_model
    if _depth_model is None:
        try:
            sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, "Depth-Anything-V2"))
            from depth_anything_v2.dpt import DepthAnythingV2
            cfgs = {
                "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192,  384]},
                "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384,  768]},
                "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
                "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536,1536,1536, 1536]},
            }
            ckpt = os.path.join(DA_CKPT_DIR, f"depth_anything_v2_{DA_ENCODER}.pth")
            if not os.path.exists(ckpt):
                print(f"Depth-Anything-V2 checkpoint not found: {ckpt}")
                return None
            import torch as _t
            m = DepthAnythingV2(**cfgs[DA_ENCODER])
            m.load_state_dict(_t.load(ckpt, map_location="cpu"))
            _depth_model = m.to(DEVICE).eval()
        except Exception as e:
            print("Depth-Anything-V2 load failed:", e)
    return _depth_model


def run_pipeline(image_bgr: np.ndarray, prompt: str) -> dict:
    res = dict(mask_np=None, depth_np=None, score_map=None,
               best_centroid=None, best_contour=None,
               region_mask=None, gdino_box=None, mean_score=0.0)

    gdino  = _get_gdino()
    box_np = None
    if gdino is not None:
        try:
            sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, "GroundingDINO"))
            from groundingdino.util.inference import load_image, predict as gd_predict
            from torchvision.ops import box_convert
            import torch as _t

            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            cv2.imwrite(tmp.name, image_bgr)
            tmp.close()
            try:
                _, img_t = load_image(tmp.name)
                boxes, logits, phrases = gd_predict(
                    gdino, img_t, caption=prompt,
                    box_threshold=0.3, text_threshold=0.25, device=DEVICE)
                if boxes is not None and len(boxes) > 0:
                    h, w   = image_bgr.shape[:2]
                    bxyxy  = (box_convert(boxes, "cxcywh", "xyxy")
                              * _t.tensor([w, h, w, h], dtype=_t.float32))
                    box_np = bxyxy[0].numpy()
                    res["gdino_box"] = box_np
                    print(f"GDINO: '{phrases[0]}' @ {box_np.astype(int)}")
                else:
                    print("GDINO: no boxes")
            finally:
                os.unlink(tmp.name)
        except Exception as e:
            import traceback
            print("GroundingDINO failed:", e)
            traceback.print_exc()

    if box_np is not None:
        try:
            pred = _get_sam2()
            rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pred.set_image(rgb)
            masks, _, _ = pred.predict(box=box_np, multimask_output=False)
            if masks is not None and len(masks) > 0:
                res["mask_np"] = (masks[0] > 0).astype(np.uint8) * 255
                print("SAM2: mask obtained")
        except Exception as e:
            print("SAM2 failed:", e)

    dm = _get_depth_model()
    if dm is not None:
        try:
            res["depth_np"] = dm.infer_image(image_bgr).astype(np.float32)
        except Exception as e:
            print("Depth inference failed:", e)

    if res["depth_np"] is not None and res["mask_np"] is not None:
        score_map = compute_quality_score(res["depth_np"], res["mask_np"])
        best_cxy, best_cnt, reg_mask = find_best_region(score_map)
        res.update(score_map=score_map, best_centroid=best_cxy,
                   best_contour=best_cnt, region_mask=reg_mask)
        masked = score_map[res["mask_np"] > 0]
        res["mean_score"] = float(masked.mean()) if len(masked) else 0.0
        print(f"Quality mean={res['mean_score']:.3f}  best_centroid={best_cxy}")

    if res["best_centroid"] is None and res["gdino_box"] is not None:
        x1, y1, x2, y2 = res["gdino_box"]
        res["best_centroid"] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        print(f"Servo target: GDino box centre {res['best_centroid']} (quality fallback)")

    return res


def _measure_flow(frame_a: np.ndarray, frame_b: np.ndarray):
    ga = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(ga, maxCorners=300, qualityLevel=0.01,
                                  minDistance=8, blockSize=7)
    if pts is None or len(pts) < 10:
        return None
    pts_next, status, _ = cv2.calcOpticalFlowPyrLK(ga, gb, pts, None)
    ok = status.ravel() == 1
    if ok.sum() < 10:
        return None
    flow = pts_next[ok] - pts[ok]
    dx = float(np.median(flow[:, 0, 0]))
    dy = float(np.median(flow[:, 0, 1]))
    return dx, dy


class RobotController:
    def __init__(self, ip: str):
        self.ip               = ip
        self._arm             = None
        self._lock            = threading.Lock()
        self.enabled          = False
        self._last_t          = 0.0
        self._jac_yz          = None
        self._jac_yz_inv      = None
        self.cal_status       = "uncalibrated"
        self._last_centroid   = None
        self._last_centroid_t = 0.0

    def _get_pos(self):
        ret = self._arm.get_position()
        if isinstance(ret, (list, tuple)) and len(ret) >= 2 and ret[0] == 0:
            return list(ret[1])
        print("get_position failed:", ret)
        return None

    def _move_abs(self, pos, wait=True):
        self._arm.set_position(
            x=pos[0], y=pos[1], z=pos[2],
            roll=pos[3], pitch=pos[4], yaw=pos[5],
            speed=VS_SPEED, mvacc=VS_MVACC, wait=wait)

    def connect(self) -> bool:
        try:
            from xarm.wrapper import XArmAPI
            arm = XArmAPI(self.ip, baud_checkset=False)
            time.sleep(0.5)
            if not arm.connected:
                print(f"Robot not connected: {self.ip}")
                return False
            arm.clean_error()
            arm.clean_warn()
            arm.motion_enable(True)
            arm.set_mode(0)
            arm.set_state(0)
            time.sleep(0.5)
            self._arm = arm
            self.enabled = False
            self.cal_status = "waiting for calibration..."
            print(f"Robot connected: {self.ip}")
            return True
        except Exception as e:
            print(f"Robot connect failed: {e}")
            return False

    def calibrate(self, get_frame_fn):
        self.cal_status = "waiting for frame..."
        print("\n=== Calibration: waiting for camera frame ===")

        deadline = time.time() + 30.0
        while time.time() < deadline:
            if get_frame_fn() is not None:
                break
            time.sleep(0.2)
        else:
            print("Calibration skipped: no camera frame available.")
            self.cal_status = "skipped (no frame)"
            self.enabled = True
            return

        pos0 = self._get_pos()
        if pos0 is None:
            print("Calibration skipped: cannot read robot position.")
            self.cal_status = "skipped (pos read fail)"
            self.enabled = True
            return

        print(f"Home: [{pos0[0]:.1f}, {pos0[1]:.1f}, {pos0[2]:.1f}] mm")

        J_yz = np.zeros((2, 2), dtype=np.float64)

        for col, (robot_idx, ax) in enumerate([(1, "Y"), (2, "Z")]):
            self.cal_status = f"calibrating {ax}..."

            time.sleep(0.3)
            frame_before = get_frame_fn()
            if frame_before is None:
                print(f"  [{ax}] no frame before move — skipping axis")
                continue

            print(f"  [{ax}] moving +{CAL_DELTA:.0f} mm...")
            fwd = list(pos0)
            fwd[robot_idx] += CAL_DELTA
            self._move_abs(fwd, wait=True)
            time.sleep(CAL_WAIT)

            frame_after = get_frame_fn()

            print(f"  [{ax}] returning home...")
            self._move_abs(pos0, wait=True)

            if frame_after is None:
                print(f"  [{ax}] no frame after move — skipping axis")
                continue

            flow = _measure_flow(frame_before, frame_after)
            if flow is None:
                print(f"  [{ax}] optical flow failed (too few trackable points) — skipping axis")
                continue

            dpx, dpy = flow
            J_yz[0, col] = dpx / CAL_DELTA
            J_yz[1, col] = dpy / CAL_DELTA
            print(f"  [{ax}] optical flow: ({dpx:+.1f}, {dpy:+.1f}) px / {CAL_DELTA:.0f} mm"
                  f"  → J_yz[:,{col}]=[{J_yz[0,col]:+.4f}, {J_yz[1,col]:+.4f}]")

        self._jac_yz = J_yz
        print(f"\n  J_yz (px/mm):\n{J_yz}")

        rank = np.linalg.matrix_rank(J_yz)
        if rank == 0:
            print("  WARNING: J_yz rank=0 — no usable calibration. Approach-only mode.")
            self.cal_status = "skipped (no flow data)"
            self.enabled = True
            return

        self._jac_yz_inv = np.linalg.pinv(J_yz)
        print(f"  J_yz_inv (mm/px):\n{self._jac_yz_inv}")
        status = "calibrated" if rank == 2 else f"partial cal (rank={rank})"
        print(f"=== Calibration complete [{status}] — servo enabled ===\n")
        self.cal_status = status
        self.enabled = True

    def stop(self):
        self.enabled = False
        if self._arm is not None:
            try:
                self._arm.emergency_stop()
                print("Robot stopped.")
            except Exception as e:
                print("Robot stop failed:", e)

    def servo_step(self, centroid: tuple, image_shape: tuple):
        now = time.time()
        if self._arm is None:
            if now - self._last_t > 5.0:
                print("Servo: arm not connected")
                self._last_t = now
            return
        if not self.enabled:
            if now - self._last_t > 5.0:
                print(f"Servo: waiting [{self.cal_status}]")
                self._last_t = now
            return
        if now - self._last_t < VS_RATE:
            return

        if now - self._last_centroid_t > 1.0:
            self._last_centroid = None
        if self._last_centroid is not None:
            jump = np.hypot(centroid[0] - self._last_centroid[0],
                            centroid[1] - self._last_centroid[1])
            if jump > MAX_JUMP_PX:
                print(f"Servo: centroid jump {jump:.0f} px — ignored (likely mis-detection)")
                return
        self._last_centroid   = centroid
        self._last_centroid_t = now

        self._last_t = now

        h, w       = image_shape[:2]
        ic_x, ic_y = w // 2, h // 2
        ex    = centroid[0] - ic_x
        ey    = centroid[1] - ic_y
        err_r = float(np.hypot(ex, ey))

        dx_mm = VS_APPROACH

        if err_r <= VS_DEAD_ZONE:
            dy_mm, dz_mm = 0.0, 0.0
        elif self._jac_yz_inv is not None:
            err  = np.array([ex, ey], dtype=np.float64)
            yz   = -CTRL_GAIN * (self._jac_yz_inv @ err)
            dy_mm = float(np.clip(yz[0], -MAX_YZ_STEP, MAX_YZ_STEP))
            dz_mm = float(np.clip(yz[1], -MAX_YZ_STEP, MAX_YZ_STEP))
        else:
            dy_mm = 0.0
            dz_mm = 0.0

        with self._lock:
            try:
                pos = self._get_pos()
                if pos is None:
                    return
                self._arm.set_position(
                    x=pos[0] + dx_mm, y=pos[1] + dy_mm, z=pos[2] + dz_mm,
                    roll=pos[3], pitch=pos[4], yaw=pos[5],
                    speed=VS_SPEED, mvacc=VS_MVACC, wait=True)
                print(f"Servo: err=({ex:+.0f},{ey:+.0f})px r={err_r:.0f}  "
                      f"Δ=({dx_mm:+.1f},{dy_mm:+.1f},{dz_mm:+.1f})mm"
                      f"  [{self.cal_status}]")
            except Exception as e:
                print("Servo step failed:", e)


class CameraStreamer(threading.Thread):
    def __init__(self, cam_index: int, stop_event: threading.Event,
                 robot: RobotController):
        super().__init__(daemon=True)
        self.cam_index  = cam_index
        self.stop_event = stop_event
        self.robot      = robot

        self._latest_left  = None
        self._frame_lock   = threading.Lock()
        self.data_lock     = threading.Lock()
        self._result: dict = {}
        self._models_ready = threading.Event()

    def _get_centroid(self):
        with self.data_lock:
            return self._result.get("best_centroid")

    def _get_frame(self):
        with self._frame_lock:
            if self._latest_left is not None:
                return self._latest_left.copy()
            return None

    def _run_calibration(self):
        print("Calibration thread: waiting for models to finish loading...")
        self._models_ready.wait()
        print("Calibration thread: models ready, starting Y/Z Jacobian calibration.")
        self.robot.calibrate(self._get_frame)

    def _seg_loop(self):
        while not self.stop_event.is_set():
            with self._frame_lock:
                frame = self._latest_left.copy() if self._latest_left is not None else None
            if frame is None:
                time.sleep(0.1)
                continue
            try:
                res = run_pipeline(frame, PROMPT)
                with self.data_lock:
                    self._result = res
                if not self._models_ready.is_set():
                    print("Models ready — calibration may now proceed.")
                    self._models_ready.set()
                if self.robot is not None and res.get("best_centroid") is not None:
                    self.robot.servo_step(res["best_centroid"], frame.shape)
            except Exception as e:
                print("Pipeline error:", e)
                time.sleep(1)

    def run(self):
        if cv2 is None:
            print("OpenCV not available.")
            return
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_ANY)
        if not cap.isOpened():
            print(f"Failed to open camera {self.cam_index}")
            return

        threading.Thread(target=self._seg_loop, daemon=True).start()
        if self.robot is not None and self.robot._arm is not None:
            threading.Thread(target=self._run_calibration, daemon=True).start()

        win = "Surface quality  |  [v] servo  [q] quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        video_writer = None
        video_path   = None

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                h, w = frame.shape[:2]
                left = frame[:, : w // 2].copy()

                with self._frame_lock:
                    self._latest_left = left.copy()
                with self.data_lock:
                    res = dict(self._result)

                rendered = self._render(left, res)

                if video_writer is None:
                    rh, rw = rendered.shape[:2]
                    video_path = time.strftime("vs_recording_%Y%m%d_%H%M%S.mp4")
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        30.0, (rw, rh))
                    print(f"Recording to {video_path}")

                video_writer.write(rendered)
                cv2.imshow(win, rendered)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.stop_event.set()
                    break
                elif key == ord("v") and self.robot is not None:
                    self.robot.enabled = not self.robot.enabled
                    print("Servo:", "ON" if self.robot.enabled else "OFF")
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
                print(f"Recording saved: {video_path}")
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _render(self, left: np.ndarray, res: dict) -> np.ndarray:
        display = left.copy()
        h, w    = display.shape[:2]

        mask_np       = res.get("mask_np")
        depth_np      = res.get("depth_np")
        score_map     = res.get("score_map")
        best_centroid = res.get("best_centroid")
        best_contour  = res.get("best_contour")
        gdino_box     = res.get("gdino_box")
        mean_score    = res.get("mean_score", 0.0)

        def _resize_to(arr, interp=cv2.INTER_NEAREST):
            if arr is not None and arr.shape[:2] != (h, w):
                return cv2.resize(arr, (w, h), interpolation=interp)
            return arr

        mask_np_r   = _resize_to(mask_np)
        score_map_r = _resize_to(score_map, cv2.INTER_LINEAR) if score_map is not None else None

        if mask_np_r is not None:
            overlay          = np.zeros_like(display)
            overlay[:, :, 1] = mask_np_r
            display = cv2.addWeighted(display, 0.72, overlay, 0.28, 0)

        if score_map_r is not None and mask_np_r is not None:
            heat  = cv2.applyColorMap(
                (np.clip(score_map_r, 0, 1) * 255).astype(np.uint8),
                cv2.COLORMAP_JET)
            m3    = (mask_np_r[:, :, None] > 0).astype(np.float32)
            display = (display * (1 - m3 * 0.6) + heat * m3 * 0.6).astype(np.uint8)

        if best_contour is not None:
            cv2.drawContours(display, [best_contour], -1, (255, 255, 255), 2)

        if gdino_box is not None:
            x1, y1, x2, y2 = gdino_box.astype(int)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)

        ic = (w // 2, h // 2)
        cv2.drawMarker(display, ic, (255, 80, 0), cv2.MARKER_CROSS, 22, 2)

        if best_centroid is not None:
            bcp = best_centroid
            cv2.drawMarker(display, bcp, (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
            cv2.arrowedLine(display, ic, bcp, (0, 220, 255), 2, tipLength=0.12)
            ex = bcp[0] - ic[0]
            ey = bcp[1] - ic[1]
            cv2.putText(display, f"err ({ex:+d},{ey:+d}) px",
                        (8, h - 58), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1)

        if score_map_r is not None and mask_np_r is not None:
            masked = score_map_r[mask_np_r > 0]
            if len(masked):
                max_s = float(masked.max())
                cv2.putText(display,
                            f"quality  mean={mean_score:.2f}  max={max_s:.2f}",
                            (8, h - 33), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

        if self.robot is not None:
            cal = self.robot.cal_status
            if not self.robot.enabled:
                txt = f"SERVO OFF  [{cal}]"
                col = (80, 80, 80)
            else:
                txt = f"SERVO ON  [{cal}]"
                col = (0, 255, 80)
            cv2.putText(display, txt, (8, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)

        if depth_np is not None:
            try:
                d = depth_np.astype(np.float32)
                d_mn, d_mx = float(np.nanmin(d)), float(np.nanmax(d))
                d_u8  = ((d - d_mn) / (d_mx - d_mn + 1e-6) * 255).astype(np.uint8)
                cmap  = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
                iw    = max(100, w // 4)
                ih    = int(iw * h / w)
                inset = cv2.resize(cmap, (iw, ih))
                display[10:10 + ih, w - iw - 10: w - 10] = inset
                cv2.putText(display, "depth", (w - iw - 8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            except Exception:
                pass

        return display


if __name__ == "__main__":
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    robot = RobotController(ROBOT_IP)
    robot.connect()

    stop_ev    = threading.Event()
    cam_thread = CameraStreamer(cam_index, stop_ev, robot)
    cam_thread.start()

    try:
        cam_thread.join()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        robot.stop()
        stop_ev.set()
        cam_thread.join(timeout=2)
        print("Done.")
