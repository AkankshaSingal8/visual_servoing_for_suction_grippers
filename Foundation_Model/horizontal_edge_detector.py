#!/usr/bin/env python3
"""
SemVS Pipeline — Grasp the top box from a stack.

Detection  → GDino / OWLv2 (open-vocab bounding box)
Splitting  → Canny + HoughLinesP find horizontal edges inside the bbox
             to locate boundaries between stacked boxes
Segment    → SAM2 with point prompts (foreground on top box,
             negative on the rest) + multimask_output=True
Grasp pt   → Mask centroid via cv2.moments (bbox-centre fallback)
Servo      → Jacobian-based IBVS on xArm
"""
import time
import sys
import threading
import os
import tempfile

import cv2
import numpy as np

THIRD_PARTY_ROOT = os.path.join(os.path.dirname(__file__), "third-party")

PROMPT = "brown box"

# ── Detector selection: "gdino" or "owlv2" ──────────────────────────
DETECTOR = os.environ.get("DETECTOR", "gdino").lower()

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

# ── Edge-split tuning ───────────────────────────────────────────────
EDGE_MIN_LINE_FRAC  = 0.35   # min line length as fraction of bbox width
EDGE_MERGE_PX       = 20     # merge lines within this many pixels
EDGE_MIN_BOX_HEIGHT = 30     # discard sub-regions shorter than this (px)

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch  = None
    DEVICE = "cpu"


# ═════════════════════════════════════════════════════════════════════
#  Utilities
# ═════════════════════════════════════════════════════════════════════

def _mask_centroid(mask: np.ndarray):
    """Return (cx, cy) from binary mask using cv2.moments, or None."""
    M = cv2.moments(mask, binaryImage=True)
    if M["m00"] < 1.0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


# ═════════════════════════════════════════════════════════════════════
#  Horizontal edge detection — find split lines between stacked boxes
# ═════════════════════════════════════════════════════════════════════

def _find_horizontal_splits(image_bgr: np.ndarray, box_xyxy: np.ndarray):
    """
    Within the bounding box, detect horizontal edges that separate
    stacked boxes.  Returns a sorted list of y-coordinates (in full-
    image space) where splits occur, NOT including the bbox top/bottom.
    """
    x1, y1, x2, y2 = box_xyxy.astype(int)
    h_img, w_img = image_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    roi = image_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    roi_h, roi_w = roi.shape[:2]
    min_len = int(roi_w * EDGE_MIN_LINE_FRAC)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # --- Method A: Canny + HoughLinesP ---
    edges = cv2.Canny(gray, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=40,
                            minLineLength=min_len,
                            maxLineGap=15)

    candidates_a = []
    if lines is not None:
        for ln in lines:
            lx1, ly1, lx2, ly2 = ln[0]
            # near-horizontal: angle < 10°
            if abs(ly2 - ly1) < 0.18 * abs(lx2 - lx1 + 1e-6):
                candidates_a.append((ly1 + ly2) / 2.0)

    # --- Method B: horizontal projection profile (Sobel-Y) ---
    sob_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    profile = np.abs(sob_y).mean(axis=1)  # 1-D signal, one value per row
    # smooth and find peaks
    kern = np.ones(7) / 7.0
    profile_smooth = np.convolve(profile, kern, mode="same")
    threshold = float(np.percentile(profile_smooth, 85))
    candidates_b = []
    in_peak = False
    peak_start = 0
    for row_idx in range(len(profile_smooth)):
        if profile_smooth[row_idx] > threshold:
            if not in_peak:
                peak_start = row_idx
                in_peak = True
        else:
            if in_peak:
                candidates_b.append((peak_start + row_idx) / 2.0)
                in_peak = False

    # --- Merge both sets of candidates ---
    all_candidates = sorted(candidates_a + candidates_b)
    if not all_candidates:
        return []

    # cluster nearby values
    merged = [all_candidates[0]]
    for c in all_candidates[1:]:
        if c - merged[-1] < EDGE_MERGE_PX:
            merged[-1] = (merged[-1] + c) / 2.0   # average
        else:
            merged.append(c)

    # filter: must be interior (not too close to top/bottom of roi)
    margin = max(EDGE_MIN_BOX_HEIGHT, roi_h * 0.08)
    splits = [y1 + int(m) for m in merged
              if margin < m < roi_h - margin]
    return sorted(splits)


def _top_box_region(box_xyxy: np.ndarray, splits: list):
    """
    Given the full-stack bbox and the interior split y-coordinates,
    return (top_y, bottom_y, cx) for the topmost sub-region.
    """
    x1, y1, x2, y2 = box_xyxy.astype(int)
    cx = int((x1 + x2) / 2)

    if not splits:
        # no splits found → treat top ~33% of bbox as the top box
        third = int((y2 - y1) / 3)
        return y1, y1 + third, cx

    first_split = splits[0]
    return y1, first_split, cx


# ═════════════════════════════════════════════════════════════════════
#  Model loaders (unchanged)
# ═════════════════════════════════════════════════════════════════════

_gdino_model     = None
_gdino_failed    = False
_owlv2_processor = None
_owlv2_model     = None
_owlv2_failed    = False
_sam2_pred       = None
_depth_model     = None


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


def _get_owlv2():
    global _owlv2_processor, _owlv2_model, _owlv2_failed
    if _owlv2_failed:
        return None, None
    if _owlv2_model is None:
        try:
            from transformers import Owlv2Processor, Owlv2ForObjectDetection
            _owlv2_processor = Owlv2Processor.from_pretrained(
                "google/owlv2-base-patch16-ensemble")
            _owlv2_model = Owlv2ForObjectDetection.from_pretrained(
                "google/owlv2-base-patch16-ensemble").to(DEVICE).eval()
            print("OWLv2 loaded")
        except Exception as e:
            print("OWLv2 load failed (won't retry):", e)
            _owlv2_failed = True
            return None, None
    return _owlv2_processor, _owlv2_model


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


# ═════════════════════════════════════════════════════════════════════
#  Detectors
# ═════════════════════════════════════════════════════════════════════

def _detect_gdino(image_bgr, prompt):
    gdino = _get_gdino()
    if gdino is None:
        return []
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
            if boxes is None or len(boxes) == 0:
                return []
            h, w = image_bgr.shape[:2]
            bxyxy = (box_convert(boxes, "cxcywh", "xyxy")
                     * _t.tensor([w, h, w, h], dtype=_t.float32)).numpy()
            return [(bxyxy[i], float(logits[i]), phrases[i])
                    for i in range(len(boxes))]
        finally:
            os.unlink(tmp.name)
    except Exception as e:
        import traceback; print("GDINO failed:", e); traceback.print_exc()
        return []


def _detect_owlv2(image_bgr, prompt):
    processor, model = _get_owlv2()
    if model is None:
        return []
    try:
        import torch as _t
        from PIL import Image
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        texts = [[prompt]]
        inputs = processor(text=texts, images=pil, return_tensors="pt").to(DEVICE)
        with _t.no_grad():
            outputs = model(**inputs)
        h, w = image_bgr.shape[:2]
        target_sizes = _t.tensor([[h, w]], device=DEVICE)
        if hasattr(processor, "post_process_grounded_object_detection"):
            results = processor.post_process_grounded_object_detection(
                outputs, target_sizes=target_sizes,
                threshold=0.1, text_labels=texts)[0]
        else:
            results = processor.post_process_object_detection(
                outputs, threshold=0.1, target_sizes=target_sizes)[0]
        boxes  = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        if len(boxes) == 0:
            return []
        return [(boxes[i], float(scores[i]), prompt) for i in range(len(boxes))]
    except Exception as e:
        import traceback; print("OWLv2 failed:", e); traceback.print_exc()
        return []


def _pick_topmost(dets):
    if not dets:
        return None
    return min(dets, key=lambda d: d[0][1])


# ═════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═════════════════════════════════════════════════════════════════════

def run_pipeline(image_bgr: np.ndarray, prompt: str) -> dict:
    res = dict(mask_np=None, depth_np=None,
               best_centroid=None, gdino_box=None,
               split_lines=[], top_region=None)

    # ── Step 1: Detect bounding box ──────────────────────────────────
    dets = (_detect_owlv2 if DETECTOR == "owlv2" else _detect_gdino)(image_bgr, prompt)
    pick = _pick_topmost(dets)
    box_np = None
    if pick is not None:
        box_np = pick[0]
        res["gdino_box"] = box_np
        print(f"{DETECTOR.upper()}: '{pick[2]}' score={pick[1]:.2f} "
              f"@ {box_np.astype(int)}  (topmost of {len(dets)})")

    if box_np is None:
        return res

    # ── Step 2: Find horizontal split lines inside the bbox ──────────
    splits = _find_horizontal_splits(image_bgr, box_np)
    res["split_lines"] = splits
    top_y, bot_y, cx = _top_box_region(box_np, splits)
    res["top_region"] = (top_y, bot_y)
    print(f"Edge splits: {len(splits)} line(s) → top box rows [{top_y}..{bot_y}]")

    # ── Step 3: SAM2 with point prompts ──────────────────────────────
    try:
        pred = _get_sam2()
        rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pred.set_image(rgb)

        # Foreground point: centre of the top sub-region
        fg_y = int((top_y + bot_y) / 2)
        fg_pt = np.array([[cx, fg_y]])
        labels = np.array([1])  # 1 = foreground

        # Negative points: centre of bbox, and bottom quarter
        x1, y1, x2, y2 = box_np.astype(int)
        neg_pts = []
        bbox_cy = int((y1 + y2) / 2)
        if bbox_cy > bot_y + EDGE_MIN_BOX_HEIGHT:
            neg_pts.append([cx, bbox_cy])
        bbox_bot_q = int(y1 + 0.75 * (y2 - y1))
        if bbox_bot_q > bot_y + EDGE_MIN_BOX_HEIGHT:
            neg_pts.append([cx, bbox_bot_q])

        if neg_pts:
            all_pts = np.vstack([fg_pt, np.array(neg_pts)])
            all_labels = np.array([1] + [0] * len(neg_pts))
        else:
            all_pts = fg_pt
            all_labels = labels

        masks, scores, _ = pred.predict(
            point_coords=all_pts,
            point_labels=all_labels,
            multimask_output=True)    # 3 masks at sub-part / part / whole

        if masks is not None and len(masks) > 0:
            # Pick the smallest mask with confidence > 0.7
            # (smallest = tightest around the top box)
            valid = [(i, masks[i].sum(), scores[i])
                     for i in range(len(masks)) if scores[i] > 0.7]
            if valid:
                best_i = min(valid, key=lambda t: t[1])[0]
            else:
                best_i = int(np.argmax(scores))
            res["mask_np"] = (masks[best_i] > 0).astype(np.uint8) * 255
            print(f"SAM2: mask {best_i} (score={scores[best_i]:.2f}, "
                  f"area={masks[best_i].sum()}) from {len(masks)} candidates")
    except Exception as e:
        print("SAM2 failed:", e)

    # ── Step 4: Depth (for inset visualisation) ──────────────────────
    dm = _get_depth_model()
    if dm is not None:
        try:
            res["depth_np"] = dm.infer_image(image_bgr).astype(np.float32)
        except Exception as e:
            print("Depth inference failed:", e)

    # ── Step 5: Grasp point — mask centroid → bbox-top fallback ──────
    if res["mask_np"] is not None:
        mc = _mask_centroid(res["mask_np"])
        if mc is not None:
            res["best_centroid"] = mc
            print(f"Grasp point: mask centroid {mc}")
        else:
            res["best_centroid"] = (cx, fg_y)
            print(f"Grasp point: top-region centre ({cx}, {fg_y}) (moments failed)")
    else:
        # no mask at all — use geometric top-region centre
        res["best_centroid"] = (cx, int((top_y + bot_y) / 2))
        print(f"Grasp point: top-region centre (no SAM2 mask)")

    return res


# ═════════════════════════════════════════════════════════════════════
#  Optical flow (unchanged)
# ═════════════════════════════════════════════════════════════════════

def _measure_flow(frame_a, frame_b):
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
    return float(np.median(flow[:, 0, 0])), float(np.median(flow[:, 0, 1]))


# ═════════════════════════════════════════════════════════════════════
#  Robot controller (unchanged)
# ═════════════════════════════════════════════════════════════════════

class RobotController:
    def __init__(self, ip):
        self.ip = ip; self._arm = None; self._lock = threading.Lock()
        self.enabled = False; self._last_t = 0.0
        self._jac_yz = None; self._jac_yz_inv = None
        self.cal_status = "uncalibrated"
        self._last_centroid = None; self._last_centroid_t = 0.0

    def _get_pos(self):
        ret = self._arm.get_position()
        if isinstance(ret, (list, tuple)) and len(ret) >= 2 and ret[0] == 0:
            return list(ret[1])
        print("get_position failed:", ret); return None

    def _move_abs(self, pos, wait=True):
        self._arm.set_position(x=pos[0], y=pos[1], z=pos[2],
                               roll=pos[3], pitch=pos[4], yaw=pos[5],
                               speed=VS_SPEED, mvacc=VS_MVACC, wait=wait)

    def connect(self):
        try:
            from xarm.wrapper import XArmAPI
            arm = XArmAPI(self.ip, baud_checkset=False); time.sleep(0.5)
            if not arm.connected:
                print(f"Robot not connected: {self.ip}"); return False
            arm.clean_error(); arm.clean_warn(); arm.motion_enable(True)
            arm.set_mode(0); arm.set_state(0); time.sleep(0.5)
            self._arm = arm; self.enabled = False
            self.cal_status = "waiting for calibration..."
            print(f"Robot connected: {self.ip}"); return True
        except Exception as e:
            print(f"Robot connect failed: {e}"); return False

    def calibrate(self, get_frame_fn):
        self.cal_status = "waiting for frame..."
        print("\n=== Calibration: waiting for camera frame ===")
        deadline = time.time() + 30.0
        while time.time() < deadline:
            if get_frame_fn() is not None: break
            time.sleep(0.2)
        else:
            print("Calibration skipped: no camera frame.")
            self.cal_status = "skipped (no frame)"; self.enabled = True; return
        pos0 = self._get_pos()
        if pos0 is None:
            print("Calibration skipped: pos read fail.")
            self.cal_status = "skipped (pos read fail)"; self.enabled = True; return
        print(f"Home: [{pos0[0]:.1f}, {pos0[1]:.1f}, {pos0[2]:.1f}] mm")
        J_yz = np.zeros((2, 2), dtype=np.float64)
        for col, (ri, ax) in enumerate([(1, "Y"), (2, "Z")]):
            self.cal_status = f"calibrating {ax}..."
            time.sleep(0.3)
            fb = get_frame_fn()
            if fb is None: print(f"  [{ax}] no frame before"); continue
            print(f"  [{ax}] moving +{CAL_DELTA:.0f} mm...")
            fwd = list(pos0); fwd[ri] += CAL_DELTA
            self._move_abs(fwd, wait=True); time.sleep(CAL_WAIT)
            fa = get_frame_fn()
            print(f"  [{ax}] returning home..."); self._move_abs(pos0, wait=True)
            if fa is None: print(f"  [{ax}] no frame after"); continue
            flow = _measure_flow(fb, fa)
            if flow is None: print(f"  [{ax}] flow failed"); continue
            dpx, dpy = flow
            J_yz[0, col] = dpx / CAL_DELTA; J_yz[1, col] = dpy / CAL_DELTA
            print(f"  [{ax}] flow: ({dpx:+.1f},{dpy:+.1f}) px/{CAL_DELTA:.0f}mm "
                  f"→ J[:,{col}]=[{J_yz[0,col]:+.4f},{J_yz[1,col]:+.4f}]")
        self._jac_yz = J_yz; print(f"\n  J_yz:\n{J_yz}")
        rank = np.linalg.matrix_rank(J_yz)
        if rank == 0:
            print("  WARNING: rank=0"); self.cal_status = "skipped (no flow)"
            self.enabled = True; return
        self._jac_yz_inv = np.linalg.pinv(J_yz)
        print(f"  J_inv:\n{self._jac_yz_inv}")
        status = "calibrated" if rank == 2 else f"partial (rank={rank})"
        print(f"=== Calibration [{status}] ===\n")
        self.cal_status = status; self.enabled = True

    def stop(self):
        self.enabled = False
        if self._arm:
            try: self._arm.emergency_stop(); print("Robot stopped.")
            except Exception as e: print("Stop failed:", e)

    def servo_step(self, centroid, image_shape):
        now = time.time()
        if self._arm is None:
            if now - self._last_t > 5.0: print("Servo: no arm"); self._last_t = now
            return
        if not self.enabled:
            if now - self._last_t > 5.0: print(f"Servo: waiting [{self.cal_status}]"); self._last_t = now
            return
        if now - self._last_t < VS_RATE: return
        if now - self._last_centroid_t > 1.0: self._last_centroid = None
        if self._last_centroid is not None:
            jump = np.hypot(centroid[0]-self._last_centroid[0],
                            centroid[1]-self._last_centroid[1])
            if jump > MAX_JUMP_PX:
                print(f"Servo: jump {jump:.0f}px — ignored"); return
        self._last_centroid = centroid; self._last_centroid_t = now; self._last_t = now
        h, w = image_shape[:2]; ic_x, ic_y = w//2, h//2
        ex = centroid[0] - ic_x; ey = centroid[1] - ic_y
        err_r = float(np.hypot(ex, ey)); dx_mm = VS_APPROACH
        if err_r <= VS_DEAD_ZONE:
            dy_mm = dz_mm = 0.0
        elif self._jac_yz_inv is not None:
            yz = -CTRL_GAIN * (self._jac_yz_inv @ np.array([ex, ey], dtype=np.float64))
            dy_mm = float(np.clip(yz[0], -MAX_YZ_STEP, MAX_YZ_STEP))
            dz_mm = float(np.clip(yz[1], -MAX_YZ_STEP, MAX_YZ_STEP))
        else:
            dy_mm = dz_mm = 0.0
        with self._lock:
            try:
                pos = self._get_pos()
                if pos is None: return
                self._arm.set_position(
                    x=pos[0]+dx_mm, y=pos[1]+dy_mm, z=pos[2]+dz_mm,
                    roll=pos[3], pitch=pos[4], yaw=pos[5],
                    speed=VS_SPEED, mvacc=VS_MVACC, wait=True)
                print(f"Servo: err=({ex:+.0f},{ey:+.0f})px r={err_r:.0f} "
                      f"Δ=({dx_mm:+.1f},{dy_mm:+.1f},{dz_mm:+.1f})mm [{self.cal_status}]")
            except Exception as e: print("Servo failed:", e)


# ═════════════════════════════════════════════════════════════════════
#  Camera + display
# ═════════════════════════════════════════════════════════════════════

class CameraStreamer(threading.Thread):
    def __init__(self, cam_index, stop_event, robot):
        super().__init__(daemon=True)
        self.cam_index = cam_index; self.stop_event = stop_event; self.robot = robot
        self._latest_left = None; self._frame_lock = threading.Lock()
        self.data_lock = threading.Lock(); self._result = {}
        self._models_ready = threading.Event()

    def _get_frame(self):
        with self._frame_lock:
            return self._latest_left.copy() if self._latest_left is not None else None

    def _run_calibration(self):
        print("Cal thread: waiting for models..."); self._models_ready.wait()
        print("Cal thread: starting."); self.robot.calibrate(self._get_frame)

    def _seg_loop(self):
        while not self.stop_event.is_set():
            with self._frame_lock:
                frame = self._latest_left.copy() if self._latest_left is not None else None
            if frame is None: time.sleep(0.1); continue
            try:
                res = run_pipeline(frame, PROMPT)
                with self.data_lock: self._result = res
                if not self._models_ready.is_set():
                    print("Models ready."); self._models_ready.set()
                if self.robot and res.get("best_centroid"):
                    self.robot.servo_step(res["best_centroid"], frame.shape)
            except Exception as e: print("Pipeline error:", e); time.sleep(1)

    def run(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_ANY)
        if not cap.isOpened(): print(f"Camera {self.cam_index} failed"); return
        threading.Thread(target=self._seg_loop, daemon=True).start()
        if self.robot and self.robot._arm:
            threading.Thread(target=self._run_calibration, daemon=True).start()
        win = "Grasp point  |  [v] servo  [q] quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        vw = None; vp = None
        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret: time.sleep(0.05); continue
                h, w = frame.shape[:2]; left = frame[:, :w//2].copy()
                with self._frame_lock: self._latest_left = left.copy()
                with self.data_lock: res = dict(self._result)
                rendered = self._render(left, res)
                if vw is None:
                    rh, rw = rendered.shape[:2]
                    vp = time.strftime("vs_recording_%Y%m%d_%H%M%S.mp4")
                    vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (rw, rh))
                    print(f"Recording to {vp}")
                vw.write(rendered); cv2.imshow(win, rendered)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"): self.stop_event.set(); break
                elif key == ord("v") and self.robot:
                    self.robot.enabled = not self.robot.enabled
                    print("Servo:", "ON" if self.robot.enabled else "OFF")
        finally:
            cap.release()
            if vw: vw.release(); print(f"Saved: {vp}")
            try: cv2.destroyAllWindows()
            except: pass

    def _render(self, left, res):
        display = left.copy()
        h, w = display.shape[:2]
        mask_np       = res.get("mask_np")
        depth_np      = res.get("depth_np")
        best_centroid = res.get("best_centroid")
        gdino_box     = res.get("gdino_box")
        split_lines   = res.get("split_lines", [])
        top_region    = res.get("top_region")

        # mask resize
        mask_r = mask_np
        if mask_r is not None and mask_r.shape[:2] != (h, w):
            mask_r = cv2.resize(mask_r, (w, h), interpolation=cv2.INTER_NEAREST)

        # SAM2 mask overlay
        if mask_r is not None:
            ov = np.zeros_like(display); ov[:, :, 1] = mask_r
            display = cv2.addWeighted(display, 0.72, ov, 0.28, 0)
            cnts, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, cnts, -1, (0, 255, 0), 2)

        # bounding box
        if gdino_box is not None:
            x1, y1, x2, y2 = gdino_box.astype(int)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)

            # draw split lines (cyan dashed)
            for sy in split_lines:
                cv2.line(display, (x1, sy), (x2, sy), (255, 255, 0), 1, cv2.LINE_AA)

            # highlight top-box region
            if top_region is not None:
                ty, by = top_region
                overlay_top = display.copy()
                cv2.rectangle(overlay_top, (x1, ty), (x2, by), (0, 200, 255), -1)
                display = cv2.addWeighted(display, 0.85, overlay_top, 0.15, 0)
                cv2.rectangle(display, (x1, ty), (x2, by), (0, 200, 255), 2)

        # image centre
        ic = (w // 2, h // 2)
        cv2.drawMarker(display, ic, (255, 80, 0), cv2.MARKER_CROSS, 22, 2)

        # grasp point
        if best_centroid is not None:
            gp = best_centroid
            cv2.circle(display, gp, 18, (0, 0, 255), 2)
            cv2.drawMarker(display, gp, (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
            cv2.putText(display, "GRASP", (gp[0]+22, gp[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)
            cv2.arrowedLine(display, ic, gp, (0, 220, 255), 2, tipLength=0.12)
            ex, ey = gp[0]-ic[0], gp[1]-ic[1]
            cv2.putText(display, f"err ({ex:+d},{ey:+d}) px",
                        (8, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1)

        # status
        if self.robot:
            cal = self.robot.cal_status
            txt = f"SERVO {'ON' if self.robot.enabled else 'OFF'}  [{cal}]"
            col = (0, 255, 80) if self.robot.enabled else (80, 80, 80)
            cv2.putText(display, txt, (8, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)

        cv2.putText(display, DETECTOR.upper(), (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 0), 1)

        # depth inset
        if depth_np is not None:
            try:
                d = depth_np.astype(np.float32)
                dmn, dmx = float(np.nanmin(d)), float(np.nanmax(d))
                d8 = ((d-dmn)/(dmx-dmn+1e-6)*255).astype(np.uint8)
                cm = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
                iw = max(100, w//4); ih = int(iw*h/w)
                ins = cv2.resize(cm, (iw, ih))
                display[10:10+ih, w-iw-10:w-10] = ins
                cv2.putText(display, "depth", (w-iw-8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            except: pass

        return display


# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SemVS Pipeline")
    parser.add_argument("cam_index", nargs="?", type=int, default=0)
    parser.add_argument("--detector", choices=["gdino", "owlv2"], default=None)
    args = parser.parse_args()
    if args.detector: DETECTOR = args.detector
    print(f"Detector: {DETECTOR.upper()}")

    robot = RobotController(ROBOT_IP); robot.connect()
    stop_ev = threading.Event()
    cam = CameraStreamer(args.cam_index, stop_ev, robot); cam.start()
    try: cam.join()
    except KeyboardInterrupt: print("Interrupted")
    finally: robot.stop(); stop_ev.set(); cam.join(timeout=2); print("Done.")