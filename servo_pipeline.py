#!/usr/bin/env python3
"""
SemVS – Semantic Visual Servoing
Updated: image-conditioned detection, stable-anchor mask propagation,
         PyZED camera + SVO recording, ≥512 px annotated video.

Changes from baseline:
  • Transparent / black-background reference image support
  • All print() → structured logging (file + console)
  • Phase 1: ref_logits injected on fresh-detection frames (pre-anchor)
  • Phase 6: MaskTracker warm-started from ref_logits/ref_mask
  • Phase 2: Bootstrap multi-point grid on first N frames
  • Phase 5: Ref-aware multi-mask selection (IoU + similarity scoring)
  • Phase 3: Ref-aspect-ratio box splitting for stacked-box scenarios
  • Phase 4: Adaptive coverage threshold from ref mask fill ratio
"""
import time
import sys
import threading
import os
import tempfile
import logging

import cv2
import numpy as np

# ═════════════════════════════════════════════════════════════════════
#  Logging setup
# ═════════════════════════════════════════════════════════════════════

def _setup_logger(name: str = "semvs",
                  log_dir: str = "logs",
                  level: int = logging.DEBUG) -> logging.Logger:
    """
    Create a logger that writes to both console (INFO+) and a
    timestamped log file (DEBUG+).
    """
    os.makedirs(log_dir, exist_ok=True)
    ts       = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"semvs_{ts}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler (INFO and above)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (DEBUG and above)
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)-5s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(fh)

    logger.info(f"Logging to {os.path.abspath(log_path)}")
    return logger

log = _setup_logger()

# ═════════════════════════════════════════════════════════════════════
#  Configuration
# ═════════════════════════════════════════════════════════════════════

THIRD_PARTY_ROOT = os.path.join(os.path.dirname(__file__), "third-party")

PROMPT = "box"

# ── Detector selection ────────────────────────────────────────────────
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

# ── Robot safety limits ──────────────────────────────────────────────
MIN_Z_MM     = 0.0

# ── Mask propagation / tracking parameters ───────────────────────────
TRACKER_WARMUP_FRAMES = 10
TRACKER_MAX_HISTORY   = 15
NEG_POINT_COUNT       = 5
NEG_POINT_MARGIN_PX   = 30
REDETECT_INTERVAL     = 30
IOU_DRIFT_THRESH      = 0.35

# ── Anchor / stable-mask parameters ──────────────────────────────────
ANCHOR_IOU_THRESH  = 0.82
ANCHOR_LOCK_FRAMES = 5

# ── Bootstrap parameters (Phase 2) ───────────────────────────────────
BOOTSTRAP_GRID_ROWS  = 3       # grid rows inside detected box during bootstrap
BOOTSTRAP_GRID_COLS  = 3       # grid cols inside detected box during bootstrap

# ── Video recording ───────────────────────────────────────────────────
MIN_RECORDING_DIM = 512
ZED_RESOLUTION    = "HD720"

# ── Reference image foreground detection ──────────────────────────────
REF_BG_DARK_THRESH  = 15       # pixel intensity below this = background
REF_BG_BORDER_CHECK = 20       # border strip width to auto-detect black bg
REF_ALPHA_THRESH    = 128      # alpha channel threshold for foreground

# ── ZED calibration for lens undistortion (OpenCV path) ───────────────
ZED_SETTINGS_DIRS = [
    "/usr/local/zed/settings/",
    os.path.expanduser("~/.ZED/settings/"),
]

def _load_zed_calibration(resolution: str = "HD"):
    """
    Load ZED camera intrinsics + distortion coefficients from the factory
    calibration file on disk.
    """
    import glob, configparser

    conf_files = []
    for d in ZED_SETTINGS_DIRS:
        conf_files.extend(glob.glob(os.path.join(d, "SN*.conf")))
    if not conf_files:
        return None, None, None

    conf_files.sort(key=os.path.getmtime, reverse=True)
    conf_path = conf_files[0]

    cp = configparser.ConfigParser()
    cp.read(conf_path)

    section = f"LEFT_CAM_{resolution}"
    if section not in cp:
        log.warning("ZED cal: section [%s] not found in %s", section, conf_path)
        return None, None, None

    fx = float(cp[section]["fx"])
    fy = float(cp[section]["fy"])
    cx = float(cp[section]["cx"])
    cy = float(cp[section]["cy"])
    k1 = float(cp[section]["k1"])
    k2 = float(cp[section]["k2"])
    p1 = float(cp[section].get("k3", "0"))
    p2 = float(cp[section].get("k4", "0"))

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    D = np.array([k1, k2, p1, p2], dtype=np.float64)

    res_map = {"2K": (2208, 1242), "FHD": (1920, 1080),
               "HD": (1280, 720), "VGA": (672, 376)}
    img_size = res_map.get(resolution, (1280, 720))

    sn = os.path.splitext(os.path.basename(conf_path))[0]
    log.info("ZED cal: loaded %s [%s]  fx=%.1f fy=%.1f cx=%.1f cy=%.1f  "
             "dist=[%.4f,%.4f,%.4f,%.4f]",
             sn, section, fx, fy, cx, cy, k1, k2, p1, p2)
    return K, D, img_size


class ZedUndistorter:
    """Pre-computes undistortion maps from ZED factory calibration."""

    def __init__(self, resolution: str = "HD"):
        self._map1 = None
        self._map2 = None
        self._new_K = None
        K, D, img_size = _load_zed_calibration(resolution)
        if K is None:
            log.warning("ZED undistortion: calibration not found — frames will "
                        "NOT be undistorted")
            return

        w, h = img_size
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            K, D, None, new_K, (w, h), cv2.CV_16SC2)
        self._new_K = new_K
        log.info("ZED undistortion: maps ready (%d×%d)", w, h)

    @property
    def available(self) -> bool:
        return self._map1 is not None

    @classmethod
    def from_frame_size(cls, width: int, height: int) -> "ZedUndistorter":
        _size_to_res = {
            (2208, 1242): "2K",
            (1920, 1080): "FHD",
            (1280, 720):  "HD",
            (672,  376):  "VGA",
        }
        res = _size_to_res.get((width, height))
        if res is None:
            log.debug("ZedUndistorter: no calibration section for %d×%d "
                      "— trying closest match", width, height)
            best, best_diff = "HD", float("inf")
            for (w, h), r in _size_to_res.items():
                diff = abs(w - width) + abs(h - height)
                if diff < best_diff:
                    best, best_diff = r, diff
            res = best
        return cls(res)

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        if self._map1 is None:
            return frame
        h, w = frame.shape[:2]
        mh, mw = self._map1.shape[:2]
        if (h, w) != (mh, mw):
            return frame
        return cv2.remap(frame, self._map1, self._map2,
                         interpolation=cv2.INTER_LINEAR)


# ── PyZED ─────────────────────────────────────────────────────────────
try:
    import pyzed.sl as sl
    PYZED_AVAILABLE = True
    log.info("PyZED available — will use ZED SDK for capture and SVO recording")
except ImportError:
    sl               = None
    PYZED_AVAILABLE  = False

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch  = None
    DEVICE = "cpu"


# ═════════════════════════════════════════════════════════════════════
#  Geometry helpers
# ═════════════════════════════════════════════════════════════════════

def _mask_centroid(mask: np.ndarray):
    """Return (cx, cy) from binary mask using cv2.moments, or None."""
    M = cv2.moments(mask, binaryImage=True)
    if M["m00"] < 1.0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def _robust_centroid(mask: np.ndarray):
    """
    Compute a robust centroid for box-like objects.

    Strategy:
      1. Morphological close to bridge gaps from internal lines / textures.
      2. Find the largest external contour.
      3. Convex hull → fill → minAreaRect centre  (robust for rectangles).
      4. Verify the centre lies inside the closed mask; fall back to
         convex-hull moments otherwise.
    """
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _mask_centroid(mask)

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:
        return _mask_centroid(mask)

    rect = cv2.minAreaRect(largest)
    cx, cy = rect[0]
    cx_i, cy_i = int(round(cx)), int(round(cy))

    h, w = closed.shape[:2]
    if 0 <= cy_i < h and 0 <= cx_i < w and closed[cy_i, cx_i] > 0:
        return (cx_i, cy_i)

    hull = cv2.convexHull(largest)
    hull_mask = np.zeros_like(mask)
    cv2.fillConvexPoly(hull_mask, hull, 255)
    c = _mask_centroid(hull_mask)
    if c is not None:
        return c

    return _mask_centroid(mask)


def _nms(boxes: np.ndarray, scores: np.ndarray,
         iou_thresh: float = 0.5) -> list:
    """Simple greedy NMS.  boxes: Nx4 (x1,y1,x2,y2), scores: N."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-6)
        order = rest[iou <= iou_thresh]
    return keep


def _mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """IoU between two binary uint8 masks (0/255)."""
    if m1.shape != m2.shape:
        m2 = cv2.resize(m2, (m1.shape[1], m1.shape[0]),
                        interpolation=cv2.INTER_NEAREST)
    inter = np.count_nonzero((m1 > 0) & (m2 > 0))
    union = np.count_nonzero((m1 > 0) | (m2 > 0))
    return inter / (union + 1e-6)


# ═════════════════════════════════════════════════════════════════════
#  MaskTracker — temporal state for SAM2 mask propagation + anchoring
# ═════════════════════════════════════════════════════════════════════

class MaskTracker:
    """
    Maintains inter-frame state for stable SAM2 mask propagation.

    Supports warm-starting from reference-image logits/mask (Phase 6)
    so the tracker has an anchor from frame 0.
    """

    def __init__(self):
        self.prev_logits:    np.ndarray | None = None
        self.prev_mask:      np.ndarray | None = None
        self.prev_centroid:  tuple | None      = None
        self.frame_count:    int               = 0
        self.mask_history:   list              = []

        # Stable-anchor fields
        self.anchor_logits:  np.ndarray | None = None
        self.anchor_mask:    np.ndarray | None = None
        self._iou_streak:    int               = 0
        self._last_iou:      float             = 1.0

        # Phase 6: track whether anchor came from ref vs live
        self._anchor_source: str | None        = None  # "ref" or "live"

    # ── Phase 6: warm-start from reference image ─────────────────────
    def seed_from_ref(self, ref_logits: np.ndarray | None,
                      ref_mask: np.ndarray | None):
        """
        Pre-seed the anchor from the reference image so the tracker has
        a fallback from frame 0.  The ref-based anchor will be replaced
        once a live anchor is confirmed.
        """
        if ref_logits is None:
            return
        self.anchor_logits  = ref_logits.copy()
        self.anchor_mask    = ref_mask.copy() if ref_mask is not None else None
        self._anchor_source = "ref"
        log.info("MaskTracker: seeded anchor from reference image "
                 "(will be replaced once live anchor is confirmed)")

    # ── housekeeping ─────────────────────────────────────────────────
    def reset(self, keep_anchor: bool = True):
        """Reset live tracking state.  Optionally keep the anchor."""
        self.prev_logits    = None
        self.prev_mask      = None
        self.prev_centroid  = None
        self.frame_count    = 0
        self.mask_history.clear()
        self._iou_streak    = 0
        self._last_iou      = 1.0
        if not keep_anchor:
            self.anchor_logits  = None
            self.anchor_mask    = None
            self._anchor_source = None

    def update(self, mask_np: np.ndarray | None,
               logits: np.ndarray | None,
               centroid: tuple | None):
        if mask_np is not None and self.prev_mask is not None:
            iou = _mask_iou(mask_np, self.prev_mask)
            self._last_iou = iou
            if iou >= ANCHOR_IOU_THRESH:
                self._iou_streak += 1
            else:
                self._iou_streak = 0
        elif mask_np is None:
            self._iou_streak = 0

        self.prev_mask      = mask_np
        self.prev_logits    = logits
        self.prev_centroid  = centroid

        if mask_np is not None:
            self.frame_count += 1
            self.mask_history.append(mask_np.copy())
            if len(self.mask_history) > TRACKER_MAX_HISTORY:
                self.mask_history.pop(0)
            self._try_lock_anchor(logits, mask_np)
        else:
            self.frame_count = max(0, self.frame_count - 1)

    def _try_lock_anchor(self, logits: np.ndarray | None,
                         mask_np: np.ndarray):
        """Promote current mask to anchor when track is stable."""
        if logits is None:
            return
        if self._iou_streak >= ANCHOR_LOCK_FRAMES:
            # Phase 6: always upgrade to live anchor
            self.anchor_logits  = logits.copy()
            self.anchor_mask    = mask_np.copy()
            prev_src            = self._anchor_source
            self._anchor_source = "live"
            if prev_src == "ref":
                log.info("MaskTracker: live anchor replaced ref-based anchor "
                         "(IoU streak=%d, last IoU=%.3f)",
                         self._iou_streak, self._last_iou)
            else:
                log.info("MaskTracker: anchor locked (IoU streak=%d, "
                         "last IoU=%.3f)",
                         self._iou_streak, self._last_iou)

    @property
    def warmed_up(self) -> bool:
        return (self.frame_count >= TRACKER_WARMUP_FRAMES
                and len(self.mask_history) >= TRACKER_WARMUP_FRAMES)

    @property
    def anchor_locked(self) -> bool:
        return self.anchor_logits is not None

    @property
    def anchor_is_live(self) -> bool:
        """True when the anchor came from live frames (not ref image)."""
        return self._anchor_source == "live"

    # ── negative-point mining ────────────────────────────────────────
    def sample_negative_points(self, h: int, w: int,
                               n: int = NEG_POINT_COUNT) -> np.ndarray | None:
        if not self.warmed_up:
            return None

        always_bg = np.ones((h, w), dtype=np.uint8) * 255
        for m in self.mask_history[-TRACKER_WARMUP_FRAMES:]:
            mr = m if m.shape[:2] == (h, w) else cv2.resize(
                m, (w, h), interpolation=cv2.INTER_NEAREST)
            always_bg = cv2.bitwise_and(always_bg, cv2.bitwise_not(mr))

        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (NEG_POINT_MARGIN_PX * 2 + 1,
                                          NEG_POINT_MARGIN_PX * 2 + 1))
        safe = cv2.erode(always_bg, kern)

        coords = np.argwhere(safe > 0)
        if len(coords) < n:
            return None

        idx = np.random.choice(len(coords), n, replace=False)
        pts = coords[idx][:, ::-1].astype(np.float32)
        return pts

    # ── positive-point mining from previous mask ─────────────────────
    def sample_positive_points(self, h: int, w: int,
                               n: int = 1) -> np.ndarray | None:
        if self.prev_mask is None:
            return None

        pts = []
        if self.prev_centroid is not None:
            pts.append(list(self.prev_centroid))

        if n > len(pts) and self.prev_mask is not None:
            mr = self.prev_mask if self.prev_mask.shape[:2] == (h, w) else \
                cv2.resize(self.prev_mask, (w, h),
                           interpolation=cv2.INTER_NEAREST)
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            interior = cv2.erode(mr, kern)
            coords = np.argwhere(interior > 0)
            if len(coords) > 0:
                extra_n = min(n - len(pts), len(coords))
                idx = np.random.choice(len(coords), extra_n, replace=False)
                for r, c in coords[idx]:
                    pts.append([c, r])

        if not pts:
            return None
        return np.array(pts[:n], dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════
#  Model singletons
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

        log.debug("BERT compat patch applied")
    except Exception as e:
        log.warning("BERT compat patch failed: %s", e)
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
            log.error("GroundingDINO load failed (won't retry): %s", e)
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
            log.info("OWLv2 loaded")
        except Exception as e:
            log.error("OWLv2 load failed (won't retry): %s", e)
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
                log.error("Depth-Anything-V2 checkpoint not found: %s", ckpt)
                return None
            import torch as _t
            m = DepthAnythingV2(**cfgs[DA_ENCODER])
            m.load_state_dict(_t.load(ckpt, map_location="cpu"))
            _depth_model = m.to(DEVICE).eval()
        except Exception as e:
            log.error("Depth-Anything-V2 load failed: %s", e)
    return _depth_model


# ═════════════════════════════════════════════════════════════════════
#  Feature extractor for visual similarity
# ═════════════════════════════════════════════════════════════════════

_feat_model     = None
_feat_transform = None


def _get_feature_extractor():
    global _feat_model, _feat_transform
    if _feat_model is not None:
        return _feat_model, _feat_transform
    try:
        import torch as _t
        from torchvision import models, transforms

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        _feat_model = _t.nn.Sequential(
            *list(backbone.children())[:-1]
        ).to(DEVICE).eval()

        _feat_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        log.info("Feature extractor (ResNet-18) loaded")
    except Exception as e:
        log.error("Feature extractor load failed: %s", e)
    return _feat_model, _feat_transform


def _extract_features(crop_bgr: np.ndarray) -> np.ndarray | None:
    model, transform = _get_feature_extractor()
    if model is None or crop_bgr.size == 0:
        return None
    try:
        import torch as _t
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = transform(rgb).unsqueeze(0).to(DEVICE)
        with _t.no_grad():
            feat = model(tensor).squeeze()
        feat = feat / (feat.norm() + 1e-8)
        return feat.cpu().numpy()
    except Exception as e:
        log.debug("Feature extraction failed: %s", e)
        return None


def _cosine_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    return float(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8))


# ═════════════════════════════════════════════════════════════════════
#  Reference-image conditioning  (updated for transparent / black bg)
# ═════════════════════════════════════════════════════════════════════

def _load_ref_image(path: str):
    """
    Load a reference image, handling both BGRA (transparent) and BGR
    (black-background) formats.

    Returns
    -------
    bgr      : np.ndarray   3-channel BGR image
    fg_mask  : np.ndarray   uint8 0/255 foreground mask
    """
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        log.error("Could not load reference image: %s", path)
        return None, None

    # ── Case 1: 4-channel image with alpha ──
    if raw.ndim == 3 and raw.shape[2] == 4:
        alpha = raw[:, :, 3]
        bgr   = raw[:, :, :3].copy()
        fg_mask = (alpha >= REF_ALPHA_THRESH).astype(np.uint8) * 255
        fg_pct  = np.count_nonzero(fg_mask) / fg_mask.size
        log.info("Ref image: BGRA format, alpha→fg mask (%.1f%% foreground)",
                 fg_pct * 100)
        return bgr, fg_mask

    # ── Case 2: 3-channel BGR — check for black / dark background ──
    bgr = raw if raw.ndim == 3 else cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Detect if the border is consistently dark
    h, w = gray.shape
    b = REF_BG_BORDER_CHECK
    border = np.concatenate([
        gray[:b, :].ravel(), gray[-b:, :].ravel(),
        gray[:, :b].ravel(), gray[:, -b:].ravel(),
    ])
    border_dark_frac = np.count_nonzero(border < REF_BG_DARK_THRESH) / len(border)

    if border_dark_frac > 0.90:
        # Black background — threshold to extract foreground
        fg_mask = (gray >= REF_BG_DARK_THRESH).astype(np.uint8) * 255
        # Clean up: close small gaps, remove small noise blobs
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kern, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kern, iterations=1)
        # Keep only the largest connected component
        n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(fg_mask)
        if n_cc > 1:
            lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            fg_mask = (labels == lbl).astype(np.uint8) * 255
        fg_pct = np.count_nonzero(fg_mask) / fg_mask.size
        log.info("Ref image: black-bg detected (border %.0f%% dark), "
                 "threshold→fg mask (%.1f%% foreground)",
                 border_dark_frac * 100, fg_pct * 100)
        return bgr, fg_mask

    # ── Case 3: Non-trivial background — fall back to Otsu ──
    return bgr, _detect_foreground_otsu(bgr)


def _detect_foreground_otsu(ref_bgr: np.ndarray):
    """Legacy Otsu-based foreground detection for generic backgrounds."""
    gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.count_nonzero(otsu) > otsu.size // 2:
        otsu = cv2.bitwise_not(otsu)
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kern, iterations=2)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN,  kern, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(otsu)
    if n <= 1:
        h, w = ref_bgr.shape[:2]
        fm = np.zeros((h, w), dtype=np.uint8)
        fm[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)] = 255
        return fm
    lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fm  = (labels == lbl).astype(np.uint8) * 255
    log.debug("Ref image: Otsu fg mask (%.1f%% foreground)",
              np.count_nonzero(fm) / fm.size * 100)
    return fm


def extract_ref_logits(ref_bgr: np.ndarray,
                       fg_mask: np.ndarray,
                       box_np: np.ndarray | None = None):
    """
    Run SAM2 on the reference image using multi-point prompts that span
    the entire object.  Uses the provided fg_mask (from alpha channel or
    black-background extraction) instead of re-computing Otsu internally.

    Returns
    -------
    logits : np.ndarray | None   shape (1, 256, 256)
    mask   : np.ndarray | None   uint8, 0/255
    """
    try:
        pred = _get_sam2()
        h, w = ref_bgr.shape[:2]
        rgb  = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
        pred.set_image(rgb)

        # --- Bounding box from fg_mask ---
        if box_np is not None:
            bx1, by1, bx2, by2 = box_np.astype(int)
        else:
            fg_bbox = cv2.boundingRect(fg_mask)
            bx, by, bw, bh = fg_bbox
            bx1, by1 = bx, by
            bx2, by2 = bx + bw, by + bh
        pad = 10
        bx1, by1 = max(0, bx1 - pad), max(0, by1 - pad)
        bx2, by2 = min(w, bx2 + pad), min(h, by2 + pad)
        sam_box = np.array([bx1, by1, bx2, by2], dtype=np.float32)
        box_w, box_h = bx2 - bx1, by2 - by1

        # --- Multi-point grid (3 cols × 5 rows) inside the fg mask ---
        points = []
        for fy in [0.15, 0.30, 0.50, 0.70, 0.85]:
            for fx in [0.25, 0.50, 0.75]:
                px = int(bx1 + fx * box_w)
                py = int(by1 + fy * box_h)
                px = min(max(px, 0), w - 1)
                py = min(max(py, 0), h - 1)
                if fg_mask[py, px] > 0:
                    points.append([px, py])
        if not points:
            points.append([int((bx1 + bx2) / 2), int((by1 + by2) / 2)])

        log.info("Ref SAM2: box=[%d,%d,%d,%d], %d positive points",
                 bx1, by1, bx2, by2, len(points))

        sam_kwargs = dict(
            box=sam_box,
            point_coords=np.array(points, dtype=np.float32),
            point_labels=np.ones(len(points), dtype=np.int32),
            multimask_output=True,
            return_logits=True,
        )

        masks, scores, logits = pred.predict(**sam_kwargs)

        if masks is None or len(masks) == 0:
            log.warning("Ref SAM2: no mask returned")
            return None, None

        best_idx   = int(np.argmax(scores))
        ref_mask   = (masks[best_idx] > 0).astype(np.uint8) * 255
        ref_logits = logits[best_idx: best_idx + 1]

        # --- Refinement: if coverage of foreground is low, re-run ---
        covered  = np.count_nonzero((fg_mask > 0) & (ref_mask > 0))
        total_fg = max(1, np.count_nonzero(fg_mask > 0))
        coverage = covered / total_fg

        if coverage < 0.70:
            log.info("Ref SAM2: coverage=%.0f%%, adding uncovered points",
                     coverage * 100)
            uncovered  = (fg_mask > 0) & (ref_mask == 0)
            unc_coords = np.argwhere(uncovered)
            if len(unc_coords) > 0:
                n_extra = min(5, len(unc_coords))
                idx = np.linspace(0, len(unc_coords) - 1, n_extra, dtype=int)
                for r, c in unc_coords[idx]:
                    points.append([int(c), int(r)])
                sam_kwargs["point_coords"] = np.array(points, dtype=np.float32)
                sam_kwargs["point_labels"] = np.ones(len(points), dtype=np.int32)
                sam_kwargs["mask_input"]   = ref_logits
                sam_kwargs["multimask_output"] = False
                pred.set_image(rgb)
                masks2, scores2, logits2 = pred.predict(**sam_kwargs)
                if masks2 is not None and len(masks2) > 0:
                    best2      = int(np.argmax(scores2))
                    ref_mask   = (masks2[best2] > 0).astype(np.uint8) * 255
                    ref_logits = logits2[best2: best2 + 1]
                    log.info("Ref SAM2 (refined): score=%.3f", scores2[best2])

        # --- Morph close to bridge white-line gap ---
        k_close  = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        ref_mask = cv2.morphologyEx(ref_mask, cv2.MORPH_CLOSE,
                                    k_close, iterations=2)
        contours, _ = cv2.findContours(ref_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ref_mask, contours, -1, 255, cv2.FILLED)

        log.info("Ref image: SAM2 mask extracted (score=%.3f, area=%d px)",
                 scores[best_idx], np.count_nonzero(ref_mask))
        return ref_logits, ref_mask

    except Exception as e:
        log.error("Reference logit extraction failed: %s", e)
        import traceback; traceback.print_exc()
    return None, None


def process_ref_image(ref_bgr: np.ndarray, fg_mask: np.ndarray,
                      prompt: str):
    """
    Full reference-image processing pipeline.

    1. Run the active detector on the reference image to locate the object.
    2. Crop the object region  → ref_crop.
    3. Run SAM2 on the reference image → ref_logits.
    4. Extract deep features from the crop → ref_features.
    5. Compute ref_fill_ratio for adaptive coverage (Phase 4).
    6. Compute ref_aspect_ratio for box splitting (Phase 3).

    Returns
    -------
    ref_crop        : np.ndarray | None
    ref_logits      : np.ndarray | None
    ref_mask        : np.ndarray | None
    ref_features    : np.ndarray | None
    ref_fill_ratio  : float              (Phase 4)
    ref_aspect_ratio: float              (Phase 3)
    """
    log.info("Processing reference image …")
    ref_box = None

    # Step 1: detect object in reference image
    if DETECTOR == "owlv2":
        dets = _detect_owlv2(ref_bgr, prompt)
    else:
        dets = _detect_gdino(ref_bgr, prompt)

    if dets:
        pick = _disambiguate_top_box(dets, ref_bgr.shape)
        if pick is not None:
            ref_box = pick[0]
            log.info("Ref image: detected '%s' score=%.2f @ %s",
                     pick[2], pick[1], ref_box.astype(int))

    # Step 2: build object crop
    h, w = ref_bgr.shape[:2]
    if ref_box is not None:
        x1, y1, x2, y2 = ref_box.astype(int)
        pad = 10
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        ref_crop = ref_bgr[y1:y2, x1:x2].copy()
    else:
        # Use fg_mask bounding rect as crop region
        fg_bbox = cv2.boundingRect(fg_mask)
        bx, by, bw, bh = fg_bbox
        pad = 10
        x1, y1 = max(0, bx - pad), max(0, by - pad)
        x2, y2 = min(w, bx + bw + pad), min(h, by + bh + pad)
        ref_crop = ref_bgr[y1:y2, x1:x2].copy()
        log.info("Ref crop: from fg_mask bbox (no detection in ref image)")
    log.info("Ref crop shape: %s", ref_crop.shape)

    # Step 3: SAM2 logits from reference image
    ref_logits, ref_mask = extract_ref_logits(ref_bgr, fg_mask,
                                              box_np=ref_box)

    # Step 4: deep features
    ref_features = _extract_features(ref_crop)
    if ref_features is not None:
        log.info("Ref features: 512-d vector extracted (norm=%.3f)",
                 np.linalg.norm(ref_features))
    else:
        log.warning("Ref features: extraction failed — similarity scoring "
                    "disabled")

    # Phase 4: compute expected fill ratio from ref mask
    if ref_box is not None and ref_mask is not None:
        bx1, by1, bx2, by2 = ref_box.astype(int)
        box_area       = max(1, (bx2 - bx1) * (by2 - by1))
        ref_fill_ratio = np.count_nonzero(ref_mask) / box_area
    elif ref_mask is not None:
        fg_bbox        = cv2.boundingRect(fg_mask)
        fg_area        = max(1, fg_bbox[2] * fg_bbox[3])
        ref_fill_ratio = np.count_nonzero(ref_mask) / fg_area
    else:
        ref_fill_ratio = 0.7  # conservative default
    log.info("Ref fill ratio: %.2f (for adaptive coverage check)", ref_fill_ratio)

    # Phase 3: compute ref aspect ratio for box splitting
    fg_bbox = cv2.boundingRect(fg_mask)
    if fg_bbox[2] > 0 and fg_bbox[3] > 0:
        ref_aspect_ratio = fg_bbox[2] / fg_bbox[3]  # width / height
    else:
        ref_aspect_ratio = 1.0
    log.info("Ref aspect ratio: %.2f (w/h)", ref_aspect_ratio)

    return ref_crop, ref_logits, ref_mask, ref_features, \
        ref_fill_ratio, ref_aspect_ratio


# ═════════════════════════════════════════════════════════════════════
#  Detection
# ═════════════════════════════════════════════════════════════════════

def _detect_gdino(image_bgr: np.ndarray, prompt: str):
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
                log.debug("GDINO: no boxes")
                return []
            h, w  = image_bgr.shape[:2]
            bxyxy = (box_convert(boxes, "cxcywh", "xyxy")
                     * _t.tensor([w, h, w, h], dtype=_t.float32)).numpy()
            dets  = [(bxyxy[i], float(logits[i]), phrases[i])
                     for i in range(len(boxes))]
            log.info("GDINO: %d detection(s)", len(dets))
            return dets
        finally:
            os.unlink(tmp.name)
    except Exception as e:
        import traceback
        log.error("GroundingDINO failed: %s", e)
        traceback.print_exc()
        return []


def _detect_owlv2(image_bgr: np.ndarray, prompt: str):
    processor, model = _get_owlv2()
    if model is None:
        return []
    try:
        import torch as _t
        from PIL import Image

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        texts  = [[prompt]]
        inputs = processor(text=texts, images=pil,
                           return_tensors="pt").to(DEVICE)

        with _t.no_grad():
            outputs = model(**inputs)

        h, w          = image_bgr.shape[:2]
        target_sizes  = _t.tensor([[h, w]], device=DEVICE)

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
            log.debug("OWLv2: no boxes")
            return []

        dets = [(boxes[i], float(scores[i]), prompt) for i in range(len(boxes))]
        log.info("OWLv2: %d detection(s)", len(dets))
        return dets

    except Exception as e:
        import traceback
        log.error("OWLv2 failed: %s", e)
        traceback.print_exc()
        return []


def _detect_owlv2_image_guided(image_bgr: np.ndarray,
                               ref_crop_bgr: np.ndarray):
    processor, model = _get_owlv2()
    if model is None:
        return []
    try:
        import torch as _t
        from PIL import Image

        rgb     = cv2.cvtColor(image_bgr,  cv2.COLOR_BGR2RGB)
        ref_rgb = cv2.cvtColor(ref_crop_bgr, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(rgb)
        pil_ref   = Image.fromarray(ref_rgb)

        inputs = processor(query_images=[pil_ref], images=[pil_frame],
                           return_tensors="pt").to(DEVICE)

        with _t.no_grad():
            outputs = model.image_guided_detection(**inputs)

        h, w         = image_bgr.shape[:2]
        target_sizes = _t.tensor([[h, w]], device=DEVICE)
        results = processor.post_process_image_guided_detection(
            outputs=outputs, threshold=0.6, nms_threshold=0.3,
            target_sizes=target_sizes)[0]

        boxes  = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        if len(boxes) == 0:
            log.debug("OWLv2 image-guided: no boxes")
            return []

        dets = [(boxes[i], float(scores[i]), "ref_image")
                for i in range(len(boxes))]
        log.info("OWLv2 image-guided: %d detection(s)", len(dets))
        return dets

    except Exception as e:
        import traceback
        log.error("OWLv2 image-guided failed: %s", e)
        traceback.print_exc()
        return []


# ═════════════════════════════════════════════════════════════════════
#  Phase 3: Box disambiguation + ref-aspect-ratio splitting
# ═════════════════════════════════════════════════════════════════════

def _split_box_by_ref_aspect(box_np: np.ndarray,
                             ref_aspect: float,
                             tolerance: float = 0.6) -> np.ndarray:
    """
    Phase 3: If the detected box's aspect ratio is much more portrait than
    the reference (suggesting it spans multiple stacked objects), crop
    it from the top to match the reference aspect ratio.

    Parameters
    ----------
    box_np     : [x1, y1, x2, y2]
    ref_aspect : width/height of the reference object
    tolerance  : if det_aspect < ref_aspect * tolerance, split

    Returns
    -------
    Adjusted box_np (may be the same if no split is needed).
    """
    x1, y1, x2, y2 = box_np
    det_w = x2 - x1
    det_h = y2 - y1
    if det_h < 1 or det_w < 1:
        return box_np

    det_aspect = det_w / det_h

    # Only split if the detection is significantly taller than expected
    if det_aspect < ref_aspect * tolerance:
        # Crop from the top: expected height = det_w / ref_aspect
        expected_h = det_w / ref_aspect
        new_y2 = y1 + min(expected_h, det_h)
        log.info("Phase 3: box aspect %.2f << ref aspect %.2f — "
                 "cropping y2 from %.0f to %.0f",
                 det_aspect, ref_aspect, y2, new_y2)
        return np.array([x1, y1, x2, new_y2], dtype=box_np.dtype)

    return box_np


def _disambiguate_top_box(dets: list, image_shape: tuple,
                          prev_centroid: tuple | None = None,
                          image_bgr: np.ndarray | None = None,
                          ref_features: np.ndarray | None = None):
    if not dets:
        return None

    h, w = image_shape[:2]
    img_area = h * w

    boxes  = np.array([d[0] for d in dets], dtype=np.float32)
    scores = np.array([d[1] for d in dets], dtype=np.float32)
    keep   = _nms(boxes, scores, iou_thresh=0.45)
    dets   = [dets[i] for i in keep]
    if not dets:
        return None

    filtered = []
    for det in dets:
        bx = det[0]
        bw, bh = bx[2] - bx[0], bx[3] - bx[1]
        area_ratio = (bw * bh) / img_area
        if area_ratio < 0.003 or area_ratio > 0.75:
            continue
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect > 8.0:
            continue
        filtered.append(det)

    if not filtered:
        return min(dets, key=lambda d: d[0][1])

    use_similarity = (ref_features is not None and image_bgr is not None)

    best, best_score = None, -1e9
    for det in filtered:
        bx, conf, _ = det
        y_centre  = (bx[1] + bx[3]) / 2.0
        pos_score = 1.0 - (y_centre / h)

        prox_score = 0.0
        if prev_centroid is not None:
            cx   = (bx[0] + bx[2]) / 2.0
            cy   = (bx[1] + bx[3]) / 2.0
            dist = np.hypot(cx - prev_centroid[0], cy - prev_centroid[1])
            prox_score = max(0.0, 1.0 - dist / (0.25 * max(h, w)))

        sim_score = 0.0
        if use_similarity:
            x1, y1, x2, y2 = bx.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                feat = _extract_features(crop)
                if feat is not None:
                    sim_score = max(0.0, _cosine_similarity(feat, ref_features))

        if use_similarity:
            combined = (sim_score  * 0.50
                      + pos_score  * 0.15
                      + conf       * 0.10
                      + prox_score * 0.25)
            log.debug("  det conf=%.2f sim=%.3f pos=%.2f prox=%.2f → %.3f",
                      conf, sim_score, pos_score, prox_score, combined)
        else:
            combined = pos_score * 0.50 + conf * 0.25 + prox_score * 0.25

        if combined > best_score:
            best_score = combined
            best = det

    return best


# ═════════════════════════════════════════════════════════════════════
#  Phase 5: Ref-aware mask selection from SAM2 multimask output
# ═════════════════════════════════════════════════════════════════════

def _select_best_mask_with_ref(masks: np.ndarray,
                               scores_sam: np.ndarray,
                               logits: np.ndarray,
                               image_bgr: np.ndarray,
                               box_np: np.ndarray | None,
                               ref_mask: np.ndarray | None,
                               ref_features: np.ndarray | None) -> int:
    """
    Phase 5: When multimask_output=True returns multiple candidates,
    score them by a combination of SAM2 confidence, IoU with ref_mask,
    and feature similarity.

    During bootstrap (before anchor locks):
      0.30 × SAM2_score + 0.40 × ref_IoU + 0.30 × feat_sim

    Returns the index of the best mask.
    """
    n_masks = len(masks)
    if n_masks == 1:
        return 0

    h, w = image_bgr.shape[:2]
    combined_scores = np.zeros(n_masks, dtype=np.float64)

    for i in range(n_masks):
        mask_i = (masks[i] > 0).astype(np.uint8) * 255
        sam_s  = float(scores_sam[i])

        # IoU with ref mask (resized to frame resolution)
        ref_iou = 0.0
        if ref_mask is not None:
            # Resize ref_mask to match the mask shape
            ref_m_r = cv2.resize(ref_mask, (mask_i.shape[1], mask_i.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            # We can't do pixel-level IoU directly since ref is from a
            # different image.  Instead, compare the mask's fill pattern
            # within the bounding box region.
            if box_np is not None:
                bx1, by1, bx2, by2 = box_np.astype(int)
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(w, bx2), min(h, by2)
                # Compute fill ratio of this mask within the box
                box_region = mask_i[by1:by2, bx1:bx2]
                mask_fill  = np.count_nonzero(box_region) / max(1, box_region.size)
                # Compute ref fill ratio (from ref mask within its bbox)
                ref_bbox   = cv2.boundingRect(ref_m_r)
                if ref_bbox[2] > 0 and ref_bbox[3] > 0:
                    rx, ry, rw, rh = ref_bbox
                    ref_region = ref_m_r[ry:ry+rh, rx:rx+rw]
                    ref_fill   = np.count_nonzero(ref_region) / max(1, ref_region.size)
                    # Score = 1 - |fill_diff|  (closer fill ratio = better)
                    ref_iou = max(0.0, 1.0 - abs(mask_fill - ref_fill))
                else:
                    ref_iou = 0.5

        # Feature similarity of masked crop
        feat_sim = 0.0
        if ref_features is not None and box_np is not None:
            bx1, by1, bx2, by2 = box_np.astype(int)
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w, bx2), min(h, by2)
            # Use the full box crop (not masked) for feature extraction
            crop = image_bgr[by1:by2, bx1:bx2]
            if crop.size > 0:
                # Apply mask to crop — zero out non-mask pixels
                mask_crop = mask_i[by1:by2, bx1:bx2]
                masked_crop = crop.copy()
                masked_crop[mask_crop == 0] = 0
                feat = _extract_features(masked_crop)
                if feat is not None:
                    feat_sim = max(0.0, _cosine_similarity(feat, ref_features))

        combined = 0.30 * sam_s + 0.40 * ref_iou + 0.30 * feat_sim
        combined_scores[i] = combined
        log.debug("  mask[%d] sam=%.3f ref_iou=%.3f feat=%.3f → %.3f",
                  i, sam_s, ref_iou, feat_sim, combined)

    best_idx = int(np.argmax(combined_scores))
    if best_idx != int(np.argmax(scores_sam)):
        log.info("Phase 5: ref-aware selection picked mask[%d] (sam picked %d)",
                 best_idx, int(np.argmax(scores_sam)))
    return best_idx


# ═════════════════════════════════════════════════════════════════════
#  Phase 2: Bootstrap grid-point generation
# ═════════════════════════════════════════════════════════════════════

def _generate_bootstrap_points(box_np: np.ndarray,
                               h: int, w: int,
                               rows: int = BOOTSTRAP_GRID_ROWS,
                               cols: int = BOOTSTRAP_GRID_COLS) -> np.ndarray:
    """
    Generate a grid of positive points inside the detected bounding box.
    Used during bootstrap (before tracker warms up) to give SAM2 more
    spatial guidance across the full object face.

    Returns Nx2 np.float32 array of (x, y) points.
    """
    x1, y1, x2, y2 = box_np.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return np.array([[int((x1+x2)/2), int((y1+y2)/2)]], dtype=np.float32)

    points = []
    for ri in range(rows):
        fy = (ri + 0.5) / rows
        for ci in range(cols):
            fx = (ci + 0.5) / cols
            px = int(x1 + fx * bw)
            py = int(y1 + fy * bh)
            points.append([px, py])

    return np.array(points, dtype=np.float32)


# ═════════════════════════════════════════════════════════════════════
#  Core pipeline  (with all 6 fix phases)
# ═════════════════════════════════════════════════════════════════════

def run_pipeline(image_bgr: np.ndarray, prompt: str,
                 tracker: MaskTracker | None = None,
                 ref_logits: np.ndarray | None = None,
                 ref_crop: np.ndarray | None = None,
                 ref_features: np.ndarray | None = None,
                 ref_mask: np.ndarray | None = None,
                 ref_fill_ratio: float = 0.7,
                 ref_aspect_ratio: float = 1.0) -> dict:
    """
    Detection → SAM2 segmentation → depth → grasp-point pipeline.

    Fix phases integrated:
      Phase 1: ref_logits injected on fresh-detection frames (pre-anchor)
      Phase 2: Bootstrap grid points on first N frames
      Phase 3: Ref-aspect box splitting for stacked boxes
      Phase 4: Adaptive coverage threshold from ref_fill_ratio
      Phase 5: Ref-aware multi-mask selection
      Phase 6: Tracker warm-started from ref (done externally)
    """
    res = dict(mask_np=None, depth_np=None,
               best_centroid=None, gdino_box=None,
               similarity=None)

    h, w = image_bgr.shape[:2]

    # ── Decide whether to run the detector this frame ─────────────────
    need_detection = True
    if tracker is not None and tracker.prev_logits is not None:
        if tracker.frame_count % REDETECT_INTERVAL != 0:
            need_detection = False

    # ── Step 1: Detect bounding box ───────────────────────────────────
    box_np = None
    if need_detection:
        prev_c = tracker.prev_centroid if tracker else None

        if ref_crop is not None and DETECTOR == "owlv2":
            dets = _detect_owlv2_image_guided(image_bgr, ref_crop)
            if not dets:
                dets = _detect_owlv2(image_bgr, prompt)
        elif DETECTOR == "owlv2":
            dets = _detect_owlv2(image_bgr, prompt)
        else:
            dets = _detect_gdino(image_bgr, prompt)

        pick = _disambiguate_top_box(dets, image_bgr.shape,
                                     prev_centroid=prev_c,
                                     image_bgr=image_bgr,
                                     ref_features=ref_features)
        if pick is not None:
            box_np = pick[0]

            # Phase 3: split box if it spans multiple stacked objects
            box_np = _split_box_by_ref_aspect(box_np, ref_aspect_ratio)

            res["gdino_box"] = box_np
            if ref_features is not None:
                x1, y1, x2, y2 = box_np.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                crop = image_bgr[y1:y2, x1:x2]
                feat = _extract_features(crop) if crop.size > 0 else None
                if feat is not None:
                    res["similarity"] = _cosine_similarity(feat, ref_features)
            sim_str = f"  sim={res['similarity']:.3f}" if res.get("similarity") else ""
            log.info("%s: '%s' score=%.2f @ %s  (top of %d)%s",
                     DETECTOR.upper(), pick[2], pick[1],
                     box_np.astype(int), len(dets), sim_str)

    # ── Step 2: SAM2 segmentation with all fix phases ─────────────────
    tracker_logits = None
    try:
        pred = _get_sam2()
        rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pred.set_image(rgb)

        # --- Build SAM2 prompt arrays ---
        point_coords = []
        point_labels = []

        if tracker is not None:
            pos_pts = tracker.sample_positive_points(h, w, n=1)
            if pos_pts is not None:
                for pt in pos_pts:
                    point_coords.append(pt)
                    point_labels.append(1)

        if tracker is not None:
            neg_pts = tracker.sample_negative_points(h, w, NEG_POINT_COUNT)
            if neg_pts is not None:
                for pt in neg_pts:
                    point_coords.append(pt)
                    point_labels.append(0)
                log.debug("SAM2: +%d pos, +%d neg point prompts",
                          len(pos_pts) if pos_pts is not None else 0,
                          len(neg_pts))

        # Phase 2: Bootstrap grid points during first N frames
        is_bootstrap = (tracker is not None and
                        not tracker.warmed_up and
                        box_np is not None)
        if is_bootstrap:
            grid_pts = _generate_bootstrap_points(box_np, h, w)
            for pt in grid_pts:
                point_coords.append(pt)
                point_labels.append(1)
            log.info("Phase 2: added %d bootstrap grid points", len(grid_pts))

        sam_kwargs = dict(multimask_output=True, return_logits=True)

        if box_np is not None:
            sam_kwargs["box"] = box_np

        if point_coords:
            sam_kwargs["point_coords"] = np.array(point_coords, dtype=np.float32)
            sam_kwargs["point_labels"] = np.array(point_labels, dtype=np.int32)

        # --- Phase 1: Mask-input priority chain (FIXED) ---
        # Previously: fresh detection → drop ALL mask priors → SAM2 picks
        #   the wrong sub-region (the white strip).
        # Now: on fresh detection, if the live anchor is NOT yet confirmed,
        #   inject ref_logits as mask prior so SAM2 is biased toward the
        #   full object shape from the reference image.
        if need_detection and box_np is not None:
            if (tracker is not None and not tracker.anchor_is_live
                    and ref_logits is not None):
                # Phase 1: use ref_logits as shape prior alongside the box
                sam_kwargs["mask_input"]       = ref_logits
                sam_kwargs["multimask_output"]  = False
                log.info("Phase 1: fresh detection + ref_logits as mask prior "
                         "(anchor not yet live)")
            else:
                # Anchor is confirmed from live frames — trust the box
                sam_kwargs["multimask_output"] = True
                log.debug("SAM2: fresh detection — box only (live anchor "
                          "confirmed)")
        elif tracker is not None and tracker.prev_logits is not None:
            sam_kwargs["mask_input"]      = tracker.prev_logits
            sam_kwargs["multimask_output"] = False
        elif tracker is not None and tracker.anchor_logits is not None:
            sam_kwargs["mask_input"] = tracker.anchor_logits
            log.debug("SAM2: using anchor logits as prior")
        elif ref_logits is not None:
            sam_kwargs["mask_input"] = ref_logits
            log.info("SAM2: using reference-image logits as initial prior")

        # --- Run SAM2 ---
        masks, scores_sam, logits = pred.predict(**sam_kwargs)

        if masks is not None and len(masks) > 0:
            # Phase 5: ref-aware mask selection when multimask
            if sam_kwargs.get("multimask_output", True) and len(masks) > 1:
                best_idx = _select_best_mask_with_ref(
                    masks, scores_sam, logits, image_bgr, box_np,
                    ref_mask, ref_features)
            else:
                best_idx = int(np.argmax(scores_sam))

            mask_out = (masks[best_idx] > 0).astype(np.uint8) * 255

            # Phase 4: Adaptive coverage check
            if box_np is not None:
                bx1, by1, bx2, by2 = box_np.astype(int)
                box_area  = max(1, (bx2 - bx1) * (by2 - by1))
                mask_area = np.count_nonzero(mask_out)
                coverage  = mask_area / box_area
                # Adaptive threshold: 70% of the expected fill ratio
                adaptive_thresh = ref_fill_ratio * 0.70
                adaptive_thresh = max(0.20, min(adaptive_thresh, 0.60))
                if coverage < adaptive_thresh:
                    log.info("Phase 4: mask covers only %.0f%% of box "
                             "(threshold=%.0f%%) — re-running box-only",
                             coverage * 100, adaptive_thresh * 100)
                    pred.set_image(rgb)
                    masks2, scores2, logits2 = pred.predict(
                        box=box_np, multimask_output=True, return_logits=True)
                    if masks2 is not None and len(masks2) > 0:
                        # Phase 5 again on the re-run
                        best2 = _select_best_mask_with_ref(
                            masks2, scores2, logits2, image_bgr, box_np,
                            ref_mask, ref_features)
                        mask_out = (masks2[best2] > 0).astype(np.uint8) * 255
                        logits   = logits2
                        scores_sam = scores2
                        best_idx = best2
                        log.info("SAM2: box-only re-run mask score=%.3f",
                                 scores2[best2])

            # Drift check
            if tracker is not None and tracker.prev_mask is not None:
                iou = _mask_iou(mask_out, tracker.prev_mask)
                if iou < IOU_DRIFT_THRESH and not need_detection:
                    log.info("SAM2: mask IoU=%.2f < %.2f — drift detected, "
                             "resetting tracker (keeping anchor)",
                             iou, IOU_DRIFT_THRESH)
                    tracker.reset(keep_anchor=True)
                    return run_pipeline(image_bgr, prompt, tracker,
                                        ref_logits=ref_logits,
                                        ref_crop=ref_crop,
                                        ref_features=ref_features,
                                        ref_mask=ref_mask,
                                        ref_fill_ratio=ref_fill_ratio,
                                        ref_aspect_ratio=ref_aspect_ratio)

            res["mask_np"]   = mask_out
            tracker_logits   = logits[best_idx: best_idx + 1]
            log.info("SAM2: mask obtained (score=%.3f)", scores_sam[best_idx])
        else:
            log.warning("SAM2: no mask returned")

    except Exception as e:
        import traceback
        log.error("SAM2 failed: %s", e)
        traceback.print_exc()

    # ── Step 3: Depth ─────────────────────────────────────────────────
    dm = _get_depth_model()
    if dm is not None:
        try:
            res["depth_np"] = dm.infer_image(image_bgr).astype(np.float32)
        except Exception as e:
            log.error("Depth inference failed: %s", e)

    # ── Step 4: Grasp point ───────────────────────────────────────────
    if res["mask_np"] is not None:
        mc = _robust_centroid(res["mask_np"])
        if mc is not None:
            res["best_centroid"] = mc
            log.debug("Grasp point: robust centroid %s", mc)
        elif res["gdino_box"] is not None:
            x1, y1, x2, y2 = res["gdino_box"]
            res["best_centroid"] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            log.debug("Grasp point: bbox centre %s (centroid failed)",
                      res["best_centroid"])
    elif res["gdino_box"] is not None:
        x1, y1, x2, y2 = res["gdino_box"]
        res["best_centroid"] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        log.debug("Grasp point: bbox centre %s (no SAM2 mask)",
                  res["best_centroid"])

    # ── Step 5: Update tracker ────────────────────────────────────────
    if tracker is not None:
        tracker.update(res["mask_np"], tracker_logits,
                       res.get("best_centroid"))

    return res


# ═════════════════════════════════════════════════════════════════════
#  Optical-flow calibration helper
# ═════════════════════════════════════════════════════════════════════

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
    dx   = float(np.median(flow[:, 0, 0]))
    dy   = float(np.median(flow[:, 0, 1]))
    return dx, dy


# ═════════════════════════════════════════════════════════════════════
#  Robot controller
# ═════════════════════════════════════════════════════════════════════

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
        log.error("get_position failed: %s", ret)
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
                log.error("Robot not connected: %s", self.ip)
                return False
            arm.clean_error()
            arm.clean_warn()
            arm.motion_enable(True)
            arm.set_mode(0)
            arm.set_state(0)
            time.sleep(0.5)
            self._arm = arm
            self.enabled     = False
            self.cal_status  = "waiting for calibration..."
            log.info("Robot connected: %s", self.ip)
            return True
        except Exception as e:
            log.error("Robot connect failed: %s", e)
            return False

    def calibrate(self, get_frame_fn):
        self.cal_status = "waiting for frame..."
        log.info("=== Calibration: waiting for camera frame ===")

        deadline = time.time() + 30.0
        while time.time() < deadline:
            if get_frame_fn() is not None:
                break
            time.sleep(0.2)
        else:
            log.warning("Calibration skipped: no camera frame available.")
            self.cal_status = "skipped (no frame)"
            self.enabled    = True
            return

        pos0 = self._get_pos()
        if pos0 is None:
            log.warning("Calibration skipped: cannot read robot position.")
            self.cal_status = "skipped (pos read fail)"
            self.enabled    = True
            return

        log.info("Home: [%.1f, %.1f, %.1f] mm", pos0[0], pos0[1], pos0[2])
        J_yz = np.zeros((2, 2), dtype=np.float64)

        for col, (robot_idx, ax) in enumerate([(1, "Y"), (2, "Z")]):
            self.cal_status = f"calibrating {ax}..."
            time.sleep(0.3)
            frame_before = get_frame_fn()
            if frame_before is None:
                log.warning("  [%s] no frame before move — skipping axis", ax)
                continue

            log.info("  [%s] moving +%.0f mm...", ax, CAL_DELTA)
            fwd = list(pos0)
            fwd[robot_idx] += CAL_DELTA
            self._move_abs(fwd, wait=True)
            time.sleep(CAL_WAIT)

            frame_after = get_frame_fn()
            log.info("  [%s] returning home...", ax)
            self._move_abs(pos0, wait=True)

            if frame_after is None:
                log.warning("  [%s] no frame after move — skipping axis", ax)
                continue

            flow = _measure_flow(frame_before, frame_after)
            if flow is None:
                log.warning("  [%s] optical flow failed — skipping axis", ax)
                continue

            dpx, dpy = flow
            J_yz[0, col] = dpx / CAL_DELTA
            J_yz[1, col] = dpy / CAL_DELTA
            log.info("  [%s] flow: (%+.1f, %+.1f) px / %.0f mm  "
                     "→ J[:,{col}]=[%+.4f, %+.4f]",
                     ax, dpx, dpy, CAL_DELTA,
                     J_yz[0, col], J_yz[1, col])

        self._jac_yz  = J_yz
        log.info("J_yz (px/mm):\n%s", J_yz)
        rank = np.linalg.matrix_rank(J_yz)

        if rank == 0:
            log.warning("J_yz rank=0 — approach-only mode.")
            self.cal_status = "skipped (no flow data)"
            self.enabled    = True
            return

        self._jac_yz_inv = np.linalg.pinv(J_yz)
        log.info("J_yz_inv (mm/px):\n%s", self._jac_yz_inv)
        status = "calibrated" if rank == 2 else f"partial cal (rank={rank})"
        log.info("=== Calibration complete [%s] ===", status)
        self.cal_status = status
        self.enabled    = True

    def stop(self):
        self.enabled = False
        if self._arm is not None:
            try:
                self._arm.emergency_stop()
                log.info("Robot stopped.")
            except Exception as e:
                log.error("Robot stop failed: %s", e)

    def servo_step(self, centroid: tuple, image_shape: tuple):
        now = time.time()
        if self._arm is None:
            if now - self._last_t > 5.0:
                log.debug("Servo: arm not connected")
                self._last_t = now
            return
        if not self.enabled:
            if now - self._last_t > 5.0:
                log.debug("Servo: waiting [%s]", self.cal_status)
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
                log.warning("Servo: centroid jump %.0f px — ignored", jump)
                return
        self._last_centroid   = centroid
        self._last_centroid_t = now
        self._last_t          = now

        h, w       = image_shape[:2]
        ic_x, ic_y = w // 2, h // 2
        ex    = centroid[0] - ic_x
        ey    = centroid[1] - ic_y
        err_r = float(np.hypot(ex, ey))
        dx_mm = VS_APPROACH

        if err_r <= VS_DEAD_ZONE:
            dy_mm, dz_mm = 0.0, 0.0
        elif self._jac_yz_inv is not None:
            err   = np.array([ex, ey], dtype=np.float64)
            yz    = -CTRL_GAIN * (self._jac_yz_inv @ err)
            dy_mm = float(np.clip(yz[0], -MAX_YZ_STEP, MAX_YZ_STEP))
            dz_mm = float(np.clip(yz[1], -MAX_YZ_STEP, MAX_YZ_STEP))
        else:
            dy_mm, dz_mm = 0.0, 0.0

        with self._lock:
            try:
                pos = self._get_pos()
                if pos is None:
                    return

                new_x = pos[0] + dx_mm
                new_y = pos[1] + dy_mm
                new_z = pos[2] + dz_mm

                if new_z < MIN_Z_MM:
                    log.warning("Servo: clamping Z %.1f → %.1f mm (base limit)",
                                new_z, MIN_Z_MM)
                    new_z = MIN_Z_MM

                self._arm.set_position(
                    x=new_x, y=new_y, z=new_z,
                    roll=pos[3], pitch=pos[4], yaw=pos[5],
                    speed=VS_SPEED, mvacc=VS_MVACC, wait=True)
                log.info("Servo: err=(%+.0f,%+.0f)px r=%.0f  "
                         "Δ=(%+.1f,%+.1f,%+.1f)mm  "
                         "pos=(%.0f,%.0f,%.0f)  [%s]",
                         ex, ey, err_r,
                         dx_mm, dy_mm, dz_mm,
                         new_x, new_y, new_z,
                         self.cal_status)
            except Exception as e:
                log.error("Servo step failed: %s", e)


# ═════════════════════════════════════════════════════════════════════
#  Camera streamer  (PyZED primary, OpenCV fallback)
# ═════════════════════════════════════════════════════════════════════

class CameraStreamer(threading.Thread):
    """
    Captures frames from the ZED camera (via PyZED or OpenCV),
    runs the segmentation pipeline in a background thread, overlays
    results, and records both a raw SVO feed (pyzed) and an annotated
    MP4 at ≥ MIN_RECORDING_DIM px.
    """

    def __init__(self, cam_index: int, stop_event: threading.Event,
                 robot: RobotController,
                 ref_image_path: str | None = None,
                 use_pyzed: bool = True):
        super().__init__(daemon=True)
        self.cam_index      = cam_index
        self.stop_event     = stop_event
        self.robot          = robot
        self.ref_image_path = ref_image_path
        self._use_pyzed     = use_pyzed and PYZED_AVAILABLE

        self._latest_left  = None
        self._frame_lock   = threading.Lock()
        self.data_lock     = threading.Lock()
        self._result: dict = {}
        self._models_ready = threading.Event()
        self._tracker      = MaskTracker()

        self._undistorter = None

        # Populated by _process_ref_image()
        self.ref_logits       = None
        self.ref_crop         = None
        self.ref_features     = None
        self.ref_mask         = None
        self.ref_fill_ratio   = 0.7
        self.ref_aspect_ratio = 1.0

    # ── helpers ──────────────────────────────────────────────────────
    def _get_frame(self):
        with self._frame_lock:
            if self._latest_left is not None:
                return self._latest_left.copy()
            return None

    def _run_calibration(self):
        log.info("Calibration thread: waiting for models …")
        self._models_ready.wait()
        log.info("Calibration thread: starting Y/Z Jacobian calibration.")
        self.robot.calibrate(self._get_frame)

    # ── reference image pre-processing ────────────────────────────────
    def _process_ref_image(self):
        if self.ref_image_path is None:
            return
        try:
            ref_bgr, fg_mask = _load_ref_image(self.ref_image_path)
            if ref_bgr is None:
                return

            (self.ref_crop, self.ref_logits, self.ref_mask,
             self.ref_features, self.ref_fill_ratio,
             self.ref_aspect_ratio) = process_ref_image(ref_bgr, fg_mask,
                                                         PROMPT)

            # Phase 6: warm-start the tracker from ref logits
            self._tracker.seed_from_ref(self.ref_logits, self.ref_mask)

        except Exception as e:
            log.error("Reference image processing error: %s", e)
            import traceback; traceback.print_exc()

    # ── segmentation loop ─────────────────────────────────────────────
    def _seg_loop(self):
        self._process_ref_image()

        while not self.stop_event.is_set():
            with self._frame_lock:
                frame = self._latest_left.copy() if self._latest_left is not None else None
            if frame is None:
                time.sleep(0.1)
                continue
            try:
                res = run_pipeline(frame, PROMPT, self._tracker,
                                   ref_logits=self.ref_logits,
                                   ref_crop=self.ref_crop,
                                   ref_features=self.ref_features,
                                   ref_mask=self.ref_mask,
                                   ref_fill_ratio=self.ref_fill_ratio,
                                   ref_aspect_ratio=self.ref_aspect_ratio)
                with self.data_lock:
                    self._result = res
                if not self._models_ready.is_set():
                    log.info("Models ready — calibration may now proceed.")
                    self._models_ready.set()
                if self.robot is not None and res.get("best_centroid") is not None:
                    self.robot.servo_step(res["best_centroid"], frame.shape)
            except Exception as e:
                log.error("Pipeline error: %s", e)
                import traceback; traceback.print_exc()
                time.sleep(1)

    # ── top-level dispatch ────────────────────────────────────────────
    def run(self):
        if self._use_pyzed:
            self._run_pyzed()
        else:
            self._run_opencv()

    # ── PyZED path ────────────────────────────────────────────────────
    def _run_pyzed(self):
        cam = sl.Camera()

        init_params = sl.InitParameters()
        res_map = {
            "HD2K":  sl.RESOLUTION.HD2K,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720":  sl.RESOLUTION.HD720,
            "VGA":    sl.RESOLUTION.VGA,
        }
        init_params.camera_resolution = res_map.get(ZED_RESOLUTION,
                                                     sl.RESOLUTION.HD720)
        init_params.camera_fps        = 30
        init_params.depth_mode        = sl.DEPTH_MODE.NONE

        err = cam.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            log.warning("PyZED: camera open failed (%s) — falling back to OpenCV", err)
            self._run_opencv()
            return

        ts = time.strftime("%Y%m%d_%H%M%S")

        svo_path   = os.path.abspath(f"vs_recording_{ts}.svo2")
        rec_params = sl.RecordingParameters()
        rec_params.video_filename    = svo_path
        rec_params.compression_mode  = sl.SVO_COMPRESSION_MODE.H264
        err = cam.enable_recording(rec_params)
        if err == sl.ERROR_CODE.SUCCESS:
            log.info("PyZED SVO recording → %s", svo_path)
        else:
            log.warning("PyZED SVO recording failed (%s)", err)

        anno_path    = os.path.abspath(f"vs_annotated_{ts}.mp4")
        video_writer = None

        mat            = sl.Mat()
        runtime_params = sl.RuntimeParameters()

        threading.Thread(target=self._seg_loop, daemon=True).start()
        if self.robot is not None and self.robot._arm is not None:
            threading.Thread(target=self._run_calibration, daemon=True).start()

        win = "Grasp point  |  [v] servo  [r] reset tracker  [q] quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        try:
            while not self.stop_event.is_set():
                if cam.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                    time.sleep(0.01)
                    continue

                cam.retrieve_image(mat, sl.VIEW.LEFT)
                frame_bgra = mat.get_data()
                frame      = frame_bgra[:, :, :3].copy()

                with self._frame_lock:
                    self._latest_left = frame.copy()
                with self.data_lock:
                    res = dict(self._result)

                rendered       = self._render(frame, res)
                rendered_write = self._ensure_min_dim(rendered)

                if video_writer is None:
                    rh, rw = rendered_write.shape[:2]
                    video_writer = cv2.VideoWriter(
                        anno_path, cv2.VideoWriter_fourcc(*"mp4v"),
                        30.0, (rw, rh))
                    if not video_writer.isOpened():
                        log.error("VideoWriter failed for %s (%d×%d)",
                                  anno_path, rw, rh)
                        video_writer.release()
                        video_writer = None
                    else:
                        log.info("Annotated recording → %s  (%d×%d)",
                                 anno_path, rw, rh)

                if video_writer is not None:
                    video_writer.write(rendered_write)
                cv2.imshow(win, rendered)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.stop_event.set()
                    break
                elif key == ord("v") and self.robot is not None:
                    self.robot.enabled = not self.robot.enabled
                    log.info("Servo: %s", "ON" if self.robot.enabled else "OFF")
                elif key == ord("r"):
                    self._tracker.reset(keep_anchor=True)
                    log.info("Tracker soft-reset (anchor kept)")
                elif key == ord("R"):
                    self._tracker.reset(keep_anchor=False)
                    log.info("Tracker FULL reset (anchor cleared)")
        finally:
            cam.disable_recording()
            cam.close()
            if video_writer is not None:
                video_writer.release()
                log.info("Annotated recording saved: %s", anno_path)
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # ── OpenCV fallback path ──────────────────────────────────────────
    def _run_opencv(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_ANY)
        if not cap.isOpened():
            log.error("Failed to open camera %d", self.cam_index)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  2560)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        threading.Thread(target=self._seg_loop, daemon=True).start()
        if self.robot is not None and self.robot._arm is not None:
            threading.Thread(target=self._run_calibration, daemon=True).start()

        win = "Grasp point  |  [v] servo  [r] reset tracker  [q] quit"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        video_writer = None
        video_path   = None
        ts           = time.strftime("%Y%m%d_%H%M%S")

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                h, w = frame.shape[:2]
                left = frame[:, : w // 2].copy()

                if self._undistorter is None:
                    lh, lw = left.shape[:2]
                    self._undistorter = ZedUndistorter.from_frame_size(lw, lh)
                left = self._undistorter.undistort(left)

                with self._frame_lock:
                    self._latest_left = left.copy()
                with self.data_lock:
                    res = dict(self._result)

                rendered       = self._render(left, res)
                rendered_write = self._ensure_min_dim(rendered)

                if video_writer is None:
                    rh, rw     = rendered_write.shape[:2]
                    video_path = os.path.abspath(f"vs_recording_{ts}.mp4")
                    video_writer = cv2.VideoWriter(
                        video_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        30.0, (rw, rh))
                    if not video_writer.isOpened():
                        log.error("VideoWriter failed for %s (%d×%d)",
                                  video_path, rw, rh)
                        video_writer.release()
                        video_writer = None
                    else:
                        log.info("Recording → %s  (%d×%d)", video_path, rw, rh)

                if video_writer is not None:
                    video_writer.write(rendered_write)
                cv2.imshow(win, rendered)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.stop_event.set()
                    break
                elif key == ord("v") and self.robot is not None:
                    self.robot.enabled = not self.robot.enabled
                    log.info("Servo: %s", "ON" if self.robot.enabled else "OFF")
                elif key == ord("r"):
                    self._tracker.reset(keep_anchor=True)
                    log.info("Tracker soft-reset (anchor kept)")
                elif key == ord("R"):
                    self._tracker.reset(keep_anchor=False)
                    log.info("Tracker FULL reset (anchor cleared)")
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
                if video_path:
                    log.info("Recording saved: %s", video_path)
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # ── video helpers ─────────────────────────────────────────────────
    @staticmethod
    def _ensure_min_dim(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        short = min(h, w)
        if short < MIN_RECORDING_DIM:
            scale = MIN_RECORDING_DIM / short
            img   = cv2.resize(img, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_LINEAR)
        return img

    # ── overlay renderer ──────────────────────────────────────────────
    def _render(self, left: np.ndarray, res: dict) -> np.ndarray:
        display = left.copy()
        h, w    = display.shape[:2]

        mask_np       = res.get("mask_np")
        depth_np      = res.get("depth_np")
        best_centroid = res.get("best_centroid")
        gdino_box     = res.get("gdino_box")

        mask_np_r = mask_np
        if mask_np_r is not None and mask_np_r.shape[:2] != (h, w):
            mask_np_r = cv2.resize(mask_np_r, (w, h),
                                   interpolation=cv2.INTER_NEAREST)

        if mask_np_r is not None:
            overlay           = np.zeros_like(display)
            overlay[:, :, 1]  = mask_np_r
            display           = cv2.addWeighted(display, 0.72, overlay, 0.28, 0)
            cnts, _ = cv2.findContours(mask_np_r, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, cnts, -1, (0, 255, 0), 2)

        if self._tracker.anchor_locked and self._tracker.anchor_mask is not None:
            am_r = self._tracker.anchor_mask
            if am_r.shape[:2] != (h, w):
                am_r = cv2.resize(am_r, (w, h), interpolation=cv2.INTER_NEAREST)
            a_cnts, _ = cv2.findContours(am_r, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, a_cnts, -1, (255, 200, 0), 1)

        if gdino_box is not None:
            x1, y1, x2, y2 = gdino_box.astype(int)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)

        ic = (w // 2, h // 2)
        cv2.drawMarker(display, ic, (255, 80, 0), cv2.MARKER_CROSS, 22, 2)

        if best_centroid is not None:
            gp = best_centroid
            cv2.circle(display, gp, 18, (0, 0, 255), 2)
            cv2.drawMarker(display, gp, (0, 0, 255), cv2.MARKER_CROSS, 26, 2)
            cv2.putText(display, "GRASP", (gp[0] + 22, gp[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)
            cv2.arrowedLine(display, ic, gp, (0, 220, 255), 2, tipLength=0.12)
            ex = gp[0] - ic[0]
            ey = gp[1] - ic[1]
            cv2.putText(display, f"err ({ex:+d},{ey:+d}) px",
                        (8, h - 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (0, 220, 255), 1)

        trk     = self._tracker
        iou_txt = f"  IoU={trk._last_iou:.2f}" if trk.frame_count > 1 else ""
        anc_src = ""
        if trk.anchor_locked:
            anc_src = f"  [ANCHOR:{trk._anchor_source or '?'}]"
        trk_txt = (f"trk: {trk.frame_count} frm"
                   f"{'  warmed' if trk.warmed_up else ''}"
                   f"{iou_txt}{anc_src}")
        cv2.putText(display, trk_txt, (8, h - 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 180, 0), 1)

        if self.ref_logits is not None:
            sim_val = res.get("similarity")
            ref_txt = "REF" if sim_val is None else f"REF sim={sim_val:.2f}"
            cv2.putText(display, ref_txt, (8, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 255), 1)

        if self.robot is not None:
            cal = self.robot.cal_status
            if not self.robot.enabled:
                txt, col = f"SERVO OFF  [{cal}]", (80, 80, 80)
            else:
                txt, col = f"SERVO ON  [{cal}]", (0, 255, 80)
            cv2.putText(display, txt, (8, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 1)

        cv2.putText(display, DETECTOR.upper(), (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 0), 1)

        if depth_np is not None:
            try:
                d = depth_np.astype(np.float32)
                d_mn, d_mx = float(np.nanmin(d)), float(np.nanmax(d))
                d_u8  = ((d - d_mn) / (d_mx - d_mn + 1e-6) * 255).astype(np.uint8)
                cmap  = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
                iw    = max(120, w // 4)
                ih    = int(iw * h / w)
                inset = cv2.resize(cmap, (iw, ih))
                display[10: 10 + ih, w - iw - 10: w - 10] = inset
                cv2.putText(display, "depth", (w - iw - 8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            except Exception:
                pass

        return display


# ═════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SemVS: Semantic Visual Servoing with image-conditioned "
                    "detection (suction-cup gripper)")
    parser.add_argument("cam_index", nargs="?", type=int, default=0,
                        help="OpenCV camera index (default: 0)")
    parser.add_argument("--detector", choices=["gdino", "owlv2"], default=None,
                        help="Override detector (default: env DETECTOR or gdino)")
    parser.add_argument("--ref-image", "--ref_image", metavar="PATH",
                        help="Path to a reference image of the target object "
                             "(supports transparent PNG or black background)")
    parser.add_argument("--no-pyzed", action="store_true",
                        help="Force OpenCV camera capture even if PyZED is "
                             "available")
    parser.add_argument("--prompt", default=None,
                        help=f"Text prompt for the detector (default: '{PROMPT}')")
    parser.add_argument("--log-dir", default="logs",
                        help="Directory for log files (default: logs)")
    args = parser.parse_args()

    if args.detector:
        DETECTOR = args.detector
    if args.prompt:
        PROMPT = args.prompt

    log.info("Detector : %s", DETECTOR.upper())
    log.info("Prompt   : %s", PROMPT)
    log.info("PyZED    : %s%s",
             "available" if PYZED_AVAILABLE else "not available",
             " (disabled by --no-pyzed)" if args.no_pyzed else "")

    if args.ref_image:
        if not os.path.exists(args.ref_image):
            log.error("Reference image not found: %s", args.ref_image)
            sys.exit(1)
        log.info("Reference image: %s", args.ref_image)
    else:
        log.info("No reference image provided — using text-prompt detection only")

    robot = RobotController(ROBOT_IP)
    robot.connect()

    stop_ev    = threading.Event()
    cam_thread = CameraStreamer(
        cam_index      = args.cam_index,
        stop_event     = stop_ev,
        robot          = robot,
        ref_image_path = args.ref_image,
        use_pyzed      = not args.no_pyzed,
    )
    cam_thread.start()

    try:
        cam_thread.join()
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    finally:
        robot.stop()
        stop_ev.set()
        cam_thread.join(timeout=2)
        log.info("Done.")
