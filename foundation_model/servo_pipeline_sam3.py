#!/usr/bin/env python3
"""
SemVS - Semantic Visual Servoing (SAM 3 edition)
Single-path pipeline using Meta Segment Anything Model 3 (Promptable
Concept Segmentation):

  SAM3 (text prompt -> all instance masks + boxes + scores in one call)
  -> reference-guided disambiguation
  -> depth (Depth-Anything-V2)
  -> grasp point.

This is a drop-in replacement for the GroundingDINO/OWLv2 + SAM2 cascade
in servo_pipeline.py. The detector and segmenter are now a single model
(facebook/sam3 via HuggingFace transformers), so:
  - no synonym padding ("box | cardboard box | package | ...")
  - no threshold-retry hack
  - no separate SAM2 mask call
  - concept-level grounding, which is more robust to printed graphics
    on package surfaces being mistaken for the box itself.

The reference image is used for:
  * a ResNet-18 feature vector to disambiguate between SAM3 instance
    candidates when more than one box-like object is present,
  * (optionally) a crop seed for image-conditioned modes (kept for
    forward compatibility; not required for SAM3 PCS).

The result-dict keys are identical to servo_pipeline.py so the existing
debug dump, JSONL summary, render, offline runner and downstream tooling
all keep working unchanged. The "gdino_box" key is preserved by name for
that reason; it now holds the chosen SAM3 box.

Usage:
  Offline single image:
    python foundation_model/servo_pipeline_sam3.py \
      --ref-image path/to/target_box_reference.png \
      --input-image path/to/frame.png \
      --output-dir runs/offline_smoke_sam3

  Offline video / frames:
    python foundation_model/servo_pipeline_sam3.py --ref-image REF \
      --input-video VIDEO --output-dir OUT
    python foundation_model/servo_pipeline_sam3.py --ref-image REF \
      --input-dir DIR --output-dir OUT

  Live camera (perception only / with robot):
    python foundation_model/servo_pipeline_sam3.py --ref-image REF --dry-run
    python foundation_model/servo_pipeline_sam3.py --ref-image REF
"""
import time
import sys
import threading
import os
import logging
import json

import cv2
import numpy as np

# =====================================================================
#  Logging setup (identical format to servo_pipeline.py)
# =====================================================================

def _setup_logger(name: str = "semvs",
                  log_dir: str = "logs",
                  level: int = logging.DEBUG) -> logging.Logger:
    """
    Create a logger that writes to both console (INFO+) and a
    timestamped log file (DEBUG+).
    """
    os.makedirs(log_dir, exist_ok=True)
    ts       = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"semvs_sam3_{ts}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-5s] %(message)s",
        datefmt="%H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)-5s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(fh)

    logger.info(f"Logging to {os.path.abspath(log_path)}")
    logger._semvs_log_path = os.path.abspath(log_path)
    return logger

log = logging.getLogger("semvs")
log.addHandler(logging.NullHandler())

# =====================================================================
#  Configuration
# =====================================================================

THIRD_PARTY_ROOT = os.path.join(os.path.dirname(__file__), "third-party")

PROMPT = "box"

# Detector identifier kept for log/debug parity with servo_pipeline.py.
# SAM3 is the only path here; the constant lets the existing render and
# debug-dump tooling tag frames with a detector name.
DETECTOR = "sam3"

# SAM3 model selection
SAM3_HF_MODEL_ID  = os.environ.get("SAM3_HF_MODEL_ID", "facebook/sam3")
SAM3_SCORE_THRESH = float(os.environ.get("SAM3_SCORE_THRESH", "0.5"))
SAM3_MASK_THRESH  = float(os.environ.get("SAM3_MASK_THRESH",  "0.5"))
# Permissive retry, mirrors the GDINO 0.25 -> 0.18 fallback in the original
SAM3_SCORE_THRESH_LO = float(os.environ.get("SAM3_SCORE_THRESH_LO", "0.30"))
SAM3_MASK_THRESH_LO  = float(os.environ.get("SAM3_MASK_THRESH_LO",  "0.40"))

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

MIN_Z_MM     = 0.0

# Mask propagation / tracking parameters
TRACKER_WARMUP_FRAMES = 10
TRACKER_MAX_HISTORY   = 15
NEG_POINT_COUNT       = 5
NEG_POINT_MARGIN_PX   = 30
# In the SAM3 pipeline, detection and segmentation are coupled in one
# forward pass, so REDETECT_INTERVAL is used to enable a fast box-prior
# fast path: when (frame_count % REDETECT_INTERVAL != 0) and a prior box
# exists, SAM3 is reprompted with that box as a positive visual prompt
# instead of running open-vocabulary detection from scratch.
REDETECT_INTERVAL     = 1
IOU_DRIFT_THRESH      = 0.35

# Anchor / stable-mask parameters
ANCHOR_IOU_THRESH  = 0.82
ANCHOR_LOCK_FRAMES = 5

# Video recording
MIN_RECORDING_DIM = 512
ZED_RESOLUTION    = "HD720"


# ZED calibration for lens undistortion (OpenCV path)
ZED_SETTINGS_DIRS = [
    "/usr/local/zed/settings/",
    os.path.expanduser("~/.ZED/settings/"),
]

def _load_zed_calibration(resolution: str = "HD"):
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
            log.warning("ZED undistortion: calibration not found - frames will "
                        "NOT be undistorted")
            return

        w, h = img_size
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            K, D, None, new_K, (w, h), cv2.CV_16SC2)
        self._new_K = new_K
        log.info("ZED undistortion: maps ready (%dx%d)", w, h)

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
            log.debug("ZedUndistorter: no calibration section for %dx%d "
                      "- trying closest match", width, height)
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


# PyZED
try:
    import pyzed.sl as sl
    PYZED_AVAILABLE = True
    log.info("PyZED available - will use ZED SDK for capture and SVO recording")
except ImportError:
    sl               = None
    PYZED_AVAILABLE  = False

DEVICE = os.environ.get("DEVICE")


def _device() -> str:
    global DEVICE
    if DEVICE is None:
        try:
            import torch as _t
            DEVICE = "cuda" if _t.cuda.is_available() else "cpu"
        except Exception:
            DEVICE = "cpu"
    return DEVICE


# =====================================================================
#  Geometry helpers
# =====================================================================

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
    Robust centroid for box-like objects: morphological close, largest
    external contour, minAreaRect centre, with hull-moment fallback.
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


# =====================================================================
#  MaskTracker - temporal state for inter-frame mask propagation
# =====================================================================

class MaskTracker:
    """
    Maintains inter-frame state for mask propagation: previous mask,
    last detected box, anchor mask, and IoU streak for drift detection.
    Anchors are derived only from live-frame masks; the reference image
    is never used as a spatial prior on live frames.
    """

    def __init__(self):
        self.prev_logits:    np.ndarray | None = None
        self.prev_mask:      np.ndarray | None = None
        self.prev_centroid:  tuple | None      = None
        self.last_box:       np.ndarray | None = None
        self.frame_count:    int               = 0
        self.mask_history:   list              = []

        self.anchor_logits:  np.ndarray | None = None
        self.anchor_mask:    np.ndarray | None = None
        self._iou_streak:    int               = 0
        self._last_iou:      float             = 1.0

        self._anchor_source: str | None        = None

    def reset(self, keep_anchor: bool = True):
        self.prev_logits    = None
        self.prev_mask      = None
        self.prev_centroid  = None
        self.last_box       = None
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
        if self._iou_streak >= ANCHOR_LOCK_FRAMES:
            self.anchor_logits  = logits.copy() if logits is not None else None
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
        return self.anchor_mask is not None

    @property
    def anchor_is_live(self) -> bool:
        return self._anchor_source == "live"

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

    def sample_positive_points(self, h: int, w: int,
                               n: int = 1) -> np.ndarray | None:
        if self.prev_mask is None:
            return None

        pts = []
        if self.prev_centroid is not None:
            pts.append(list(self.prev_centroid))

        if n > len(pts):
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


# =====================================================================
#  Model singletons
# =====================================================================

_sam3_processor = None
_sam3_model     = None
_sam3_failed    = False
_depth_model    = None


def _get_sam3():
    """
    Load SAM 3 (facebook/sam3) via HuggingFace transformers. Returns
    (processor, model) or (None, None) on any failure (the caller
    degrades gracefully so the pipeline still produces a frame).

    Override the model with: SAM3_HF_MODEL_ID env var
      "facebook/sam3"   (default)
      "facebook/sam3.1"

    NOTE: facebook/sam3 and facebook/sam3.1 are GATED repos. You must:
      1. Visit https://huggingface.co/facebook/sam3 and accept the
         license ("Agree and access repository").
      2. Authenticate locally:
           pip install -U "huggingface_hub[cli]"
           huggingface-cli login   # paste an HF token
         or set HF_TOKEN in the environment.
    Without this, the from_pretrained call returns 401 Client Error.
    """
    global _sam3_processor, _sam3_model, _sam3_failed
    if _sam3_failed:
        return None, None
    if _sam3_model is None:
        try:
            from transformers import Sam3Processor, Sam3Model
            _sam3_processor = Sam3Processor.from_pretrained(SAM3_HF_MODEL_ID)
            _sam3_model = (Sam3Model
                           .from_pretrained(SAM3_HF_MODEL_ID)
                           .to(_device()).eval())
            log.info("SAM3 loaded: %s on %s",
                     SAM3_HF_MODEL_ID, _device())
        except Exception as e:
            msg = str(e)
            log.error("SAM3 load failed (won't retry): %s", e)
            if ("gated" in msg.lower() or "401" in msg
                    or "restricted" in msg.lower()):
                log.error(
                    "SAM3 hint: '%s' is a gated HuggingFace repo. "
                    "1) Accept the license at "
                    "https://huggingface.co/%s  "
                    "2) Run `huggingface-cli login` (or set HF_TOKEN). "
                    "Then re-run the pipeline.",
                    SAM3_HF_MODEL_ID, SAM3_HF_MODEL_ID)
            _sam3_failed = True
            return None, None
    return _sam3_processor, _sam3_model


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
            _depth_model = m.to(_device()).eval()
        except Exception as e:
            log.error("Depth-Anything-V2 load failed: %s", e)
    return _depth_model


# =====================================================================
#  Feature extractor for visual similarity (kept for ref disambiguation)
# =====================================================================

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
        ).to(_device()).eval()

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
        tensor = transform(rgb).unsqueeze(0).to(_device())
        with _t.no_grad():
            feat = model(tensor).squeeze()
        feat = feat / (feat.norm() + 1e-8)
        return feat.cpu().numpy()
    except Exception as e:
        log.debug("Feature extraction failed: %s", e)
        return None


def _cosine_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    return float(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8))


# =====================================================================
#  Reference-image conditioning
# =====================================================================

def _load_ref_image(path: str):
    """Load a reference image as 3-channel BGR. Strips alpha if present."""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        log.error("Could not load reference image: %s", path)
        return None
    if raw.ndim == 2:
        return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    if raw.shape[2] == 4:
        return raw[:, :, :3].copy()
    return raw


def process_ref_image(ref_bgr: np.ndarray, prompt: str):
    """
    Reference-image preprocessing using SAM3 itself for the detection
    step (replaces the GDino/OWLv2 + ResNet path on the reference).

    Runs SAM3 on the reference image with the given concept prompt,
    picks the most prominent instance (largest area, score-weighted),
    crops it with a small padding margin, and extracts a ResNet-18
    feature vector from the crop. The crop is also returned in case a
    downstream caller wants an image exemplar.

    Returns
    -------
    ref_crop     : np.ndarray | None  (BGR crop of the reference object)
    ref_features : np.ndarray | None  (512-d ResNet-18 feature vector)
    """
    log.info("Processing reference image ...")
    h, w = ref_bgr.shape[:2]

    dets, masks = _detect_sam3(ref_bgr, prompt)

    ref_box = None
    if dets:
        # Score = SAM3 confidence * sqrt(area_fraction): prefer prominent,
        # confidently-segmented instances over tiny high-score artifacts.
        scored = []
        img_area = float(h * w)
        for i, (box, score, label) in enumerate(dets):
            x1, y1, x2, y2 = box
            area_frac = max(0.0, ((x2 - x1) * (y2 - y1)) / img_area)
            scored.append((score * (area_frac ** 0.5), i, box, score, label))
        scored.sort(key=lambda t: t[0], reverse=True)
        _, idx, ref_box, ref_score, ref_label = scored[0]
        log.info("Ref image: detected '%s' score=%.2f @ %s",
                 ref_label, ref_score, np.asarray(ref_box).astype(int))

    if ref_box is not None:
        x1, y1, x2, y2 = np.asarray(ref_box).astype(int)
        pad = 10
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        ref_crop = ref_bgr[y1:y2, x1:x2].copy()
    else:
        log.info("Ref image: no detection - using full image as crop")
        ref_crop = ref_bgr.copy()

    ref_features = _extract_features(ref_crop)
    if ref_features is not None:
        log.info("Ref features: 512-d vector (norm=%.3f)",
                 np.linalg.norm(ref_features))
    else:
        log.warning("Ref features: extraction failed - similarity disabled")

    return ref_crop, ref_features


# =====================================================================
#  SAM3 detection + segmentation (single forward pass)
# =====================================================================

def _sam3_run(image_bgr: np.ndarray,
              prompt: str,
              prior_box_xyxy: np.ndarray | None = None,
              prior_label: int = 1,
              score_thresh: float | None = None,
              mask_thresh: float | None = None):
    """
    Single SAM3 forward pass. Returns the post-processed instance dict
    {"masks": Tensor[N,H,W] bool, "boxes": Tensor[N,4] xyxy abs px,
     "scores": Tensor[N]} or None on any failure.

    `prior_box_xyxy` is an optional in-image visual prompt (positive box
    by default) that constrains attention to a known prior region. It is
    not a cross-image exemplar; that is handled by the ResNet reranker.
    """
    processor, model = _get_sam3()
    if model is None:
        return None

    if score_thresh is None:
        score_thresh = SAM3_SCORE_THRESH
    if mask_thresh is None:
        mask_thresh = SAM3_MASK_THRESH

    try:
        import torch as _t
        from PIL import Image

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        kwargs = dict(images=pil, return_tensors="pt")
        caption = (prompt or "").strip().rstrip(".")
        if caption:
            kwargs["text"] = caption
        if prior_box_xyxy is not None:
            box_list = [[float(v) for v in np.asarray(prior_box_xyxy).flatten()[:4]]]
            kwargs["input_boxes"] = [box_list]
            kwargs["input_boxes_labels"] = [[int(prior_label)]]

        inputs = processor(**kwargs).to(_device())

        with _t.no_grad():
            outputs = model(**inputs)

        orig_sizes = inputs.get("original_sizes")
        if orig_sizes is not None and hasattr(orig_sizes, "tolist"):
            target_sizes = orig_sizes.tolist()
        elif orig_sizes is not None:
            target_sizes = list(orig_sizes)
        else:
            target_sizes = [[image_bgr.shape[0], image_bgr.shape[1]]]

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=float(score_thresh),
            mask_threshold=float(mask_thresh),
            target_sizes=target_sizes,
        )[0]
        return results
    except Exception as e:
        import traceback
        log.error("SAM3 forward failed: %s", e)
        traceback.print_exc()
        return None


_sam3_unavailable_warned = False


def _detect_sam3(image_bgr: np.ndarray, prompt: str,
                 prior_box_xyxy: np.ndarray | None = None):
    """
    Runs SAM3 PCS and returns aligned (dets, masks) where:
      dets  : list of (box_xyxy_np, score, label) - one per instance
      masks : list of uint8 0/255 binary masks of shape (H, W),
              i-th mask aligns with i-th det.

    Includes a permissive-threshold retry if the first pass returns
    nothing, mirroring the GDINO 0.25 -> 0.18 fallback in the original.

    Short-circuits silently after the first per-frame call once SAM3 has
    permanently failed to load (e.g. gated-repo 401), so the log doesn't
    spam "empty / retrying / no detections" on every frame.
    """
    global _sam3_unavailable_warned
    caption = (prompt or "box").strip() or "box"
    h, w = image_bgr.shape[:2]

    if _sam3_failed:
        if not _sam3_unavailable_warned:
            log.warning("SAM3 unavailable for this run (load failed). "
                        "Returning no detections; per-frame retries silenced.")
            _sam3_unavailable_warned = True
        return [], []

    results = _sam3_run(image_bgr, caption, prior_box_xyxy=prior_box_xyxy)

    def _empty(r):
        return (r is None
                or "masks" not in r or r["masks"] is None
                or len(r["masks"]) == 0)

    if _empty(results):
        # If the first call failed because the model just got marked as
        # permanently unavailable, don't run the retry pass.
        if _sam3_failed:
            return [], []
        log.info("SAM3: empty at %.2f/%.2f, retrying at %.2f/%.2f",
                 SAM3_SCORE_THRESH, SAM3_MASK_THRESH,
                 SAM3_SCORE_THRESH_LO, SAM3_MASK_THRESH_LO)
        results = _sam3_run(image_bgr, caption,
                            prior_box_xyxy=prior_box_xyxy,
                            score_thresh=SAM3_SCORE_THRESH_LO,
                            mask_thresh=SAM3_MASK_THRESH_LO)

    if _empty(results):
        log.warning("SAM3: no detections for prompt '%s'", caption)
        return [], []

    def _to_numpy(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    boxes_np  = _to_numpy(results["boxes"]).astype(np.float32)
    scores_np = _to_numpy(results["scores"]).astype(np.float32)
    masks_raw = _to_numpy(results["masks"])

    masks_list: list[np.ndarray] = []
    for m in masks_raw:
        m_bin = (m > 0).astype(np.uint8) * 255
        if m_bin.shape[:2] != (h, w):
            m_bin = cv2.resize(m_bin, (w, h), interpolation=cv2.INTER_NEAREST)
        masks_list.append(m_bin)

    dets = [(boxes_np[i], float(scores_np[i]), caption)
            for i in range(len(boxes_np))]
    log.info("SAM3: %d detection(s) for '%s'", len(dets), caption)
    return dets, masks_list


# =====================================================================
#  Box disambiguation (identical scoring to servo_pipeline.py)
# =====================================================================

def _disambiguate_top_box(dets: list, image_shape: tuple,
                          prev_centroid: tuple | None = None,
                          image_bgr: np.ndarray | None = None,
                          ref_features: np.ndarray | None = None,
                          return_index: bool = False):
    """
    Pick the best detection among several candidates. Same combined
    scoring as servo_pipeline.py: position (top of image), confidence,
    proximity to prior centroid, and ResNet-18 cosine similarity to the
    reference crop when available.
    """
    if not dets:
        return None

    h, w = image_shape[:2]
    img_area = h * w

    boxes  = np.array([d[0] for d in dets], dtype=np.float32)
    scores = np.array([d[1] for d in dets], dtype=np.float32)
    keep   = _nms(boxes, scores, iou_thresh=0.45)
    kept_dets = [(dets[i], i) for i in keep]
    if not kept_dets:
        return None

    filtered = []
    for det, orig_i in kept_dets:
        bx = det[0]
        bw, bh = bx[2] - bx[0], bx[3] - bx[1]
        area_ratio = (bw * bh) / img_area
        if area_ratio < 0.003 or area_ratio > 0.75:
            continue
        aspect = max(bw, bh) / (min(bw, bh) + 1e-6)
        if aspect > 8.0:
            continue
        filtered.append((det, orig_i))

    if not filtered:
        det, orig_i = min(kept_dets, key=lambda t: t[0][0][1])
        return (orig_i, det) if return_index else det

    use_similarity = (ref_features is not None and image_bgr is not None)

    best, best_orig_i, best_score = None, None, -1e9
    for det, orig_i in filtered:
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
            log.debug("  det conf=%.2f sim=%.3f pos=%.2f prox=%.2f -> %.3f",
                      conf, sim_score, pos_score, prox_score, combined)
        else:
            combined = pos_score * 0.50 + conf * 0.25 + prox_score * 0.25

        if combined > best_score:
            best_score  = combined
            best        = det
            best_orig_i = orig_i

    if best is None:
        return None
    return (best_orig_i, best) if return_index else best


# =====================================================================
#  Core pipeline
# =====================================================================

def _box_center(box_np: np.ndarray, h: int, w: int) -> np.ndarray:
    x1, y1, x2, y2 = box_np
    cx = float(min(max((x1 + x2) * 0.5, 0), w - 1))
    cy = float(min(max((y1 + y2) * 0.5, 0), h - 1))
    return np.array([[cx, cy]], dtype=np.float32)


def run_pipeline(image_bgr: np.ndarray, prompt: str,
                 tracker: MaskTracker | None = None,
                 ref_crop: np.ndarray | None = None,
                 ref_features: np.ndarray | None = None) -> dict:
    """
    SAM3 segmentation -> reference-guided disambiguation -> depth ->
    grasp-point pipeline.

    Detection and segmentation come from a single SAM3 forward pass with
    Promptable Concept Segmentation. Multi-instance disambiguation uses
    the same scoring as the original GDino+SAM2 path (position, confidence,
    proximity, ResNet similarity).

    The result dict mirrors servo_pipeline.py exactly: keys "gdino_box",
    "mask_np", "depth_np", "best_centroid", "similarity", "dets_all",
    "sam_score", "sam_center_pt", "detector_used", "detection_was_skipped".
    The "gdino_box" key is preserved by name for compatibility with the
    debug-dump and JSONL tooling; it now holds the chosen SAM3 box.
    """
    res = dict(mask_np=None, depth_np=None,
               best_centroid=None, gdino_box=None,
               similarity=None,
               dets_all=None,
               sam_score=None,
               sam_center_pt=None,
               detector_used=None,
               detection_was_skipped=False)

    h, w = image_bgr.shape[:2]

    # ── Decide whether to run open-vocabulary detection this frame ───
    use_box_prior = False
    if (tracker is not None
            and tracker.last_box is not None
            and REDETECT_INTERVAL > 1
            and tracker.frame_count % REDETECT_INTERVAL != 0):
        use_box_prior = True
        res["detection_was_skipped"] = True

    box_np = tracker.last_box if tracker is not None else None
    prior_box = box_np if use_box_prior else None

    # ── Step 1+2: SAM3 detect+segment in a single forward pass ───────
    prev_c = tracker.prev_centroid if tracker else None
    dets, masks_list = _detect_sam3(image_bgr, prompt, prior_box_xyxy=prior_box)
    res["detector_used"] = "sam3_box_prior" if use_box_prior else "sam3"

    res["dets_all"] = [
        (np.asarray(d[0], dtype=np.float32).tolist(),
         float(d[1]), str(d[2])) for d in dets]

    chosen_idx = None
    chosen = None
    if dets:
        pick = _disambiguate_top_box(dets, image_bgr.shape,
                                     prev_centroid=prev_c,
                                     image_bgr=image_bgr,
                                     ref_features=ref_features,
                                     return_index=True)
        if pick is not None:
            chosen_idx, chosen = pick
            box_np = chosen[0]
            res["gdino_box"] = box_np
            res["sam_score"] = float(chosen[1])

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
                     "SAM3", chosen[2], chosen[1],
                     box_np.astype(int), len(dets), sim_str)

    # ── Pick the SAM3 mask aligned with the chosen box ──────────────
    mask_out = None
    center_pt = None
    if chosen is not None and chosen_idx is not None and chosen_idx < len(masks_list):
        mask_out = masks_list[chosen_idx]
        center_pt = _box_center(box_np, h, w)
        res["sam_center_pt"] = (float(center_pt[0, 0]), float(center_pt[0, 1]))

    if mask_out is not None:
        # IoU drift check against prior mask, mirroring servo_pipeline.py
        if (tracker is not None and tracker.prev_mask is not None
                and use_box_prior):
            iou = _mask_iou(mask_out, tracker.prev_mask)
            if iou < IOU_DRIFT_THRESH:
                log.info("SAM3: IoU=%.2f < %.2f - forcing redetect "
                         "next frame", iou, IOU_DRIFT_THRESH)
                tracker.reset(keep_anchor=True)

        res["mask_np"] = mask_out
        log.info("SAM3: mask obtained (score=%.3f, area=%d px)",
                 float(chosen[1]), int(np.count_nonzero(mask_out)))
    else:
        if box_np is None:
            log.warning("SAM3 returned no usable instance. Try a more "
                        "descriptive --prompt or lower SAM3_SCORE_THRESH.")
        else:
            log.warning("SAM3: no mask aligned with chosen box "
                        "(idx=%s, n_masks=%d)", chosen_idx, len(masks_list))

    if box_np is not None:
        res["gdino_box"] = box_np
        if tracker is not None:
            tracker.last_box = box_np

    # ── Step 3: Depth ───────────────────────────────────────────────
    dm = _get_depth_model()
    if dm is not None:
        try:
            res["depth_np"] = dm.infer_image(image_bgr).astype(np.float32)
        except Exception as e:
            log.error("Depth inference failed: %s", e)

    # ── Step 4: Grasp point ─────────────────────────────────────────
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
        log.debug("Grasp point: bbox centre %s (no SAM3 mask)",
                  res["best_centroid"])

    # ── Step 5: Update tracker ──────────────────────────────────────
    if tracker is not None:
        tracker.update(res["mask_np"], None, res.get("best_centroid"))

    return res


# =====================================================================
#  Optical-flow calibration helper (used by the robot Y/Z calibration)
# =====================================================================

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


# =====================================================================
#  Robot controller (unchanged from servo_pipeline.py)
# =====================================================================

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
                log.warning("  [%s] no frame before move - skipping axis", ax)
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
                log.warning("  [%s] no frame after move - skipping axis", ax)
                continue

            flow = _measure_flow(frame_before, frame_after)
            if flow is None:
                log.warning("  [%s] optical flow failed - skipping axis", ax)
                continue

            dpx, dpy = flow
            J_yz[0, col] = dpx / CAL_DELTA
            J_yz[1, col] = dpy / CAL_DELTA
            log.info("  [%s] flow: (%+.1f, %+.1f) px / %.0f mm  "
                     "-> J[:,{col}]=[%+.4f, %+.4f]",
                     ax, dpx, dpy, CAL_DELTA,
                     J_yz[0, col], J_yz[1, col])

        self._jac_yz  = J_yz
        log.info("J_yz (px/mm):\n%s", J_yz)
        rank = np.linalg.matrix_rank(J_yz)

        if rank == 0:
            log.warning("J_yz rank=0 - approach-only mode.")
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
                log.warning("Servo: centroid jump %.0f px - ignored", jump)
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
                    log.warning("Servo: clamping Z %.1f -> %.1f mm (base limit)",
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


# =====================================================================
#  Camera streamer (PyZED primary, OpenCV fallback)
# =====================================================================

class CameraStreamer(threading.Thread):
    """
    Captures frames from the ZED camera (via PyZED or OpenCV), runs the
    SAM3 segmentation pipeline in a background thread, overlays results,
    and records both a raw SVO feed (pyzed) and an annotated MP4 at
    >= MIN_RECORDING_DIM px.
    """

    def __init__(self, cam_index: int, stop_event: threading.Event,
                 robot: RobotController,
                 ref_image_path: str | None = None,
                 use_pyzed: bool = True,
                 debug_dir: str | None = None,
                 debug_every: int = 30):
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

        self.ref_crop         = None
        self.ref_features     = None

        self._debug_dir       = debug_dir
        self._debug_every     = max(1, int(debug_every or 1))
        self._debug_frame_idx = 0

    def _get_frame(self):
        with self._frame_lock:
            if self._latest_left is not None:
                return self._latest_left.copy()
            return None

    def _run_calibration(self):
        log.info("Calibration thread: waiting for models ...")
        self._models_ready.wait()
        log.info("Calibration thread: starting Y/Z Jacobian calibration.")
        self.robot.calibrate(self._get_frame)

    def _process_ref_image(self):
        if self.ref_image_path is None:
            return
        try:
            ref_bgr = _load_ref_image(self.ref_image_path)
            if ref_bgr is None:
                return
            self.ref_crop, self.ref_features = process_ref_image(ref_bgr, PROMPT)
        except Exception as e:
            log.error("Reference image processing error: %s", e)
            import traceback; traceback.print_exc()

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
                                   ref_crop=self.ref_crop,
                                   ref_features=self.ref_features)
                with self.data_lock:
                    self._result = res

                if (self._debug_dir
                        and self._debug_frame_idx % self._debug_every == 0):
                    overlay = None
                    try:
                        overlay = self._render(frame, res)
                    except Exception:
                        overlay = None
                    _dump_debug_frame(self._debug_dir,
                                      self._debug_frame_idx,
                                      frame, res, overlay_bgr=overlay,
                                      source="live")
                self._debug_frame_idx += 1

                if not self._models_ready.is_set():
                    log.info("Models ready - calibration may now proceed.")
                    self._models_ready.set()
                if self.robot is not None and res.get("best_centroid") is not None:
                    self.robot.servo_step(res["best_centroid"], frame.shape)
            except Exception as e:
                log.error("Pipeline error: %s", e)
                import traceback; traceback.print_exc()
                time.sleep(1)

    def run(self):
        if self._use_pyzed:
            self._run_pyzed()
        else:
            self._run_opencv()

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
            log.warning("PyZED: camera open failed (%s) - falling back to OpenCV", err)
            self._run_opencv()
            return

        ts = time.strftime("%Y%m%d_%H%M%S")

        svo_path   = os.path.abspath(f"vs_recording_{ts}.svo2")
        rec_params = sl.RecordingParameters()
        rec_params.video_filename    = svo_path
        rec_params.compression_mode  = sl.SVO_COMPRESSION_MODE.H264
        err = cam.enable_recording(rec_params)
        if err == sl.ERROR_CODE.SUCCESS:
            log.info("PyZED SVO recording -> %s", svo_path)
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
                        log.error("VideoWriter failed for %s (%dx%d)",
                                  anno_path, rw, rh)
                        video_writer.release()
                        video_writer = None
                    else:
                        log.info("Annotated recording -> %s  (%dx%d)",
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
                        log.error("VideoWriter failed for %s (%dx%d)",
                                  video_path, rw, rh)
                        video_writer.release()
                        video_writer = None
                    else:
                        log.info("Recording -> %s  (%dx%d)", video_path, rw, rh)

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

    @staticmethod
    def _ensure_min_dim(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        short = min(h, w)
        if short < MIN_RECORDING_DIM:
            scale = MIN_RECORDING_DIM / short
            img   = cv2.resize(img, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_LINEAR)
        return img

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

        if self.ref_features is not None:
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


# =====================================================================
#  Offline image/video runner (identical schema to servo_pipeline.py)
# =====================================================================

def _result_summary(res: dict, frame_shape: tuple, tracker: MaskTracker,
                    frame_index: int, source: str, elapsed_s: float) -> dict:
    h, w = frame_shape[:2]
    mask = res.get("mask_np")
    depth = res.get("depth_np")
    centroid = res.get("best_centroid")
    box = res.get("gdino_box")

    if hasattr(box, "astype"):
        bbox = box.astype(float).tolist()
    elif box is not None:
        bbox = [float(v) for v in box]
    else:
        bbox = None

    summary = {
        "frame_index": frame_index,
        "source": source,
        "elapsed_s": elapsed_s,
        "image_width": w,
        "image_height": h,
        "centroid": list(map(int, centroid)) if centroid is not None else None,
        "bbox": bbox,
        "similarity": (float(res["similarity"])
                       if res.get("similarity") is not None else None),
        "mask_area_px": int(np.count_nonzero(mask)) if mask is not None else 0,
        "tracker_frame_count": tracker.frame_count,
        "tracker_warmed_up": tracker.warmed_up,
        "anchor_locked": tracker.anchor_locked,
        "anchor_source": tracker._anchor_source,
        "last_iou": float(tracker._last_iou),
    }

    if depth is not None:
        d = depth.astype(np.float32)
        summary["depth_min"] = float(np.nanmin(d))
        summary["depth_max"] = float(np.nanmax(d))
        summary["depth_mean"] = float(np.nanmean(d))

    if centroid is not None:
        cx, cy = centroid
        summary["pixel_error"] = [int(cx - w // 2), int(cy - h // 2)]

    return summary


# =====================================================================
#  Debug dump (per-frame artifacts for sharing with reviewers)
# =====================================================================

DEBUG_PALETTE = [
    (0, 255, 0), (0, 200, 255), (255, 200, 0), (255, 0, 200),
    (0, 255, 255), (255, 100, 100), (100, 255, 100), (180, 180, 0),
]


def _init_debug_dir(base_dir: str | None,
                    args,
                    ref_image_path: str | None) -> str | None:
    if not base_dir:
        return None

    ts = time.strftime("%Y%m%d_%H%M%S")
    debug_dir = os.path.abspath(os.path.join(base_dir, f"debug_sam3_{ts}"))
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "frames"), exist_ok=True)
    os.makedirs(os.path.join(debug_dir, "ref"), exist_ok=True)

    cfg = {
        "timestamp": ts,
        "cwd": os.getcwd(),
        "detector": DETECTOR,
        "sam3_model_id": SAM3_HF_MODEL_ID,
        "sam3_score_thresh": SAM3_SCORE_THRESH,
        "sam3_mask_thresh": SAM3_MASK_THRESH,
        "prompt": PROMPT,
        "redetect_interval": REDETECT_INTERVAL,
        "iou_drift_thresh": IOU_DRIFT_THRESH,
        "min_z_mm": MIN_Z_MM,
        "args": {k: getattr(args, k, None) for k in vars(args)} if args else {},
    }
    log_path = getattr(log, "_semvs_log_path", None)
    if log_path:
        cfg["log_file"] = log_path

    with open(os.path.join(debug_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, default=str)

    if ref_image_path and os.path.exists(ref_image_path):
        try:
            import shutil
            shutil.copy(ref_image_path,
                        os.path.join(debug_dir, "ref",
                                     os.path.basename(ref_image_path)))
        except Exception as e:
            log.warning("debug-dump: could not copy ref image: %s", e)

    log.info("Debug dump: %s", debug_dir)
    log.info("Debug dump: share this absolute path to send results back.")
    return debug_dir


def _finalize_debug_dir(debug_dir: str | None) -> None:
    if not debug_dir:
        return
    log_path = getattr(log, "_semvs_log_path", None)
    if log_path and os.path.exists(log_path):
        try:
            import shutil
            shutil.copy(log_path, os.path.join(debug_dir, "log.txt"))
        except Exception as e:
            log.warning("debug-dump: could not copy log file: %s", e)
    log.info("Debug dump finalized: %s", debug_dir)
    log.info("Push to share with Claude:  git add %s && git commit -m "
             "'debug dump' && git push", debug_dir)


def _write_debug_index(debug_dir: str | None,
                       summaries: list,
                       args,
                       source: str) -> None:
    if not debug_dir:
        return
    try:
        n = len(summaries)
        n_with_box = sum(1 for s in summaries if s.get("bbox") is not None)
        n_with_centroid = sum(1 for s in summaries
                              if s.get("centroid") is not None)
        first_with_box = next((s for s in summaries
                               if s.get("bbox") is not None), None)
        first_no_box = next((s for s in summaries
                             if s.get("bbox") is None), None)

        lines = [
            f"# SemVS debug dump (SAM3 pipeline)",
            f"",
            f"- Source: `{source}`",
            f"- Frames: **{n}**, with detector box: **{n_with_box}**, "
            f"with centroid: **{n_with_centroid}**",
            f"- Detector: `{DETECTOR}` ({SAM3_HF_MODEL_ID})",
            f"- Prompt: `{PROMPT}`",
            f"- CLI args: `{vars(args) if args else {}}`",
            f"",
            f"## What's here",
            f"- `config.json` - run configuration",
            f"- `log.txt` - full structured log (DEBUG and above)",
            f"- `ref/` - copy of the --ref-image used",
            f"- `frames/frame_NNNNNN/`",
            f"  - `input.png` - input frame",
            f"  - `detections.png` - input + every SAM3 instance candidate "
            f"(coloured, score-labelled), the chosen box in red, the SAM3 "
            f"box centre as a yellow dot",
            f"  - `detections.json` - SAM3 candidates, chosen box, "
            f"box centre, SAM3 score, similarity, centroid",
            f"  - `sam_mask.png` - final 0/255 mask from SAM3",
            f"  - `overlay.png` - high-level rendered overlay",
            f"  - `summary.json` - per-frame summary",
            f"",
            f"## Frames worth looking at first",
        ]
        if first_with_box is not None:
            i = first_with_box.get("frame_index")
            lines.append(f"- First frame with a detection: "
                         f"`frames/frame_{i:06d}/`")
        if first_no_box is not None:
            i = first_no_box.get("frame_index")
            lines.append(f"- First frame with NO detection: "
                         f"`frames/frame_{i:06d}/`")
        if not (first_with_box or first_no_box):
            lines.append("- (no frames recorded)")

        with open(os.path.join(debug_dir, "INDEX.md"), "w") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        log.warning("debug-dump: INDEX.md failed: %s", e)


def _draw_box(img: np.ndarray, box, color, thickness: int = 2,
              label: str | None = None):
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(img, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def _dump_debug_frame(debug_dir: str | None,
                      frame_index: int,
                      frame_bgr: np.ndarray,
                      res: dict,
                      overlay_bgr: np.ndarray | None = None,
                      source: str | None = None) -> None:
    if not debug_dir:
        return
    try:
        fdir = os.path.join(debug_dir, "frames", f"frame_{frame_index:06d}")
        os.makedirs(fdir, exist_ok=True)

        cv2.imwrite(os.path.join(fdir, "input.png"), frame_bgr)

        det_overlay = frame_bgr.copy()
        dets_all = res.get("dets_all") or []
        for i, (box, score, label) in enumerate(dets_all):
            color = DEBUG_PALETTE[i % len(DEBUG_PALETTE)]
            _draw_box(det_overlay, box, color, 1,
                      f"{label} {score:.2f}")
        chosen = res.get("gdino_box")
        if chosen is not None:
            _draw_box(det_overlay, chosen.astype(float), (0, 0, 255), 3,
                      "CHOSEN")
        center_pt = res.get("sam_center_pt")
        if center_pt is not None:
            cx, cy = int(round(center_pt[0])), int(round(center_pt[1]))
            cv2.circle(det_overlay, (cx, cy), 6, (0, 255, 255), -1)
        cv2.imwrite(os.path.join(fdir, "detections.png"), det_overlay)

        det_meta = {
            "frame_index": frame_index,
            "source": source,
            "detector_used": res.get("detector_used"),
            "detection_was_skipped": res.get("detection_was_skipped"),
            "candidates": [
                {"box_xyxy": list(map(float, box)),
                 "score": float(score),
                 "label": label}
                for (box, score, label) in dets_all
            ],
            "chosen_box_xyxy": (chosen.astype(float).tolist()
                                if chosen is not None else None),
            "sam_center_pt": (list(center_pt)
                              if center_pt is not None else None),
            "sam_score": res.get("sam_score"),
            "similarity": (float(res["similarity"])
                           if res.get("similarity") is not None else None),
            "best_centroid": (list(map(int, res["best_centroid"]))
                              if res.get("best_centroid") is not None
                              else None),
        }
        with open(os.path.join(fdir, "detections.json"), "w") as f:
            json.dump(det_meta, f, indent=2)

        mask = res.get("mask_np")
        if mask is not None:
            cv2.imwrite(os.path.join(fdir, "sam_mask.png"), mask)
            mask_area = int(np.count_nonzero(mask))
        else:
            mask_area = 0

        if overlay_bgr is not None:
            cv2.imwrite(os.path.join(fdir, "overlay.png"), overlay_bgr)

        summary = dict(det_meta)
        summary["mask_area_px"] = mask_area
        summary["frame_shape"] = list(frame_bgr.shape)
        with open(os.path.join(fdir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    except Exception as e:
        log.warning("debug-dump: frame %d failed: %s", frame_index, e)


def _make_offline_streamer(ref_image_path: str | None) -> CameraStreamer:
    stop_event = threading.Event()
    streamer = CameraStreamer(
        cam_index=0,
        stop_event=stop_event,
        robot=None,
        ref_image_path=ref_image_path,
        use_pyzed=False,
    )
    streamer._process_ref_image()
    return streamer


def _run_offline_frame(streamer: CameraStreamer, frame: np.ndarray,
                       frame_index: int, source: str, output_dir: str,
                       jsonl_fh, save_overlays: bool = True,
                       debug_dir: str | None = None,
                       debug_every: int = 1) -> dict:
    start = time.time()
    res = run_pipeline(
        frame,
        PROMPT,
        streamer._tracker,
        ref_crop=streamer.ref_crop,
        ref_features=streamer.ref_features,
    )
    elapsed_s = time.time() - start
    summary = _result_summary(
        res, frame.shape, streamer._tracker, frame_index, source, elapsed_s)

    overlay = None
    if save_overlays:
        overlay = streamer._render(frame, res)
        overlay_path = os.path.join(output_dir, f"overlay_{frame_index:06d}.png")
        cv2.imwrite(overlay_path, overlay)
        summary["overlay_path"] = overlay_path

    if debug_dir and (debug_every <= 1 or frame_index % debug_every == 0):
        _dump_debug_frame(debug_dir, frame_index, frame, res,
                          overlay_bgr=overlay, source=source)

    jsonl_fh.write(json.dumps(summary) + "\n")
    jsonl_fh.flush()

    return summary


def _iter_image_paths(input_image: str | None, input_dir: str | None):
    if input_image:
        yield input_image
        return

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for name in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, name)
        if os.path.splitext(name.lower())[1] in valid_exts:
            yield path


def run_offline(args) -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or os.path.join("runs", f"offline_sam3_{ts}")
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "frames.jsonl")

    debug_dir = _init_debug_dir(args.debug_dir, args, args.ref_image)

    streamer = _make_offline_streamer(args.ref_image)
    summaries = []
    debug_every = max(1, int(getattr(args, "debug_every", 1) or 1))

    try:
        with open(jsonl_path, "w") as jsonl_fh:
            if args.input_video:
                cap = cv2.VideoCapture(args.input_video)
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open video: {args.input_video}")
                frame_index = 0
                while True:
                    if args.max_frames is not None and frame_index >= args.max_frames:
                        break
                    ok, frame = cap.read()
                    if not ok:
                        break
                    summaries.append(_run_offline_frame(
                        streamer, frame, frame_index, args.input_video, output_dir,
                        jsonl_fh, save_overlays=not args.no_overlays,
                        debug_dir=debug_dir, debug_every=debug_every))
                    frame_index += 1
                cap.release()
            else:
                for frame_index, path in enumerate(
                        _iter_image_paths(args.input_image, args.input_dir)):
                    if args.max_frames is not None and frame_index >= args.max_frames:
                        break
                    frame = cv2.imread(path)
                    if frame is None:
                        log.warning("Skipping unreadable image: %s", path)
                        continue
                    summaries.append(_run_offline_frame(
                        streamer, frame, frame_index, path, output_dir,
                        jsonl_fh, save_overlays=not args.no_overlays,
                        debug_dir=debug_dir, debug_every=debug_every))

        summary_path = os.path.join(output_dir, "summary.json")
        success_count = sum(1 for s in summaries if s.get("centroid") is not None)
        with open(summary_path, "w") as f:
            json.dump({
                "frames": len(summaries),
                "frames_with_centroid": success_count,
                "output_dir": output_dir,
                "jsonl_path": jsonl_path,
                "debug_dir": debug_dir,
            }, f, indent=2)

        log.info("Offline run complete: %d frame(s), %d with centroid",
                 len(summaries), success_count)
        log.info("Offline outputs: %s", os.path.abspath(output_dir))

        if debug_dir:
            _write_debug_index(debug_dir, summaries, args, source="offline")
    finally:
        _finalize_debug_dir(debug_dir)


# =====================================================================
#  Entry point
# =====================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SemVS: Semantic Visual Servoing with SAM 3 "
                    "(suction-cup gripper)")
    parser.add_argument("cam_index", nargs="?", type=int, default=0,
                        help="OpenCV camera index (default: 0)")
    parser.add_argument("--ref-image", "--ref_image", metavar="PATH",
                        help="Path to a reference image of the target object. "
                             "Used to bias SAM3 candidate choice via ResNet "
                             "feature similarity. Alpha is ignored.")
    parser.add_argument("--no-pyzed", action="store_true",
                        help="Force OpenCV camera capture even if PyZED is "
                             "available")
    parser.add_argument("--no-robot", action="store_true",
                        help="Run vision/camera only; do not connect to xArm")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run perception and logging without robot commands")
    parser.add_argument("--input-image", metavar="PATH",
                        help="Run offline on a single image instead of a camera")
    parser.add_argument("--input-dir", metavar="PATH",
                        help="Run offline on all images in a directory")
    parser.add_argument("--input-video", metavar="PATH",
                        help="Run offline on a video file")
    parser.add_argument("--output-dir", metavar="PATH",
                        help="Directory for offline overlays and JSON metrics")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum offline frames to process")
    parser.add_argument("--no-overlays", action="store_true",
                        help="Skip overlay PNGs during offline runs")
    parser.add_argument("--prompt", default=None,
                        help=f"Concept prompt for SAM3 (default: '{PROMPT}')")
    parser.add_argument("--sam3-model", default=None,
                        help=f"HuggingFace model id for SAM3 "
                             f"(default: '{SAM3_HF_MODEL_ID}'). "
                             f"Try 'facebook/sam3' or 'facebook/sam3.1'.")
    parser.add_argument("--score-thresh", type=float, default=None,
                        help=f"SAM3 instance score threshold "
                             f"(default: {SAM3_SCORE_THRESH})")
    parser.add_argument("--mask-thresh", type=float, default=None,
                        help=f"SAM3 mask threshold "
                             f"(default: {SAM3_MASK_THRESH})")
    parser.add_argument("--log-dir", default="logs",
                        help="Directory for log files (default: logs)")
    parser.add_argument("--debug", action="store_true",
                        help="Per-frame debug dump under artifacts/. "
                             "Saves input frame, all SAM3 candidates, "
                             "chosen box, mask, overlay, and the full "
                             "log file. Offline: every frame. Live: every "
                             "30th frame. Push artifacts/debug_sam3_<ts>/ "
                             "to share results.")
    args = parser.parse_args()

    args.debug_dir = "artifacts" if args.debug else None
    args.debug_every = 1 if (args.input_image or args.input_dir
                              or args.input_video) else 30

    log = _setup_logger(log_dir=args.log_dir)

    if args.prompt:
        PROMPT = args.prompt
    if args.sam3_model:
        SAM3_HF_MODEL_ID = args.sam3_model
    if args.score_thresh is not None:
        SAM3_SCORE_THRESH = float(args.score_thresh)
    if args.mask_thresh is not None:
        SAM3_MASK_THRESH = float(args.mask_thresh)

    log.info("Detector : %s", DETECTOR.upper())
    log.info("SAM3     : %s", SAM3_HF_MODEL_ID)
    log.info("Score th : %.2f  Mask th: %.2f",
             SAM3_SCORE_THRESH, SAM3_MASK_THRESH)
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
        log.info("No reference image provided - using text-prompt detection only")

    offline_inputs = [
        bool(args.input_image),
        bool(args.input_dir),
        bool(args.input_video),
    ]
    if sum(offline_inputs) > 1:
        log.error("Use only one of --input-image, --input-dir, or --input-video")
        sys.exit(1)
    if any(offline_inputs):
        if args.input_image and not os.path.exists(args.input_image):
            log.error("Input image not found: %s", args.input_image)
            sys.exit(1)
        if args.input_dir and not os.path.isdir(args.input_dir):
            log.error("Input directory not found: %s", args.input_dir)
            sys.exit(1)
        if args.input_video and not os.path.exists(args.input_video):
            log.error("Input video not found: %s", args.input_video)
            sys.exit(1)
        run_offline(args)
        sys.exit(0)

    robot = RobotController(ROBOT_IP)
    if args.no_robot or args.dry_run:
        reason = "--dry-run" if args.dry_run else "--no-robot"
        log.info("Robot disabled (%s)", reason)
    else:
        robot.connect()

    debug_dir = _init_debug_dir(args.debug_dir, args, args.ref_image)

    stop_ev    = threading.Event()
    cam_thread = CameraStreamer(
        cam_index      = args.cam_index,
        stop_event     = stop_ev,
        robot          = robot,
        ref_image_path = args.ref_image,
        use_pyzed      = not args.no_pyzed,
        debug_dir      = debug_dir,
        debug_every    = max(1, int(getattr(args, "debug_every", 30) or 30)),
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
        _finalize_debug_dir(debug_dir)
        log.info("Done.")
