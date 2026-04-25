#!/usr/bin/env python3
"""
SemVS – Semantic Visual Servoing
Updated: image-conditioned detection, stable-anchor mask propagation,
         PyZED camera + SVO recording, ≥512 px annotated video.
"""
import logging
import time
import sys
import threading
import os
import tempfile

import cv2
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"))
    logger.addHandler(_handler)
    logger.setLevel(logging.DEBUG)

THIRD_PARTY_ROOT = os.path.join(os.path.dirname(__file__), "third-party")

PROMPT = "box"

# ── Detector selection ────────────────────────────────────────────────
DETECTOR = os.environ.get("DETECTOR", "gdino").lower()   # override via env var

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
MIN_Z_MM     = 0.0      # minimum Z height (mm) — base level; prevents
                         # the arm from commanding poses below its base

# ── Mask propagation / tracking parameters ───────────────────────────
TRACKER_WARMUP_FRAMES = 10       # frames before negative prompting kicks in
TRACKER_MAX_HISTORY   = 15       # mask history buffer size
NEG_POINT_COUNT       = 5        # number of negative sample points
NEG_POINT_MARGIN_PX   = 30       # erosion margin for safe negative zone
REDETECT_INTERVAL     = 30       # re-run detector every N frames
IOU_DRIFT_THRESH      = 0.35     # reset tracker when mask IoU drops below this

# ── Anchor / stable-mask parameters ──────────────────────────────────
ANCHOR_IOU_THRESH  = 0.82   # inter-frame IoU required to count as "stable"
ANCHOR_LOCK_FRAMES = 5      # consecutive stable frames before locking anchor

# ── Video recording ───────────────────────────────────────────────────
MIN_RECORDING_DIM = 512     # minimum pixel dimension of the recorded video
ZED_RESOLUTION    = "HD720" # PyZED resolution (HD720 = 1280×720)

# ── ZED calibration for lens undistortion (OpenCV path) ───────────────
ZED_SETTINGS_DIRS = [
    "/usr/local/zed/settings/",
    os.path.expanduser("~/.ZED/settings/"),
]

def _load_zed_calibration(resolution: str = "HD"):
    """
    Load ZED camera intrinsics + distortion coefficients from the factory
    calibration file on disk.  Picks the most recently modified SN*.conf
    file (likely the connected camera).

    Parameters
    ----------
    resolution : "2K", "FHD", "HD", or "VGA"

    Returns
    -------
    camera_matrix : np.ndarray (3,3) or None
    dist_coeffs   : np.ndarray (4,) [k1, k2, p1, p2] or None
    image_size    : (width, height) or None
    """
    import glob, configparser

    conf_files = []
    for d in ZED_SETTINGS_DIRS:
        conf_files.extend(glob.glob(os.path.join(d, "SN*.conf")))
    if not conf_files:
        return None, None, None

    # Pick the most recently modified file (likely the active camera)
    conf_files.sort(key=os.path.getmtime, reverse=True)
    conf_path = conf_files[0]

    cp = configparser.ConfigParser()
    cp.read(conf_path)

    section = f"LEFT_CAM_{resolution}"
    if section not in cp:
        logger.info(f"ZED cal: section [{section}] not found in {conf_path}")
        return None, None, None

    fx = float(cp[section]["fx"])
    fy = float(cp[section]["fy"])
    cx = float(cp[section]["cx"])
    cy = float(cp[section]["cy"])
    k1 = float(cp[section]["k1"])
    k2 = float(cp[section]["k2"])
    # ZED conf stores: k1, k2, k3(=p1), k4(=p2)
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
    logger.info(f"ZED cal: loaded {sn} [{section}]  fx={fx:.1f} fy={fy:.1f} "
          f"cx={cx:.1f} cy={cy:.1f}  dist=[{k1:.4f},{k2:.4f},{p1:.4f},{p2:.4f}]")
    return K, D, img_size


class ZedUndistorter:
    """
    Pre-computes undistortion maps from ZED factory calibration and
    applies them to raw UVC frames.  No-op if calibration is unavailable.
    """

    def __init__(self, resolution: str = "HD"):
        self._map1 = None
        self._map2 = None
        self._new_K = None
        K, D, img_size = _load_zed_calibration(resolution)
        if K is None:
            logger.info("ZED undistortion: calibration not found — frames will NOT "
                  "be undistorted")
            return

        w, h = img_size
        # Compute optimal new camera matrix (alpha=0 → crop black edges,
        # alpha=1 → keep all pixels).  alpha=0 is cleaner for servo.
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0)
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            K, D, None, new_K, (w, h), cv2.CV_16SC2)
        self._new_K = new_K
        logger.info(f"ZED undistortion: maps ready ({w}×{h})")

    @property
    def available(self) -> bool:
        return self._map1 is not None

    @classmethod
    def from_frame_size(cls, width: int, height: int) -> "ZedUndistorter":
        """Create an undistorter by matching actual frame dimensions to the
        calibration section, removing the need for a hardcoded resolution map."""
        _size_to_res = {
            (2208, 1242): "2K",
            (1920, 1080): "FHD",
            (1280, 720):  "HD",
            (672,  376):  "VGA",
        }
        res = _size_to_res.get((width, height))
        if res is None:
            logger.info(f"ZedUndistorter: no calibration section for {width}×{height} "
                  f"— trying closest match")
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
            # Resolution mismatch — skip rather than crash
            return frame
        return cv2.remap(frame, self._map1, self._map2,
                         interpolation=cv2.INTER_LINEAR)


# ── PyZED ─────────────────────────────────────────────────────────────
try:
    import pyzed.sl as sl
    PYZED_AVAILABLE = True
    logger.info("PyZED available — will use ZED SDK for capture and SVO recording")
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

    Key additions over the basic tracker:
    * ``anchor_logits`` / ``anchor_mask`` — a locked "best known" mask that
      is reused as SAM2 mask_input whenever the live track is unavailable or
      has drifted.  The anchor is set once ANCHOR_LOCK_FRAMES consecutive
      frames each show inter-frame IoU ≥ ANCHOR_IOU_THRESH.
    * ``reset(keep_anchor)`` — soft reset that retains the anchor so the
      tracker re-locks faster after a disturbance.
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
        self._iou_streak:    int               = 0   # consecutive stable frames
        self._last_iou:      float             = 1.0

    # ── housekeeping ─────────────────────────────────────────────────
    def reset(self, keep_anchor: bool = True):
        """Reset live tracking state.  Optionally keep the anchor for warm re-start."""
        self.prev_logits    = None
        self.prev_mask      = None
        self.prev_centroid  = None
        self.frame_count    = 0
        self.mask_history.clear()
        self._iou_streak    = 0
        self._last_iou      = 1.0
        if not keep_anchor:
            self.anchor_logits = None
            self.anchor_mask   = None

    def update(self, mask_np: np.ndarray | None,
               logits: np.ndarray | None,
               centroid: tuple | None):
        # Track inter-frame IoU for anchor locking
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
        """Promote current mask to anchor when the track has been stable long enough."""
        if logits is None:
            return
        if self._iou_streak >= ANCHOR_LOCK_FRAMES:
            self.anchor_logits = logits.copy()
            self.anchor_mask   = mask_np.copy()
            logger.info(f"MaskTracker: anchor locked (IoU streak={self._iou_streak}, "
                  f"last IoU={self._last_iou:.3f})")

    @property
    def warmed_up(self) -> bool:
        return (self.frame_count >= TRACKER_WARMUP_FRAMES
                and len(self.mask_history) >= TRACKER_WARMUP_FRAMES)

    @property
    def anchor_locked(self) -> bool:
        return self.anchor_logits is not None

    # ── negative-point mining ────────────────────────────────────────
    def sample_negative_points(self, h: int, w: int,
                               n: int = NEG_POINT_COUNT) -> np.ndarray | None:
        """
        Return n (x,y) points that were consistently outside every recent mask.
        Returns Nx2 np.float32 array or None.
        """
        if not self.warmed_up:
            return None

        always_bg = np.ones((h, w), dtype=np.uint8) * 255
        for m in self.mask_history[-TRACKER_WARMUP_FRAMES:]:
            mr = m if m.shape[:2] == (h, w) else cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            always_bg = cv2.bitwise_and(always_bg, cv2.bitwise_not(mr))

        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (NEG_POINT_MARGIN_PX * 2 + 1,
                                          NEG_POINT_MARGIN_PX * 2 + 1))
        safe = cv2.erode(always_bg, kern)

        coords = np.argwhere(safe > 0)
        if len(coords) < n:
            return None

        idx = np.random.choice(len(coords), n, replace=False)
        pts = coords[idx][:, ::-1].astype(np.float32)   # (x, y)
        return pts

    # ── positive-point mining from previous mask ─────────────────────
    def sample_positive_points(self, h: int, w: int,
                               n: int = 1) -> np.ndarray | None:
        """
        Sample n positive points from the interior of the previous mask.
        Prefer the centroid; add extra random interior points if n > 1.
        Returns Nx2 np.float32 array or None.
        """
        if self.prev_mask is None:
            return None

        pts = []
        if self.prev_centroid is not None:
            pts.append(list(self.prev_centroid))

        if n > len(pts) and self.prev_mask is not None:
            mr = self.prev_mask if self.prev_mask.shape[:2] == (h, w) else cv2.resize(self.prev_mask, (w, h),
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

        logger.info("BERT compat patch applied")
    except Exception as e:
        logger.error("BERT compat patch failed: %s", e)
        logger.exception("Details:")


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
            logger.error("GroundingDINO load failed (won't retry): %s", e)
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
            logger.info("OWLv2 loaded")
        except Exception as e:
            logger.error("OWLv2 load failed (won't retry): %s", e)
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
                logger.info(f"Depth-Anything-V2 checkpoint not found: {ckpt}")
                return None
            import torch as _t
            m = DepthAnythingV2(**cfgs[DA_ENCODER])
            m.load_state_dict(_t.load(ckpt, map_location="cpu"))
            _depth_model = m.to(DEVICE).eval()
        except Exception as e:
            logger.error("Depth-Anything-V2 load failed: %s", e)
    return _depth_model


# ═════════════════════════════════════════════════════════════════════
#  Feature extractor for visual similarity (colour-agnostic detection)
# ═════════════════════════════════════════════════════════════════════

_feat_model     = None
_feat_transform = None


def _get_feature_extractor():
    """
    Singleton: load a headless ResNet-18 for feature extraction.
    Returns (model, transform).  torchvision is already a dependency.
    """
    global _feat_model, _feat_transform
    if _feat_model is not None:
        return _feat_model, _feat_transform
    try:
        import torch as _t
        from torchvision import models, transforms

        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        _feat_model = _t.nn.Sequential(
            *list(backbone.children())[:-1]      # strip final FC → (B, 512, 1, 1)
        ).to(DEVICE).eval()

        _feat_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        logger.info("Feature extractor (ResNet-18) loaded")
    except Exception as e:
        logger.error("Feature extractor load failed: %s", e)
    return _feat_model, _feat_transform


def _extract_features(crop_bgr: np.ndarray) -> np.ndarray | None:
    """Return an L2-normalised 512-d feature vector for a BGR image crop."""
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
        logger.info(f"Feature extraction failed: {e}")
        return None


def _cosine_similarity(f1: np.ndarray, f2: np.ndarray) -> float:
    """Cosine similarity between two normalised feature vectors."""
    return float(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2) + 1e-8))


# ═════════════════════════════════════════════════════════════════════
#  Reference-image conditioning
# ═════════════════════════════════════════════════════════════════════

def _detect_foreground_for_ref(ref_bgr: np.ndarray):
    """
    Quick Otsu-based foreground detection for the reference image.
    Returns a binary fg_mask (uint8 0/255) and bbox (x, y, w, h).
    """
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
        return fm, (int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6))
    lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fm  = (labels == lbl).astype(np.uint8) * 255
    return fm, cv2.boundingRect(fm)


def extract_ref_logits(ref_bgr: np.ndarray,
                       box_np: np.ndarray | None = None):
    """
    Run SAM2 on the reference image using multi-point prompts that span
    the entire object.  This solves the "white line" problem where SAM2
    only captures one colour region of a multi-colour object.

    The approach:
      1. If no detector box is given, detect the foreground via Otsu.
      2. Place a 3×5 grid of positive points across the bounding box,
         keeping only those inside the foreground mask.
      3. Feed the bounding box + multi-point prompts to SAM2.
      4. Morphologically close the result to bridge any remaining gap.

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

        # --- Foreground mask for point sampling ---
        fg_mask, fg_bbox = _detect_foreground_for_ref(ref_bgr)

        # --- Bounding box ---
        if box_np is not None:
            bx1, by1, bx2, by2 = box_np.astype(int)
        else:
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

        logger.info(f"Ref SAM2: box=[{bx1},{by1},{bx2},{by2}], "
              f"{len(points)} positive points")

        sam_kwargs = dict(
            box=sam_box,
            point_coords=np.array(points, dtype=np.float32),
            point_labels=np.ones(len(points), dtype=np.int32),
            multimask_output=True,
            return_logits=True,
        )

        masks, scores, logits = pred.predict(**sam_kwargs)

        if masks is None or len(masks) == 0:
            logger.info("Ref SAM2: no mask returned")
            return None, None

        best_idx   = int(np.argmax(scores))
        ref_mask   = (masks[best_idx] > 0).astype(np.uint8) * 255
        ref_logits = logits[best_idx: best_idx + 1]

        # --- Refinement: if coverage of foreground is low, re-run ---
        covered  = np.count_nonzero((fg_mask > 0) & (ref_mask > 0))
        total_fg = max(1, np.count_nonzero(fg_mask > 0))
        coverage = covered / total_fg

        if coverage < 0.70:
            logger.info(f"Ref SAM2: coverage={coverage:.0%}, adding uncovered points")
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
                    logger.info(f"Ref SAM2 (refined): score={scores2[best2]:.3f}")

        # --- Morph close to bridge white-line gap ---
        k_close  = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        ref_mask = cv2.morphologyEx(ref_mask, cv2.MORPH_CLOSE,
                                    k_close, iterations=2)
        contours, _ = cv2.findContours(ref_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ref_mask, contours, -1, 255, cv2.FILLED)

        logger.info(f"Ref image: SAM2 mask extracted (score={scores[best_idx]:.3f}, "
              f"area={np.count_nonzero(ref_mask)} px)")
        return ref_logits, ref_mask

    except Exception as e:
        logger.error("Reference logit extraction failed: %s", e)
        logger.exception("Details:")
    return None, None


def process_ref_image(ref_bgr: np.ndarray, prompt: str):
    """
    Full reference-image processing pipeline.

    1. Run the active detector on the reference image to locate the object.
    2. Crop the object region  → ``ref_crop``  (used for OWLv2 image-guided mode).
    3. Run SAM2 on the reference image → ``ref_logits`` (used as SAM2 prior).
    4. Extract deep features from the crop → ``ref_features`` (for similarity-based
       disambiguation among multiple detections in the live scene).

    Returns
    -------
    ref_crop     : np.ndarray | None   BGR crop of the object
    ref_logits   : np.ndarray | None   SAM2 low-res logits (1, 256, 256)
    ref_mask     : np.ndarray | None   uint8 binary mask from ref image
    ref_features : np.ndarray | None   512-d normalised feature vector
    """
    logger.info("Processing reference image …")
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
            logger.info(f"Ref image: detected '{pick[2]}' score={pick[1]:.2f} "
                  f"@ {ref_box.astype(int)}")

    # Step 2: build object crop for OWLv2 image-guided mode
    h, w = ref_bgr.shape[:2]
    if ref_box is not None:
        x1, y1, x2, y2 = ref_box.astype(int)
        pad = 10
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        ref_crop = ref_bgr[y1:y2, x1:x2].copy()
        logger.info(f"Ref crop: {ref_crop.shape}")
    else:
        ref_crop = ref_bgr[h // 3: 2 * h // 3, w // 3: 2 * w // 3].copy()
        logger.info("Ref crop: centre region (no detection in ref image)")

    # Step 3: SAM2 logits from reference image
    ref_logits, ref_mask = extract_ref_logits(ref_bgr, box_np=ref_box)

    # Step 4: deep features for similarity-based disambiguation
    ref_features = _extract_features(ref_crop)
    if ref_features is not None:
        logger.info(f"Ref features: 512-d vector extracted (norm={np.linalg.norm(ref_features):.3f})")
    else:
        logger.info("Ref features: extraction failed — similarity scoring disabled")

    return ref_crop, ref_logits, ref_mask, ref_features


# ═════════════════════════════════════════════════════════════════════
#  Detection
# ═════════════════════════════════════════════════════════════════════

def _detect_gdino(image_bgr: np.ndarray, prompt: str):
    """Return list of (xyxy_np, score, phrase) or empty list."""
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
                logger.info("GDINO: no boxes")
                return []
            h, w  = image_bgr.shape[:2]
            bxyxy = (box_convert(boxes, "cxcywh", "xyxy")
                     * _t.tensor([w, h, w, h], dtype=_t.float32)).numpy()
            dets  = [(bxyxy[i], float(logits[i]), phrases[i])
                     for i in range(len(boxes))]
            logger.info(f"GDINO: {len(dets)} detection(s)")
            return dets
        finally:
            os.unlink(tmp.name)
    except Exception as e:
        logger.exception("GroundingDINO failed: %s", e)
        return []


def _detect_owlv2(image_bgr: np.ndarray, prompt: str):
    """Text-conditioned OWLv2 detection. Returns list of (xyxy, score, phrase)."""
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
            logger.info("OWLv2: no boxes")
            return []

        dets = [(boxes[i], float(scores[i]), prompt) for i in range(len(boxes))]
        logger.info(f"OWLv2: {len(dets)} detection(s)")
        return dets

    except Exception as e:
        logger.exception("OWLv2 failed: %s", e)
        return []


def _detect_owlv2_image_guided(image_bgr: np.ndarray,
                               ref_crop_bgr: np.ndarray):
    """
    One-shot OWLv2 detection: uses ``ref_crop_bgr`` as the query image
    instead of a text description.  Falls back gracefully on any error.
    Returns list of (xyxy_np, score, "ref_image") or empty list.
    """
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
            logger.info("OWLv2 image-guided: no boxes")
            return []

        dets = [(boxes[i], float(scores[i]), "ref_image")
                for i in range(len(boxes))]
        logger.info(f"OWLv2 image-guided: {len(dets)} detection(s)")
        return dets

    except Exception as e:
        logger.exception("OWLv2 image-guided failed: %s", e)
        return []


# ═════════════════════════════════════════════════════════════════════
#  Box disambiguation for stacked-box scenarios
# ═════════════════════════════════════════════════════════════════════

def _disambiguate_top_box(dets: list, image_shape: tuple,
                          prev_centroid: tuple | None = None,
                          image_bgr: np.ndarray | None = None,
                          ref_features: np.ndarray | None = None):
    """
    From a list of (xyxy, score, phrase) detections, pick the best box.

    When ``ref_features`` (from the reference image) are provided, the
    dominant signal is **visual similarity** — the detection whose crop
    most resembles the reference is preferred.  This lets the system
    distinguish among identically-shaped boxes that differ only in colour
    pattern, without relying on a colour-specific text prompt.

    When no reference features are available, falls back to the original
    heuristic (top-of-frame preference + confidence + proximity).

    Pipeline:
      1. NMS to collapse near-duplicate detections.
      2. Filter out implausibly small / large boxes.
      3. Score by [similarity + position + confidence + proximity].
    """
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
            logger.info(f"  det conf={conf:.2f} sim={sim_score:.3f} "
                  f"pos={pos_score:.2f} prox={prox_score:.2f} → {combined:.3f}")
        else:
            combined = pos_score * 0.50 + conf * 0.25 + prox_score * 0.25

        if combined > best_score:
            best_score = combined
            best = det

    return best


# ═════════════════════════════════════════════════════════════════════
#  Core pipeline  (image-conditioned, with mask propagation + anchoring)
# ═════════════════════════════════════════════════════════════════════

def run_pipeline(image_bgr: np.ndarray, prompt: str,
                 tracker: MaskTracker | None = None,
                 ref_logits: np.ndarray | None = None,
                 ref_crop: np.ndarray | None = None,
                 ref_features: np.ndarray | None = None) -> dict:
    """
    Detection → SAM2 segmentation → depth → grasp-point pipeline.

    Parameters
    ----------
    image_bgr    : current live frame (BGR)
    prompt       : text prompt for the detector (fallback when no ref_crop)
    tracker      : MaskTracker instance for temporal state
    ref_logits   : SAM2 low-res logits from the reference image — used as
                   the initial mask_input on the very first frame and as a
                   fallback when the anchor has not yet been established.
    ref_crop     : BGR reference image crop — used to drive OWLv2 in
                   image-guided (one-shot) mode when DETECTOR == "owlv2".
    ref_features : 512-d normalised feature vector from the reference crop —
                   used to disambiguate among multiple detections by visual
                   similarity instead of relying on a colour-specific prompt.

    Mask-input priority
    -------------------
    1. tracker.prev_logits   – best recent temporal propagation
    2. tracker.anchor_logits – last confirmed stable mask (soft re-init)
    3. ref_logits            – reference image prior (very first frame)
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

        # Image-guided OWLv2 takes priority when a reference crop is available
        if ref_crop is not None and DETECTOR == "owlv2":
            dets = _detect_owlv2_image_guided(image_bgr, ref_crop)
            if not dets:  # fallback to text-based
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
            logger.info(f"{DETECTOR.upper()}: '{pick[2]}' score={pick[1]:.2f} "
                  f"@ {box_np.astype(int)}  (top of {len(dets)}){sim_str}")

    # ── Step 2: SAM2 segmentation with mask propagation ───────────────
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
                logger.info(f"SAM2: +{len(pos_pts) if pos_pts is not None else 0} pos, "
                      f"+{len(neg_pts)} neg point prompts")

        sam_kwargs = dict(multimask_output=True, return_logits=True)

        if box_np is not None:
            sam_kwargs["box"] = box_np

        if point_coords:
            sam_kwargs["point_coords"] = np.array(point_coords, dtype=np.float32)
            sam_kwargs["point_labels"] = np.array(point_labels, dtype=np.int32)

        # --- Mask-input priority chain ---
        # When a fresh detector box is available, let it drive SAM2
        # WITHOUT mask_input bias.  This prevents partial-mask propagation
        # for multi-colour objects where the text prompt matches only one
        # colour region (e.g. "yellow box" on a yellow-and-white box).
        if need_detection and box_np is not None:
            sam_kwargs["multimask_output"] = True
            logger.info("SAM2: fresh detection — using box only (no mask prior)")
        elif tracker is not None and tracker.prev_logits is not None:
            sam_kwargs["mask_input"]      = tracker.prev_logits
            sam_kwargs["multimask_output"] = False
        elif tracker is not None and tracker.anchor_logits is not None:
            sam_kwargs["mask_input"] = tracker.anchor_logits
            logger.info("SAM2: using anchor logits as prior")
        elif ref_logits is not None and (tracker is None or tracker.frame_count == 0):
            sam_kwargs["mask_input"] = ref_logits
            logger.info("SAM2: using reference-image logits as initial prior")

        # --- Run SAM2 ---
        masks, scores_sam, logits = pred.predict(**sam_kwargs)

        if masks is not None and len(masks) > 0:
            best_idx = int(np.argmax(scores_sam))
            mask_out = (masks[best_idx] > 0).astype(np.uint8) * 255

            # Coverage check: if the mask covers much less than the
            # detector box, the prior may still be biasing SAM2.
            # Re-run with box-only (no mask_input, no point prompts).
            if box_np is not None:
                bx1, by1, bx2, by2 = box_np.astype(int)
                box_area  = max(1, (bx2 - bx1) * (by2 - by1))
                mask_area = np.count_nonzero(mask_out)
                coverage  = mask_area / box_area
                if coverage < 0.35:
                    logger.info(f"SAM2: mask covers only {coverage:.0%} of box — "
                          f"re-running with box-only prompt")
                    pred.set_image(rgb)
                    masks2, scores2, logits2 = pred.predict(
                        box=box_np, multimask_output=True, return_logits=True)
                    if masks2 is not None and len(masks2) > 0:
                        best2    = int(np.argmax(scores2))
                        mask_out = (masks2[best2] > 0).astype(np.uint8) * 255
                        logits   = logits2
                        scores_sam = scores2
                        best_idx = best2
                        logger.info(f"SAM2: box-only mask score={scores2[best2]:.3f}")

            # Drift check
            if tracker is not None and tracker.prev_mask is not None:
                iou = _mask_iou(mask_out, tracker.prev_mask)
                if iou < IOU_DRIFT_THRESH and not need_detection:
                    logger.info(f"SAM2: mask IoU={iou:.2f} < {IOU_DRIFT_THRESH} — "
                          f"drift detected, resetting tracker (keeping anchor)")
                    tracker.reset(keep_anchor=True)
                    return run_pipeline(image_bgr, prompt, tracker,
                                        ref_logits=ref_logits,
                                        ref_crop=ref_crop,
                                        ref_features=ref_features)

            res["mask_np"]   = mask_out
            tracker_logits   = logits[best_idx: best_idx + 1]
            logger.info(f"SAM2: mask obtained (score={scores_sam[best_idx]:.3f})")
        else:
            logger.info("SAM2: no mask returned")

    except Exception as e:
        logger.exception("SAM2 failed: %s", e)

    # ── Step 3: Depth (inset visualisation) ──────────────────────────
    dm = _get_depth_model()
    if dm is not None:
        try:
            res["depth_np"] = dm.infer_image(image_bgr).astype(np.float32)
        except Exception as e:
            logger.error("Depth inference failed: %s", e)

    # ── Step 4: Grasp point — robust centroid → bbox fallback ─────────
    if res["mask_np"] is not None:
        mc = _robust_centroid(res["mask_np"])
        if mc is not None:
            res["best_centroid"] = mc
            logger.info(f"Grasp point: robust centroid {mc}")
        elif res["gdino_box"] is not None:
            x1, y1, x2, y2 = res["gdino_box"]
            res["best_centroid"] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            logger.info(f"Grasp point: bbox centre {res['best_centroid']} (centroid failed)")
    elif res["gdino_box"] is not None:
        x1, y1, x2, y2 = res["gdino_box"]
        res["best_centroid"] = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        logger.info(f"Grasp point: bbox centre {res['best_centroid']} (no SAM2 mask)")

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
        logger.error("get_position failed: %s", ret)
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
                logger.info(f"Robot not connected: {self.ip}")
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
            logger.info(f"Robot connected: {self.ip}")
            return True
        except Exception as e:
            logger.info(f"Robot connect failed: {e}")
            return False

    def calibrate(self, get_frame_fn):
        self.cal_status = "waiting for frame..."
        logger.info("\n=== Calibration: waiting for camera frame ===")

        deadline = time.time() + 30.0
        while time.time() < deadline:
            if get_frame_fn() is not None:
                break
            time.sleep(0.2)
        else:
            logger.info("Calibration skipped: no camera frame available.")
            self.cal_status = "skipped (no frame)"
            self.enabled    = True
            return

        pos0 = self._get_pos()
        if pos0 is None:
            logger.info("Calibration skipped: cannot read robot position.")
            self.cal_status = "skipped (pos read fail)"
            self.enabled    = True
            return

        logger.info(f"Home: [{pos0[0]:.1f}, {pos0[1]:.1f}, {pos0[2]:.1f}] mm")
        J_yz = np.zeros((2, 2), dtype=np.float64)

        for col, (robot_idx, ax) in enumerate([(1, "Y"), (2, "Z")]):
            self.cal_status = f"calibrating {ax}..."
            time.sleep(0.3)
            frame_before = get_frame_fn()
            if frame_before is None:
                logger.info(f"  [{ax}] no frame before move — skipping axis")
                continue

            logger.info(f"  [{ax}] moving +{CAL_DELTA:.0f} mm...")
            fwd = list(pos0)
            fwd[robot_idx] += CAL_DELTA
            self._move_abs(fwd, wait=True)
            time.sleep(CAL_WAIT)

            frame_after = get_frame_fn()
            logger.info(f"  [{ax}] returning home...")
            self._move_abs(pos0, wait=True)

            if frame_after is None:
                logger.info(f"  [{ax}] no frame after move — skipping axis")
                continue

            flow = _measure_flow(frame_before, frame_after)
            if flow is None:
                logger.info(f"  [{ax}] optical flow failed — skipping axis")
                continue

            dpx, dpy = flow
            J_yz[0, col] = dpx / CAL_DELTA
            J_yz[1, col] = dpy / CAL_DELTA
            logger.info(f"  [{ax}] flow: ({dpx:+.1f}, {dpy:+.1f}) px / {CAL_DELTA:.0f} mm"
                  f"  → J[:,{col}]=[{J_yz[0,col]:+.4f}, {J_yz[1,col]:+.4f}]")

        self._jac_yz  = J_yz
        logger.info(f"\n  J_yz (px/mm):\n{J_yz}")
        rank = np.linalg.matrix_rank(J_yz)

        if rank == 0:
            logger.info("  WARNING: J_yz rank=0 — approach-only mode.")
            self.cal_status = "skipped (no flow data)"
            self.enabled    = True
            return

        self._jac_yz_inv = np.linalg.pinv(J_yz)
        logger.info(f"  J_yz_inv (mm/px):\n{self._jac_yz_inv}")
        status = "calibrated" if rank == 2 else f"partial cal (rank={rank})"
        logger.info(f"=== Calibration complete [{status}] ===\n")
        self.cal_status = status
        self.enabled    = True

    def stop(self):
        self.enabled = False
        if self._arm is not None:
            try:
                self._arm.emergency_stop()
                logger.info("Robot stopped.")
            except Exception as e:
                logger.error("Robot stop failed: %s", e)

    def servo_step(self, centroid: tuple, image_shape: tuple):
        now = time.time()
        if self._arm is None:
            if now - self._last_t > 5.0:
                logger.info("Servo: arm not connected")
                self._last_t = now
            return
        if not self.enabled:
            if now - self._last_t > 5.0:
                logger.info(f"Servo: waiting [{self.cal_status}]")
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
                logger.info(f"Servo: centroid jump {jump:.0f} px — ignored")
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

                # Clamp Z so the arm never goes below its base height
                if new_z < MIN_Z_MM:
                    logger.info(f"Servo: clamping Z {new_z:.1f} → {MIN_Z_MM:.1f} mm "
                          f"(base limit)")
                    new_z = MIN_Z_MM

                self._arm.set_position(
                    x=new_x, y=new_y, z=new_z,
                    roll=pos[3], pitch=pos[4], yaw=pos[5],
                    speed=VS_SPEED, mvacc=VS_MVACC, wait=True)
                logger.info(f"Servo: err=({ex:+.0f},{ey:+.0f})px r={err_r:.0f}  "
                      f"Δ=({dx_mm:+.1f},{dy_mm:+.1f},{dz_mm:+.1f})mm"
                      f"  pos=({new_x:.0f},{new_y:.0f},{new_z:.0f})"
                      f"  [{self.cal_status}]")
            except Exception as e:
                logger.error("Servo step failed: %s", e)


# ═════════════════════════════════════════════════════════════════════
#  Camera streamer  (PyZED primary, OpenCV fallback)
# ═════════════════════════════════════════════════════════════════════

class CameraStreamer(threading.Thread):
    """
    Captures frames from the ZED camera (via PyZED or OpenCV),
    runs the segmentation pipeline in a background thread, overlays
    results, and records both a raw SVO feed (pyzed) and an annotated
    MP4 at ≥ MIN_RECORDING_DIM px.

    Parameters
    ----------
    cam_index     : OpenCV camera index (used only in OpenCV fallback mode)
    stop_event    : threading.Event to request shutdown
    robot         : RobotController instance (may have _arm=None)
    ref_image_bgr : optional BGR reference image used to condition detection
    use_pyzed     : if False, force OpenCV even when PyZED is available
    """

    def __init__(self, cam_index: int, stop_event: threading.Event,
                 robot: RobotController,
                 ref_image_bgr: np.ndarray | None = None,
                 use_pyzed: bool = True):
        super().__init__(daemon=True)
        self.cam_index     = cam_index
        self.stop_event    = stop_event
        self.robot         = robot
        self.ref_image_bgr = ref_image_bgr
        self._use_pyzed    = use_pyzed and PYZED_AVAILABLE

        self._latest_left  = None
        self._frame_lock   = threading.Lock()
        self.data_lock     = threading.Lock()
        self._result: dict = {}
        self._models_ready = threading.Event()
        self._tracker      = MaskTracker()

        self._undistorter = None   # created lazily from actual frame size

        # Populated by _process_ref_image() before the main seg loop
        self.ref_logits    = None
        self.ref_crop      = None
        self.ref_features  = None

    # ── helpers ──────────────────────────────────────────────────────
    def _get_frame(self):
        with self._frame_lock:
            if self._latest_left is not None:
                return self._latest_left.copy()
            return None

    def _run_calibration(self):
        logger.info("Calibration thread: waiting for models …")
        self._models_ready.wait()
        logger.info("Calibration thread: starting Y/Z Jacobian calibration.")
        self.robot.calibrate(self._get_frame)

    # ── reference image pre-processing (runs once at seg loop start) ─
    def _process_ref_image(self):
        if self.ref_image_bgr is None:
            return
        try:
            self.ref_crop, self.ref_logits, _, self.ref_features = process_ref_image(self.ref_image_bgr, PROMPT)
        except Exception as e:
            logger.error("Reference image processing error: %s", e)
            logger.exception("Details:")

    # ── segmentation loop ─────────────────────────────────────────────
    def _seg_loop(self):
        # Load models + process reference image before the main loop
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
                                   ref_features=self.ref_features)
                with self.data_lock:
                    self._result = res
                if not self._models_ready.is_set():
                    logger.info("Models ready — calibration may now proceed.")
                    self._models_ready.set()
                if self.robot is not None and res.get("best_centroid") is not None:
                    self.robot.servo_step(res["best_centroid"], frame.shape)
            except Exception as e:
                logger.error("Pipeline error: %s", e)
                logger.exception("Details:")
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
        init_params.depth_mode        = sl.DEPTH_MODE.NONE  # depth via DA-v2

        err = cam.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            logger.info(f"PyZED: camera open failed ({err}) — falling back to OpenCV")
            self._run_opencv()
            return

        ts = time.strftime("%Y%m%d_%H%M%S")

        # Raw stereo SVO recording
        svo_path   = os.path.abspath(time.strftime(f"vs_recording_{ts}.svo2"))
        rec_params = sl.RecordingParameters()
        rec_params.video_filename    = svo_path
        rec_params.compression_mode  = sl.SVO_COMPRESSION_MODE.H264
        err = cam.enable_recording(rec_params)
        if err == sl.ERROR_CODE.SUCCESS:
            logger.info(f"PyZED SVO recording → {svo_path}")
        else:
            logger.info(f"PyZED SVO recording failed ({err}) — continuing without raw SVO")

        # Annotated overlay MP4
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
                frame      = frame_bgra[:, :, :3].copy()  # BGRA → BGR
                # sl.VIEW.LEFT is already rectified by the ZED SDK

                with self._frame_lock:
                    self._latest_left = frame.copy()
                with self.data_lock:
                    res = dict(self._result)

                rendered = self._render(frame, res)

                # Ensure ≥ MIN_RECORDING_DIM on the shorter side
                rendered_write = self._ensure_min_dim(rendered)

                if video_writer is None:
                    rh, rw = rendered_write.shape[:2]
                    video_writer = cv2.VideoWriter(
                        anno_path, cv2.VideoWriter_fourcc(*"mp4v"),
                        30.0, (rw, rh))
                    if not video_writer.isOpened():
                        logger.info(f"ERROR: OpenCV VideoWriter failed for {anno_path} "
                              f"({rw}×{rh} mp4v). Install ffmpeg / try "
                              f"`sudo apt install ffmpeg` or run with "
                              f"--no-pyzed and a working codec backend.")
                        video_writer.release()
                        video_writer = None
                    else:
                        logger.info(f"Annotated recording → {anno_path}  ({rw}×{rh})")

                if video_writer is not None:
                    video_writer.write(rendered_write)
                cv2.imshow(win, rendered)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.stop_event.set()
                    break
                elif key == ord("v") and self.robot is not None:
                    self.robot.enabled = not self.robot.enabled
                    logger.info("Servo:", "ON" if self.robot.enabled else "OFF")
                elif key == ord("r"):
                    self._tracker.reset(keep_anchor=True)
                    logger.info("Tracker soft-reset (anchor kept)")
                elif key == ord("R"):  # shift-R → full reset
                    self._tracker.reset(keep_anchor=False)
                    logger.info("Tracker FULL reset (anchor cleared)")
        finally:
            cam.disable_recording()
            cam.close()
            if video_writer is not None:
                video_writer.release()
                logger.info(f"Annotated recording saved: {anno_path}")
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # ── OpenCV fallback path ──────────────────────────────────────────
    def _run_opencv(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_ANY)
        if not cap.isOpened():
            logger.info(f"Failed to open camera {self.cam_index}")
            return

        # Request a reasonable resolution from UVC
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  2560)  # ZED side-by-side at HD720
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
                left = frame[:, : w // 2].copy()   # left eye from side-by-side

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
                        logger.info(f"ERROR: OpenCV VideoWriter failed for {video_path} "
                              f"({rw}×{rh} mp4v). Try: `sudo apt install ffmpeg` "
                              f"or use a backend that supports MP4 encoding.")
                        video_writer.release()
                        video_writer = None
                    else:
                        logger.info(f"Recording → {video_path}  ({rw}×{rh})")

                if video_writer is not None:
                    video_writer.write(rendered_write)
                cv2.imshow(win, rendered)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.stop_event.set()
                    break
                elif key == ord("v") and self.robot is not None:
                    self.robot.enabled = not self.robot.enabled
                    logger.info("Servo:", "ON" if self.robot.enabled else "OFF")
                elif key == ord("r"):
                    self._tracker.reset(keep_anchor=True)
                    logger.info("Tracker soft-reset (anchor kept)")
                elif key == ord("R"):
                    self._tracker.reset(keep_anchor=False)
                    logger.info("Tracker FULL reset (anchor cleared)")
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
                if video_path:
                    logger.info(f"Recording saved: {video_path}")
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # ── video helpers ─────────────────────────────────────────────────
    @staticmethod
    def _ensure_min_dim(img: np.ndarray) -> np.ndarray:
        """Upscale image so its shorter side is ≥ MIN_RECORDING_DIM."""
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

        mask_np = res.get("mask_np")

        # Resize mask to display resolution
        mask_np_r = mask_np
        if mask_np_r is not None and mask_np_r.shape[:2] != (h, w):
            mask_np_r = cv2.resize(mask_np_r, (w, h),
                                   interpolation=cv2.INTER_NEAREST)

        # SAM2 mask overlay (green tint + contour)
        if mask_np_r is not None:
            overlay           = np.zeros_like(display)
            overlay[:, :, 1]  = mask_np_r
            display           = cv2.addWeighted(display, 0.72, overlay, 0.28, 0)
            cnts, _ = cv2.findContours(mask_np_r, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display, cnts, -1, (0, 255, 0), 2)

        return display


# ═════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SemVS: Semantic Visual Servoing with image-conditioned detection")
    parser.add_argument("cam_index", nargs="?", type=int, default=0,
                        help="OpenCV camera index (default: 0, ignored when using PyZED)")
    parser.add_argument("--detector", choices=["gdino", "owlv2"], default=None,
                        help="Override detector (default: env DETECTOR or gdino)")
    parser.add_argument("--ref-image", "--ref_image", metavar="PATH",
                        help="Path to a reference image of the target object. "
                             "Used to condition SAM2 and (for OWLv2) image-guided detection.")
    parser.add_argument("--no-pyzed", action="store_true",
                        help="Force OpenCV camera capture even if PyZED is available")
    parser.add_argument("--prompt", default=None,
                        help=f"Text prompt for the detector (default: '{PROMPT}')")
    args = parser.parse_args()

    if args.detector:
        DETECTOR = args.detector
    if args.prompt:
        PROMPT = args.prompt

    logger.info(f"Detector : {DETECTOR.upper()}")
    logger.info(f"Prompt   : {PROMPT}")
    logger.info(f"PyZED    : {'available' if PYZED_AVAILABLE else 'not available'}"
          f"{' (disabled by --no-pyzed)' if args.no_pyzed else ''}")

    # Load reference image if provided
    ref_image_bgr = None
    if args.ref_image:
        ref_image_bgr = cv2.imread(args.ref_image)
        if ref_image_bgr is None:
            logger.info(f"ERROR: could not load reference image: {args.ref_image}")
            sys.exit(1)
        logger.info(f"Reference image loaded: {args.ref_image}  "
              f"({ref_image_bgr.shape[1]}×{ref_image_bgr.shape[0]})")
    else:
        logger.info("No reference image provided — using text-prompt detection only")

    robot = RobotController(ROBOT_IP)
    robot.connect()

    stop_ev    = threading.Event()
    cam_thread = CameraStreamer(
        cam_index     = args.cam_index,
        stop_event    = stop_ev,
        robot         = robot,
        ref_image_bgr = ref_image_bgr,
        use_pyzed     = not args.no_pyzed,
    )
    cam_thread.start()

    try:
        cam_thread.join()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        robot.stop()
        stop_ev.set()
        cam_thread.join(timeout=2)
        logger.info("Done.")
