#!/usr/bin/env python3
"""
Visual Servoing Pipeline — DINOv2 Detection + SAM2 Video Tracking

Uses DINOv2 feature matching for initial object detection, then SAM2's
mask-logit propagation for fast frame-to-frame video tracking.  The robot
is moving and the ZED camera feed updates continuously, so per-frame
re-detection is too slow.  Instead:

  1. DETECT  (first frame / re-detection):
     DINOv2 patch matching → bbox → SAM2 bbox prompt → topmost mask + logits

  2. TRACK   (subsequent frames):
     SAM2 set_image + predict(mask_input=prev_logits) → updated mask + logits
     No DINOv2 needed — runs ~5-10× faster than full detection.

  3. RE-DETECT triggers:
     • SAM2 tracking score drops below threshold
     • Mask area changes drastically (>3× or <0.3× previous)
     • Every N frames (configurable, default 60)
     • Mask disappears entirely

Usage:
    python semvs_dinov2_servo.py \\
        --reference input_image_transparent.png \\
        [--cam 0] [--resnet-weight 0.0] [--threshold-pct 80] \\
        [--redetect-interval 60] [--track-score-thresh 0.5]
"""
import time
import sys
import threading
import os
import argparse

import cv2
import numpy as np

THIRD_PARTY_ROOT = os.path.join(os.path.dirname(__file__), "third-party")

SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT   = os.path.join(THIRD_PARTY_ROOT, "sam2/checkpoints/sam2.1_hiera_large.pt")
DA_CKPT_DIR = os.path.join(THIRD_PARTY_ROOT, "Depth-Anything-V2/checkpoints")
DA_ENCODER  = "vits"

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

CAL_DELTA = 8.0
CAL_WAIT  = 1.5

try:
    import torch
    import torch.nn.functional as F
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch  = None
    DEVICE = "cpu"


# ═════════════════════════════════════════════════════════════════════
#  DINOv2 feature extraction
# ═════════════════════════════════════════════════════════════════════

_dinov2_model = None


def _get_dinov2():
    global _dinov2_model
    if _dinov2_model is None:
        print("Loading DINOv2 (ViT-B/14)...")
        _dinov2_model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14",
            pretrained=True,
        )
        _dinov2_model = _dinov2_model.to(DEVICE).eval()
        print(f"DINOv2 loaded on {DEVICE}")
    return _dinov2_model


def _preprocess_for_dinov2(image_bgr: np.ndarray, patch_size: int = 14):
    h, w = image_bgr.shape[:2]
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    resized = cv2.resize(image_bgr, (new_w, new_h),
                         interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb  = (rgb - mean) / std
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    return tensor, new_h, new_w


def extract_patch_features(image_bgr: np.ndarray, patch_size: int = 14):
    model = _get_dinov2()
    tensor, proc_h, proc_w = _preprocess_for_dinov2(image_bgr, patch_size)
    with torch.no_grad():
        out = model.forward_features(tensor)
        patch_tokens = out["x_norm_patchtokens"]
    n_h = proc_h // patch_size
    n_w = proc_w // patch_size
    features = patch_tokens[0].reshape(n_h, n_w, -1).cpu().numpy()
    return features, proc_h, proc_w


# ═════════════════════════════════════════════════════════════════════
#  ResNet18 feature similarity
# ═════════════════════════════════════════════════════════════════════

_resnet_model = None


def _get_resnet():
    global _resnet_model
    if _resnet_model is None:
        import torchvision.models as models
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        _resnet_model = torch.nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        ).to(DEVICE).eval()
        print(f"ResNet18 feature extractor loaded on {DEVICE}")
    return _resnet_model


def _resnet_preprocess(image_bgr: np.ndarray):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb  = (rgb - mean) / std
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def compute_resnet_similarity(ref_bgr, ref_alpha, scene_bgr,
                              scene_proc_h, scene_proc_w, patch_size=14):
    model = _get_resnet()
    sh_patches = scene_proc_h // patch_size
    sw_patches = scene_proc_w // patch_size
    with torch.no_grad():
        ref_tensor = _resnet_preprocess(ref_bgr)
        ref_feats  = model(ref_tensor)
        fh, fw = ref_feats.shape[2], ref_feats.shape[3]
        ref_alpha_resized = cv2.resize(
            ref_alpha, (fw, fh), interpolation=cv2.INTER_NEAREST)
        fg_mask = torch.from_numpy(
            (ref_alpha_resized > 128).astype(np.float32)).to(DEVICE)
        if fg_mask.sum() < 1:
            return np.ones((sh_patches, sw_patches), dtype=np.float32)
        ref_feats_flat = ref_feats[0]
        fg_mask_3d     = fg_mask.unsqueeze(0)
        ref_fg_feat    = (ref_feats_flat * fg_mask_3d).sum(
            dim=(1, 2)) / fg_mask.sum()
        ref_fg_feat    = F.normalize(ref_fg_feat, dim=0)
        scene_resized = cv2.resize(
            scene_bgr, (scene_proc_w, scene_proc_h),
            interpolation=cv2.INTER_LINEAR)
        scene_tensor = _resnet_preprocess(scene_resized)
        scene_feats  = model(scene_tensor)
        scene_feats_norm = F.normalize(scene_feats[0], dim=0)
        sim = torch.einsum('d,dhw->hw', ref_fg_feat,
                           scene_feats_norm).cpu().numpy()
        resnet_map = cv2.resize(
            sim, (sw_patches, sh_patches), interpolation=cv2.INTER_LINEAR)
    return resnet_map


# ═════════════════════════════════════════════════════════════════════
#  Feature matching helpers
# ═════════════════════════════════════════════════════════════════════

def compute_similarity_map(ref_features, ref_mask_patches, scene_features):
    fg_mask  = ref_mask_patches > 0
    fg_feats = ref_features[fg_mask]
    if len(fg_feats) == 0:
        return np.zeros(scene_features.shape[:2], dtype=np.float32)
    fg_feats = fg_feats / (np.linalg.norm(
        fg_feats, axis=1, keepdims=True) + 1e-8)
    sh, sw, D  = scene_features.shape
    scene_flat = scene_features.reshape(-1, D)
    scene_flat = scene_flat / (np.linalg.norm(
        scene_flat, axis=1, keepdims=True) + 1e-8)
    sim_scores = np.zeros(scene_flat.shape[0], dtype=np.float32)
    chunk_size = 512
    for i in range(0, len(scene_flat), chunk_size):
        chunk = scene_flat[i:i + chunk_size]
        sim_scores[i:i + chunk_size] = (chunk @ fg_feats.T).mean(axis=1)
    return sim_scores.reshape(sh, sw)


def combine_similarity_maps(dinov2_sim, resnet_sim, alpha=0.5):
    d_min, d_max = dinov2_sim.min(), dinov2_sim.max()
    d_norm = ((dinov2_sim - d_min) / (d_max - d_min + 1e-8)
              if d_max - d_min > 1e-8 else np.zeros_like(dinov2_sim))
    r_min, r_max = resnet_sim.min(), resnet_sim.max()
    r_norm = ((resnet_sim - r_min) / (r_max - r_min + 1e-8)
              if r_max - r_min > 1e-8 else np.ones_like(resnet_sim))
    return alpha * d_norm + (1.0 - alpha) * r_norm


def similarity_to_bbox(sim_map, scene_h, scene_w, patch_size=14,
                       threshold_percentile=80):
    sim_upscaled = cv2.resize(
        sim_map, (scene_w, scene_h), interpolation=cv2.INTER_LINEAR)
    thresh_val = np.percentile(sim_upscaled, threshold_percentile)
    binary = (sim_upscaled >= thresh_val).astype(np.uint8) * 255
    kern   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kern, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kern, iterations=1)

    n_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    if n_cc <= 1:
        peak = np.unravel_index(np.argmax(sim_upscaled), sim_upscaled.shape)
        cy, cx = peak
        pad = 50
        return (max(0, cx - pad), max(0, cy - pad),
                min(scene_w, cx + pad), min(scene_h, cy + pad)), sim_upscaled
    best_cc, best_score = 1, -1.0
    min_area = 0.002 * scene_h * scene_w
    for cc_id in range(1, n_cc):
        if stats[cc_id, cv2.CC_STAT_AREA] < min_area:
            continue
        cc_mask  = labels == cc_id
        mean_sim = float(np.mean(sim_upscaled[cc_mask]))
        if mean_sim > best_score:
            best_score = mean_sim
            best_cc    = cc_id
    x, y, bw, bh, _ = stats[best_cc]
    pad = 15
    return (max(0, x - pad), max(0, y - pad),
            min(scene_w, x + bw + pad),
            min(scene_h, y + bh + pad)), sim_upscaled


# ═════════════════════════════════════════════════════════════════════
#  SAM2 predictor (shared instance for both detect and track)
# ═════════════════════════════════════════════════════════════════════

_sam2_pred = None


def _get_sam2():
    global _sam2_pred
    if _sam2_pred is None:
        sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, "sam2"))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model      = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=DEVICE)
        _sam2_pred = SAM2ImagePredictor(model)
        print(f"SAM2 loaded on {DEVICE}")
    return _sam2_pred


def _select_topmost_mask(mask: np.ndarray) -> np.ndarray:
    n_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n_cc <= 2:
        return mask
    best_cc, best_top_y = 1, stats[1, cv2.CC_STAT_TOP]
    for cc_id in range(2, n_cc):
        if stats[cc_id, cv2.CC_STAT_AREA] < 100:
            continue
        top_y = stats[cc_id, cv2.CC_STAT_TOP]
        if top_y < best_top_y:
            best_top_y = top_y
            best_cc    = cc_id
    return (labels == best_cc).astype(np.uint8) * 255


def _sam2_detect_with_bbox(image_bgr, bbox, select_topmost=True):
    """
    Full SAM2 detection from a bounding box prompt.
    Returns (mask, score, logits) — logits are kept for video tracking.
    """
    x1, y1, x2, y2 = bbox
    box_np = np.array([x1, y1, x2, y2], dtype=np.float32)

    pred = _get_sam2()
    rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pred.set_image(rgb)

    masks, scores, logits = pred.predict(
        box=box_np, multimask_output=True)

    best_idx    = int(np.argmax(scores))
    mask        = (masks[best_idx] > 0).astype(np.uint8) * 255
    score       = float(scores[best_idx])
    best_logits = logits[best_idx:best_idx + 1]  # (1, 256, 256)

    # Morphological cleanup
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=2)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)

    if not select_topmost:
        return mask, score, best_logits

    # ── Topmost-object selection ──────────────────────────────
    fg_rows = np.any(mask > 0, axis=1)
    fg_cols = np.any(mask > 0, axis=0)
    if not np.any(fg_rows):
        return mask, score, best_logits

    mask_y1 = int(np.argmax(fg_rows))
    mask_y2 = int(len(fg_rows) - np.argmax(fg_rows[::-1]))
    mask_x1 = int(np.argmax(fg_cols))
    mask_x2 = int(len(fg_cols) - np.argmax(fg_cols[::-1]))
    mask_h  = mask_y2 - mask_y1

    if mask_h == 0:
        return mask, score, best_logits

    # Strategy 1: connected components
    top_mask = _select_topmost_mask(mask)
    n_cc     = cv2.connectedComponentsWithStats(mask)[0] - 1
    if n_cc > 1:
        print(f"  Detect: split {n_cc} components, selected topmost")
        fg     = np.argwhere(top_mask > 0)
        cy, cx = fg.mean(axis=0).astype(int)
        point  = np.array([[cx, cy]], dtype=np.float32)
        label  = np.array([1], dtype=np.int32)
        masks2, scores2, logits2 = pred.predict(
            point_coords=point, point_labels=label,
            multimask_output=True)
        best2   = int(np.argmax(scores2))
        refined = (masks2[best2] > 0).astype(np.uint8) * 255
        refined = cv2.morphologyEx(
            refined, cv2.MORPH_CLOSE, kern, iterations=2)
        return refined, float(scores2[best2]), logits2[best2:best2 + 1]

    # Strategy 2: top-crop bbox re-segmentation
    crop_y2 = min(mask_y1 + int(mask_h * 0.4), image_bgr.shape[0])
    top_box = np.array(
        [mask_x1, mask_y1, mask_x2, crop_y2], dtype=np.float32)
    masks2, scores2, logits2 = pred.predict(
        box=top_box, multimask_output=True)
    if masks2 is not None and len(masks2) > 0:
        best2   = int(np.argmax(scores2))
        refined = (masks2[best2] > 0).astype(np.uint8) * 255
        refined = cv2.morphologyEx(
            refined, cv2.MORPH_CLOSE, kern, iterations=2)
        cnts, _ = cv2.findContours(
            refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(refined, cnts, -1, 255, cv2.FILLED)
        return refined, float(scores2[best2]), logits2[best2:best2 + 1]

    return mask, score, best_logits


def _sam2_track_with_logits(image_bgr, prev_logits, prev_bbox=None):
    """
    Fast SAM2 video tracking using previous-frame logits as mask prompt.

    SAM2's mask_input accepts low-resolution logits (1, 256, 256) from
    a previous prediction.  Combined with set_image on the new frame,
    this propagates the mask without needing DINOv2 re-detection.

    The previous bbox is expanded by 20% and supplied as a secondary
    prompt for more robust tracking when the object moves significantly.

    Returns (mask, score, logits).
    """
    pred = _get_sam2()
    rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pred.set_image(rgb)

    # Build prediction kwargs — logits as primary prompt
    predict_kwargs = dict(
        mask_input=prev_logits,      # (1, 256, 256) from prev frame
        multimask_output=False,      # single mask for tracking stability
    )

    # Add a loosened bbox prompt for robustness during motion
    if prev_bbox is not None:
        x1, y1, x2, y2 = prev_bbox
        bw, bh  = x2 - x1, y2 - y1
        pad_x   = int(bw * 0.1)
        pad_y   = int(bh * 0.1)
        h, w    = image_bgr.shape[:2]
        loose_box = np.array([
            max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(w, x2 + pad_x), min(h, y2 + pad_y),
        ], dtype=np.float32)
        predict_kwargs["box"] = loose_box

    masks, scores, logits = pred.predict(**predict_kwargs)

    best_idx    = int(np.argmax(scores))
    mask        = (masks[best_idx] > 0).astype(np.uint8) * 255
    score       = float(scores[best_idx])
    best_logits = logits[best_idx:best_idx + 1]

    # Light cleanup only — preserve mask continuity during tracking
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=1)

    return mask, score, best_logits


# ═════════════════════════════════════════════════════════════════════
#  VideoTracker — detect / track state machine
# ═════════════════════════════════════════════════════════════════════

class VideoTracker:
    """
    State machine for DINOv2-detect → SAM2-track loop.

    States
    ------
    NEED_DETECT : no valid tracking state, run full DINOv2 + SAM2
    TRACKING    : propagate mask via SAM2 logit input (fast path)
    """

    NEED_DETECT = "NEED_DETECT"
    TRACKING    = "TRACKING"

    def __init__(self, ref_cache, threshold_pct=80,
                 redetect_interval=60,
                 track_score_thresh=0.5,
                 area_change_thresh=3.0):
        self.ref_cache           = ref_cache
        self.threshold_pct       = threshold_pct
        self.redetect_interval   = redetect_interval
        self.track_score_thresh  = track_score_thresh
        self.area_change_thresh  = area_change_thresh

        # Tracking state
        self.state         = self.NEED_DETECT
        self._prev_logits  = None   # (1, 256, 256) SAM2 low-res logits
        self._prev_bbox    = None   # (x1, y1, x2, y2) tight mask bbox
        self._prev_area    = 0
        self._frame_count  = 0
        self._detect_count = 0
        self._track_count  = 0

    @staticmethod
    def _mask_to_bbox(mask):
        """Extract tight bounding box from binary mask."""
        fg = np.argwhere(mask > 0)
        if len(fg) == 0:
            return None
        y1, x1 = fg.min(axis=0)
        y2, x2 = fg.max(axis=0)
        return (int(x1), int(y1), int(x2), int(y2))

    def _needs_redetect(self, track_score, track_area):
        """Check if we should fall back to full DINOv2 re-detection."""
        if track_score < self.track_score_thresh:
            print(f"  Re-detect trigger: score {track_score:.3f} < "
                  f"{self.track_score_thresh}")
            return True
        if track_area < 100:
            print("  Re-detect trigger: mask vanished")
            return True
        if self._prev_area > 0:
            ratio = track_area / self._prev_area
            if (ratio > self.area_change_thresh
                    or ratio < 1.0 / self.area_change_thresh):
                print(f"  Re-detect trigger: area ratio {ratio:.2f}")
                return True
        if (self.redetect_interval > 0
                and self._track_count >= self.redetect_interval):
            print(f"  Re-detect trigger: periodic "
                  f"(every {self.redetect_interval} frames)")
            return True
        return False

    def process_frame(self, image_bgr: np.ndarray) -> dict:
        """
        Process one frame.  Decides detect vs track automatically.
        Returns dict with mask_np, sam_score, best_centroid, bbox,
        mode, sim_upscaled.
        """
        self._frame_count += 1
        result = dict(
            mask_np=None, sam_score=0.0, best_centroid=None,
            bbox=None, mode=self.state, sim_upscaled=None,
        )

        if self.state == self.NEED_DETECT:
            return self._run_detect(image_bgr, result)
        else:
            return self._run_track(image_bgr, result)

    def _run_detect(self, image_bgr, result):
        """Full DINOv2 detection + SAM2 with topmost selection."""
        t0     = time.time()
        sh, sw = image_bgr.shape[:2]
        ps     = self.ref_cache.patch_size

        # DINOv2 scene features
        scene_features, scene_proc_h, scene_proc_w = \
            extract_patch_features(image_bgr, ps)

        # Similarity map
        dinov2_sim = compute_similarity_map(
            self.ref_cache.ref_features,
            self.ref_cache.ref_mask_patches,
            scene_features)

        rw = self.ref_cache.resnet_weight
        if rw > 0.0:
            resnet_sim = compute_resnet_similarity(
                self.ref_cache.ref_bgr, self.ref_cache.ref_alpha,
                image_bgr, scene_proc_h, scene_proc_w, ps)
            sim_map = combine_similarity_maps(
                dinov2_sim, resnet_sim, alpha=1.0 - rw)
        else:
            sim_map = dinov2_sim

        # Bbox from similarity
        bbox, sim_upscaled = similarity_to_bbox(
            sim_map, sh, sw, ps, self.threshold_pct)

        # SAM2 detection with topmost-object selection → get logits
        mask, score, logits = _sam2_detect_with_bbox(
            image_bgr, bbox, select_topmost=True)

        mask_area = np.count_nonzero(mask)
        dt = time.time() - t0

        self._detect_count += 1
        self._track_count   = 0
        print(f"DETECT #{self._detect_count}: score={score:.3f} "
              f"area={mask_area}px  dt={dt:.2f}s")

        if mask_area > 100 and score > 0.3:
            # Transition to tracking — store logits for propagation
            self.state        = self.TRACKING
            self._prev_logits = logits
            self._prev_bbox   = self._mask_to_bbox(mask) or bbox
            self._prev_area   = mask_area
        else:
            print("  Detection too weak — will retry next frame")

        centroid = self._mask_centroid(mask)

        result.update(
            mask_np=mask, sam_score=score,
            best_centroid=centroid, bbox=bbox,
            mode="detect", sim_upscaled=sim_upscaled,
        )
        return result

    def _run_track(self, image_bgr, result):
        """Fast SAM2 video tracking via logit propagation."""
        t0 = time.time()

        mask, score, logits = _sam2_track_with_logits(
            image_bgr, self._prev_logits, self._prev_bbox)

        mask_area = np.count_nonzero(mask)
        dt = time.time() - t0

        self._track_count += 1

        # Check if re-detection is needed
        if self._needs_redetect(score, mask_area):
            self.state = self.NEED_DETECT
            print(f"TRACK #{self._track_count}: score={score:.3f} "
                  f"area={mask_area}  dt={dt:.3f}s → re-detect next")
        else:
            # Update tracking state for next frame
            self._prev_logits = logits
            new_bbox = self._mask_to_bbox(mask)
            if new_bbox is not None:
                self._prev_bbox = new_bbox
            self._prev_area = mask_area
            if self._track_count % 10 == 0:
                print(f"TRACK #{self._track_count}: score={score:.3f} "
                      f"area={mask_area}  dt={dt:.3f}s")

        centroid = self._mask_centroid(mask)

        result.update(
            mask_np=mask, sam_score=score,
            best_centroid=centroid,
            bbox=self._prev_bbox,
            mode="track",
        )
        return result

    def force_redetect(self):
        """Externally trigger re-detection (e.g. keyboard shortcut)."""
        self.state = self.NEED_DETECT
        print("Forced re-detection scheduled")

    @staticmethod
    def _mask_centroid(mask):
        fg = np.argwhere(mask > 0)
        if len(fg) == 0:
            return None
        cy, cx = fg.mean(axis=0).astype(int)
        return (int(cx), int(cy))


# ═════════════════════════════════════════════════════════════════════
#  Depth model + quality scoring
# ═════════════════════════════════════════════════════════════════════

_depth_model = None


def _get_depth_model():
    global _depth_model
    if _depth_model is None:
        try:
            sys.path.insert(0, os.path.join(
                THIRD_PARTY_ROOT, "Depth-Anything-V2"))
            from depth_anything_v2.dpt import DepthAnythingV2
            cfgs = {
                "vits": {"encoder": "vits", "features": 64,
                         "out_channels": [48, 96, 192, 384]},
                "vitb": {"encoder": "vitb", "features": 128,
                         "out_channels": [96, 192, 384, 768]},
                "vitl": {"encoder": "vitl", "features": 256,
                         "out_channels": [256, 512, 1024, 1024]},
                "vitg": {"encoder": "vitg", "features": 384,
                         "out_channels": [1536, 1536, 1536, 1536]},
            }
            ckpt = os.path.join(
                DA_CKPT_DIR, f"depth_anything_v2_{DA_ENCODER}.pth")
            if not os.path.exists(ckpt):
                print(f"Depth-Anything-V2 checkpoint not found: {ckpt}")
                return None
            m = DepthAnythingV2(**cfgs[DA_ENCODER])
            m.load_state_dict(torch.load(ckpt, map_location="cpu"))
            _depth_model = m.to(DEVICE).eval()
        except Exception as e:
            print("Depth-Anything-V2 load failed:", e)
    return _depth_model


def compute_quality_score(depth, mask):
    score = np.zeros(depth.shape[:2], dtype=np.float32)
    m = mask > 0
    if not m.any():
        return score
    d       = depth.astype(np.float32)
    d_min   = float(np.nanmin(d[m]))
    d_range = float(np.nanmax(d[m])) - d_min + 1e-6
    d_norm  = (d - d_min) / d_range

    gx       = cv2.Sobel(d_norm, cv2.CV_32F, 1, 0, ksize=5)
    gy       = cv2.Sobel(d_norm, cv2.CV_32F, 0, 1, ksize=5)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    k       = np.ones((11, 11), np.float32) / 121.0
    mean_d  = cv2.filter2D(d_norm, -1, k)
    mean_d2 = cv2.filter2D(d_norm ** 2, -1, k)
    local_std = np.sqrt(np.maximum(mean_d2 - mean_d ** 2, 0.0))

    g_scale = float(np.percentile(grad_mag[m], 75)) + 1e-6
    s_scale = float(np.percentile(local_std[m], 75)) + 1e-6

    combined = (np.exp(-2.0 * grad_mag / g_scale) *
                np.exp(-2.0 * local_std / s_scale)).astype(np.float32)
    combined[~m] = 0.0
    return combined


def find_best_region(score_map, threshold=0.45, min_area=300):
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
    cxy = (int(centroids[best_lbl][0]), int(centroids[best_lbl][1]))
    cnts, _ = cv2.findContours(
        region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cxy, (cnts[0] if cnts else None), region_mask


# ═════════════════════════════════════════════════════════════════════
#  Reference feature cache
# ═════════════════════════════════════════════════════════════════════

class ReferenceCache:
    def __init__(self, ref_bgr, ref_alpha, resnet_weight=0.0,
                 patch_size=14):
        self.ref_bgr       = ref_bgr
        self.ref_alpha     = ref_alpha
        self.resnet_weight = resnet_weight
        self.patch_size    = patch_size

        print("\n--- Precomputing reference DINOv2 features ---")
        self.ref_features, self.ref_proc_h, self.ref_proc_w = \
            extract_patch_features(ref_bgr, patch_size)

        ref_alpha_resized = cv2.resize(
            ref_alpha, (self.ref_proc_w, self.ref_proc_h),
            interpolation=cv2.INTER_NEAREST)
        rh_patches = self.ref_proc_h // patch_size
        rw_patches = self.ref_proc_w // patch_size
        self.ref_mask_patches = np.zeros(
            (rh_patches, rw_patches), dtype=np.uint8)
        for py in range(rh_patches):
            for px in range(rw_patches):
                patch_region = ref_alpha_resized[
                    py * patch_size:(py + 1) * patch_size,
                    px * patch_size:(px + 1) * patch_size,
                ]
                if np.mean(patch_region > 128) > 0.5:
                    self.ref_mask_patches[py, px] = 1

        fg = np.count_nonzero(self.ref_mask_patches)
        print(f"  Reference fg patches: {fg}/{self.ref_mask_patches.size}")


# ═════════════════════════════════════════════════════════════════════
#  Full pipeline step (tracker + depth + quality)
# ═════════════════════════════════════════════════════════════════════

def run_pipeline(image_bgr: np.ndarray, tracker: VideoTracker) -> dict:
    """
    One perception step.  The VideoTracker decides detect vs track.
    Depth + quality scoring runs at reduced rate during tracking.
    """
    # ── Segmentation (detect or track) ───────────────────────────
    res = tracker.process_frame(image_bgr)

    # ── Depth estimation — every frame on detect, every 5th on track
    run_depth = (res["mode"] == "detect"
                 or tracker._track_count % 5 == 0)

    if run_depth:
        dm = _get_depth_model()
        if dm is not None:
            try:
                res["depth_np"] = dm.infer_image(
                    image_bgr).astype(np.float32)
            except Exception as e:
                print("Depth failed:", e)
                res["depth_np"] = None
        else:
            res["depth_np"] = None
    else:
        res["depth_np"] = None

    # ── Quality scoring ──────────────────────────────────────────
    res["score_map"]    = None
    res["best_contour"] = None
    res["region_mask"]  = None
    res["mean_score"]   = 0.0

    if (res["depth_np"] is not None
            and res["mask_np"] is not None
            and np.count_nonzero(res["mask_np"]) > 0):
        score_map = compute_quality_score(
            res["depth_np"], res["mask_np"])
        best_cxy, best_cnt, reg_mask = find_best_region(score_map)
        res["score_map"]    = score_map
        res["best_contour"] = best_cnt
        res["region_mask"]  = reg_mask
        masked = score_map[res["mask_np"] > 0]
        res["mean_score"] = float(masked.mean()) if len(masked) else 0.0

        # Override centroid with quality-region centroid if available
        if best_cxy is not None:
            res["best_centroid"] = best_cxy

    return res


# ═════════════════════════════════════════════════════════════════════
#  Optical flow for calibration
# ═════════════════════════════════════════════════════════════════════

def _measure_flow(frame_a, frame_b):
    ga = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(
        ga, maxCorners=300, qualityLevel=0.01,
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
            self._arm    = arm
            self.enabled = False
            self.cal_status = "waiting for calibration..."
            print(f"Robot connected: {self.ip}")
            return True
        except Exception as e:
            print(f"Robot connect failed: {e}")
            return False

    def calibrate(self, get_frame_fn):
        self.cal_status = "waiting for frame..."
        deadline = time.time() + 30.0
        while time.time() < deadline:
            if get_frame_fn() is not None:
                break
            time.sleep(0.2)
        else:
            self.cal_status = "skipped (no frame)"
            self.enabled = True
            return

        pos0 = self._get_pos()
        if pos0 is None:
            self.cal_status = "skipped (pos read fail)"
            self.enabled = True
            return

        J_yz = np.zeros((2, 2), dtype=np.float64)
        for col, (robot_idx, ax) in enumerate([(1, "Y"), (2, "Z")]):
            self.cal_status = f"calibrating {ax}..."
            time.sleep(0.3)
            frame_before = get_frame_fn()
            if frame_before is None:
                continue
            fwd = list(pos0)
            fwd[robot_idx] += CAL_DELTA
            self._move_abs(fwd, wait=True)
            time.sleep(CAL_WAIT)
            frame_after = get_frame_fn()
            self._move_abs(pos0, wait=True)
            if frame_after is None:
                continue
            flow = _measure_flow(frame_before, frame_after)
            if flow is None:
                continue
            dpx, dpy = flow
            J_yz[0, col] = dpx / CAL_DELTA
            J_yz[1, col] = dpy / CAL_DELTA

        self._jac_yz = J_yz
        rank = np.linalg.matrix_rank(J_yz)
        if rank == 0:
            self.cal_status = "skipped (no flow data)"
            self.enabled = True
            return
        self._jac_yz_inv = np.linalg.pinv(J_yz)
        self.cal_status = ("calibrated" if rank == 2
                           else f"partial cal (rank={rank})")
        self.enabled = True

    def stop(self):
        self.enabled = False
        if self._arm is not None:
            try:
                self._arm.emergency_stop()
            except Exception:
                pass

    def servo_step(self, centroid, image_shape):
        now = time.time()
        if self._arm is None or not self.enabled:
            return
        if now - self._last_t < VS_RATE:
            return
        if now - self._last_centroid_t > 1.0:
            self._last_centroid = None
        if self._last_centroid is not None:
            jump = np.hypot(centroid[0] - self._last_centroid[0],
                            centroid[1] - self._last_centroid[1])
            if jump > MAX_JUMP_PX:
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
            dy_mm = dz_mm = 0.0
        elif self._jac_yz_inv is not None:
            err   = np.array([ex, ey], dtype=np.float64)
            yz    = -CTRL_GAIN * (self._jac_yz_inv @ err)
            dy_mm = float(np.clip(yz[0], -MAX_YZ_STEP, MAX_YZ_STEP))
            dz_mm = float(np.clip(yz[1], -MAX_YZ_STEP, MAX_YZ_STEP))
        else:
            dy_mm = dz_mm = 0.0

        with self._lock:
            try:
                pos = self._get_pos()
                if pos is None:
                    return
                self._arm.set_position(
                    x=pos[0] + dx_mm, y=pos[1] + dy_mm,
                    z=pos[2] + dz_mm,
                    roll=pos[3], pitch=pos[4], yaw=pos[5],
                    speed=VS_SPEED, mvacc=VS_MVACC, wait=True)
                print(f"Servo: err=({ex:+.0f},{ey:+.0f})px "
                      f"r={err_r:.0f}  "
                      f"Δ=({dx_mm:+.1f},{dy_mm:+.1f},{dz_mm:+.1f})mm")
            except Exception as e:
                print("Servo step failed:", e)


# ═════════════════════════════════════════════════════════════════════
#  Camera streamer (detect/track loop + keyboard controls)
# ═════════════════════════════════════════════════════════════════════

class CameraStreamer(threading.Thread):
    def __init__(self, cam_index, stop_event, robot, tracker):
        super().__init__(daemon=True)
        self.cam_index  = cam_index
        self.stop_event = stop_event
        self.robot      = robot
        self.tracker    = tracker

        self._latest_left  = None
        self._frame_lock   = threading.Lock()
        self.data_lock     = threading.Lock()
        self._result: dict = {}
        self._models_ready = threading.Event()

    def _get_frame(self):
        with self._frame_lock:
            if self._latest_left is not None:
                return self._latest_left.copy()
            return None

    def _run_calibration(self):
        self._models_ready.wait()
        self.robot.calibrate(self._get_frame)

    def _seg_loop(self):
        while not self.stop_event.is_set():
            with self._frame_lock:
                frame = (self._latest_left.copy()
                         if self._latest_left is not None else None)
            if frame is None:
                time.sleep(0.1)
                continue
            try:
                res = run_pipeline(frame, self.tracker)
                with self.data_lock:
                    self._result = res
                if not self._models_ready.is_set():
                    print("Models ready — calibration may proceed.")
                    self._models_ready.set()
                if (self.robot is not None
                        and res.get("best_centroid") is not None):
                    self.robot.servo_step(
                        res["best_centroid"], frame.shape)
            except Exception as e:
                import traceback
                print("Pipeline error:", e)
                traceback.print_exc()
                time.sleep(1)

    def run(self):
        cap = cv2.VideoCapture(self.cam_index, cv2.CAP_ANY)
        if not cap.isOpened():
            print(f"Failed to open camera {self.cam_index}")
            return

        threading.Thread(target=self._seg_loop, daemon=True).start()
        if self.robot is not None and self.robot._arm is not None:
            threading.Thread(target=self._run_calibration,
                             daemon=True).start()

        win = ("DINOv2+SAM2 Track  |  "
               "[v]servo [d]redetect [q]quit")
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
                left = frame[:, :w // 2].copy()

                with self._frame_lock:
                    self._latest_left = left.copy()
                with self.data_lock:
                    res = dict(self._result)

                rendered = self._render(left, res)

                if video_writer is None:
                    rh, rw = rendered.shape[:2]
                    video_path = time.strftime(
                        "dinov2_track_%Y%m%d_%H%M%S.mp4")
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
                    print("Servo:",
                          "ON" if self.robot.enabled else "OFF")
                elif key == ord("d"):
                    self.tracker.force_redetect()
        finally:
            cap.release()
            if video_writer is not None:
                video_writer.release()
                print(f"Recording saved: {video_path}")
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _render(self, left, res):
        display = left.copy()
        h, w    = display.shape[:2]

        mask_np       = res.get("mask_np")
        depth_np      = res.get("depth_np")
        score_map     = res.get("score_map")
        best_centroid = res.get("best_centroid")
        best_contour  = res.get("best_contour")
        bbox          = res.get("bbox")
        mean_score    = res.get("mean_score", 0.0)
        sam_score     = res.get("sam_score", 0.0)
        sim_upscaled  = res.get("sim_upscaled")
        mode          = res.get("mode", "?")

        def _resize_to(arr, interp=cv2.INTER_NEAREST):
            if arr is not None and arr.shape[:2] != (h, w):
                return cv2.resize(arr, (w, h), interpolation=interp)
            return arr

        mask_np_r   = _resize_to(mask_np)
        score_map_r = (_resize_to(score_map, cv2.INTER_LINEAR)
                       if score_map is not None else None)

        # ── Mask overlay: green=detect, teal=track ───────────────
        if mask_np_r is not None:
            overlay = np.zeros_like(display)
            if mode == "track":
                overlay[:, :, 0] = (mask_np_r * 0.6).astype(np.uint8)
                overlay[:, :, 1] = mask_np_r
            else:
                overlay[:, :, 1] = mask_np_r
            display = cv2.addWeighted(display, 0.72, overlay, 0.28, 0)

        # ── Quality heatmap ──────────────────────────────────────
        if score_map_r is not None and mask_np_r is not None:
            heat = cv2.applyColorMap(
                (np.clip(score_map_r, 0, 1) * 255).astype(np.uint8),
                cv2.COLORMAP_JET)
            m3 = (mask_np_r[:, :, None] > 0).astype(np.float32)
            display = (display * (1 - m3 * 0.6)
                       + heat * m3 * 0.6).astype(np.uint8)

        if best_contour is not None:
            cv2.drawContours(
                display, [best_contour], -1, (255, 255, 255), 2)

        # ── Bbox ─────────────────────────────────────────────────
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = (255, 255, 0) if mode == "detect" else (255, 180, 0)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

        # ── Image centre + servo target ──────────────────────────
        ic = (w // 2, h // 2)
        cv2.drawMarker(display, ic, (255, 80, 0),
                       cv2.MARKER_CROSS, 22, 2)

        if best_centroid is not None:
            bcp = best_centroid
            cv2.drawMarker(display, bcp, (0, 0, 255),
                           cv2.MARKER_CROSS, 26, 2)
            cv2.arrowedLine(display, ic, bcp, (0, 220, 255), 2,
                            tipLength=0.12)
            ex = bcp[0] - ic[0]
            ey = bcp[1] - ic[1]
            cv2.putText(display, f"err ({ex:+d},{ey:+d}) px",
                        (8, h - 58), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (0, 220, 255), 1)

        # ── Mode + score HUD ─────────────────────────────────────
        mode_color = (0, 255, 80) if mode == "track" else (0, 200, 255)
        mode_txt = (f"{mode.upper()}  SAM={sam_score:.2f}  "
                    f"q={mean_score:.2f}")
        cv2.putText(display, mode_txt, (8, h - 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, mode_color, 1)

        # ── Servo status ─────────────────────────────────────────
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

        # ── Inset: similarity on detect, depth on track ──────────
        inset_img   = None
        inset_label = None
        if sim_upscaled is not None:
            su = sim_upscaled.astype(np.float32)
            su_mn, su_mx = su.min(), su.max()
            su_u8 = ((su - su_mn) / (su_mx - su_mn + 1e-6)
                     * 255).astype(np.uint8)
            inset_img   = cv2.applyColorMap(su_u8, cv2.COLORMAP_JET)
            inset_label = "DINOv2 sim"
        elif depth_np is not None:
            d     = depth_np.astype(np.float32)
            d_mn  = float(np.nanmin(d))
            d_mx  = float(np.nanmax(d))
            d_u8  = ((d - d_mn) / (d_mx - d_mn + 1e-6)
                     * 255).astype(np.uint8)
            inset_img   = cv2.applyColorMap(d_u8, cv2.COLORMAP_JET)
            inset_label = "depth"

        if inset_img is not None:
            try:
                iw    = max(100, w // 4)
                ih    = int(iw * h / w)
                inset = cv2.resize(inset_img, (iw, ih))
                display[10:10 + ih, w - iw - 10:w - 10] = inset
                cv2.putText(display, inset_label,
                            (w - iw - 8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (200, 200, 200), 1)
            except Exception:
                pass

        return display


# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DINOv2 detection + SAM2 video tracking servo",
    )
    parser.add_argument("--reference", "-r", required=True,
                        help="Reference image (RGBA with transparent bg)")
    parser.add_argument("--cam", type=int, default=0,
                        help="Camera index (default: 0)")
    parser.add_argument("--resnet-weight", type=float, default=0.0,
                        help="ResNet18 fusion weight (0=DINOv2 only)")
    parser.add_argument("--threshold-pct", type=float, default=80,
                        help="Percentile threshold for similarity map")
    parser.add_argument("--redetect-interval", type=int, default=60,
                        help="Force re-detection every N tracking frames "
                             "(0=never, default=60)")
    parser.add_argument("--track-score-thresh", type=float, default=0.5,
                        help="SAM2 score below which re-detection triggers")
    parser.add_argument("--no-robot", action="store_true",
                        help="Perception only, no robot connection")
    args = parser.parse_args()

    # ── Load reference image ─────────────────────────────────────
    ref_full = cv2.imread(args.reference, cv2.IMREAD_UNCHANGED)
    if ref_full is None:
        print(f"ERROR: could not load reference: {args.reference}")
        sys.exit(1)

    if ref_full.shape[2] == 4:
        ref_bgr   = ref_full[:, :, :3]
        ref_alpha = ref_full[:, :, 3]
        print(f"Reference: {ref_bgr.shape[1]}x{ref_bgr.shape[0]}, "
              f"fg pixels: {np.count_nonzero(ref_alpha > 128)}")
    else:
        ref_bgr   = ref_full
        ref_alpha = np.ones(ref_full.shape[:2], dtype=np.uint8) * 255
        print("WARNING: reference has no alpha — treating full image "
              "as object")

    # ── Build reference cache + tracker ──────────────────────────
    ref_cache = ReferenceCache(
        ref_bgr, ref_alpha, resnet_weight=args.resnet_weight)

    tracker = VideoTracker(
        ref_cache,
        threshold_pct=args.threshold_pct,
        redetect_interval=args.redetect_interval,
        track_score_thresh=args.track_score_thresh,
    )

    # ── Robot ────────────────────────────────────────────────────
    robot = RobotController(ROBOT_IP)
    if not args.no_robot:
        robot.connect()
    else:
        print("Running in perception-only mode (--no-robot)")

    # ── Camera + servo loop ──────────────────────────────────────
    stop_ev    = threading.Event()
    cam_thread = CameraStreamer(
        args.cam, stop_ev, robot, tracker)
    cam_thread.start()

    try:
        cam_thread.join()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        robot.stop()
        stop_ev.set()
        cam_thread.join(timeout=2)
        print("Done.")


if __name__ == "__main__":
    main()
