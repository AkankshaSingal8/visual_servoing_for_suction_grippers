"""
Detection backends for the visual servoing pipeline.

Each backend exposes a callable with signature:
    runner(frame_bgr: np.ndarray) -> dict

Return-dict keys (all backends):
    mask_np              np.ndarray (H,W) uint8 or None
    gdino_box            [x1, y1, x2, y2] or None
    sam_score            float or None
    best_centroid        (float, float) or None
    detector_used        str  — "sam3" | "gdino" | "gdino+sam2"
    detection_was_skipped bool
    dets_all             list or None
    similarity           float or None

Available backends
------------------
sam3        GroundingDINO-free single-model detection+segmentation via
            facebook/sam3 (HuggingFace transformers).
gdino       GroundingDINO (HF transformers) detection only; no mask.
            Returns bounding-box centre as best_centroid.
gdino+sam2  GroundingDINO detection followed by SAM2 image-predictor
            segmentation.  Falls back to gdino-only if SAM2 is absent.
"""

import logging
import os
import sys

import cv2
import numpy as np

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

AVAILABLE_BACKENDS = ["sam3", "gdino", "gdino+sam2"]

# ---------------------------------------------------------------------------
# Repo-root discovery
# ---------------------------------------------------------------------------

def _find_repo_root(start: str) -> str | None:
    current = os.path.abspath(start)
    for _ in range(10):
        if (os.path.exists(os.path.join(current, "ral_paper_plan.md"))
                or os.path.isdir(os.path.join(current, "foundation_model"))):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


_THIS_DIR  = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = _find_repo_root(_THIS_DIR) or os.path.dirname(_THIS_DIR)
_THIRD_PARTY = os.path.join(_REPO_ROOT, "foundation_model", "third-party")

# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

_device_cache: str | None = None


def _device() -> str:
    global _device_cache
    if _device_cache is None:
        try:
            import torch as _t
            _device_cache = "cuda" if _t.cuda.is_available() else "cpu"
        except Exception:
            _device_cache = "cpu"
    return _device_cache


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _mask_centroid(mask: np.ndarray):
    M = cv2.moments(mask, binaryImage=True)
    if M["m00"] < 1.0:
        return None
    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))


def _box_center(box):
    x1, y1, x2, y2 = box
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


# ---------------------------------------------------------------------------
# Backend 1: SAM3
# ---------------------------------------------------------------------------

def make_sam3_runner(ref_image_path: str, prompt: str = "box"):
    try:
        from foundation_model.servo_pipeline_sam3 import (
            run_pipeline, MaskTracker, process_ref_image, _load_ref_image)
    except Exception:
        try:
            from servo_pipeline_sam3 import (
                run_pipeline, MaskTracker, process_ref_image, _load_ref_image)
        except Exception as exc:
            log.warning("SAM3 pipeline unavailable (%s); using stub runner.", exc)

            def _stub(frame_bgr: np.ndarray) -> dict:
                return _gdino_stub_result()
            return _stub

    tracker   = MaskTracker()
    ref_crop  = None
    ref_feats = None

    if ref_image_path and os.path.exists(ref_image_path):
        ref_bgr = _load_ref_image(ref_image_path)
        if ref_bgr is not None:
            ref_crop, ref_feats = process_ref_image(ref_bgr, prompt)

    def runner(frame_bgr: np.ndarray) -> dict:
        res = run_pipeline(frame_bgr, prompt, tracker,
                           ref_crop=ref_crop, ref_features=ref_feats)
        res.setdefault("detector_used", "sam3")
        if res.get("detector_used") is None:
            res["detector_used"] = "sam3"
        return res

    return runner


# ---------------------------------------------------------------------------
# Backend 2: GDINO only
# ---------------------------------------------------------------------------

def _gdino_stub_result() -> dict:
    return dict(
        mask_np=None,
        gdino_box=None,
        sam_score=0.0,
        best_centroid=None,
        detector_used="gdino",
        detection_was_skipped=True,
        dets_all=None,
        similarity=None,
    )


def make_gdino_runner(ref_image_path: str, prompt: str = "box"):
    gdino_path = os.path.join(_THIRD_PARTY, "GroundingDINO")
    if os.path.isdir(gdino_path) and gdino_path not in sys.path:
        sys.path.insert(0, gdino_path)

    state = dict(
        processor=None,
        model=None,
        failed=False,
    )

    gdino_hf_model_id = os.environ.get(
        "GDINO_HF_MODEL_ID", "IDEA-Research/grounding-dino-tiny")

    def _load_gdino():
        if state["failed"] or state["model"] is not None:
            return state["processor"], state["model"]
        try:
            from transformers import (
                AutoProcessor, AutoModelForZeroShotObjectDetection)
            log.info("Loading GroundingDINO (HF): %s on %s",
                     gdino_hf_model_id, _device())
            state["processor"] = AutoProcessor.from_pretrained(gdino_hf_model_id)
            state["model"] = (
                AutoModelForZeroShotObjectDetection
                .from_pretrained(gdino_hf_model_id)
                .to(_device()).eval()
            )
            log.info("GroundingDINO loaded: %s", gdino_hf_model_id)
        except Exception as exc:
            log.warning("GroundingDINO load failed (won't retry): %s", exc)
            state["failed"] = True
        return state["processor"], state["model"]

    def _run_gdino(frame_bgr: np.ndarray, text_prompt: str):
        processor, model = _load_gdino()
        if model is None:
            return []

        caption_raw = text_prompt.strip()
        if caption_raw.lower() in {"box", "box."}:
            text_labels = [["box", "cardboard box", "package", "carton", "parcel"]]
        else:
            parts = [p.strip() for p in caption_raw.replace(".", "\n").splitlines()
                     if p.strip()]
            text_labels = [parts] if parts else [[caption_raw]]

        try:
            import torch as _t
            from PIL import Image

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            h, w = frame_bgr.shape[:2]

            inputs = processor(images=pil, text=text_labels,
                               return_tensors="pt").to(_device())

            with _t.no_grad():
                outputs = model(**inputs)

            def _post(thresh: float, text_thresh: float):
                return processor.post_process_grounded_object_detection(
                    outputs,
                    input_ids=inputs["input_ids"],
                    threshold=thresh,
                    text_threshold=text_thresh,
                    target_sizes=[(h, w)])[0]

            result = _post(0.25, 0.25)
            if len(result["boxes"]) == 0:
                log.info("GDINO: empty at 0.25/0.25, retrying at 0.18/0.18")
                result = _post(0.18, 0.18)

            boxes  = result["boxes"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            labels = (result.get("text_labels") or result.get("labels")
                      or [""] * len(boxes))
            labels = [str(lb) for lb in labels]

            if len(boxes) == 0:
                log.warning("GDINO: no detections for %s", text_labels[0])
                return []

            dets = [(boxes[i].astype(np.float32), float(scores[i]),
                     labels[i] if i < len(labels) else "")
                    for i in range(len(boxes))]
            log.info("GDINO: %d detection(s) for %s", len(dets), text_labels[0])
            return dets

        except Exception as exc:
            log.error("GroundingDINO inference failed: %s", exc)
            return []

    def runner(frame_bgr: np.ndarray) -> dict:
        res = dict(
            mask_np=None,
            gdino_box=None,
            sam_score=None,
            best_centroid=None,
            detector_used="gdino",
            detection_was_skipped=False,
            dets_all=None,
            similarity=None,
        )

        if state["failed"] and state["model"] is None:
            res["detection_was_skipped"] = True
            return res

        dets = _run_gdino(frame_bgr, prompt)
        res["dets_all"] = [
            (np.asarray(d[0], dtype=np.float32).tolist(), float(d[1]), str(d[2]))
            for d in dets
        ]

        if not dets:
            return res

        best = max(dets, key=lambda d: d[1])
        box_np = best[0]
        res["gdino_box"] = box_np.tolist()
        res["sam_score"] = float(best[1])
        res["best_centroid"] = _box_center(box_np)
        return res

    return runner


# ---------------------------------------------------------------------------
# Backend 3: GDINO + SAM2
# ---------------------------------------------------------------------------

def make_gdino_sam2_runner(ref_image_path: str, prompt: str = "box"):
    sam2_path = os.path.join(_THIRD_PARTY, "sam2")
    if os.path.isdir(sam2_path) and sam2_path not in sys.path:
        sys.path.insert(0, sam2_path)

    gdino_path = os.path.join(_THIRD_PARTY, "GroundingDINO")
    if os.path.isdir(gdino_path) and gdino_path not in sys.path:
        sys.path.insert(0, gdino_path)

    state = dict(
        gdino_processor=None,
        gdino_model=None,
        gdino_failed=False,
        sam2_pred=None,
        sam2_failed=False,
    )

    gdino_hf_model_id = os.environ.get(
        "GDINO_HF_MODEL_ID", "IDEA-Research/grounding-dino-tiny")

    def _find_sam2_checkpoint() -> str | None:
        ckpt_dir = os.path.join(sam2_path, "checkpoints")
        preferred = os.path.join(ckpt_dir, "sam2.1_hiera_large.pt")
        if os.path.exists(preferred):
            return preferred
        if os.path.isdir(ckpt_dir):
            for name in sorted(os.listdir(ckpt_dir)):
                if name.endswith(".pt"):
                    return os.path.join(ckpt_dir, name)
        return None

    def _load_gdino():
        if state["gdino_failed"] or state["gdino_model"] is not None:
            return state["gdino_processor"], state["gdino_model"]
        try:
            from transformers import (
                AutoProcessor, AutoModelForZeroShotObjectDetection)
            log.info("Loading GroundingDINO (HF): %s on %s",
                     gdino_hf_model_id, _device())
            state["gdino_processor"] = AutoProcessor.from_pretrained(
                gdino_hf_model_id)
            state["gdino_model"] = (
                AutoModelForZeroShotObjectDetection
                .from_pretrained(gdino_hf_model_id)
                .to(_device()).eval()
            )
            log.info("GroundingDINO loaded: %s", gdino_hf_model_id)
        except Exception as exc:
            log.warning("GroundingDINO load failed (won't retry): %s", exc)
            state["gdino_failed"] = True
        return state["gdino_processor"], state["gdino_model"]

    def _load_sam2():
        if state["sam2_failed"] or state["sam2_pred"] is not None:
            return state["sam2_pred"]
        ckpt = _find_sam2_checkpoint()
        if ckpt is None:
            log.warning("SAM2: no checkpoint found under %s; "
                        "falling back to GDINO-only.", sam2_path)
            state["sam2_failed"] = True
            return None
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            log.info("Loading SAM2 from %s on %s", ckpt, _device())
            model = build_sam2(sam2_cfg, ckpt, device=_device())
            state["sam2_pred"] = SAM2ImagePredictor(model)
            log.info("SAM2 image predictor ready")
        except Exception as exc:
            log.warning("SAM2 load failed (%s); falling back to GDINO-only.", exc)
            state["sam2_failed"] = True
        return state["sam2_pred"]

    def _run_gdino(frame_bgr: np.ndarray, text_prompt: str):
        processor, model = _load_gdino()
        if model is None:
            return []

        caption_raw = text_prompt.strip()
        if caption_raw.lower() in {"box", "box."}:
            text_labels = [["box", "cardboard box", "package", "carton", "parcel"]]
        else:
            parts = [p.strip() for p in caption_raw.replace(".", "\n").splitlines()
                     if p.strip()]
            text_labels = [parts] if parts else [[caption_raw]]

        try:
            import torch as _t
            from PIL import Image

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            h, w = frame_bgr.shape[:2]

            inputs = processor(images=pil, text=text_labels,
                               return_tensors="pt").to(_device())
            with _t.no_grad():
                outputs = model(**inputs)

            def _post(thresh: float, text_thresh: float):
                return processor.post_process_grounded_object_detection(
                    outputs,
                    input_ids=inputs["input_ids"],
                    threshold=thresh,
                    text_threshold=text_thresh,
                    target_sizes=[(h, w)])[0]

            result = _post(0.25, 0.25)
            if len(result["boxes"]) == 0:
                log.info("GDINO: empty at 0.25/0.25, retrying at 0.18/0.18")
                result = _post(0.18, 0.18)

            boxes  = result["boxes"].cpu().numpy()
            scores = result["scores"].cpu().numpy()
            labels = (result.get("text_labels") or result.get("labels")
                      or [""] * len(boxes))
            labels = [str(lb) for lb in labels]

            if len(boxes) == 0:
                log.warning("GDINO: no detections for %s", text_labels[0])
                return []

            dets = [(boxes[i].astype(np.float32), float(scores[i]),
                     labels[i] if i < len(labels) else "")
                    for i in range(len(boxes))]
            log.info("GDINO: %d detection(s) for %s", len(dets), text_labels[0])
            return dets

        except Exception as exc:
            log.error("GroundingDINO inference failed: %s", exc)
            return []

    def runner(frame_bgr: np.ndarray) -> dict:
        res = dict(
            mask_np=None,
            gdino_box=None,
            sam_score=None,
            best_centroid=None,
            detector_used="gdino+sam2",
            detection_was_skipped=False,
            dets_all=None,
            similarity=None,
        )

        dets = _run_gdino(frame_bgr, prompt)
        res["dets_all"] = [
            (np.asarray(d[0], dtype=np.float32).tolist(), float(d[1]), str(d[2]))
            for d in dets
        ]

        if not dets:
            return res

        best = max(dets, key=lambda d: d[1])
        box_np = best[0].astype(np.float32)
        res["gdino_box"] = box_np.tolist()
        res["sam_score"] = float(best[1])
        res["best_centroid"] = _box_center(box_np)

        pred = _load_sam2()
        if pred is None:
            return res

        try:
            import torch as _t
            h, w = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pred.set_image(rgb)

            cx = float((box_np[0] + box_np[2]) * 0.5)
            cy = float((box_np[1] + box_np[3]) * 0.5)
            cx = float(min(max(cx, 0), w - 1))
            cy = float(min(max(cy, 0), h - 1))
            center_pt = np.array([[cx, cy]], dtype=np.float32)

            masks, scores_sam, _ = pred.predict(
                box=box_np,
                point_coords=center_pt,
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=False,
            )

            if masks is not None and len(masks) > 0:
                mask_out = (masks[0] > 0).astype(np.uint8) * 255
                res["mask_np"]   = mask_out
                res["sam_score"] = float(scores_sam[0])
                c = _mask_centroid(mask_out)
                if c is not None:
                    res["best_centroid"] = c
                log.info("SAM2: mask obtained (score=%.3f, area=%d px)",
                         float(scores_sam[0]), int(np.count_nonzero(mask_out)))
            else:
                log.warning("SAM2: no mask returned; keeping GDINO box centre")

        except Exception as exc:
            log.error("SAM2 inference failed: %s; keeping GDINO box centre", exc)

        return res

    return runner


# ---------------------------------------------------------------------------
# Factory dispatcher
# ---------------------------------------------------------------------------

def make_runner(backend: str, ref_image_path: str, prompt: str = "box"):
    """Dispatch to the appropriate backend factory."""
    if backend == "sam3":
        return make_sam3_runner(ref_image_path, prompt)
    if backend == "gdino":
        return make_gdino_runner(ref_image_path, prompt)
    if backend == "gdino+sam2":
        return make_gdino_sam2_runner(ref_image_path, prompt)
    raise ValueError(
        f"Unknown backend {backend!r}. Choose from {AVAILABLE_BACKENDS}.")
