"""
Simplified last-mile visual servoing pipeline.

State machine: FAR → NEAR → TERMINAL

FAR:   SAM3 detects box every frame; servo XY toward centroid.
NEAR:  SAM3 runs every frame. If detected → update centroid.
       If SAM3 fails (box exits frame) → hold last centroid, keep stepping Z.
       estimate_near_tilt() called every NEAR frame for orientation feedback.
TERMINAL: depth < TERM_DEPTH_MM (30 mm).

No CoTracker3, no SAM2, no DINOv2, no multi-signal fusion.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib

import cv2
import numpy as np

from foundation_model.servo_lastmile import (
    State,
    bbox_area_fraction,
    bbox_touches_border,
    make_default_sam3_runner,
    _setup_logger,
    _init_debug_dir,
    _finalize_debug_dir,
)
from foundation_model.servo_lastmile_v2 import (
    TERM_DEPTH_MM,
    LastMilePipelineV2,
    StateMachineV2,
    TiltEstimate,
    VideoRecorder,
    _serialize_result_v2,
    estimate_near_tilt,
)

log = _setup_logger("lastmile_simple")

# SAM3 score threshold for NEAR-phase centroid updates.
SAM3_NEAR_MIN_SCORE = 0.50
# Max box area fraction before SAM3 is considered unreliable (box fills frame).
NEAR_AREA_MAX_FRAC = 0.85


class _NoopSignal:
    """Drop-in stub for SignalB/C/D. Loads no models, allocates no memory."""
    def reset(self, *args, **kwargs): pass
    def step(self, *args, **kwargs): return (None, 0.0, None)


def _empty_result_simple() -> dict:
    return dict(
        state=None,
        frame_idx=None,
        best_centroid=None,
        z_mm=None,
        sam3_score=None,
        mask_np=None,
        gdino_box=None,
        phase=None,                  # "sam3" | "hold" | None
        sam3_detected=False,
        holding_last_centroid=False,
        tilt_deg=None,
        roll_deg=None,
        pitch_deg=None,
        near_plane_n_inliers=None,
        near_plane_normal=None,
        _tilt=None,                  # TiltEstimate object — stripped by _serialize_result_v2
    )


class LastMilePipelineSimple(LastMilePipelineV2):
    """
    SAM3-only pipeline with hold-last-centroid fallback and tilt correction.

    Subclasses LastMilePipelineV2 to inherit model loading and the
    depth-only TERMINAL trigger (StateMachineV2). Overrides step()
    entirely — does not call super().step().

    Signals B/C/D are replaced with _NoopSignal so _enter_lock() from the
    base class still runs (it initialises SignalB/C), but they are then
    immediately replaced and never touched again.
    """

    def __init__(
        self,
        sam3_runner,
        depth_provider=None,
        intrinsics=None,
        ee_pose_provider=None,
        hand_eye_path=None,
    ):
        super().__init__(
            sam3_runner=sam3_runner,
            ee_pose_provider=ee_pose_provider,
            depth_provider=depth_provider,
            hand_eye_path=hand_eye_path,
            intrinsics=intrinsics,
            use_sixdof=False,
        )
        _noop = _NoopSignal()
        self.signal_B = _noop
        self.signal_C = _noop
        self.signal_D = _noop

        self._last_centroid: tuple[float, float] | None = None
        self._holding_last_centroid: bool = False

    def reset(self):
        self.fsm.reset()
        self._last_centroid = None
        self._holding_last_centroid = False

    def step(self, frame_bgr: np.ndarray) -> dict:
        """
        Process one frame. Returns result dict compatible with _serialize_result_v2.

        Control flow is completely independent from the base step() — no super() call.
        """
        res = _empty_result_simple()
        self.fsm.frame_idx += 1
        res["frame_idx"] = self.fsm.frame_idx

        h, w = frame_bgr.shape[:2]
        depth_map = self.depth_provider(frame_bgr)

        # ── TERMINAL (checked first so it short-circuits immediately) ────────
        if self.fsm.state == State.TERM:
            res["state"] = State.TERM
            res["best_centroid"] = (w / 2.0, h / 2.0)
            return res

        # ── FAR ──────────────────────────────────────────────────────────────
        if self.fsm.state == State.FAR:
            sam3_out = self.sam3_runner(frame_bgr)
            box  = sam3_out.get("gdino_box")
            mask = sam3_out.get("mask_np")
            score = float(sam3_out.get("sam_score") or 0.0)

            centroid = None
            if box is not None and mask is not None and score >= SAM3_NEAR_MIN_SCORE:
                ys, xs = np.where(mask > 0)
                if xs.size >= 50:
                    centroid = (float(xs.mean()), float(ys.mean()))

            z_mm = self._depth_at_px(depth_map, centroid)

            res["state"] = "FAR"
            res["best_centroid"] = centroid
            res["sam3_score"] = score
            res["mask_np"] = mask
            res["gdino_box"] = box
            res["z_mm"] = z_mm

            # FAR → NEAR transition uses base-class evaluate_far + _enter_lock.
            if self.fsm.evaluate_far(sam3_out, z_mm, frame_bgr.shape):
                lock = self._enter_lock(frame_bgr, sam3_out, depth_map)
                if lock is not None:
                    self.fsm.lock_state = lock
                    self.fsm.state = State.NEAR
                    self._last_centroid = lock.centroid_px
                    self._holding_last_centroid = False
                    log.info("FAR→NEAR  lock=(%.0f,%.0f)  z=%.0fmm",
                             lock.centroid_px[0], lock.centroid_px[1], z_mm or 0)
            return res

        # ── NEAR ─────────────────────────────────────────────────────────────
        assert self.fsm.state == State.NEAR

        sam3_out = self.sam3_runner(frame_bgr)
        box  = sam3_out.get("gdino_box")
        mask = sam3_out.get("mask_np")
        score = float(sam3_out.get("sam_score") or 0.0)

        box_visible = (
            box is not None
            and mask is not None
            and score >= SAM3_NEAR_MIN_SCORE
            and not bbox_touches_border(np.asarray(box), w, h)
            and bbox_area_fraction(np.asarray(box), w, h) < NEAR_AREA_MAX_FRAC
        )

        if box_visible:
            ys, xs = np.where(mask > 0)
            if xs.size >= 50:
                self._last_centroid = (float(xs.mean()), float(ys.mean()))
                self._holding_last_centroid = False
        else:
            # Box exited the frame (close range) — keep stepping Z with last centroid.
            self._holding_last_centroid = True

        z_mm = self._depth_at_px(depth_map, self._last_centroid)

        # Surface-normal tilt estimation (best-effort; None when depth unavailable).
        tilt: TiltEstimate | None = None
        if depth_map is not None and self._last_centroid is not None:
            tilt = estimate_near_tilt(
                depth_map, self.K, self._last_centroid, rng=self._rng
            )
        if tilt is not None:
            res["tilt_deg"]            = tilt.tilt_deg
            res["roll_deg"]            = tilt.roll_deg
            res["pitch_deg"]           = tilt.pitch_deg
            res["near_plane_normal"]   = tilt.normal.tolist()
            res["near_plane_n_inliers"] = tilt.n_inliers
        res["_tilt"] = tilt

        # NEAR → TERMINAL check (depth-only, inherited from StateMachineV2).
        if self.fsm.evaluate_near_to_term(
            c_fused=self._last_centroid,
            z_mm=z_mm,
            image_center=(w / 2.0, h / 2.0),
        ):
            self.fsm.state = State.TERM
            log.info("NEAR→TERMINAL  z=%.0fmm", z_mm or 0)

        res["state"]                = "NEAR"
        res["frame_idx"]            = self.fsm.frame_idx
        res["best_centroid"]        = self._last_centroid
        res["z_mm"]                 = z_mm
        res["sam3_score"]           = score
        res["mask_np"]              = mask
        res["gdino_box"]            = box
        res["sam3_detected"]        = box_visible and not self._holding_last_centroid
        res["holding_last_centroid"] = self._holding_last_centroid
        res["phase"]                = "hold" if self._holding_last_centroid else "sam3"

        return res


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def _make_overlay_simple(frame: np.ndarray, res: dict) -> np.ndarray:
    """Debug overlay: RED cross in sam3 phase; YELLOW cross in hold phase."""
    out = frame.copy()
    c = res.get("best_centroid")
    if c is None:
        return out
    cx, cy = int(round(c[0])), int(round(c[1]))
    phase = res.get("phase") or "sam3"
    color = (0, 255, 255) if phase == "hold" else (0, 0, 255)   # YELLOW or RED
    label = "HOLD" if phase == "hold" else "SAM3"
    cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)
    tilt = res.get("tilt_deg")
    text = f"{label}  {res.get('state','')}  z={res.get('z_mm') or 0:.0f}mm"
    if tilt is not None:
        text += f"  tilt={tilt:.1f}deg"
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return out


# ---------------------------------------------------------------------------
# Offline runner (standalone, no robot)
# ---------------------------------------------------------------------------

def run_offline_simple(args: argparse.Namespace) -> None:
    """Replay a saved video, write frames.jsonl + overlay JPEGs."""
    pipeline = LastMilePipelineSimple(
        sam3_runner=make_default_sam3_runner(
            args.ref_image, prompt=getattr(args, "prompt", "box")
        ),
    )
    pipeline.reset()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "frames.jsonl"

    cap = cv2.VideoCapture(args.input_video)
    frame_idx = 0
    max_frames = getattr(args, "max_frames", None)

    with open(jsonl_path, "w") as fout:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames and frame_idx >= max_frames:
                break

            res = pipeline.step(frame)
            fout.write(json.dumps(_serialize_result_v2(res)) + "\n")

            if not getattr(args, "no_overlays", False):
                overlay = _make_overlay_simple(frame, res)
                cv2.imwrite(str(out_dir / f"frame_{frame_idx:05d}.jpg"), overlay)

            log.info("[%4d] %-8s phase=%-4s z=%s tilt=%s hold=%s",
                     frame_idx,
                     res.get("state") or "?",
                     res.get("phase") or "-",
                     f"{res['z_mm']:.0f}" if res.get("z_mm") else "—",
                     f"{res['tilt_deg']:.1f}deg" if res.get("tilt_deg") else "—",
                     res.get("holding_last_centroid", False))
            frame_idx += 1

    cap.release()
    log.info("Done. %d frames → %s", frame_idx, jsonl_path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Simplified SAM3 last-mile servo — offline")
    p.add_argument("--input-video",  required=True)
    p.add_argument("--ref-image",    required=True)
    p.add_argument("--output-dir",   required=True)
    p.add_argument("--no-overlays",  action="store_true")
    p.add_argument("--max-frames",   type=int, default=None)
    p.add_argument("--prompt",       default="box")
    args = p.parse_args(argv)
    run_offline_simple(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
