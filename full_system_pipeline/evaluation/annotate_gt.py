"""
annotate_gt.py — Interactive GT grasp-point annotation for a servoing run.

Usage:
    python annotate_gt.py --run-dir runs/v2_servo10/ [--frame N] [--ref-image PATH]

Opens a matplotlib window showing frame N from the run video (raw.mp4, or overlay.mp4
as fallback). Click the image to place the GT grasp point (red cross). Press ENTER or
'q' to confirm and save.

Output: <run-dir>/gt_centroid.json
    {"gt_centroid": [x, y], "frame_idx": N, "annotated_at": "<ISO timestamp>"}
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def _find_video(run_dir):
    for name in ("raw.mp4", "overlay.mp4"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            return path
    return None


def _load_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx >= total:
        cap.release()
        raise ValueError(f"Frame {frame_idx} out of range (video has {total} frames)")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _first_near_frame(run_dir):
    jsonl_path = os.path.join(run_dir, "frames.jsonl")
    if not os.path.exists(jsonl_path):
        return 0
    prev_state = None
    with open(jsonl_path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            state = frame.get("state")
            if prev_state == "FAR" and state == "NEAR":
                return i
            prev_state = state
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Run directory")
    parser.add_argument("--frame", type=int, default=None,
                        help="Frame index to annotate (default: first FAR→NEAR transition)")
    parser.add_argument("--ref-image", default=None,
                        help="Optional reference image to show side-by-side")
    args = parser.parse_args()

    video_path = _find_video(args.run_dir)
    if video_path is None:
        print(f"Error: no raw.mp4 or overlay.mp4 found in {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    frame_idx = args.frame if args.frame is not None else _first_near_frame(args.run_dir)
    frame_rgb = _load_frame(video_path, frame_idx)

    ref_rgb = None
    if args.ref_image is not None:
        ref_bgr = cv2.imread(args.ref_image)
        if ref_bgr is None:
            print(f"Warning: could not load reference image {args.ref_image}", file=sys.stderr)
        else:
            ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)

    ncols = 2 if ref_rgb is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    ax_main = axes[0]
    ax_main.imshow(frame_rgb)
    ax_main.set_title(f"Frame {frame_idx} — click GT grasp point, then press ENTER or 'q'")
    ax_main.axis("off")

    if ref_rgb is not None:
        axes[1].imshow(ref_rgb)
        axes[1].set_title("Reference image")
        axes[1].axis("off")

    plt.tight_layout()

    clicked_point = [None]
    cross_artists = []

    def _redraw_cross(x, y):
        for artist in cross_artists:
            artist.remove()
        cross_artists.clear()
        marker, = ax_main.plot(x, y, marker="+", color="red", markersize=20,
                               markeredgewidth=2.5, linestyle="none")
        label = ax_main.text(x + 10, y - 10, f"({x:.1f}, {y:.1f})",
                             color="red", fontsize=9)
        cross_artists.extend([marker, label])
        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes is not ax_main:
            return
        clicked_point[0] = (event.xdata, event.ydata)
        _redraw_cross(event.xdata, event.ydata)

    def _on_key(event):
        if event.key in ("enter", "q"):
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    plt.show()

    if clicked_point[0] is None:
        print("No point selected. Exiting without saving.")
        sys.exit(0)

    x, y = clicked_point[0]
    output = {
        "gt_centroid": [round(x, 2), round(y, 2)],
        "frame_idx": frame_idx,
        "annotated_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = os.path.join(args.run_dir, "gt_centroid.json")
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)

    print(f"GT centroid saved: {output['gt_centroid']} → {out_path}")


if __name__ == "__main__":
    main()
