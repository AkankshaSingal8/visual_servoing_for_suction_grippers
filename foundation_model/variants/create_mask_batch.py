#!/usr/bin/env python3
"""
Create a segmentation mask for an input image and save with transparent background.

Uses SAM2 with automatic foreground detection (no text prompt needed).
Handles objects with internal colour boundaries (e.g. white lines on coloured
boxes) by placing a grid of positive point prompts across the entire object.

Usage:
    python create_mask.py path/to/image.jpg [--output dir] [--no-display]
"""
import sys
import os
import argparse

import cv2
import numpy as np

THIRD_PARTY_ROOT = os.path.join(os.path.dirname(__file__), "third-party")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT   = os.path.join(THIRD_PARTY_ROOT, "sam2/checkpoints/sam2.1_hiera_large.pt")

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    torch  = None
    DEVICE = "cpu"

_sam2_pred = None

def _get_sam2():
    global _sam2_pred
    if _sam2_pred is None:
        sys.path.insert(0, os.path.join(THIRD_PARTY_ROOT, "sam2"))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model      = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=DEVICE)
        _sam2_pred = SAM2ImagePredictor(model)
    return _sam2_pred


# ═════════════════════════════════════════════════════════════════════
#  Foreground detection (no text prompt — pure image analysis)
# ═════════════════════════════════════════════════════════════════════

def detect_foreground(image_bgr: np.ndarray):
    """
    Separate the main object from the background using adaptive thresholding.

    Returns
    -------
    fg_mask    : uint8 binary mask (0/255) of the largest foreground blob
    bbox       : (x, y, w, h) bounding box of the foreground
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Otsu threshold — works well when object is brighter or darker than bg
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Pick the variant (or its inverse) where foreground area is smaller
    # (the object should be smaller than the background)
    fg_count  = np.count_nonzero(otsu)
    bg_count  = otsu.size - fg_count
    if fg_count > bg_count:
        otsu = cv2.bitwise_not(otsu)

    # Morphological clean-up
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kern, iterations=2)
    otsu = cv2.morphologyEx(otsu, cv2.MORPH_OPEN,  kern, iterations=1)

    # Largest connected component = the main object
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(otsu)
    if n <= 1:
        # Fallback: use the centre 60 % of the image
        h, w = image_bgr.shape[:2]
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1 = int(w * 0.2), int(h * 0.2)
        x2, y2 = int(w * 0.8), int(h * 0.8)
        fg_mask[y1:y2, x1:x2] = 255
        return fg_mask, (x1, y1, x2 - x1, y2 - y1)

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    fg_mask = (labels == largest_label).astype(np.uint8) * 255

    x, y, bw, bh = cv2.boundingRect(fg_mask)
    return fg_mask, (x, y, bw, bh)


# ═════════════════════════════════════════════════════════════════════
#  Multi-point SAM2 segmentation (handles internal colour splits)
# ═════════════════════════════════════════════════════════════════════

def segment_whole_object(image_bgr: np.ndarray, fg_mask: np.ndarray,
                         bbox: tuple):
    """
    Run SAM2 with a bounding-box prompt + a grid of positive points
    sampled from the foreground mask.  This ensures both the coloured
    and white regions of an object like the tissue box are captured.

    Returns
    -------
    mask_initial : uint8 0/255 — raw SAM2 output (before morph closing)
    mask_final   : uint8 0/255 — after morph closing to bridge gaps
    logits       : (1, 256, 256) low-res SAM2 logits
    points_vis   : list of (x, y) points used as prompts (for visualisation)
    """
    x, y, bw, bh = bbox
    pad = 10
    h_img, w_img = image_bgr.shape[:2]
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + bw + pad)
    y2 = min(h_img, y + bh + pad)
    box_np = np.array([x1, y1, x2, y2], dtype=np.float32)

    # Sample a 3x5 grid of positive points inside the foreground mask.
    # The tall grid (5 rows) ensures coverage above and below the white line.
    points = []
    for fy in [0.15, 0.30, 0.50, 0.70, 0.85]:
        for fx in [0.25, 0.50, 0.75]:
            px = int(x + fx * bw)
            py = int(y + fy * bh)
            py = min(py, h_img - 1)
            px = min(px, w_img - 1)
            if fg_mask[py, px] > 0:
                points.append([px, py])

    # Always include at least the centre
    cx, cy = int(x + bw / 2), int(y + bh / 2)
    if not points:
        points.append([cx, cy])

    point_coords = np.array(points, dtype=np.float32)
    point_labels = np.ones(len(points), dtype=np.int32)

    print(f"SAM2: box=[{x1},{y1},{x2},{y2}], {len(points)} positive points")

    pred = _get_sam2()
    rgb  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pred.set_image(rgb)

    masks, scores, logits = pred.predict(
        box=box_np,
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
        return_logits=True,
    )

    best_idx = int(np.argmax(scores))
    mask_initial = (masks[best_idx] > 0).astype(np.uint8) * 255
    ref_logits   = logits[best_idx: best_idx + 1]
    print(f"SAM2: mask score={scores[best_idx]:.3f}")

    # ── Iterative refinement ─────────────────────────────────────────
    # Check coverage: if the mask misses a big chunk of the bounding box
    # foreground, re-run with extra points in the uncovered region.
    covered   = np.count_nonzero((fg_mask > 0) & (mask_initial > 0))
    total_fg  = max(1, np.count_nonzero(fg_mask > 0))
    coverage  = covered / total_fg

    if coverage < 0.70:
        print(f"SAM2: coverage={coverage:.0%} — re-running with extra points "
              f"in uncovered region")
        uncovered = (fg_mask > 0) & (mask_initial == 0)
        unc_coords = np.argwhere(uncovered)
        if len(unc_coords) > 0:
            n_extra = min(5, len(unc_coords))
            idx     = np.linspace(0, len(unc_coords) - 1, n_extra, dtype=int)
            for r, c in unc_coords[idx]:
                points.append([int(c), int(r)])

            point_coords = np.array(points, dtype=np.float32)
            point_labels = np.ones(len(points), dtype=np.int32)

            pred.set_image(rgb)
            masks2, scores2, logits2 = pred.predict(
                box=box_np,
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=ref_logits,
                multimask_output=False,
                return_logits=True,
            )
            if masks2 is not None and len(masks2) > 0:
                best2 = int(np.argmax(scores2))
                mask_initial = (masks2[best2] > 0).astype(np.uint8) * 255
                ref_logits   = logits2[best2: best2 + 1]
                print(f"SAM2 (refined): score={scores2[best2]:.3f}")

    # ── Morphological close to bridge colour-boundary gaps ───────────
    k_close    = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask_final = cv2.morphologyEx(mask_initial, cv2.MORPH_CLOSE,
                                  k_close, iterations=2)
    # Fill interior holes
    contours, _ = cv2.findContours(mask_final, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_final, contours, -1, 255, cv2.FILLED)

    return mask_initial, mask_final, ref_logits, points


# ═════════════════════════════════════════════════════════════════════
#  Transparent-background image
# ═════════════════════════════════════════════════════════════════════

def make_transparent(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create a BGRA image where pixels outside the mask are transparent."""
    if mask.shape[:2] != image_bgr.shape[:2]:
        mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    b, g, r = cv2.split(image_bgr)
    alpha   = mask.copy()
    return cv2.merge([b, g, r, alpha])


# ═════════════════════════════════════════════════════════════════════
#  Visualisation
# ═════════════════════════════════════════════════════════════════════

def visualise(image_bgr, fg_mask, points, mask_initial, mask_final,
              transparent, output_dir, display=True):
    """
    Create a multi-panel figure showing every step of the pipeline.

    Panels:
      1. Input image
      2. Foreground detection + sample points
      3. SAM2 initial mask
      4. SAM2 mask after morph close (final)
      5. Transparent-background result
    """
    import matplotlib
    if not display:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = image_bgr.shape[:2]
    rgb   = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # Panel 1: Input image
    axes[0].imshow(rgb)
    axes[0].set_title("1. Input image", fontsize=12)
    axes[0].axis("off")

    # Panel 2: Foreground + points
    fg_vis = rgb.copy()
    fg_overlay = np.zeros_like(fg_vis)
    fg_overlay[:, :, 2] = fg_mask  # red channel
    fg_vis = cv2.addWeighted(fg_vis, 0.6, fg_overlay, 0.4, 0)
    for px, py in points:
        cv2.circle(fg_vis, (px, py), 6, (0, 255, 0), -1)
        cv2.circle(fg_vis, (px, py), 7, (255, 255, 255), 2)
    axes[1].imshow(fg_vis)
    axes[1].set_title("2. Foreground + SAM2 points", fontsize=12)
    axes[1].axis("off")

    # Panel 3: SAM2 initial mask
    init_vis = rgb.copy()
    init_overlay = np.zeros_like(init_vis)
    init_overlay[:, :, 1] = mask_initial
    init_vis = cv2.addWeighted(init_vis, 0.5, init_overlay, 0.5, 0)
    axes[2].imshow(init_vis)
    axes[2].set_title("3. SAM2 raw mask", fontsize=12)
    axes[2].axis("off")

    # Panel 4: Final mask after morph close
    final_vis = rgb.copy()
    final_overlay = np.zeros_like(final_vis)
    final_overlay[:, :, 1] = mask_final
    final_vis = cv2.addWeighted(final_vis, 0.5, final_overlay, 0.5, 0)
    axes[3].imshow(final_vis)
    axes[3].set_title("4. Final mask (morph closed)", fontsize=12)
    axes[3].axis("off")

    # Panel 5: Transparent background (checkerboard behind)
    checker = _checkerboard(h, w, sq=16)
    trans_rgba = cv2.cvtColor(transparent, cv2.COLOR_BGRA2RGBA)
    alpha_f = trans_rgba[:, :, 3:4].astype(np.float32) / 255.0
    blended = (checker * (1 - alpha_f) +
               trans_rgba[:, :, :3].astype(np.float32) * alpha_f)
    axes[4].imshow(blended.astype(np.uint8))
    axes[4].set_title("5. Transparent background", fontsize=12)
    axes[4].axis("off")

    plt.tight_layout()
    vis_path = os.path.join(output_dir, "mask_pipeline_steps.png")
    fig.savefig(vis_path, dpi=150, bbox_inches="tight")
    print(f"Visualisation saved: {vis_path}")

    if display:
        plt.show()
    else:
        plt.close(fig)


def _checkerboard(h: int, w: int, sq: int = 16) -> np.ndarray:
    """Grey/white checkerboard for showing transparency."""
    board = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(0, h, sq):
        for x in range(0, w, sq):
            colour = 220.0 if ((x // sq) + (y // sq)) % 2 == 0 else 180.0
            board[y: y + sq, x: x + sq] = colour
    return board

def process_single_image(image_path, output_dir, display=True):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"Skipping (cannot load): {image_path}")
        return

    h, w = image_bgr.shape[:2]
    print(f"\nProcessing: {image_path} ({w}x{h})")

    basename = os.path.splitext(os.path.basename(image_path))[0]

    # Step 1: Foreground detection
    fg_mask, bbox = detect_foreground(image_bgr)

    # Step 2: SAM2 segmentation
    mask_initial, mask_final, logits, points = segment_whole_object(
        image_bgr, fg_mask, bbox
    )

    # Step 3: Transparent image
    transparent = make_transparent(image_bgr, mask_final)

    # Save outputs
    # mask_path  = os.path.join(output_dir, f"{basename}_mask.png")
    trans_path = os.path.join(output_dir, f"{basename}.png")

    # cv2.imwrite(mask_path, mask_final)
    cv2.imwrite(trans_path, transparent)

    # print(f"Saved: {mask_path}")
    print(f"Saved: {trans_path}")

    # Step 4: Visualization (optional per image)
    # visualise(
    #     image_bgr,
    #     fg_mask,
    #     points,
    #     mask_initial,
    #     mask_final,
    #     transparent,
    #     output_dir=os.path.join(output_dir, f"{basename}_viz"),
    #     display=display,
    # )

# ═════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Batch segmentation on folder of images"
    )
    parser.add_argument("input_folder", help="Path to folder containing images")
    parser.add_argument("--output", "-o", default="outputs",
                        help="Output directory")
    parser.add_argument("--no-display", action="store_true")

    args = parser.parse_args()

    input_folder = args.input_folder
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)

    # Supported extensions
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp")

    image_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith(valid_exts)
    ]

    if not image_files:
        print("No images found in folder.")
        sys.exit(1)

    print(f"Found {len(image_files)} images")

    for img_path in image_files:
        process_single_image(
            img_path,
            output_dir,
            display=not args.no_display
        )

    print("\nBatch processing complete.")


if __name__ == "__main__":
    main()
