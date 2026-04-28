#!/usr/bin/env bash
# run_demo.sh — SemVS v2 demo launcher
#
# Usage:
#   ./run_demo.sh                        # live, dry-run, no cup (current setup)
#   ./run_demo.sh --with-cup             # live, dry-run, suction cup attached
#   ./run_demo.sh --robot                # live, robot enabled, no cup
#   ./run_demo.sh --robot --with-cup     # live, robot enabled, with cup
#   ./run_demo.sh --video CLIP.mp4       # offline on a recorded video
#   ./run_demo.sh --sixdof               # enable 6-DOF Signal A (needs depth)
#   ./run_demo.sh --no-window            # headless / SSH session

set -euo pipefail

# ---------------------------------------------------------------------------
# Hardware parameters — edit these to match your physical setup
# ---------------------------------------------------------------------------

# Path to hand-eye calibration file (T_ee_cam, 4x4, .npy).
# Run hand-eye calibration once and save here. Leave blank until then.
HAND_EYE_PATH="config/hand_eye.npy"

# Conda environment that has cv2, torch, pyzed, etc.
CONDA_ENV="semvs"

# Text prompt for SAM3 object detection
PROMPT="box"

# Reference image of the target object (used by SAM3 / DINOv2)
REF_IMAGE="masked_objects/protein_bar.png"

# Output root — a timestamped subfolder is created inside automatically
OUTPUT_ROOT="runs"

# Log directory
LOG_DIR="logs"

# ---------------------------------------------------------------------------
# Suction cup geometry (Improvement 2 — EE occlusion masking)
# ---------------------------------------------------------------------------
# Measure these from your physical cup when attached:
#   EE_TIP_OFFSET  = distance from EE flange face to suction cup face (mm)
#   EE_BODY_RADIUS = outer radius of the cup / gripper body at the tip (mm)

EE_TIP_OFFSET_WITH_CUP=120    # mm — adjust after measuring
EE_BODY_RADIUS_WITH_CUP=40    # mm — adjust after measuring (cup diameter / 2)

# With no cup the camera body has negligible footprint; masking is disabled.
EE_TIP_OFFSET_NO_CUP=0
EE_BODY_RADIUS_NO_CUP=0

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------

WITH_CUP=0
ROBOT_ENABLED=0
SIXDOF=0
NO_WINDOW=0
OFFLINE_VIDEO=""
DRY_RUN_FLAG="--dry-run"
EXTRA_ARGS=""

for arg in "$@"; do
    case "$arg" in
        --with-cup)   WITH_CUP=1 ;;
        --robot)      ROBOT_ENABLED=1 ;;
        --sixdof)     SIXDOF=1 ;;
        --no-window)  NO_WINDOW=1 ;;
        --video)      shift; OFFLINE_VIDEO="$1" ;;   # next arg is path
        *)
            # Pass unknown args straight through (e.g. --max-frames 100)
            EXTRA_ARGS="$EXTRA_ARGS $arg"
            ;;
    esac
done

# If --video PATH was passed as two separate tokens, handle that form too
for i in $(seq 1 $#); do
    if [ "${!i}" = "--video" ]; then
        next=$((i+1))
        OFFLINE_VIDEO="${!next}"
    fi
done

# ---------------------------------------------------------------------------
# Build argument list
# ---------------------------------------------------------------------------

ARGS=""

# --- Input source ---
if [ -n "$OFFLINE_VIDEO" ]; then
    ARGS="$ARGS --input-video $OFFLINE_VIDEO"
    DRY_RUN_FLAG=""   # offline mode doesn't need dry-run
else
    # Live mode: enable dry-run unless --robot was passed
    if [ "$ROBOT_ENABLED" -eq 0 ]; then
        ARGS="$ARGS --dry-run"
    fi
fi

# --- Reference image ---
if [ -f "$REF_IMAGE" ]; then
    ARGS="$ARGS --ref-image $REF_IMAGE"
else
    echo "WARNING: ref image not found at $REF_IMAGE — SAM3 will run without reference."
fi

# --- Hand-eye calibration ---
if [ -f "$HAND_EYE_PATH" ]; then
    ARGS="$ARGS --hand-eye $HAND_EYE_PATH"
else
    echo "INFO: hand-eye file not found at $HAND_EYE_PATH — Signal A disabled (identity fallback)."
fi

# --- EE occlusion masking geometry ---
if [ "$WITH_CUP" -eq 1 ]; then
    ARGS="$ARGS --ee-tip-offset $EE_TIP_OFFSET_WITH_CUP"
    ARGS="$ARGS --ee-body-radius $EE_BODY_RADIUS_WITH_CUP"
    echo "INFO: suction cup geometry active (offset=${EE_TIP_OFFSET_WITH_CUP}mm, radius=${EE_BODY_RADIUS_WITH_CUP}mm)"
else
    ARGS="$ARGS --ee-tip-offset $EE_TIP_OFFSET_NO_CUP"
    ARGS="$ARGS --ee-body-radius $EE_BODY_RADIUS_NO_CUP"
    echo "INFO: no suction cup — EE occlusion masking disabled"
fi

# --- Optional flags ---
[ "$SIXDOF" -eq 1 ]    && ARGS="$ARGS --sixdof"
[ "$NO_WINDOW" -eq 1 ] && ARGS="$ARGS --no-window"

# --- Output dir (timestamped) ---
TS=$(date +%Y%m%d_%H%M%S)
CUP_TAG=$([ "$WITH_CUP" -eq 1 ] && echo "cup" || echo "nocup")
SOURCE_TAG=$([ -n "$OFFLINE_VIDEO" ] && echo "offline" || echo "live")
OUT_DIR="${OUTPUT_ROOT}/v2_${SOURCE_TAG}_${CUP_TAG}_${TS}"
ARGS="$ARGS --output-dir $OUT_DIR"

# --- Prompt + log dir ---
ARGS="$ARGS --prompt $PROMPT --log-dir $LOG_DIR"

# --- Always save debug artifacts ---
ARGS="$ARGS --debug"

# Pass through any extra unknown args
ARGS="$ARGS $EXTRA_ARGS"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

echo ""
echo "======================================================="
echo " SemVS v2 demo"
echo " cup=$WITH_CUP  robot=$ROBOT_ENABLED  sixdof=$SIXDOF"
echo " output -> $OUT_DIR"
echo "======================================================="
echo ""

conda run -n "$CONDA_ENV" python -m foundation_model.servo_lastmile_v2 $ARGS
