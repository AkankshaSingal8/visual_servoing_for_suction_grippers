#!/usr/bin/env python3
"""
Offline smoke test — no robot, no camera required.

Usage:
    python full_system_pipeline/smoke_test.py

Pass criterion: all checks print PASS and exit code is 0.
"""

import os, sys, json, subprocess, shutil, traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

REF_IMAGE  = "assets/objects/protein_bar.jpeg"
INPUT_VIDEO = "runs/v2_proteinbar_servo12/raw.mp4"
LIVE_RUN   = "runs/v2_proteinbar_servo12/"
OUT_BASE   = "runs/_smoketest"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

results = []

def check(name, ok, detail=""):
    tag = PASS if ok else FAIL
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    results.append((name, ok))

def run(cmd, timeout=300):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr

shutil.rmtree(OUT_BASE, ignore_errors=True)

# ---------------------------------------------------------------------------
print("\n=== 1/7  Imports ===")
modules = [
    "full_system_pipeline.detection_backends",
    "full_system_pipeline.pipeline.robot_controller_ext",
    "full_system_pipeline.pipeline.servo_lastmile_v2_ext",
    "full_system_pipeline.evaluation.analyze_run",
    "full_system_pipeline.evaluation.annotate_gt",
    "full_system_pipeline.evaluation.run_experiment",
    "full_system_pipeline.evaluation.aggregate_results",
    "full_system_pipeline.evaluation.failure_analysis",
]
for m in modules:
    try:
        __import__(m)
        check(m, True)
    except Exception as e:
        check(m, False, str(e))

# ---------------------------------------------------------------------------
print("\n=== 2/7  SAM3 offline ===")
rc, out = run([sys.executable,
    "full_system_pipeline/pipeline/servo_lastmile_v2_ext.py",
    "--detector", "sam3",
    "--ref-image", REF_IMAGE,
    "--input-video", INPUT_VIDEO,
    "--output-dir", f"{OUT_BASE}/sam3/",
])
jsonl = f"{OUT_BASE}/sam3/frames.jsonl"
frames_sam3 = 0
if os.path.exists(jsonl):
    with open(jsonl) as f:
        frames_sam3 = sum(1 for _ in f)
check("exit code 0",     rc == 0, f"rc={rc}")
check("frames.jsonl written", os.path.exists(jsonl))
check("103 frames logged",    frames_sam3 == 103, f"got {frames_sam3}")

# ---------------------------------------------------------------------------
print("\n=== 3/7  GDINO offline ===")
rc, out = run([sys.executable,
    "full_system_pipeline/pipeline/servo_lastmile_v2_ext.py",
    "--detector", "gdino",
    "--ref-image", REF_IMAGE,
    "--input-video", INPUT_VIDEO,
    "--output-dir", f"{OUT_BASE}/gdino/",
])
jsonl_gdino = f"{OUT_BASE}/gdino/frames.jsonl"
frames_gdino = 0
non_null_gdino = 0
if os.path.exists(jsonl_gdino):
    with open(jsonl_gdino) as f:
        rows = [json.loads(l) for l in f]
    frames_gdino = len(rows)
    non_null_gdino = sum(1 for r in rows if r.get("best_centroid") is not None)
check("exit code 0",      rc == 0, f"rc={rc}")
check("103 frames logged", frames_gdino == 103, f"got {frames_gdino}")
check("centroids non-null", non_null_gdino == 103, f"{non_null_gdino}/103 non-null")
check("detector_used=gdino",
      os.path.exists(jsonl_gdino) and
      json.loads(open(jsonl_gdino).readline()).get("detector_used") == "gdino")

# ---------------------------------------------------------------------------
print("\n=== 4/7  GDINO+SAM2 offline ===")
rc, out = run([sys.executable,
    "full_system_pipeline/pipeline/servo_lastmile_v2_ext.py",
    "--detector", "gdino+sam2",
    "--ref-image", REF_IMAGE,
    "--input-video", INPUT_VIDEO,
    "--output-dir", f"{OUT_BASE}/gdino_sam2/",
], timeout=360)
jsonl_gs2 = f"{OUT_BASE}/gdino_sam2/frames.jsonl"
frames_gs2 = 0
non_null_gs2 = 0
if os.path.exists(jsonl_gs2):
    with open(jsonl_gs2) as f:
        rows = [json.loads(l) for l in f]
    frames_gs2 = len(rows)
    non_null_gs2 = sum(1 for r in rows if r.get("best_centroid") is not None)
check("exit code 0",      rc == 0, f"rc={rc}")
check("103 frames logged", frames_gs2 == 103, f"got {frames_gs2}")
check("centroids non-null", non_null_gs2 == 103, f"{non_null_gs2}/103 non-null")
check("detector_used=gdino+sam2",
      os.path.exists(jsonl_gs2) and
      json.loads(open(jsonl_gs2).readline()).get("detector_used") == "gdino+sam2")

# ---------------------------------------------------------------------------
print("\n=== 5/7  Detectors produce different centroids ===")
if os.path.exists(f"{OUT_BASE}/sam3/frames.jsonl") and os.path.exists(jsonl_gdino):
    with open(f"{OUT_BASE}/sam3/frames.jsonl") as f:
        c_sam3 = [json.loads(l).get("best_centroid") for l in f]
    with open(jsonl_gdino) as f:
        c_gdino = [json.loads(l).get("best_centroid") for l in f]
    with open(jsonl_gs2) as f:
        c_gs2 = [json.loads(l).get("best_centroid") for l in f]
    # At least some frames should differ between SAM3 and GDINO
    differ_sg = sum(1 for a, b in zip(c_sam3, c_gdino)
                    if a and b and (abs(a[0]-b[0]) > 0.5 or abs(a[1]-b[1]) > 0.5))
    differ_sg2 = sum(1 for a, b in zip(c_sam3, c_gs2)
                     if a and b and (abs(a[0]-b[0]) > 0.5 or abs(a[1]-b[1]) > 0.5))
    check("SAM3 vs GDINO differ on ≥1 frame",   differ_sg >= 1,  f"{differ_sg} frames differ")
    check("SAM3 vs GDINO+SAM2 differ on ≥1 frame", differ_sg2 >= 1, f"{differ_sg2} frames differ")
else:
    check("centroid comparison (skipped — prior step failed)", False)

# ---------------------------------------------------------------------------
print("\n=== 6/7  analyze_run on live run (NEAR+TERMINAL data) ===")
rc, out = run([sys.executable,
    "full_system_pipeline/evaluation/analyze_run.py",
    LIVE_RUN,
    "--out-dir", f"{OUT_BASE}/analysis/",
])
expected_plots = [
    "error_over_distance.png", "signal_centroids.png",
    "tilt_timeline.png", "phase_summary.png",
    "signal_dominance.png", "summary.json",
]
check("exit code 0", rc == 0, f"rc={rc}")
for p in expected_plots:
    check(f"  {p}", os.path.exists(f"{OUT_BASE}/analysis/{p}"))
if os.path.exists(f"{OUT_BASE}/analysis/summary.json"):
    s = json.load(open(f"{OUT_BASE}/analysis/summary.json"))
    check("summary has n_near_sam3 > 0", s.get("n_near_sam3", 0) > 0, str(s.get("n_near_sam3")))
    check("summary has n_terminal > 0",  s.get("n_terminal",  0) > 0, str(s.get("n_terminal")))

# ---------------------------------------------------------------------------
print("\n=== 7/7  Tilt controller logic (no robot) ===")
try:
    from full_system_pipeline.pipeline.robot_controller_ext import RobotControllerExt

    class FakeArm:
        def __init__(self): self.calls = []
        def set_tool_position(self, **kw):
            self.calls.append(kw); return 0, None

    arm = FakeArm()
    ctrl = RobotControllerExt.__new__(RobotControllerExt)
    ctrl._arm = arm
    ctrl.enabled = True
    ctrl._dry_run = False

    # Dead-zone: tilt < 1 deg — must not fire
    acted = ctrl.correct_tilt(roll_deg=0.5, pitch_deg=0.3, n_inliers=600)
    check("dead-zone skips command", not acted and len(arm.calls) == 0)

    # Enough tilt + inliers — must fire, clamped to ±3 deg
    acted = ctrl.correct_tilt(roll_deg=10.0, pitch_deg=-10.0, n_inliers=600)
    check("fires when tilt > 1° and inliers ≥ 500", acted and len(arm.calls) == 1)
    if arm.calls:
        dr = arm.calls[-1].get("roll", 0)
        dp = arm.calls[-1].get("pitch", 0)
        check("roll clamped to ±3°",  abs(dr) <= 3.0, f"got {dr:.2f}")
        check("pitch clamped to ±3°", abs(dp) <= 3.0, f"got {dp:.2f}")

    # Insufficient inliers — must skip
    arm.calls.clear()
    acted = ctrl.correct_tilt(roll_deg=10.0, pitch_deg=-10.0, n_inliers=200)
    check("skips when inliers < 500", not acted and len(arm.calls) == 0)

    # Disabled — must skip
    ctrl.enabled = False
    acted = ctrl.correct_tilt(roll_deg=10.0, pitch_deg=-10.0, n_inliers=600)
    check("skips when disabled", not acted)
    ctrl.enabled = True

    # Dry-run report
    rpt = ctrl.correct_tilt_dry(roll_deg=4.0, pitch_deg=-6.0, n_inliers=700)
    check("dry-run would_apply=True",  rpt.get("would_apply") == True)
    check("dry-run delta_roll ≤ 3°",   abs(rpt.get("delta_roll", 99)) <= 3.0)
    check("dry-run delta_pitch ≤ 3°",  abs(rpt.get("delta_pitch", 99)) <= 3.0)

except Exception as e:
    check("tilt controller test (exception)", False, traceback.format_exc())

# ---------------------------------------------------------------------------
print()
total  = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed
print(f"{'='*50}")
print(f"  Results: {passed}/{total} passed  ({failed} failed)")
print(f"{'='*50}")

if failed:
    print("\nFailed checks:")
    for name, ok in results:
        if not ok:
            print(f"  - {name}")

sys.exit(0 if failed == 0 else 1)
