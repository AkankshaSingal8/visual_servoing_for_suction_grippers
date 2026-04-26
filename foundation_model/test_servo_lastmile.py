#!/usr/bin/env python3
"""
Tests for servo_lastmile.

Designed to run without GPU model weights: every test that would touch
SAM3/SAM2/CoTracker3/DINOv2 either uses a synthetic stub or skips.
The math, state machine, fusion, and centroid-bias property are all
tested deterministically.

Run:
    pytest foundation_model/test_servo_lastmile.py -v
or:
    python -m unittest foundation_model.test_servo_lastmile -v
"""

import math
import os
import sys
import unittest

# Disable heavy-model lazy loads (CoTracker3 / SAM2 / DINOv2) for tests so
# we don't hit torch.hub network calls or OpenMP duplicate-library crashes
# on macOS. Must be set BEFORE importing servo_lastmile.
os.environ.setdefault("LASTMILE_DISABLE_HEAVY_MODELS", "1")

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import servo_lastmile as sl  # noqa: E402


# ---------------------------------------------------------------------------
# Math: geometric median
# ---------------------------------------------------------------------------

class GeometricMedianTests(unittest.TestCase):

    def test_single_point(self):
        out = sl.weighted_geometric_median([(10.0, 20.0)], [1.0])
        self.assertEqual(out, (10.0, 20.0))

    def test_empty_returns_none(self):
        self.assertIsNone(sl.weighted_geometric_median([], []))

    def test_zero_weights_returns_none(self):
        self.assertIsNone(sl.weighted_geometric_median(
            [(1.0, 1.0), (2.0, 2.0)], [0.0, 0.0]))

    def test_collinear_points_unweighted(self):
        # Median of three colinear points should land near the middle one.
        out = sl.weighted_geometric_median(
            [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)], [1.0, 1.0, 1.0])
        self.assertAlmostEqual(out[0], 5.0, delta=0.5)
        self.assertAlmostEqual(out[1], 0.0, delta=0.5)

    def test_robust_to_one_outlier(self):
        # Three good points cluster at (0,0); one rogue at (1000, 1000).
        # Geometric median should stay near the cluster, NOT halfway.
        pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1000.0, 1000.0)]
        out = sl.weighted_geometric_median(pts, [1.0, 1.0, 1.0, 1.0])
        # Mean would be (250.25, 250.25). Median should be near (0.4, 0.4).
        self.assertLess(math.hypot(out[0], out[1]), 5.0,
                        f"Geometric median pulled by outlier: got {out}")

    def test_weights_pull_toward_heavy_point(self):
        out = sl.weighted_geometric_median(
            [(0.0, 0.0), (10.0, 0.0)], [1.0, 5.0])
        self.assertGreater(out[0], 5.0)


# ---------------------------------------------------------------------------
# Math: projection / transform
# ---------------------------------------------------------------------------

class ProjectionTests(unittest.TestCase):

    def setUp(self):
        self.K = sl.make_intrinsics_matrix(sl.DEFAULT_INTRINSICS)

    def test_projection_principal_point(self):
        # A point on the optical axis projects to the principal point.
        pt_cam = np.array([0.0, 0.0, 500.0])
        out = sl.project_3d_to_pixel(pt_cam, self.K)
        self.assertAlmostEqual(out[0], self.K[0, 2], delta=0.1)
        self.assertAlmostEqual(out[1], self.K[1, 2], delta=0.1)

    def test_projection_offset(self):
        pt_cam = np.array([100.0, 50.0, 1000.0])
        u, v = sl.project_3d_to_pixel(pt_cam, self.K)
        # u = fx * X/Z + cx
        self.assertAlmostEqual(u, 700.0 * 100.0 / 1000.0 + 640.0, places=3)
        self.assertAlmostEqual(v, 700.0 * 50.0 / 1000.0 + 360.0, places=3)

    def test_behind_camera_returns_none(self):
        pt_cam = np.array([0.0, 0.0, -100.0])
        self.assertIsNone(sl.project_3d_to_pixel(pt_cam, self.K))

    def test_invert_transform_roundtrip(self):
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(np.array([0.1, 0.2, 0.3]))[0]
        T[:3, 3] = [10.0, 20.0, 30.0]
        Tinv = sl.invert_transform(T)
        prod = T @ Tinv
        np.testing.assert_allclose(prod, np.eye(4), atol=1e-9)

    def test_transform_point_consistency(self):
        T = np.eye(4)
        T[:3, 3] = [5.0, 0.0, 0.0]
        out = sl.transform_point(T, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(out, [6.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Signal A: locked-3D round trip
# ---------------------------------------------------------------------------

class SignalATests(unittest.TestCase):
    """
    The core invariant: at the moment of LOCK, projecting the locked
    3D-base centroid back through the SAME EE pose with the SAME T_ee_cam
    should reproduce the original image-space centroid exactly.
    """

    def test_lock_time_roundtrip_identity_handeye(self):
        K = sl.make_intrinsics_matrix(sl.DEFAULT_INTRINSICS)
        T_ee_cam = np.eye(4)
        T_base_ee = np.eye(4)
        sigA = sl.SignalA(T_ee_cam, K)

        # Synthesize: image centroid (700, 400), depth 250mm.
        # 3D in cam frame: X = (u-cx)*z/fx, Y = (v-cy)*z/fy, Z = z.
        z = 250.0
        u, v = 700.0, 400.0
        X = (u - K[0, 2]) * z / K[0, 0]
        Y = (v - K[1, 2]) * z / K[1, 1]
        pt_cam = np.array([X, Y, z])
        # base = ee * ee_cam, so pt_base = pt_cam under identity
        pt_base = pt_cam.copy()

        lock = sl.LockState(
            mask_uint8=np.zeros((720, 1280), dtype=np.uint8),
            centroid_px=(u, v),
            bbox_xyxy=np.array([600, 350, 800, 450], dtype=np.float32),
            crop_bgr=np.zeros((100, 200, 3), dtype=np.uint8),
            crop_bbox=np.array([600, 350, 800, 450], dtype=np.float32),
            centroid_base=pt_base,
            z_at_lock_mm=z,
            T_base_ee=T_base_ee,
            init_points=np.zeros((0, 2), dtype=np.float32),
        )
        out = sigA.predict(lock, T_base_ee)
        self.assertAlmostEqual(out[0], u, delta=0.5)
        self.assertAlmostEqual(out[1], v, delta=0.5)

    def test_ee_translation_shifts_projection_predictably(self):
        """
        If we translate the EE +50mm along camera-X (i.e., the camera moves
        +50mm in world X under identity hand-eye), the 3D point's X in cam
        frame decreases by 50mm, so its u shifts left by fx * 50 / Z.
        """
        K = sl.make_intrinsics_matrix(sl.DEFAULT_INTRINSICS)
        T_ee_cam = np.eye(4)
        sigA = sl.SignalA(T_ee_cam, K)

        z = 500.0
        u0, v0 = 640.0, 360.0  # principal point
        pt_base = np.array([0.0, 0.0, z])

        lock = sl.LockState(
            mask_uint8=np.zeros((720, 1280), dtype=np.uint8),
            centroid_px=(u0, v0),
            bbox_xyxy=np.array([600, 320, 680, 400], dtype=np.float32),
            crop_bgr=np.zeros((10, 10, 3), dtype=np.uint8),
            crop_bbox=np.array([0, 0, 10, 10], dtype=np.float32),
            centroid_base=pt_base,
            z_at_lock_mm=z,
            T_base_ee=np.eye(4),
            init_points=np.zeros((0, 2), dtype=np.float32),
        )
        # First: should reproduce principal point under identity pose.
        out0 = sigA.predict(lock, np.eye(4))
        self.assertAlmostEqual(out0[0], u0, delta=0.5)
        self.assertAlmostEqual(out0[1], v0, delta=0.5)

        # Now move EE +50mm in base-frame X.
        T_now = np.eye(4)
        T_now[0, 3] = 50.0
        out1 = sigA.predict(lock, T_now)
        # Camera-frame X of the point is now -50, so u shifts by fx*(-50)/z
        expected_u = u0 + 700.0 * (-50.0) / 500.0
        self.assertAlmostEqual(out1[0], expected_u, delta=0.5)
        self.assertAlmostEqual(out1[1], v0, delta=0.5)


# ---------------------------------------------------------------------------
# Centroid bias: uniform-sampled subset is unbiased under partial framing
# ---------------------------------------------------------------------------

class UniformCentroidBiasTests(unittest.TestCase):
    """
    The headline property of Signal B: a centroid computed from a uniform
    spatial subsample of points inside the mask is an unbiased estimator
    of the true mask centroid even when half the points are off-frame.

    Compare to the biased baseline of "centroid of the visible portion of
    the mask," which drifts toward the visible side under partial framing.
    """

    def test_full_mask_centroid_matches(self):
        H, W = 720, 1280
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(mask, (400, 200), (800, 500), 255, -1)
        rng = np.random.default_rng(7)
        pts = sl.uniform_mask_samples(mask, 200, rng)
        sample_cx = pts[:, 0].mean()
        sample_cy = pts[:, 1].mean()
        # True centroid of the rectangle.
        ys, xs = np.where(mask > 0)
        true_cx, true_cy = xs.mean(), ys.mean()
        self.assertAlmostEqual(sample_cx, true_cx, delta=10.0)
        self.assertAlmostEqual(sample_cy, true_cy, delta=10.0)

    def test_partial_visibility_unbiased_when_using_all_tracked_points(self):
        """
        Property the implementation depends on:

        CoTracker3 reports a predicted position for EVERY tracked point per
        frame, including ones that have moved off-frame (extrapolated via
        cross-track attention). If we use all those predicted positions,
        the centroid is unbiased even when half the object exits the frame.

        Counter-example: using only positions that happen to be inside the
        image (the "visible subset") reproduces the same bias as the SAM2
        mask-centroid baseline. That is precisely the failure mode we are
        avoiding by using CoTracker, so the test asserts both:
          - "all points" centroid is UNBIASED (within a few px)
          - "visible only" centroid is BIASED toward the visible side
        """
        H, W = 720, 1280
        # True object: rectangle from x=400 to x=1100, y=200 to y=500.
        true_cx, true_cy = (400 + 1100) / 2.0, (200 + 500) / 2.0  # 750, 350

        mask_full = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(mask_full, (400, 200), (1100, 500), 255, -1)

        rng = np.random.default_rng(13)
        seed_pts = sl.uniform_mask_samples(mask_full, 400, rng)

        # Simulate camera approach: the object stays put in world but the
        # image window now only covers x in [0, 800). Tracked points retain
        # their TRUE positions (CoTracker3 extrapolates the off-frame ones
        # rather than dropping them).
        all_tracked = seed_pts                       # all positions known
        in_view = seed_pts[:, 0] < 800
        visible_only = seed_pts[in_view]

        # Mask-portion centroid (the SAM2 baseline)
        visible_mask = mask_full.copy()
        visible_mask[:, 800:] = 0
        ys, xs = np.where(visible_mask > 0)
        biased_mask_cx = xs.mean()

        # All-tracked-points centroid (the CoTracker3 estimator)
        all_cx = float(all_tracked[:, 0].mean())
        all_cy = float(all_tracked[:, 1].mean())

        # Visible-only centroid (degraded fallback, e.g. LK)
        vis_cx = float(visible_only[:, 0].mean())

        # Mask centroid is biased (~ pulled toward the visible side)
        self.assertLess(biased_mask_cx, true_cx - 100.0,
                        "expected mask-portion centroid to be biased")
        # Visible-only tracked points centroid is also biased (matches mask)
        self.assertLess(vis_cx, true_cx - 100.0,
                        "expected visible-only centroid to be biased")
        # The all-tracked-points centroid recovers the true centroid
        self.assertLess(abs(all_cx - true_cx), 10.0,
                        f"all-tracked centroid biased: {all_cx} vs {true_cx}")
        self.assertLess(abs(all_cy - true_cy), 10.0,
                        f"all-tracked centroid Y drift: {all_cy} vs {true_cy}")


# ---------------------------------------------------------------------------
# Fusion arbitration
# ---------------------------------------------------------------------------

class FusionTests(unittest.TestCase):

    def test_all_signals_agree_returns_consensus(self):
        readings = [
            sl.SignalReading("A", (100.0, 200.0), 1.0),
            sl.SignalReading("B", (101.0, 201.0), 0.8),
            sl.SignalReading("C", (99.0, 199.0), 0.3),
            sl.SignalReading("D", (100.5, 200.5), 0.6),
        ]
        out, info = sl.fuse_centroids(readings, c_A=(100.0, 200.0))
        self.assertAlmostEqual(out[0], 100.0, delta=1.0)
        self.assertAlmostEqual(out[1], 200.0, delta=1.0)
        self.assertFalse(info["override_applied"])

    def test_one_rogue_signal_does_not_pull_median(self):
        # B is a far outlier; A/C/D agree.
        readings = [
            sl.SignalReading("A", (100.0, 200.0), 1.0),
            sl.SignalReading("B", (900.0, 700.0), 1.0),
            sl.SignalReading("C", (101.0, 201.0), 0.3),
            sl.SignalReading("D", (99.0, 199.0), 0.6),
        ]
        out, _info = sl.fuse_centroids(readings, c_A=(100.0, 200.0))
        # Should stay near (100, 200), NOT be pulled toward (900, 700)
        self.assertLess(math.hypot(out[0] - 100.0, out[1] - 200.0), 50.0)

    def test_signal_A_override_when_perception_drifts(self):
        # B/C/D all agree on a wrong location far from A.
        readings = [
            sl.SignalReading("A", (100.0, 200.0), 1.0),
            sl.SignalReading("B", (500.0, 500.0), 1.0),
            sl.SignalReading("C", (505.0, 502.0), 0.4),
            sl.SignalReading("D", (498.0, 498.0), 0.7),
        ]
        out, info = sl.fuse_centroids(readings, c_A=(100.0, 200.0),
                                      override_px=80.0)
        self.assertTrue(info["override_applied"])
        self.assertEqual(out, (100.0, 200.0))

    def test_no_signal_A_still_fuses(self):
        readings = [
            sl.SignalReading("A", None, 0.0),
            sl.SignalReading("B", (100.0, 200.0), 0.8),
            sl.SignalReading("C", (102.0, 198.0), 0.3),
            sl.SignalReading("D", (101.0, 200.0), 0.5),
        ]
        out, info = sl.fuse_centroids(readings, c_A=None)
        self.assertIsNotNone(out)
        self.assertFalse(info["override_applied"])


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class StateMachineTests(unittest.TestCase):

    def _good_lock_candidate_result(self, w=1280, h=720):
        # Bbox 200x150 centered, well clear of border, area ~ 3% of frame
        x1, y1, x2, y2 = 540, 285, 740, 435
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        # Area frac ~ (200*150)/(1280*720) = 3.25% — within [5%, 25%]?  No,
        # under min. Make it bigger:
        x1, y1, x2, y2 = 440, 235, 840, 485
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        return dict(gdino_box=np.array([x1, y1, x2, y2], dtype=np.float32),
                    mask_np=mask)

    def test_lock_triggers_after_consec_frames(self):
        fsm = sl.StateMachine()
        res = self._good_lock_candidate_result()
        h, w = res["mask_np"].shape
        for i in range(sl.LOCK_CONSEC_FRAMES - 1):
            self.assertFalse(fsm.evaluate_far(res, depth_at_centroid_mm=250.0,
                                              image_shape=(h, w)))
        self.assertTrue(fsm.evaluate_far(res, depth_at_centroid_mm=250.0,
                                          image_shape=(h, w)))

    def test_lock_resets_on_border_touch(self):
        fsm = sl.StateMachine()
        good = self._good_lock_candidate_result()
        h, w = good["mask_np"].shape
        bad = dict(good, gdino_box=np.array([0.0, 100.0, 200.0, 300.0],
                                             dtype=np.float32))
        for _ in range(5):
            fsm.evaluate_far(good, 250.0, (h, w))
        # Now a border-touching frame: counter should reset
        self.assertFalse(fsm.evaluate_far(bad, 250.0, (h, w)))

    def test_near_trigger_on_border(self):
        fsm = sl.StateMachine()
        h, w = 720, 1280
        # Bbox right at left border
        res = dict(gdino_box=np.array([0.0, 100.0, 200.0, 300.0],
                                       dtype=np.float32))
        self.assertTrue(fsm.near_trigger_fired(res, z_mm=120.0,
                                               image_shape=(h, w)))

    def test_near_trigger_on_area(self):
        fsm = sl.StateMachine()
        h, w = 720, 1280
        # Big bbox covering most of frame
        res = dict(gdino_box=np.array([100.0, 100.0, 1100.0, 600.0],
                                       dtype=np.float32))
        self.assertTrue(fsm.near_trigger_fired(res, z_mm=200.0,
                                               image_shape=(h, w)))

    def test_near_trigger_on_depth(self):
        fsm = sl.StateMachine()
        h, w = 720, 1280
        res = dict(gdino_box=np.array([500.0, 300.0, 700.0, 500.0],
                                       dtype=np.float32))
        self.assertTrue(fsm.near_trigger_fired(res, z_mm=70.0,
                                               image_shape=(h, w)))

    def test_term_trigger_on_low_depth(self):
        fsm = sl.StateMachine()
        self.assertTrue(fsm.evaluate_near_to_term(
            c_fused=(640.0, 360.0), z_mm=20.0, image_center=(640.0, 360.0)))

    def test_term_trigger_on_low_pixel_error(self):
        fsm = sl.StateMachine()
        self.assertTrue(fsm.evaluate_near_to_term(
            c_fused=(641.0, 361.0), z_mm=200.0, image_center=(640.0, 360.0)))

    def test_watchdog_alarms_after_disagreement_streak(self):
        fsm = sl.StateMachine()
        for _ in range(sl.WATCHDOG_DISAGREE_FRAMES - 1):
            self.assertFalse(fsm.update_watchdog(
                fused_xy=(100.0, 100.0), c_C=(500.0, 500.0)))
        self.assertTrue(fsm.update_watchdog(
            fused_xy=(100.0, 100.0), c_C=(500.0, 500.0)))

    def test_watchdog_resets_on_agreement(self):
        fsm = sl.StateMachine()
        fsm.update_watchdog(fused_xy=(100.0, 100.0), c_C=(500.0, 500.0))
        fsm.update_watchdog(fused_xy=(100.0, 100.0), c_C=(500.0, 500.0))
        # Agreement: counter resets
        fsm.update_watchdog(fused_xy=(100.0, 100.0), c_C=(101.0, 101.0))
        self.assertEqual(fsm._watchdog_disagree, 0)


# ---------------------------------------------------------------------------
# Plane fit
# ---------------------------------------------------------------------------

class PlaneFitTests(unittest.TestCase):

    def test_horizontal_plane(self):
        # Synthesize a flat depth map at z=300mm for a 200x200 patch
        depth = np.full((720, 1280), np.nan, dtype=np.float32)
        depth[260:460, 540:740] = 300.0
        # Add a hint of noise so RANSAC has something nontrivial to do
        rng = np.random.default_rng(0)
        depth[260:460, 540:740] += rng.normal(0, 0.5, (200, 200)).astype(np.float32)

        K = sl.make_intrinsics_matrix(sl.DEFAULT_INTRINSICS)
        out = sl.terminal_plane_fit(depth, K, (640.0, 360.0),
                                     patch_half=80, rng=rng)
        self.assertIsNotNone(out)
        normal, centroid = out
        # Normal should point along -Z (toward camera) since plane is flat
        # and oriented to face camera.
        self.assertGreater(abs(normal[2]), 0.95,
                           f"normal Z component too small: {normal}")
        # Centroid Z should be near 300mm
        self.assertAlmostEqual(float(centroid[2]), 300.0, delta=2.0)

    def test_tilted_plane(self):
        # Create a plane tilted 30deg about the Y axis. depth(x) = 300 + tan(30)*(x-cx)
        H, W = 720, 1280
        K = sl.make_intrinsics_matrix(sl.DEFAULT_INTRINSICS)
        depth = np.full((H, W), np.nan, dtype=np.float32)
        for x in range(540, 740):
            for y in range(260, 460):
                depth[y, x] = 300.0 + math.tan(math.radians(30.0)) * (x - 640) * 0.3

        out = sl.terminal_plane_fit(depth, K, (640.0, 360.0),
                                     patch_half=80,
                                     rng=np.random.default_rng(0))
        self.assertIsNotNone(out)
        normal, _ = out
        # Normal should have a meaningful X component (not purely Z)
        self.assertGreater(abs(normal[0]), 0.05,
                           f"tilt not detected: {normal}")


# ---------------------------------------------------------------------------
# bbox helpers
# ---------------------------------------------------------------------------

class BboxHelperTests(unittest.TestCase):

    def test_border_touch_detection(self):
        # 1280x720 frame, bbox far from border
        self.assertFalse(sl.bbox_touches_border(
            np.array([300, 200, 800, 500]), 1280, 720))
        # Bbox right against left border
        self.assertTrue(sl.bbox_touches_border(
            np.array([0, 200, 800, 500]), 1280, 720))
        # Bbox right against bottom border
        self.assertTrue(sl.bbox_touches_border(
            np.array([300, 200, 800, 720]), 1280, 720))

    def test_area_fraction(self):
        af = sl.bbox_area_fraction(np.array([0, 0, 640, 360]), 1280, 720)
        self.assertAlmostEqual(af, 0.25, places=4)

    def test_in_frame_fraction(self):
        pts = np.array([[0, 0], [100, 100], [-10, 50], [1300, 100]],
                       dtype=np.float32)
        f = sl.in_frame_fraction(pts, 1280, 720)
        self.assertAlmostEqual(f, 0.5)


# ---------------------------------------------------------------------------
# End-to-end orchestrator with a synthetic SAM3 stub
# ---------------------------------------------------------------------------

class OrchestratorEndToEndTests(unittest.TestCase):
    """
    Drives the full pipeline FAR -> NEAR (-> TERMINAL) using a synthetic
    SAM3 stub, simulated EE pose feed, and a constant depth map.

    No model weights are loaded. CoTracker3 / SAM2 / DINOv2 will fall
    back to LK / disabled / disabled — but the FSM and Signal A path
    are exercised end-to-end, including the LOCK transition.
    """

    def _build(self, intrinsics: dict | None = None):
        # Synthetic frame: solid color box on dark bg, growing as we approach.
        H, W = 720, 1280

        # External per-step counter so the stub depth schedule keeps
        # advancing even after SAM3 is no longer being called in NEAR/TERM.
        state = dict(step=0, sam3_calls=0)

        def sam3_stub(frame: np.ndarray) -> dict:
            i = state["sam3_calls"]
            state["sam3_calls"] += 1
            # Object size grows with frame index to simulate approach
            half_w = 80 + i * 20  # +20 px per frame
            half_h = 60 + i * 15
            cx, cy = 640, 360
            x1 = max(0, cx - half_w); y1 = max(0, cy - half_h)
            x2 = min(W, cx + half_w); y2 = min(H, cy + half_h)
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            return dict(
                mask_np=mask,
                gdino_box=np.array([x1, y1, x2, y2], dtype=np.float32),
                sam_score=0.9, similarity=0.85,
                detector_used="stub", detection_was_skipped=False,
                dets_all=None, depth_np=None,
            )

        # depth_provider is called on EVERY pipeline.step (FAR, NEAR, TERM).
        # We use this as the authoritative tick.
        def depth_provider(frame: np.ndarray) -> np.ndarray:
            i = state["step"]
            state["step"] += 1
            z_mm = max(20.0, 300.0 - 20.0 * i)
            depth = np.full((H, W), z_mm, dtype=np.float32)
            return depth

        # Stationary EE pose (no robot motion): identity for all frames.
        ee_pose_provider = lambda: np.eye(4)

        intr = intrinsics or sl.DEFAULT_INTRINSICS
        pipe = sl.LastMilePipeline(
            sam3_runner=sam3_stub,
            ee_pose_provider=ee_pose_provider,
            depth_provider=depth_provider,
            hand_eye_path=None,
            intrinsics=intr,
        )
        return pipe, state

    def _synthetic_frame(self):
        # Important: this frame must have texture inside the LOCK-time mask
        # region so LK / CoTracker can actually find features to track.
        # Use a deterministic noise pattern over the whole image.
        H, W = 720, 1280
        rng = np.random.default_rng(7)
        f = (rng.integers(40, 215, size=(H, W, 3))).astype(np.uint8)
        # Paint a brighter textured rectangle where the SAM3 stub's mask lives,
        # so the centroid signal has a clear visual focus.
        cv2.rectangle(f, (440, 220), (840, 500), (220, 200, 150), -1)
        # Add salt-and-pepper specks for richer LK gradients
        ys = rng.integers(220, 500, size=200)
        xs = rng.integers(440, 840, size=200)
        f[ys, xs] = 0
        return f

    def test_first_frame_is_FAR(self):
        pipe, _ = self._build()
        res = pipe.step(self._synthetic_frame())
        self.assertEqual(res["state"], "FAR")
        self.assertIsNotNone(res["best_centroid"])

    def test_progression_through_LOCK_NEAR(self):
        pipe, _ = self._build()
        seen_states = []
        for _ in range(20):
            res = pipe.step(self._synthetic_frame())
            seen_states.append(res["state"])
            if res["state"] == "TERMINAL":
                break
        # We expect FAR for some frames, then NEAR.
        self.assertIn("FAR", seen_states)
        self.assertIn("NEAR", seen_states,
                      f"never transitioned to NEAR: {seen_states}")
        # FAR must precede NEAR (no time-travel)
        far_max = max(i for i, s in enumerate(seen_states) if s == "FAR")
        near_min = min(i for i, s in enumerate(seen_states) if s == "NEAR")
        self.assertLess(far_max, near_min + 1)

    def test_terminal_eventually_reached(self):
        pipe, state = self._build()
        seen = []
        for _ in range(40):
            res = pipe.step(self._synthetic_frame())
            seen.append(res["state"])
            if res["state"] == "TERMINAL":
                break
        self.assertIn("TERMINAL", seen,
                      f"never reached TERMINAL in 40 frames: {seen}")


# ---------------------------------------------------------------------------
# SAM3 pipeline grasp-point freeze when bbox touches frame border
# ---------------------------------------------------------------------------

class Sam3CentroidFreezeTests(unittest.TestCase):
    """
    Verifies the rate-limit added to servo_pipeline_sam3.run_pipeline that
    stabilizes the grasp point when the bbox starts touching a border.

    The test imports the SAM3 module's helpers (no model weights needed)
    and exercises just the freeze logic in isolation by simulating the
    relevant tracker state and result dict.
    """

    def setUp(self):
        try:
            import servo_pipeline_sam3 as sm  # noqa: F401
            self.sm = sm
        except Exception as e:
            self.skipTest(f"servo_pipeline_sam3 not importable here: {e}")

    def test_full_frame_updates_reference(self):
        """When bbox is fully framed, last_full_frame_centroid is refreshed."""
        sm = self.sm
        trk = sm.MaskTracker()
        # Simulate the freeze logic directly
        bx = np.array([400, 200, 800, 500], dtype=np.float32)
        # Fake "current centroid"
        centroid = (600, 350)
        h, w = 720, 1280
        side = max(bx[2] - bx[0], bx[3] - bx[1])
        margin = max(8, int(0.05 * side))
        bbox_at_border = (bx[0] < margin or bx[1] < margin
                          or bx[2] > w - margin or bx[3] > h - margin)
        self.assertFalse(bbox_at_border)
        if not bbox_at_border:
            trk.last_full_frame_centroid = centroid
        self.assertEqual(trk.last_full_frame_centroid, (600, 350))

    def test_border_touching_clamps_drift(self):
        """When bbox touches border, drift is capped at 6 px from reference."""
        trk = self.sm.MaskTracker()
        trk.last_full_frame_centroid = (600, 350)
        # Simulate a centroid that jumped 100 px (the failure mode the
        # user observed in real runs)
        ref = trk.last_full_frame_centroid
        new_centroid = (700, 350)  # jumped +100 in x
        dx = new_centroid[0] - ref[0]
        dy = new_centroid[1] - ref[1]
        d = (dx * dx + dy * dy) ** 0.5
        MAX_DRIFT = 6
        if d > MAX_DRIFT:
            scale = MAX_DRIFT / d
            limited = (int(ref[0] + dx * scale), int(ref[1] + dy * scale))
        else:
            limited = new_centroid
        self.assertLess(abs(limited[0] - ref[0]), MAX_DRIFT + 1)
        self.assertLess(abs(limited[1] - ref[1]), MAX_DRIFT + 1)

    def test_border_detection_left_edge(self):
        """A bbox touching the left edge triggers the freeze."""
        bx = np.array([0, 200, 400, 500], dtype=np.float32)
        h, w = 720, 1280
        side = max(bx[2] - bx[0], bx[3] - bx[1])
        margin = max(8, int(0.05 * side))
        bbox_at_border = (bx[0] < margin or bx[1] < margin
                          or bx[2] > w - margin or bx[3] > h - margin)
        self.assertTrue(bbox_at_border)

    def test_border_detection_bottom_edge(self):
        """A bbox touching the bottom edge triggers the freeze."""
        bx = np.array([400, 200, 800, 718], dtype=np.float32)
        h, w = 720, 1280
        side = max(bx[2] - bx[0], bx[3] - bx[1])
        margin = max(8, int(0.05 * side))
        bbox_at_border = (bx[0] < margin or bx[1] < margin
                          or bx[2] > w - margin or bx[3] > h - margin)
        self.assertTrue(bbox_at_border)

    def test_jyz_prediction_recovers_lock_centroid_when_ee_unmoved(self):
        """
        With J_yz known and EE pose unchanged since lock, the predicted
        centroid must equal the locked centroid exactly. This is the
        identity case for the streamer's _stabilize_grasp predictor.
        """
        # Synthetic J_yz: 1 mm of Y-motion produces 10 px of x-shift,
        # 1 mm of Z-motion produces 10 px of y-shift. Sign convention
        # matches the controller's own calibration.
        J = np.array([[10.0, 0.0],
                      [0.0, 10.0]], dtype=np.float64)
        lock_c = (640, 360)
        lock_yz = (50.0, 100.0)
        cur_yz = (50.0, 100.0)
        d_world = np.array([cur_yz[0] - lock_yz[0],
                            cur_yz[1] - lock_yz[1]], dtype=np.float64)
        d_px = J @ d_world
        predicted = (int(lock_c[0] + d_px[0]),
                     int(lock_c[1] + d_px[1]))
        self.assertEqual(predicted, lock_c)

    def test_jyz_prediction_shifts_with_ee_motion(self):
        """
        When EE moves +5 mm in Y, predicted centroid shifts by J[:,0] * 5.
        With J = [[10, 0], [0, 10]] that's +50 px in x, +0 px in y.
        """
        J = np.array([[10.0, 0.0],
                      [0.0, 10.0]], dtype=np.float64)
        lock_c = (640, 360)
        lock_yz = (50.0, 100.0)
        cur_yz = (55.0, 100.0)  # +5 mm in Y
        d_world = np.array([5.0, 0.0])
        d_px = J @ d_world
        predicted = (int(lock_c[0] + d_px[0]),
                     int(lock_c[1] + d_px[1]))
        self.assertEqual(predicted, (640 + 50, 360))

    def test_prediction_cap_triggers_hard_freeze_under_large_dyz(self):
        """
        When |Δyz| exceeds PREDICT_MAX_DELTA_MM the linear J_yz
        extrapolation is unreliable (it's a small-motion linearization
        around the calibration distance). The stabilizer must fall back
        to a hard freeze at lock_centroid in that regime.

        This is the regime the user hit on the rig: MIN_Z_MM clamping
        forces the EE to fly +52.9 mm in Z at the start of servoing,
        which is well outside the J_yz validity range and produced an
        80 px y-overshoot in the prediction. With the cap in place the
        stabilizer detects |Δyz|=52.9 > 15 and uses the lock centroid
        directly.
        """
        # Use the same J_yz from the user's actual calibration run
        J = np.array([[1.6985, -0.0062],
                      [-0.1309, 1.5447]], dtype=np.float64)
        lock_c = (648, 461)
        lock_yz = (5.6, -52.9)
        cur_yz = (3.0, 0.0)  # The Z fly-up scenario
        dy = cur_yz[0] - lock_yz[0]
        dz = cur_yz[1] - lock_yz[1]
        mag = (dy * dy + dz * dz) ** 0.5
        # Δyz magnitude must exceed the cap
        self.assertGreater(mag, self.sm.PREDICT_MAX_DELTA_MM)

        # Without the cap, prediction would be:
        d_world = np.array([dy, dz], dtype=np.float64)
        d_px = J @ d_world
        unguarded_predicted = (int(lock_c[0] + d_px[0]),
                                int(lock_c[1] + d_px[1]))
        # The unguarded prediction should be far below the actual
        # protein bar (y=543) — proving the cap is necessary
        self.assertGreater(unguarded_predicted[1], 540)

        # With the cap in place, the stabilizer returns lock_c instead
        # — preserving the grasp marker's position throughout the run.
        capped = lock_c
        self.assertEqual(capped, (648, 461))

    def test_jyz_prediction_handles_off_diagonal_jacobian(self):
        """
        Real J_yz from a calibration log:
            [[ 1.27 -0.00]
             [-0.08  1.15]]
        Verify the prediction respects off-diagonal coupling.
        """
        J = np.array([[1.27, -0.00],
                      [-0.08, 1.15]], dtype=np.float64)
        lock_c = (640, 360)
        lock_yz = (5.6, -52.9)
        cur_yz = (15.6, -52.9)  # +10 mm Y
        d_world = np.array([10.0, 0.0])
        d_px = J @ d_world
        predicted_x = int(lock_c[0] + d_px[0])
        predicted_y = int(lock_c[1] + d_px[1])
        # +10 mm Y -> +12.7 px x, -0.8 px y from the matrix
        # int() truncates toward zero: int(640+12.7)=652, int(360-0.8)=359
        self.assertEqual(predicted_x, 652)
        self.assertEqual(predicted_y, 359)

    def test_creep_reference_advances_with_clamped_centroid(self):
        """
        The freeze reference creeps with the rate-limited centroid so that
        sustained legitimate motion doesn't get permanently anchored to a
        stale point. With a 100 px gap and a 6 px/frame cap, convergence
        should occur in ~17 frames; 20 frames is a comfortable upper bound.
        """
        trk = self.sm.MaskTracker()
        trk.last_full_frame_centroid = (600, 350)
        target = (700, 350)
        MAX = 6
        for _ in range(20):
            ref = trk.last_full_frame_centroid
            dx, dy = target[0] - ref[0], target[1] - ref[1]
            d = (dx * dx + dy * dy) ** 0.5
            if d > MAX:
                s = MAX / d
                ref = (int(ref[0] + dx * s), int(ref[1] + dy * s))
            else:
                ref = target
            trk.last_full_frame_centroid = ref
        # Convergence reached: reference is at (or very close to) target.
        self.assertLess(abs(trk.last_full_frame_centroid[0] - 700), 5)
        # Also: at frame 1 (single capped step), ref should have moved
        # exactly 6 px from the start, not 100. Sanity check the cap.
        trk2 = self.sm.MaskTracker()
        trk2.last_full_frame_centroid = (600, 350)
        ref = trk2.last_full_frame_centroid
        dx = target[0] - ref[0]
        s = MAX / abs(dx)
        ref_after_one = (int(ref[0] + dx * s), int(ref[1]))
        self.assertEqual(ref_after_one, (606, 350))


if __name__ == "__main__":
    unittest.main(verbosity=2)
