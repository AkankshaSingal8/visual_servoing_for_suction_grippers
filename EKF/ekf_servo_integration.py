"""
Integration guide: EKF pose estimation into semvs_dinov2_servo.py

This file shows the key modifications needed. The three main changes are:

1. Add EKF + DepthScaler to the main setup
2. Replace the direct centroid → Jacobian servo with EKF predict/update → PBVS
3. Update the render HUD to show EKF state

Below are the modified functions / sections with CHANGED markers.
"""

import time
import numpy as np
from ekf_servo import PoseEKF, PBVSController, CameraIntrinsics, DepthScaler


# ═════════════════════════════════════════════════════════════════════
#  1. SETUP — add to main() after building the tracker
# ═════════════════════════════════════════════════════════════════════

def setup_ekf():
    """
    Create the EKF, PBVS controller, and depth scaler.
    Call this in main() and pass these objects to CameraStreamer.
    """
    # Camera intrinsics — replace with your ZED Mini calibration
    # ZED Mini typical values at 720p:
    cam = CameraIntrinsics(fx=700.0, fy=700.0, cx=640.0, cy=360.0)

    # EKF with tuned noise parameters
    ekf = PoseEKF(
        cam,
        process_noise_pos=0.01,   # how much we trust const-vel model
        process_noise_vel=0.1,    # velocity can change this much
        meas_noise_uv=6.0,       # pixel centroid noise (std dev)
        meas_noise_z=0.10,       # depth noise in metres
    )

    # PBVS controller
    pbvs = PBVSController(
        gain=0.4,
        target_depth=0.35,    # desired distance to object (metres)
        max_vel=0.015,        # max 15 mm/s per axis
        dead_zone_m=0.008,    # 8mm dead zone
    )

    # Monocular depth → metric converter
    depth_scaler = DepthScaler(default_scale=0.001)

    return ekf, pbvs, cam, depth_scaler


# ═════════════════════════════════════════════════════════════════════
#  2. MODIFIED PIPELINE STEP — replaces run_pipeline()
# ═════════════════════════════════════════════════════════════════════

def run_pipeline_ekf(image_bgr, tracker, ekf, depth_scaler,
                     dt, robot_vel_cam=None):
    """
    One perception step with EKF integration.

    Flow:
        Frame → Detect/Track → mask centroid (u,v)
              → Depth Anything → median depth under mask → metric Z
              → EKF predict(dt, robot_vel)
              → EKF update(u, v, Z)   or  update_2d(u, v) if no depth
              → filtered 3D position for PBVS
    """
    # ── 1. Segmentation (detect or track) ────────────────────────
    res = tracker.process_frame(image_bgr)

    # ── 2. EKF Predict ───────────────────────────────────────────
    #    Always predict first, even if no measurement this frame.
    #    robot_vel_cam compensates for robot-induced image motion.
    ekf.predict(dt, robot_vel_cam=robot_vel_cam)

    centroid = res.get("best_centroid")
    if centroid is None:
        # No detection — return prediction only
        if ekf.is_initialised:
            res["ekf_position"] = ekf.position
            res["ekf_velocity"] = ekf.velocity
            res["ekf_pixel"] = ekf.predicted_pixel()
            res["ekf_uncertainty"] = ekf.position_uncertainty()
        return res

    u, v = float(centroid[0]), float(centroid[1])

    # ── 3. Depth estimation ──────────────────────────────────────
    run_depth = (res["mode"] == "detect" or tracker._track_count % 5 == 0)
    depth_metric = None

    if run_depth:
        from semvs_dinov2_servo import _get_depth_model
        dm = _get_depth_model()
        if dm is not None:
            try:
                depth_relative = dm.infer_image(image_bgr).astype(np.float32)
                res["depth_np"] = depth_relative

                # Get metric depth under the mask
                mask = res.get("mask_np")
                if mask is not None:
                    depth_metric = depth_scaler.estimate_object_depth(
                        depth_relative, mask, percentile=50)
            except Exception as e:
                print("Depth failed:", e)

    # ── 4. EKF Update ────────────────────────────────────────────
    if depth_metric is not None and depth_metric > 0.05:
        # Full 3-DOF update: pixel centroid + metric depth
        ekf.update(u, v, depth_metric)
        res["depth_metric"] = depth_metric
    else:
        # 2D-only update: still constrains X/Y, depth drifts slowly
        ekf.update_2d_only(u, v)
        res["depth_metric"] = None

    # ── 5. Pack EKF results ──────────────────────────────────────
    res["ekf_position"] = ekf.position
    res["ekf_velocity"] = ekf.velocity
    res["ekf_pixel"] = ekf.predicted_pixel()
    res["ekf_uncertainty"] = ekf.position_uncertainty()

    return res


# ═════════════════════════════════════════════════════════════════════
#  3. MODIFIED SERVO STEP — replaces RobotController.servo_step()
# ═════════════════════════════════════════════════════════════════════

def servo_step_pbvs(robot, pbvs, ekf_position, ekf_velocity):
    """
    Position-based servo step using EKF-filtered 3D pose.

    Instead of:
        pixel error → calibrated Jacobian → robot delta

    We now do:
        EKF 3D position → PBVS control law → robot velocity
    """
    if robot._arm is None or not robot.enabled:
        return

    now = time.time()
    if now - robot._last_t < VS_RATE:
        return
    robot._last_t = now

    # PBVS computes velocity in robot frame
    v_robot, err_m = pbvs.compute_velocity(ekf_position)

    if err_m < pbvs.dead_zone_m:
        return

    # Convert m/s velocity to mm position delta (one servo period)
    # v_robot is [vx, vy, vz] in m/s, multiply by dt and convert to mm
    dt_step = VS_RATE
    dx_mm = float(v_robot[0]) * dt_step * 1000.0
    dy_mm = float(v_robot[1]) * dt_step * 1000.0
    dz_mm = float(v_robot[2]) * dt_step * 1000.0

    # Add constant approach along X (robot frame)
    dx_mm += VS_APPROACH

    with robot._lock:
        try:
            pos = robot._get_pos()
            if pos is None:
                return
            robot._arm.set_position(
                x=pos[0] + dx_mm, y=pos[1] + dy_mm,
                z=pos[2] + dz_mm,
                roll=pos[3], pitch=pos[4], yaw=pos[5],
                speed=VS_SPEED, mvacc=VS_MVACC, wait=True)
            print(f"PBVS: pos=({ekf_position[0]:.3f},"
                  f"{ekf_position[1]:.3f},{ekf_position[2]:.3f})m  "
                  f"err={err_m:.4f}m  "
                  f"Δ=({dx_mm:+.1f},{dy_mm:+.1f},{dz_mm:+.1f})mm")
        except Exception as e:
            print("PBVS step failed:", e)


# ═════════════════════════════════════════════════════════════════════
#  4. MODIFIED SEG LOOP — replaces CameraStreamer._seg_loop()
# ═════════════════════════════════════════════════════════════════════

# Constants (import from main file or redefine)
VS_RATE = 0.3
VS_APPROACH = 3.0
VS_SPEED = 150
VS_MVACC = 500


def modified_seg_loop(streamer, ekf, pbvs, depth_scaler):
    """
    Drop-in replacement for CameraStreamer._seg_loop().
    Adds EKF predict/update and PBVS servo.
    """
    last_time = time.time()

    while not streamer.stop_event.is_set():
        with streamer._frame_lock:
            frame = (streamer._latest_left.copy()
                     if streamer._latest_left is not None else None)
        if frame is None:
            time.sleep(0.1)
            continue

        try:
            now = time.time()
            dt = now - last_time
            last_time = now

            # TODO: compute robot_vel_cam from last robot command
            # For now, pass None (no egomotion compensation)
            robot_vel_cam = None

            res = run_pipeline_ekf(
                frame, streamer.tracker,
                ekf, depth_scaler,
                dt, robot_vel_cam)

            with streamer.data_lock:
                streamer._result = res

            if not streamer._models_ready.is_set():
                print("Models ready — calibration may proceed.")
                streamer._models_ready.set()

            # PBVS servo using EKF-filtered position
            ekf_pos = res.get("ekf_position")
            ekf_vel = res.get("ekf_velocity")
            if (streamer.robot is not None
                    and ekf_pos is not None
                    and ekf.is_initialised):
                servo_step_pbvs(
                    streamer.robot, pbvs, ekf_pos, ekf_vel)

        except Exception as e:
            import traceback
            print("Pipeline error:", e)
            traceback.print_exc()
            time.sleep(1)


# ═════════════════════════════════════════════════════════════════════
#  5. MODIFIED RENDER — add EKF visualisation to _render()
# ═════════════════════════════════════════════════════════════════════

def render_ekf_overlay(display, res, h, w):
    """
    Add EKF-specific overlays to the rendered frame.
    Call this at the end of CameraStreamer._render().
    """
    import cv2

    ekf_pixel = res.get("ekf_pixel")
    ekf_pos = res.get("ekf_position")
    ekf_unc = res.get("ekf_uncertainty")
    depth_m = res.get("depth_metric")

    # ── EKF filtered centroid (blue circle, distinct from raw) ───
    if ekf_pixel is not None:
        eu, ev = ekf_pixel
        # Draw uncertainty ellipse (scaled to pixels)
        if ekf_unc is not None:
            radius = max(5, int(ekf_unc * 500))  # scale for visibility
            cv2.circle(display, (eu, ev), radius,
                       (255, 100, 100), 1, cv2.LINE_AA)
        # Filtered centroid marker
        cv2.drawMarker(display, (eu, ev), (255, 50, 50),
                       cv2.MARKER_TILTED_CROSS, 18, 2)

    # ── 3D position HUD ─────────────────────────────────────────
    if ekf_pos is not None:
        txt = (f"EKF: X={ekf_pos[0]:.3f} Y={ekf_pos[1]:.3f} "
               f"Z={ekf_pos[2]:.3f} m")
        cv2.putText(display, txt, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (255, 150, 100), 1)

    if depth_m is not None:
        cv2.putText(display, f"depth: {depth_m:.3f} m", (8, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (200, 200, 100), 1)

    if ekf_unc is not None:
        cv2.putText(display, f"unc: {ekf_unc:.4f}", (8, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (180, 180, 180), 1)

    return display


# ═════════════════════════════════════════════════════════════════════
#  6. MODIFIED main() — shows what changes in main()
# ═════════════════════════════════════════════════════════════════════

def main_with_ekf():
    """
    Modified main() — only showing the new/changed lines.
    Everything else (argparse, reference loading, tracker, robot)
    stays the same.
    """
    # ... [existing argparse, reference loading, ref_cache, tracker] ...

    # ── NEW: set up EKF + PBVS ───────────────────────────────────
    ekf, pbvs, cam_intrinsics, depth_scaler = setup_ekf()

    # ... [existing robot setup] ...

    # ── NEW: pass ekf/pbvs to CameraStreamer ─────────────────────
    # Modify CameraStreamer.__init__ to accept ekf, pbvs, depth_scaler
    # and use modified_seg_loop instead of _seg_loop

    # ... [rest of main unchanged] ...
