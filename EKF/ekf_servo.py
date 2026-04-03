"""
EKF-based object pose estimation for visual servoing.

State:  x = [X, Y, Z, Vx, Vy, Vz]^T   (object position + velocity in camera frame)

Process model (constant velocity + robot egomotion compensation):
    x_{k|k-1} = F @ x_{k-1} + B @ u_k + w_k

Measurement model (pinhole projection + depth):
    z = h(x) = [fx * X/Z + cx,  fy * Y/Z + cy,  Z]^T + v_k

The EKF linearises h(x) via the analytical Jacobian (interaction matrix)
at each update step, giving filtered 3D position and velocity estimates
for position-based visual servoing (PBVS).
"""

import numpy as np


class CameraIntrinsics:
    """Pinhole camera parameters."""

    def __init__(self, fx=500.0, fy=500.0, cx=320.0, cy=240.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def project(self, X, Y, Z):
        """3D camera-frame point → 2D pixel."""
        Z = max(Z, 1e-4)
        u = self.fx * X / Z + self.cx
        v = self.fy * Y / Z + self.cy
        return u, v

    def backproject(self, u, v, Z):
        """2D pixel + depth → 3D camera-frame point."""
        Z = max(Z, 1e-4)
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        return X, Y, Z


class PoseEKF:
    """
    Extended Kalman Filter for object pose in camera frame.

    State: [X, Y, Z, Vx, Vy, Vz]
        X, Y, Z  — object position in camera frame (metres)
        Vx,Vy,Vz — velocity in camera frame (m/s)

    Measurements (up to 3):
        u, v  — 2D centroid in pixel coordinates
        Z     — monocular depth estimate (metres)
    """

    DIM_STATE = 6
    DIM_MEAS = 3  # (u, v, Z)

    def __init__(self, cam: CameraIntrinsics,
                 process_noise_pos=0.005,
                 process_noise_vel=0.05,
                 meas_noise_uv=8.0,
                 meas_noise_z=0.15):
        self.cam = cam

        # ── State and covariance ─────────────────────────────────
        self.x = np.zeros(self.DIM_STATE)  # filled on first measurement
        self.P = np.eye(self.DIM_STATE)

        # ── Process noise Q ──────────────────────────────────────
        self.Q = np.diag([
            process_noise_pos ** 2,
            process_noise_pos ** 2,
            process_noise_pos ** 2,
            process_noise_vel ** 2,
            process_noise_vel ** 2,
            process_noise_vel ** 2,
        ])

        # ── Measurement noise R ──────────────────────────────────
        self.R = np.diag([
            meas_noise_uv ** 2,
            meas_noise_uv ** 2,
            meas_noise_z ** 2,
        ])

        self._initialised = False

    @property
    def position(self):
        return self.x[:3].copy()

    @property
    def velocity(self):
        return self.x[3:6].copy()

    @property
    def is_initialised(self):
        return self._initialised

    # ─────────────────────────────────────────────────────────────
    #  Initialisation from first measurement
    # ─────────────────────────────────────────────────────────────

    def initialise(self, u, v, Z):
        """Seed the filter from the first valid observation."""
        X, Y, Z = self.cam.backproject(u, v, Z)
        self.x = np.array([X, Y, Z, 0.0, 0.0, 0.0])
        self.P = np.diag([0.05, 0.05, 0.1, 0.5, 0.5, 0.5])
        self._initialised = True

    # ─────────────────────────────────────────────────────────────
    #  Predict
    # ─────────────────────────────────────────────────────────────

    def predict(self, dt, robot_vel_cam=None):
        """
        Propagate state forward.

        Parameters
        ----------
        dt : float
            Time step (seconds).
        robot_vel_cam : array-like (3,), optional
            Robot end-effector velocity in camera frame [vx, vy, vz] (m/s).
            Used for egomotion compensation — the object appears to move
            opposite to the camera's motion.
        """
        if not self._initialised:
            return

        # State transition: constant velocity
        F = np.eye(self.DIM_STATE)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        self.x = F @ self.x

        # Egomotion compensation: object velocity in camera frame
        # shifts by negative robot velocity
        if robot_vel_cam is not None:
            v_cam = np.asarray(robot_vel_cam, dtype=np.float64)
            # Robot moving +X in camera frame → object appears to move -X
            self.x[0] -= v_cam[0] * dt
            self.x[1] -= v_cam[1] * dt
            self.x[2] -= v_cam[2] * dt

        # Covariance propagation
        # Scale process noise by dt so faster updates → less noise injected
        Q_scaled = self.Q * dt
        self.P = F @ self.P @ F.T + Q_scaled

        # Enforce minimum depth (object can't be behind camera)
        if self.x[2] < 0.05:
            self.x[2] = 0.05

    # ─────────────────────────────────────────────────────────────
    #  Update
    # ─────────────────────────────────────────────────────────────

    def update(self, u, v, Z):
        """
        EKF update with measurement z = [u, v, Z].

        The observation model is:
            h(x) = [ fx * X/Z + cx ]
                   [ fy * Y/Z + cy ]
                   [      Z        ]

        and its Jacobian H = dh/dx evaluated at the current state.
        """
        if not self._initialised:
            self.initialise(u, v, Z)
            return

        X, Y, Zs = self.x[0], self.x[1], max(self.x[2], 1e-4)

        # ── Predicted measurement h(x) ───────────────────────────
        u_pred = self.cam.fx * X / Zs + self.cam.cx
        v_pred = self.cam.fy * Y / Zs + self.cam.cy
        z_pred = np.array([u_pred, v_pred, Zs])

        # ── Measurement Jacobian H (3×6) ─────────────────────────
        #
        #  dh/dX = [fx/Z,     0, -fx*X/Z²,  0, 0, 0]
        #  dh/dY = [0,     fy/Z, -fy*Y/Z²,  0, 0, 0]
        #  dh/dZ = [0,        0,        1,  0, 0, 0]
        #
        H = np.zeros((self.DIM_MEAS, self.DIM_STATE))
        H[0, 0] = self.cam.fx / Zs
        H[0, 2] = -self.cam.fx * X / (Zs ** 2)
        H[1, 1] = self.cam.fy / Zs
        H[1, 2] = -self.cam.fy * Y / (Zs ** 2)
        H[2, 2] = 1.0

        # ── Innovation ───────────────────────────────────────────
        z_meas = np.array([u, v, Z])
        y = z_meas - z_pred  # innovation

        # ── Innovation covariance S and Kalman gain K ────────────
        S = H @ self.P @ H.T + self.R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fallback: pseudo-inverse
            K = self.P @ H.T @ np.linalg.pinv(S)

        # ── State and covariance update ──────────────────────────
        self.x = self.x + K @ y
        I_KH = np.eye(self.DIM_STATE) - K @ H
        # Joseph form for numerical stability
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Enforce minimum depth
        if self.x[2] < 0.05:
            self.x[2] = 0.05

    # ─────────────────────────────────────────────────────────────
    #  Update with 2D only (no depth measurement)
    # ─────────────────────────────────────────────────────────────

    def update_2d_only(self, u, v):
        """
        EKF update with 2D centroid only (no depth).
        Useful when depth estimation is skipped on tracking frames.
        Measurement: z = [u, v], dim=2.
        """
        if not self._initialised:
            return

        X, Y, Zs = self.x[0], self.x[1], max(self.x[2], 1e-4)

        u_pred = self.cam.fx * X / Zs + self.cam.cx
        v_pred = self.cam.fy * Y / Zs + self.cam.cy
        z_pred = np.array([u_pred, v_pred])

        H = np.zeros((2, self.DIM_STATE))
        H[0, 0] = self.cam.fx / Zs
        H[0, 2] = -self.cam.fx * X / (Zs ** 2)
        H[1, 1] = self.cam.fy / Zs
        H[1, 2] = -self.cam.fy * Y / (Zs ** 2)

        R_2d = self.R[:2, :2]
        z_meas = np.array([u, v])
        y = z_meas - z_pred

        S = H @ self.P @ H.T + R_2d
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)

        self.x = self.x + K @ y
        I_KH = np.eye(self.DIM_STATE) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_2d @ K.T

        if self.x[2] < 0.05:
            self.x[2] = 0.05

    # ─────────────────────────────────────────────────────────────
    #  Convenience
    # ─────────────────────────────────────────────────────────────

    def predicted_pixel(self):
        """Project current state estimate to pixel coordinates."""
        if not self._initialised:
            return None
        X, Y, Z = self.x[0], self.x[1], max(self.x[2], 1e-4)
        u, v = self.cam.project(X, Y, Z)
        return int(round(u)), int(round(v))

    def position_covariance(self):
        """Return 3×3 position covariance block."""
        return self.P[:3, :3].copy()

    def position_uncertainty(self):
        """Scalar uncertainty: sqrt of trace of position covariance."""
        return float(np.sqrt(np.trace(self.P[:3, :3])))


class PBVSController:
    """
    Position-Based Visual Servoing using EKF-filtered 3D pose.

    Generates Cartesian velocity commands to drive the object to
    a desired position in the camera frame (typically on the optical
    axis at a target depth).

    v_cmd = -lambda * (t_obj - t_desired)

    The robot-to-camera transform (R_cam, t_cam) maps camera-frame
    velocities to robot base-frame velocities.
    """

    def __init__(self,
                 gain=0.5,
                 target_depth=0.30,
                 max_vel=0.02,
                 dead_zone_m=0.005):
        """
        Parameters
        ----------
        gain : float
            Proportional gain lambda.
        target_depth : float
            Desired object depth Z* in camera frame (metres).
        max_vel : float
            Maximum commanded velocity per axis (m/s).
        dead_zone_m : float
            Position error below which we stop commanding motion.
        """
        self.gain = gain
        self.target_depth = target_depth
        self.max_vel = max_vel
        self.dead_zone_m = dead_zone_m

        # Camera-to-robot rotation (identity = camera aligned with robot)
        # Override after calibration for your setup
        self.R_cam_to_robot = np.eye(3)

    def compute_velocity(self, obj_position_cam):
        """
        Compute robot Cartesian velocity command.

        Parameters
        ----------
        obj_position_cam : array (3,)
            Object [X, Y, Z] in camera frame (metres).

        Returns
        -------
        v_robot : array (3,)
            Velocity command [vx, vy, vz] in robot base frame (m/s).
        err_norm : float
            Euclidean position error (metres).
        """
        # Desired position: on optical axis at target depth
        desired = np.array([0.0, 0.0, self.target_depth])
        error = obj_position_cam - desired
        err_norm = float(np.linalg.norm(error))

        if err_norm < self.dead_zone_m:
            return np.zeros(3), err_norm

        # PBVS control law: camera-frame velocity
        v_cam = -self.gain * error

        # Clamp per-axis
        v_cam = np.clip(v_cam, -self.max_vel, self.max_vel)

        # Transform to robot frame
        v_robot = self.R_cam_to_robot @ v_cam

        return v_robot, err_norm


class DepthScaler:
    """
    Convert relative monocular depth (from Depth Anything V2) to
    approximate metric depth using a simple affine model:

        Z_metric = scale * d_relative + offset

    Calibrated either:
      (a) from a known object size + distance, or
      (b) from the visual servoing Jacobian calibration baseline.

    Until calibrated, uses a default scale factor.
    """

    def __init__(self, default_scale=0.001, default_offset=0.0):
        self.scale = default_scale
        self.offset = default_offset
        self.calibrated = False

    def calibrate(self, d_relative, Z_metric_known):
        """
        One-point calibration: given a known metric depth and the
        corresponding relative depth reading, set scale + offset.
        Assumes offset ≈ 0 for single-point calibration.
        """
        if d_relative > 1e-6:
            self.scale = Z_metric_known / d_relative
            self.offset = 0.0
            self.calibrated = True

    def to_metric(self, d_relative):
        """Convert relative depth map or scalar to metric."""
        return self.scale * d_relative + self.offset

    def estimate_object_depth(self, depth_map_relative, mask, percentile=50):
        """
        Estimate metric depth of the object by taking the median
        relative depth under the mask and converting to metric.
        """
        m = mask > 0
        if not np.any(m):
            return None
        d_rel = float(np.percentile(depth_map_relative[m], percentile))
        return self.to_metric(d_rel)
