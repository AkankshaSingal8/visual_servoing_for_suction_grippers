#!/usr/bin/env python3
"""
RobotControllerExt — RobotController subclass with wrist tilt correction.

Adds proportional surface-normal feedback so the suction face aligns with
the box surface during approach. Tilt estimates come from estimate_near_tilt
(roll_deg, pitch_deg) and are converted to relative wrist rotations sent via
set_tool_position(relative=True).
"""

from __future__ import annotations

import logging

try:
    from foundation_model.servo_pipeline_sam3 import RobotController
except ImportError:
    from servo_pipeline_sam3 import RobotController

log = logging.getLogger("robot_controller_ext")


class RobotControllerExt(RobotController):
    """
    Extends RobotController with wrist tilt correction.
    Uses surface normal feedback (roll_deg, pitch_deg) to rotate the EE
    wrist during approach so the suction face aligns with the box surface.
    """

    TILT_DEAD_ZONE_DEG = 1.0
    TILT_MAX_STEP_DEG  = 3.0
    TILT_MIN_INLIERS   = 500
    TILT_GAIN          = 0.6

    def __init__(self, ip: str):
        super().__init__(ip)
        self.tilt_correction_enabled = True

    def correct_tilt(self, roll_deg: float, pitch_deg: float, n_inliers: int) -> bool:
        """
        Apply wrist correction for measured surface tilt.

        roll_deg: rotation around camera X axis (from estimate_near_tilt)
        pitch_deg: rotation around camera Y axis
        n_inliers: RANSAC inlier count (confidence gate)

        Returns True if correction was applied, False if skipped (dead-zone, low confidence, disabled).
        """
        if not self.enabled:
            return False

        if n_inliers < self.TILT_MIN_INLIERS:
            log.debug(
                "correct_tilt: skipped (n_inliers=%d < %d)",
                n_inliers, self.TILT_MIN_INLIERS,
            )
            return False

        if abs(roll_deg) < self.TILT_DEAD_ZONE_DEG and abs(pitch_deg) < self.TILT_DEAD_ZONE_DEG:
            log.debug(
                "correct_tilt: skipped (roll=%.2f° pitch=%.2f° both within dead-zone ±%.1f°)",
                roll_deg, pitch_deg, self.TILT_DEAD_ZONE_DEG,
            )
            return False

        delta_roll  = -self.TILT_GAIN * roll_deg
        delta_pitch = -self.TILT_GAIN * pitch_deg

        delta_roll  = max(-self.TILT_MAX_STEP_DEG, min(self.TILT_MAX_STEP_DEG, delta_roll))
        delta_pitch = max(-self.TILT_MAX_STEP_DEG, min(self.TILT_MAX_STEP_DEG, delta_pitch))

        log.debug(
            "correct_tilt: roll_in=%.2f° pitch_in=%.2f°  "
            "delta_roll=%.2f° delta_pitch=%.2f°  inliers=%d",
            roll_deg, pitch_deg, delta_roll, delta_pitch, n_inliers,
        )

        try:
            self._arm.set_tool_position(
                roll=delta_roll,
                pitch=delta_pitch,
                relative=True,
                wait=False,
                speed=30,
            )
        except Exception as exc:
            log.error("correct_tilt: set_tool_position failed: %s", exc)
            return False

        return True

    def correct_tilt_dry(self, roll_deg: float, pitch_deg: float, n_inliers: int) -> dict:
        """
        Compute what correct_tilt would do without moving the robot.
        Returns dict with: would_apply (bool), delta_roll, delta_pitch, reason (str).
        """
        if not self.enabled:
            return dict(would_apply=False, delta_roll=0.0, delta_pitch=0.0,
                        reason="servo disabled")

        if n_inliers < self.TILT_MIN_INLIERS:
            return dict(would_apply=False, delta_roll=0.0, delta_pitch=0.0,
                        reason=f"low confidence (n_inliers={n_inliers} < {self.TILT_MIN_INLIERS})")

        if abs(roll_deg) < self.TILT_DEAD_ZONE_DEG and abs(pitch_deg) < self.TILT_DEAD_ZONE_DEG:
            return dict(would_apply=False, delta_roll=0.0, delta_pitch=0.0,
                        reason=f"dead-zone (|roll|={abs(roll_deg):.2f}° |pitch|={abs(pitch_deg):.2f}° < {self.TILT_DEAD_ZONE_DEG}°)")

        delta_roll  = -self.TILT_GAIN * roll_deg
        delta_pitch = -self.TILT_GAIN * pitch_deg
        delta_roll  = max(-self.TILT_MAX_STEP_DEG, min(self.TILT_MAX_STEP_DEG, delta_roll))
        delta_pitch = max(-self.TILT_MAX_STEP_DEG, min(self.TILT_MAX_STEP_DEG, delta_pitch))

        return dict(would_apply=True, delta_roll=delta_roll, delta_pitch=delta_pitch,
                    reason="correction computed")
