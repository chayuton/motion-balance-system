"""
Balance Analysis Engine — Core algorithms for detecting loss of balance and falls.

Implements three detection rules:
1. Trunk Angle: Measures trunk tilt relative to vertical
2. Center of Mass vs Base of Support: Checks if CoM is within foot boundaries
3. Y-Axis Velocity: Detects rapid downward movement (fall signature)
"""

import time
import numpy as np
from collections import deque
from typing import Optional, Dict, Tuple

from config import config
from models import BalanceState, AnalysisResult


class BalanceAnalyzer:
    """
    Analyzes body landmarks to determine balance state.
    Combines multiple heuristic rules for robust detection.
    """

    # MediaPipe Pose landmark indices
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    def __init__(self):
        """Initialize the balance analyzer with tracking state."""
        self._prev_nose_y: Optional[float] = None
        self._prev_timestamp: Optional[float] = None
        self._fall_confirm_count: int = 0
        self._last_fall_time: float = 0.0
        self._velocity_history: deque = deque(maxlen=10)
        self._state_history: deque = deque(maxlen=30)
        self._current_state: BalanceState = BalanceState.NORMAL

    def calculate_trunk_angle(self, landmarks: list) -> float:
        """
        Calculate the trunk tilt angle relative to vertical.

        Computes the angle between the trunk centerline (shoulder midpoint
        to hip midpoint) and the vertical axis. In image coordinates,
        Y increases downward, so vertical = (0, 1).

        Args:
            landmarks: List of 33 MediaPipe pose landmarks

        Returns:
            Angle in degrees (0 = perfectly upright)
        """
        ls = landmarks[self.LEFT_SHOULDER]
        rs = landmarks[self.RIGHT_SHOULDER]
        lh = landmarks[self.LEFT_HIP]
        rh = landmarks[self.RIGHT_HIP]

        # Check landmark visibility
        if min(ls.visibility, rs.visibility, lh.visibility, rh.visibility) < 0.3:
            return 0.0

        # Midpoints
        shoulder_mid_x = (ls.x + rs.x) / 2
        shoulder_mid_y = (ls.y + rs.y) / 2
        hip_mid_x = (lh.x + rh.x) / 2
        hip_mid_y = (lh.y + rh.y) / 2

        # Vector from hip to shoulder
        dx = shoulder_mid_x - hip_mid_x
        dy = shoulder_mid_y - hip_mid_y  # Negative because shoulder is above hip

        # Angle with vertical axis
        # In image coords, vertical up = (0, -1)
        # Using atan2(horizontal_component, vertical_component)
        angle_rad = np.arctan2(abs(dx), abs(dy))
        angle_deg = np.degrees(angle_rad)

        return round(angle_deg, 2)

    def check_com_vs_bos(self, landmarks: list) -> Dict:
        """
        Check if Center of Mass is within the Base of Support.

        Uses hip midpoint as approximate CoM and ankle positions
        to define the BoS boundary.

        Args:
            landmarks: List of 33 MediaPipe pose landmarks

        Returns:
            Dictionary with CoM analysis results
        """
        lh = landmarks[self.LEFT_HIP]
        rh = landmarks[self.RIGHT_HIP]
        la = landmarks[self.LEFT_ANKLE]
        ra = landmarks[self.RIGHT_ANKLE]

        # Check visibility
        if min(lh.visibility, rh.visibility, la.visibility, ra.visibility) < 0.3:
            return {
                "com_x": 0.0,
                "bos_center_x": 0.0,
                "bos_width": 0.0,
                "deviation": 0.0,
                "is_within_bos": True,
                "stability_ratio": 0.0,
            }

        # Approximate Center of Mass as hip midpoint (X-axis)
        com_x = (lh.x + rh.x) / 2

        # Base of Support boundaries (ankle positions)
        bos_left = min(la.x, ra.x)
        bos_right = max(la.x, ra.x)
        bos_center = (bos_left + bos_right) / 2
        bos_width = bos_right - bos_left

        # Add margin to BoS
        margin = config.com_margin
        effective_bos_left = bos_left - margin
        effective_bos_right = bos_right + margin

        # Calculate deviation
        deviation = abs(com_x - bos_center)
        is_within = effective_bos_left <= com_x <= effective_bos_right

        # Stability ratio: 0 = perfectly centered, >1 = outside BoS
        half_bos = (bos_width / 2 + margin) if bos_width > 0 else 0.01
        stability_ratio = deviation / half_bos

        return {
            "com_x": round(com_x, 4),
            "bos_center_x": round(bos_center, 4),
            "bos_width": round(bos_width, 4),
            "deviation": round(deviation, 4),
            "is_within_bos": is_within,
            "stability_ratio": round(stability_ratio, 4),
        }

    def detect_fall_velocity(
        self, landmarks: list, current_time: float
    ) -> Dict:
        """
        Detect rapid downward movement indicating a fall.

        Tracks the Y-position of the nose landmark across frames
        and calculates velocity. Rapid downward movement exceeding
        the threshold suggests a fall event.

        Args:
            landmarks: List of 33 MediaPipe pose landmarks
            current_time: Current timestamp in seconds

        Returns:
            Dictionary with velocity analysis results
        """
        nose = landmarks[self.NOSE]

        result = {
            "velocity": 0.0,
            "is_rapid_descent": False,
        }

        if nose.visibility < 0.3:
            return result

        if self._prev_nose_y is not None and self._prev_timestamp is not None:
            dt = current_time - self._prev_timestamp
            if dt > 0:
                # Velocity in normalized coordinates per second
                # Positive velocity = moving downward (Y increases down)
                velocity = (nose.y - self._prev_nose_y) / dt
                result["velocity"] = round(velocity, 4)

                # Check against threshold
                result["is_rapid_descent"] = velocity > config.velocity_threshold

                self._velocity_history.append(velocity)

        self._prev_nose_y = nose.y
        self._prev_timestamp = current_time

        return result

    def analyze(self, landmarks: list) -> AnalysisResult:
        """
        Perform complete balance analysis combining all three rules.

        State transitions:
        - Normal → Imbalanced: trunk angle exceeds threshold OR CoM outside BoS
        - Normal → Fall_Detected: rapid Y-velocity detected
        - Imbalanced → Fall_Detected: rapid Y-velocity detected
        - Imbalanced → Normal: all metrics within thresholds
        - Fall_Detected → Normal: after cooldown period

        Args:
            landmarks: List of 33 MediaPipe pose landmarks

        Returns:
            AnalysisResult with state and all metric values
        """
        current_time = time.time()

        # Rule 1: Trunk angle
        trunk_angle = self.calculate_trunk_angle(landmarks)

        # Rule 2: CoM vs BoS
        com_result = self.check_com_vs_bos(landmarks)

        # Rule 3: Fall velocity
        velocity_result = self.detect_fall_velocity(landmarks, current_time)

        # --- State machine logic ---
        new_state = BalanceState.NORMAL

        # Check for fall (highest priority)
        if velocity_result["is_rapid_descent"]:
            self._fall_confirm_count += 1
            if self._fall_confirm_count >= config.fall_confirm_frames:
                new_state = BalanceState.FALL_DETECTED
                self._last_fall_time = current_time
        else:
            self._fall_confirm_count = max(0, self._fall_confirm_count - 1)

        # Check for imbalance (if not already falling)
        if new_state != BalanceState.FALL_DETECTED:
            is_trunk_tilted = trunk_angle > config.trunk_angle_threshold
            is_com_outside = not com_result["is_within_bos"]

            if is_trunk_tilted or is_com_outside:
                new_state = BalanceState.IMBALANCED

        # Fall cooldown: maintain Fall_Detected state for cooldown period
        if (
            self._current_state == BalanceState.FALL_DETECTED
            and (current_time - self._last_fall_time) < config.fall_cooldown_seconds
        ):
            new_state = BalanceState.FALL_DETECTED

        self._current_state = new_state
        self._state_history.append(new_state)

        return AnalysisResult(
            state=new_state,
            trunk_angle=trunk_angle,
            com_deviation=com_result["deviation"],
            velocity=velocity_result["velocity"],
            is_within_bos=com_result["is_within_bos"],
            stability_ratio=com_result["stability_ratio"],
        )

    def reset(self) -> None:
        """Reset analyzer state (e.g., when switching video sources)."""
        self._prev_nose_y = None
        self._prev_timestamp = None
        self._fall_confirm_count = 0
        self._last_fall_time = 0.0
        self._velocity_history.clear()
        self._state_history.clear()
        self._current_state = BalanceState.NORMAL

    @property
    def current_state(self) -> BalanceState:
        """Get the current balance state."""
        return self._current_state
