"""
Pose Estimator Module — MediaPipe Pose wrapper for skeleton detection.
Handles pose landmark detection and visualization on video frames.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, List, Tuple

from config import config


class PoseEstimator:
    """Wrapper around MediaPipe Pose for body landmark detection."""

    # Landmark indices for key body points
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    def __init__(self):
        """Initialize MediaPipe Pose with configuration parameters."""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=config.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )

        # Custom drawing specs for skeleton visualization
        self._landmark_spec = self.mp_drawing.DrawingSpec(
            color=(0, 230, 118),  # Green (#00e676)
            thickness=2,
            circle_radius=3,
        )
        self._connection_spec = self.mp_drawing.DrawingSpec(
            color=(100, 200, 255),  # Light cyan
            thickness=2,
            circle_radius=1,
        )

    def detect(self, frame: np.ndarray) -> Optional[list]:
        """
        Detect pose landmarks in a frame.

        Args:
            frame: BGR image (numpy array from OpenCV)

        Returns:
            List of landmark objects with (x, y, z, visibility) or None
        """
        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            return results.pose_landmarks.landmark
        return None

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: list,
        state: str = "Normal",
    ) -> np.ndarray:
        """
        Draw skeleton overlay on the frame with state-based coloring.

        Args:
            frame: BGR image to draw on
            landmarks: List of detected landmarks
            state: Current balance state for color coding

        Returns:
            Frame with skeleton drawn
        """
        annotated = frame.copy()
        h, w, _ = annotated.shape

        # Choose colors based on state
        if state == "Fall_Detected":
            landmark_color = (23, 23, 255)  # Red (#ff1744)
            connection_color = (23, 23, 255)
        elif state == "Imbalanced":
            landmark_color = (0, 171, 255)  # Amber (#ffab00)
            connection_color = (0, 171, 255)
        else:
            landmark_color = (0, 230, 118)  # Green (#00e676)
            connection_color = (200, 200, 100)  # Light cyan

        landmark_spec = self.mp_drawing.DrawingSpec(
            color=landmark_color, thickness=2, circle_radius=4
        )
        connection_spec = self.mp_drawing.DrawingSpec(
            color=connection_color, thickness=2
        )

        # Create a NormalizedLandmarkList for drawing
        landmark_list = mp.framework.formats.landmark_pb2.NormalizedLandmarkList()
        for lm in landmarks:
            landmark_proto = landmark_list.landmark.add()
            landmark_proto.x = lm.x
            landmark_proto.y = lm.y
            landmark_proto.z = lm.z
            landmark_proto.visibility = lm.visibility

        self.mp_drawing.draw_landmarks(
            annotated,
            landmark_list,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_spec,
            connection_spec,
        )

        # Draw midpoints for trunk analysis visualization
        self._draw_trunk_line(annotated, landmarks, landmark_color)
        self._draw_bos_region(annotated, landmarks, connection_color)

        return annotated

    def _draw_trunk_line(
        self,
        frame: np.ndarray,
        landmarks: list,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw the trunk centerline (shoulder midpoint to hip midpoint)."""
        h, w, _ = frame.shape

        ls = landmarks[self.LEFT_SHOULDER]
        rs = landmarks[self.RIGHT_SHOULDER]
        lh = landmarks[self.LEFT_HIP]
        rh = landmarks[self.RIGHT_HIP]

        # Check visibility
        if min(ls.visibility, rs.visibility, lh.visibility, rh.visibility) < 0.5:
            return

        mid_shoulder = (
            int((ls.x + rs.x) / 2 * w),
            int((ls.y + rs.y) / 2 * h),
        )
        mid_hip = (
            int((lh.x + rh.x) / 2 * w),
            int((lh.y + rh.y) / 2 * h),
        )

        # Draw trunk centerline (thicker, dashed effect)
        cv2.line(frame, mid_shoulder, mid_hip, color, 3, cv2.LINE_AA)

        # Draw midpoint markers
        cv2.circle(frame, mid_shoulder, 6, color, -1, cv2.LINE_AA)
        cv2.circle(frame, mid_hip, 6, color, -1, cv2.LINE_AA)

    def _draw_bos_region(
        self,
        frame: np.ndarray,
        landmarks: list,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw the Base of Support region between feet."""
        h, w, _ = frame.shape

        la = landmarks[self.LEFT_ANKLE]
        ra = landmarks[self.RIGHT_ANKLE]

        if min(la.visibility, ra.visibility) < 0.5:
            return

        left_pt = (int(la.x * w), int(la.y * h))
        right_pt = (int(ra.x * w), int(ra.y * h))

        # Draw BoS line between ankles
        cv2.line(frame, left_pt, right_pt, color, 2, cv2.LINE_AA)

        # Draw CoM projection (hip midpoint X at ankle Y level)
        lh = landmarks[self.LEFT_HIP]
        rh = landmarks[self.RIGHT_HIP]
        if min(lh.visibility, rh.visibility) >= 0.5:
            com_x = int((lh.x + rh.x) / 2 * w)
            ankle_y = int((la.y + ra.y) / 2 * h)
            cv2.circle(frame, (com_x, ankle_y), 8, (0, 255, 255), 2, cv2.LINE_AA)

    def reinitialize(self) -> None:
        """Reinitialize pose model with current config values."""
        self.pose.close()
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=config.model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.pose.close()
