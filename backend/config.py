"""
Configuration module for Motion & Balance Loss Detection System.
Reads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    """Application configuration loaded from environment variables."""

    # MediaPipe Pose parameters
    min_detection_confidence: float = float(
        os.getenv("MIN_DETECTION_CONFIDENCE", "0.5")
    )
    min_tracking_confidence: float = float(
        os.getenv("MIN_TRACKING_CONFIDENCE", "0.5")
    )
    model_complexity: int = int(os.getenv("MODEL_COMPLEXITY", "1"))

    # Balance analysis thresholds
    trunk_angle_threshold: float = float(
        os.getenv("TRUNK_ANGLE_THRESHOLD", "45.0")
    )
    com_margin: float = float(os.getenv("COM_MARGIN", "0.05"))
    velocity_threshold: float = float(os.getenv("VELOCITY_THRESHOLD", "0.3"))

    # Fall detection
    fall_confirm_frames: int = int(os.getenv("FALL_CONFIRM_FRAMES", "3"))
    fall_cooldown_seconds: float = float(
        os.getenv("FALL_COOLDOWN_SECONDS", "3.0")
    )

    # Video source: "0" for webcam, file path, or RTSP URL
    video_source: str = os.getenv("VIDEO_SOURCE", "0")
    video_directory: str = os.getenv("VIDEO_DIRECTORY", "/app/videos")

    # Processing
    fps_limit: int = int(os.getenv("FPS_LIMIT", "30"))
    jpeg_quality: int = int(os.getenv("JPEG_QUALITY", "80"))

    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))

    def update(self, **kwargs) -> None:
        """Update configuration values at runtime."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                expected_type = type(getattr(self, key))
                setattr(self, key, expected_type(value))

    def to_dict(self) -> dict:
        """Return configuration as a dictionary."""
        return {
            "min_detection_confidence": self.min_detection_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
            "model_complexity": self.model_complexity,
            "trunk_angle_threshold": self.trunk_angle_threshold,
            "com_margin": self.com_margin,
            "velocity_threshold": self.velocity_threshold,
            "fall_confirm_frames": self.fall_confirm_frames,
            "fall_cooldown_seconds": self.fall_cooldown_seconds,
            "video_source": self.video_source,
            "video_directory": self.video_directory,
            "fps_limit": self.fps_limit,
            "jpeg_quality": self.jpeg_quality,
        }


# Global configuration instance
config = Config()
