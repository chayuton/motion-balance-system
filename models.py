"""
Pydantic models for API request/response schemas.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class BalanceState(str, Enum):
    """Possible balance states detected by the system."""
    NORMAL = "Normal"
    IMBALANCED = "Imbalanced"
    FALL_DETECTED = "Fall_Detected"


class LandmarkPoint(BaseModel):
    """A single body landmark point."""
    x: float = Field(..., description="Normalized X coordinate (0.0 - 1.0)")
    y: float = Field(..., description="Normalized Y coordinate (0.0 - 1.0)")
    z: float = Field(..., description="Depth relative to hip midpoint")
    visibility: float = Field(
        ..., description="Landmark visibility confidence (0.0 - 1.0)"
    )


class AnalysisResult(BaseModel):
    """Result of balance analysis for a single frame."""
    state: BalanceState = Field(..., description="Current balance state")
    trunk_angle: float = Field(
        ..., description="Trunk tilt angle in degrees (0 = upright)"
    )
    com_deviation: float = Field(
        ...,
        description="Center of Mass deviation from Base of Support center",
    )
    velocity: float = Field(
        ..., description="Y-axis velocity of head (nose landmark)"
    )
    is_within_bos: bool = Field(
        ..., description="Whether CoM is within Base of Support"
    )
    stability_ratio: float = Field(
        ...,
        description="Stability ratio (0 = perfect balance, >1 = unstable)",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of analysis",
    )


class FrameData(BaseModel):
    """Complete frame data sent via WebSocket."""
    frame: str = Field(..., description="Base64-encoded JPEG frame")
    state: BalanceState = Field(..., description="Current balance state")
    trunk_angle: float = Field(..., description="Trunk angle in degrees")
    com_deviation: float = Field(..., description="CoM deviation value")
    velocity: float = Field(..., description="Y-axis velocity")
    is_within_bos: bool = Field(True, description="CoM within BoS")
    stability_ratio: float = Field(0.0, description="Stability ratio")
    fps: float = Field(..., description="Current processing FPS")
    timestamp: str = Field(..., description="ISO format timestamp")
    landmarks: Optional[List[LandmarkPoint]] = Field(
        None, description="List of 33 pose landmarks"
    )


class SystemStatus(BaseModel):
    """System operational status."""
    is_running: bool = Field(..., description="Whether processing is active")
    video_source: str = Field(..., description="Current video source")
    current_state: BalanceState = Field(
        ..., description="Latest balance state"
    )
    fps: float = Field(..., description="Current processing FPS")
    total_frames_processed: int = Field(
        ..., description="Total frames processed since start"
    )
    uptime_seconds: float = Field(..., description="System uptime in seconds")


class ConfigUpdate(BaseModel):
    """Request model for updating configuration at runtime."""
    min_detection_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="MediaPipe detection confidence"
    )
    min_tracking_confidence: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="MediaPipe tracking confidence"
    )
    trunk_angle_threshold: Optional[float] = Field(
        None, ge=0.0, le=90.0, description="Trunk angle threshold in degrees"
    )
    com_margin: Optional[float] = Field(
        None, ge=0.0, le=0.5, description="CoM margin for BoS check"
    )
    velocity_threshold: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="Y-velocity fall threshold"
    )


class ConfigResponse(BaseModel):
    """Response model for configuration values."""
    min_detection_confidence: float
    min_tracking_confidence: float
    model_complexity: int
    trunk_angle_threshold: float
    com_margin: float
    velocity_threshold: float
    fall_confirm_frames: int
    fall_cooldown_seconds: float
    video_source: str
    video_directory: str
    fps_limit: int
    jpeg_quality: int


class VideoInfo(BaseModel):
    """Information about an available video file."""
    filename: str
    size_mb: float
    path: str
