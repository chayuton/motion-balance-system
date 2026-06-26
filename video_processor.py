"""
Video Processor Module — Handles video input from various sources.
Supports webcam, video files, and RTSP streams.
"""

import os
import cv2
import time
import logging
from typing import Optional, Tuple, List

from config import config

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Manages video capture from multiple source types.
    Provides frame-by-frame access with FPS control.
    """

    def __init__(self, source: Optional[str] = None):
        """
        Initialize video processor.

        Args:
            source: Video source - can be:
                - Integer string ("0", "1") for webcam index
                - File path for video file
                - RTSP URL for IP camera
                - None to use config default
        """
        self._source = source or config.video_source
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_file: bool = False
        self._frame_count: int = 0
        self._start_time: float = 0.0
        self._last_frame_time: float = 0.0
        self._fps_actual: float = 0.0
        self._fps_counter: int = 0
        self._fps_timer: float = 0.0

    def open(self) -> bool:
        """
        Open the video source.

        Returns:
            True if successfully opened, False otherwise
        """
        try:
            source = self._resolve_source(self._source)

            if isinstance(source, int):
                self._cap = cv2.VideoCapture(source)
                self._is_file = False
                logger.info(f"Opened webcam at index {source}")
            elif source.startswith("rtsp://") or source.startswith("http://"):
                self._cap = cv2.VideoCapture(source)
                self._is_file = False
                logger.info(f"Opened stream: {source}")
            else:
                if not os.path.exists(source):
                    logger.error(f"Video file not found: {source}")
                    return False
                self._cap = cv2.VideoCapture(source)
                self._is_file = True
                logger.info(f"Opened video file: {source}")

            if not self._cap.isOpened():
                logger.error(f"Failed to open video source: {self._source}")
                return False

            self._start_time = time.time()
            self._fps_timer = time.time()
            return True

        except Exception as e:
            logger.error(f"Error opening video source: {e}")
            return False

    def _resolve_source(self, source: str):
        """Resolve the source string to appropriate type."""
        # Check if it's a webcam index
        try:
            return int(source)
        except ValueError:
            pass

        # Check if it's a path relative to video directory
        if not os.path.isabs(source) and not source.startswith(("rtsp://", "http://")):
            full_path = os.path.join(config.video_directory, source)
            if os.path.exists(full_path):
                return full_path

        return source

    def get_frame(self) -> Tuple[bool, Optional["np.ndarray"]]:
        """
        Read the next frame from the video source.
        Applies FPS limiting to control processing rate.

        Returns:
            Tuple of (success, frame) where frame is BGR numpy array
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        # FPS limiting
        if config.fps_limit > 0:
            target_interval = 1.0 / config.fps_limit
            elapsed = time.time() - self._last_frame_time
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)

        ret, frame = self._cap.read()

        if not ret:
            if self._is_file:
                # Loop video files
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    return False, None
                logger.info("Video file looped back to beginning")
            else:
                return False, None

        self._frame_count += 1
        self._last_frame_time = time.time()
        self._update_fps()

        return True, frame

    def _update_fps(self) -> None:
        """Calculate actual processing FPS."""
        self._fps_counter += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self._fps_actual = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_timer = time.time()

    def set_source(self, source: str) -> bool:
        """
        Change the video source.

        Args:
            source: New video source string

        Returns:
            True if new source opened successfully
        """
        self.release()
        self._source = source
        self._frame_count = 0
        return self.open()

    def release(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Video source released")

    @property
    def is_opened(self) -> bool:
        """Check if video source is currently open."""
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        """Get current actual processing FPS."""
        return round(self._fps_actual, 1)

    @property
    def frame_count(self) -> int:
        """Get total number of frames processed."""
        return self._frame_count

    @property
    def source_fps(self) -> float:
        """Get the source video's native FPS."""
        if self._cap is not None:
            return self._cap.get(cv2.CAP_PROP_FPS)
        return 0.0

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Get frame dimensions (width, height)."""
        if self._cap is not None:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (0, 0)

    @property
    def source_info(self) -> str:
        """Get human-readable source description."""
        return self._source

    def __del__(self):
        """Cleanup on deletion."""
        self.release()


def list_video_files(directory: Optional[str] = None) -> List[dict]:
    """
    List available video files in the specified directory.

    Args:
        directory: Path to scan for videos (defaults to config.video_directory)

    Returns:
        List of dicts with filename, size_mb, and path
    """
    video_dir = directory or config.video_directory
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    videos = []

    if not os.path.exists(video_dir):
        logger.warning(f"Video directory does not exist: {video_dir}")
        return videos

    for filename in sorted(os.listdir(video_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in video_extensions:
            filepath = os.path.join(video_dir, filename)
            size_bytes = os.path.getsize(filepath)
            videos.append({
                "filename": filename,
                "size_mb": round(size_bytes / (1024 * 1024), 2),
                "path": filepath,
            })

    return videos
