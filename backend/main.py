"""
Main FastAPI Application — Motion & Balance Loss Detection System.

Provides REST API endpoints and WebSocket streaming for real-time
pose estimation and balance analysis.
"""

import asyncio
import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import config
from models import (
    AnalysisResult,
    BalanceState,
    ConfigResponse,
    ConfigUpdate,
    FrameData,
    LandmarkPoint,
    SystemStatus,
    VideoInfo,
)
from pose_estimator import PoseEstimator
from balance_analyzer import BalanceAnalyzer
from video_processor import VideoProcessor, list_video_files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# --- Global State ---
class AppState:
    """Application state shared across endpoints."""

    def __init__(self):
        self.pose_estimator: PoseEstimator = PoseEstimator()
        self.balance_analyzer: BalanceAnalyzer = BalanceAnalyzer()
        self.video_processor: VideoProcessor = VideoProcessor()
        self.is_running: bool = False
        self.current_state: BalanceState = BalanceState.NORMAL
        self.current_fps: float = 0.0
        self.total_frames: int = 0
        self.start_time: float = time.time()
        self.latest_result: dict = {}
        self.active_connections: List[WebSocket] = []
        self._processing_task: asyncio.Task = None


app_state = AppState()


# --- Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info("=" * 60)
    logger.info("Motion & Balance Loss Detection System v1.0")
    logger.info("=" * 60)
    logger.info(f"Configuration: {config.to_dict()}")

    # Try to open default video source
    if app_state.video_processor.open():
        logger.info(
            f"Video source ready: {app_state.video_processor.source_info}"
        )
        app_state.is_running = True
        # Start background processing
        app_state._processing_task = asyncio.create_task(
            _process_frames_loop()
        )
    else:
        logger.warning(
            "Could not open default video source. "
            "Upload a video or configure VIDEO_SOURCE."
        )

    yield

    # Shutdown
    logger.info("Shutting down...")
    app_state.is_running = False
    if app_state._processing_task:
        app_state._processing_task.cancel()
        try:
            await app_state._processing_task
        except asyncio.CancelledError:
            pass
    app_state.video_processor.release()
    app_state.pose_estimator.close()
    logger.info("Shutdown complete.")


# --- FastAPI App ---
app = FastAPI(
    title="Motion & Balance Loss Detection System",
    description=(
        "Real-time pose estimation and balance analysis system "
        "using MediaPipe Pose. Detects loss of balance and falls "
        "from video streams."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Background Frame Processing ---
async def _process_frames_loop():
    """
    Main processing loop that runs in the background.
    Reads frames, runs pose estimation, analyzes balance,
    and broadcasts results to WebSocket clients.
    """
    logger.info("Frame processing loop started.")

    while app_state.is_running:
        try:
            # Read frame (run in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            ret, frame = await loop.run_in_executor(
                None, app_state.video_processor.get_frame
            )

            if not ret or frame is None:
                await asyncio.sleep(0.1)
                continue

            # Run pose estimation
            landmarks = await loop.run_in_executor(
                None, app_state.pose_estimator.detect, frame
            )

            state = BalanceState.NORMAL
            trunk_angle = 0.0
            com_deviation = 0.0
            velocity = 0.0
            is_within_bos = True
            stability_ratio = 0.0
            landmark_points = None

            if landmarks:
                # Run balance analysis
                result = app_state.balance_analyzer.analyze(landmarks)
                state = result.state
                trunk_angle = result.trunk_angle
                com_deviation = result.com_deviation
                velocity = result.velocity
                is_within_bos = result.is_within_bos
                stability_ratio = result.stability_ratio

                # Draw skeleton on frame
                frame = await loop.run_in_executor(
                    None,
                    app_state.pose_estimator.draw_landmarks,
                    frame,
                    landmarks,
                    state.value,
                )

                # Convert landmarks to serializable format
                landmark_points = [
                    LandmarkPoint(
                        x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility
                    )
                    for lm in landmarks
                ]

            # Draw state overlay on frame
            frame = _draw_state_overlay(frame, state, trunk_angle)

            # Encode frame to JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
            _, buffer = cv2.imencode(".jpg", frame, encode_params)
            frame_b64 = base64.b64encode(buffer).decode("utf-8")

            # Update app state
            app_state.current_state = state
            app_state.current_fps = app_state.video_processor.fps
            app_state.total_frames += 1

            # Create frame data
            frame_data = FrameData(
                frame=frame_b64,
                state=state,
                trunk_angle=trunk_angle,
                com_deviation=com_deviation,
                velocity=velocity,
                is_within_bos=is_within_bos,
                stability_ratio=stability_ratio,
                fps=app_state.video_processor.fps,
                timestamp=datetime.now().isoformat(),
                landmarks=landmark_points,
            )

            app_state.latest_result = frame_data.model_dump()

            # Broadcast to all connected WebSocket clients
            await _broadcast_frame(frame_data)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in processing loop: {e}", exc_info=True)
            await asyncio.sleep(0.1)

    logger.info("Frame processing loop stopped.")


def _draw_state_overlay(
    frame: np.ndarray, state: BalanceState, trunk_angle: float
) -> np.ndarray:
    """Draw status text and alert overlay on the frame."""
    h, w, _ = frame.shape
    overlay = frame.copy()

    # State label colors (BGR)
    colors = {
        BalanceState.NORMAL: (118, 230, 0),      # Green
        BalanceState.IMBALANCED: (0, 171, 255),   # Amber
        BalanceState.FALL_DETECTED: (23, 23, 255), # Red
    }
    color = colors.get(state, (255, 255, 255))

    # Draw semi-transparent background bar at top
    bar_height = 50
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # State text
    state_text = f"Status: {state.value}"
    cv2.putText(
        frame, state_text, (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA,
    )

    # Trunk angle text
    angle_text = f"Angle: {trunk_angle:.1f} deg"
    cv2.putText(
        frame, angle_text, (w - 250, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA,
    )

    # Alert overlay for danger states
    if state == BalanceState.FALL_DETECTED:
        # Red border flash
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

        # Large warning text
        warning_text = "!! FALL DETECTED !!"
        text_size = cv2.getTextSize(
            warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3
        )[0]
        text_x = (w - text_size[0]) // 2
        text_y = h - 40
        # Background
        cv2.rectangle(
            frame,
            (text_x - 10, text_y - text_size[1] - 10),
            (text_x + text_size[0] + 10, text_y + 10),
            (0, 0, 180),
            -1,
        )
        cv2.putText(
            frame, warning_text, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA,
        )

    elif state == BalanceState.IMBALANCED:
        # Amber border
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 171, 255), 3)

        warning_text = "IMBALANCED"
        text_size = cv2.getTextSize(
            warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2
        )[0]
        text_x = (w - text_size[0]) // 2
        text_y = h - 40
        cv2.rectangle(
            frame,
            (text_x - 10, text_y - text_size[1] - 10),
            (text_x + text_size[0] + 10, text_y + 10),
            (0, 120, 180),
            -1,
        )
        cv2.putText(
            frame, warning_text, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA,
        )

    return frame


async def _broadcast_frame(frame_data: FrameData):
    """Send frame data to all connected WebSocket clients."""
    if not app_state.active_connections:
        return

    # Serialize once for all clients
    data = frame_data.model_dump()
    # Remove full landmarks from broadcast to reduce bandwidth
    # (clients can request full data via REST API)
    broadcast_data = {k: v for k, v in data.items() if k != "landmarks"}

    disconnected = []
    for ws in app_state.active_connections:
        try:
            await ws.send_json(broadcast_data)
        except Exception:
            disconnected.append(ws)

    # Clean up disconnected clients
    for ws in disconnected:
        if ws in app_state.active_connections:
            app_state.active_connections.remove(ws)


# === REST API Endpoints ===


@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    """Get current system status."""
    return SystemStatus(
        is_running=app_state.is_running,
        video_source=app_state.video_processor.source_info,
        current_state=app_state.current_state,
        fps=app_state.current_fps,
        total_frames_processed=app_state.total_frames,
        uptime_seconds=round(time.time() - app_state.start_time, 1),
    )


@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration values."""
    return ConfigResponse(**config.to_dict())


@app.put("/api/config")
async def update_config(update: ConfigUpdate):
    """Update configuration thresholds at runtime."""
    updates = update.model_dump(exclude_none=True)
    if not updates:
        return JSONResponse(
            status_code=400,
            content={"detail": "No configuration values provided"},
        )

    config.update(**updates)

    # Reinitialize pose estimator if detection/tracking confidence changed
    if "min_detection_confidence" in updates or "min_tracking_confidence" in updates:
        app_state.pose_estimator.reinitialize()

    logger.info(f"Configuration updated: {updates}")
    return {"message": "Configuration updated", "updated": updates}


@app.get("/api/videos", response_model=List[VideoInfo])
async def list_videos():
    """List available video files in the videos directory."""
    videos = list_video_files()
    return [VideoInfo(**v) for v in videos]


@app.post("/api/video/select/{filename}")
async def select_video(filename: str):
    """Select a video file to process."""
    filepath = os.path.join(config.video_directory, filename)
    if not os.path.exists(filepath):
        return JSONResponse(
            status_code=404,
            content={"detail": f"Video file not found: {filename}"},
        )

    # Stop current processing
    app_state.is_running = False
    if app_state._processing_task:
        app_state._processing_task.cancel()
        try:
            await app_state._processing_task
        except asyncio.CancelledError:
            pass

    # Reset analyzer state
    app_state.balance_analyzer.reset()

    # Switch to new source
    success = app_state.video_processor.set_source(filepath)
    if not success:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Failed to open video: {filename}"},
        )

    # Restart processing
    app_state.is_running = True
    app_state._processing_task = asyncio.create_task(_process_frames_loop())

    logger.info(f"Switched to video: {filename}")
    return {"message": f"Now processing: {filename}"}


@app.post("/api/video/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for analysis."""
    # Validate file extension
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".webm"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        return JSONResponse(
            status_code=400,
            content={
                "detail": (
                    f"Unsupported file type: {ext}. "
                    f"Allowed: {', '.join(allowed_extensions)}"
                )
            },
        )

    # Ensure video directory exists
    os.makedirs(config.video_directory, exist_ok=True)

    # Save file
    filepath = os.path.join(config.video_directory, file.filename)
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    size_mb = len(content) / (1024 * 1024)
    logger.info(f"Video uploaded: {file.filename} ({size_mb:.2f} MB)")

    return {
        "message": f"Video uploaded: {file.filename}",
        "filename": file.filename,
        "size_mb": round(size_mb, 2),
    }


@app.post("/api/start")
async def start_processing():
    """Start or resume video processing."""
    if app_state.is_running:
        return {"message": "Already running"}

    if not app_state.video_processor.is_opened:
        if not app_state.video_processor.open():
            return JSONResponse(
                status_code=500,
                content={"detail": "Cannot open video source"},
            )

    app_state.is_running = True
    app_state._processing_task = asyncio.create_task(_process_frames_loop())
    logger.info("Processing started")
    return {"message": "Processing started"}


@app.post("/api/stop")
async def stop_processing():
    """Stop video processing."""
    if not app_state.is_running:
        return {"message": "Already stopped"}

    app_state.is_running = False
    if app_state._processing_task:
        app_state._processing_task.cancel()
        try:
            await app_state._processing_task
        except asyncio.CancelledError:
            pass

    logger.info("Processing stopped")
    return {"message": "Processing stopped"}


@app.get("/api/latest")
async def get_latest_result():
    """Get the latest analysis result (without frame data)."""
    if not app_state.latest_result:
        return {"message": "No data available yet"}

    result = {
        k: v for k, v in app_state.latest_result.items() if k != "frame"
    }
    return result


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# === WebSocket Endpoints ===


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time video streaming.
    Sends processed frames with analysis data as JSON.
    """
    await websocket.accept()
    app_state.active_connections.append(websocket)
    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info(
        f"WebSocket client connected: {client_host} "
        f"(total: {len(app_state.active_connections)})"
    )

    try:
        while True:
            # Keep connection alive, handle client messages
            data = await websocket.receive_text()

            # Handle control messages from client
            try:
                import json
                msg = json.loads(data)
                if msg.get("action") == "ping":
                    await websocket.send_json({"action": "pong"})
                elif msg.get("action") == "get_landmarks":
                    # Send full landmark data on request
                    if app_state.latest_result:
                        await websocket.send_json(app_state.latest_result)
            except (json.JSONDecodeError, Exception):
                pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in app_state.active_connections:
            app_state.active_connections.remove(websocket)
        logger.info(
            f"WebSocket client disconnected: {client_host} "
            f"(total: {len(app_state.active_connections)})"
        )


# === Entry Point ===
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=False,
        log_level="info",
    )
