# 🏃 Motion & Balance Loss Detection System

ระบบตรวจจับการเคลื่อนไหวและวิเคราะห์การเสียการทรงตัว (Motion & Balance Loss Detection System) v1.0

Real-time pose estimation and balance analysis system using **MediaPipe Pose**, **OpenCV**, and **FastAPI**. Detects loss of balance and potential falls from video streams or files.

---

## ✨ Features

- **Real-time Pose Estimation**: Detects 33 body landmarks using Google MediaPipe Pose
- **Balance Analysis**: Three-rule detection system:
  - 📐 **Trunk Angle**: Measures body tilt relative to vertical
  - ⚖️ **Center of Mass vs Base of Support**: Checks if CoM is within foot boundaries
  - ⬇️ **Fall Velocity**: Detects rapid downward head movement
- **Three Detection States**: `Normal` → `Imbalanced` → `Fall_Detected`
- **Web Dashboard**: Premium dark-themed real-time visualization
- **REST API + WebSocket**: For external system integration
- **Docker Ready**: One command deployment with `docker-compose`

---

## 🚀 Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/install/)

### 1. Clone / Download

```bash
cd "movement data"
```

### 2. Add Test Videos (Optional)

Place your video files (MP4, AVI, MOV) in the `videos/` directory:

```bash
cp /path/to/your/video.mp4 ./videos/
```

### 3. Build & Run

```bash
docker-compose up -d --build
```

### 4. Access the System

| Service | URL | Description |
|---------|-----|-------------|
| **Web Dashboard** | http://localhost:8080 | Real-time visualization |
| **API Documentation** | http://localhost:8000/docs | Interactive Swagger UI |
| **Health Check** | http://localhost:8000/health | System health status |

---

## 📡 API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/status` | Current system status (state, FPS, frame count) |
| `GET` | `/api/config` | Current configuration values |
| `PUT` | `/api/config` | Update thresholds at runtime |
| `GET` | `/api/videos` | List available video files |
| `POST` | `/api/video/select/{filename}` | Select a video to process |
| `POST` | `/api/video/upload` | Upload a video file |
| `POST` | `/api/start` | Start processing |
| `POST` | `/api/stop` | Stop processing |
| `GET` | `/api/latest` | Latest analysis result (no frame) |
| `GET` | `/health` | Health check |

### WebSocket

Connect to `ws://localhost:8000/ws/stream` to receive real-time data:

```json
{
  "frame": "<base64 JPEG>",
  "state": "Normal",
  "trunk_angle": 5.2,
  "com_deviation": 0.03,
  "velocity": 0.01,
  "is_within_bos": true,
  "stability_ratio": 0.15,
  "fps": 28.5,
  "timestamp": "2026-06-20T15:54:59"
}
```

---

## ⚙️ Configuration

All parameters are configurable via **environment variables** in `docker-compose.yml`:

### MediaPipe Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `MIN_DETECTION_CONFIDENCE` | `0.5` | Pose detection confidence (0.0-1.0) |
| `MIN_TRACKING_CONFIDENCE` | `0.5` | Pose tracking confidence (0.0-1.0) |
| `MODEL_COMPLEXITY` | `1` | Model complexity (0, 1, or 2) |

### Balance Analysis Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `TRUNK_ANGLE_THRESHOLD` | `45.0` | Trunk tilt threshold in degrees |
| `COM_MARGIN` | `0.05` | CoM margin for BoS check |
| `VELOCITY_THRESHOLD` | `0.3` | Y-velocity fall threshold |
| `FALL_CONFIRM_FRAMES` | `3` | Frames to confirm fall |
| `FALL_COOLDOWN_SECONDS` | `3.0` | Cooldown after fall detection |

### Processing

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_SOURCE` | `0` | Video source (webcam index, file path, or RTSP URL) |
| `FPS_LIMIT` | `30` | Maximum processing FPS |
| `JPEG_QUALITY` | `80` | JPEG encoding quality (1-100) |

---

## 🐳 Docker Architecture

```
┌─────────────────────────────────────────────────┐
│                Docker Network                    │
│                                                  │
│  ┌──────────────────────┐  ┌──────────────────┐ │
│  │   pose-engine:8000   │  │   web-ui:80      │ │
│  │                      │  │                   │ │
│  │  FastAPI + MediaPipe  │◄─│  Nginx + HTML/   │ │
│  │  + OpenCV + Balance   │  │  CSS/JS          │ │
│  │  Analysis             │  │                   │ │
│  └──────────┬───────────┘  └───────────────────┘ │
│             │                                     │
│        ┌────┴────┐                                │
│        │ /videos │ ← Docker Volume (Host mount)   │
│        └─────────┘                                │
└─────────────────────────────────────────────────┘
         ↕ Port 8000          ↕ Port 8080
    API / WebSocket         Web Dashboard
```

### Resource Limits

| Container | CPU | Memory |
|-----------|-----|--------|
| pose-engine | 2.0 cores | 2 GB |
| web-ui | 0.5 cores | 512 MB |

---

## 🔧 Development

### Run Without Docker

```bash
# Backend
cd backend
pip install -r requirements.txt
python main.py

# Frontend (serve with any HTTP server)
cd frontend
python -m http.server 8080
```

### Webcam Access (Linux Only)

Uncomment in `docker-compose.yml`:

```yaml
devices:
  - /dev/video0:/dev/video0
```

### RTSP Camera

Set the `VIDEO_SOURCE` environment variable:

```yaml
environment:
  - VIDEO_SOURCE=rtsp://admin:password@192.168.1.100:554/stream
```

---

## 📊 Detection Algorithm

### State Machine

```
        ┌─────────────────────────┐
        │                         │
        ▼                         │
   ┌─────────┐  angle/CoM   ┌────┴──────┐
   │ Normal  │──────────────►│Imbalanced │
   │         │◄──────────────│           │
   └────┬────┘  recovered    └─────┬─────┘
        │                          │
        │  velocity                │ velocity
        │                          │
        ▼                          ▼
   ┌─────────────────────────────────┐
   │        Fall_Detected            │
   │   (cooldown → Normal)           │
   └─────────────────────────────────┘
```

### Rules

1. **Trunk Angle** > 45°: Trunk centerline (shoulder mid → hip mid) tilted from vertical
2. **CoM outside BoS**: Hip midpoint X-coordinate outside ankle boundaries ± margin
3. **Y-Velocity** > 0.3: Nose landmark moving downward rapidly (fall signature)

---

## 📝 License

This project is developed for research and healthcare applications.

---

## 🙏 Acknowledgments

- [Google MediaPipe](https://mediapipe.dev/) — Pose estimation model
- [FastAPI](https://fastapi.tiangolo.com/) — Backend API framework
- [OpenCV](https://opencv.org/) — Computer vision library
