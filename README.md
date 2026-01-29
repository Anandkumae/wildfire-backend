# 🔥 Wildfire Detection - Backend API

FastAPI backend for real-time fire and smoke detection using YOLO deep learning model. Supports video analysis, live webcam feeds, and network camera (IP Webcam) integration.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![YOLO](https://img.shields.io/badge/YOLO-v8-red)

## ✨ Features

- **Real-time Fire Detection**: YOLO-based object detection for fire and smoke
- **Video Analysis**: Frame-by-frame detection with streaming results
- **Live Camera Support**: Real-time detection from webcam or network cameras
- **CORS Proxy**: Bypass CORS restrictions for IP Webcam integration
- **Server-Sent Events**: Stream detection results in real-time
- **Low Latency**: Optimized for fast detection (< 100ms per frame)

## 🏗️ Architecture

```
backend/
├── app.py                  # Main FastAPI application
├── services/
│   └── yolo_service.py    # YOLO detection service
├── models/
│   ├── README.md          # Model setup instructions
│   └── best.pt            # YOLO model (not in repo)
├── uploads/               # Temporary file storage
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- YOLO model trained for fire/smoke detection (`.pt` file)
- 4GB+ RAM recommended

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Anandkumae/wildfire-backend.git
   cd wildfire-backend
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add YOLO model:**
   - Place your trained model file (e.g., `best.pt`) in the `models/` directory
   - See `models/README.md` for detailed instructions

5. **Run the server:**
   ```bash
   uvicorn app:app --reload
   ```

6. **Access the API:**
   - API: `http://localhost:8000`
   - Docs: `http://localhost:8000/docs`
   - Health: `http://localhost:8000/`

## 📊 API Endpoints

### Health Check

```http
GET /
```

Returns API status and version.

**Response:**
```json
{
  "message": "Wildfire Detection API",
  "status": "active"
}
```

---

### Video/Image Detection

```http
POST /detect/fire-smoke
```

Upload video or image file for fire/smoke detection.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (video or image file)

**Response:**
```json
{
  "detections": [
    {
      "class": 0,
      "confidence": 0.85
    }
  ],
  "has_fire": true,
  "total_detections": 1
}
```

---

### Streaming Detection (SSE)

```http
POST /detect/fire-smoke/stream
```

Stream detection results frame-by-frame using Server-Sent Events.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (video file)

**Response:**
- Content-Type: `text/event-stream`
- Streams JSON events for each frame with detections

**Event Format:**
```json
{
  "frame": 42,
  "detections": [...],
  "has_fire": true,
  "timestamp": "2026-01-30T03:30:00"
}
```

---

### Frame Detection (Webcam)

```http
POST /detect/frame
```

Detect fire/smoke in a single frame (base64 encoded).

**Request:**
```json
{
  "frame": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
}
```

**Response:**
```json
{
  "detections": [
    {
      "class": 0,
      "confidence": 0.92,
      "bbox": [100, 150, 300, 400]
    }
  ],
  "has_fire": true,
  "timestamp": "2026-01-30T03:30:00.123456",
  "frame_size": [480, 640]
}
```

---

### Camera Proxy (CORS Bypass)

```http
GET /proxy/camera?url={camera_url}
```

Proxy endpoint to fetch IP Webcam stream and serve from same origin.

**Parameters:**
- `url`: IP Webcam stream URL (e.g., `http://192.168.1.100:8080/shot.jpg`)

**Response:**
- Content-Type: `image/jpeg`
- Returns camera image with CORS headers

---

## 🔧 Configuration

### Detection Threshold

Adjust confidence threshold in `app.py`:

```python
# Lower threshold = more sensitive (more false positives)
# Higher threshold = less sensitive (may miss fires)
results = yolo.model(frame, conf=0.25, verbose=False)
```

**Recommended values:**
- `0.25` - Balanced (default)
- `0.15` - More sensitive
- `0.40` - Less sensitive, higher accuracy

### CORS Settings

Update CORS origins in `app.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**For production:**
```python
allow_origins=["https://your-frontend-domain.com"]
```

### Model Path

Change model path in `app.py`:

```python
yolo = FireSmokeDetector('models/your-model.pt')
```

## 🧪 Testing

### Using cURL

**Test health endpoint:**
```bash
curl http://localhost:8000/
```

**Test image detection:**
```bash
curl -X POST http://localhost:8000/detect/fire-smoke \
  -F "file=@test_image.jpg"
```

**Test streaming detection:**
```bash
curl -X POST http://localhost:8000/detect/fire-smoke/stream \
  -F "file=@test_video.mp4"
```

### Using Swagger UI

1. Start the server
2. Open `http://localhost:8000/docs`
3. Try out endpoints interactively

### Using Python

```python
import requests

# Upload image
with open('fire_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/detect/fire-smoke',
        files={'file': f}
    )
    print(response.json())
```

## 📦 Dependencies

Core dependencies (see `requirements.txt`):

- **FastAPI** - Modern web framework
- **Uvicorn** - ASGI server
- **Ultralytics** - YOLO implementation
- **OpenCV** - Image/video processing
- **NumPy** - Numerical operations
- **Requests/HTTPX** - HTTP client for proxy

## 🔒 Security

### Production Checklist

- [ ] Update CORS origins to specific domains
- [ ] Add authentication/API keys
- [ ] Enable HTTPS
- [ ] Rate limiting for endpoints
- [ ] Input validation and sanitization
- [ ] Secure file upload handling
- [ ] Environment variables for sensitive config
- [ ] Logging and monitoring

### Environment Variables

Create `.env` file:

```env
MODEL_PATH=models/best.pt
UPLOAD_DIR=uploads
MAX_FILE_SIZE=100000000
CONFIDENCE_THRESHOLD=0.25
ALLOWED_ORIGINS=http://localhost:5173,https://your-domain.com
```

Load in `app.py`:
```python
from dotenv import load_dotenv
import os

load_dotenv()
MODEL_PATH = os.getenv('MODEL_PATH', 'models/best.pt')
```

## 🐛 Troubleshooting

### Model not loading

**Error:** `FileNotFoundError: models/best.pt`

**Solution:**
1. Add your YOLO model to `models/` directory
2. Check model path in `app.py`
3. See `models/README.md` for setup instructions

### CORS errors

**Error:** `Access to fetch blocked by CORS policy`

**Solution:**
1. Add frontend URL to `allow_origins` in `app.py`
2. Restart server after changes

### Out of memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
1. Use smaller model (YOLOv8n instead of YOLOv8x)
2. Reduce image resolution
3. Process fewer frames per second
4. Use CPU instead of GPU

### Slow detection

**Issue:** Detection takes > 1 second per frame

**Solution:**
1. Use GPU if available
2. Use smaller YOLO model (n/s instead of m/l/x)
3. Reduce input image size
4. Lower confidence threshold (processes faster)

## 📈 Performance

### Benchmarks (on CPU)

| Model | FPS | Accuracy | Memory |
|-------|-----|----------|--------|
| YOLOv8n | ~10 | Good | 2GB |
| YOLOv8s | ~7 | Better | 3GB |
| YOLOv8m | ~4 | Best | 4GB |

### Optimization Tips

1. **Use GPU**: 10-20x faster than CPU
2. **Batch processing**: Process multiple frames together
3. **Frame skipping**: Analyze every 2nd or 3rd frame
4. **Model quantization**: Reduce model size
5. **Async processing**: Use background tasks

## 🔗 Integration

### Frontend Integration

**React/JavaScript:**
```javascript
// Upload video
const formData = new FormData();
formData.append('file', videoFile);

const response = await fetch('http://localhost:8000/detect/fire-smoke', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result);
```

**Streaming detection:**
```javascript
const eventSource = new EventSource(
  'http://localhost:8000/detect/fire-smoke/stream'
);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.has_fire) {
    alert('Fire detected!');
  }
};
```

### Mobile Integration

Use the `/detect/frame` endpoint with base64 encoded images from mobile camera.

## 📝 Model Requirements

Your YOLO model should:

- Be in PyTorch `.pt` format
- Have classes for fire and/or smoke detection
- Be trained on diverse fire/smoke dataset
- Support standard YOLO input (RGB images)

**Model classes example:**
```python
{
    0: 'fire',
    1: 'smoke'
}
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🔗 Related Projects

- **Frontend**: [wildfire-frontend](https://github.com/Anandkumae/wildfire-frontend)
- **YOLO**: [Ultralytics](https://github.com/ultralytics/ultralytics)

## 📧 Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues for solutions

---

**⚠️ Important:** This is a detection system for early warning. Always follow proper fire safety protocols and contact emergency services when fire is detected.

**🔥 Built with FastAPI + YOLO for real-time fire detection**
