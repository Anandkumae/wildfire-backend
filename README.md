# Wildfire System - Backend API

FastAPI backend for real-time fire and smoke detection using YOLO deep learning model with integrated emergency alert system. Supports video analysis, live webcam feeds, satellite monitoring, and automated notifications.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![YOLO](https://img.shields.io/badge/YOLO-v8-red)
![Twilio](https://img.shields.io/badge/Twilio-SMS-orange)

## Features

### Detection Capabilities
- **Real-time Fire Detection**: YOLO-based object detection for fire and smoke
- **Video Analysis**: Frame-by-frame detection with streaming results
- **Live Camera Support**: Real-time detection from webcam or network cameras
- **Satellite Monitoring**: NASA MODIS satellite data integration for large-scale monitoring
- **CORS Proxy**: Bypass CORS restrictions for IP Webcam integration
- **Server-Sent Events**: Stream detection results in real-time
- **Low Latency**: Optimized for fast detection (< 100ms per frame)

### Alert System
- **85% Confidence Threshold**: Alerts only sent for high-confidence detections (>=85%)
- **Dual Notification**: Both email and SMS alerts for critical detections
- **Smart Cooldown**: 30-minute cooldown prevents alert spam per location
- **Multiple Sources**: Manual uploads, live cameras, and satellite detections
- **Emergency Contacts**: Configurable email and phone number lists

## Architecture

```
backend/
|-- app.py                          # Main FastAPI application
|-- services/
|   |-- yolo_service.py            # YOLO detection service
|   |-- notification_service.py    # Email & SMS alert system
|   -- satellite_service.py        # NASA satellite monitoring
|-- models/
|   |-- README.md                  # Model setup instructions
|   -- best.pt                     # YOLO model (not in repo)
|-- uploads/                       # Temporary file storage
|-- .env                           # Configuration file
|-- requirements.txt               # Python dependencies
-- README.md                       # This file
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- YOLO model trained for fire/smoke detection (`.pt` file)
- 4GB+ RAM recommended
- Gmail account with App Password (for email alerts)
- Twilio account (for SMS alerts)

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

5. **Configure Environment:**
   ```bash
   # Copy and edit the environment file
   cp .env.example .env
   # Edit .env with your credentials
   ```

6. **Run the server:**
   ```bash
   uvicorn app:app --reload
   ```

7. **Access the API:**
   - API: `http://localhost:8000`
   - Docs: `http://localhost:8000/docs`
   - Health: `http://localhost:8000/`

## Configuration

### Environment Variables (.env)

```bash
# Email Configuration (Gmail)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=wildfire-alert@gmail.com
SMTP_PASSWORD=your_gmail_app_password
FROM_EMAIL=wildfire-alert@gmail.com
EMERGENCY_GROUP_EMAILS=a2056164@gmail.com,akashmoreasm6000@gmail.com

# SMS Configuration (Twilio)
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=+1234567890
EMERGENCY_PHONE_NUMBERS=+919876543210,+919876543211

# Detection Settings
CONFIDENCE_THRESHOLD=0.25
ALERT_CONFIDENCE_THRESHOLD=85
COOLDOWN_MINUTES=30
```

### Alert Threshold Configuration

The system uses an **85% confidence threshold** for sending alerts:

- **Detection Threshold**: 0.25 (25%) for general detection
- **Alert Threshold**: 85% for sending email/SMS notifications
- **Cooldown**: 30 minutes between alerts per location

## API Endpoints

### Health Check
```http
GET /
```

### Video/Image Detection
```http
POST /detect/fire-smoke
```
Upload video or image file for fire/smoke detection.

### Streaming Detection (SSE)
```http
POST /detect/fire-smoke/stream
```
Stream detection results frame-by-frame using Server-Sent Events.

### Frame Detection (Webcam)
```http
POST /detect/frame
```
Detect fire/smoke in a single frame (base64 encoded).

### Satellite Monitoring
```http
GET /satellite/fires
```
Get current fire data from NASA MODIS satellites.

### Camera Proxy (CORS Bypass)
```http
GET /proxy/camera?url={camera_url}
```
Proxy endpoint to fetch IP Webcam stream and serve from same origin.

## Alert System

### How Alerts Work

1. **Detection Occurs**: Any source detects fire/smoke
2. **Confidence Check**: If confidence < 85% -> No alerts
3. **Alert Triggered**: If confidence >= 85%:
   - Email sent to emergency group
   - SMS sent to emergency contacts
   - 30-minute cooldown engaged

### Alert Message Format

**Email Subject**: `URGENT: Fire Detected - [Source] Alert`

**SMS Message**:
```
WILDFIRE SYSTEM ALERT
Confidence: 92%
Location: 34.05, -118.24
Source: Live Optic Sensor
Maps: https://www.google.com/maps?q=34.05,-118.24
URGENT: Verify and alert emergency services!
```

### Supported Detection Sources

- **Manual Upload**: User-uploaded images/videos
- **Live Camera**: Real-time webcam/IP camera feeds
- **Satellite**: NASA MODIS satellite monitoring

## Dependencies

Core dependencies (see `requirements.txt`):

- **FastAPI** - Modern web framework
- **Uvicorn** - ASGI server
- **Ultralytics** - YOLO implementation
- **OpenCV** - Image/video processing
- **NumPy** - Numerical operations
- **Requests** - HTTP client for proxy
- **Twilio** - SMS service integration
- **Earth Engine API** - Satellite data
- **Geemap** - Geospatial visualization

## Security

### Production Checklist

- [ ] Update CORS origins to specific domains
- [ ] Add authentication/API keys
- [ ] Enable HTTPS
- [ ] Rate limiting for endpoints
- [ ] Input validation and sanitization
- [ ] Secure file upload handling
- [ ] Environment variables for sensitive config
- [ ] Logging and monitoring

## Performance

### Benchmarks (on CPU)

| Model | FPS | Accuracy | Memory |
|-------|-----|----------|--------|
| YOLOv8n | ~10 | Good | 2GB |
| YOLOv8s | ~7 | Better | 3GB |
| YOLOv8m | ~4 | Best | 4GB |

## Integration Examples

### Frontend Integration (React)
```javascript
// Upload video for detection
const formData = new FormData();
formData.append('file', videoFile);

const response = await fetch('http://localhost:8000/detect/fire-smoke', {
  method: 'POST',
  body: formData
});

const result = await response.json();
if (result.has_fire && result.detections[0].confidence >= 0.85) {
  // High confidence fire detected - alerts will be sent automatically
  console.log('Emergency alerts triggered!');
}
```

### Streaming Detection
```javascript
const eventSource = new EventSource(
  'http://localhost:8000/detect/fire-smoke/stream'
);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.has_fire) {
    console.log('Fire detected with confidence:', data.detections[0].confidence);
  }
};
```

## Troubleshooting

### Email Issues
- **Error**: `Username and Password not accepted`
- **Solution**: Use Gmail App Password, not regular password

### SMS Issues
- **Error**: `TwilioRestException`
- **Solution**: Verify Twilio credentials and phone numbers

### Model Loading
- **Error**: `FileNotFoundError: models/best.pt`
- **Solution**: Add YOLO model to models/ directory

### CORS Errors
- **Error**: `Access to fetch blocked by CORS policy`
- **Solution**: Add frontend URL to allow_origins in app.py

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues for solutions

---

**Important**: This is a detection system for early warning. Always follow proper fire safety protocols and contact emergency services when fire is detected.

**Built with FastAPI + YOLO + Twilio for comprehensive wildfire detection and alerting**
