from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import shutil, os
import json
import asyncio

from services.yolo_service import FireSmokeDetector
from services.satellite_service import SatelliteFireDetector

app = FastAPI(title="Forest Fire AI System")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

yolo = FireSmokeDetector("models/fire_smoke_yolo_best.pt")
satellite = SatelliteFireDetector("models/satellite_wildfire_resnet18.pth")

@app.post("/detect/fire-smoke")
async def detect_fire_smoke(file: UploadFile = File(...)):
    """Original endpoint - returns all detections after processing"""
    path = f"{UPLOAD_DIR}/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    detections = yolo.detect(path)
    return {"detections": detections}

@app.post("/detect/fire-smoke/stream")
async def detect_fire_smoke_stream(file: UploadFile = File(...)):
    """
    STREAMING endpoint - sends detection results frame-by-frame in real-time!
    Alerts are sent IMMEDIATELY when fire is detected in any frame.
    """
    path = f"{UPLOAD_DIR}/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    async def event_generator():
        """Generate Server-Sent Events for each frame"""
        try:
            for result in yolo.detect_video_stream(path):
                # Send event immediately when frame is processed
                event_data = json.dumps(result)
                yield f"data: {event_data}\n\n"
                
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)
            
            # Send completion event
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            # Send error event
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

@app.post("/detect/satellite-fire")
async def detect_satellite_fire(file: UploadFile = File(...)):
    path = f"{UPLOAD_DIR}/{file.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    prediction = satellite.predict(path)
    return prediction

@app.post("/detect/frame")
async def detect_frame(data: dict):
    """
    Real-time frame detection for webcam/live camera feeds.
    Accepts base64 encoded image frame and returns detection results immediately.
    Optimized for low-latency real-time detection.
    """
    import base64
    import numpy as np
    import cv2
    from datetime import datetime
    
    try:
        # Extract base64 image data
        image_data = data.get('frame', '')
        if not image_data:
            return {"error": "No frame data provided"}
        
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64 to image
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Failed to decode image"}
        
        print(f"🖼️ Frame decoded: {frame.shape}")
        
        # Run YOLO detection on frame with lower confidence threshold
        results = yolo.model(frame, conf=0.25, verbose=False)  # Lowered from 0.4 to 0.25
        detections = []
        
        for r in results:
            for box in r.boxes:
                detection = {
                    "class": int(box.cls),
                    "confidence": float(box.conf),
                    "bbox": box.xyxy.tolist()[0] if hasattr(box, 'xyxy') else None
                }
                detections.append(detection)
                print(f"🔍 Detection: class={detection['class']}, confidence={detection['confidence']:.2f}")
        
        has_fire = len(detections) > 0
        
        if has_fire:
            print(f"🔥 FIRE DETECTED! {len(detections)} detection(s)")
        else:
            print(f"✅ No fire detected in frame")
        
        return {
            "detections": detections,
            "has_fire": has_fire,
            "timestamp": datetime.now().isoformat(),
            "frame_size": frame.shape[:2]
        }
        
    except Exception as e:
        print(f"❌ Error in frame detection: {str(e)}")
        return {"error": str(e), "detections": [], "has_fire": False}

@app.get("/proxy/camera")
async def proxy_camera(url: str):
    """
    Proxy endpoint to fetch IP Webcam stream and serve it from same origin.
    This bypasses CORS restrictions by serving the image from our backend.
    """
    import requests
    from fastapi.responses import Response
    
    print(f"🔄 Proxy request for URL: {url}")
    
    try:
        print(f"📡 Fetching from IP Webcam using requests library...")
        
        # Use requests library (synchronous but more reliable)
        response = requests.get(url, timeout=10, stream=False)
        
        print(f"✅ Response status: {response.status_code}")
        print(f"📦 Content length: {len(response.content)} bytes")
        
        if response.status_code == 200:
            return Response(
                content=response.content,
                media_type=response.headers.get('content-type', 'image/jpeg'),
                headers={
                    'Access-Control-Allow-Origin': '*',
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                }
            )
        else:
            error_msg = f"Failed to fetch camera stream: HTTP {response.status_code}"
            print(f"❌ {error_msg}")
            return {"error": error_msg}
    except requests.exceptions.Timeout as e:
        error_msg = f"Timeout connecting to camera: {str(e)}"
        print(f"⏱️ {error_msg}")
        return {"error": error_msg}
    except requests.exceptions.ConnectionError as e:
        error_msg = f"Cannot connect to camera: {str(e)}"
        print(f"🔌 {error_msg}")
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Proxy error: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}
