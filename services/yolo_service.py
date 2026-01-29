from ultralytics import YOLO
import cv2
import os

class FireSmokeDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        print(f"🔥 Fire/Smoke Detection Model Loaded: {model_path}")
        print(f"📋 Model classes: {self.model.names}")

    def detect(self, image_path):
        """Detect fire/smoke in a single image or video (returns all detections)"""
        results = self.model(image_path, conf=0.4)
        detections = []

        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": int(box.cls),
                    "confidence": float(box.conf)
                })

        return detections

    def detect_video_stream(self, video_path):
        """
        Stream detection results frame-by-frame for real-time alerts.
        Yields detection results immediately when fire is found in any frame.
        """
        # Check if file is a video
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        file_ext = os.path.splitext(video_path)[1].lower()
        
        if file_ext not in video_extensions:
            # Not a video, process as image
            detections = self.detect(video_path)
            yield {
                "frame": 0,
                "total_frames": 1,
                "detections": detections,
                "has_fire": len(detections) > 0
            }
            return

        # Process video frame by frame
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_number += 1

            # Run detection on this frame
            results = self.model(frame, conf=0.4, verbose=False)
            detections = []

            for r in results:
                for box in r.boxes:
                    detections.append({
                        "class": int(box.cls),
                        "confidence": float(box.conf)
                    })

            # Yield result immediately if fire detected
            has_fire = len(detections) > 0
            
            yield {
                "frame": frame_number,
                "total_frames": total_frames,
                "detections": detections,
                "has_fire": has_fire,
                "progress": (frame_number / total_frames) * 100 if total_frames > 0 else 100
            }

            # If fire detected, this will be sent immediately to frontend!

        cap.release()
