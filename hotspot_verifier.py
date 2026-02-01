"""
Intelligent Satellite Hotspot Verification System

This module cross-references NASA FIRMS thermal hotspots with computer vision
to visually confirm actual wildfires, reducing false alarms.

Process:
1. Get thermal hotspots from NASA FIRMS (temperature-based)
2. Fetch satellite imagery for each hotspot location
3. Run YOLO fire detection on the imagery
4. Only confirm as wildfire if BOTH thermal + visual detection agree
"""

import requests
from datetime import datetime
import cv2
import numpy as np
from io import BytesIO
from PIL import Image


class HotspotVerifier:
    """Verifies satellite thermal hotspots using computer vision"""
    
    def __init__(self, yolo_model):
        """
        Args:
            yolo_model: Trained YOLO fire detection model
        """
        self.yolo_model = yolo_model
        self.sentinel_hub_url = "https://services.sentinel-hub.com/ogc/wms/"
        
    def verify_hotspot(self, lat, lon, confidence, frp):
        """
        Verify if a thermal hotspot is an actual wildfire using computer vision.
        
        Args:
            lat: Latitude of hotspot
            lon: Longitude of hotspot
            confidence: FIRMS confidence percentage
            frp: Fire Radiative Power
            
        Returns:
            dict: {
                'is_verified': bool,
                'thermal_confidence': float,
                'visual_confidence': float,
                'verification_method': str,
                'status': str
            }
        """
        print(f"\n🔍 Verifying hotspot at ({lat}, {lon})")
        print(f"   Thermal confidence: {confidence}%, FRP: {frp} MW")
        
        # Try to get satellite imagery for this location
        imagery = self._fetch_satellite_imagery(lat, lon)
        
        if imagery is None:
            # No imagery available - rely on thermal data only
            print(f"   ⚠️ No satellite imagery available")
            return {
                'is_verified': confidence >= 80 and frp >= 20,  # High threshold without visual
                'thermal_confidence': confidence,
                'visual_confidence': 0,
                'verification_method': 'thermal_only',
                'status': 'unverified_no_imagery'
            }
        
        # Run YOLO detection on satellite imagery
        visual_result = self._detect_fire_in_imagery(imagery)
        
        if visual_result['has_fire']:
            # BOTH thermal and visual detection agree - HIGH CONFIDENCE
            print(f"   ✅ VERIFIED: Visual confirmation ({visual_result['confidence']:.1%})")
            return {
                'is_verified': True,
                'thermal_confidence': confidence,
                'visual_confidence': visual_result['confidence'] * 100,
                'verification_method': 'thermal_and_visual',
                'status': 'verified_wildfire'
            }
        else:
            # Thermal detection but NO visual confirmation - likely false alarm
            print(f"   ❌ NOT VERIFIED: No visual fire detected")
            return {
                'is_verified': False,
                'thermal_confidence': confidence,
                'visual_confidence': 0,
                'verification_method': 'visual_rejected',
                'status': 'false_alarm_rejected'
            }
    
    def _fetch_satellite_imagery(self, lat, lon, zoom_km=2):
        """
        Fetch satellite imagery for a location.
        In production, this would use Sentinel Hub or Google Earth Engine.
        For demo, we'll use a placeholder approach.
        
        Args:
            lat: Latitude
            lon: Longitude
            zoom_km: Area size in kilometers
            
        Returns:
            numpy array: RGB image or None
        """
        # TODO: In production, integrate with:
        # - Sentinel Hub API for real satellite imagery
        # - Google Earth Engine
        # - NASA EOSDIS
        
        # For now, return None to indicate imagery not available
        # This will be implemented when you have API keys
        return None
    
    def _detect_fire_in_imagery(self, image):
        """
        Run YOLO fire detection on satellite imagery.
        
        Args:
            image: numpy array (RGB)
            
        Returns:
            dict: {'has_fire': bool, 'confidence': float, 'detections': list}
        """
        # Run YOLO detection
        results = self.yolo_model(image, conf=0.2, verbose=False)
        
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    'confidence': float(box.conf),
                    'bbox': box.xyxy.tolist()[0] if hasattr(box, 'xyxy') else None
                })
        
        has_fire = len(detections) > 0
        avg_confidence = sum(d['confidence'] for d in detections) / len(detections) if detections else 0
        
        return {
            'has_fire': has_fire,
            'confidence': avg_confidence,
            'detections': detections
        }


def verify_all_hotspots(hotspots, yolo_model):
    """
    Verify all thermal hotspots using computer vision.
    
    Args:
        hotspots: List of thermal hotspot dicts from FIRMS
        yolo_model: YOLO fire detection model
        
    Returns:
        dict: {
            'verified_fires': list,
            'false_alarms': list,
            'unverified': list,
            'stats': dict
        }
    """
    verifier = HotspotVerifier(yolo_model)
    
    verified_fires = []
    false_alarms = []
    unverified = []
    
    print(f"\n🛰️ Verifying {len(hotspots)} thermal hotspots with computer vision...")
    
    for hotspot in hotspots:
        result = verifier.verify_hotspot(
            hotspot['lat'],
            hotspot['lon'],
            hotspot['confidence'],
            hotspot['frp']
        )
        
        # Add verification result to hotspot
        hotspot['verification'] = result
        
        # Categorize based on verification
        if result['status'] == 'verified_wildfire':
            verified_fires.append(hotspot)
        elif result['status'] == 'false_alarm_rejected':
            false_alarms.append(hotspot)
        else:
            unverified.append(hotspot)
    
    stats = {
        'total_hotspots': len(hotspots),
        'verified_fires': len(verified_fires),
        'false_alarms_rejected': len(false_alarms),
        'unverified': len(unverified),
        'verification_rate': len(verified_fires) / len(hotspots) * 100 if hotspots else 0
    }
    
    print(f"\n📊 Verification Results:")
    print(f"   ✅ Verified fires: {stats['verified_fires']}")
    print(f"   ❌ False alarms rejected: {stats['false_alarms_rejected']}")
    print(f"   ⚠️ Unverified (no imagery): {stats['unverified']}")
    print(f"   📈 Verification rate: {stats['verification_rate']:.1f}%")
    
    return {
        'verified_fires': verified_fires,
        'false_alarms': false_alarms,
        'unverified': unverified,
        'stats': stats
    }
