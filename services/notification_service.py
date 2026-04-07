import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import threading

# Simple in-memory cooldown tracker
# Key: context (e.g., 'satellite', 'yolo')
# Value: datetime of last alert
_last_alert_times = {}
_lock = threading.Lock()

COOLDOWN_MINUTES = 30
CONFIDENCE_THRESHOLD = 85  # Only send alerts if confidence >= 85%

def can_send_alert(context="global"):
    """Check if the cooldown period has passed for a given alert context."""
    global _last_alert_times
    now = datetime.now()
    
    with _lock:
        if context in _last_alert_times:
            last_sent = _last_alert_times[context]
            if now - last_sent < timedelta(minutes=COOLDOWN_MINUTES):
                return False
        
        _last_alert_times[context] = now
        return True

def send_incident_email(lat, lon, confidence, source="System Detection"):
    """
    Sends an incident report email to the emergency group when confidence >= 85%.
    
    Args:
        lat: Latitude of the incident
        lon: Longitude of the incident
        confidence: Detection confidence percentage
        source: Source of the detection (e.g. 'NASA Satellite', 'Optic Sensor')
    """
    
    # Check confidence threshold
    if confidence < CONFIDENCE_THRESHOLD:
        print(f" [Notification Service] Confidence {confidence}% below threshold {CONFIDENCE_THRESHOLD}%. Email not sent.")
        return False
    
    # Environment variables for SMTP
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASS = os.getenv("SMTP_PASSWORD")
    FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USER)
    
    # Recipient list (comma-separated string in .env)
    RECIPIENTS_STR = os.getenv("EMERGENCY_GROUP_EMAILS", "")
    if not RECIPIENTS_STR or not SMTP_USER or not SMTP_PASS:
        print("⚠️ [Notification Service] Missing SMTP configuration or recipient list. Email skipped.")
        return False

    recipients = [r.strip() for r in RECIPIENTS_STR.split(",") if r.strip()]
    
    if not recipients:
        print("⚠️ [Notification Service] No valid recipients found.")
        return False

    # Google Maps URL
    maps_url = f"https://www.google.com/maps?q={lat},{lon}"
    
    # Email Content
    subject = f"🚨 URGENT: Fire Detected - {source} Alert"
    
    body = f"""
🔥 FIRE INCIDENT REPORT - WILDFIRE SYSTEM
=====================================================

An active fire incident has been detected and verified by the {source}.

DETAILS:
-----------------------------------------------------
📍 COORDINATES: {lat}, {lon}
🎯 CONFIDENCE: {confidence}%
🕒 TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🛰️ DETECTION SOURCE: {source}

🌍 VIEW ON GOOGLE MAPS:
{maps_url}

ACTION REQUIRED:
-----------------------------------------------------
Please verify the incident and alert local emergency services if necessary.

---
This is an automated emergency alert from your Wildfire System.
    """

    try:
        msg = MIMEMultipart()
        msg['From'] = FROM_EMAIL
        msg['To'] = ", ".join(recipients) # Showing all in To (or use BCC for privacy)
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
        server.quit()
        
        print(f"✅ [Notification Service] Emergency email sent to {len(recipients)} recipients.")
        return True
    except Exception as e:
        print(f" [Notification Service] Failed to send email: {str(e)}")
        return False

def send_alerts(lat, lon, confidence, source="System Detection"):
    """
    Sends email alerts if confidence threshold (85%) is met.
    
    Args:
        lat: Latitude of the incident
        lon: Longitude of the incident
        confidence: Detection confidence percentage
        source: Source of the detection
    """
    print(f" [Notification Service] Processing alert: {confidence}% confidence from {source}")
    
    # Send email (only if confidence >= 85%)
    email_sent = send_incident_email(lat, lon, confidence, source)
    
    return email_sent
