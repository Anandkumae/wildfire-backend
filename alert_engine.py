from config import *

def filter_fire_events(df):
    filtered = df[
        (df["confidence"] >= MIN_CONFIDENCE) &
        (df["frp"] >= MIN_FRP)
    ]

    alerts = []
    for _, row in filtered.iterrows():
        alerts.append({
            "lat": row["latitude"],
            "lon": row["longitude"],
            "confidence": row["confidence"],
            "frp": row["frp"],
            "date": row["acq_date"],
            "time": row["acq_time"]
        })

    return alerts
