import requests
import pandas as pd
from config import *

def fetch_modis_data():
    # Use NASA FIRMS Area API (Country API is currently unavailable)
    # Format: https://firms.modaps.eosdis.nasa.gov/api/area/csv/[MAP_KEY]/[SOURCE]/[EXTENT]/[DAYS]
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{FIRMS_MAP_KEY}/{FIRMS_SOURCE}/"
        f"{WEST},{SOUTH},{EAST},{NORTH}/{DAYS}"
    )

    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching NASA FIRMS data: {e}")
        raise

    with open("data/modis_latest.csv", "wb") as f:
        f.write(response.content)

    print("✅ MODIS satellite data downloaded")

    df = pd.read_csv("data/modis_latest.csv")
    return df
