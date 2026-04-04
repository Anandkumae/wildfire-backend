import requests
import pandas as pd
from config import *

def fetch_modis_data():
    # Use NASA FIRMS Country API for India (much faster than area API)
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/"
        f"{FIRMS_MAP_KEY}/{FIRMS_SOURCE}/IND/{DAYS}"
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
