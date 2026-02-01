import requests
import pandas as pd
from config import *

def fetch_modis_data():
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{FIRMS_MAP_KEY}/{FIRMS_SOURCE}/"
        f"{WEST},{SOUTH},{EAST},{NORTH}/{DAYS}"
    )

    response = requests.get(url)
    response.raise_for_status()

    with open("data/modis_latest.csv", "wb") as f:
        f.write(response.content)

    print("✅ MODIS satellite data downloaded")

    df = pd.read_csv("data/modis_latest.csv")
    return df
