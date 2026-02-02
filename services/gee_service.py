"""
Google Earth Engine Service for Wildfire Detection

This service integrates with Google Earth Engine to fetch:
1. Real satellite imagery (Sentinel-2) for visual confirmation
2. Surface temperature data (MODIS) for thermal analysis
3. Comprehensive hotspot analysis combining multiple data sources

Architecture:
- Local Python script connects to GEE cloud
- Requests only specific location + time data (no full dataset downloads)
- Returns processed imagery and temperature readings
"""

import ee
import geemap
import os
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Output directory for satellite images
OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "satellite_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def initialize_gee():
    """
    Initialize Google Earth Engine connection.
    
    This uses locally stored credentials from 'earthengine authenticate'.
    No API keys needed - authentication is done once via OAuth.
    
    Raises:
        Exception: If GEE is not authenticated
    """
    try:
        # Try to initialize with default project
        try:
            ee.Initialize()
        except Exception as init_error:
            # If initialization fails due to missing project, try with opt_url parameter
            # This uses the high-volume endpoint which doesn't require a project
            if "no project found" in str(init_error).lower():
                print("⚠️ No default project found, using high-volume endpoint...")
                ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')
            else:
                raise init_error
        
        print("✅ Google Earth Engine initialized successfully")
    except Exception as e:
        print(f"❌ GEE initialization failed: {e}")
        print("💡 Run 'earthengine authenticate' in terminal to set up credentials")
        print("💡 Or set a default project: earthengine set_project YOUR_PROJECT_ID")
        raise


def get_satellite_image(lat, lon, date=None):
    """
    Get real Sentinel-2 satellite image for a location.
    
    Sentinel-2 provides high-resolution RGB imagery (10m resolution).
    Filters for low cloud coverage to get clear images.
    
    Args:
        lat: Latitude
        lon: Longitude
        date: Date string in 'YYYY-MM-DD' format (defaults to yesterday)
        
    Returns:
        ee.Image: Sentinel-2 image or None if no imagery available
    """
    if date is None:
        # Default to 3 days ago (satellite data has ~1-2 day delay)
        date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    
    # Create date range (±7 days for better coverage)
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    start_date = (date_obj - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
    
    point = ee.Geometry.Point(lon, lat)
    
    try:
        # Fetch Sentinel-2 Surface Reflectance imagery (harmonized version)
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50))  # Accept up to 50% cloud coverage
            .sort('CLOUDY_PIXEL_PERCENTAGE')  # Get least cloudy image first
        )
        
        # Check if any images are available
        count = collection.size().getInfo()
        if count == 0:
            print(f"⚠️ No Sentinel-2 imagery available for this location/date")
            print(f"   Try expanding date range or accepting higher cloud coverage")
            return None
        
        image = collection.first()
        return image
    except Exception as e:
        print(f"⚠️ Error fetching Sentinel-2 imagery: {e}")
        return None


def save_rgb_image(image, lat, lon, output_filename=None):
    """
    Extract and save RGB visual image from satellite data.
    
    This creates the actual satellite photo that users will see.
    Uses Sentinel-2 RGB bands (B4=Red, B3=Green, B2=Blue).
    
    Args:
        image: ee.Image from Sentinel-2
        lat: Latitude (for filename)
        lon: Longitude (for filename)
        output_filename: Optional custom filename
        
    Returns:
        str: Path to saved image file or None if failed
    """
    if image is None:
        print("⚠️ No image provided to save")
        return None
    
    # Generate filename
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"sat_{lat:.4f}_{lon:.4f}_{timestamp}"
        tif_filename = f"{base_filename}.tif"
        png_filename = f"{base_filename}.png"
    else:
        base_filename = output_filename.replace('.tif', '').replace('.png', '')
        tif_filename = f"{base_filename}.tif"
        png_filename = f"{base_filename}.png"
    
    tif_path = OUTPUT_DIR / tif_filename
    png_path = OUTPUT_DIR / png_filename
    
    # Define region around the point (2km buffer)
    region = ee.Geometry.Point(lon, lat).buffer(2000)
    
    try:
        # Select RGB bands and visualize
        try:
            rgb_image = image.select(["B4", "B3", "B2"]).visualize(
                min=0,
                max=3000,
                gamma=1.4
            )
        except Exception as select_error:
            print(f"❌ Error selecting bands: {select_error}")
            print("⚠️ Image may be null or missing required bands")
            return None
        
        # Export image as GeoTIFF using geemap
        try:
            geemap.ee_export_image(
                rgb_image,
                filename=str(tif_path),
                scale=10,  # 10m resolution
                region=region,
                file_per_band=False
            )
        except Exception as export_error:
            print(f"❌ Error exporting image: {export_error}")
            print("⚠️ No satellite imagery available for this location/date")
            return None
        
        # Convert GeoTIFF to PNG for browser compatibility
        try:
            from PIL import Image
            import rasterio
            
            # Read the GeoTIFF
            with rasterio.open(str(tif_path)) as src:
                # Read RGB bands
                data = src.read()
                
                # Convert to PIL Image (transpose from (bands, height, width) to (height, width, bands))
                if data.shape[0] == 3:
                    img_array = np.transpose(data, (1, 2, 0))
                else:
                    img_array = data[0]  # Single band
                
                # Normalize to 0-255 range
                img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min()) * 255).astype(np.uint8)
                
                # Create PIL image and save as PNG
                pil_img = Image.fromarray(img_array)
                pil_img.save(str(png_path), 'PNG')
                
            print(f"✅ Satellite image saved: {png_path}")
            return str(png_path)
            
        except Exception as convert_error:
            print(f"⚠️ Could not convert to PNG: {convert_error}")
            print(f"✅ Satellite image saved as GeoTIFF: {tif_path}")
            return str(tif_path)
        
    except Exception as e:
        print(f"❌ Error saving satellite image: {e}")
        return None


def get_surface_temperature(lat, lon, date=None):
    """
    Get surface temperature from MODIS thermal data.
    
    MODIS provides land surface temperature at 1km resolution.
    This is the actual thermal reading from space.
    
    Args:
        lat: Latitude
        lon: Longitude
        date: Date string in 'YYYY-MM-DD' format (defaults to yesterday)
        
    Returns:
        dict: {
            'temperature_celsius': float,
            'temperature_kelvin': float,
            'acquisition_date': str
        } or None if no data available
    """
    if date is None:
        date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    
    # Create date range (±7 days for better coverage)
    date_obj = datetime.strptime(date, "%Y-%m-%d")
    start_date = (date_obj - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
    
    point = ee.Geometry.Point(lon, lat)
    
    try:
        # Fetch MODIS Land Surface Temperature
        print(f"   Searching for MODIS data from {start_date} to {end_date}")
        modis_collection = (
            ee.ImageCollection("MODIS/061/MOD11A1")
            .filterBounds(point)
            .filterDate(start_date, end_date)
        )
        
        # Check if collection has any images
        count = modis_collection.size().getInfo()
        print(f"   Found {count} MODIS images")
        
        if count == 0:
            print("⚠️ No MODIS temperature data available for this date/location")
            return None
        
        modis = modis_collection.first()
        
        # Select daytime temperature band
        temp = modis.select("LST_Day_1km")
        
        # Convert from raw values to Celsius
        # MODIS LST is in Kelvin * 50, so: (value * 0.02) - 273.15
        temp_celsius = temp.multiply(0.02).subtract(273.15)
        
        # Get temperature value at the point with a buffer
        region = ee.Geometry.Point(lon, lat).buffer(1000)  # 1km buffer
        print(f"   Sampling temperature in 1km buffer region...")
        
        temp_value = temp_celsius.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=1000  # 1km resolution
        ).getInfo()
        
        print(f"   Temperature value received: {temp_value}")
        
        temp_c = temp_value.get('LST_Day_1km') if temp_value else None
        
        if temp_c is None or temp_c == 0:
            print("⚠️ No temperature reading at this location (null or zero value)")
            return None
        
        print(f"✅ Temperature: {temp_c:.2f}°C")
        
        return {
            'temperature_celsius': round(temp_c, 2),
            'temperature_kelvin': round(temp_c + 273.15, 2),
            'acquisition_date': date
        }
    except Exception as e:
        print(f"❌ Error fetching temperature data: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_hotspot(lat, lon, date=None):
    """
    Comprehensive hotspot analysis using Google Earth Engine.
    
    Combines:
    - Sentinel-2 RGB imagery (what it looks like)
    - MODIS temperature data (how hot it is)
    - Metadata (satellite source, date, etc.)
    
    This is the main function called by the API endpoint.
    
    Args:
        lat: Latitude
        lon: Longitude
        date: Date string in 'YYYY-MM-DD' format (defaults to yesterday)
        
    Returns:
        dict: Comprehensive hotspot analysis data
    """
    print(f"\n🛰️ Analyzing hotspot at ({lat}, {lon})")
    
    # Initialize GEE if not already done
    try:
        initialize_gee()
    except Exception as e:
        return {
            "error": "Google Earth Engine not authenticated",
            "message": "Run 'earthengine authenticate' to set up GEE access",
            "lat": lat,
            "lon": lon
        }
    
    # Fetch satellite image
    print("📡 Fetching Sentinel-2 imagery...")
    image = get_satellite_image(lat, lon, date)
    
    image_path = None
    image_url = None
    satellite_source = None
    cloud_coverage = None
    
    if image is not None:
        # Save RGB image
        print("💾 Saving RGB image...")
        image_path = save_rgb_image(image, lat, lon)
        
        if image_path:
            # Convert to relative URL for frontend
            image_filename = os.path.basename(image_path)
            image_url = f"/outputs/satellite_images/{image_filename}"
            satellite_source = "Sentinel-2"
            
            # Get cloud coverage
            try:
                cloud_coverage = image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
            except:
                cloud_coverage = None
    
    # Fetch temperature data
    print("🌡️ Fetching MODIS temperature data...")
    temp_data = get_surface_temperature(lat, lon, date)
    
    # Compile results
    result = {
        "lat": lat,
        "lon": lon,
        "satellite_image_url": image_url,
        "satellite_image_path": image_path,
        "satellite_source": satellite_source,
        "cloud_coverage": cloud_coverage,
        "temperature_data": temp_data,
        "acquisition_date": date or (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    print("✅ Hotspot analysis complete!")
    return result


# Cleanup old images (optional - run periodically)
def cleanup_old_images(days=7):
    """
    Remove satellite images older than specified days.
    
    Args:
        days: Number of days to keep images
    """
    cutoff = datetime.now() - timedelta(days=days)
    
    for image_file in OUTPUT_DIR.glob("sat_*.png"):
        if image_file.stat().st_mtime < cutoff.timestamp():
            image_file.unlink()
            print(f"🗑️ Deleted old image: {image_file.name}")
