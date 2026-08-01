"""End-to-end refresh: pull the most recent usable Sentinel-2 composite for
Ubon Ratchathani from Google Earth Engine, export it, and download the
GeoTIFF locally - no manual Drive download step.

Uses the same OAuth credentials Earth Engine already has cached (its granted
scopes include Drive access), so no separate Drive auth is needed.

Run manually:  python refresh_ubon_data.py
Meant to be re-run on a schedule (e.g. every 7 days) to keep the dashboard's
province-wide turbidity layer current.
"""
import datetime as dt
import io
import json
import os
import time

import ee
import ee.oauth as oauth
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

PROJECT = "gee-training-498303"
DRIVE_FOLDER_NAME = "GEE_Ubon_Turbidity"
LOCAL_OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["B2", "B3", "B4", "B8", "NDWI", "MNDWI", "NDTI", "NDSSI"]
EXPORT_SCALE = 20  # meters; full 10m province-wide export is very large (see refresh notes)
WINDOW_DAYS = 7
MAX_LOOKBACK_DAYS = 35
LATEST_POINTER = "ubon_latest_composite.txt"


def init_ee():
    ee.Initialize(project=PROJECT)


def drive_service():
    cred_path = os.path.expanduser("~/.config/earthengine/credentials")
    with open(cred_path) as f:
        d = json.load(f)
    creds = Credentials(
        token=None,
        refresh_token=d["refresh_token"],
        client_id=oauth.CLIENT_ID,
        client_secret=oauth.CLIENT_SECRET,
        token_uri="https://oauth2.googleapis.com/token",
        scopes=d["scopes"],
    )
    return build("drive", "v3", credentials=creds)


def process_s2(image):
    scl = image.select("SCL")
    mask_out = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    img_clean = image.updateMask(mask_out).divide(10000)
    ndwi = img_clean.normalizedDifference(["B3", "B8"]).rename("NDWI")
    water_mask = ndwi.gt(0)
    mndwi = img_clean.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    ndti = img_clean.normalizedDifference(["B4", "B3"]).rename("NDTI")
    ndssi = img_clean.normalizedDifference(["B8", "B4"]).rename("NDSSI")  # matches trained model
    bands = img_clean.select(["B2", "B3", "B4", "B8"]).addBands([ndwi, mndwi, ndti, ndssi])
    return bands.updateMask(water_mask).copyProperties(image, ["system:time_start"])


def build_composite(end_date):
    provinces = ee.FeatureCollection("FAO/GAUL/2015/level1")
    ubon = provinces.filter(ee.Filter.eq("ADM1_NAME", "Ubon Ratchathani"))

    lookback = WINDOW_DAYS
    while lookback <= MAX_LOOKBACK_DAYS:
        start = end_date.advance(-lookback, "day")
        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(ubon)
            .filterDate(start, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
            .map(process_s2)
        )
        size = col.size().getInfo()
        print(f"  window {lookback}d: {size} scene(s)")
        if size > 0:
            composite = col.median().clip(ubon).select(FEATURES)
            return composite, ubon, lookback, size
        lookback += WINDOW_DAYS
    raise RuntimeError("No usable Sentinel-2 scenes found even after extending the lookback window")


def export_and_download(label):
    print("Initializing Earth Engine...")
    init_ee()
    end_date = ee.Date(dt.datetime.utcnow().strftime("%Y-%m-%d"))
    print("Searching for usable imagery...")
    composite, ubon, lookback, n_images = build_composite(end_date)

    filename = f"Ubon_S2_{label}"
    print(f"Submitting export: {filename} (scale={EXPORT_SCALE}m, lookback={lookback}d, scenes={n_images})")
    task = ee.batch.Export.image.toDrive(
        image=composite,
        description=filename,
        fileNamePrefix=filename,
        folder=DRIVE_FOLDER_NAME,
        scale=EXPORT_SCALE,
        region=ubon.geometry(),
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()

    while True:
        status = task.status()
        state = status.get("state")
        print(f"  task state: {state}")
        if state in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        time.sleep(15)

    if state != "COMPLETED":
        raise RuntimeError(f"Export task did not complete: {status}")

    print("Export complete, downloading from Drive...")
    drive = drive_service()
    query = f"name = '{filename}.tif' and trashed = false"
    results = drive.files().list(q=query, fields="files(id, name, size)").execute()
    files = results.get("files", [])
    if not files:
        raise RuntimeError(f"Exported file {filename}.tif not found in Drive")
    file_id = files[0]["id"]
    out_path = os.path.join(LOCAL_OUT_DIR, f"{filename}.tif")
    request = drive.files().get_media(fileId=file_id)
    with io.FileIO(out_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Downloaded to {out_path} ({size_mb:.1f} MB)")

    with open(os.path.join(LOCAL_OUT_DIR, LATEST_POINTER), "w") as f:
        f.write(f"{filename}.tif")

    return out_path


if __name__ == "__main__":
    label = dt.datetime.utcnow().strftime("%Y%m%d")
    export_and_download(label)
