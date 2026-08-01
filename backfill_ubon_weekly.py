"""Backfill historical weekly (7-day) Sentinel-2 composites for Ubon
Ratchathani, Nov 2024 - Jan 2025 - the period matching the PCD field-visit
data already in this project (Sentinel2_Extract_Ubon_New.csv spans Feb 2024
to Feb 2025). Populates real historical dates for the dashboard's time
slider / date picker, instead of the single Nov 2024 scene it had before.

Reuses the water-masking/feature pipeline from refresh_ubon_data.py. Submits
all export tasks up front (GEE runs them concurrently server-side), then
polls and downloads each as it finishes.

Run manually:  python backfill_ubon_weekly.py
"""
import datetime as dt
import io
import os
import time

import ee

import refresh_ubon_data as core

START = dt.date(2024, 11, 1)
END = dt.date(2025, 1, 31)
WINDOW_DAYS = 7


def week_windows(start, end, step_days):
    windows = []
    cur = start
    while cur < end:
        nxt = min(cur + dt.timedelta(days=step_days), end)
        windows.append((cur, nxt))
        cur = nxt
    return windows


def submit_all():
    core.init_ee()
    provinces = ee.FeatureCollection("FAO/GAUL/2015/level1")
    ubon = provinces.filter(ee.Filter.eq("ADM1_NAME", "Ubon Ratchathani"))

    tasks = []
    for start, end in week_windows(START, END, WINDOW_DAYS):
        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(ubon)
            .filterDate(start.isoformat(), end.isoformat())
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
            .map(core.process_s2)
        )
        size = col.size().getInfo()
        label = f"Ubon_S2_{start.strftime('%Y%m%d')}"
        print(f"{start} - {end}: {size} scene(s)", "-> skipped (no imagery)" if size == 0 else "")
        if size == 0:
            continue
        composite = col.median().clip(ubon).select(core.FEATURES)
        task = ee.batch.Export.image.toDrive(
            image=composite,
            description=label,
            fileNamePrefix=label,
            folder=core.DRIVE_FOLDER_NAME,
            scale=core.EXPORT_SCALE,
            region=ubon.geometry(),
            maxPixels=1e13,
            fileFormat="GeoTIFF",
        )
        task.start()
        tasks.append((label, task, start))
        print(f"  submitted: {label}")
    return tasks


def wait_and_download(tasks):
    downloaded = []
    pending = list(tasks)
    while pending:
        still_pending = []
        for label, task, start in pending:
            state = task.status().get("state")
            print(f"  {label}: {state}")
            if state == "COMPLETED":
                path = download_one(label)
                downloaded.append((label, start, path))
            elif state in ("FAILED", "CANCELLED"):
                print(f"  {label} FAILED: {task.status()}")
            else:
                still_pending.append((label, task, start))
        pending = still_pending
        if pending:
            time.sleep(20)
    return downloaded


def download_one(label):
    drive = core.drive_service()
    query = f"name = '{label}.tif' and trashed = false"
    results = drive.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])
    if not files:
        raise RuntimeError(f"{label}.tif not found in Drive")
    file_id = files[0]["id"]
    out_path = os.path.join(core.LOCAL_OUT_DIR, f"{label}.tif")
    request = drive.files().get_media(fileId=file_id)
    with io.FileIO(out_path, "wb") as fh:
        from googleapiclient.http import MediaIoBaseDownload
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    print(f"  downloaded: {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB)")
    return out_path


if __name__ == "__main__":
    tasks = submit_all()
    print(f"\n{len(tasks)} export task(s) submitted. Waiting for completion...")
    results = wait_and_download(tasks)
    print(f"\nDone. {len(results)} composite(s) downloaded:")
    for label, start, path in sorted(results, key=lambda r: r[1]):
        print(f"  {start.isoformat()} -> {path}")
