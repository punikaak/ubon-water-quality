"""Export water-masked, full-province Sentinel-2 composites for Ubon
Ratchathani, using Google Earth Engine (GEE) - adapted from a JS Code Editor
script the user wrote, with fixes so the output is compatible with the
already-trained turbidity model in this project, plus a rolling weekly mode
for recurring refreshes.

FEATURE-DEFINITION FIX (important): the original JS script computes
NDSSI as normalizedDifference(['B2','B8']). But best_model_neural_network_mlp.pkl
(see turbidity_model.py) was trained on NDSSI = normalizedDifference(['B8','B4'])
- the definition used throughout this project (WQ_Project.py, geotiff_train.py).
Feeding the model a differently-defined "NDSSI" band would silently produce
wrong predictions (same band name, different physical quantity). This script
keeps the ORIGINAL definition. The JS script also only computed NDWI + NDSSI;
the model needs all 8 features (B2,B3,B4,B8,NDWI,MNDWI,NDTI,NDSSI), so B11 and
the missing indices are added.

Boundary: FAO/GAUL/2015/level1 (official UN dataset, more authoritative than
the OpenStreetMap boundary geo_boundary.py falls back to).

TWO EXPORT MODES:
  - "rolling" (default): composite of the most recent WINDOW_DAYS days, for a
    scheduled run every ~7 days so the dashboard always has fresh imagery.
    Sentinel-2 revisits Ubon every ~5 days, so a 10-day window balances
    freshness against having enough cloud-free pixels to composite.
  - "fixed_months": specific calendar months (e.g. dry vs wet season), for a
    one-off seasonal comparison.

RUN THIS FROM A GEE-AUTHENTICATED ENVIRONMENT (this sandbox has none - see
chat). After running:
  1. Check task progress at https://code.earthengine.google.com/tasks
  2. Once each task finishes, the GeoTIFF appears in Google Drive under
     "GEE_Ubon_Turbidity" - if Google Drive for Desktop is syncing that
     folder locally, it will show up on disk automatically.
  3. Tell the dashboard maintainer the local sync path once, and every future
     run just needs this script re-run (e.g. on a 7-day schedule) - no more
     manual steps after that.
"""
import datetime as dt

import ee

try:
    ee.Initialize(project="gee-training-498303")
except Exception:
    ee.Authenticate()
    ee.Initialize(project="gee-training-498303")

# =========================================================================
# 1. Study area: Ubon Ratchathani province (FAO/GAUL - official UN boundary)
# =========================================================================
provinces = ee.FeatureCollection("FAO/GAUL/2015/level1")
ubon = provinces.filter(ee.Filter.eq("ADM1_NAME", "Ubon Ratchathani"))

# =========================================================================
# 2. Cloud/shadow masking (SCL) + water-only masking (NDWI), all 8 model
#    features computed with THIS project's existing definitions.
# =========================================================================
def process_s2(image):
    scl = image.select("SCL")
    # 3=cloud shadow, 8=cloud medium prob, 9=cloud high prob, 10=cirrus
    mask_out = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    img_clean = image.updateMask(mask_out).divide(10000)

    ndwi = img_clean.normalizedDifference(["B3", "B8"]).rename("NDWI")
    water_mask = ndwi.gt(0)

    mndwi = img_clean.normalizedDifference(["B3", "B11"]).rename("MNDWI")
    ndti = img_clean.normalizedDifference(["B4", "B3"]).rename("NDTI")
    # NDSSI kept as (B8-B4)/(B8+B4) to match the already-trained model - see
    # module docstring. (The JS draft used (B2-B8)/(B2+B8) instead.)
    ndssi = img_clean.normalizedDifference(["B8", "B4"]).rename("NDSSI")

    bands = img_clean.select(["B2", "B3", "B4", "B8"]).addBands([ndwi, mndwi, ndti, ndssi])
    water_only = bands.updateMask(water_mask)
    return water_only.copyProperties(image, ["system:time_start"])


FEATURES = ["B2", "B3", "B4", "B8", "NDWI", "MNDWI", "NDTI", "NDSSI"]
DRIVE_FOLDER = "GEE_Ubon_Turbidity"

# =========================================================================
# 3. Configure export mode
# =========================================================================
MODE = "rolling"  # "rolling" or "fixed_months"
WINDOW_DAYS = 10  # rolling mode: composite of the last N days (S2 revisit ~5 days)
YEAR = 2024        # fixed_months mode only
EXPORT_MONTHS = [2, 8]  # fixed_months mode only: dry season (Feb), wet season (Aug)


def export_composite(image, name):
    task = ee.batch.Export.image.toDrive(
        image=image.select(FEATURES),
        description=name,
        fileNamePrefix=name,
        folder=DRIVE_FOLDER,
        scale=10,
        region=ubon.geometry(),
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )
    task.start()
    print(f"Started export task: {name}  (Google Drive folder: {DRIVE_FOLDER})")


if MODE == "rolling":
    today = dt.date.today()
    end = ee.Date(today.isoformat())
    start = end.advance(-WINDOW_DAYS, "day")

    s2_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(ubon)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        .map(process_s2)
    )
    composite = s2_collection.median().clip(ubon)
    name = f"Ubon_S2_rolling_{today.isoformat()}"
    export_composite(composite, name)

elif MODE == "fixed_months":
    start_date = ee.Date.fromYMD(YEAR, 1, 1)
    end_date = ee.Date.fromYMD(YEAR, 12, 31)
    s2_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(ubon)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        .map(process_s2)
    )
    for m in EXPORT_MONTHS:
        month_start = ee.Date.fromYMD(YEAR, m, 1)
        month_end = month_start.advance(1, "month")
        composite = s2_collection.filterDate(month_start, month_end).median().clip(ubon)
        export_composite(composite, f"Ubon_S2_{YEAR}_{m:02d}")

else:
    raise ValueError(f"Unknown MODE: {MODE}")

print("\nMonitor progress at https://code.earthengine.google.com/tasks")
