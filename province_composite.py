"""Load full-province, water-masked Sentinel-2 composites exported by
refresh_ubon_data.py / backfill_ubon_weekly.py (files named Ubon_S2_YYYYMMDD.tif,
8 bands already in turbidity_model.FEATURES order, computed server-side in GEE
with the same NDWI/MNDWI/NDTI/NDSSI definitions the trained model expects - no
local index recomputation needed).

Two sources, tried in order:
  1. Local disk (fast path - what the Task Scheduler refresh writes to on the
     dev machine).
  2. Google Drive, via drive_client.py (what a cloud deployment uses, since it
     has no persistent local disk of its own - see drive_client.py for why
     Drive rather than Cloud Storage).
"""
import datetime as dt
import glob
import os
import re

import numpy as np
import rasterio

import turbidity_model as tm

FILENAME_RE = re.compile(r"Ubon_S2_(\d{8})\.tif$")
CACHE_DIR = ".composite_cache"
DISPLAY_CACHE_DIR = "display_rasters"

# Ground resolution the map's turbidity raster is drawn at, in metres.
#
# A distance rather than a pixel budget, which is what this used to be
# (max_dim=1400 on the long side). That was wrong in a way that hid itself:
# the composites are ~20m, 6986x10505, so a 1400px cap downsampled them 8x and
# the map drew 155m pixels - the Mun River, 200-400m wide, was one to three
# pixels across. Raising the GEE export scale would not have helped either,
# since a finer source only made the factor larger.
#
# Derived from each file's own pixel size, so the two ends stay independent:
# change EXPORT_SCALE in refresh_ubon_data.py and the display grid, and what
# it costs to ship, stay put.
#
# Why 40 and not finer. The whole province is drawn as ONE PNG overlay, so the
# cost is set by the full grid, not by the ~1% of it that is water. Measured
# per composite, on the wettest date:
#
#     155m (old)   0.9 Mpx    0.1MB on disk    65KB in the payload
#      40m         18 Mpx     2-4MB            ~300KB
#      20m         73 Mpx     1.3-4.2MB        ~1.2MB
#
# 20m works but is not safe: colouring a 73 Mpx grid transiently needs ~850MB
# (the BoundaryNorm index alone is 560MB of int64), which overruns the 1GB a
# Streamlit Community Cloud container gets. 40m needs ~220MB. Since the
# archive is ~20m anyway, 40m also means each display pixel averages a 2x2
# block rather than resampling one - slightly less noisy, at 4x the detail the
# map had before.
DISPLAY_RESOLUTION_M = 40


def list_available_composites(folder="."):
    """Returns [(date, path), ...] sorted by date. Local files if any exist;
    otherwise falls back to listing (not downloading) what's in Drive - see
    ensure_local() to actually fetch one of those for use."""
    out = _scan_local(folder)
    if out:
        return out
    return _list_remote()


def _scan_local(folder):
    out = []
    for path in glob.glob(os.path.join(folder, "Ubon_S2_*.tif")):
        m = FILENAME_RE.search(os.path.basename(path))
        if not m:
            continue
        d = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
        out.append((d, path))
    return sorted(out, key=lambda t: t[0])


def _list_remote():
    import drive_client
    out = []
    for f in drive_client.list_remote_composites():
        m = FILENAME_RE.search(f["name"])
        if not m:
            continue
        d = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
        out.append((d, f"drive:{f['id']}:{f['name']}"))
    return sorted(out, key=lambda t: t[0])


def ensure_local(path: str) -> str:
    """If `path` is a local file, return it as-is. If it's a "drive:<id>:<name>"
    marker (from the Drive fallback above), download it to a local cache
    directory (once) and return that local path."""
    if not path.startswith("drive:"):
        return path
    _, file_id, filename = path.split(":", 2)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cached = os.path.join(CACHE_DIR, filename)
    if os.path.exists(cached):
        return cached
    import drive_client
    return drive_client.download_file(file_id, filename, CACHE_DIR)


def source_resolution_m(src) -> float:
    """One source pixel, in metres of ground.

    Read from the file rather than assumed: these are written in EPSG:4326, so
    the pixel size on disk is in degrees and depends on the scale the GEE
    export used. Latitude is used because a degree of it is very nearly
    constant, where a degree of longitude is not.
    """
    if src.crs and src.crs.is_geographic:
        return abs(src.transform.e) * 110_540
    return abs(src.transform.e)


def downsample_factor(src, target_res_m=DISPLAY_RESOLUTION_M) -> int:
    """How many source pixels go into one display pixel - see DISPLAY_RESOLUTION_M."""
    return max(1, int(round(target_res_m / source_resolution_m(src))))


def load_composite(path, target_res_m=DISPLAY_RESOLUTION_M, strip_rows=500):
    """Returns (turbidity_map, valid_mask, bounds) at ~`target_res_m` per pixel.

    Reads and predicts the file in horizontal row-strips instead of loading
    the whole province (70M+ pixels x 8 bands) into memory at once, which
    reliably exhausts memory in a long-running server process even though
    the file itself is only ~20MB on disk. Each strip is downsampled ("any
    water in block" for the mask, masked-mean for turbidity) immediately
    after prediction, so peak memory stays at one strip's worth.
    """
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        bounds = src.bounds

        factor = downsample_factor(src, target_res_m)
        strip_rows = max(factor, (strip_rows // factor) * factor)  # keep divisible by factor

        out_h = h // factor
        out_w = w // factor
        turb_sum = np.zeros((out_h, out_w), dtype=np.float64)
        turb_count = np.zeros((out_h, out_w), dtype=np.int32)
        mask_down = np.zeros((out_h, out_w), dtype=bool)

        out_row = 0
        for row0 in range(0, (h // factor) * factor, strip_rows):
            rows = min(strip_rows, h - row0)
            rows = (rows // factor) * factor
            if rows == 0:
                break
            window = rasterio.windows.Window(0, row0, w, rows)
            strip = src.read(window=window)  # (8, rows, w)
            valid_strip = np.isfinite(strip).all(axis=0)

            flat = strip.reshape(strip.shape[0], -1).T
            valid_flat = valid_strip.ravel()
            turb_flat = np.full(flat.shape[0], np.nan, dtype=np.float32)
            if valid_flat.any():
                turb_flat[valid_flat] = tm.predict(flat[valid_flat]).astype(np.float32)
            turb_strip = turb_flat.reshape(rows, w)

            out_rows = rows // factor
            usable_w = out_w * factor
            turb_strip = turb_strip[:, :usable_w]
            valid_strip = valid_strip[:, :usable_w]

            mask_blocks = valid_strip.reshape(out_rows, factor, out_w, factor)
            mask_down[out_row:out_row + out_rows] = mask_blocks.any(axis=(1, 3))

            turb_blocks = np.nan_to_num(turb_strip.reshape(out_rows, factor, out_w, factor), nan=0.0)
            turb_sum[out_row:out_row + out_rows] = turb_blocks.sum(axis=(1, 3))
            turb_count[out_row:out_row + out_rows] = mask_blocks.sum(axis=(1, 3))

            out_row += out_rows

    turbidity_map = np.divide(turb_sum, turb_count, out=np.zeros_like(turb_sum), where=turb_count > 0)
    return turbidity_map.astype(np.float32), mask_down, bounds


# This used to also build and return a stretched RGB composite, from the B4/B3/B2
# bands, alongside the turbidity. Nothing ever drew it: both callers unpacked it
# into a discard. It was harmless at the old 155m grid and is not at 20m, where
# the array it allocates is 880MB - enough on its own to put a precompute or a
# cold cloud render over the memory limit, for a picture no one looks at.


# ------------------------------------------------------- display rasters ---
# load_composite() runs the MLP over every pixel of the full-resolution
# province (~70M pixels, 8 bands) and then averages the predictions down to
# the ~1300x900 grid the map actually draws. That is ~11s of inference per
# composite, and on a cloud host it is preceded by a ~28MB Drive download of
# the GeoTIFF - paid again for every date a visitor clicks, not just at
# startup.
#
# Nothing downstream of the dashboard needs the full-resolution result, or
# the RGB channels at all. So the downsampled output is precomputed once
# (precompute_history.py) into a small .npz per date and shipped in the repo.
# Only ~1% of the province is water, so the arrays compress to ~80KB each and
# read back in ~20ms - the GeoTIFF never has to be opened, or downloaded, to
# render the map.

def display_cache_path(path: str) -> str | None:
    """Where the precomputed display raster for `path` lives, or None if the
    filename isn't a recognisable composite. Accepts both a local .tif path
    and a "drive:<id>:<name>" marker, so it keys the same either way."""
    m = FILENAME_RE.search(os.path.basename(path))
    if not m:
        return None
    return os.path.join(DISPLAY_CACHE_DIR, f"Ubon_S2_{m.group(1)}.npz")


def save_display(path: str, turbidity, mask, bounds) -> str | None:
    """Write the display raster for `path`. Returns the file written."""
    out = display_cache_path(path)
    if out is None:
        return None
    os.makedirs(DISPLAY_CACHE_DIR, exist_ok=True)
    np.savez_compressed(
        out,
        turbidity=turbidity.astype(np.float32),
        mask=mask,
        bounds=np.array([bounds.left, bounds.bottom, bounds.right, bounds.top], dtype=np.float64),
    )
    return out


def load_display(path: str):
    """(turbidity, mask, bounds) for one composite - the map's view of it.

    Reads the precomputed raster when one exists and otherwise falls back to
    load_composite(), so a composite added since the last precompute run is
    still correct, just as slow as it used to be.
    """
    cached = display_cache_path(path)
    if cached and os.path.exists(cached):
        with np.load(cached) as z:
            left, bottom, right, top = z["bounds"]
            return z["turbidity"], z["mask"], rasterio.coords.BoundingBox(left, bottom, right, top)
    return load_composite(ensure_local(path))


def sample_at(turbidity_map, valid_mask, bounds, lat, lon, search_radius=5):
    """Turbidity at (lat, lon) for this composite, or None if no valid (water)
    pixel exists within `search_radius` pixels of that point - stations sit
    right at the water's edge, and the downsampled/simplified mask sometimes
    misses them by a pixel or two.
    """
    h, w = turbidity_map.shape
    col = int((lon - bounds.left) / (bounds.right - bounds.left) * w)
    row = int((bounds.top - lat) / (bounds.top - bounds.bottom) * h)
    if not (0 <= row < h and 0 <= col < w):
        return None
    for r in range(search_radius + 1):
        r0, r1 = max(0, row - r), min(h, row + r + 1)
        c0, c1 = max(0, col - r), min(w, col + r + 1)
        window_mask = valid_mask[r0:r1, c0:c1]
        if window_mask.any():
            window_vals = turbidity_map[r0:r1, c0:c1]
            return float(window_vals[window_mask].mean()) if r > 0 else float(turbidity_map[row, col])
    return None
