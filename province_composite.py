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


def load_composite(path, max_dim=1400, strip_rows=500):
    """Returns (rgb, turbidity_map, valid_mask, bounds), already downsampled to
    at most `max_dim` on the long side.

    Reads and predicts the file in horizontal row-strips instead of loading
    the whole province (70M+ pixels x 8 bands) into memory at once, which
    reliably exhausts memory in a long-running server process even though
    the file itself is only ~20MB on disk. Each strip is downsampled (mean
    for color, "any water in block" for the mask, masked-mean for turbidity)
    immediately after prediction, so peak memory stays at one strip's worth.
    """
    with rasterio.open(path) as src:
        h, w = src.height, src.width
        bounds = src.bounds
        b4_idx = tm.FEATURES.index("B4")
        b3_idx = tm.FEATURES.index("B3")
        b2_idx = tm.FEATURES.index("B2")

        factor = max(1, int(np.ceil(max(h, w) / max_dim)))
        strip_rows = max(factor, (strip_rows // factor) * factor)  # keep divisible by factor

        out_h = h // factor
        out_w = w // factor
        rgb_raw = np.zeros((out_h, out_w, 3), dtype=np.float32)
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
            rgb_strip = np.dstack([strip[b4_idx], strip[b3_idx], strip[b2_idx]])[:, :usable_w]
            turb_strip = turb_strip[:, :usable_w]
            valid_strip = valid_strip[:, :usable_w]

            rgb_blocks = rgb_strip.reshape(out_rows, factor, out_w, factor, 3)
            rgb_raw[out_row:out_row + out_rows] = np.nan_to_num(rgb_blocks, nan=0.0).mean(axis=(1, 3))

            mask_blocks = valid_strip.reshape(out_rows, factor, out_w, factor)
            mask_down[out_row:out_row + out_rows] = mask_blocks.any(axis=(1, 3))

            turb_blocks = np.nan_to_num(turb_strip.reshape(out_rows, factor, out_w, factor), nan=0.0)
            turb_sum[out_row:out_row + out_rows] = turb_blocks.sum(axis=(1, 3))
            turb_count[out_row:out_row + out_rows] = mask_blocks.sum(axis=(1, 3))

            out_row += out_rows

    turbidity_map = np.divide(turb_sum, turb_count, out=np.zeros_like(turb_sum), where=turb_count > 0)

    def stretch(b, low=2, high=98):
        valid = b[mask_down]
        if valid.size == 0:
            return np.zeros_like(b)
        lo, hi = np.percentile(valid, [low, high])
        return np.clip((b - lo) / (hi - lo + 1e-9), 0, 1)

    rgb = np.dstack([stretch(rgb_raw[..., 0]), stretch(rgb_raw[..., 1]), stretch(rgb_raw[..., 2])])
    return rgb, turbidity_map.astype(np.float32), mask_down, bounds


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
    still correct, just as slow as it used to be. Deliberately drops the RGB
    channels load_composite() also returns: no caller uses them.
    """
    cached = display_cache_path(path)
    if cached and os.path.exists(cached):
        with np.load(cached) as z:
            left, bottom, right, top = z["bounds"]
            return z["turbidity"], z["mask"], rasterio.coords.BoundingBox(left, bottom, right, top)
    _rgb, turbidity, mask, bounds = load_composite(ensure_local(path))
    return turbidity, mask, bounds


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
