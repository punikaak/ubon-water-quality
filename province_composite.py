"""Load full-province, water-masked Sentinel-2 composites exported by
refresh_ubon_data.py / backfill_ubon_weekly.py (files named Ubon_S2_YYYYMMDD.tif
in this folder, 8 bands already in turbidity_model.FEATURES order, computed
server-side in GEE with the same NDWI/MNDWI/NDTI/NDSSI definitions the trained
model expects - no local index recomputation needed).
"""
import datetime as dt
import glob
import os
import re

import numpy as np
import rasterio

import turbidity_model as tm

FILENAME_RE = re.compile(r"Ubon_S2_(\d{8})\.tif$")


def list_available_composites(folder="."):
    """Returns [(date, path), ...] sorted by date, for every Ubon_S2_*.tif present."""
    out = []
    for path in glob.glob(os.path.join(folder, "Ubon_S2_*.tif")):
        m = FILENAME_RE.search(os.path.basename(path))
        if not m:
            continue
        d = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
        out.append((d, path))
    return sorted(out, key=lambda t: t[0])


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
