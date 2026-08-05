"""Precompute everything the deployed dashboard would otherwise derive from
the GeoTIFFs at request time. Writes two artifacts, both committed:

  ubon_history.json  - the sidebar's province and per-station trend values
  display_rasters/   - the downsampled turbidity raster the map draws

Why this exists
---------------
Opening one composite costs ~11s of MLP inference over the full-resolution
province, and on Streamlit Community Cloud it is preceded by a ~28MB Google
Drive download, because the cloud container has no persistent disk.

That was being paid twice over. The two sidebar series are defined over
*every* composite in range, so drawing them live meant loading all twelve -
a 4-6 minute blank screen on a cold visit. And the map's own raster was
recomputed from scratch for every date a visitor clicked, so simply stepping
along the timeline cost ~12s a step locally and far worse on the cloud.

Both outputs are far smaller than their inputs: the aggregates are one float
per date, and only ~1% of the province is water, so a display raster
compresses to ~80KB (vs ~28MB for the GeoTIFF it came from) and loads in
~20ms. Shipping them in the repo takes the GeoTIFFs off the render path
entirely - the deployed app never opens or downloads one.

Run this locally whenever new composites land:

    python precompute_history.py

It reads whatever Ubon_S2_*.tif files it can see (local disk first, Drive
otherwise), rewrites ubon_history.json and repopulates display_rasters/.
The dashboard treats both as caches, not sources of truth: any composite
date missing from them is still computed live, so a newly refreshed week
shows up correctly before anyone gets around to re-running this.
"""
import datetime as dt
import json
import sys

import pandas as pd

import province_composite as pc

HISTORY_PATH = "ubon_history.json"
VALIDATION_CSV = "Sentinel2_Extract_Ubon_New.csv"


def station_key(lat: float, lon: float) -> str:
    """Stable dict key for a station coordinate. Rounded because the value is
    a JSON key on one side and a float from pandas on the other; 5 decimals is
    ~1m, far finer than the stations are distinguishable."""
    return f"{lat:.5f},{lon:.5f}"


def load_stations() -> list[tuple[str, float, float]]:
    df = pd.read_csv(VALIDATION_CSV)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    grouped = df.groupby("Code", as_index=False).agg(
        station_la=("station_la", "first"), station_lo=("station_lo", "first"))
    return [(r.Code, float(r.station_la), float(r.station_lo)) for r in grouped.itertuples()]


def main() -> int:
    stations = load_stations()
    composites = pc.list_available_composites(".")
    if not composites:
        print("No composites found (local or Drive) - nothing to precompute.")
        return 1

    print(f"{len(composites)} composites, {len(stations)} stations")
    province: dict[str, float] = {}
    per_station: dict[str, dict[str, float]] = {station_key(la, lo): {} for _, la, lo in stations}

    for i, (date, path) in enumerate(composites, start=1):
        iso = date.isoformat()
        print(f"  [{i}/{len(composites)}] {iso} ...", end="", flush=True)
        local = pc.ensure_local(path)
        _rgb, turb, mask, bounds = pc.load_composite(local)
        # Same pass also writes the map's display raster - see
        # province_composite.load_display() for why the dashboard reads that
        # instead of the GeoTIFF.
        pc.save_display(path, turb, mask, bounds)

        if mask.any():
            province[iso] = float(turb[mask].mean())
        for _code, lat, lon in stations:
            val = pc.sample_at(turb, mask, bounds, lat, lon)
            if val is not None:
                per_station[station_key(lat, lon)][iso] = val
        print(f" province mean {province.get(iso, float('nan')):.1f} NTU")

    payload = {
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
        "province": province,
        "stations": per_station,
    }
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    print(f"wrote {HISTORY_PATH} ({len(province)} dates)")
    print(f"wrote {len(composites)} display rasters to {pc.DISPLAY_CACHE_DIR}/ - commit both")
    return 0


if __name__ == "__main__":
    sys.exit(main())
