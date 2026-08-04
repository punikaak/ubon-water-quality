"""Precompute the per-composite aggregates the dashboard's sidebar needs, so
the deployed app doesn't have to open every composite on a cold start.

Why this exists
---------------
Two sidebar series - the province-wide trend and the per-station turbidity
trend - are defined over *every* composite in range, not just the selected
one. Computing them live means loading all of them: locally that is ~12s of
MLP inference each; on Streamlit Community Cloud it is that plus a ~28MB
Google Drive download each, because the cloud container has no persistent
disk. Nine composites put a 4-6 minute blank screen in front of every cold
visitor, and the free tier sleeps an idle app, so "cold" is the common case.

The aggregates themselves are tiny - one float per (date) and per
(station, date). Writing them to a small JSON that ships in the repo turns
that cold start into a single composite load (the selected date, which the
map needs regardless).

Run this locally whenever new composites land:

    python precompute_history.py

It reads whatever Ubon_S2_*.tif files it can see (local disk first, Drive
otherwise) and rewrites ubon_history.json. The dashboard treats the file as
a cache, not a source of truth: any composite date missing from it is still
computed live, so a newly refreshed week shows up correctly before anyone
gets around to re-running this.
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
