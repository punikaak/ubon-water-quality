"""Reference-layer data for the Mekong Water Quality dashboard.

- Station place names: OpenStreetMap Nominatim reverse geocoding, cached to
  station_locations.json.
- Major roads (trunk + primary) within Ubon: OpenStreetMap Overpass, cached
  to ubon_roads.json.

No administrative boundary geometry
-----------------------------------
There is none here, and none anywhere else in this project. Province and
district outlines - and the separate Ubon Ratchathani outline that used to
supply the map's fit bounds - have all been removed, along with the local
shapefile archives they were built from and the script that converted them.

Nothing was moved elsewhere or left disabled behind a flag: the map draws no
boundaries, and the per-district turbidity ranking that needed district
polygons for its zonal statistics is gone rather than stubbed. The map's
opening rectangle is now four literal numbers in dashboard.py.

Note that the place names below are *not* boundary data - Nominatim returns
them as text for a point, with no geometry attached.

All fetches are one-time; if a cache file already exists on disk it's used
as-is (delete the file to force a re-fetch).
"""
import functools
import json
import os
import time

import requests

NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"

STATION_LOCATIONS_CACHE = "station_locations.json"

ROADS_CACHE = "ubon_roads.json"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
UBON_RELATION_ID = 1908830  # OSM relation id, used only to scope the road query


def station_locations(stations, lang="en"):
    """{code: "Tambon, Amphoe, Province"} in the requested language ("en" or
    "th") for each (code, lat, lon) in `stations`, via Nominatim reverse
    geocoding (no local data source has sub-district detail for these
    stations). Cached to disk as {code: {"en": ..., "th": ...}} - both
    languages are fetched together the first time a code is seen, so the
    TH/EN toggle never needs a fresh network call after that. Only hits the
    network for codes not already in the cache file, respecting Nominatim's
    1 req/sec policy on those.
    """
    cache = {}
    if os.path.exists(STATION_LOCATIONS_CACHE):
        with open(STATION_LOCATIONS_CACHE, encoding="utf-8") as f:
            cache = json.load(f)

    missing = [(code, lat, lon) for code, lat, lon in stations if code not in cache]
    if missing:
        headers = {"User-Agent": "MekongWaterQualityDashboard/1.0 (research project)"}
        request_count = 0
        for code, lat, lon in missing:
            entry = {}
            for lc in ("en", "th"):
                if request_count:
                    time.sleep(1)
                request_count += 1
                resp = requests.get(
                    NOMINATIM_REVERSE_URL,
                    params={"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 15, "addressdetails": 1,
                            "accept-language": lc},
                    headers=headers, timeout=20,
                )
                resp.raise_for_status()
                addr = resp.json().get("address", {})
                tambon = addr.get("suburb") or addr.get("village") or addr.get("hamlet") or addr.get("town") or ""
                amphoe = addr.get("county") or addr.get("state_district") or ""
                province = addr.get("province") or addr.get("state") or ""
                entry[lc] = ", ".join(p for p in (tambon, amphoe, province) if p)
            cache[code] = entry
        with open(STATION_LOCATIONS_CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    return {code: cache.get(code, {}).get(lang, "") for code, _, _ in stations}


def station_location_parts(stations, lang="en"):
    """{code: {"subdistrict": ..., "district": ..., "province": ...}} - the
    same reverse-geocoded place as station_locations(), split back into its
    administrative levels so a caller can label them individually.

    Splitting the joined string rather than storing the parts separately:
    the on-disk cache holds the joined form (see station_locations), and
    re-fetching every station to change that shape would cost a Nominatim
    call per station per language for no new information. The join is ours
    and uses ", ", which none of the components contain.

    Nominatim yields either two levels (district, province) or three
    (subdistrict as well), so the parts are read from the right-hand end,
    where the province always sits.
    """
    out = {}
    for code, joined in station_locations(stations, lang=lang).items():
        parts = [p.strip() for p in joined.split(",") if p.strip()]
        out[code] = {
            "province": parts[-1] if len(parts) >= 1 else "",
            "district": parts[-2] if len(parts) >= 2 else "",
            "subdistrict": parts[-3] if len(parts) >= 3 else "",
        }
    return out


@functools.lru_cache(maxsize=1)
def load_roads() -> list:
    """[[ [lat,lon], ... ], ...] polylines for trunk/primary roads in Ubon."""
    if not os.path.exists(ROADS_CACHE):
        raise FileNotFoundError(
            f"{ROADS_CACHE} not found. Run: python -c \"import geo_boundary as g; g.fetch_roads()\" once."
        )
    with open(ROADS_CACHE, encoding="utf-8") as f:
        data = json.load(f)
    return [
        [(pt["lat"], pt["lon"]) for pt in el["geometry"]]
        for el in data.get("elements", [])
        if el.get("type") == "way" and "geometry" in el and len(el["geometry"]) >= 2
    ]


# --------------------------------------------------------------- fetchers --
# One-time fetch functions. Run manually (see the load_roads message above);
# not called automatically since a countrywide Overpass query is slow and
# shouldn't run on every dashboard load.
#
# The boundary fetchers that used to live here are gone for good: first
# fetch_thailand_provinces() and fetch_ubon_districts(), which pulled
# FAO/GAUL/2015 levels 1 and 2 from Earth Engine, then load_boundary() and its
# Nominatim lookup of the Ubon relation. Nothing here fetches administrative
# geometry any more, from any source. Re-adding one would put boundary data
# back into a project that was deliberately cleared of it.


def fetch_roads(highway_types="trunk|primary"):
    area_id = 3600000000 + UBON_RELATION_ID
    query = f"""
    [out:json][timeout:90];
    area({area_id})->.a;
    (
      way["highway"~"{highway_types}"](area.a);
    );
    out geom;
    """
    headers = {"User-Agent": "MekongWaterQualityDashboard/1.0 (research project)"}
    resp = requests.post(OVERPASS_URL, data={"data": query}, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    with open(ROADS_CACHE, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return data
