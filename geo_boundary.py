"""Boundary and reference-layer geometry for the Mekong Water Quality dashboard.

- All-Thailand province boundaries (77): local TH_Province shapefile, cached
  to thailand_provinces.geojson with both English and Thai names.
- Ubon district (amphoe) boundaries (25): local GISTDA FGDS 1:50k amphoe
  shapefile, cached to ubon_districts.geojson with both names.
  Ubon's own outline in thailand_provinces.geojson is the union of these 25,
  not the province shapefile's own version of it - the two datasets disagree
  along that border by a few hundred metres, which drew as a doubled edge.
  Both of the above are written by import_shapefiles.py - see that module for
  why the shapefiles are converted rather than read directly.
- Station place names: OpenStreetMap Nominatim reverse geocoding, cached to
  station_locations.json.
- Ubon Ratchathani outline: OpenStreetMap Nominatim (relation 1908830),
  cached to ubon_boundary.geojson. The dashboard no longer uses this - it
  takes Ubon's outline from the province shapefile above, so the boundary it
  fits the map to is the same one it draws - but the Earth Engine export
  scripts still keep it as a fallback study area.
- Major roads (trunk + primary) within Ubon: OpenStreetMap Overpass, cached
  to ubon_roads.json.

All fetches are one-time; if a cache file already exists on disk it's used
as-is (delete the file to force a re-fetch).
"""
import functools
import json
import os
import time

import requests
from shapely.geometry import shape

BOUNDARY_CACHE = "ubon_boundary.geojson"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
QUERY = "Ubon Ratchathani Province, Thailand"

STATION_LOCATIONS_CACHE = "station_locations.json"

THAILAND_PROVINCES_CACHE = "thailand_provinces.geojson"
UBON_DISTRICTS_CACHE = "ubon_districts.geojson"
ROADS_CACHE = "ubon_roads.json"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
UBON_RELATION_ID = 1908830  # OSM relation id, from the Nominatim lookup above


def _fetch_boundary_geojson() -> dict:
    headers = {"User-Agent": "MekongWaterQualityDashboard/1.0 (research project)"}
    resp = requests.get(
        NOMINATIM_URL,
        params={"q": QUERY, "format": "jsonv2", "polygon_geojson": 1, "limit": 1},
        headers=headers,
        timeout=20,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise RuntimeError(f"Nominatim returned no results for '{QUERY}'")
    return results[0]["geojson"]


@functools.lru_cache(maxsize=1)
def load_boundary(simplify_tol: float = 0.008):
    """Returns a shapely Polygon for Ubon Ratchathani, simplified for map rendering."""
    if os.path.exists(BOUNDARY_CACHE):
        with open(BOUNDARY_CACHE, encoding="utf-8") as f:
            geo = json.load(f)
    else:
        geo = _fetch_boundary_geojson()
        with open(BOUNDARY_CACHE, "w", encoding="utf-8") as f:
            json.dump(geo, f)
    poly = shape(geo)
    return poly.simplify(simplify_tol, preserve_topology=True)


def boundary_rings_latlon(poly):
    """[[lat, lon], ...] rings for folium.Polygon (exterior + holes)."""
    rings = [list(poly.exterior.coords)]
    rings += [list(ring.coords) for ring in poly.interiors]
    return [[(lat, lon) for lon, lat in ring] for ring in rings]


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


def _require_cache(path, hint, label):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"{path} not found ({label}). {hint}")


_SHAPEFILE_HINT = "Run: python import_shapefiles.py (needs the local shapefile folders)."


@functools.lru_cache(maxsize=1)
def load_thailand_provinces() -> dict:
    """All 77 Thailand provinces: ADM1_NAME, plus ADM1_NAME_TH for the Thai UI."""
    return _require_cache(THAILAND_PROVINCES_CACHE, _SHAPEFILE_HINT, "Thailand province boundaries")


@functools.lru_cache(maxsize=1)
def load_ubon_districts() -> dict:
    """Ubon Ratchathani's 25 districts: ADM2_NAME, plus ADM2_NAME_TH."""
    return _require_cache(UBON_DISTRICTS_CACHE, _SHAPEFILE_HINT, "Ubon district boundaries")


@functools.lru_cache(maxsize=8)
def load_province(name: str):
    """One province's outline, as a shapely geometry, from the same shapefile
    data the map draws.

    Used for the map's initial fit bounds. That used to come from the separate
    OpenStreetMap outline in load_boundary(), which meant the rectangle the map
    fitted to and the outline it drew were two different pieces of geometry;
    reading both from one source keeps them in step.
    """
    for feature in load_thailand_provinces()["features"]:
        if feature["properties"].get("ADM1_NAME") == name:
            return shape(feature["geometry"])
    raise KeyError(f"{name!r} not found in {THAILAND_PROVINCES_CACHE}")


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
# One-time fetch functions. Run manually (see _require_cache messages above);
# not called automatically since Thailand-wide EE queries and countrywide
# Overpass queries are slow and shouldn't run on every dashboard load.

# fetch_thailand_provinces() and fetch_ubon_districts() used to live here,
# pulling FAO/GAUL/2015 levels 1 and 2 from Earth Engine. Both boundary sets
# now come from the local Thai shapefiles instead (import_shapefiles.py),
# which are more complete - GAUL had 20 of Ubon's 25 districts - and carry
# Thai names, which GAUL did not. The fetchers are gone rather than kept as a
# fallback: they wrote to these same two files, so calling one would quietly
# replace the better data with the worse.


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
