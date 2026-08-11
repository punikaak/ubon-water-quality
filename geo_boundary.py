"""Boundary and reference-layer geometry for the Mekong Water Quality dashboard.

- District (amphoe) boundaries, all 930 in Thailand: from Amphoe Shapefile.zip
  (GISTDA's FGDS 1:50k amphoe layer), cached to thailand_districts.geojson
  with English and Thai names and the province each belongs to.
- Province boundaries, all 77: from Province Shapefile.zip (TH_Province),
  cached to thailand_provinces.geojson.
  Those two local archives are the only source of boundary geometry here, and
  neither layer is derived from the other - each is drawn as its own shapefile
  has it. The province file is also read for its province names, which the
  amphoe layer lacks. Both caches are written by import_shapefiles.py; see
  that module for why the archives are converted rather than read at runtime.
- Station place names: OpenStreetMap Nominatim reverse geocoding, cached to
  station_locations.json. Text for a point, with no geometry attached - not
  boundary data.
- Major roads (trunk + primary) within Ubon: OpenStreetMap Overpass, cached
  to ubon_roads.json.

The two archives above are the only source of boundary geometry in this
project. There is no OpenStreetMap fallback: load_boundary() and its
ubon_boundary.geojson cache, which fetched Ubon's outline from Nominatim, are
gone. Ubon's outline comes from the province shapefile like every other
province's, so the box the map fits to and the line it draws are one piece of
geometry rather than two that disagree.

All fetches are one-time; if a cache file already exists on disk it's used
as-is (delete the file to force a re-fetch).
"""
import functools
import json
import os
import time

import requests
from shapely.geometry import shape

NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"

STATION_LOCATIONS_CACHE = "station_locations.json"

THAILAND_PROVINCES_CACHE = "thailand_provinces.geojson"
DISTRICTS_CACHE = "thailand_districts.geojson"
WATER_CACHE = "thailand_water.geojson"
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


def _require_cache(path, hint, label):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(f"{path} not found ({label}). {hint}")


_SHAPEFILE_HINT = "Run: python import_shapefiles.py (needs the two local shapefile .zip archives)."
_WATER_HINT = "Run: python import_water.py (needs the local Stream .zip archive)."


@functools.lru_cache(maxsize=1)
def load_thailand_provinces() -> dict:
    """All 77 Thailand provinces: ADM1_NAME, plus ADM1_NAME_TH for the Thai UI."""
    return _require_cache(THAILAND_PROVINCES_CACHE, _SHAPEFILE_HINT, "Thailand province boundaries")


@functools.lru_cache(maxsize=1)
def load_districts() -> dict:
    """All 930 Thai amphoe: ADM2_NAME/_TH, plus the ADM1_NAME/_TH and
    ADM1_CODE of the province each belongs to.

    Country-wide rather than Ubon-only, so callers that want one province
    filter on ADM1_NAME - see dashboard.district_ntu, which ranks Ubon's
    districts and would otherwise rasterise all 930 over a grid that covers
    one province.
    """
    return _require_cache(DISTRICTS_CACHE, _SHAPEFILE_HINT, "district boundaries")


@functools.lru_cache(maxsize=1)
def load_water() -> dict:
    """Thailand's water areas, from the local Stream.zip archive.

    The archive's layers are named Wetland_<region> and every polygon in them
    is tagged LU_GROUP="Wetland", so this is wetland cover rather than river
    channels - see import_water.py, which is the only thing that writes this
    cache.
    """
    return _require_cache(WATER_CACHE, _WATER_HINT, "water layer")


@functools.lru_cache(maxsize=8)
def load_province(name: str):
    """One province's outline, as a shapely geometry, from the same shapefile
    data the map draws.

    Used for the map's initial fit bounds. That used to come from a separate
    OpenStreetMap outline, which meant the rectangle the map fitted to and the
    outline it drew were two different pieces of geometry that disagreed by
    kilometres. Both now read from the province shapefile, so they agree by
    construction; the OSM outline and its loader have been deleted.
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

# No boundary fetcher lives here any more, from any source. There were two:
# fetch_thailand_provinces() / fetch_ubon_districts(), pulling FAO/GAUL/2015
# levels 1 and 2 from Earth Engine, and the Nominatim lookup behind
# load_boundary(). Both boundary sets now come from the two local shapefile
# archives instead (import_shapefiles.py), which are more complete - GAUL had
# 20 of Ubon's 25 districts - and carry Thai names, which GAUL did not.
#
# They are deleted rather than kept as a fallback: they wrote to the same
# cache files the archives feed, so calling one would quietly replace the
# chosen data with a different dataset.


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
