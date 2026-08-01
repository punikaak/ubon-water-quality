"""Boundary and reference-layer geometry for the Mekong Water Quality dashboard.

- Ubon Ratchathani province boundary: OpenStreetMap Nominatim (relation
  1908830), cached to ubon_boundary.geojson. Used to center the map and as
  the water-masking extent for province_composite.py.
- All-Thailand province boundaries: FAO/GAUL/2015/level1 via Earth Engine,
  cached to thailand_provinces.geojson (76 provinces, simplified to 1km
  tolerance server-side before download).
- Ubon district (amphoe) boundaries: FAO/GAUL/2015/level2 via Earth Engine,
  filtered to Ubon Ratchathani, cached to ubon_districts.geojson.
- Major roads (trunk + primary) within Ubon: OpenStreetMap Overpass, cached
  to ubon_roads.json.

All fetches are one-time; if a cache file already exists on disk it's used
as-is (delete the file to force a re-fetch).
"""
import functools
import json
import os

import requests
from shapely.geometry import shape

BOUNDARY_CACHE = "ubon_boundary.geojson"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
QUERY = "Ubon Ratchathani Province, Thailand"

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


def _require_cache(path, fetch_fn, label):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    raise FileNotFoundError(
        f"{path} not found. Run: python -c \"import geo_boundary as g; g.{fetch_fn}()\" "
        f"once from a GEE-authenticated environment to fetch {label}."
    )


@functools.lru_cache(maxsize=1)
def load_thailand_provinces() -> dict:
    """GeoJSON FeatureCollection of all Thailand provinces (ADM1_NAME property)."""
    return _require_cache(THAILAND_PROVINCES_CACHE, "fetch_thailand_provinces", "Thailand province boundaries")


@functools.lru_cache(maxsize=1)
def load_ubon_districts() -> dict:
    """GeoJSON FeatureCollection of Ubon Ratchathani's districts (ADM2_NAME property)."""
    return _require_cache(UBON_DISTRICTS_CACHE, "fetch_ubon_districts", "Ubon district boundaries")


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

def fetch_thailand_provinces(simplify_m: int = 1000):
    import ee
    ee.Initialize(project="gee-training-498303")
    provinces = ee.FeatureCollection("FAO/GAUL/2015/level1")
    thailand = provinces.filter(ee.Filter.eq("ADM0_NAME", "Thailand"))
    simplified = thailand.map(lambda f: f.simplify(simplify_m).select(["ADM1_NAME"]))
    geo = simplified.getInfo()
    with open(THAILAND_PROVINCES_CACHE, "w", encoding="utf-8") as f:
        json.dump(geo, f)
    return geo


def fetch_ubon_districts(simplify_m: int = 200):
    import ee
    ee.Initialize(project="gee-training-498303")
    districts = ee.FeatureCollection("FAO/GAUL/2015/level2")
    ubon = districts.filter(ee.Filter.eq("ADM1_NAME", "Ubon Ratchathani"))
    simplified = ubon.map(lambda f: f.simplify(simplify_m).select(["ADM2_NAME"]))
    geo = simplified.getInfo()
    with open(UBON_DISTRICTS_CACHE, "w", encoding="utf-8") as f:
        json.dump(geo, f)
    return geo


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
