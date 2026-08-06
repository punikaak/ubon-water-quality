"""Build the dashboard's province and district boundaries from the local
Thai shapefiles (Province Shapefile/, Tambon Shapefile/).

Why a conversion step rather than reading the shapefiles directly
----------------------------------------------------------------
The two shapefiles total ~68MB, TH_Tambon.shp alone being over GitHub's
50MB file limit, so they cannot ship with the repo - and the deployed app on
Streamlit Cloud has nothing but the repo. Reading them at runtime would work
on this machine and fail on the website.

So they are converted once, here, into the same two GeoJSON caches the app
already loads (see geo_boundary.load_thailand_provinces / load_ubon_districts).
Those are small enough to commit, which keeps the deployed site working and
means no other module has to know where the geometry came from.

What it does
------------
- Reprojects from UTM zone 47N (both .prj files) to WGS84, which is what
  Leaflet needs.
- Provinces: all 77, straight from TH_Province.shp.
- Districts: TH_Tambon.shp is subdistrict-level (8105 tambon polygons), but
  it carries the amphoe each tambon belongs to, so Ubon's 219 tambons are
  dissolved by amphoe into its 25 districts - the level the map's District
  layer has always shown.
- Keeps the Thai name alongside the English one, which the previous source
  (FAO GAUL via Earth Engine) did not have.

Run it after replacing either shapefile:

    python import_shapefiles.py

Then commit the two .geojson files it writes. The shapefiles themselves are
gitignored on purpose (see .gitignore).
"""
import json
import os
import sys
from collections import defaultdict

import shapefile  # pyshp - see requirements-dev.txt
from rasterio.warp import transform as warp_transform
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

PROVINCE_SHP_NAME = "TH_Province.shp"
TAMBON_SHP_NAME = "TH_Tambon.shp"
# Directories skipped when hunting for them - none can contain a shapefile and
# .git in particular is full of files it would be pointless to stat.
_SKIP_DIRS = {".git", "__pycache__", ".composite_cache", "display_rasters", ".streamlit"}


def find_shapefile(filename, root="."):
    """Locate a shapefile by name, wherever it sits under the project.

    Searched rather than hard-coded because the layout has already changed
    once: TH_Tambon.shp began in a nested "Tambon Shapefile/Tambon/" and later
    moved up to "Tambon Shapefile/". A fixed path turns that kind of tidy-up
    into a crash, and the filenames are distinctive enough to find.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if name.lower() == filename.lower():
                return os.path.join(dirpath, name)
    return None
# Encodings differ between the two files; each states its own in the .cpg
# beside it, and guessing wrong turns every Thai name into mojibake.
PROVINCE_ENCODING = "tis-620"
TAMBON_ENCODING = "utf-8"

# Both .prj files say WGS_1984_UTM_Zone_47N with central meridian 99 and a
# 500km false easting.
SRC_CRS = "EPSG:32647"
DST_CRS = "EPSG:4326"

FOCUS_PROVINCE = "UBON RATCHATHANI"
PROVINCES_OUT = "thailand_provinces.geojson"
DISTRICTS_OUT = "ubon_districts.geojson"

# Degrees, and both are chosen against what actually consumes the output
# rather than by eye:
#   Provinces are re-simplified to 0.02 by the display path before rendering
#   (dashboard.load_provinces_for_display), so anything finer than that is
#   discarded anyway; 0.005 (~500m) leaves margin if that ever changes and is
#   still finer than the 1km the previous Earth Engine source shipped.
#   Districts are rasterised for the per-district turbidity ranking
#   (dashboard.district_ntu) onto a grid whose pixels are ~0.003 across, so
#   0.001 is already sub-pixel there and well under a line's width on screen.
# Both matter for repo weight: these files ship to Streamlit Cloud.
PROVINCE_TOLERANCE = 0.005
DISTRICT_TOLERANCE = 0.001
# Ubon itself is the subject of the map, drawn as a heavy highlight and looked
# at closely, so it is kept far finer than the 76 provinces that are only
# context around it. Raw it is 11129 vertices (450KB of GeoJSON); this keeps
# it within 55m of that for 67KB, which is well under a line's width at any
# zoom this map reaches. The 0.005 the others use would put it 554m out -
# enough to visibly straighten the Mekong meanders along its eastern edge.
FOCUS_TOLERANCE = 0.0005

# The old "minor district" designation. Every one of Ubon's five was upgraded
# to a full amphoe years ago; the prefix survives only in this dataset's
# labels, and carrying it onto the map would be wrong as well as noisy.
KING_AMPHOE_EN = "KING AMPHOE"
THAI_PREFIXES = ("กิ่งอำเภอ", "อำเภอ", "จังหวัด")


def clean_en(raw: str) -> str:
    """Uppercase shapefile label -> display form ("DET UDOM" -> "Det Udom")."""
    s = " ".join(raw.split())
    if s.upper().startswith(KING_AMPHOE_EN):
        s = s[len(KING_AMPHOE_EN):].strip()
    return s.title()


def clean_th(raw: str) -> str:
    """Thai label with its administrative word stripped - the map labels the
    level itself, so repeating it in every name just costs width."""
    s = " ".join(raw.split())
    for prefix in THAI_PREFIXES:
        if s.startswith(prefix):
            return s[len(prefix):].strip()
    return s


def to_wgs84(geom):
    """Reproject a shapely geometry from the shapefiles' UTM 47N to WGS84."""
    def fn(xs, ys):
        lon, lat = warp_transform(SRC_CRS, DST_CRS, list(xs), list(ys))
        return lon, lat
    return shapely_transform(fn, geom)


def prepare(geom, tolerance):
    """Reproject, repair and simplify one boundary.

    buffer(0) before anything else: a few polygons in these files have
    self-touching rings, which shapely refuses to union or simplify. It is the
    standard repair and a no-op on geometry that was already valid.
    """
    if not geom.is_valid:
        geom = geom.buffer(0)
    geom = to_wgs84(geom)
    if tolerance:
        geom = geom.simplify(tolerance, preserve_topology=True)
    return geom


def build_provinces(path):
    reader = shapefile.Reader(path, encoding=PROVINCE_ENCODING)
    features = []
    for rec, shp in zip(reader.records(), reader.shapes()):
        is_focus = rec["PROV_NAME"].strip().upper() == FOCUS_PROVINCE
        geom = prepare(shape(shp.__geo_interface__),
                       FOCUS_TOLERANCE if is_focus else PROVINCE_TOLERANCE)
        if geom.is_empty:
            continue
        features.append({
            "type": "Feature",
            "properties": {
                "ADM1_NAME": clean_en(rec["PROV_NAME"]),
                "ADM1_NAME_TH": clean_th(rec["PROV_NAMT"]),
            },
            "geometry": mapping(geom),
        })
    return {"type": "FeatureCollection", "features": features}


def build_districts(path):
    """Ubon's amphoe, dissolved up from the tambon polygons that make them."""
    reader = shapefile.Reader(path, encoding=TAMBON_ENCODING)
    groups = defaultdict(list)
    names = {}
    for rec, shp in zip(reader.records(), reader.shapes()):
        if rec["P_NAME_E"].strip().upper() != FOCUS_PROVINCE:
            continue
        key = rec["A_NAME_E"].strip().upper()
        geom = shape(shp.__geo_interface__)
        if not geom.is_valid:
            geom = geom.buffer(0)
        groups[key].append(geom)
        names.setdefault(key, (clean_en(rec["A_NAME_E"]), clean_th(rec["A_NAME_T"])))

    features = []
    for key, parts in sorted(groups.items()):
        # Dissolve first, simplify after: simplifying each tambon separately
        # would move shared edges by different amounts and leave slivers and
        # gaps along every internal boundary once they were merged.
        merged = unary_union(parts)
        merged = prepare(merged, DISTRICT_TOLERANCE)
        if merged.is_empty:
            continue
        name_en, name_th = names[key]
        features.append({
            "type": "Feature",
            "properties": {"ADM2_NAME": name_en, "ADM2_NAME_TH": name_th},
            "geometry": mapping(merged),
        })
    return {"type": "FeatureCollection", "features": features}


def write(path, collection):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(collection, f, ensure_ascii=False)
    kb = os.path.getsize(path) / 1024
    print(f"  wrote {path}: {len(collection['features'])} features, {kb:.0f} KB")


def main() -> int:
    province_shp = find_shapefile(PROVINCE_SHP_NAME)
    tambon_shp = find_shapefile(TAMBON_SHP_NAME)
    missing = [n for n, p in ((PROVINCE_SHP_NAME, province_shp),
                              (TAMBON_SHP_NAME, tambon_shp)) if p is None]
    if missing:
        print(f"Could not find {', '.join(missing)} anywhere under this folder.")
        return 1

    print(f"Provinces from {province_shp} ...")
    write(PROVINCES_OUT, build_provinces(province_shp))
    print(f"Districts from {tambon_shp} ({FOCUS_PROVINCE.title()}, dissolved from tambon) ...")
    write(DISTRICTS_OUT, build_districts(tambon_shp))
    print("Commit both .geojson files; the shapefiles stay gitignored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
