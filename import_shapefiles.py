"""Build the dashboard's province and district boundaries from the local Thai
shapefiles (Province Shapefile/, Amphoe Shapefile/).

Why a conversion step rather than reading the shapefiles directly
----------------------------------------------------------------
The two shapefiles total ~51MB, and the deployed app on Streamlit Cloud has
nothing but the repo, which cannot carry files that size. Reading them at
runtime would work on this machine and fail on the website.

So they are converted once, here, into the two GeoJSON caches the app loads
(see geo_boundary.load_thailand_provinces / load_districts). Those are small
enough to commit, which keeps the deployed site working and means no other
module has to know where the geometry came from.

The two sources
---------------
- Provinces: TH_Province.shp - all 77, UTM zone 47N, TIS-620 names.
- Amphoe: L05_AdminBoundary_Amphoe_*.shp - GISTDA's 1:50k FGDS layer, WGS84,
  UTF-8. All 930 of Thailand's amphoe. It identifies a district's province by
  code only, so the province names are joined in from the file above.

They disagree about encoding, projection and which sidecar file declares the
encoding, and the folders have been reorganised more than once - so rather
than hard-coding any of that, each source is located by filename pattern and
its encoding and CRS read from the .prj and .cpg/.cst beside it.

Each layer is read from its own file and nothing is derived from the other.
Note that the two datasets disagree along province borders by a few hundred
metres, so with both layers shown a province edge and the district edges
along it will not sit exactly on top of each other.

Run it after replacing either shapefile:

    python import_shapefiles.py

Then commit the two .geojson files it writes. The shapefiles themselves are
gitignored on purpose (see .gitignore).
"""
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

import shapefile  # pyshp - see requirements-dev.txt
from rasterio.warp import transform as warp_transform
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

# Filename patterns rather than paths. The province file has kept its name
# while moving folders; the amphoe file's name carries a dataset version
# ("..._2011_50k_FGDS_beta") that will change when the dataset does.
PROVINCE_GLOB = "**/*Province*.shp"
DISTRICT_GLOB = "**/*Amphoe*.shp"

DST_CRS = "EPSG:4326"
FOCUS_PROVINCE = "UBON RATCHATHANI"
PROVINCES_OUT = "thailand_provinces.geojson"
DISTRICTS_OUT = "thailand_districts.geojson"

# Simplification has to happen - every byte of these layers is re-sent to the
# browser on each Streamlit rerun, since streamlit-folium reserialises the
# whole map - but the tolerance is proportional to each polygon's own size
# rather than a fixed distance.
#
# A fixed distance is the obvious thing and it is wrong here, because Thai
# amphoe span four orders of magnitude in area. At a fixed 0.01deg (~1.1km),
# a 5000km2 rural amphoe is untouched while Bangkok's khet - 6 to 28km2 - are
# mangled: measured against the source, the worst district came out at 30%
# IoU and most of Bangkok's under 75%. Tightening the fixed value does not
# fix it either; 0.001 costs 5.6MB and still leaves the worst at 85%.
#
# Scaling by sqrt(area) holds every feature to the same *relative* fidelity
# whatever its size. Measured over all 930 amphoe: worst 93.8%, median 97.4%,
# for 1.8MB.
SHAPE_TOLERANCE_FRACTION = 0.02
# Ubon and its districts are the subject of this map rather than context, so
# they get a fixed fine tolerance instead - finer than the rule above would
# hand them.
FOCUS_TOLERANCE = 0.0005

# The old "minor district" designation. Every one of these was upgraded to a
# full amphoe years ago; where a dataset still carries the prefix, putting it
# on the map would be wrong as well as noisy.
KING_AMPHOE_EN = "KING AMPHOE"
THAI_PREFIXES = ("กิ่งอำเภอ", "อำเภอ", "จังหวัด")


def find_one(pattern):
    """The single shapefile matching `pattern`, or None.

    Skips the QGIS lock files that appear beside an open layer, which glob
    otherwise picks up because they end in .shp.<something>.
    """
    hits = [p for p in glob.glob(pattern, recursive=True)
            if p.lower().endswith(".shp") and ".git" not in p]
    return sorted(hits)[0] if hits else None


def sidecar_encoding(shp_path, default="utf-8"):
    """The character encoding a shapefile declares for its .dbf.

    Read rather than assumed: the province file says TIS-620 in a .cpg, the
    amphoe file says UTF-8 in a .cst, and decoding Thai with the wrong one
    turns every name into mojibake without raising anything.
    """
    stem = os.path.splitext(shp_path)[0]
    for ext in (".cpg", ".cst", ".CPG", ".CST"):
        try:
            with open(stem + ext, encoding="ascii") as f:
                declared = f.read().strip()
            if declared:
                return declared
        except OSError:
            continue
    return default


def sidecar_crs(shp_path, default="EPSG:4326"):
    """The CRS a shapefile declares in its .prj, as far as is needed here.

    Deliberately a two-case reading rather than a real WKT parser: these
    files are either already geographic WGS84 or Thailand's UTM zone 47N, and
    pulling in a full projection library to tell those apart would be the only
    reason this script needed one.
    """
    try:
        with open(os.path.splitext(shp_path)[0] + ".prj", encoding="ascii", errors="ignore") as f:
            wkt = f.read()
    except OSError:
        return default
    if "PROJCS" not in wkt:
        return "EPSG:4326"
    if re.search(r"UTM.?[_ ]?Zone.?[_ ]?47", wkt, re.I):
        return "EPSG:32647"
    raise ValueError(f"{shp_path}: projected CRS that this script does not know how to read:\n{wkt}")


def clean_en(raw: str) -> str:
    """Display form of an English label. Left alone unless it is all-caps -
    the amphoe layer is already properly cased ("Phra Nakhon") and .title()
    would not improve it."""
    s = " ".join(raw.split())
    if s.upper().startswith(KING_AMPHOE_EN):
        s = s[len(KING_AMPHOE_EN):].strip()
    return s.title() if s.isupper() else s


def clean_th(raw: str) -> str:
    """Thai label with its administrative word stripped - the map labels the
    level itself, so repeating it in every name just costs width."""
    s = " ".join(raw.split())
    for prefix in THAI_PREFIXES:
        if s.startswith(prefix):
            return s[len(prefix):].strip()
    return s


def tolerance_for(geom, is_focus):
    """Simplification tolerance for one polygon - see SHAPE_TOLERANCE_FRACTION."""
    if is_focus:
        return FOCUS_TOLERANCE
    return math.sqrt(geom.area) * SHAPE_TOLERANCE_FRACTION


def to_wgs84(geom, src_crs):
    if src_crs == DST_CRS:
        return geom
    return shapely_transform(
        lambda xs, ys: tuple(warp_transform(src_crs, DST_CRS, list(xs), list(ys))), geom)




def province_names(path):
    """{province code: (english, thai)} from the province shapefile.

    Used to label districts with the province they belong to: the amphoe
    layer identifies a province only by a numeric code, and nothing else in
    the project maps those codes to names.
    """
    enc = sidecar_encoding(path, "tis-620")
    out = {}
    for rec in shapefile.Reader(path, encoding=enc).records():
        out[str(rec["PROV_CODE"]).strip()] = (clean_en(rec["PROV_NAME"]),
                                              clean_th(rec["PROV_NAMT"]))
    return out


def build_districts(path, names_by_code):
    """Every amphoe in Thailand, one feature each.

    Grouped by AMP_CODE first: 37 amphoe are split across several records in
    this dataset (islands, mostly), and emitting those as separate features
    would put duplicate names in the layer and leave their internal edges
    drawn as boundaries.

    The province is keyed off AMP_CODE's first two digits rather than the
    PRV_CODE column. Both agree wherever PRV_CODE is filled in, but 16
    records have it blank - among them the whole of Nong Bua Lamphu - and
    those would otherwise be dropped, taking their province's outline with
    them.
    """
    crs, enc = sidecar_crs(path), sidecar_encoding(path, "utf-8")
    reader = shapefile.Reader(path, encoding=enc)

    groups = defaultdict(list)
    labels = {}
    for rec, shp in zip(reader.records(), reader.shapes()):
        code = str(rec["AMP_CODE"]).strip()
        groups[code].append(shape(shp.__geo_interface__))
        # Same 16 records carry no name either. Falling back to the code keeps
        # the feature identifiable instead of showing an empty tooltip.
        if code not in labels or not labels[code][0]:
            labels[code] = (clean_en(rec["AMP_NAME_E"]), clean_th(rec["AMP_NAME_T"]))

    features = []
    for code, parts in sorted(groups.items()):
        prov_code = code[:2]
        prov_en, prov_th = names_by_code.get(prov_code, ("", ""))
        geom = parts[0] if len(parts) == 1 else unary_union(parts)
        if not geom.is_valid:
            geom = geom.buffer(0)
        geom = to_wgs84(geom, crs)
        geom = geom.simplify(tolerance_for(geom, prov_en.upper() == FOCUS_PROVINCE),
                             preserve_topology=True)
        if geom.is_empty:
            continue
        name_en, name_th = labels[code]
        features.append({
            "type": "Feature",
            "properties": {
                "ADM2_NAME": name_en or f"Amphoe {code}",
                "ADM2_NAME_TH": name_th or f"อำเภอ {code}",
                "ADM1_CODE": prov_code,
                "ADM1_NAME": prov_en,
                "ADM1_NAME_TH": prov_th,
            },
            "geometry": mapping(geom),
        })
    return {"type": "FeatureCollection", "features": features}


def build_provinces(path):
    """All 77 provinces, straight from the province shapefile."""
    crs, enc = sidecar_crs(path), sidecar_encoding(path, "tis-620")
    reader = shapefile.Reader(path, encoding=enc)
    features = []
    for rec, shp in zip(reader.records(), reader.shapes()):
        name_en = clean_en(rec["PROV_NAME"])
        geom = shape(shp.__geo_interface__)
        if not geom.is_valid:
            geom = geom.buffer(0)
        geom = to_wgs84(geom, crs)
        geom = geom.simplify(tolerance_for(geom, name_en.upper() == FOCUS_PROVINCE),
                             preserve_topology=True)
        if geom.is_empty:
            continue
        features.append({
            "type": "Feature",
            "properties": {"ADM1_NAME": name_en, "ADM1_NAME_TH": clean_th(rec["PROV_NAMT"])},
            "geometry": mapping(geom),
        })
    return {"type": "FeatureCollection", "features": features}


def write(path, collection):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(collection, f, ensure_ascii=False)
    kb = os.path.getsize(path) / 1024
    print(f"  wrote {path}: {len(collection['features'])} features, {kb:,.0f} KB")


def main() -> int:
    province_shp = find_one(PROVINCE_GLOB)
    district_shp = find_one(DISTRICT_GLOB)
    for label, path in (("province", province_shp), ("amphoe", district_shp)):
        if path is None:
            print(f"No {label} shapefile found under this folder.")
            return 1
        print(f"{label:9} <- {path}  [{sidecar_crs(path)}, {sidecar_encoding(path)}]")

    print(f"\nProvinces (all of Thailand, {FOCUS_PROVINCE.title()} kept finer) ...")
    write(PROVINCES_OUT, build_provinces(province_shp))
    print(f"Districts (all of Thailand, {FOCUS_PROVINCE.title()} kept finer) ...")
    write(DISTRICTS_OUT, build_districts(district_shp, province_names(province_shp)))
    print("Commit both .geojson files; the shapefiles stay gitignored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
