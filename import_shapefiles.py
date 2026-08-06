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
- Amphoe: L05_AdminBoundary_Amphoe_*.shp - GISTDA's 1:50k FGDS layer, WGS84,
  UTF-8. All 930 of Thailand's amphoe. This is the only geometry that ends up
  drawn: the province layer is built from it too (see below).
- Provinces: TH_Province.shp - UTM zone 47N, TIS-620. Used for its province
  *names*, which the amphoe layer does not carry - it identifies a province
  only by code.

They disagree about encoding, projection and which sidecar file declares the
encoding, and the folders have been reorganised more than once - so rather
than hard-coding any of that, each source is located by filename pattern and
its encoding and CRS read from the .prj and .cpg/.cst beside it.

Why provinces are built from the amphoe rather than read from TH_Province
-------------------------------------------------------------------------
The two datasets disagree along every province border by a few hundred
metres. Drawing the province layer from one and the district layer from the
other therefore puts two nearly-parallel lines on the map, which reads as a
smeared double edge rather than a boundary. A province is the union of its
districts, so building it that way is both correct and what makes the two
layers coincide exactly.

Run it after replacing either shapefile:

    python import_shapefiles.py

Then commit the two .geojson files it writes. The shapefiles themselves are
gitignored on purpose (see .gitignore).
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import shapefile  # pyshp - see requirements-dev.txt
from rasterio.warp import transform as warp_transform
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
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

# Degrees. Two tiers, because the country-wide layers are ~930 districts and
# 77 provinces and every byte of them is re-sent to the browser on each
# Streamlit rerun (streamlit-folium reserialises the whole map), while only
# Ubon is ever looked at closely:
#   Ubon's districts, and the province outline built from them, are what the
#   reader actually inspects, and the display path draws the focus province
#   as cached rather than re-simplifying it - so 0.0005 (~55m) is what is
#   seen.
#   Everything else is context at country zoom, where 0.01 (~1.1km) is around
#   a pixel; the other 76 provinces are re-simplified to 0.02 for drawing
#   regardless.
# Measured whole-country cost: 0.0005 everywhere would be 8.9MB, 0.005 1.6MB,
# this split 1.1MB.
FOCUS_TOLERANCE = 0.0005
CONTEXT_TOLERANCE = 0.01

# The old "minor district" designation. Every one of these was upgraded to a
# full amphoe years ago; where a dataset still carries the prefix, putting it
# on the map would be wrong as well as noisy.
KING_AMPHOE_EN = "KING AMPHOE"
THAI_PREFIXES = ("กิ่งอำเภอ", "อำเภอ", "จังหวัด")

# The province shapefile carries a stale English label for province 38: it
# says NONG KHAI, but the Thai name beside it is บึงกาฬ and its amphoe are
# Bung Kan, Seka, Si Wilai and so on. It is Bueng Kan, split out of Nong Khai
# in 2011; only the English column was never updated. Uncorrected, the map
# labels two different provinces "Nong Khai".
ENGLISH_NAME_FIXES = {"38": "Bueng Kan"}


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


def to_wgs84(geom, src_crs):
    if src_crs == DST_CRS:
        return geom
    return shapely_transform(
        lambda xs, ys: tuple(warp_transform(src_crs, DST_CRS, list(xs), list(ys))), geom)


def outer_ring_only(geom):
    """Drop the interior rings from a dissolved boundary.

    Unioning a province's districts leaves a hairline gap wherever two
    neighbours' shared edge simplified a fraction differently - a few hundred
    of them country-wide, each a tiny fraction of a square kilometre. They are
    digitisation slivers rather than enclaves, and these layers are drawn as
    outlines rather than fills, so each one would paint as a speck of stray
    boundary inside a province. The exterior ring, which is what gets drawn,
    is untouched.
    """
    parts = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    cleaned = [Polygon(p.exterior) for p in parts if not p.is_empty]
    return cleaned[0] if len(cleaned) == 1 else MultiPolygon(cleaned)


def province_names(path):
    """{province code: (english, thai)} from the province shapefile.

    This file is read for its names alone. The amphoe layer identifies a
    province only by a numeric code, and nothing else in the project maps
    those codes to names.
    """
    enc = sidecar_encoding(path, "tis-620")
    out = {}
    for rec in shapefile.Reader(path, encoding=enc).records():
        code = str(rec["PROV_CODE"]).strip()
        out[code] = (ENGLISH_NAME_FIXES.get(code, clean_en(rec["PROV_NAME"])),
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
        tol = FOCUS_TOLERANCE if prov_en.upper() == FOCUS_PROVINCE else CONTEXT_TOLERANCE
        geom = geom.simplify(tol, preserve_topology=True)
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


def build_provinces(district_collection):
    """Every province, as the union of its own districts.

    The union is taken over the *already simplified* district geometries, so
    a shared edge is the identical vertex list in both layers and the province
    outline cannot sit a pixel off the district edges drawn on top of it.

    Grouped by province code, not name: two provinces in the source share the
    English name "Nong Khai" (see ENGLISH_NAME_FIXES), and grouping by name
    silently merged them into one 76-province layer.
    """
    groups = defaultdict(list)
    names = {}
    for f in district_collection["features"]:
        key = f["properties"]["ADM1_CODE"]
        groups[key].append(shape(f["geometry"]))
        names[key] = (f["properties"]["ADM1_NAME"], f["properties"]["ADM1_NAME_TH"])

    features = []
    for code, parts in sorted(groups.items()):
        geom = parts[0] if len(parts) == 1 else unary_union(parts)
        if not geom.is_valid:
            geom = geom.buffer(0)
        geom = outer_ring_only(geom)
        if geom.is_empty:
            continue
        name_en, name_th = names[code]
        features.append({
            "type": "Feature",
            "properties": {"ADM1_NAME": name_en, "ADM1_NAME_TH": name_th},
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

    names = province_names(province_shp)
    print(f"\nDistricts (all of Thailand, {FOCUS_PROVINCE.title()} kept finer) ...")
    districts = build_districts(district_shp, names)
    write(DISTRICTS_OUT, districts)

    print("Provinces (each built from its own districts) ...")
    write(PROVINCES_OUT, build_provinces(districts))
    print("Commit both .geojson files; the shapefiles stay gitignored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
