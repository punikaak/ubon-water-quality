"""Build the dashboard's province and district boundaries from the local Thai
shapefiles (Province Shapefile/, Amphoe Shapefile/).

Why a conversion step rather than reading the shapefiles directly
----------------------------------------------------------------
The two shapefiles total ~51MB, and the deployed app on Streamlit Cloud has
nothing but the repo, which cannot carry files that size. Reading them at
runtime would work on this machine and fail on the website.

So they are converted once, here, into the same two GeoJSON caches the app
already loads (see geo_boundary.load_thailand_provinces / load_ubon_districts).
Those are small enough to commit, which keeps the deployed site working and
means no other module has to know where the geometry came from.

The two sources
---------------
- Provinces: TH_Province.shp - all 77, UTM zone 47N, TIS-620 names.
- Districts: L05_AdminBoundary_Amphoe_*.shp - GISTDA's 1:50k FGDS amphoe
  layer, already WGS84 with UTF-8 names. Ubon's 25 amphoe are taken from it
  directly.

They disagree about encoding, projection and which sidecar file declares the
encoding, and the folders have already been reorganised twice - so rather
than hard-coding any of that, each source is located by filename pattern and
its encoding and CRS read from the .prj and .cpg/.cst beside it.

An earlier version built the district layer by dissolving a tambon
(subdistrict) shapefile up to amphoe. The amphoe layer replaces that
outright: dissolving unions hundreds of polygons along shared edges that do
not quite coincide, which leaves slivers and a visibly ragged outline, and an
authoritative amphoe layer has no such problem.

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
DISTRICTS_OUT = "ubon_districts.geojson"

# Degrees, chosen against what actually consumes the output rather than by eye:
#   The 76 provinces that are only context around Ubon are re-simplified to
#   0.02 by the display path anyway (dashboard.load_provinces_for_display).
#   Ubon itself is exempt from that step and drawn as cached, so its tolerance
#   is what the reader actually sees - hence far finer.
#   Districts are rasterised for the per-district turbidity ranking
#   (dashboard.district_ntu) onto a grid whose pixels are ~0.003 across, so
#   0.0005 is already sub-pixel there and well under a line's width on screen.
# All three matter for repo weight: these files ship to Streamlit Cloud.
PROVINCE_TOLERANCE = 0.005
FOCUS_TOLERANCE = 0.0005
DISTRICT_TOLERANCE = 0.0005

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


def prepare(geom, src_crs, tolerance):
    """Repair, reproject and simplify one boundary.

    buffer(0) before anything else: a self-touching ring makes shapely refuse
    to simplify, and it is the standard repair - a no-op on geometry that was
    already valid, which is all of the amphoe layer and all but a few
    provinces.
    """
    if not geom.is_valid:
        geom = geom.buffer(0)
    if src_crs != DST_CRS:
        geom = shapely_transform(
            lambda xs, ys: tuple(warp_transform(src_crs, DST_CRS, list(xs), list(ys))), geom)
    if tolerance:
        geom = geom.simplify(tolerance, preserve_topology=True)
    return geom


def focus_code_of(path):
    """Ubon's province code, which is how the amphoe layer identifies it."""
    enc = sidecar_encoding(path, "tis-620")
    for rec in shapefile.Reader(path, encoding=enc).records():
        if rec["PROV_NAME"].strip().upper() == FOCUS_PROVINCE:
            return str(rec["PROV_CODE"]).strip()
    return None


def outer_ring_only(geom):
    """Drop the interior rings from a dissolved boundary.

    Unioning the districts leaves a hairline gap wherever two neighbours'
    shared edge simplified a fraction differently - 314 of them here, the
    largest 0.05km2 against the province's 16063, 0.016% of it in total.
    They are digitisation slivers rather than enclaves, and this layer is
    drawn as an outline rather than a fill, so each one would paint as a
    speck of stray boundary inside the province. Ubon has no real enclaves,
    so every interior goes; the exterior ring, which is what gets drawn, is
    untouched.
    """
    parts = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    cleaned = [Polygon(p.exterior) for p in parts if not p.is_empty]
    return cleaned[0] if len(cleaned) == 1 else MultiPolygon(cleaned)


def build_provinces(path, focus_geom=None):
    """All 77 provinces.

    `focus_geom` replaces Ubon's own outline with one built elsewhere - see
    main(), which passes the union of its amphoe. The two shapefiles disagree
    slightly along that border, and drawing the province from one while
    drawing the districts from the other put two nearly-parallel lines a few
    hundred metres apart on the map, which read as a smeared double edge
    rather than a boundary. A province is the union of its districts, so
    building it that way is both correct and the thing that makes the two
    layers coincide exactly.
    """
    crs, enc = sidecar_crs(path), sidecar_encoding(path, "tis-620")
    reader = shapefile.Reader(path, encoding=enc)
    features = []
    for rec, shp in zip(reader.records(), reader.shapes()):
        name_en = rec["PROV_NAME"]
        is_focus = name_en.strip().upper() == FOCUS_PROVINCE
        if is_focus and focus_geom is not None:
            geom = focus_geom
        else:
            geom = prepare(shape(shp.__geo_interface__), crs,
                           FOCUS_TOLERANCE if is_focus else PROVINCE_TOLERANCE)
        if geom.is_empty:
            continue
        features.append({
            "type": "Feature",
            "properties": {
                "ADM1_NAME": clean_en(name_en),
                "ADM1_NAME_TH": clean_th(rec["PROV_NAMT"]),
            },
            "geometry": mapping(geom),
        })
    return {"type": "FeatureCollection", "features": features}


def build_districts(path, focus_code):
    """Ubon's amphoe, taken straight from the amphoe layer.

    The layer identifies a district's province by code, not name, so the code
    is carried over from the province file - which is also what guarantees the
    two layers are talking about the same province.
    """
    crs, enc = sidecar_crs(path), sidecar_encoding(path, "utf-8")
    reader = shapefile.Reader(path, encoding=enc)
    features = []
    for rec, shp in zip(reader.records(), reader.shapes()):
        if str(rec["PRV_CODE"]).strip() != focus_code:
            continue
        geom = prepare(shape(shp.__geo_interface__), crs, DISTRICT_TOLERANCE)
        if geom.is_empty:
            continue
        features.append({
            "type": "Feature",
            "properties": {
                "ADM2_NAME": clean_en(rec["AMP_NAME_E"]),
                "ADM2_NAME_TH": clean_th(rec["AMP_NAME_T"]),
            },
            "geometry": mapping(geom),
        })
    features.sort(key=lambda f: f["properties"]["ADM2_NAME"])
    return {"type": "FeatureCollection", "features": features}


def write(path, collection):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(collection, f, ensure_ascii=False)
    kb = os.path.getsize(path) / 1024
    print(f"  wrote {path}: {len(collection['features'])} features, {kb:.0f} KB")


def main() -> int:
    province_shp = find_one(PROVINCE_GLOB)
    district_shp = find_one(DISTRICT_GLOB)
    for label, path in (("province", province_shp), ("amphoe", district_shp)):
        if path is None:
            print(f"No {label} shapefile found under this folder.")
            return 1
        print(f"{label:9} <- {path}  [{sidecar_crs(path)}, {sidecar_encoding(path)}]")

    focus_code = focus_code_of(province_shp)
    if focus_code is None:
        print(f"{FOCUS_PROVINCE} not found in {province_shp}.")
        return 1

    print(f"Districts of {FOCUS_PROVINCE.title()} (province code {focus_code}) ...")
    districts = build_districts(district_shp, focus_code)
    write(DISTRICTS_OUT, districts)

    # Union the districts *after* simplification, not before: the shared edge
    # between two neighbours is then the identical vertex list in both, so the
    # union's outer ring is made of the very segments the district layer draws
    # and the two outlines cannot disagree by a pixel.
    focus_geom = unary_union([shape(f["geometry"]) for f in districts["features"]])
    if not focus_geom.is_valid:
        focus_geom = focus_geom.buffer(0)
    focus_geom = outer_ring_only(focus_geom)
    print("Provinces (Ubon's outline built from those districts) ...")
    write(PROVINCES_OUT, build_provinces(province_shp, focus_geom))
    print("Commit both .geojson files; the shapefiles stay gitignored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
