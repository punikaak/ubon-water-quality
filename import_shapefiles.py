"""Build the dashboard's province and district boundaries from the local Thai
shapefile archives - Province Shapefile.zip and Amphoe Shapefile.zip.

Those two zips are the only source of boundary geometry in this project. No
part of the map's province or district lines comes from OpenStreetMap, from
FAO GAUL, or from anywhere else, and neither layer is derived from the other:
each is read from its own shapefile and drawn as that shapefile has it.

Why a conversion step rather than reading the zips directly
-----------------------------------------------------------
They total ~39MB zipped (~54MB inside), and the deployed app on Streamlit
Cloud has nothing but the repo, which cannot carry files that size. Reading
them at runtime would work on this machine and fail on the website.

So they are converted once, here, into the two GeoJSON caches the app loads
(see geo_boundary.load_thailand_provinces / load_districts). Those are small
enough to commit, which keeps the deployed site working and means no other
module has to know where the geometry came from. The GeoJSON is a rendering
of these zips and nothing else - re-run this script and it is rebuilt from
them.

The two sources
---------------
- Provinces: TH_Province.shp - all 77, UTM zone 47N, TIS-620 names.
- Amphoe: L05_AdminBoundary_Amphoe_*.shp - GISTDA's 1:50k FGDS layer, WGS84,
  UTF-8. All 930 of Thailand's amphoe. It identifies a district's province by
  code only, so the province names are joined in from the file above.

They disagree about encoding, projection and which sidecar file declares the
encoding, and have been reorganised more than once - so rather than
hard-coding any of that, each archive is located by filename pattern, its
shapefile found inside, and its encoding and CRS read from the .prj and
.cpg/.cst packed alongside.

Note that the two datasets disagree along province borders by a few hundred
metres, so with both layers shown a province edge and the district edges
along it will not sit exactly on top of each other. That is what the source
data says; it is not corrected here, because correcting it would mean drawing
something neither shapefile contains.

Run it after replacing either zip:

    python import_shapefiles.py

Then commit the two .geojson files it writes. The zips themselves, and any
folder unpacked from them, are gitignored on purpose (see .gitignore).
"""
import glob
import io
import json
import os
import re
import sys
import zipfile
from collections import defaultdict

import shapefile  # pyshp - see requirements-dev.txt
from rasterio.warp import transform as warp_transform
from shapely.geometry import mapping, shape
from shapely.ops import transform as shapely_transform
from shapely.ops import unary_union

# Filename patterns rather than paths. The province archive has kept its name
# while moving folders; the amphoe shapefile inside carries a dataset version
# ("..._2011_50k_FGDS_beta") that will change when the dataset does.
PROVINCE_GLOB = "*Province*.zip"
DISTRICT_GLOB = "*Amphoe*.zip"

DST_CRS = "EPSG:4326"
PROVINCES_OUT = "thailand_provinces.geojson"
DISTRICTS_OUT = "thailand_districts.geojson"

# One tolerance, in degrees, for every feature in both layers. ~110m.
#
# It must be uniform, and that is the whole point. The previous scheme scaled
# the tolerance by sqrt(each polygon's area) and then exempted Ubon with a
# fixed fine value, which produced two visible defects:
#
#   - Provinces were cut to 3,028m of error and districts to 1,507m. At the
#     zoom this map opens on that is 10-20 pixels, so borders rendered as
#     straight chords across their real shape - a cartoon of the shapefile.
#   - Worse, adjacent features got different tolerances. Ubon was held to 56m
#     while the provinces touching it were cut to 2,541m: a 46x mismatch
#     across a shared border, so the same border drew in two places at once.
#     The province and district layers disagreed for the same reason, being
#     simplified at scales an order of magnitude apart.
#
# A shared border is one line. It can only stay one line if both sides of it,
# in both layers, are thinned by the same amount.
#
# The size this costs is paid for by COORD_DECIMALS below rather than by
# coarsening the geometry.
SHAPE_TOLERANCE = 0.001

# Coordinates are written to 5 decimal places, ~1.1m at this latitude. The
# source is nothing like that accurate, so this discards no real information -
# but json.dump writes floats at full repr precision otherwise, and
# "100.50069326100004" costs 19 characters to say "100.50069".
#
# That is not a micro-optimisation here: it halves the file, which is what
# buys the 3x finer tolerance above at a comparable payload.
COORD_DECIMALS = 5

# The old "minor district" designation. Every one of these was upgraded to a
# full amphoe years ago; where a dataset still carries the prefix, putting it
# on the map would be wrong as well as noisy.
KING_AMPHOE_EN = "KING AMPHOE"
THAI_PREFIXES = ("กิ่งอำเภอ", "อำเภอ", "จังหวัด")


def find_zip(pattern):
    """The single archive matching `pattern` in this folder, or None."""
    hits = sorted(glob.glob(pattern))
    return hits[0] if hits else None


class ZippedShapefile:
    """The one shapefile inside a local .zip, read without unpacking it.

    pyshp is handed the .shp/.shx/.dbf as in-memory streams rather than
    paths, so nothing is ever written to disk. That is deliberate: the zip is
    this project's source of truth for boundaries, and an extracted copy
    sitting beside it is one more thing that can drift out of step with it or
    be read by mistake.

    Encoding and CRS are read from the .cpg/.cst and .prj packed in the same
    archive rather than assumed - see the module docstring for why the two
    archives cannot share one assumption.
    """

    def __init__(self, zip_path, default_encoding="utf-8"):
        self.zip_path = zip_path
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            self.member = self._find_member(names)
            stem = self.member[:-len(".shp")]
            self.encoding = self._encoding(zf, names, stem, default_encoding)
            self.crs = self._crs(zf, names, stem)
            parts = {ext: io.BytesIO(zf.read(self._sidecar(names, stem, ext)))
                     for ext in (".shp", ".shx", ".dbf")}
        self.reader = shapefile.Reader(shp=parts[".shp"], shx=parts[".shx"],
                                       dbf=parts[".dbf"], encoding=self.encoding)

    def _find_member(self, names):
        """The archive's .shp entry.

        Matched on the exact extension, so the QGIS lock files left beside an
        open layer - "...shp.HOSTNAME.1234.sr.lock" - are not mistaken for the
        layer itself.
        """
        hits = sorted(n for n in names if n.lower().endswith(".shp"))
        if len(hits) != 1:
            raise ValueError(f"{self.zip_path}: expected one .shp inside, found {len(hits)}")
        return hits[0]

    @staticmethod
    def _sidecar(names, stem, ext):
        """The archive entry `stem` + `ext`, in whatever case it was zipped."""
        for name in names:
            if name.lower() == (stem + ext).lower():
                return name
        raise KeyError(f"{stem}{ext} missing from the archive")

    def _encoding(self, zf, names, stem, default):
        """The character encoding this shapefile declares for its .dbf.

        Read rather than assumed: the province file says TIS-620 in a .cpg,
        the amphoe file says UTF-8 in a .cst, and decoding Thai with the wrong
        one turns every name into mojibake without raising anything.
        """
        for ext in (".cpg", ".cst"):
            try:
                declared = zf.read(self._sidecar(names, stem, ext)).decode("ascii").strip()
            except KeyError:
                continue
            if declared:
                return declared
        return default

    def _crs(self, zf, names, stem):
        """The CRS this shapefile declares in its .prj, as far as is needed here.

        Deliberately a two-case reading rather than a real WKT parser: these
        files are either already geographic WGS84 or Thailand's UTM zone 47N,
        and pulling in a full projection library to tell those apart would be
        the only reason this script needed one.
        """
        try:
            wkt = zf.read(self._sidecar(names, stem, ".prj")).decode("ascii", errors="ignore")
        except KeyError:
            return DST_CRS
        if "PROJCS" not in wkt:
            return "EPSG:4326"
        if re.search(r"UTM.?[_ ]?Zone.?[_ ]?47", wkt, re.I):
            return "EPSG:32647"
        raise ValueError(
            f"{self.zip_path}: projected CRS this script does not know how to read:\n{wkt}")


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


def thin(geom):
    """One polygon, simplified and rounded ready to write.

    Deliberately takes no arguments beyond the geometry: every feature in both
    layers is thinned identically, which is what keeps a shared border drawn
    as one line rather than two. See SHAPE_TOLERANCE.
    """
    return geom.simplify(SHAPE_TOLERANCE, preserve_topology=True)


def rounded(obj):
    """GeoJSON coordinates at COORD_DECIMALS places.

    Applied to the mapping() output rather than the geometry, so nothing
    downstream of here has to know about it.
    """
    if isinstance(obj, float):
        return round(obj, COORD_DECIMALS)
    if isinstance(obj, (list, tuple)):
        return [rounded(o) for o in obj]
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    return obj


def to_wgs84(geom, src_crs):
    if src_crs == DST_CRS:
        return geom
    return shapely_transform(
        lambda xs, ys: tuple(warp_transform(src_crs, DST_CRS, list(xs), list(ys))), geom)




def province_names(layer):
    """{province code: (english, thai)} from the province shapefile.

    Used to label districts with the province they belong to: the amphoe
    layer identifies a province only by a numeric code, and nothing else in
    the project maps those codes to names.
    """
    out = {}
    for rec in layer.reader.records():
        out[str(rec["PROV_CODE"]).strip()] = (clean_en(rec["PROV_NAME"]),
                                              clean_th(rec["PROV_NAMT"]))
    return out


def build_districts(layer, names_by_code):
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
    crs, reader = layer.crs, layer.reader

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
        geom = thin(to_wgs84(geom, crs))
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
            "geometry": rounded(mapping(geom)),
        })
    return {"type": "FeatureCollection", "features": features}


def build_provinces(layer):
    """All 77 provinces, straight from the province shapefile."""
    crs, reader = layer.crs, layer.reader
    features = []
    for rec, shp in zip(reader.records(), reader.shapes()):
        name_en = clean_en(rec["PROV_NAME"])
        geom = shape(shp.__geo_interface__)
        if not geom.is_valid:
            geom = geom.buffer(0)
        geom = thin(to_wgs84(geom, crs))
        if geom.is_empty:
            continue
        features.append({
            "type": "Feature",
            "properties": {"ADM1_NAME": name_en, "ADM1_NAME_TH": clean_th(rec["PROV_NAMT"])},
            "geometry": rounded(mapping(geom)),
        })
    return {"type": "FeatureCollection", "features": features}


def write(path, collection):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(collection, f, ensure_ascii=False)
    kb = os.path.getsize(path) / 1024
    print(f"  wrote {path}: {len(collection['features'])} features, {kb:,.0f} KB")


def main() -> int:
    layers = {}
    for label, pattern, enc in (("province", PROVINCE_GLOB, "tis-620"),
                                ("amphoe", DISTRICT_GLOB, "utf-8")):
        zip_path = find_zip(pattern)
        if zip_path is None:
            print(f"No {label} archive matching {pattern!r} in this folder.")
            return 1
        layers[label] = ZippedShapefile(zip_path, default_encoding=enc)
        print(f"{label:9} <- {zip_path} :: {layers[label].member}  "
              f"[{layers[label].crs}, {layers[label].encoding}]")

    print(f"\nProvinces (all 77, every one thinned by the same {SHAPE_TOLERANCE}deg) ...")
    write(PROVINCES_OUT, build_provinces(layers["province"]))
    print(f"Districts (all 930, thinned by the same {SHAPE_TOLERANCE}deg) ...")
    write(DISTRICTS_OUT, build_districts(layers["amphoe"], province_names(layers["province"])))
    print("Commit both .geojson files; the archives stay gitignored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
