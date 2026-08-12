"""Build the dashboard's water layer from the local Stream.zip archive.

What the archive actually is
---------------------------
Despite the name, Stream.zip holds no stream geometry. It contains six
regional POLYGON layers - Wetland_Central / East / North / Northeast / South /
West - and every one of their 421,319 polygons carries the single attribute
LU_GROUP = "Wetland". So this is Thailand's wetland cover, split by region,
not a river network.

It is drawn as the map's "Water" layer because that is what it represents on
the ground, but nothing here should be read as a channel or a centreline.

Two things about the archive that break naive reading:

  - The .cpg declares "ANSI 874", which is not a codec name Python knows.
    import_shapefiles.ZippedShapefile normalises it to cp874; without that the
    archive raises LookupError before a single feature is read.
  - Shape_Area is not in square metres (it sums to ~16,000 over the whole
    country), so the size cut below measures the geometry instead. In UTM
    zone 47N that is metres by construction, and needs no interpretation.

Why a conversion step
---------------------
Same as import_shapefiles.py: 301MB zipped cannot ride along to Streamlit
Cloud, so it is converted once, here, into a committed GeoJSON that
geo_boundary.load_water() reads.

What gets dropped, and why
--------------------------
421,319 polygons cannot be drawn. This map opens around zoom 10 where one
pixel is ~148m of ground, so a polygon under ~0.02 km2 is smaller than the
pixel it would occupy while still costing its coordinates on every rerun.
MIN_AREA_KM2 keeps the ones with area worth drawing; the run prints how many
survive and what share of the total wetland area they carry.

Run it after replacing the archive:

    python import_water.py

Then commit thailand_water.geojson. The archive stays gitignored.
"""
import glob
import json
import os
import sys

from shapely.geometry import mapping, shape

from import_shapefiles import ZippedShapefile, to_wgs84

ZIP_GLOB = "*Stream*.zip"
MEMBERS = ("Wetland_Central.shp", "Wetland_East.shp", "Wetland_North.shp",
           "Wetland_Northeast.shp", "Wetland_South.shp", "Wetland_West.shp")
WATER_OUT = "thailand_water.geojson"

# Metres, applied in the source CRS before reprojection - the archive is UTM
# zone 47N, so the tolerance is a real ground distance rather than a fraction
# of a degree that means different things east-west and north-south.
#
# 60m, not the 250m this started at. The polygons that survive MIN_AREA_KM2
# are big but extremely detailed - 1,416 of them carry 4.17M vertices, ~2,900
# each - and 250m of allowed error kept only 3.9% of those, which is what made
# the layer read as flat-sided blocks rather than wetland. Measured:
#
#     tol     vertices   share of raw   file
#     250m     163,371           3.9%   3.9MB
#     120m     229,084           5.5%   5.8MB
#      60m     325,681           7.8%   8.2MB
#      30m     475,692          11.4%  11.8MB
#
# 60m is chosen against the zoom this is actually read at: one screen pixel is
# ~148m of ground at zoom 10 and ~18m at zoom 13, so 60m is invisible across
# the range the map opens in and only starts to show past the point where a
# reader is inspecting one wetland rather than the province.
SHAPE_TOLERANCE_M = 60

# Smallest polygon worth drawing, km2. See the module docstring.
MIN_AREA_KM2 = 1.0

# 5 places, ~1.1m. At 4 (~11m) the rounding was a sixth of the tolerance and
# visibly ragged the edges of the smaller polygons once the tolerance came
# down; the extra place costs far less than the detail it preserves.
COORD_DECIMALS = 5


def rounded(obj):
    if isinstance(obj, float):
        return round(obj, COORD_DECIMALS)
    if isinstance(obj, (list, tuple)):
        return [rounded(o) for o in obj]
    if isinstance(obj, dict):
        return {k: rounded(v) for k, v in obj.items()}
    return obj


def build(zip_path):
    min_area_m2 = MIN_AREA_KM2 * 1e6
    features = []
    kept_area = dropped_area = 0.0
    total = dropped = 0

    for member in MEMBERS:
        layer = ZippedShapefile(zip_path, member=member)
        reader = layer.reader
        region_kept = 0
        for shp in reader.iterShapes():
            total += 1
            geom = shape(shp.__geo_interface__)
            if geom.is_empty:
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
            area = geom.area                      # m2: the source CRS is UTM
            if area < min_area_m2:
                dropped += 1
                dropped_area += area
                continue
            kept_area += area
            geom = geom.simplify(SHAPE_TOLERANCE_M, preserve_topology=True)
            if geom.is_empty:
                continue
            geom = to_wgs84(geom, layer.crs)
            if geom.is_empty:
                continue
            features.append({
                "type": "Feature",
                "properties": {"kind": "water"},
                "geometry": rounded(mapping(geom)),
            })
            region_kept += 1
        print(f"  {member:24} kept {region_kept:7,}", flush=True)
        reader.close()

    area_share = 100 * kept_area / (kept_area + dropped_area) if total else 0
    print(f"\n  kept {len(features):,} of {total:,} polygons "
          f"({100*len(features)/total:.1f}%), carrying {area_share:.1f}% of the "
          f"total wetland area; dropped {dropped:,} below {MIN_AREA_KM2} km2")
    return {"type": "FeatureCollection", "features": features}


def main() -> int:
    hits = sorted(glob.glob(ZIP_GLOB))
    if not hits:
        print(f"No archive matching {ZIP_GLOB!r} in this folder.")
        return 1
    zip_path = hits[0]
    print(f"water <- {zip_path} ({os.path.getsize(zip_path)/1e6:.0f} MB), "
          f"{len(MEMBERS)} regional layers\n")

    collection = build(zip_path)
    with open(WATER_OUT, "w", encoding="utf-8") as f:
        json.dump(collection, f, ensure_ascii=False)
    mb = os.path.getsize(WATER_OUT) / 1e6
    print(f"\n  wrote {WATER_OUT}: {len(collection['features']):,} features, "
          f"{mb:.1f} MB (thinned {SHAPE_TOLERANCE_M}m, {COORD_DECIMALS} dp)")
    print("Commit thailand_water.geojson; the archive stays gitignored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
