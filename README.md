# Mekong Water Quality

Satellite-derived turbidity monitoring for Ubon Ratchathani, Thailand.
Sentinel-2 → calibrated MLP → an interactive Leaflet map, with RID
streamflow context and a station risk ranking.

## Run locally

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Deploying as a real website

The dashboard code is ready to deploy; three steps need your own account
login (GitHub, Google) and can't be done for you by an assistant:

### 1. Push this repo to GitHub

```bash
gh repo create mekong-water-quality --private --source=. --remote=origin --push
```

(No `gh` CLI? Create the repo at github.com/new, then:)

```bash
git remote add origin https://github.com/<you>/mekong-water-quality.git
git push -u origin master
```

First, fix the placeholder git identity this repo was committed with:

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```

### 2. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
2. "New app" → pick this repo → main file `dashboard.py`.
3. Before or after first deploy, open **Settings → Secrets** and paste in
   the Google credentials that let the app read satellite composites from
   Drive (it has no local disk of its own). Generate them with:

   ```bash
   python make_secrets.py     # writes .streamlit/secrets.toml, gitignored
   ```

   Then open that file and copy all of it into the Secrets box. Never commit
   it - the refresh token grants Drive access to your Google account until
   revoked at https://myaccount.google.com/permissions.

### 3. Keep it refreshed weekly

`.github/workflows/weekly-refresh.yml` runs the Earth Engine pull on
GitHub's servers every Sunday, so the site stays current without this
machine needing to be on. It needs one repo secret:

1. Open your local Earth Engine credentials file
   (`~/.config/earthengine/credentials`) in a text editor - it's JSON.
2. GitHub repo → Settings → Secrets and variables → Actions → New repository secret.
3. Name: `GEE_CREDENTIALS_JSON`. Value: paste the entire file content.

The existing Windows Task Scheduler job (`MekongWaterQuality_WeeklyRefresh`)
still works as a local backup/dev-time refresh - the two don't conflict,
both just add composites to the same Drive folder.

### 4. After new composites land, refresh the precomputed data

```bash
python precompute_history.py   # rewrites ubon_history.json + display_rasters/
git add ubon_history.json display_rasters/ && git commit
```

One pass writes both artifacts that keep the GeoTIFFs off the render path:

- `ubon_history.json` - the sidebar's province and per-station trend values,
  one float per composite date. Those two charts span *every* composite in
  range, so without this the app opens all of them on a cold visit.
- `display_rasters/*.npz` - the downsampled turbidity raster the map draws,
  one per date. Without these, every date a visitor clicks on the timeline
  re-runs the model over the full-resolution province.

Either way the underlying cost is a ~28MB Drive download plus ~11s of
inference per composite. Precomputed, a date costs a file read of a few
hundred KB and ~20ms - only ~1% of the province is water, so the arrays
compress hard.

The grid those rasters are written on is set by
`province_composite.DISPLAY_RESOLUTION_M`, in **metres of ground, not pixels**.
It is 40m. It used to be a pixel budget (1400 on the long side) which, against
~20m composites, quietly downsampled them 8x and drew the map at 155m - the
Mun River, 200-400m wide, was one to three pixels across. Raising the GEE
export scale would not have helped, because a finer source only made the
downsample factor larger; expressing the target as a distance decouples the
two ends.

40m rather than finer because the province is drawn as one PNG overlay, so
cost follows the whole grid, not the 1% of it that is water:

| grid | on disk, per date | in the map payload | peak to colour |
|---|---|---|---|
| 155m (old) | ~0.1MB | ~65KB | ~30MB |
| **40m** | **2-4MB** | **~370-470KB** | **~118MB** |
| 20m | 1.3-4.2MB | ~1.2MB | ~850MB |

20m is the archive's own resolution and it renders, but colouring a 73Mpx grid
transiently needs more memory than a Streamlit Community Cloud container has.

Skipping this step is not fatal: any date missing from either artifact is
still computed live, it just costs what it used to for that date.

### 5. The streamflow snapshot

`mun_levels.json` holds the daily Mun River stage the sidebar chart plots.
It is committed rather than fetched live because the RID service appears to
refuse requests from outside Thailand: the same code that works on a Thai
connection returns nothing from Streamlit Cloud, leaving the chart reading
"gauge service unavailable" on the deployed site only. The window is fixed
history, so a snapshot carries no staleness risk. Regenerate it from a
machine that can reach the service:

```bash
python -c "import datetime, rid_streamflow as r; r.save_snapshot(datetime.date(2024,10,2), datetime.date(2024,12,31))"
```

The start date is deliberately a month before the displayed range - the
30-day averaging option needs that run-up (see `load_level_history`).

## Cold starts

Streamlit Community Cloud sleeps an idle app, so the first visit after a
quiet spell pays full container startup. What that still covers: the Python
imports, listing the available composites, and the Leaflet map build.

Measured locally on a fresh server process: **6.2s** to a fully painted map,
and **~1.3s** to switch to a date not yet seen in that process. Both used to
be ~11s worse, because each one re-ran the model over the full-resolution
province - see step 4 above, which moved that off the request path.

A warm reload (server process already up) is ~2.1s, essentially all of it
the map build and Streamlit's own boot. Note the cloud still makes one Drive
API call at startup to *list* what composites exist; it no longer downloads
any of them to render.

## Boundary data

Both layers cover the whole country: all 77 provinces and all 930 amphoe.
Every province and district line on the map comes from two local archives and
nothing else - `Province Shapefile.zip` and `Amphoe Shapefile.zip`. No part of
either layer comes from OpenStreetMap, FAO GAUL, or any other source.

The archives are **not** in the repo - ~39MB zipped, and the deployed app has
nothing but the repo - so they are converted once into two GeoJSON caches:

```bash
python import_shapefiles.py   # writes thailand_provinces + thailand_districts .geojson
```

Those two files are committed, and the deployed app reads only them. If you
replace either archive, re-run the script and commit the output. The script
reads the `.shp`/`.shx`/`.dbf` straight out of the zip as in-memory streams and
never unpacks anything to disk, so the archive stays the one copy of the source
geometry. It finds each archive by filename pattern, finds the shapefile inside
it, and reads that layer's encoding and CRS from the packed `.prj` and
`.cpg`/`.cst`, because the two disagree on all three (UTM 47N / TIS-620 against
WGS84 / UTF-8) and have been reorganised more than once.

Each layer is read from its own shapefile and nothing is derived from the
other. The province shapefile is also read for the province *names*, which the
amphoe layer does not carry - it identifies a district's province by code.

One consequence worth knowing: the two datasets disagree along province
borders by a few hundred metres, so with both layers shown, a province edge
and the district edges running along it do not sit exactly on top of each
other.

### Sources that are deliberately *not* used

Each was removed, and each would otherwise be the obvious thing to reach for:

- **The basemap's own borders.** The default tile layer is Esri Light Gray
  Canvas, which draws none. It replaced CartoDB positron, which bakes
  OpenStreetMap's province *and* amphoe boundaries into its tiles as dashed
  pink lines. Those are a different dataset from the shapefiles drawn on top,
  so every border appeared twice, a few hundred metres apart. Tiles are
  images: the lines cannot be restyled or switched off, and `light_nolabels`
  removes only the text. The other basemaps in the picker (Dark, Classic,
  Terrain) still carry them; Satellite does not.
- **OpenStreetMap.** `ubon_boundary.geojson` and `geo_boundary.load_boundary()`
  fetched Ubon's outline from Nominatim and used it for the map's fit bounds.
  That put the rectangle the map fitted to kilometres away from the line it
  drew. Ubon's outline now comes from the province shapefile like every other
  province's, so the two agree by construction.
- **FAO/GAUL via Earth Engine.** The three export scripts
  (`refresh_ubon_data.py`, `backfill_ubon_weekly.py`,
  `export_ubon_monthly_gee.py`) took their study area from
  `FAO/GAUL/2015/level1`. They no longer do, so **`STUDY_AREA` must be set by
  hand before any of them will run** - see the note beside it in
  `refresh_ubon_data.py`. It is left unset rather than defaulted to a
  rectangle on purpose: a wrong footprint does not fail, it exports real
  imagery of the wrong ground. Those scripts run offline against Earth Engine
  and deliberately do not reuse the app's simplified GeoJSON caches.

Station place names in the sidebar are not boundary data either - Nominatim
returns them as text for a point, with no geometry attached.

### Simplification: one tolerance, everywhere

Both layers are thinned before being written, because streamlit-folium
reserialises the whole map on every rerun, so every byte is paid on each
interaction. `SHAPE_TOLERANCE` in `import_shapefiles.py` is **0.001° (~110m),
applied identically to every feature in both layers**.

The uniformity is the point. An earlier scheme scaled the tolerance by
`sqrt(area)` and then exempted Ubon with a fixed fine value. It produced two
visible defects:

| | worst error |
|---|---|
| Provinces | 3,028 m |
| Districts | 1,507 m |
| Ubon | 56 m — while the provinces touching it were cut to 2,541 m |

At the zoom this map opens on, 3km is 10-20 pixels, so borders drew as
straight chords across their real shape. Worse, adjacent features were thinned
by different amounts - a 46× mismatch across Ubon's shared borders, and a
comparable one between the province and district layers - so the *same* border
drew in two places at once, as a doubled line.

A shared border is one line. It stays one line only if both sides of it, in
both layers, are thinned by the same amount. Now:

| | worst error |
|---|---|
| 77 provinces | 159 m |
| 930 districts | 207 m |

The finer tolerance is paid for by `COORD_DECIMALS`, not by geometry.
Coordinates are written to 5 decimal places (~1.1m); `json.dump` otherwise
writes full float repr, spending 19 characters on `100.50069326100004` to say
`100.50069`. That halves the file, which buys the 3× finer tolerance back.

The price: the two caches total 4.5MB, against 2.2MB before. `SHAPE_TOLERANCE`
is the single knob if that is the wrong trade - 0.002° roughly halves the size
and doubles the error, and is still 8× better than what came before it.

### Details that only matter if you regenerate

- Districts are grouped by `AMP_CODE` before use: 37 amphoe are split across
  several records, and emitting those separately would duplicate names and
  draw their internal edges as boundaries.
- The province of a district comes from `AMP_CODE`'s first two digits, not the
  `PRV_CODE` column. They agree wherever `PRV_CODE` is filled in, but 16
  records leave it blank - among them the whole of Nong Bua Lamphu, whose six
  districts would otherwise be unattributable to any province. The same 16
  records have no name either, so those fall back to the code.
- The province shapefile gives two different provinces the English name
  "Nong Khai". Code 38 is really Bueng Kan, split off in 2011, and only its
  English column was never updated - its Thai name is บึงกาฬ and its amphoe
  are Bung Kan, Seka, Si Wilai. The data is passed through as-is, so both
  appear as "Nong Khai" on hover.

## Why Google Drive instead of Cloud Storage

The original plan used Google Cloud Storage, but bucket creation failed:
the GCP project (`gee-training-498303`) has no billing account attached,
and GCS requires one even for near-zero usage. Enabling billing means
adding a payment method in the GCP Console - an account-owner action, not
something automatable. Google Drive works with the same credentials and no
billing requirement, so that's the storage backend for now
(`drive_client.py`). Revisit Cloud Storage if billing ever gets enabled.

## Project layout

- `dashboard.py` - the Streamlit app
- `turbidity_model.py` - the calibrated MLP inference pipeline
- `geo_boundary.py` - province/district boundaries (local shapefile archives), plus road and place-name layers (OpenStreetMap)
- `import_shapefiles.py` - converts the local shapefile archives into the two boundary caches (see below)
- `province_composite.py` - loads Sentinel-2 composites, from local disk or Drive
- `drive_client.py` - shared Google Drive access (dashboard read path)
- `rid_streamflow.py` - RID streamflow API + the `mun_levels.json` snapshot it writes (see module docstring for the API's limitations)
- `refresh_ubon_data.py` / `backfill_ubon_weekly.py` - GEE export + Drive upload/download scripts
- `precompute_history.py` - writes `ubon_history.json` and `display_rasters/` (see step 4)
- `display_rasters/` - precomputed map rasters, committed; what the app renders instead of the GeoTIFFs
- `make_secrets.py` - renders local Google credentials into the Streamlit Cloud secrets block
- `requirements.txt` - deployed-app dependencies; `requirements-dev.txt` adds what the refresh scripts need
