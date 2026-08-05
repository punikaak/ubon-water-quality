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
inference per composite. Precomputed, a date costs a ~80KB file read and
~20ms - only ~1% of the province is water, so the arrays compress hard.

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

Province and district outlines come from Thai shapefiles kept locally in
`Province Shapefile/` and `Tambon Shapefile/`. They are **not** in the repo -
together they are ~68MB and `TH_Tambon.shp` alone exceeds GitHub's 50MB file
limit - so they are converted once into two small GeoJSON caches that are:

```bash
python import_shapefiles.py   # writes thailand_provinces.geojson + ubon_districts.geojson
```

Those two files are committed, and the deployed app reads only them. If you
replace either shapefile, re-run the script and commit the output.

The tambon (subdistrict) file is dissolved by amphoe to produce the district
layer. That is more complete than the FAO GAUL data this replaced - 25 of
Ubon's districts against GAUL's 20 - and it carries Thai names, so boundary
hover labels now follow the interface language. As a cross-check on the
conversion, the 25 dissolved districts fill the province outline from the
separate province shapefile to within 0.02% by area.

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
- `geo_boundary.py` - province/district/road reference layers (local shapefiles + OpenStreetMap)
- `import_shapefiles.py` - converts the local Thai shapefiles into the two boundary caches (see below)
- `province_composite.py` - loads Sentinel-2 composites, from local disk or Drive
- `drive_client.py` - shared Google Drive access (dashboard read path)
- `rid_streamflow.py` - RID streamflow API + the `mun_levels.json` snapshot it writes (see module docstring for the API's limitations)
- `refresh_ubon_data.py` / `backfill_ubon_weekly.py` - GEE export + Drive upload/download scripts
- `precompute_history.py` - writes `ubon_history.json` and `display_rasters/` (see step 4)
- `display_rasters/` - precomputed map rasters, committed; what the app renders instead of the GeoTIFFs
- `make_secrets.py` - renders local Google credentials into the Streamlit Cloud secrets block
- `requirements.txt` - deployed-app dependencies; `requirements-dev.txt` adds what the refresh scripts need
