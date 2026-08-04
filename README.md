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
3. Before or after first deploy, open **Settings → Secrets** and paste the
   contents of `.streamlit/secrets.toml.example` with real values filled in
   (see that file for where to get them). This is what lets the deployed
   app read satellite composites from Google Drive, since it has no local
   disk of its own.

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

### 4. After new composites land, refresh the history cache

```bash
python precompute_history.py   # rewrites ubon_history.json, then commit it
```

`ubon_history.json` holds the sidebar's province and per-station trend
values, one float per composite date. Without it the deployed app has to
open every composite in range to draw those two charts - each one a ~28MB
Drive download plus ~12s of inference, so a cold visit would sit blank for
minutes. Skipping this step is not fatal: any date missing from the file is
still computed live, it just costs what it used to for that date.

## Cold starts

Streamlit Community Cloud sleeps an idle app, so the first visit after a
quiet spell pays full startup cost. What that covers: one composite
download + inference, the RID streamflow API call, and the Leaflet map
build. Measured ~34s locally with the composite already on disk; expect
longer on the first cloud load while the composite comes down from Drive.
Subsequent visits hit Streamlit's cache and are immediate.

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
- `geo_boundary.py` - province/district/road reference layers (Earth Engine + OpenStreetMap)
- `province_composite.py` - loads Sentinel-2 composites, from local disk or Drive
- `drive_client.py` - shared Google Drive access (dashboard read path)
- `rid_streamflow.py` - RID streamflow API (best-effort; see module docstring for its limitations)
- `refresh_ubon_data.py` / `backfill_ubon_weekly.py` - GEE export + Drive upload/download scripts
- `precompute_history.py` - writes `ubon_history.json`, the sidebar trend cache (see step 4)
- `requirements.txt` - deployed-app dependencies; `requirements-dev.txt` adds what the refresh scripts need
