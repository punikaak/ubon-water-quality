"""Google Drive access for the Ubon_S2_*.tif composites, shared by the local
refresh scripts and the deployed dashboard.

Credentials come from one of two places:
  - Cloud (Streamlit Community Cloud): st.secrets["gcp_drive"] - a table with
    refresh_token/client_id/client_secret/scopes, set once in the app's
    Settings > Secrets. Never commit these values to the repo.
  - Local dev: the same OAuth token Earth Engine already has cached at
    ~/.config/earthengine/credentials (its granted scopes include Drive).

Why Drive and not Google Cloud Storage: GCS bucket creation on this GCP
project (gee-training-498303) returned "billing account ... disabled" -
enabling billing needs a payment method added in the GCP Console, which is
the account owner's call, not something done here. Drive already works with
the same credentials and has no billing requirement, so it's the storage
backend for the deployed app until/unless that changes.
"""
import io
import json
import os

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

DRIVE_FOLDER_NAME = "GEE_Ubon_Turbidity"

# Earth Engine's own public installed-app OAuth identifiers. Hard-coded rather
# than read from ee.oauth because earthengine-api is a LOCAL-only dependency
# (see requirements-dev.txt) - importing it at module scope crashed the
# deployed app, for which Drive is the only composite source. These are not
# secrets: they ship inside the public earthengine-api package. Any real
# secrets.toml overrides them anyway (see _load_credentials_dict).
EE_CLIENT_ID = ("517222506229-vsmmajv00ul0bs7p89v5m89qs8eb9359"
                ".apps.googleusercontent.com")
EE_CLIENT_SECRET = "RUP0RZ6e0pPhDzsqIJ7KlNd1"


def _ee_oauth_defaults() -> tuple[str, str]:
    """Prefer the values the installed earthengine-api reports, so a future
    rotation upstream is picked up locally; fall back to the constants above
    where that package isn't installed (i.e. the deployed app)."""
    try:
        import ee.oauth as oauth
        return oauth.CLIENT_ID, oauth.CLIENT_SECRET
    except Exception:
        return EE_CLIENT_ID, EE_CLIENT_SECRET


def _load_credentials_dict() -> dict:
    try:
        import streamlit as st
        if "gcp_drive" in st.secrets:
            return dict(st.secrets["gcp_drive"])
    except Exception:
        pass

    cred_path = os.path.expanduser("~/.config/earthengine/credentials")
    if os.path.exists(cred_path):
        with open(cred_path, encoding="utf-8") as f:
            return json.load(f)

    raise RuntimeError(
        "No Google credentials found. Locally: run 'earthengine authenticate' once. "
        "On Streamlit Cloud: add a [gcp_drive] table to the app's secrets with "
        "refresh_token/client_id/client_secret/scopes."
    )


def get_drive_service():
    d = _load_credentials_dict()
    default_id, default_secret = _ee_oauth_defaults()
    creds = Credentials(
        token=None,
        refresh_token=d["refresh_token"],
        client_id=d.get("client_id") or default_id,
        client_secret=d.get("client_secret") or default_secret,
        token_uri="https://oauth2.googleapis.com/token",
        scopes=d["scopes"] if isinstance(d["scopes"], list) else list(d["scopes"]),
    )
    return build("drive", "v3", credentials=creds)


def list_remote_composites() -> list[dict]:
    """[{id, name}, ...] for every Ubon_S2_*.tif file in Drive."""
    drive = get_drive_service()
    query = "name contains 'Ubon_S2_' and name contains '.tif' and trashed = false"
    files, page_token = [], None
    while True:
        resp = drive.files().list(
            q=query, fields="nextPageToken, files(id, name)", pageSize=200, pageToken=page_token
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def download_file(file_id: str, filename: str, local_dir: str = ".") -> str:
    drive = get_drive_service()
    out_path = os.path.join(local_dir, filename)
    request = drive.files().get_media(fileId=file_id)
    with io.FileIO(out_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return out_path
