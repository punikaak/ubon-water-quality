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

import ee.oauth as oauth
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

DRIVE_FOLDER_NAME = "GEE_Ubon_Turbidity"


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
    creds = Credentials(
        token=None,
        refresh_token=d["refresh_token"],
        client_id=d.get("client_id", oauth.CLIENT_ID),
        client_secret=d.get("client_secret", oauth.CLIENT_SECRET),
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
