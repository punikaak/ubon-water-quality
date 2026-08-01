"""Streamflow/water level data for the RID Lower NE Hydro Center (hydro-4.rid.go.th).

Live source: https://hyd-app-db.rid.go.th (UtokID=4 = Lower Northeastern Region),
the same API that backs http://hydro-4.rid.go.th's "water level at 06:00" page.
As of this writing the service accepts the request (HTTP 200) but returns an
empty/null body for every payload encoding tried (plain JSON, bracket-form,
dot-form) - it likely needs a session/token nuance not visible from outside
the page's own JS. fetch_daily_streamflow() is kept as a best-effort live
attempt.

Note: Sentinel2_Ready_Train_With_Streamflow.csv (the only "streamflow" data
already in this project) covers Nakhon Phanom / Sakon Nakhon / Bueng Kan /
Nong Khai stations on the upper Mekong tributaries - a different river system
from the Mun River / Ubon Ratchathani stations this dashboard covers. It is
NOT a valid offline fallback here, so none is used: if the live call fails,
get_streamflow() reports that honestly instead of substituting unrelated data.
"""
import datetime as dt

import requests

API_URL = "https://hyd-app-db.rid.go.th/webservice/getDailyWaterLevelListReportAD.ashx?option=2"
PAGE_URL = "https://hyd-app-db.rid.go.th/hydro4d_admsl.html"
UTOK_ID = 4  # Lower Northeastern Region Hydrological Irrigation Center

# RID gauge codes on the Mun River system near the Ubon PCD stations
STATIONS_OF_INTEREST = {
    "M.7": "Mun River, Warin Chamrap, Ubon Ratchathani",
    "M.9": "Huai Samran, Mueang, Si Sa Ket",
    "M.11B": "Mun River, Phibun Mangsahan, Ubon Ratchathani",
    "M.32": "Lam Se Bai, Pa Tio, Yasothon",
}


def _thai_date(date: dt.date) -> str:
    return f"{date.day:02d}/{date.month:02d}/{date.year + 543}"


def fetch_daily_streamflow(date: dt.date | None = None, timeout: float = 8.0) -> dict:
    """Best-effort live fetch. Returns {} if the RID service gives nothing usable."""
    date = date or dt.date.today()
    date_str = _thai_date(date)
    try:
        session = requests.Session()
        session.get(PAGE_URL, timeout=timeout)
        resp = session.post(
            API_URL,
            data={"DW[UtokID]": str(UTOK_ID), "DW[TimeCurrent]": date_str},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    rows = data.get("rows") or []
    out = {}
    for row in rows:
        code = row.get("stationcode")
        if code not in STATIONS_OF_INTEREST:
            continue
        q1 = str(row.get("waterlevelvalueQ1", ""))
        parts = q1.split("|")
        out[code] = {
            "name": STATIONS_OF_INTEREST[code],
            "waterlevel_m": _safe_float(parts[0]) if len(parts) > 0 else None,
            "discharge_cms": _safe_float(parts[1]) if len(parts) > 1 else None,
            "capacity_percent": row.get("capacitypercent"),
            "status": row.get("wlstatus"),
            "source": "live",
        }
    return out


def get_streamflow(rid_gauge_code: str | None = None, date: dt.date | None = None) -> dict:
    """Live RID reading if available; otherwise reports unavailable (no substitute data exists)."""
    live = fetch_daily_streamflow(date=date)
    if live:
        return {"source": "live", "stations": live}
    return {"source": "unavailable", "stations": {}}


def _safe_float(s: str):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None
