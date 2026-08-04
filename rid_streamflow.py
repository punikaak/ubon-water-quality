"""Streamflow / water level for the RID Lower NE Hydro Center (hydro-4.rid.go.th).

Source: https://hyd-app-db.rid.go.th - the service backing hydro-4's daily
water level pages, reached through its HDService.svc JSON API.

Two things about this API are worth knowing before changing anything here:

1. The `getDailyWaterLevelListReport*.ashx` endpoints the public page appears
   to use return a literal `null` body for every payload encoding tried
   (plain, bracket-form, dot-form), with or without a warmed-up session. They
   are not usable from outside the page. `HDService.svc` *is*.

2. `HDService.svc` identifies stations by an internal numeric id, not by the
   public "M.7"-style gauge code - passing the gauge code returns an
   all-null record. The ids below were found by probing
   getStationFromStationID over a numeric range and reading back the
   stationcode of each hit.

`GetDailyStageReportChartFromStationID6Months` returns one row per day for
the six months ending at the requested date, which is what makes a fixed
historical window (e.g. Nov-Dec 2024) retrievable at all.

Each row's `hvalues` is [measured, ref, ref]: index 0 is the level actually
recorded that day, and the other two are the 2541/2554 flood comparison
series the page overlays (see GetDailyWaterLevelChartsType, whose type names
are those Buddhist years). Only index 0 is real observed data.

`qvalues` (discharge, m^3/s) is present in the schema but comes back null
for these gauges, so what this module can honestly report is stage/water
level in metres, not a discharge rate.
"""
import datetime as dt

import requests

PAGE_URL = "https://hyd-app-db.rid.go.th/hydro4d_admsl.html"
HDSERVICE = "https://hyd-app-db.rid.go.th/webservice/HDService.svc/"

# Internal numeric StationIDs (see note 2 above) for the Mun River system
# gauges nearest the Ubon PCD water-quality stations.
MUN_STATIONS = {
    "M.7": {"id": 279, "name": "Mun River, Warin Chamrap, Ubon Ratchathani"},
    "M.11B": {"id": 281, "name": "Mun River, Phibun Mangsahan, Ubon Ratchathani"},
    "M.9": {"id": 280, "name": "Huai Samran, Mueang, Si Sa Ket"},
}


def _thai_date(date: dt.date) -> str:
    """DD/MM/YYYY in the Buddhist era, which is what the API expects."""
    return f"{date.day:02d}/{date.month:02d}/{date.year + 543}"


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0", "Referer": PAGE_URL})
    try:
        s.get(PAGE_URL, timeout=15)
    except requests.RequestException:
        pass  # the warm-up is best-effort; the API works without it
    return s


def _parse_dotnet_date(raw: str) -> dt.date:
    """"/Date(1735599600000+0700)/" -> date, in the +07 offset the API sends."""
    ms = int(raw[6:].split("+")[0].split(")")[0])
    return (dt.datetime(1970, 1, 1) + dt.timedelta(milliseconds=ms, hours=7)).date()


def fetch_level_history(end_date: dt.date, timeout: float = 30.0) -> dict:
    """{gauge_code: [(date, level_m), ...]} for the 6 months ending `end_date`.

    Returns {} if the service is unreachable or returns nothing usable -
    callers should treat an empty result as "no data", not as zero.
    """
    session = _session()
    payload_date = _thai_date(end_date)
    out = {}
    for code, meta in MUN_STATIONS.items():
        try:
            resp = session.post(
                HDSERVICE + "GetDailyStageReportChartFromStationID6Months",
                json={"hydro": {"StationID": str(meta["id"]), "TimeCurrent": payload_date}},
                headers={"Content-Type": "application/json; charset=utf-8"},
                timeout=timeout,
            )
            resp.raise_for_status()
            rows = resp.json()
        except (requests.RequestException, ValueError):
            continue
        if not isinstance(rows, list):
            continue

        series = []
        for row in rows:
            values = row.get("hvalues") or []
            if not values or values[0] is None:
                continue
            try:
                series.append((_parse_dotnet_date(str(row.get("time", ""))), float(values[0])))
            except (ValueError, IndexError):
                continue
        if series:
            out[code] = sorted(series)
    return out


def level_history_between(start: dt.date, end: dt.date, timeout: float = 30.0) -> dict:
    """{gauge_code: [(date, level_m), ...]} restricted to [start, end].

    One call per gauge covers six months back from `end`, so any window
    shorter than that (which is the point of this function) needs no paging.
    """
    history = fetch_level_history(end, timeout=timeout)
    return {
        code: [(d, v) for d, v in series if start <= d <= end]
        for code, series in history.items()
        if any(start <= d <= end for d, _ in series)
    }


def station_name(code: str) -> str:
    return MUN_STATIONS.get(code, {}).get("name", code)
