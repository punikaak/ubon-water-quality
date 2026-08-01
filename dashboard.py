"""Mekong Water Quality - satellite-derived turbidity monitoring, Ubon Ratchathani.

Sentinel-2 -> MLP (calibrated) turbidity map, RID streamflow context, and a
high-risk station ranking. The Leaflet map fills the full viewport. Left
panel is Streamlit's native (foldable) sidebar. Right panel (legend) folds
via a toggle button since Streamlit has no native right sidebar.

Run with:  streamlit run dashboard.py
"""
import folium
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

import geo_boundary as geo
import province_composite as pc
import rid_streamflow as rid
import turbidity_model as tm
import turbidity_style as style

st.set_page_config(page_title="Mekong Water Quality", layout="wide")

VALIDATION_CSV = "Sentinel2_Extract_Ubon_New.csv"
RID_GAUGES = rid.STATIONS_OF_INTEREST
FOCUS_PROVINCE = "Ubon Ratchathani"

# One representative tile (z=7, x=101, y=58 - covers Ubon Ratchathani) from
# each provider, reused both as the live basemap and as the small preview
# thumbnail in the sidebar picker.
BASEMAPS = {
    "Dark": {
        "tiles": "CartoDB dark_matter", "attr": None,
        "thumb": "https://a.basemaps.cartocdn.com/dark_all/7/101/58.png",
        "desc": "Displays a map in dark theme",
    },
    "Light": {
        "tiles": "CartoDB positron", "attr": None,
        "thumb": "https://a.basemaps.cartocdn.com/light_all/7/101/58.png",
        "desc": "Displays a map in light theme",
    },
    "Classic": {
        "tiles": "OpenStreetMap", "attr": None,
        "thumb": "https://a.tile.openstreetmap.org/7/101/58.png",
        "desc": "Displays the default road map view",
    },
    "Terrain": {
        "tiles": "https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr": "Map data: OpenStreetMap contributors, SRTM | Map style: OpenTopoMap (CC-BY-SA)",
        "thumb": "https://a.tile.opentopomap.org/7/101/58.png",
        "desc": "Displays the terrain road map view",
    },
    "Satellite": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri World Imagery",
        "thumb": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/7/58/101",
        "desc": "High-resolution aerial imagery (Esri World Imagery)",
    },
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }
    #MainMenu, footer {visibility: hidden;}
    header [data-testid="stToolbarActions"] {visibility: hidden;}
    .block-container { padding: 0 0.6rem 0.4rem 0.6rem; max-width: 100%; }
    iframe[title="streamlit_folium.st_folium"] { height: calc(100vh - 84px) !important; min-height: 560px; }

    .topbar { display:flex; align-items:baseline; justify-content:space-between;
        padding: 8px 6px; border-bottom: 2px solid #e7eaf0; margin-bottom: 6px; }
    .topbar-title { font-size: 1.25rem; font-weight: 700; color: #1e3a4a; }
    .topbar-sub { font-size: 0.8rem; color: #6b7684; margin-left: 10px; }

    .card { border-radius: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.12);
        margin-bottom: 14px; overflow:hidden; background: var(--background-color, #fff); }
    .card-body { padding: 12px 14px; }
    .card-head { padding: 8px 14px; font-weight:700; font-size:1.02rem; }
    .head-teal { background: linear-gradient(90deg,#7be8c4,#3ed99b); color:#0d4a35; }

    .legend-item { display:flex; align-items:center; gap:8px; padding:4px 0; font-size:0.86rem; color:#2b2b3a; }
    .legend-swatch { width:12px; height:12px; border-radius:3px; display:inline-block; flex-shrink:0; }
    .legend-heading { font-weight:700; font-size:0.82rem; text-transform:uppercase; letter-spacing:.03em;
        color:#5a6474; margin: 10px 0 4px 0; }
    .legend-caption { font-size:0.76rem; color:#8592A3; margin-top:6px; }

    .sb-metric { border:1px solid #E7EAF0; border-radius:12px; padding:10px 12px; margin-bottom:10px; }
    .sb-value { font-size:1.5rem; font-weight:700; }
    .sb-label { font-size:0.72rem; color:#8592A3; text-transform:uppercase; letter-spacing:.04em; }

    .risk-row { display:flex; justify-content:space-between; align-items:center;
        padding: 7px 10px; border-radius: 10px; margin-bottom:4px; }
    .risk-row:hover { background:#F4F6F9; }
    .risk-pill { display:inline-block; padding: 2px 10px; border-radius: 999px; font-weight:600;
        font-size: 0.78rem; color: #2b2b3a; }

    .basemap-thumb { width:100%; border-radius:8px; aspect-ratio: 1.4; object-fit:cover; }
    .basemap-desc { font-size:0.74rem; color:#8592A3; margin-top:-6px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- data ----
@st.cache_data(show_spinner="Loading satellite composite...")
def load_province_composite(path: str):
    local_path = pc.ensure_local(path)
    return pc.load_composite(local_path)


@st.cache_data(show_spinner=False)
def load_validation():
    df = pd.read_csv(VALIDATION_CSV)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    df = df.dropna(subset=tm.FEATURES + ["Turbidity_"])
    df["Date"] = pd.to_datetime(df["Date"])
    X = df[tm.FEATURES].to_numpy()
    y = df["Turbidity_"].to_numpy()
    y_pred = tm.predict(X)
    df = df.assign(Predicted_NTU=y_pred)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
    return df, y, y_pred, r2, rmse


def turbidity_overlay_rgba(turbidity_map, water_mask):
    breakpoints = [c["max"] for c in style.CLASSES[:-1]]
    colors = [c["color"] for c in style.CLASSES]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm([0] + breakpoints + [breakpoints[-1] * 3], cmap.N)
    rgba = cmap(norm(turbidity_map))
    rgba[..., 3] = np.where(water_mask, 0.85, 0.0)
    return rgba


def render_map_legend():
    rows = "".join(
        f'<div class="legend-item"><span class="legend-swatch" style="background:#9aa3ad;"></span>{label}</div>'
        for label in [
            "Basemap", "Province boundary", "District boundary", "Turbidity", "Ground stations",
        ]
    )
    turbidity_rows = "".join(
        f'<div class="legend-item"><span class="legend-swatch" style="background:{c["color"]}"></span>{c["label"]}</div>'
        for c in style.CLASSES
    )
    html = (
        '<div class="card"><div class="card-head head-teal">Map Legend</div><div class="card-body">'
        f'<div class="legend-heading">Layers</div>{rows}'
        f'<div class="legend-heading">Turbidity Levels</div>{turbidity_rows}'
        '<div class="legend-caption">General reference scale for this dashboard, not an official Thai PCD standard.</div>'
        '</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


df_val, y, y_pred, r2, rmse = load_validation()
station_summary = (
    df_val.groupby("Code", as_index=False)
    .agg(Name=("Name", "first"), station_la=("station_la", "first"), station_lo=("station_lo", "first"),
         Turbidity_Actual=("Turbidity_", "mean"), Predicted_NTU=("Predicted_NTU", "mean"))
    .sort_values("Predicted_NTU", ascending=False)
)
latest_row = df_val.sort_values("Date").iloc[-1]

# ------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.markdown("### Mekong Water Quality")
    st.caption("Situation overview - Ubon Ratchathani")

    st.markdown("#### Latest Turbidity")
    c = style.classify(latest_row["Predicted_NTU"])
    st.markdown(
        f'<div class="sb-metric"><div class="sb-value">{latest_row["Predicted_NTU"]:.1f} NTU</div>'
        f'<div class="sb-label">predicted &middot; {latest_row["Code"]} &middot; '
        f'{latest_row["Date"].date().isoformat()}</div>'
        f'<span class="legend-swatch" style="background:{c["color"]}"></span> {c["label"]}'
        f'&nbsp;&nbsp;<span style="color:#8592A3;">(PCD actual: {latest_row["Turbidity_"]:.1f} NTU)</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Streamflow Summary")
    gauge_label = st.selectbox("Gauge", [f"{code} - {name}" for code, name in RID_GAUGES.items()],
                                label_visibility="collapsed")
    gauge_code = gauge_label.split(" - ")[0]
    flow = rid.get_streamflow(gauge_code)
    if flow["source"] == "live" and flow["stations"]:
        info = flow["stations"].get(gauge_code) or next(iter(flow["stations"].values()))
        d1, d2 = st.columns(2)
        d1.metric("Discharge", f'{info["discharge_cms"]:.1f} cms' if info["discharge_cms"] is not None else "n/a")
        d2.metric("Level", f'{info["waterlevel_m"]:.2f} m' if info["waterlevel_m"] is not None else "n/a")
        st.caption(f"Live RID | {info.get('status', '-')}")
    else:
        st.warning("Live RID feed unavailable right now.")
        st.caption("No offline substitute used (only unrelated-basin data exists locally).")

    st.markdown("#### Stations, High Risk First")
    for _, r in station_summary.iterrows():
        cls = style.classify(r["Predicted_NTU"])
        st.markdown(
            f'<div class="risk-row"><span>{r["Code"]}</span>'
            f'<span class="risk-pill" style="background:{cls["color"]}">{r["Predicted_NTU"]:.1f} NTU &middot; {cls["label"]}</span></div>',
            unsafe_allow_html=True,
        )

    if "basemap" not in st.session_state:
        st.session_state.basemap = "Light"

    with st.expander("Base Map", expanded=False):
        for name, cfg in BASEMAPS.items():
            with st.container(border=True):
                c_img, c_text = st.columns([1, 2])
                with c_img:
                    st.markdown(f'<img class="basemap-thumb" src="{cfg["thumb"]}">', unsafe_allow_html=True)
                with c_text:
                    is_selected = st.session_state.basemap == name
                    if st.button(name, key=f"basemap_{name}", use_container_width=True,
                                 type="primary" if is_selected else "secondary"):
                        st.session_state.basemap = name
                        st.rerun()
                    st.markdown(f'<div class="basemap-desc">{cfg["desc"]}</div>', unsafe_allow_html=True)

# --------------------------------------------------------------- top bar --
if "legend_open" not in st.session_state:
    st.session_state.legend_open = True

top_l, top_r = st.columns([6, 1])
with top_l:
    st.markdown(
        '<div class="topbar"><span class="topbar-title">Mekong Water Quality</span>'
        '<span class="topbar-sub">Satellite-derived turbidity monitoring - Ubon Ratchathani, Thailand</span></div>',
        unsafe_allow_html=True,
    )
with top_r:
    if st.button("Hide legend" if st.session_state.legend_open else "Show legend"):
        st.session_state.legend_open = not st.session_state.legend_open
        st.rerun()

if st.session_state.legend_open:
    col_map, col_right = st.columns([3.3, 1])
else:
    col_map = st.container()
    col_right = None

with col_map:
    available = pc.list_available_composites(".")
    if not available:
        st.error(
            "No province composites found (expected Ubon_S2_YYYYMMDD.tif files). "
            "Run refresh_ubon_data.py or backfill_ubon_weekly.py first."
        )
    else:
        dates = [d for d, _ in available]
        picked_date = st.selectbox(
            "Imagery date (7-day composite ending this date)", dates,
            index=len(dates) - 1, format_func=lambda d: d.strftime("%d %b %Y"),
            label_visibility="collapsed",
        )
        picked_path = dict(available)[picked_date]

        rgb, turbidity_map, valid_mask, bounds = load_province_composite(picked_path)

        boundary = geo.load_boundary()
        b_minx, b_miny, b_maxx, b_maxy = boundary.bounds

        fmap = folium.Map(zoom_start=8, tiles=None)
        fmap.fit_bounds([[b_miny, b_minx], [b_maxy, b_maxx]])

        # --- Basemap chosen via the sidebar's "Base Map" card picker ---
        chosen = BASEMAPS[st.session_state.basemap]
        folium.TileLayer(
            tiles=chosen["tiles"], attr=chosen["attr"], name=st.session_state.basemap, control=False,
        ).add_to(fmap)

        # --- All Thailand provinces, Ubon Ratchathani highlighted ---
        try:
            provinces_geojson = geo.load_thailand_provinces()

            def province_style(feature):
                is_focus = feature["properties"].get("ADM1_NAME") == FOCUS_PROVINCE
                return {
                    "color": "#e05a2b" if is_focus else "#9aa3ad",
                    "weight": 3 if is_focus else 1,
                    "fillOpacity": 0,
                }

            folium.GeoJson(
                provinces_geojson, name="Provinces", style_function=province_style,
                tooltip=folium.GeoJsonTooltip(fields=["ADM1_NAME"], aliases=[""]),
                show=True,
            ).add_to(fmap)
        except FileNotFoundError as e:
            st.info(str(e))

        # --- Ubon districts (off by default - secondary detail) ---
        try:
            districts_geojson = geo.load_ubon_districts()
            folium.GeoJson(
                districts_geojson, name="Districts",
                style_function=lambda f: {"color": "#6b7684", "weight": 1, "dashArray": "3,3", "fillOpacity": 0},
                tooltip=folium.GeoJsonTooltip(fields=["ADM2_NAME"], aliases=[""]),
                show=False,
            ).add_to(fmap)
        except FileNotFoundError:
            pass

        overlay_rgba = turbidity_overlay_rgba(turbidity_map, valid_mask)
        folium.raster_layers.ImageOverlay(
            image=overlay_rgba,
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            opacity=0.9, name="Turbidity", show=True,
        ).add_to(fmap)

        station_layer = folium.FeatureGroup(name="Ground Stations", show=True)
        for _, r in station_summary.iterrows():
            cls = style.classify(r["Predicted_NTU"])
            folium.CircleMarker(
                location=[r["station_la"], r["station_lo"]],
                radius=8, color="#2b2b3a", weight=1, fill=True,
                fill_color=cls["color"], fill_opacity=0.95,
                popup=folium.Popup(
                    f"<b>{r['Code']}</b><br>Predicted: {r['Predicted_NTU']:.1f} NTU"
                    f"<br>Actual (PCD): {r['Turbidity_Actual']:.1f} NTU<br>Class: {cls['label']}",
                    max_width=200,
                ),
                tooltip=r["Code"],
            ).add_to(station_layer)
        station_layer.add_to(fmap)

        folium.LayerControl(collapsed=True).add_to(fmap)
        st_folium(fmap, use_container_width=True, height=750, returned_objects=[])

if col_right is not None:
    with col_right:
        render_map_legend()
