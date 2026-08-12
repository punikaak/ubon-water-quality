"""Mekong Water Quality - satellite-derived turbidity monitoring, Ubon Ratchathani.

Sentinel-2 -> MLP (calibrated) turbidity map, rendered full-bleed with a
fold-out control rail (legend, layer toggles, basemap picker) styled after
ADPC's Air4Laos dashboard.

The left sidebar is a fixed situation overview: latest province-wide
turbidity, a per-station turbidity trend, daily Mun River water level from
the RID gauges, districts ranked by turbidity with a risk class, and the
station list. The headline figure, district ranking and map markers all
follow whichever composite date is selected on the map's timeline; the two
trend charts always cover the full analysis window (RANGE_START..RANGE_END).

Run with:  streamlit run dashboard.py
"""
import base64
import datetime as dt
import html
import io
import json
import os

import altair as alt
import folium
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import streamlit.components.v1 as components
from streamlit_folium import st_folium

import geo_boundary as geo
import map_controls
import province_composite as pc
import rid_streamflow as rid
import turbidity_model as tm
import turbidity_style as style

st.set_page_config(page_title="Mekong Water Quality", layout="wide")

VALIDATION_CSV = "Sentinel2_Extract_Ubon_New.csv"
HISTORY_CACHE = "ubon_history.json"  # written by precompute_history.py
FOCUS_PROVINCE = "Ubon Ratchathani"

# This dashboard's original analysis window - composites outside this range
# (e.g. from the ongoing weekly refresh automation) are excluded so the
# timeline only ever shows the period this deployment was built to cover.
RANGE_START = dt.date(2024, 11, 1)
RANGE_END = dt.date(2024, 12, 31)

# Actual on-map symbol colors, reused so the legend and the layer-toggle
# panel never drift out of sync with what's really drawn on the map.
PROVINCE_LINE_COLOR = "#9aa3ad"
# The highlight around Ubon itself, drawn heavier than the other provinces'
# outlines. Black rather than a hue: the turbidity overlay it encloses is
# itself a colour scale, so a coloured boundary competed with the data for
# attention and, at the orange end of that scale, blended into it.
PROVINCE_FOCUS_COLOR = "#000000"
# Lighter and thinner than the province line above it, not darker. The
# district layer is every amphoe in Thailand - 930 of them - and it used to be
# the darker of the two, dashed: at country zoom that reads as a grey haze
# with the province borders lost inside it. Subdividing lines should sit
# under the lines they subdivide.
DISTRICT_LINE_COLOR = "#c2c8d0"

# Water layer (Thailand's wetland areas - see import_water.py).
#
# Quiet steel blue, measured rather than picked. This layer is the bed the
# reading sits in; the turbidity ramp on top is the thing being read, and a
# heavy water fill competes with it for attention even though it cannot cover
# it (the raster is drawn after, and opaque).
#
# What matters is the COMPOSITED colour, not the swatch - a translucent fill
# blends toward the pale canvas. And fading it is not simply safe: the lowest
# turbidity class "Excellent" is #A2C0FC, itself a light blue, so the paler
# the water gets the CLOSER it moves to that class. Measured in CIE Lab
# against the classes actually painted on the map (the "unavailable" grey
# lives only in the sidebar choropleth, so it does not constrain this):
#
#     fill      op   composite   vs ramp   vs canvas
#     #1A4E8A  .75     #4F76A4      29.5        53.7   <- loud
#     #1A4E8A  .70     #597EA9      26.9        50.0
#     #3A6C96  .62     #7E9EB8      22.8        35.0   <- this
#     #5B87A8  .50     #A4BBCC      23.3        22.6   <- too faint
#     #5B87A8  .35     #BACBD7      26.2        15.6   <- invisible
#
# There is no setting that is both far from the ramp and strongly visible;
# the two constraints pull opposite ways, so this takes a point between them.
#
# Note the ramp column barely moves across this range while the canvas column
# doubles - so "darker" costs almost nothing in confusability here, and the
# earlier worry about a heavy fill was really the z-order bug below, not the
# shade. The raster is opaque and now genuinely on top, so this cannot dim a
# reading whatever its opacity.
WATER_FILL_COLOR = "#3a6c96"
WATER_LINE_COLOR = "#2b5478"
WATER_FILL_OPACITY = 0.62
# What WATER_FILL_COLOR at WATER_FILL_OPACITY composites to over the Light
# basemap - the colour the reader actually sees. Used for the legend swatch,
# which sits on white and so cannot borrow the map's own blending.
WATER_LEGEND_COLOR = "#7e9eb8"

# Fill for a district with no water pixel on the selected date, in the sidebar
# choropleth. Deliberately outside the turbidity ramp: any colour from the
# ramp would state a reading that was never taken.
DISTRICT_NODATA_COLOR = "#e6e5dd"
STATION_STROKE_COLOR = "#2b2b3a"
HEADER_NAVY = "#1e3a5f"  # "si krom tha" - the dark navy used for the floating title card

# Switchable basemaps, presented via the custom fold-out rail in
# map_controls.py (styled after air4laos.adpc.net), not Leaflet's default
# LayerControl widget.
BASEMAPS = {
    # Esri's Light Gray Canvas rather than CartoDB positron, which was the
    # default here before. Positron draws OpenStreetMap's own administrative
    # boundaries into its tiles as dashed pink lines - province and amphoe
    # both. Those are a different dataset from the shapefiles this app draws,
    # so every border appeared twice, in two places, a few hundred metres
    # apart. Switching the tiles off is not an option (they are baked into the
    # raster) and neither is light_nolabels, which drops only the text.
    #
    # This basemap has no administrative boundaries at all, so the only
    # borders on the map are the ones drawn from the local shapefiles.
    "Light": {
        "tiles": "https://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri, HERE, Garmin, OpenStreetMap contributors",
    },
    "Dark": {"tiles": "CartoDB dark_matter", "attr": None},
    "Classic": {"tiles": "OpenStreetMap", "attr": None},
    "Terrain": {
        "tiles": "https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
        "attr": "Map data: OpenStreetMap contributors, SRTM | Map style: OpenTopoMap (CC-BY-SA)",
    },
    "Satellite": {
        "tiles": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attr": "Esri World Imagery",
    },
}
DEFAULT_BASEMAP = "Light"

# Chart series colors - the first slots of the dataviz skill's validated
# categorical order (blue, orange, aqua), taken in that fixed order rather
# than picked by eye, so adjacent series stay separable under colour-vision
# deficiency.
COLOR_PREDICTED = "#2a78d6"
COLOR_ACTUAL = "#eb6834"
GAUGE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
# Days per plotted point on the streamflow chart. 5 over a 61-day range gives
# 12 points - enough to read a rise or fall, without the daily noise the raw
# gauge series carries.
BIN_DAYS = 5

P = dict(
    app_bg="#ffffff", sidebar_bg="#fafbfc", text="#2b2b3a", muted="#6b7684",
    border="#e7eaf0",
)

# ------------------------------------------------------------ translations --
TRANSLATIONS = {
    "en": {
        "page_title": "Mekong Water Quality - Thailand",
        "page_subtitle": "Satellite-derived turbidity monitoring - Ubon Ratchathani",
        "situation_overview": "Situation overview - Ubon Ratchathani",
        "latest_turbidity": "Latest Turbidity",
        "station_select": "Station",
        "no_coverage": "No composite coverage at this station's location.",
        "legend_layers": "Layers",
        "legend_turbidity_levels": "Turbidity Levels",
        "legend_province": "Province boundary",
        "legend_district": "District boundary",
        "legend_pcd_stations": "PCD stations",
        "legend_water": "Water",
        "legend_caption": "General reference scale for this dashboard, not an official Thai PCD standard.",
        "legend_label": "Legend",
        "pcd_stations_label": "PCD Stations",
        "province_label": "Province",
        "district_label": "District",
        "turbidity_label": "Turbidity",
        "basemap_label": "Base Map",
        "pcd_dept": "PCD - Thailand Pollution Control Department",
        "predicted_satellite": "Predicted (satellite)",
        "actual_pcd": "Actual (PCD)",
        "measured_pcd_avg": "Measured (PCD avg)",
        "class_label": "Class",
        "turbidity_trend": "Turbidity Trend",
        "province_average": "Province-wide average",
        "vs_previous": "vs previous week",
        "no_change": "no change",
        "streamflow_heading": "Streamflow / Discharge",
        "discharge": "Discharge",
        "water_level": "Water level",
        "streamflow_unavailable": "Streamflow gauge service unavailable right now.",
        "month_label": "Month",
        "month_all": "All",
        "period_mean": "5-day mean",
        "period_from": "5 days from",
        "month_nov": "Nov",
        "month_dec": "Dec",
        "level_m": "Level (m)",
        "gauge": "Gauge",
        "district_ranking": "District Ranking",
        "district_ranking_note": "Mean turbidity of water within each district",
        "no_districts": "District boundaries not available.",
        "no_water_pixels": "no water detected",
        "district_map_hint": "Hover or tap a district for its reading",
        "water_label": "Water",
        "water_source": "Wetland areas - Thai land-use survey",
        "risk": "Risk",
        "window_label": "01 Nov - 31 Dec 2024",
        # --- Information modal. The {…} fields are filled from the model
        # constants in turbidity_model.py rather than typed out here, so the
        # accuracy figures on screen cannot drift from the model that produced
        # them. ---
        # --- Station marker popup ---
        "popup_subdistrict": "Subdistrict",
        "popup_district": "District",
        "popup_province": "Province",
        "pixel_no_water": "No water detected in this pixel on this date.",
        "pixel_note": "Value under the point, {date}.",
        "popup_predicted": "Predicted Turbidity",
        "popup_measured": "Measured Turbidity",
        "popup_level": "Turbidity Level",
        "popup_note": (
            "Predicted value is for {date}. Measured value is this station's average "
            "across the PCD record, not a reading taken that day."
        ),
        "info_label": "Information",
        "info_data_source": "Data Source",
        "info_src_turbidity": (
            "<b>Turbidity (satellite):</b> Sentinel-2 Level-2A surface reflectance "
            "(COPERNICUS/S2_SR_HARMONIZED), composited from scenes with under 80% cloud "
            "cover and exported at 20 m resolution through "
            '<a href="https://earthengine.google.com" target="_blank" rel="noopener">Google Earth '
            "Engine</a>. Reflectance is converted to turbidity by a calibrated neural network "
            "(MLP) - R&sup2; {r2:.3f}, RMSE {rmse:.2f} NTU against {n} Ubon ground samples."
        ),
        "info_src_stations": (
            "<b>Ground stations:</b> Measured turbidity from the "
            '<a href="https://www.pcd.go.th" target="_blank" rel="noopener">Thailand Pollution '
            "Control Department</a> (PCD), used to calibrate and validate the model above and "
            "shown next to each satellite estimate. The place named in each station's popup is "
            'reverse-geocoded from <a href="https://www.openstreetmap.org" target="_blank" '
            'rel="noopener">OpenStreetMap</a>.'
        ),
        "info_src_streamflow": (
            "<b>Streamflow / water level:</b> Daily stage from the "
            '<a href="http://hydro-4.com/" target="_blank" rel="noopener">RID</a> Lower-NE '
            "gauges on the Mun River, downloaded for 01 Nov - 31 Dec 2024. The service "
            "publishes no discharge (m&sup3;/s) figure for these gauges - the field exists "
            "but is empty on every day in the range - so water level is shown, being the "
            "quantity actually measured. The faint line is the daily reading; the solid "
            "line is the average."
        ),
        "info_src_coverage": (
            "<b>Coverage:</b> Weekly composites across {window}. Heavy cloud on a given week "
            "can leave gaps in that date's water mask."
        ),
        "info_note": (
            "Satellite turbidity is a model estimate, not a measurement. Values are indicative "
            "and are not an official Thai PCD figure."
        ),
    },
    "th": {
        "page_title": "คุณภาพน้ำแม่น้ำโขง - ประเทศไทย",
        "page_subtitle": "ติดตามความขุ่นของน้ำด้วยดาวเทียม - จังหวัดอุบลราชธานี",
        "situation_overview": "ภาพรวมสถานการณ์ - จังหวัดอุบลราชธานี",
        "latest_turbidity": "ความขุ่นล่าสุด",
        "station_select": "สถานี",
        "no_coverage": "ไม่มีข้อมูลดาวเทียมครอบคลุมตำแหน่งสถานีนี้",
        "legend_layers": "ชั้นข้อมูล",
        "legend_turbidity_levels": "ระดับความขุ่น",
        "legend_province": "ขอบเขตจังหวัด",
        "legend_district": "ขอบเขตอำเภอ",
        "legend_pcd_stations": "สถานีคุณภาพน้ำ",
        "legend_water": "แหล่งน้ำ",
        "legend_caption": "ค่าอ้างอิงทั่วไปสำหรับแดชบอร์ดนี้ ไม่ใช่มาตรฐานทางการของกรมควบคุมมลพิษ",
        "legend_label": "คำอธิบาย",
        "pcd_stations_label": "สถานีคุณภาพน้ำ",
        "province_label": "จังหวัด",
        "district_label": "อำเภอ",
        "turbidity_label": "ความขุ่น",
        "basemap_label": "แผนที่ฐาน",
        "pcd_dept": "คพ. - กรมควบคุมมลพิษ",
        "predicted_satellite": "พยากรณ์ (ดาวเทียม)",
        "actual_pcd": "ค่าจริง (คพ.)",
        "measured_pcd_avg": "ค่าวัดจริง (เฉลี่ย คพ.)",
        "class_label": "ระดับ",
        "turbidity_trend": "แนวโน้มความขุ่น",
        "province_average": "ค่าเฉลี่ยทั้งจังหวัด",
        "vs_previous": "เทียบกับสัปดาห์ก่อน",
        "no_change": "ไม่เปลี่ยนแปลง",
        "streamflow_heading": "ปริมาณน้ำท่า / อัตราการไหล",
        "discharge": "อัตราการไหล",
        "water_level": "ระดับน้ำ",
        "streamflow_unavailable": "ไม่สามารถเชื่อมต่อระบบสถานีวัดน้ำได้ในขณะนี้",
        "month_label": "เดือน",
        "month_all": "ทั้งหมด",
        "period_mean": "ค่าเฉลี่ย 5 วัน",
        "period_from": "5 วันนับจาก",
        "month_nov": "พ.ย.",
        "month_dec": "ธ.ค.",
        "level_m": "ระดับน้ำ (ม.)",
        "gauge": "สถานีวัดน้ำ",
        "district_ranking": "อันดับความขุ่นรายอำเภอ",
        "district_ranking_note": "ค่าเฉลี่ยความขุ่นของน้ำในแต่ละอำเภอ",
        "no_districts": "ไม่มีข้อมูลขอบเขตอำเภอ",
        "no_water_pixels": "ไม่พบพื้นที่น้ำ",
        "district_map_hint": "ชี้หรือแตะที่อำเภอเพื่อดูค่า",
        "water_label": "แหล่งน้ำ",
        "water_source": "พื้นที่ชุ่มน้ำ - ข้อมูลการใช้ที่ดิน",
        "risk": "ความเสี่ยง",
        "window_label": "1 พ.ย. - 31 ธ.ค. 2567",
        "popup_subdistrict": "ตำบล",
        "popup_district": "อำเภอ",
        "popup_province": "จังหวัด",
        "pixel_no_water": "ไม่พบน้ำในพิกเซลนี้ในวันที่เลือก",
        "pixel_note": "ค่า ณ จุดที่กด · {date}",
        "popup_predicted": "ความขุ่นที่ประเมิน",
        "popup_measured": "ความขุ่นที่ตรวจวัด",
        "popup_level": "ระดับความขุ่น",
        "popup_note": (
            "ค่าประเมินเป็นของวันที่ {date} ส่วนค่าตรวจวัดเป็นค่าเฉลี่ยของสถานีนี้"
            "จากข้อมูลกรมควบคุมมลพิษ ไม่ใช่ค่าที่วัดในวันดังกล่าว"
        ),
        "info_label": "ข้อมูล",
        "info_data_source": "แหล่งที่มาของข้อมูล",
        "info_src_turbidity": (
            "<b>ความขุ่นจากดาวเทียม:</b> ค่าการสะท้อนแสงพื้นผิวจาก Sentinel-2 ระดับ 2A "
            "(COPERNICUS/S2_SR_HARMONIZED) รวมภาพจากฉากที่มีเมฆปกคลุมน้อยกว่า 80% "
            "และส่งออกที่ความละเอียด 20 เมตร ผ่าน "
            '<a href="https://earthengine.google.com" target="_blank" rel="noopener">Google Earth '
            "Engine</a> จากนั้นแปลงเป็นค่าความขุ่นด้วยแบบจำลองโครงข่ายประสาทเทียม (MLP) "
            "ที่ผ่านการปรับเทียบ - R&sup2; {r2:.3f}, RMSE {rmse:.2f} NTU "
            "เทียบกับตัวอย่างภาคพื้นดินในจังหวัดอุบลราชธานี {n} ตัวอย่าง"
        ),
        "info_src_stations": (
            "<b>สถานีภาคพื้นดิน:</b> ค่าความขุ่นที่ตรวจวัดจริงโดย"
            '<a href="https://www.pcd.go.th" target="_blank" rel="noopener">กรมควบคุมมลพิษ</a> (คพ.) '
            "ใช้สำหรับปรับเทียบและตรวจสอบความถูกต้องของแบบจำลองข้างต้น "
            "และแสดงคู่กับค่าที่ประเมินจากดาวเทียม ส่วนชื่อสถานที่ตั้งของแต่ละสถานีได้จากการค้นพิกัดย้อนกลับด้วย "
            '<a href="https://www.openstreetmap.org" target="_blank" rel="noopener">OpenStreetMap</a>'
        ),
        "info_src_streamflow": (
            "<b>ปริมาณน้ำ / ระดับน้ำ:</b> ระดับน้ำรายวันจากสถานีวัดน้ำแม่น้ำมูล "
            '<a href="http://hydro-4.com/" target="_blank" rel="noopener">กรมชลประทาน</a> '
            "(สำนักงานอุทกวิทยาภาคตะวันออกเฉียงเหนือตอนล่าง) ดึงข้อมูลช่วง 1 พ.ย. - 31 ธ.ค. 2567 "
            "ระบบไม่ได้ส่งค่าอัตราการไหล (ลบ.ม./วินาที) สำหรับสถานีเหล่านี้ "
            "โดยมีฟิลด์ข้อมูลแต่ว่างเปล่าทุกวันในช่วงนี้ จึงแสดงเป็นระดับน้ำซึ่งเป็นค่าที่วัดได้จริง "
            "เส้นจางคือค่ารายวัน เส้นทึบคือค่าเฉลี่ย"
        ),
        "info_src_coverage": (
            "<b>ช่วงข้อมูล:</b> ภาพรวมรายสัปดาห์ ระหว่าง {window} "
            "สัปดาห์ที่มีเมฆหนาอาจทำให้พื้นที่น้ำของวันนั้นขาดหายไปบางส่วน"
        ),
        "info_note": (
            "ค่าความขุ่นจากดาวเทียมเป็นค่าประเมินจากแบบจำลอง ไม่ใช่ค่าที่ตรวจวัดโดยตรง "
            "ใช้เป็นข้อมูลเบื้องต้นเท่านั้น และไม่ใช่ค่าทางการของกรมควบคุมมลพิษ"
        ),
    },
}

if "lang" not in st.session_state:
    st.session_state.lang = "en"
LANG = st.session_state.lang
T = TRANSLATIONS[LANG]

# Whichever font is listed first wins for the characters it covers, and the
# other only fills the gaps. Poppins has no Thai block, so on the English UI
# it renders the Latin text and Noto Sans Thai quietly handles any Thai that
# appears (place names, mostly). Putting Noto Sans Thai first for the Thai UI
# makes it set the whole interface - including the digits and Latin fragments
# like "NTU" and "M.7" - so the page reads as one typeface rather than two
# mixed mid-sentence.
FONT_STACK = ("'Noto Sans Thai', 'Poppins', sans-serif" if LANG == "th"
              else "'Poppins', 'Noto Sans Thai', sans-serif")


def scale_icon_data_uri():
    """A segmented turbidity-scale bar, as a base64 data URI for CSS.

    Used as the face of the sidebar open/close control instead of
    Streamlit's default chevron, so the control reads as "the water quality
    panel". Built from style.CLASSES rather than hard-coded swatches so it
    cannot drift out of sync with the legend. Base64 (not raw SVG in the
    url()) purely to sidestep escaping - the markup contains both '#' and
    quotes, which are awkward inside a CSS url() nested in an f-string.
    """
    seg = 18 / len(style.CLASSES)
    bars = "".join(
        f'<rect x="{3 + i * seg:.3f}" y="9" width="{seg:.3f}" height="6" fill="{c["color"]}"/>'
        for i, c in enumerate(style.CLASSES)
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
        '<defs><clipPath id="wqr"><rect x="3" y="9" width="18" height="6" rx="3"/></clipPath></defs>'
        f'<g clip-path="url(#wqr)">{bars}</g>'
        '<rect x="3" y="9" width="18" height="6" rx="3" fill="none" '
        'stroke="#2b2b3a" stroke-width="0.9" stroke-opacity="0.55"/>'
        "</svg>"
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


SIDEBAR_ICON = scale_icon_data_uri()


def arrow_icon_data_uri(direction: str, color: str = HEADER_NAVY):
    """Outline prev/next triangle as a base64 data URI for CSS.

    Drawn rather than typed: the obvious route is the U+25C1/U+25B7 glyphs
    as button text, but Poppins has no geometric-shapes block, so the
    browser silently falls back to a font whose triangle is tiny - raising
    font-size barely moves it. An SVG is the same size at any font stack.
    """
    path = "M16 4 L6 12 L16 20 Z" if direction == "prev" else "M8 4 L18 12 L8 20 Z"
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
        f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2" '
        'stroke-linejoin="round" stroke-linecap="round"/></svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


ARROW_PREV_ICON = arrow_icon_data_uri("prev")
ARROW_NEXT_ICON = arrow_icon_data_uri("next")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Noto+Sans+Thai:wght@400;500;600;700&display=swap');
    html, body, [class*="css"]  {{ font-family: {FONT_STACK}; }}
    /* Streamlit sets font-family directly on headings, widget labels and
       markdown blocks, which beats inheriting from body - so the Thai UI kept
       rendering those in Source Sans while everything else switched. This
       overrides them all in one go.
       The :not() matters: Streamlit draws its icons as ligatures in a
       Material Symbols font, and repainting that element with a text font
       turns each icon into its literal name ("keyboard_double_arrow_left").
       Excluding the icon span leaves those alone. Our own map icons are
       inline SVG and unaffected either way. */
    .stApp :not([data-testid="stIconMaterial"]) {{ font-family: {FONT_STACK} !important; }}
    #MainMenu, footer {{visibility: hidden;}}
    /* Header is collapsed to nothing rather than display:none, so it stops
       taking up space but Streamlit's "Running..." status widget can still
       show through. Fully hiding the header also hid that indicator, which
       left every click with no feedback at all during the ~1-2s rerun -
       the toggle button in particular just looked dead. */
    header[data-testid="stHeader"] {{ background: transparent; height: 0;
        pointer-events: none; }}
    header[data-testid="stHeader"] [data-testid="stToolbarActions"] {{ display: none; }}
    [data-testid="stStatusWidget"] {{ pointer-events: auto; }}
    [data-testid="stAppDeployButton"] {{display: none;}}
    /* relative: an anchor for .page-header below, so it floats at the
       map's own top-left corner instead of the raw viewport's (which would
       land it over the sidebar, since fixed/absolute positioning otherwise
       has no notion of "after the sidebar"). */
    /* height+overflow: Streamlit still reserves normal-flow gap space around
       the now-absolutely-positioned header/timeline elements (they collapse
       to 0 height but the ~16px inter-element gap around each remains),
       inflating this container past 100vh - which then throws off the
       bottom:14px math on .st-key-timeline_bar below. Pinning the real
       height and clipping the (empty, invisible) excess keeps that math
       anchored to what's actually visible. */
    .block-container {{ padding: 0; max-width: 100%; position: relative;
        height: 100vh; overflow: hidden; }}
    /* width:100% - streamlit_folium sizes the iframe from the full page
       width, ignoring the sidebar, so on desktop it was 1680px wide starting
       at x=300 and overflowed the viewport by exactly the sidebar's 300px.
       That rendered the map's right-hand 300px off-screen and took the
       bottom-right OSM credit with it. Constraining it to its actual
       container fixes both. */
    iframe[title="streamlit_folium.st_folium"] {{ height: calc(100vh - 4px) !important;
        width: 100% !important; min-height: 380px; display: block; }}

    html, body, .stApp {{ background: {P['app_bg']}; }}
    [data-testid="stSidebar"] {{ background: {P['sidebar_bg']}; }}
    .stApp, .stApp p, .stApp span, .stApp label, .stMarkdown {{ color: {P['text']}; }}

    /* Title floats on top of the map (air4laos-style), not in its own bar
       pushing the map down - full-bleed map, text stamped on top of it.
       left+right (not just left): stretches the bar full-width; the flex
       row inside is then centered within that full width. */
    .page-header {{ position:absolute; top:20px; left:14px; right:14px; z-index:999;
        background:{HEADER_NAVY}; border-radius:10px; box-shadow:0 2px 14px rgba(0,0,0,0.28);
        padding:14px 18px; display:flex; align-items:baseline; justify-content:center; gap:10px; }}
    /* !important: .stApp span (above) targets every span including these
       two and otherwise wins on specificity (class+type beats a bare class). */
    .page-title {{ font-size: 1.15rem; font-weight: 700; color: #ffffff !important; }}
    .page-subtitle {{ font-size: 0.85rem; color: #b7c4d4 !important; }}

    .legend-swatch {{ width:12px; height:12px; border-radius:3px; display:inline-block; flex-shrink:0; }}

    /* Sidebar top. Streamlit reserves a 60px header (a 32px logo spacer plus
       room for the collapse control) and then pads the content by another
       16px, putting the first line 76px down an otherwise empty column.
       Nothing lives in that header here: the collapse button is position:fixed
       (see stSidebarCollapseButton below), so it has already left the flow and
       collapsing the header cannot strand it. */
    [data-testid="stSidebarHeader"] {{ height:0 !important; min-height:0 !important;
        padding:0 !important; margin:0 !important; }}
    [data-testid="stLogoSpacer"] {{ display:none !important; }}
    /* Streamlit pads the sidebar's foot by 96px, which read as a large empty
       gap under the last section. 18px still clears the bottom edge. */
    [data-testid="stSidebarUserContent"] {{ padding-top:14px !important;
        padding-bottom:18px !important; }}

    /* The sidebar's own title. A caption before, which set it in muted 0.75rem
       and made the panel look like it began with a footnote. */
    .sb-heading {{ font-size:1.02rem; font-weight:700; color:{P['text']};
        line-height:1.3; margin:0 0 10px 0; padding-bottom:8px;
        border-bottom:1px solid {P['border']}; }}

    .sb-metric {{ border:1px solid {P['border']}; border-radius:12px; padding:10px 12px; margin-bottom:10px; }}
    .sb-value {{ font-size:1.5rem; font-weight:700; color:{P['text']}; }}
    .sb-label {{ font-size:0.72rem; color:{P['muted']}; text-transform:uppercase; letter-spacing:.04em; }}
    /* Delta chip next to the hero number - direction is carried by an arrow
       glyph and the text itself, never by color alone. */
    .sb-delta {{ font-size:0.78rem; font-weight:600; margin-left:6px; }}
    .sb-sub {{ font-size:0.72rem; color:{P['muted']}; margin-top:2px; }}

    /* Streamflow gauge rows. */
    .sf-row {{ display:flex; justify-content:space-between; align-items:baseline;
        padding:6px 0 0 0; font-size:0.82rem; }}
    .sf-name {{ font-size:0.68rem; color:{P['muted']}; padding-bottom:6px; line-height:1.3; }}
    .sf-value {{ font-weight:700; }}

    /* Floating translucent bar along the bottom, same treatment as the
       header - anchored with clearance on the right so it doesn't sit
       under the layer rail (which lives inside the map iframe, fixed to
       its own corner, so it isn't reachable from out here). */
    /* width:auto - Streamlit puts width:100% on every container, and against
       an absolutely-positioned box that wins over `right`, so the bar was
       laid out as (left:14px + full container width) and overhung the right
       edge by 14px instead of matching .page-header's inset. Only with the
       width released do left+right both take effect. */
    /* padding-top 20 vs bottom 6: not a typo. The selected-date readout is
       absolutely positioned ~8px ABOVE the slider's own box, so it eats 8px
       of the top padding before anything is visible; the tick labels below
       sit 6px inside their row. 20-8 = 12 above, 6+6 = 12 below - equal
       breathing room, which is what the eye actually measures. */
    .st-key-timeline_bar {{ position:absolute; left:14px; right:14px; width:auto !important;
        bottom:14px; z-index:998;
        background:rgba(255,255,255,0.5); backdrop-filter: blur(3px); border-radius:12px;
        box-shadow:0 2px 14px rgba(0,0,0,0.12); padding:20px 14px 6px 14px; }}
    div[data-testid="stSlider"] {{ padding-top: 0; }}
    div[data-testid="stSlider"] > div > div > div:first-of-type {{ opacity: 0.85; }}
    /* Streamlit's own min/max end labels ("01 Nov 2024" / "27 Dec 2024"
       under each end of the track). Hidden because .wq-tick-row below
       already labels every composite date - keeping both meant the two end
       dates were printed twice, in two different formats, on two rows. */
    div[data-testid="stSliderTickBar"] {{ display: none !important; }}
    /* One label per composite date, evenly spaced under the slider (few
       enough dates now - see RANGE_START/RANGE_END - that labelling every
       point is readable instead of an unreadable comb). Now nested inside
       the slider's own column (see the timeline_bar block) so its width
       matches the slider instead of the whole row; the 6px side padding
       matches the track's own inset from the slider widget's outer edge
       (measured empirically) so labels line up with the actual tick
       positions instead of the widget's outer bounding box. */
    .wq-tick-row {{ display:flex; justify-content:space-between; padding:0 6px; margin-top:-34px; }}
    .wq-tick-label {{ font-size:0.78rem; color:{P['muted']}; }}
    .wq-tick-label.wq-tick-current {{ font-weight:700; color:{HEADER_NAVY}; }}
    /* Prev/next/lang as bare buttons - no button box/border. */
    div[data-testid="stButton"] button {{ border: none; background: none; box-shadow: none;
        padding: 2px 6px; font-size: 1.5rem; color: {P['text']}; }}
    div[data-testid="stButton"] button:hover {{ background: rgba(255,255,255,0.6); border-radius: 6px; }}
    div[data-testid="stButton"] button:disabled {{ opacity: 0.3; }}

    /* Calendar badge at the left end of the timeline bar - marks the bar as
       a date control and names the year the ticks belong to (the tick
       labels themselves are day+month only, so the year is otherwise only
       visible in the selected-date readout above the slider). */
    /* Icon and year side by side, not stacked. Stacked, the badge as a whole
       centred correctly but the year - being its second line - sat 16px below
       the centre line everything else in the bar shares (arrows 37, slider
       40, language 39), so it read as misaligned against the EN/TH pill. In a
       row both halves sit on that one line. */
    /* The badge's box, sized to the language toggle's 38px so the two centre
       lines coincide. It lives on this wrapper rather than on .wq-cal itself
       because Streamlit's markdown wrappers in between collapse to 16px -
       a min-height on the badge overflowed them downward instead of centring
       inside them, which is what left the year sitting low. */
    /* The column this sits in is already aligned with the language toggle
       (both centre on the same line). What was off is INSIDE it: Streamlit
       nests four wrappers around a markdown block and they collapse to a
       couple of pixels, so the badge hung below whatever those wrappers
       centred. Rather than centring a collapsed box, the whole chain is given
       the badge's height and told to centre its contents. */
    .st-key-cal_badge {{ min-height:38px; }}
    .wq-cal {{ display:flex; flex-direction:row; align-items:center; justify-content:center;
        gap:5px; color:{HEADER_NAVY}; min-height:38px; }}
    .wq-cal svg {{ width:22px; height:22px; flex:0 0 auto; }}
    /* Same size and weight as the EN/TH labels it sits opposite - they are
       the two ends of one bar and a smaller, lighter year read as an error
       rather than a hierarchy. */
    .wq-cal-year {{ font-size:0.74rem; font-weight:700; color:{P['muted']};
        letter-spacing:.02em; line-height:1; white-space:nowrap; }}

    /* Prev/next pair: outline triangles, tight together, no button boxes. */
    .st-key-date_nav {{ display:flex; align-items:center; }}
    .st-key-date_nav div[data-testid="stHorizontalBlock"] {{ gap:0 !important; }}
    /* 1.9rem, not the 1.5rem the generic button rule above already sets -
       anything at or below that leaves the arrows looking unchanged. The
       explicit height matters as much as the font size: zeroing padding and
       min-height collapsed the button box to 14px, which clipped a 30px
       glyph down to a sliver, so the arrows looked unchanged however large
       the font was set. */
    /* Compact button box, large drawn triangle. The glyph that used to be
       the label is hidden (font-size:0) and the arrow comes from a
       background SVG instead - see arrow_icon_data_uri() for why. */
    .st-key-date_nav button {{ padding:0 !important;
        height:30px !important; min-height:30px !important; width:30px !important;
        background-repeat:no-repeat !important; background-position:center !important;
        background-size:24px 24px !important; }}
    .st-key-date_nav button div, .st-key-date_nav button p {{ font-size:0 !important; }}
    .st-key-nav_prev button {{ background-image:url("{ARROW_PREV_ICON}") !important; }}
    .st-key-nav_next button {{ background-image:url("{ARROW_NEXT_ICON}") !important; }}
    .st-key-date_nav button:hover:not(:disabled) {{ background-color:transparent !important; }}
    .st-key-date_nav button:disabled {{ opacity:0.28 !important; }}

    /* Language pill: both codes always visible, active one filled. The
       active side is the *disabled* button (you cannot switch to the
       language you are already in), so it is styled through :disabled -
       and the default 0.3 dimming for disabled buttons is overridden here,
       since here "disabled" means "current", not "unavailable". */
    /* Styled as one switch, not two buttons: a navy pill with a white knob
       that sits under whichever language is active. The knob is the
       *disabled* button (you cannot select the language already in use), so
       the active look is applied through :disabled - hence the opacity
       override, since here disabled means "current", not "unavailable". */
    .st-key-lang_toggle {{ background:{HEADER_NAVY}; border-radius:999px; padding:4px;
        width:92px; margin-left:auto; box-shadow:0 2px 8px rgba(0,0,0,0.18); }}
    .st-key-lang_toggle div[data-testid="stHorizontalBlock"] {{ gap:0 !important; }}
    .st-key-lang_toggle button {{ font-size:0.74rem !important; font-weight:700 !important;
        border-radius:999px !important; height:30px !important; min-height:30px !important;
        padding:0 !important; background:transparent !important;
        border:none !important; box-shadow:none !important; }}
    /* The inner <p>/<div> needs the colour too: the global `.stApp p` rule
       further up otherwise repaints the label dark, so the inactive side
       came out near-black on navy instead of white. */
    .st-key-lang_toggle button, .st-key-lang_toggle button p,
    .st-key-lang_toggle button div {{ color:#ffffff !important; }}
    .st-key-lang_toggle button:disabled {{ opacity:1 !important; background:#ffffff !important; }}
    .st-key-lang_toggle button:disabled, .st-key-lang_toggle button:disabled p,
    .st-key-lang_toggle button:disabled div {{ color:{HEADER_NAVY} !important; }}
    .st-key-lang_toggle button:hover:not(:disabled) {{
        background:rgba(255,255,255,0.16) !important; }}

    /* Sidebar open/close control, both states: Streamlit's chevron replaced
       by the turbidity-scale icon on a white disc. Two different elements
       are involved - stSidebarCollapseButton lives in the sidebar header
       while it is open, stExpandSidebarButton appears at the top-left of
       the page once it is closed - so both are styled identically and the
       control looks like the same button in either state.
       opacity/visibility are forced because Streamlit fades the collapse
       button in only on hover, which made it easy to miss. */
    /* Each of these two testids is a wrapper in one Streamlit build and the
       <button> itself in another, so both shapes are matched. */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"] {{ opacity: 1 !important; visibility: visible !important; }}
    /* Moved out of the sidebar header / page corner to sit directly under
       the map's zoom control, in the bottom-left stack: zoom ends at 828px
       down a 1000px viewport and the timeline bar starts at 916px, so
       bottom:114px centres a 38px control in the gap between them.
       The two states need different `left` values because only one exists
       at a time and the map's left edge moves with the sidebar: with the
       sidebar open the zoom column sits at x=326-356, with it closed at
       x=26-56. Each value centres the control under the zoom column for
       the state it belongs to, so it looks like one button that stays put
       relative to the zoom buttons. */
    [data-testid="stSidebarCollapseButton"] {{
        position: fixed !important; left: 315px !important; bottom: 114px !important;
        width: 40px !important; height: 40px !important; z-index: 1002 !important; }}
    [data-testid="stExpandSidebarButton"] {{
        position: fixed !important; left: 25px !important; bottom: 114px !important;
        z-index: 1002 !important; }}
    /* Once collapsed, the collapse button still exists and - being
       position:fixed - escapes the zero-width sidebar, landing right on top
       of the expand button that has just replaced it. Hide it in that state
       so only one control is ever on screen. */
    [data-testid="stSidebar"][aria-expanded="false"] [data-testid="stSidebarCollapseButton"] {{
        display: none !important; }}
    [data-testid="stSidebarCollapseButton"] button,
    button[data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"] button,
    /* 40px, matching the map's Information button directly above it - the two
       sit in the same bottom-left column, so a size difference between them
       read as a mistake rather than a hierarchy. Both grew with the layer
       rail, and the zoom pair below is 38px, so the whole column is 38-40px
       now instead of the old 30-34px. */
    button[data-testid="stExpandSidebarButton"] {{
        width: 40px !important; height: 40px !important; border-radius: 50% !important;
        background: #ffffff url("{SIDEBAR_ICON}") center / 23px 23px no-repeat !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.22) !important; border: none !important;
        padding: 0 !important; }}
    [data-testid="stSidebarCollapseButton"] button:hover,
    button[data-testid="stSidebarCollapseButton"]:hover,
    [data-testid="stExpandSidebarButton"] button:hover,
    button[data-testid="stExpandSidebarButton"]:hover {{ filter: brightness(1.04); }}
    /* The chevron itself. It is NOT an <svg> - Streamlit renders it as a
       Material Symbols ligature in <span data-testid="stIconMaterial">
       (text content literally "keyboard_double_arrow_left"), so hiding svg
       does nothing and the glyph sits on top of the icon. Target that span
       specifically: hiding all children instead also hid the <button> in
       the open state, where the testid is on a wrapper around it, and
       collapsed the control to 0x0. The button keeps its aria-label, so
       nothing accessible is lost. */
    [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"],
    [data-testid="stExpandSidebarButton"] [data-testid="stIconMaterial"] {{
        display: none !important; }}

    /* Stand this button down while the map's Information modal is open. The
       modal is drawn inside the map iframe, so its backdrop cannot cover
       page-level chrome like this one: on a phone the button lands on top of
       the card and stays clickable through what looks like a backdrop. The
       class is set on <body> by map_controls' modal (see the wq-modal-open
       comment there) precisely so the page keeps ownership of this decision.
       Visibility rather than display: display:none on a focused element moves
       focus to the top of the document. */
    body.wq-modal-open [data-testid="stExpandSidebarButton"],
    body.wq-modal-open [data-testid="stSidebarCollapseButton"] {{
        visibility: hidden !important; }}

    /* Ranked district rows. */
    .rank-row {{ display:flex; align-items:center; gap:8px; padding:5px 0 0 0; font-size:0.8rem; }}
    .rank-num {{ width:18px; color:{P['muted']}; font-size:0.72rem; flex-shrink:0; }}
    .rank-name {{ flex:1; }}
    .rank-ntu {{ font-weight:700; }}
    .rank-risk {{ display:inline-block; padding:1px 8px; border-radius:999px;
        font-size:0.68rem; font-weight:600; color:#2b2b3a; }}

    /* The district choropleth's own styling lives inside its iframe (see
       _CHOROPLETH_TEMPLATE) - a stylesheet only applies to its own document,
       so rules written here would never reach it. */

    /* Streamlit Cloud's badge and owner avatar are NOT styled here. They live
       in the host page above this one - Cloud serves the app in a nested
       iframe - and a stylesheet only applies to its own document, so rules
       written here never reached them (confirmed by transform:none on the
       deployed page). They are repositioned by script at the end of this
       file, which can cross that boundary because the two frames are
       same-origin. */

    /* ---------------------------------------------------------- phones ---
       Everything above is sized for a desktop viewport. On a ~390px screen
       the same layout collapses: the title wraps to two lines, and the
       timeline bar grows to 290px - a third of the screen - because
       Streamlit stamps min-width:calc(100% - 24px) on every column below its
       own breakpoint, so all four timeline columns stack into their own
       rows. The bar then covers the map's zoom control and the sidebar
       button, which sit at fixed offsets from the bottom.
       640px matches where Streamlit's own column stacking kicks in. */
    @media (max-width: 640px) {{
      /* dvh, not vh: mobile browsers count the collapsing URL bar inside
         100vh, so a vh-sized map is taller than the visible area and the
         bottom-anchored timeline sits off-screen until you scroll. Declared
         after the vh rules so browsers without dvh keep the old value. */
      .block-container {{ height: 100dvh; }}
      iframe[title="streamlit_folium.st_folium"] {{ height: calc(100dvh - 4px) !important; }}

      /* Title and subtitle stack instead of sharing a baseline - side by
         side they wrapped mid-phrase ("Mekong Water / Quality - Thailand"). */
      .page-header {{ top:8px; left:8px; right:8px; padding:10px 13px;
          flex-direction:column; align-items:flex-start; justify-content:flex-start; gap:2px; }}
      .page-title {{ font-size:0.98rem; line-height:1.2; }}
      .page-subtitle {{ font-size:0.72rem; line-height:1.25; }}

      /* This was 54px, holding the bar clear of the Streamlit Cloud badge and
         owner avatar - they live in the *host* page on top of this app's
         iframe, occupied the bottom 46px of the phone layout, and were
         swallowing taps meant for the language switch. Those two are now
         removed outright from the host document (see the components.html
         block at the end of this file), so the strip they occupied is ours
         again and the bar sits at the same 14px margin as on desktop. Every
         other control in the bottom-left stack below moved down with it. */
      /* position:fixed, not the desktop rule's absolute: absolute measures
         `bottom` from the block container, whose bottom edge sits ~28px above
         the viewport's, so the bar floated higher than the number asked for.
         Fixed measures from the viewport, which is the edge being aimed at. */
      .st-key-timeline_bar {{ position:fixed; left:8px; right:8px; bottom:14px;
          padding:16px 10px 5px 10px; }}
      /* Undo that forced min-width so the columns can share rows again, then
         reorder into two: the slider alone on top (it needs the full width
         to be draggable), and calendar / arrows / language beneath it. */
      .st-key-timeline_bar div[data-testid="stHorizontalBlock"] {{
          flex-wrap:wrap !important; align-items:center !important;
          gap:0 !important; row-gap:2px !important; }}
      .st-key-timeline_bar div[data-testid="stColumn"] {{
          min-width:0 !important; flex:0 0 auto !important; width:auto !important;
          margin-right:0 !important; }}
      /* margin-right above is not redundant with the gap rules: alongside
         that min-width, Streamlit gives each column a ~34px right margin at
         this size. Inside the two nested column pairs that meant the prev/
         next arrows needed 102px of a 68px column and the EN/TH pair 114px
         of 76px, so both wrapped - which is why the language switch showed
         as EN stacked above TH rather than as a single pill. */
      .st-key-date_nav div[data-testid="stHorizontalBlock"],
      .st-key-lang_toggle div[data-testid="stHorizontalBlock"] {{
          flex-wrap:nowrap !important; }}
      /* Written with the child combinator, and kept below the :nth-child
         rules further down, because those match on position alone - and the
         arrow pair and the EN/TH pair are *themselves* columns nested inside
         columns 3 and 4. Without this, prev/next and EN/TH each inherited
         the outer row's order:2 / order:1 and rendered swapped, and the
         nth-child(1) auto-margin stretched TH to 70px of a 76px pill so EN
         overflowed outside it. The extra attribute selector also lifts
         specificity above those rules, which a flat selector could not. */
      .st-key-date_nav div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
      .st-key-lang_toggle div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
          order:0 !important; flex:1 1 50% !important; width:auto !important;
          margin:0 !important; min-width:0 !important; }}
      .st-key-timeline_bar div[data-testid="stColumn"]:nth-child(2) {{
          order:1; flex:0 0 100% !important; width:100% !important; }}
      /* auto, and !important, because the blanket margin-right:0 above would
         otherwise win and leave the calendar, arrows and language switch all
         bunched against the left edge. This is the spacer that pushes the
         arrows and language switch over to the right. */
      .st-key-timeline_bar div[data-testid="stColumn"]:nth-child(1) {{
          order:2; margin-right:auto !important; }}
      .st-key-timeline_bar div[data-testid="stColumn"]:nth-child(3) {{ order:3; }}
      .st-key-timeline_bar div[data-testid="stColumn"]:nth-child(4) {{ order:4; }}

      /* Nine dates across ~330px render as one unbroken string
         ("01 Nov08 Nov15 Nov..."). Every other label is dropped - the ticks
         are evenly spaced, so the survivors still line up with the track,
         and the exact selected date is spelled out above the thumb anyway. */
      .wq-tick-label:nth-child(even) {{ display:none; }}
      .wq-tick-label {{ font-size:0.62rem; }}
      .wq-tick-row {{ margin-top:-30px; padding:0 4px; }}

      /* 32px here: the phone rules shrink the toggle to a 26px button in 3px
         padding, so the box it has to line up with is smaller too, and the
         year follows the toggle's smaller label size. */
      .st-key-cal_badge {{ min-height:32px; }}
      .wq-cal {{ min-height:32px; }}
      .wq-cal {{ gap:4px; }}
      .wq-cal svg {{ width:18px; height:18px; }}
      .wq-cal-year {{ font-size:0.66rem; }}
      .st-key-date_nav button {{ height:34px !important; min-height:34px !important;
          width:34px !important; background-size:23px 23px !important; }}
      .st-key-lang_toggle {{ width:76px; padding:3px; }}
      .st-key-lang_toggle button {{ height:26px !important; min-height:26px !important;
          font-size:0.66rem !important; }}

      /* Both sidebar states share one position on a phone: the sidebar
         overlays the map rather than shrinking it, so there is no second
         layout to offset against. Raised to clear the two-row timeline. */
      /* Two positions, not one. Closed, the button is the only thing on
         screen and belongs in the map's bottom-left stack. Open, the sidebar
         overlays the map from x=0 to x=300, so a button at left:12 sits on
         top of the sidebar's own chart; 308 puts it just past the sidebar's
         edge, the same relationship the desktop layout already uses. */
      [data-testid="stExpandSidebarButton"] {{ left:17px !important; bottom:146px !important;
          width:36px !important; height:36px !important; }}
      [data-testid="stSidebarCollapseButton"] {{ left:306px !important; bottom:146px !important;
          width:36px !important; height:36px !important; }}
      /* 36px here, not the desktop 40: this matches both the Information
         button and Leaflet's own zoom buttons at this breakpoint, so the
         three controls in the column are one size. */
      button[data-testid="stExpandSidebarButton"],
      [data-testid="stExpandSidebarButton"] button,
      button[data-testid="stSidebarCollapseButton"],
      [data-testid="stSidebarCollapseButton"] button {{
          width:36px !important; height:36px !important; background-size:21px 21px !important; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- data ----
@st.cache_data(show_spinner="Loading satellite composite...")
def load_province_composite(path: str):
    """(turbidity, mask, bounds) for one composite date.

    Hits the precomputed display raster shipped in the repo, so picking a
    date costs a ~20ms file read rather than a Drive download plus ~11s of
    full-resolution inference. See province_composite.load_display().
    """
    return pc.load_display(path)


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


def mercator_row_map(bounds, h):
    """Source row for each row of a Web Mercator image of `h` rows.

    folium's ImageOverlay stretches the image linearly between two lat/lon
    corners, but Leaflet draws in Web Mercator, whose y axis is
    ln(tan(pi/4 + lat/2)) - not linear in latitude. The composites are
    EPSG:4326, so their rows are evenly spaced in latitude, and handing them
    straight to ImageOverlay puts every row except the two edges too far
    north. Over this province's 1.89 degrees of latitude that peaks at 233m
    near lat 15.15 - wider than the Mun River, so the overlay slid clean off
    the water it was meant to sit on.

    Longitude needs no equivalent fix: Mercator x IS linear in longitude. So
    this is a row gather, not a warp.

    Nearest row, never interpolated. The overlay is categorical - class
    colours plus a hard alpha edge - and blending rows would both invent
    intermediate classes and fray the water's edge into half-transparent
    pixels.
    """
    south, north = bounds.bottom, bounds.top
    y_south = np.log(np.tan(np.pi / 4 + np.radians(south) / 2))
    y_north = np.log(np.tan(np.pi / 4 + np.radians(north) / 2))
    # Output row i is at an evenly spaced Mercator y; find the latitude that
    # falls at, then the equal-latitude source row holding it.
    y = y_north + (np.arange(h) + 0.5) / h * (y_south - y_north)
    lat = np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)
    src = ((north - lat) / (north - south) * h).astype(np.intp)
    return np.clip(src, 0, h - 1, out=src)


def turbidity_overlay_rgba(turbidity_map, water_mask, bounds, band_rows=512):
    """The map's turbidity image, as uint8 RGBA, on a Web Mercator grid.

    Two things here exist only to keep the peak allocation bounded, because
    this runs on every rerun and the raster is the whole province.

    bytes=True: matplotlib's colormaps return float64, 32 bytes per pixel. At
    the display grid that was a 587MB array, and at finer grids it simply
    fails to allocate. The image is quantised to 8 bits per channel on its way
    to PNG regardless, so building the float version discards nothing - it
    just costs eight times the memory to reach the same picture.

    Banding: even in uint8 the intermediate BoundaryNorm index is int64, twice
    the size of the output image. Colouring in horizontal bands keeps that
    intermediate to one band's worth, so the peak is the output array plus a
    slice rather than the whole grid several times over.
    """
    breakpoints = [c["max"] for c in style.CLASSES[:-1]]
    colors = [c["color"] for c in style.CLASSES]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm([0] + breakpoints + [breakpoints[-1] * 3], cmap.N)

    h, w = turbidity_map.shape
    rows = mercator_row_map(bounds, h)
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    for y0 in range(0, h, band_rows):
        y1 = min(y0 + band_rows, h)
        # Gather the source rows this output band needs, so the reprojection
        # rides along inside the existing band loop and never materialises a
        # second full-size image.
        take = rows[y0:y1]
        band = cmap(norm(turbidity_map[take]), bytes=True)
        # 255 on water, 0 everywhere else. Fully opaque where there IS a
        # reading, fully transparent where there is none - the alpha channel
        # is what shapes the overlay to the water, so the 0 side has to stay
        # 0 or the whole province rectangle would be painted.
        band[..., 3] = np.where(water_mask[take], 255, 0)
        rgba[y0:y1] = band
    return rgba


def build_legend_html():
    layer_rows = (
        f'<div class="wq-legend-item"><span class="wq-legend-line" style="background:{PROVINCE_LINE_COLOR}"></span>{T["legend_province"]}</div>'
        f'<div class="wq-legend-item"><span class="wq-legend-line" style="background:{DISTRICT_LINE_COLOR};height:2px"></span>{T["legend_district"]}</div>'
        f'<div class="wq-legend-item"><span class="wq-legend-circle" style="border:2px solid {STATION_STROKE_COLOR}"></span>{T["legend_pcd_stations"]}</div>'
        # WATER_LEGEND_COLOR, not WATER_FILL_COLOR: the layer is drawn at 62%
        # over a pale basemap, so the raw fill would put a swatch here noticeably
        # darker than anything on the map. This is what that fill composites to.
        f'<div class="wq-legend-item"><span class="wq-legend-swatch" style="background:{WATER_LEGEND_COLOR}"></span>{T["legend_water"]}</div>'
    )
    turbidity_rows = []
    prev_max = 0
    for c in style.CLASSES:
        range_label = f"&gt;{prev_max} NTU" if c["max"] == float("inf") else f"{prev_max}-{c['max']:.0f} NTU"
        turbidity_rows.append(
            f'<div class="wq-legend-item"><span class="wq-legend-swatch" style="background:{c["color"]}"></span>'
            f'{c["label"]} <span class="wq-legend-range">({range_label})</span></div>'
        )
        prev_max = c["max"]
    turbidity_rows = "".join(turbidity_rows)
    return (
        f'<div class="wq-legend-heading">{T["legend_layers"]}</div>{layer_rows}'
        f'<div class="wq-legend-heading">{T["legend_turbidity_levels"]}</div>{turbidity_rows}'
        f'<div class="wq-legend-caption">{T["legend_caption"]}</div>'
    )


def build_info_html():
    """Body of the Information modal - where every number on this map comes
    from, in the current language.

    The model's accuracy figures are pulled from turbidity_model rather than
    written into the translation strings, so retraining updates what the
    dashboard claims about itself instead of leaving a stale number on screen.
    """
    rows = [
        T["info_src_turbidity"].format(
            r2=tm.VALIDATION_R2, rmse=tm.VALIDATION_RMSE, n=tm.VALIDATION_N),
        T["info_src_stations"],
        T["info_src_streamflow"],
        T["info_src_coverage"].format(window=T["window_label"]),
    ]
    return (
        f'<div class="wq-info-section">{T["info_data_source"]}</div>'
        '<div class="wq-info-box">'
        + "".join(f'<p class="wq-info-row">{r}</p>' for r in rows)
        + '</div>'
        f'<div class="wq-info-note">{T["info_note"]}</div>'
    )


df_val, y, y_pred, r2, rmse = load_validation()
station_summary = (
    df_val.groupby("Code", as_index=False)
    .agg(Name=("Name", "first"), station_la=("station_la", "first"), station_lo=("station_lo", "first"),
         Turbidity_Actual=("Turbidity_", "mean"), Predicted_NTU=("Predicted_NTU", "mean"))
    .sort_values("Predicted_NTU", ascending=False)
)

available = [
    (d, p) for d, p in pc.list_available_composites(".")
    if RANGE_START <= d <= RANGE_END
]
if not available:
    st.error(
        f"No province composites found between {RANGE_START:%d %b %Y} and {RANGE_END:%d %b %Y} "
        "(expected Ubon_S2_YYYYMMDD.tif files). Run refresh_ubon_data.py or backfill_ubon_weekly.py first."
    )
    st.stop()

dates = [d for d, _ in available]
# Widget renders below the map (as a timeline bar), but its value is needed
# above to pick which composite to load - initialize the session_state key
# first and read from there; st.select_slider(key=...) further down both
# displays and updates that same state.
if "picked_date" not in st.session_state or st.session_state.picked_date not in dates:
    st.session_state.picked_date = dates[-1]
picked_date = st.session_state.picked_date
picked_path = dict(available)[picked_date]

turbidity_map, valid_mask, bounds = load_province_composite(picked_path)


@st.cache_data(show_spinner=False)
def load_history_cache():
    """Precomputed per-composite aggregates from precompute_history.py, or an
    empty stub if that file was never generated.

    Both series below are defined over *every* composite in range, so computing
    them live means opening all of them - ~12s of inference each locally, and
    on top of that a ~28MB Drive download each in the cloud, where there's no
    persistent disk. That's minutes of blank screen per cold visit. The values
    are one float per date, so they ship precomputed instead (see
    precompute_history.py for how to regenerate).
    """
    try:
        with open(HISTORY_CACHE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"province": {}, "stations": {}}


@st.cache_data(show_spinner="Loading station history...")
def station_history(lat, lon):
    """Predicted turbidity at (lat, lon), sampled from every available
    composite - the "day by day / week by week" series for one station.
    """
    cached = load_history_cache()["stations"].get(f"{lat:.5f},{lon:.5f}", {})
    rows = []
    for d, path in available:
        # Cache miss = a composite added since the last precompute run. Fall
        # back to loading it so a freshly refreshed week is never silently
        # missing from the trend; it just costs what it used to.
        if d.isoformat() in cached:
            rows.append({"Date": pd.Timestamp(d), "NTU": cached[d.isoformat()]})
            continue
        turb_map, mask, b = load_province_composite(path)
        val = pc.sample_at(turb_map, mask, b, lat, lon)
        if val is not None:
            rows.append({"Date": pd.Timestamp(d), "NTU": val})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Loading province trend...")
def province_history():
    """Province-wide mean turbidity per composite date - the situation-overview
    series. Unlike station_history() (one sampled point per station), this
    averages every valid water pixel in the province, so it answers "how is
    the province as a whole trending" rather than "how is this one station".
    """
    cached = load_history_cache()["province"]
    rows = []
    for d, path in available:
        if d.isoformat() in cached:  # see station_history() on the fallback
            rows.append({"Date": pd.Timestamp(d), "NTU": cached[d.isoformat()]})
            continue
        turb_map, mask, _bounds = load_province_composite(path)
        if mask.any():
            rows.append({"Date": pd.Timestamp(d), "NTU": float(turb_map[mask].mean())})
    return pd.DataFrame(rows)


def load_level_history(start, end, lead_days=0):
    """Daily Mun River stage over [start, end] as a tidy DataFrame.

    `lead_days` extends the fetch *earlier* than `start` without widening what
    a caller then displays. A rolling average needs that many days already
    behind the first plotted point, otherwise the left-hand end of every
    smoothed line is either blank or computed from a shrinking window and
    slopes for no physical reason. One API call covers six months back
    regardless, so the lead-in is free.

    Reads the committed snapshot first and only calls the live service if
    that is missing. The window here is fixed history, so the snapshot is not
    a staleness risk - and the service is not reachable from every network
    (it appears to refuse foreign IPs, which is where a cloud host sits), so
    a live call was leaving the deployed chart permanently empty while the
    same code worked locally. See rid_streamflow.save_snapshot().

    Wrapped so a network failure degrades to an empty frame ("no data")
    rather than taking the page down.
    """
    fetch_from = start - dt.timedelta(days=lead_days)
    history = rid.load_snapshot()
    if history:
        history = {
            code: [(d, v) for d, v in series if fetch_from <= d <= end]
            for code, series in history.items()
        }
        history = {c: s for c, s in history.items() if s}
    if not history:
        try:
            history = rid.level_history_between(fetch_from, end)
        except Exception:
            return pd.DataFrame(columns=["Date", "Gauge", "Level"])
    rows = [
        {"Date": pd.Timestamp(d), "Gauge": code, "Level": level}
        for code, series in history.items()
        for d, level in series
    ]
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def district_ntu(path: str):
    """Mean turbidity per Ubon district for one composite, highest first.

    Zonal statistics rather than per-station values: this answers "which
    district is worst" for every district with water in it, not only the
    ones that happen to contain a PCD station. Districts are burned into a
    label grid aligned to the raster once (cheap - ~10ms) and averaged with
    a boolean mask per zone.
    """
    import rasterio.features
    import rasterio.transform

    try:
        districts = geo.load_districts()
    except FileNotFoundError:
        return pd.DataFrame(columns=["District", "NTU"])

    # The district layer covers the whole country; this ranking is about one
    # province, and the composite only spans that province anyway. Filtering
    # first keeps the rasterise to 25 polygons rather than 930, all but a
    # handful of which would burn no pixels at all.
    features = [f for f in districts["features"]
                if f["properties"].get("ADM1_NAME") == FOCUS_PROVINCE]

    turb, mask, b = load_province_composite(path)
    h, w = turb.shape
    transform = rasterio.transform.from_bounds(b.left, b.bottom, b.right, b.top, w, h)
    names = [f["properties"]["ADM2_NAME"] for f in features]
    zones = rasterio.features.rasterize(
        [(f["geometry"], i + 1) for i, f in enumerate(features)],
        out_shape=(h, w), transform=transform, fill=0,
        # uint8, not int32: 25 labels fit in a byte, and the display raster is
        # 73M pixels, where the wider dtype costs 294MB for the label grid
        # alone.
        dtype="uint8",
    )

    # Summed per label across water pixels in one pass, rather than a boolean
    # test per district. The old loop built "(zones == i) & mask" 25 times,
    # each allocating two more full-grid arrays; together with the int32 grid
    # that peaked at 514MB, over the limit a cloud container gets. This walks
    # ~1M water pixels instead of 25 x 73M, and gives identical means.
    labels = zones[mask]
    values = turb[mask]
    sums = np.bincount(labels, weights=values, minlength=len(names) + 1)
    counts = np.bincount(labels, minlength=len(names) + 1)

    rows = [{"District": name, "NTU": float(sums[i] / counts[i])}
            for i, name in enumerate(names, start=1) if counts[i]]
    return pd.DataFrame(rows).sort_values("NTU", ascending=False).reset_index(drop=True)


# Per-station turbidity for the *currently selected* composite date - this is
# what both the map markers and the sidebar station list show, so picking a
# different date updates both instead of only the map's own raster overlay.
station_now = station_summary.copy()
station_now["Predicted_NTU"] = [
    pc.sample_at(turbidity_map, valid_mask, bounds, r.station_la, r.station_lo) or r.Predicted_NTU
    for r in station_summary.itertuples()
]
station_now = station_now.sort_values("Predicted_NTU", ascending=False)

_station_coords = [(r.Code, r.station_la, r.station_lo) for r in station_summary.itertuples()]
# The joined form for the one-line captions (sidebar, ranking rows), and the
# split form for the marker popup, which labels each level separately. Both
# read the same cache, so this is one lookup's worth of work, not two.
station_geo = geo.station_locations(_station_coords, lang=LANG)
station_places = geo.station_location_parts(_station_coords, lang=LANG)

# Province-wide mean for the selected composite. Free - this composite's
# raster is already in memory.
province_now = float(turbidity_map[valid_mask].mean()) if valid_mask.any() else None

# ---------------------------------------------------------------- title ---
st.markdown(
    f'<div class="page-header"><span class="page-title">{T["page_title"]}</span>'
    f'<span class="page-subtitle">{T["page_subtitle"]}</span></div>',
    unsafe_allow_html=True,
)

# Center/zoom persist across reruns entirely client-side (see
# map_controls.add_view_persistence, called below) - only the very first-ever
# visit has nothing saved, so this fallback only has to be reasonably close;
# it's immediately corrected (see add_view_persistence) either from a saved
# position or a proper fitBounds to the province boundary.
b_minx, b_miny, b_maxx, b_maxy = geo.load_province(FOCUS_PROVINCE).bounds
center_lat = station_summary["station_la"].mean()
center_lon = station_summary["station_lo"].mean()
fmap = folium.Map(location=[center_lat, center_lon], zoom_start=8, tiles=None, zoom_control=False)
fmap.get_root().header.add_child(folium.Element(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Noto+Sans+Thai:wght@400;500;600;700&display=swap');
    .leaflet-popup-content, .leaflet-tooltip {{
        font-family: {FONT_STACK} !important;
    }}
    /* No focus ring on click - was showing as a black box around whatever
       shape (often a huge invisible province polygon) took the click. */
    .leaflet-container *:focus, .leaflet-container *:focus-visible {{
        outline: none !important;
    }}
    </style>
    """
))

# --- Switchable basemaps, one TileLayer per style; the fold-out rail
# (map_controls.add_layer_rail, added below) swaps the active one. ---
basemap_tile_layers = {}
for name, cfg in BASEMAPS.items():
    layer = folium.TileLayer(
        tiles=cfg["tiles"], attr=cfg["attr"], name=name, control=False, show=(name == DEFAULT_BASEMAP),
    )
    layer.add_to(fmap)
    basemap_tile_layers[name] = layer

def _drop_admin_word(value, label):
    """Strip the administrative word from a place name when the row's label
    already carries it - "District: Mueang Ubon Ratchathani District" reads
    as a mistake. English puts it last ("... District"), Thai first
    ("อำเภอ..."), so both ends are checked. Never returns empty: a name that
    is nothing but the word is left as it was.
    """
    v = value.strip()
    if v.lower().endswith(" " + label.lower()):
        v = v[: -len(label) - 1].strip()
    elif v.startswith(label):
        v = v[len(label):].strip()
    return v or value.strip()


# Top of the encoded range for the hidden value image below. Above the highest
# reading seen in the 10m composites (~1,290 NTU) with headroom.
VALUE_PNG_MAX_NTU = 1400.0

# The value image is stored at this many display cells per side, i.e. 80m
# against the map's 40m. See value_png_data_uri for the trade.
VALUE_PNG_BLOCK = 2


def send_once(key, loader):
    """The layer's GeoJSON on a session's first render, None on every later one.

    Paired with map_controls.add_geojson_layer, which parks the data on the top
    window: that survives a rerun (Streamlit rebuilds the map iframe but never
    reloads the page around it), so re-sending it would be re-sending something
    the browser already holds. This is what took a date change from 50.2MB to
    a few hundred KB.

    Keyed per session, so a page reload - which clears both the session and the
    window it cached into - sends it again, keeping the two in step.
    """
    sent = st.session_state.setdefault("_wq_layers_sent", set())
    if key in sent:
        return None
    sent.add(key)
    return loader()


@st.cache_data(show_spinner=False)
def turbidity_overlay_png(path, lang):
    """The turbidity overlay as a PALETTED PNG data URI.

    The image only ever holds eight colours - seven turbidity classes plus
    transparent - so storing four bytes a pixel was paying for a range it
    never uses. As an indexed PNG with a transparent index 0 it is pixel
    identical and 71% smaller: 0.91MB to 0.26MB on average, which comes
    straight off every date change because this is re-sent whenever the date
    does.

    `lang` is in the cache key only because the palette is language
    independent today, and a stale image would be the silent failure if that
    ever stopped being true.
    """
    turb, mask, bnds = pc.load_display(path)
    breakpoints = [c["max"] for c in style.CLASSES[:-1]]

    # Same Web Mercator row gather as the RGBA path - see mercator_row_map.
    rows = mercator_row_map(bnds, turb.shape[0])
    index = (np.digitize(turb[rows], breakpoints) + 1).astype(np.uint8)
    index[~mask[rows]] = 0

    palette = [0, 0, 0]
    for c in style.CLASSES:
        palette += [int(c["color"][i:i + 2], 16) for i in (1, 3, 5)]
    palette += [0] * (768 - len(palette))

    image = Image.fromarray(index, mode="P")
    image.putpalette(palette)
    buf = io.BytesIO()
    image.save(buf, format="PNG", transparency=0, optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


@st.cache_data(show_spinner=False)
def value_png_data_uri(path):
    """The turbidity values as a hidden PNG, for client-side readout.

    Why an image and not vector cells. The obvious way to make each pixel
    clickable is a polygon per pixel, and it does not survive contact with
    the numbers: at the 40m display grid one date is 307,803 water cells =
    46MB of GeoJSON, and merging same-class neighbours still leaves 47,294
    polygons and 21.6MB. Leaflet draws each as an SVG path and stalls in the
    tens of thousands. The PNG that draws the same information is 0.9MB.

    So the picture stays a picture, and a second picture carries the numbers.
    The reader never sees this one - JS samples it with a canvas on tap, which
    is what makes the readout instant instead of a server round trip.

    Encoding: grey = sqrt(NTU / MAX), alpha = water mask. The square root
    spends the 256 levels where the data is - about 1 NTU per step below
    100 NTU, coarsening to ~4 NTU near the top of the range - rather than
    spreading them evenly over a range whose upper half is nearly empty.
    Sized against the model, not by taste: its RMSE is 12.2 NTU, so 1-2 NTU
    of quantisation adds 2-5% to an error that is already there, while 16-bit
    exactness would cost another 590KB per date to state a precision the
    model does not have.

    Alpha separates "no reading" from "0 NTU", which grey alone cannot.
    """
    turb, mask, _bounds = pc.load_display(path)

    # Half the display grid, so this image is a quarter of the pixels. It is
    # the larger of the two sent per date change - 1.04MB against the colour
    # overlay's 0.91MB - and it shrinks far better than it loses accuracy:
    # 1.04MB -> 0.29MB, a 72% cut, for a reading averaged over 80m instead of
    # 40m. The COLOUR overlay stays at 40m, so nothing about the map's
    # appearance changes; only the number a tap reports is coarser.
    #
    # Blockwise, not turb[::2, ::2]: taking every other pixel would drop a
    # river narrower than two cells out of the mask entirely and report "no
    # water" where the map plainly shows some. Any water in the block keeps
    # the block, and its value is the mean of the water in it.
    h, w = (turb.shape[0] // VALUE_PNG_BLOCK) * VALUE_PNG_BLOCK, \
           (turb.shape[1] // VALUE_PNG_BLOCK) * VALUE_PNG_BLOCK
    blocks = mask[:h, :w].reshape(h // VALUE_PNG_BLOCK, VALUE_PNG_BLOCK,
                                  w // VALUE_PNG_BLOCK, VALUE_PNG_BLOCK)
    vals = np.where(mask, turb, 0.0)[:h, :w].reshape(blocks.shape)
    wet = blocks.any(axis=(1, 3))
    count = blocks.sum(axis=(1, 3))
    total = vals.sum(axis=(1, 3))
    turb_small = np.divide(total, count, out=np.zeros_like(total),
                           where=count > 0)

    scaled = np.sqrt(np.clip(turb_small, 0, VALUE_PNG_MAX_NTU) / VALUE_PNG_MAX_NTU)
    grey = np.round(scaled * 255).astype(np.uint8)
    alpha = np.where(wet, 255, 0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.dstack([grey, alpha]), mode="LA").save(
        buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def station_popup_html(code, predicted_ntu, measured_ntu, cls):
    """The card shown when a station marker is clicked: station code as the
    heading, then where it is, then its numbers.

    The level is a filled pill in its own class colour rather than a word, so
    it reads the same way as the marker it came from, the legend and the
    district card - one severity scale, shown one way everywhere.

    The note at the foot is what the old inline labels ("Predicted
    (satellite)", "Measured (PCD avg)") used to carry. Those qualifiers
    matter - the predicted figure moves with the selected date while the
    measured one is a fixed average over the whole PCD record - so they moved
    into a caption rather than being dropped when the labels were shortened.
    """
    parts = station_places.get(code, {})
    levels = (("subdistrict", T["popup_subdistrict"]),
              ("district", T["popup_district"]),
              ("province", T["popup_province"]))
    named = {key: _drop_admin_word(parts.get(key, ""), label) if parts.get(key) else ""
             for key, label in levels}
    # A subdistrict often shares its district's name (Khong Chiam sits in
    # Khong Chiam District). Printing it under both labels tells the reader
    # nothing, so the broader level is the one kept.
    if named["subdistrict"] and named["subdistrict"].casefold() == named["district"].casefold():
        named["subdistrict"] = ""
    rows = [f'<div class="wq-pop-row"><b>{label}:</b> {named[key]}</div>'
            for key, label in levels if named[key]]
    place_block = f'<div class="wq-pop-group">{"".join(rows)}</div>' if rows else ""
    return (
        f'<div class="wq-pop-title">{code}</div>'
        + place_block
        + '<div class="wq-pop-group">'
        f'<div class="wq-pop-row"><b>{T["popup_predicted"]}:</b> {predicted_ntu:.1f} NTU</div>'
        f'<div class="wq-pop-row"><b>{T["popup_measured"]}:</b> {measured_ntu:.1f} NTU</div>'
        f'<div class="wq-pop-row"><b>{T["popup_level"]}:</b> '
        f'<span class="wq-pop-pill" style="background:{cls["color"]}">'
        f'{predicted_ntu:.1f} NTU &middot; {cls["label"]}</span></div>'
        '</div>'
        f'<div class="wq-pop-note">{T["popup_note"].format(date=f"{picked_date:%d %b %Y}")}</div>'
    )


station_layer = folium.FeatureGroup(name="Ground Stations", show=True)
for _, r in station_now.iterrows():
    cls = style.classify(r["Predicted_NTU"])
    folium.CircleMarker(
        location=[r["station_la"], r["station_lo"]],
        radius=8, color=STATION_STROKE_COLOR, weight=1, fill=True,
        fill_color=cls["color"], fill_opacity=0.95,
        popup=folium.Popup(
            station_popup_html(r["Code"], r["Predicted_NTU"], r["Turbidity_Actual"], cls),
            # Wide enough for "Mueang Ubon Ratchathani" to stay on one line;
            # min_width stops short codes collapsing the card to a sliver.
            max_width=320, min_width=250,
        ),
        tooltip=r["Code"],
    ).add_to(station_layer)
# Not added to fmap yet - added last, below, after the boundary/raster
# layers, so station points always draw on top of them instead of being
# visually crossed out by a province/district line passing through.
stations_def = {
    "key": "stations", "label": T["pcd_stations_label"], "layer": station_layer, "default_on": True,
    "title": T["pcd_dept"],
}

# Boundary hover labels follow the interface language. The shapefiles these
# come from carry both names (see import_shapefiles.py); the previous Earth
# Engine source had English only, which is why these were English-only before.
PROVINCE_NAME_FIELD = "ADM1_NAME_TH" if LANG == "th" else "ADM1_NAME"
DISTRICT_NAME_FIELD = "ADM2_NAME_TH" if LANG == "th" else "ADM2_NAME"

# District name labels. Dark text with a white halo rather than a filled chip:
# the halo keeps the name legible over the Satellite basemap without painting
# 25 opaque boxes across the water the map exists to show.
# position:absolute is load-bearing, not decoration. As a plain block child of
# the 0x0 DivIcon container the div inherits width 0 - the text still paints,
# because overflow is visible, but getBoundingClientRect reports a zero-width
# box and declutter_labels can then never detect a collision. Out of flow it
# shrinks to fit its own text, so the measured box is the text.
DISTRICT_LABEL_CSS = (
    "position:absolute;left:0;top:0;"
    "font:600 11px/1.15 'Poppins','Noto Sans Thai',sans-serif;"
    "color:#3d4652;"
    "text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff,0 0 3px #fff;"
    "white-space:nowrap;transform:translate(-50%,-50%);"
    "pointer-events:none;"
)


# Self-contained page for the sidebar choropleth iframe. It inherits nothing
# from the app - not the font, not the palette - so everything it needs is
# substituted in. Braces are doubled where they are literal CSS/JS.
_CHOROPLETH_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Noto+Sans+Thai:wght@400;600;700&display=swap');
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family:{font}; background:transparent; }}
#wrap {{ position:relative; width:100%; }}
svg {{ width:100%; height:auto; display:block; }}
path {{ stroke:#fff; stroke-width:1; vector-effect:non-scaling-stroke;
        cursor:pointer; transition:opacity .12s ease; }}
#wrap.act path {{ opacity:.4; }}
#wrap.act path.on {{ opacity:1; stroke:{muted}; stroke-width:1.6; }}

/* The card sits in normal flow rather than over the map, and the frame is
   always drawn - only its contents come and go. That keeps a stable, visible
   place for the reading, so nothing appears out of nowhere and the map never
   shifts down as the pointer moves between districts. */
#tip {{ background:#fff; border:1px solid {border}; border-radius:10px;
        padding:9px 12px; margin-bottom:8px;
        box-shadow:0 2px 6px rgba(20,25,40,.07);
        pointer-events:none; }}
/* The rows, not the card, are what fades. They keep their boxes either way,
   which is what holds the frame at one height. */
.row {{ display:flex; align-items:baseline; gap:6px; line-height:1.35;
        opacity:0; transition:opacity .1s ease; }}
#tip.on .row {{ opacity:1; }}
.row + .row {{ margin-top:4px; }}
.rl {{ font-weight:700; color:{text}; flex:0 0 auto; }}
#nm {{ font-size:14px; }}
#nmt {{ font-weight:400; color:{text}; }}
#tv {{ font-size:12px; }}
#pill {{ display:inline-block; padding:2px 10px; border-radius:999px;
         font-size:11.5px; font-weight:700; color:{text}; white-space:nowrap; }}
/* An empty pill is 4px tall (padding only), a filled one ~19px, which moved
   the frame by 3px every time a district was picked. A non-breaking space
   gives the empty one a line box, so the frame holds one height. */
#pill:empty::before {{ content:'\\00a0'; }}
#hint {{ font-size:10px; color:{muted}; padding:3px 1px 0; }}
</style>
<div id="wrap">
  <div id="tip">
    <div class="row" id="nm"><span class="rl">{district_label}:</span><span id="nmt"></span></div>
    <div class="row" id="tv"><span class="rl">{turbidity_label}:</span><span id="pill"></span></div>
  </div>
  <svg viewBox="0 0 {vb_w} {vb_h}" role="img" aria-label="{label}">{paths}</svg>
</div>
<div id="hint">{hint}</div>
<script>
const D = {data};
const wrap = document.getElementById('wrap'), tip = document.getElementById('tip');
const nmt = document.getElementById('nmt'), pill = document.getElementById('pill');
let current = null;

function show(el) {{
  const d = D[el.id];
  if (!d) return;
  if (current) current.classList.remove('on');
  current = el; el.classList.add('on');
  nmt.textContent = d.name;
  pill.innerHTML = d.detail;
  // The pill carries the class colour, so the reading and the shape on the
  // map say the same thing without a separate key.
  pill.style.background = d.color;
  wrap.classList.add('act'); tip.classList.add('on');
}}
function hide() {{
  if (current) current.classList.remove('on');
  current = null;
  wrap.classList.remove('act'); tip.classList.remove('on');
}}

document.querySelectorAll('svg path').forEach(function (p) {{
  // pointerenter carries the cursor; pointerdown carries the tap. A touch
  // device fires no enter at all, which is why CSS :hover alone could not
  // satisfy this.
  p.addEventListener('pointerenter', function () {{ show(p); }});
  p.addEventListener('pointerdown', function (e) {{ e.preventDefault(); show(p); }});
}});
// Moving the cursor off the map clears it - but ONLY for a mouse. A touch
// pointer is destroyed the moment the finger lifts, which fires pointerleave
// too, and hiding on that tore the banner down instantly on every tap: the
// text was set and then cleared before it could be seen.
document.querySelector('svg').addEventListener('pointerleave', function (e) {{
  if (e.pointerType === 'mouse') hide();
}});
// What clears it on touch: tapping anywhere that is not a district.
document.addEventListener('pointerdown', function (e) {{
  if (e.target.tagName.toLowerCase() !== 'path') hide();
}});

// The sidebar decides our width, so the height can only be known here.
//
// Measured from the last element's bottom, NOT documentElement.scrollHeight:
// the html element stretches to fill whatever height the iframe currently
// has, so scrollHeight can never report less than the value it is being used
// to set. That fed back and pinned the frame at its initial guess, leaving
// 111px of blank sidebar under the hint.
function fit() {{
  const last = document.getElementById('hint');
  const h = Math.ceil(last.getBoundingClientRect().bottom) + 2;
  const f = window.frameElement;
  if (!f) return;
  f.style.height = h + 'px';
  // Streamlit reserves the `height` argument on the iframe's CONTAINER, and
  // resizing only the iframe leaves that container at its original size - the
  // sidebar then keeps the difference as blank space (126px of it here).
  //
  // flex-basis, not height. The container is a flex item in Streamlit's
  // column layout with `flex: 0 0 <reserved>px`, and in a column flex
  // container the basis IS the height - so setting `height` had no effect at
  // all, even inline and even with !important. Both are set: the basis is
  // what actually governs, the height keeps the two consistent.
  const box = f.parentElement;
  if (box) {{
    box.style.setProperty('flex', '0 0 ' + h + 'px', 'important');
    box.style.setProperty('height', h + 'px', 'important');
  }}
}}
window.addEventListener('load', fit);
window.addEventListener('resize', fit);
fit();
</script>
"""


@st.cache_data(show_spinner=False)
def district_svg_shapes(name_field, province=FOCUS_PROVINCE, width=286, tolerance=0.0025):
    """(width, height, [(key, display_name, svg_path)]) for a sidebar choropleth.

    An inline SVG rather than a second folium map. A Leaflet instance costs
    another iframe, its own tile requests and a full re-serialisation on every
    rerun - for a thumbnail in a 300px column that never pans or zooms. The
    geometry is already loaded, so this is a projection and a string.

    Mercator y, matching the main map, so the two show the same province the
    same shape. At this size the choice is worth about half a pixel; it is
    made anyway because having a thumbnail that disagrees with the map beside
    it is the sort of thing nobody ever tracks down later.

    `key` is always ADM2_NAME because district_ntu() ranks by that regardless
    of interface language; `display_name` follows the language. Joining on the
    displayed name would silently produce an all-grey map in Thai.
    """
    import math

    from shapely.geometry import shape

    features = [f for f in geo.load_districts()["features"]
                if f["properties"].get("ADM1_NAME") == province]
    if not features:
        return 0, 0, []

    def merc_y(lat):
        # Degrees, not radians. x below is raw longitude in degrees, and
        # log(tan(...)) comes out in radians - mixing the two squashed the
        # whole province into a 7px-tall sliver, scaled wrong by 180/pi.
        return math.degrees(math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))

    geoms = [shape(f["geometry"]).simplify(tolerance, preserve_topology=True)
             for f in features]
    lons = [c for g in geoms for c in (g.bounds[0], g.bounds[2])]
    lats = [c for g in geoms for c in (g.bounds[1], g.bounds[3])]
    x0, x1 = min(lons), max(lons)
    y0, y1 = merc_y(min(lats)), merc_y(max(lats))
    scale = width / (x1 - x0)
    height = (y1 - y0) * scale

    def ring(coords):
        pts = [f"{(lon - x0) * scale:.1f},{(y1 - merc_y(lat)) * scale:.1f}"
               for lon, lat in coords]
        return "M" + "L".join(pts) + "Z"

    shapes = []
    for feature, geom in zip(features, geoms):
        parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        d = "".join(ring(p.exterior.coords) + "".join(ring(i.coords) for i in p.interiors)
                    for p in parts)
        props = feature["properties"]
        shapes.append((props["ADM2_NAME"], props.get(name_field) or props["ADM2_NAME"], d))
    return width, height, shapes


def district_choropleth_html(ranking):
    """The district ranking drawn as a map: each amphoe filled by its
    turbidity class, in the same colours as the legend and the map overlay,
    with a detail banner on hover or tap.

    Districts absent from `ranking` are the ones whose polygon contained no
    water pixel on this date - cloud, or simply no mapped water - and are
    drawn in the no-data grey rather than the lowest class, which would read
    as "clean" for somewhere nothing was measured at all.

    Rendered through components.html rather than st.markdown, because this
    needs real JavaScript and st.markdown strips <script>. CSS alone was tried
    first and does not cover the ask:

      - :hover works, and handles the cursor case.
      - :focus does NOT match on an SVG <path>, even with tabindex and even
        when document.activeElement is that path - measured, not assumed.
      - Mobile "sticky hover" does not fire either: a real emulated tap left
        the banner at opacity 0.

    So the tap case has no CSS-only mechanism. pointerenter drives the cursor,
    pointerdown drives the tap, and both feed the same banner.

    The iframe sizes itself: its height depends on the width the sidebar gives
    it, which is not knowable here, so the page reports its own height back
    through window.frameElement once laid out.
    """
    width, height, shapes = district_svg_shapes(DISTRICT_NAME_FIELD)
    if not shapes:
        return "", 0

    ntu = {r.District: r.NTU for r in ranking.itertuples()}

    paths, detail = [], {}
    for i, (key, name, d) in enumerate(shapes):
        value = ntu.get(key)
        if value is None:
            fill = DISTRICT_NODATA_COLOR
            line2 = T["no_water_pixels"]
        else:
            cls = style.classify(value)
            fill = cls["color"]
            line2 = f"{value:.1f} NTU &middot; {cls['label']}"
        paths.append(f'<path id="dp{i}" d="{d}" fill="{fill}"/>')
        detail[f"dp{i}"] = {"name": name, "detail": line2, "color": fill}

    aspect = height / width if width else 1.0
    return _CHOROPLETH_TEMPLATE.format(
        font=FONT_STACK,
        text=P["text"], muted=P["muted"], border=P["border"],
        vb_w=f"{width:.0f}", vb_h=f"{height:.0f}",
        paths="".join(paths),
        data=json.dumps(detail, ensure_ascii=False),
        label=html.escape(T["district_ranking"]),
        hint=html.escape(T["district_map_hint"]),
        district_label=html.escape(T["district_label"]),
        turbidity_label=html.escape(T["turbidity_label"]),
    ), aspect


@st.cache_data(show_spinner=False)
def district_label_points(name_field, province=FOCUS_PROVINCE):
    """(lat, lon, name) for each of `province`'s districts.

    Reads the boundaries itself rather than taking them as an argument:
    st.cache_data hashes its inputs, and hashing a 930-feature GeoJSON on
    every rerun would cost more than the work it is caching.

    name_field has no default on purpose. st.cache_data keys on the arguments
    it is passed, so a defaulted field would be absent from the key and the
    English labels would be served to the Thai interface - which is exactly
    what happened before the caller was made to pass it.

    Ubon's 25 rather than all 930. The map opens fitted to Ubon, so national
    labels would be off-screen text that still ships on every rerun - and at
    that zoom the ones that did land on screen would overplot each other into
    an unreadable smear. Districts elsewhere keep the hover tooltip they have
    always had.

    representative_point(), not centroid: a centroid can fall outside a
    concave amphoe or land in the water between the parts of a multipolygon,
    which would put the name somewhere the district isn't. For multipolygons
    the point is taken from the largest part, so the label sits on the main
    body rather than on an outlying sliver.

    Returns (lat, lon, name, rank), rank 0 being the largest district by area.
    Zoomed out there is not room for all 25 names, so map_controls.declutter_
    labels() drops whichever ones collide - and it needs an ordering to decide
    what to drop. Area is the honest one: the big rural amphoe are the ones
    with space to print a name in, and the cluster that actually overlaps is
    the small districts ringing Ubon city.
    """
    from shapely.geometry import shape

    points = []
    for feature in geo.load_districts()["features"]:
        props = feature["properties"]
        if props.get("ADM1_NAME") != province:
            continue
        name = props.get(name_field)
        if not name:
            continue
        geom = shape(feature["geometry"])
        area = geom.area
        if geom.geom_type == "MultiPolygon":
            geom = max(geom.geoms, key=lambda g: g.area)
        pt = geom.representative_point()
        points.append((pt.y, pt.x, name, area))

    points.sort(key=lambda p: -p[3])
    return [(lat, lon, name, rank) for rank, (lat, lon, name, _a) in enumerate(points)]

# Boundary draw order, lowest first: districts, then the other 76 provinces,
# then Ubon. Leaflet paints these SVG paths in the order they are added, so
# whatever is added last sits on top - and the Ubon highlight is meant to be
# read over everything else, not crossed by the 930 district lines that used
# to be drawn after it.

# --- Districts, country-wide. On by default: it is a full national layer
# rather than the secondary detail it was when it held one province, and
# leaving it switched off meant a visitor never saw it without hunting through
# the rail for a toggle. ---
# --- Water: Thailand's wetland areas, from the local archive.
#
# In its OWN pane, below the overlay pane, and that is load-bearing rather
# than tidiness. Adding it first is not enough: Leaflet gives the vector <svg>
# z-index 200 and an ImageOverlay z-index 1, both absolutely positioned inside
# the same overlay pane - so z-index decides, not document order, and the
# water painted OVER the turbidity raster even though the raster came later in
# the DOM. Measured: 33,151 pixels in one viewport became a pure class colour
# the moment the water layer was switched off, i.e. every one of them had been
# tinted by 50% water.
#
# Demoting the whole overlay pane instead would drag the district and province
# lines under the opaque raster with it, and those lines follow the rivers in
# places - the Mun IS a district boundary along part of its length - so they
# would vanish exactly where the water is. A pane at 350 sits above the tiles
# and below the overlay pane, which puts water under the reading while leaving
# the boundaries over it.
WATER_PANE = "waterpane"
folium.map.CustomPane(WATER_PANE, z_index=350, pointer_events=False).add_to(fmap)

# Fetched by the browser rather than embedded - 7.9MB of wetland polygons that
# never change with the date. See map_controls.add_geojson_layer.
water_layer = folium.FeatureGroup(name="Water", show=True)
water_layer.add_to(fmap)
map_controls.add_geojson_layer(
    fmap, water_layer, "water", send_once("water", geo.load_water),
    style={"color": WATER_LINE_COLOR, "weight": 0.3, "opacity": 0.4,
           "fill": True, "fillColor": WATER_FILL_COLOR,
           "fillOpacity": WATER_FILL_OPACITY},
    pane=WATER_PANE,
)
water_def = {"key": "water", "label": T["water_label"],
             "layer": water_layer, "default_on": True,
             "title": T["water_source"]}

district_def = None
try:
    # Outlines and name labels travel together in one FeatureGroup, so the
    # rail's existing District button switches both: labels floating over a
    # map with no boundaries under them would read as loose noise.
    #
    # The outlines are fetched by the browser (5.9MB that never changes with
    # the date); the 25 labels are generated here, being a few KB.
    district_layer = folium.FeatureGroup(name="Districts", show=True)
    # add_to BEFORE add_geojson_layer, as with the water and province
    # layers. folium emits every child's JS in the order the children were
    # added, so a fetch script registered first would run before the
    # `var district_layer = L.featureGroup(...)` it assigns into.
    district_layer.add_to(fmap)
    map_controls.add_geojson_layer(
        fmap, district_layer, "districts",
        send_once("districts", geo.load_districts),
        # Solid, not dashed: dashes on 930 outlines are noise at any zoom that
        # shows more than a province or two.
        style={"color": DISTRICT_LINE_COLOR, "weight": 0.8,
               "fill": False, "fillOpacity": 0},
        tooltip_field=DISTRICT_NAME_FIELD,
    )
    for lat, lon, label, rank in district_label_points(DISTRICT_NAME_FIELD):
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                # 0x0 container, and the text centred on it by transform. The
                # container has no area, so labels cannot swallow clicks meant
                # for the station markers underneath.
                icon_size=(0, 0), icon_anchor=(0, 0),
                html=f'<div class="wq-dlabel" data-wq-rank="{rank}" '
                     f'style="{DISTRICT_LABEL_CSS}">{label}</div>',
            ),
        ).add_to(district_layer)
    district_def = {"key": "district", "label": T["district_label"], "layer": district_layer, "default_on": True}
except FileNotFoundError:
    pass

# --- All Thailand provinces, Ubon Ratchathani highlighted. Fetched by the
# browser like the districts; the Ubon-last ordering that keeps its highlight
# on top is done there too (see add_geojson_layer's focus_* arguments). ---
province_layer = folium.FeatureGroup(name="Provinces", show=True)
province_layer.add_to(fmap)
map_controls.add_geojson_layer(
    fmap, province_layer, "provinces",
    send_once("provinces", geo.load_thailand_provinces),
    # fill:False (not just fillOpacity:0) - otherwise the invisible fill still
    # counts as "painted" for hit-testing and the whole province polygon
    # (which covers every station) swallows clicks meant for the markers
    # underneath it.
    #
    # weight 1.4 rather than 1: with the district layer on, a province line the
    # same width as its own subdivisions stops reading as the higher level.
    style={"color": PROVINCE_LINE_COLOR, "weight": 1.4,
           "fill": False, "fillOpacity": 0},
    tooltip_field=PROVINCE_NAME_FIELD,
    focus_field="ADM1_NAME", focus_value=FOCUS_PROVINCE,
    focus_style={"color": PROVINCE_FOCUS_COLOR, "weight": 3,
                 "fill": False, "fillOpacity": 0},
)
province_def = {"key": "province", "label": T["province_label"],
                "layer": province_layer, "default_on": True}


# Added after the water layer and the boundaries, so it draws over them:
# Leaflet paints in insertion order, and this is the reading the map exists
# to show.
#
# opacity=1, and the per-pixel alpha above is 255 too. Both mattered - the two
# multiply, so the old 0.9 here on top of 217/255 there left the reading at
# 76% and the water layer showing through it.
turbidity_layer = folium.raster_layers.ImageOverlay(
    image=turbidity_overlay_png(picked_path, LANG),
    bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
    opacity=1, name="Turbidity", show=True,
)
turbidity_layer.add_to(fmap)
turbidity_def = {"key": "turbidity", "label": T["turbidity_label"], "layer": turbidity_layer, "default_on": True}

station_layer.add_to(fmap)

# --- Tap-to-read, entirely client-side. See map_controls.add_pixel_readout.
map_controls.add_pixel_readout(
    fmap, value_png_data_uri(picked_path), bounds,
    {
        "classes": [{"max": c["max"], "color": c["color"], "label": c["label"]}
                    for c in style.CLASSES],
        "maxNtu": VALUE_PNG_MAX_NTU,
        "districtField": DISTRICT_NAME_FIELD,
        "provinceField": PROVINCE_NAME_FIELD,
        "districtLabel": T["popup_district"],
        "provinceLabel": T["popup_province"],
        "predictedLabel": T["popup_predicted"],
        "noWater": T["pixel_no_water"],
        "note": T["pixel_note"].format(date=f"{picked_date:%d %b %Y}"),
    },
    district_layer=district_layer if district_def else None,
)

# Water last, so its rail button sits below the turbidity one.
overlay_defs = [d for d in [stations_def, province_def, district_def, turbidity_def,
                            water_def] if d is not None]
map_controls.add_layer_rail(
    fmap, basemap_tile_layers, DEFAULT_BASEMAP, overlay_defs, build_legend_html(),
    legend_label=T["legend_label"], basemap_label=T["basemap_label"],
    font_stack=FONT_STACK,
    info_html=build_info_html(), info_label=T["info_label"],
)
map_controls.add_view_persistence(fmap, [[b_miny, b_minx], [b_maxy, b_maxx]])
map_controls.add_zoom_control(fmap)
map_controls.compact_attribution(fmap)
# After the view is restored, so the first pass measures the zoom the reader
# actually lands on rather than folium's initial one.
map_controls.declutter_labels(fmap)

# height is generously large; the CSS rule on this iframe (see <style> above)
# clips/fills it to the actual viewport, so this just needs to cover the tallest
# realistic screen and avoid leaving blank space below a shorter fixed render.
# returned_objects=[]: panning/zooming is handled entirely client-side (see
# add_view_persistence) precisely so it does NOT feed back into a Streamlit
# return value - that would rerun the whole script on every pan/zoom tick.
# returned_objects is limited to last_clicked on purpose. st_folium reruns the
# whole app whenever its return value changes, so asking for pan/zoom state
# would rerun on every mouse move across the map (which is why view
# persistence is done client-side - see map_controls.add_view_persistence).
# A click is a deliberate act, and one rerun per tap is what it costs.
# returned_objects=[] - nothing comes back from the map, so nothing it does
# can trigger a rerun. The tap-to-read popup is handled inside the iframe (see
# map_controls.add_pixel_readout); routing clicks through Streamlit instead
# cost 6.5s per tap, because every rerun re-serialises all the layers.
st_folium(fmap, use_container_width=True, height=1400, returned_objects=[])

idx = dates.index(picked_date)

with st.container(key="timeline_bar"):
    # calendar badge | slider | prev/next | language
    nav_cal, nav_slider, nav_arrows, nav_lang = st.columns([1.1, 18, 1.7, 2.2])
    with nav_cal:
        years = sorted({d.year for d in dates})
        year_label = str(years[0]) if len(years) == 1 else f"{years[0]}-{years[-1]}"
        # Keyed container purely to get a stable hook for centring: the
        # markdown wrappers Streamlit puts around this collapse to 16px, so a
        # min-height on the badge itself overflowed downward instead of
        # centring, leaving the year sitting below the EN/TH pill.
        with st.container(key="cal_badge"):
            st.markdown(
                f'<div class="wq-cal"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                f'stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
                f'<rect x="3" y="5" width="18" height="16" rx="2.5"/>'
                f'<path d="M3 10h18M8 3v4M16 3v4"/>'
                f'<path d="M7.5 14h2M11 14h2M14.5 14h2M7.5 17.5h2M11 17.5h2"/></svg>'
                f'<div class="wq-cal-year">{year_label}</div></div>',
                unsafe_allow_html=True,
            )

    # The prev/next block is executed BEFORE the slider even though it sits
    # to the right of it: `with nav_arrows` writes into that column wherever
    # this code appears, so placement is unaffected by execution order.
    # Order matters here because both controls own the same state. Once
    # st.select_slider(key="picked_date") has been instantiated in a run,
    # Streamlit refuses further assignment to st.session_state.picked_date
    # and drops it silently - which left the arrows visibly doing nothing.
    # Running them first means the assignment lands before the widget exists.
    with nav_arrows:
        # Both arrows in one column, side by side, so they read as a single
        # prev/next pair rather than bracketing the slider.
        with st.container(key="date_nav"):
            prev_col, next_col = st.columns(2)
            # Labels are kept (and hidden with CSS) rather than blanked, so
            # the buttons still have accessible names.
            with prev_col:
                with st.container(key="nav_prev"):
                    if st.button("Previous date", disabled=idx == 0, width="stretch"):
                        st.session_state.picked_date = dates[idx - 1]
                        st.rerun()
            with next_col:
                with st.container(key="nav_next"):
                    if st.button("Next date", disabled=idx == len(dates) - 1, width="stretch"):
                        st.session_state.picked_date = dates[idx + 1]
                        st.rerun()

    with nav_slider:
        st.select_slider(
            f"Imagery date, {dates[0]:%d %b %Y} to {dates[-1]:%d %b %Y}", options=dates,
            key="picked_date", format_func=lambda d: d.strftime("%d %b %Y"),
            label_visibility="collapsed",
        )
        # One label per composite date (few enough, now that the range is
        # fixed to RANGE_START..RANGE_END, to label every point instead of
        # needing to thin them out) - current selection bolded. Placed
        # INSIDE this column (not as a separate top-level element after the
        # whole st.columns row) so it inherits the slider's own width rather
        # than the full row's - that mismatch was why labels didn't line up
        # with the actual tick positions before.
        tick_row_html = "".join(
            f'<span class="wq-tick-label{" wq-tick-current" if d == picked_date else ""}">{d:%d %b}</span>'
            for d in dates
        )
        st.markdown(f'<div class="wq-tick-row">{tick_row_html}</div>', unsafe_allow_html=True)
    with nav_lang:
        # Both languages are always shown, the active one highlighted, so the
        # control reads as a state rather than as "press this to switch" -
        # the previous single button showed only the language you'd get,
        # which is ambiguous about which one is currently on.
        with st.container(key="lang_toggle"):
            en_col, th_col = st.columns(2)
            for col, code in ((en_col, "en"), (th_col, "th")):
                with col:
                    with st.container(key=f"lang_{code}{'_on' if LANG == code else ''}"):
                        if st.button(code.upper(), width="stretch", disabled=LANG == code):
                            st.session_state.lang = code
                            st.rerun()

# The sidebar is rendered here, after the map, so the map paints first on a
# cold load - the sections below loop over every composite and make a network
# call to the RID gauge service, and are the slowest things on the page.
with st.sidebar:
    st.markdown(f'<div class="sb-heading">{T["situation_overview"]}</div>',
                unsafe_allow_html=True)

    # --- Headline: latest province-wide turbidity ---
    st.markdown(f"#### {T['latest_turbidity']}")
    if province_now is None:
        st.caption(T["no_coverage"])
    else:
        c = style.classify(province_now)
        st.markdown(
            f'<div class="sb-metric"><div class="sb-value">{province_now:.1f} NTU</div>'
            f'<div class="sb-label">{T["province_average"]} &middot; {picked_date:%d %b %Y}</div>'
            f'<div style="margin-top:6px;"><span class="legend-swatch" style="background:{c["color"]}"></span> '
            f'{c["label"]}</div></div>',
            unsafe_allow_html=True,
        )
        # Change vs the preceding composite. Direction is carried by an arrow
        # and the number, not by color alone.
        prov = province_history()
        prior = prov[prov["Date"] < pd.Timestamp(picked_date)] if not prov.empty else prov
        if not prior.empty:
            delta = province_now - float(prior.iloc[-1]["NTU"])
            if abs(delta) < 0.05:
                chip = f'<span class="sb-delta" style="color:{P["muted"]}">= {T["no_change"]}</span>'
            else:
                arrow, color = ("▲", "#c0392b") if delta > 0 else ("▼", "#1d7a4c")
                chip = f'<span class="sb-delta" style="color:{color}">{arrow} {abs(delta):.1f} NTU</span>'
            st.markdown(f'<div class="sb-sub">{T["vs_previous"]} {chip}</div>', unsafe_allow_html=True)

    # --- Turbidity trend, per station, over the whole analysis window ---
    st.markdown(f"#### {T['turbidity_trend']}")
    st.caption(T["window_label"])
    qs_code = st.selectbox(T["station_select"], station_summary["Code"], label_visibility="collapsed")
    qs_row = station_summary.set_index("Code").loc[qs_code]
    history = station_history(qs_row["station_la"], qs_row["station_lo"])
    if history.empty:
        st.caption(T["no_coverage"])
    else:
        predicted = history.assign(Series=T["predicted_satellite"])
        # PCD ground samples for this station, if any fall in the window -
        # they are sparse and rarely coincide with a composite date, so they
        # are drawn as separate points rather than a second line.
        actual_pts = df_val.loc[
            (df_val["Code"] == qs_code)
            & (df_val["Date"] >= pd.Timestamp(RANGE_START))
            & (df_val["Date"] <= pd.Timestamp(RANGE_END)),
            ["Date", "Turbidity_"],
        ].rename(columns={"Turbidity_": "NTU"})

        color_scale = alt.Scale(
            domain=[T["predicted_satellite"], T["actual_pcd"]],
            range=[COLOR_PREDICTED, COLOR_ACTUAL],
        )
        has_actual = not actual_pts.empty
        # Legend only when there really are two series; a lone series is
        # already named by the section heading.
        legend = alt.Legend(title=None, orient="bottom") if has_actual else None
        layers = [
            alt.Chart(predicted).mark_line(
                strokeWidth=2,
                point=alt.OverlayMarkDef(size=45, filled=True, color=COLOR_PREDICTED),
            ).encode(
                x=alt.X("Date:T", title=None, axis=alt.Axis(format="%d %b")),
                y=alt.Y("NTU:Q", title="NTU", scale=alt.Scale(zero=False)),
                color=alt.Color("Series:N", scale=color_scale, legend=legend),
                tooltip=[
                    alt.Tooltip("Date:T", title="Date", format="%d %b %Y"),
                    alt.Tooltip("NTU:Q", title="NTU", format=".1f"),
                ],
            )
        ]
        if has_actual:
            layers.append(
                alt.Chart(actual_pts.assign(Series=T["actual_pcd"])).mark_circle(size=70).encode(
                    x="Date:T", y="NTU:Q",
                    color=alt.Color("Series:N", scale=color_scale, legend=legend),
                    tooltip=[
                        alt.Tooltip("Date:T", title="Date", format="%d %b %Y"),
                        alt.Tooltip("NTU:Q", title="NTU", format=".1f"),
                    ],
                )
            )
        st.altair_chart(
            alt.layer(*layers)
            .properties(height=165)
            .configure_axis(gridColor="#e1e0d9", domainColor="#c3c2b7", tickColor="#c3c2b7",
                            labelColor="#52514e", titleColor="#52514e", labelFontSize=10)
            .configure_view(strokeWidth=0)
            .configure_legend(labelFontSize=10, symbolSize=60),
            width="stretch",
        )
        st.caption(f"{qs_code} &middot; {station_geo.get(qs_code, '')}")

    # --- Streamflow / water level, RID Mun River gauges ---
    st.markdown(f"#### {T['streamflow_heading']}")
    st.caption(T["window_label"])

    # No run-up fetched any more: a monthly mean needs only the days inside
    # the month, where the rolling averages this used to draw needed up to 30
    # days of history before the first plotted point.
    levels = load_level_history(RANGE_START, RANGE_END)
    if levels.empty:
        st.markdown(
            f'<div class="sb-sub">{T["streamflow_unavailable"]}</div>', unsafe_allow_html=True,
        )
    else:
        # required=True: without it a segmented control is clearable, and
        # clicking the selected choice could leave *nothing* selected - the
        # buttons all went unlit while the chart silently fell back to the
        # full range. The `or` fallback guards the very first render.
        month_choices = {T["month_all"]: None, T["month_nov"]: 11, T["month_dec"]: 12}
        sel_month = st.segmented_control(
            T["month_label"], list(month_choices), default=T["month_all"],
            key="flow_month", required=True,
        ) or T["month_all"]

        shown = levels[levels["Date"] >= pd.Timestamp(RANGE_START)].copy()

        # 5-day bins, restarted at each month boundary rather than run
        # continuously from 01 Nov. A continuous run would put a bin across
        # 29 Nov - 03 Dec, which the month selector below could not place in
        # either month without either double-counting it or dropping it.
        #
        # Day 31 is clipped into the 26th's bin, so December ends with one
        # 6-day bin instead of a 1-day bin whose mean would sit on a single
        # reading and swing wildly against its neighbours.
        day = shown["Date"].dt.day
        bin_day = (((day - 1) // BIN_DAYS).clip(upper=(31 // BIN_DAYS) - 1)
                   * BIN_DAYS + 1)
        shown["BinStart"] = pd.to_datetime(
            dict(year=shown["Date"].dt.year, month=shown["Date"].dt.month, day=bin_day)
        )

        binned = (shown.groupby(["Gauge", "BinStart"], as_index=False)["Level"]
                  .mean().sort_values("BinStart"))

        # Axis fixed to 1..5 m, and measured from the WHOLE range rather than
        # the month on show, so switching month moves the lines and never the
        # scale - which is the only way two months can be compared by eye.
        #
        # The top clears the data rather than stopping at 5: the highest bin
        # is 5.20 m (M.9, early November), so a hard 1-5 domain would have
        # quietly clipped it off the top of the chart. Labels stay at
        # 1,2,3,4,5; the extra headroom is unlabelled.
        y_top = max(5.0, float(binned["Level"].max())) * 1.03

        month = month_choices[sel_month]
        if month is not None:
            binned = binned[binned["BinStart"].dt.month == month]

        gauges = sorted(binned["Gauge"].unique())
        colour = alt.Color(
            "Gauge:N",
            scale=alt.Scale(domain=gauges, range=GAUGE_COLORS[: len(gauges)]),
            legend=alt.Legend(title=None, orient="bottom"),
        )
        # Points joined by a line, not bars. These are metres above datum, not
        # counts, so the axis cannot start at zero without flattening every
        # difference - and a bar drawn from a truncated baseline states a
        # proportion that isn't true. A point sits at its value and claims
        # nothing about the distance to zero.
        base = alt.Chart(binned).encode(
            x=alt.X("BinStart:T", title=None, axis=alt.Axis(format="%d %b")),
            y=alt.Y("Level:Q", title=T["level_m"],
                    scale=alt.Scale(domain=[1, y_top], nice=False, clamp=False),
                    # labelOverlap=False: Vega's default drops alternate
                    # labels when it thinks they crowd, which at this height
                    # left the axis reading 1, 3, 5 even though all five ticks
                    # were in the DOM.
                    axis=alt.Axis(values=[1, 2, 3, 4, 5], labelOverlap=False)),
            color=colour,
            tooltip=[
                alt.Tooltip("Gauge:N", title=T["gauge"]),
                alt.Tooltip("BinStart:T", title=T["period_from"], format="%d %b %Y"),
                alt.Tooltip("Level:Q", title=T["period_mean"], format=".2f"),
            ],
        )
        flow_chart = (
            (base.mark_line(strokeWidth=2) + base.mark_point(size=45, filled=True))
            .properties(height=165)
            .configure_axis(gridColor="#e1e0d9", domainColor="#c3c2b7", tickColor="#c3c2b7",
                            labelColor="#52514e", titleColor="#52514e", labelFontSize=10)
            .configure_view(strokeWidth=0)
            .configure_legend(labelFontSize=10, symbolSize=60)
        )
        st.altair_chart(flow_chart, width="stretch")
        for code in gauges:
            st.markdown(
                f'<div class="sf-name"><b>{code}</b> &middot; {rid.station_name(code)}</div>',
                unsafe_allow_html=True,
            )
    # The note that used to sit here - what the gauges are, why water level is
    # shown rather than discharge, and what the two chart lines mean - now lives
    # only in the Information modal (info_src_streamflow).

    # --- District ranking by turbidity ---
    st.markdown(f"#### {T['district_ranking']}")
    districts = district_ntu(picked_path)
    if districts.empty:
        st.caption(T["no_districts"])
    else:
        st.caption(f'{T["district_ranking_note"]} &middot; {picked_date:%d %b %Y}')
        choropleth, aspect = district_choropleth_html(districts)
        if choropleth:
            # A starting height only. The iframe measures itself once the
            # sidebar has given it a width and corrects this - see fit().
            components.html(choropleth, height=int(300 * aspect) + 108,
                            scrolling=False)


# ---------------------------------------------------- cloud chrome hiding ---
# Hides Streamlit Cloud's "Hosted with Streamlit" badge and the owner avatar,
# which it pins to the bottom-right corner on top of the app - over the
# language switch, swallowing its taps, for anonymous visitors as much as for
# the signed-in owner.
#
# Note for whoever reads this next: that badge is Community Cloud's
# attribution for hosting the app for free, and its terms ask that it not be
# removed. Hiding it was an explicit, informed decision by the app owner, not
# an oversight - if this project ever moves to paid or self-hosting the block
# below can simply be deleted.
#
# Why script and not CSS: Cloud serves the app inside a nested iframe, and the
# badge lives in the host document above it, which a stylesheet written here
# can never reach - an earlier CSS attempt had no effect at all (transform
# stayed 'none' on the deployed page). The two frames are same-origin (both on
# the app's own hostname), so script can cross that boundary. Selectors come
# from the deployed DOM: the badge carries no data-testid, only an href and a
# content-hashed class, so href is what it is matched on.
#
# Re-applied on an interval because Cloud re-renders its own chrome - on
# reconnect, for instance - which drops styles set once at load. Wrapped in
# try/catch so that if the reach ever stops working the app simply carries on;
# the timeline bar keeps its own bottom clearance either way, so the controls
# stay usable even then.
components.html(
    """
    <script>
    (function () {
        function place() {
            try {
                var d = window.top.document;
                var els = [
                    d.querySelector('a[href^="https://streamlit.io/cloud"]'),
                    d.querySelector('[class*="_profileContainer"]')
                ];
                els.forEach(function (el) {
                    if (!el) { return; }
                    el.style.setProperty('display', 'none', 'important');
                });
            } catch (e) { /* cross-origin or no host frame - leave as-is */ }
        }
        place();
        setInterval(place, 2000);
    })();
    </script>
    """,
    height=0,
)
