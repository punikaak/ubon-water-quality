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
import json

import altair as alt
import folium
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import streamlit as st
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
DISTRICT_LINE_COLOR = "#6b7684"
STATION_STROKE_COLOR = "#2b2b3a"
HEADER_NAVY = "#1e3a5f"  # "si krom tha" - the dark navy used for the floating title card

# Switchable basemaps, presented via the custom fold-out rail in
# map_controls.py (styled after air4laos.adpc.net), not Leaflet's default
# LayerControl widget.
BASEMAPS = {
    "Light": {"tiles": "CartoDB positron", "attr": None},
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
# Averaging windows offered for the streamflow chart, shortest first. Rolling
# averages rather than N-day buckets: over a 61-day range, 30-day buckets give
# two points and a single-month view would give one, which is not a trend.
SMOOTH_WINDOWS = [7, 14, 30]

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
        "stations_heading": "Stations",
        "station_select": "Station",
        "no_coverage": "No composite coverage at this station's location.",
        "legend_layers": "Layers",
        "legend_turbidity_levels": "Turbidity Levels",
        "legend_province": "Province boundary",
        "legend_district": "District boundary",
        "legend_pcd_stations": "PCD stations",
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
        "streamflow_note": (
            "Daily stage from the RID Lower-NE gauges on the Mun River, downloaded for "
            "01 Nov - 31 Dec 2024. The service publishes no discharge (m^3/s) figure for "
            "these gauges - the field exists but is empty on every day in the range - so "
            "water level is shown, being the quantity actually measured. The faint line is "
            "the daily reading; the solid line is the average."
        ),
        "month_label": "Month",
        "month_all": "Nov-Dec",
        "month_nov": "Nov",
        "month_dec": "Dec",
        "smooth_label": "Averaging window",
        "smooth_days": "{n}-day",
        "daily_reading": "Daily",
        "avg_reading": "{n}-day average",
        "level_m": "Level (m)",
        "gauge": "Gauge",
        "district_ranking": "District Ranking",
        "district_ranking_note": "Mean turbidity of water pixels within each district, highest first.",
        "no_districts": "District boundaries not available.",
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
            "<b>Streamflow / water level:</b> Daily Mun River stage from the "
            '<a href="https://hydro-4.rid.go.th" target="_blank" rel="noopener">Royal Irrigation '
            "Department</a> (RID) Lower North-East Hydro Center. The service publishes no "
            "discharge (m&sup3;/s) figure for these gauges - the field exists but is empty on "
            "every day in the range - so water level is shown, being the quantity actually "
            "measured."
        ),
        "info_src_boundaries": (
            "<b>Administrative boundaries:</b> Thai administrative shapefiles held with this "
            "project - all 77 provinces, and Ubon Ratchathani's 25 districts built by merging "
            "the subdistrict (tambon) polygons into their amphoe. Reprojected from UTM zone 47N."
        ),
        "info_src_basemaps": (
            "<b>Base maps:</b> CARTO (Light, Dark), OpenStreetMap (Classic), OpenTopoMap "
            "(Terrain) and Esri World Imagery (Satellite)."
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
        "stations_heading": "สถานีตรวจวัด",
        "station_select": "สถานี",
        "no_coverage": "ไม่มีข้อมูลดาวเทียมครอบคลุมตำแหน่งสถานีนี้",
        "legend_layers": "ชั้นข้อมูล",
        "legend_turbidity_levels": "ระดับความขุ่น",
        "legend_province": "ขอบเขตจังหวัด",
        "legend_district": "ขอบเขตอำเภอ",
        "legend_pcd_stations": "สถานีคุณภาพน้ำ",
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
        "streamflow_note": (
            "ระดับน้ำรายวันจากสถานีวัดน้ำแม่น้ำมูล กรมชลประทาน (สำนักงานอุทกวิทยาภาคตะวันออกเฉียงเหนือตอนล่าง) "
            "ดึงข้อมูลช่วง 1 พ.ย. - 31 ธ.ค. 2567 ระบบไม่ได้ส่งค่าอัตราการไหล (ลบ.ม./วินาที) สำหรับสถานีเหล่านี้ "
            "โดยมีฟิลด์ข้อมูลแต่ว่างเปล่าทุกวันในช่วงนี้ จึงแสดงเป็นระดับน้ำซึ่งเป็นค่าที่วัดได้จริง "
            "เส้นจางคือค่ารายวัน เส้นทึบคือค่าเฉลี่ย"
        ),
        "month_label": "เดือน",
        "month_all": "พ.ย.-ธ.ค.",
        "month_nov": "พ.ย.",
        "month_dec": "ธ.ค.",
        "smooth_label": "ช่วงเฉลี่ย",
        "smooth_days": "{n} วัน",
        "daily_reading": "รายวัน",
        "avg_reading": "เฉลี่ย {n} วัน",
        "level_m": "ระดับน้ำ (ม.)",
        "gauge": "สถานีวัดน้ำ",
        "district_ranking": "อันดับความขุ่นรายอำเภอ",
        "district_ranking_note": "ค่าเฉลี่ยความขุ่นของพื้นที่น้ำในแต่ละอำเภอ เรียงจากมากไปน้อย",
        "no_districts": "ไม่มีข้อมูลขอบเขตอำเภอ",
        "risk": "ความเสี่ยง",
        "window_label": "1 พ.ย. - 31 ธ.ค. 2567",
        "popup_subdistrict": "ตำบล",
        "popup_district": "อำเภอ",
        "popup_province": "จังหวัด",
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
            "<b>ปริมาณน้ำ / ระดับน้ำ:</b> ระดับน้ำรายวันของแม่น้ำมูล จาก"
            '<a href="https://hydro-4.rid.go.th" target="_blank" rel="noopener">กรมชลประทาน</a> '
            "ศูนย์อุทกวิทยาชลประทานภาคตะวันออกเฉียงเหนือตอนล่าง "
            "ทั้งนี้ระบบไม่ได้เผยแพร่ค่าอัตราการไหล (ลบ.ม./วินาที) ของสถานีเหล่านี้ "
            "โดยมีช่องข้อมูลแต่ว่างเปล่าทุกวันในช่วงที่แสดง จึงแสดงระดับน้ำ "
            "ซึ่งเป็นค่าที่ตรวจวัดจริงแทน"
        ),
        "info_src_boundaries": (
            "<b>ขอบเขตการปกครอง:</b> ไฟล์รูปร่าง (shapefile) เขตการปกครองของไทยที่จัดเก็บไว้กับโครงการนี้ "
            "ครอบคลุมทั้ง 77 จังหวัด และ 25 อำเภอของจังหวัดอุบลราชธานี "
            "ซึ่งสร้างขึ้นโดยการรวมขอบเขตระดับตำบลเข้าเป็นอำเภอ "
            "ข้อมูลถูกแปลงพิกัดจากระบบ UTM โซน 47N"
        ),
        "info_src_basemaps": (
            "<b>แผนที่ฐาน:</b> CARTO (สว่าง, มืด), OpenStreetMap (คลาสสิก), "
            "OpenTopoMap (ภูมิประเทศ) และ Esri World Imagery (ภาพถ่ายดาวเทียม)"
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
        padding:10px 14px; display:flex; align-items:baseline; justify-content:center; gap:8px; }}
    /* !important: .stApp span (above) targets every span including these
       two and otherwise wins on specificity (class+type beats a bare class). */
    .page-title {{ font-size: 0.95rem; font-weight: 700; color: #ffffff !important; }}
    .page-subtitle {{ font-size: 0.72rem; color: #b7c4d4 !important; }}

    .legend-swatch {{ width:12px; height:12px; border-radius:3px; display:inline-block; flex-shrink:0; }}

    .sb-metric {{ border:1px solid {P['border']}; border-radius:12px; padding:10px 12px; margin-bottom:10px; }}
    .sb-value {{ font-size:1.5rem; font-weight:700; color:{P['text']}; }}
    .sb-label {{ font-size:0.72rem; color:{P['muted']}; text-transform:uppercase; letter-spacing:.04em; }}
    /* Delta chip next to the hero number - direction is carried by an arrow
       glyph and the text itself, never by color alone. */
    .sb-delta {{ font-size:0.78rem; font-weight:600; margin-left:6px; }}
    .sb-sub {{ font-size:0.72rem; color:{P['muted']}; margin-top:2px; }}

    /* Streamflow gauge rows - same visual family as .risk-row. */
    .sf-row {{ display:flex; justify-content:space-between; align-items:baseline;
        padding:6px 0 0 0; font-size:0.82rem; }}
    .sf-name {{ font-size:0.68rem; color:{P['muted']}; padding-bottom:6px; line-height:1.3; }}
    .sf-value {{ font-weight:700; }}
    .sf-note {{ font-size:0.68rem; color:{P['muted']}; line-height:1.4;
        border-left:3px solid {P['border']}; padding:2px 0 2px 8px; margin-top:6px; }}

    .risk-row {{ display:flex; justify-content:space-between; align-items:center;
        padding: 6px 10px 0 10px; border-radius: 10px; color:{P['text']}; }}
    .risk-row-wrap:hover {{ background: {P['sidebar_bg']}; }}
    .risk-pill {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-weight:600;
        font-size: 0.78rem; color: #2b2b3a; }}
    .risk-location {{ font-size:0.68rem; color:{P['muted']}; padding: 0 10px 6px 10px; line-height:1.3; }}

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
    .wq-cal {{ display:flex; flex-direction:column; align-items:center; justify-content:center;
        gap:1px; color:{HEADER_NAVY}; padding-top:4px; }}
    .wq-cal svg {{ width:22px; height:22px; }}
    .wq-cal-year {{ font-size:0.6rem; font-weight:600; color:{P['muted']}; letter-spacing:.02em; }}

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
        position: fixed !important; left: 314px !important; bottom: 114px !important;
        width: 34px !important; height: 34px !important; z-index: 1002 !important; }}
    [data-testid="stExpandSidebarButton"] {{
        position: fixed !important; left: 20px !important; bottom: 114px !important;
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
    /* 34px, matching the map's Information button directly above it - the two
       sit in the same bottom-left column, so a size difference between them
       read as a mistake rather than a hierarchy. */
    button[data-testid="stExpandSidebarButton"] {{
        width: 34px !important; height: 34px !important; border-radius: 50% !important;
        background: #ffffff url("{SIDEBAR_ICON}") center / 20px 20px no-repeat !important;
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
      .page-header {{ top:8px; left:8px; right:8px; padding:7px 11px;
          flex-direction:column; align-items:flex-start; justify-content:flex-start; gap:1px; }}
      .page-title {{ font-size:0.82rem; line-height:1.2; }}
      .page-subtitle {{ font-size:0.6rem; line-height:1.25; }}

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

      .wq-cal {{ padding-top:0; }}
      .wq-cal svg {{ width:18px; height:18px; }}
      .wq-cal-year {{ font-size:0.54rem; }}
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
      [data-testid="stExpandSidebarButton"] {{ left:12px !important; bottom:146px !important;
          width:30px !important; height:30px !important; }}
      [data-testid="stSidebarCollapseButton"] {{ left:308px !important; bottom:146px !important;
          width:30px !important; height:30px !important; }}
      /* 30px here, not the desktop 34: this matches both the Information
         button and Leaflet's own zoom buttons at this breakpoint, so the
         three controls in the column are one size. */
      button[data-testid="stExpandSidebarButton"],
      [data-testid="stExpandSidebarButton"] button,
      button[data-testid="stSidebarCollapseButton"],
      [data-testid="stSidebarCollapseButton"] button {{
          width:30px !important; height:30px !important; background-size:18px 18px !important; }}
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


def turbidity_overlay_rgba(turbidity_map, water_mask):
    breakpoints = [c["max"] for c in style.CLASSES[:-1]]
    colors = [c["color"] for c in style.CLASSES]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm([0] + breakpoints + [breakpoints[-1] * 3], cmap.N)
    rgba = cmap(norm(turbidity_map))
    rgba[..., 3] = np.where(water_mask, 0.85, 0.0)
    return rgba


def build_legend_html():
    layer_rows = (
        f'<div class="wq-legend-item"><span class="wq-legend-line" style="background:{PROVINCE_LINE_COLOR}"></span>{T["legend_province"]}</div>'
        f'<div class="wq-legend-item"><span class="wq-legend-line-dashed" style="border-top:2px dashed {DISTRICT_LINE_COLOR}"></span>{T["legend_district"]}</div>'
        f'<div class="wq-legend-item"><span class="wq-legend-circle" style="border:2px solid {STATION_STROKE_COLOR}"></span>{T["legend_pcd_stations"]}</div>'
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
        T["info_src_boundaries"],
        T["info_src_basemaps"],
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


@st.cache_data(show_spinner=False)
def load_provinces_for_display(tolerance=0.02):
    """Thailand province outlines, simplified again for rendering.

    The cached GeoJSON is ~700KB of 77 polygons, and streamlit-folium
    re-sends the whole map (this included) to the browser on *every* rerun -
    Leaflet re-drawing it is the dominant cost of every interaction, ~5s
    measured. Simplifying to ~2km cuts it to under a third with no visible
    difference at the zoom levels this map ever shows.

    Except for Ubon itself, which is exempt. It is the subject of the map
    rather than context around it, and at 0.02 this step was moving its
    outline by up to 2.2km - almost as far as the entirely different
    OpenStreetMap boundary sits from it (3.4km). The result was that
    switching the source to the province shapefile made no visible
    difference, because this was smoothing the new detail straight back off.
    It is one polygon of 77, so drawing it as cached costs ~56KB.
    """
    from shapely.geometry import mapping, shape

    features = []
    for f in geo.load_thailand_provinces()["features"]:
        geom = shape(f["geometry"])
        if f["properties"].get("ADM1_NAME") != FOCUS_PROVINCE:
            geom = geom.simplify(tolerance, preserve_topology=True)
        if geom.is_empty:
            continue
        features.append({
            "type": "Feature",
            "properties": {
                "ADM1_NAME": f["properties"].get("ADM1_NAME"),
                # Carried through so the hover label can be Thai in Thai mode;
                # dropping it here would strand it in the cache file.
                "ADM1_NAME_TH": f["properties"].get("ADM1_NAME_TH"),
            },
            "geometry": mapping(geom),
        })
    return {"type": "FeatureCollection", "features": features}


@st.cache_data(ttl=3600, show_spinner="Loading streamflow gauges...")
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
        districts = geo.load_ubon_districts()
    except FileNotFoundError:
        return pd.DataFrame(columns=["District", "NTU"])

    turb, mask, b = load_province_composite(path)
    h, w = turb.shape
    transform = rasterio.transform.from_bounds(b.left, b.bottom, b.right, b.top, w, h)
    names = [f["properties"]["ADM2_NAME"] for f in districts["features"]]
    zones = rasterio.features.rasterize(
        [(f["geometry"], i + 1) for i, f in enumerate(districts["features"])],
        out_shape=(h, w), transform=transform, fill=0, dtype="int32",
    )

    rows = []
    for i, name in enumerate(names, start=1):
        sel = (zones == i) & mask
        if sel.any():
            rows.append({"District": name, "NTU": float(turb[sel].mean())})
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


def station_popup_html(code, predicted_ntu, measured_ntu, class_label):
    """The card shown when a station marker is clicked: station code as the
    heading, then where it is, then its numbers.

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
        f'<div class="wq-pop-row"><b>{T["popup_level"]}:</b> {class_label}</div>'
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
            station_popup_html(r["Code"], r["Predicted_NTU"], r["Turbidity_Actual"], cls["label"]),
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

# --- All Thailand provinces, Ubon Ratchathani highlighted ---
province_def = None
try:
    provinces_geojson = load_provinces_for_display()

    def province_style(feature):
        is_focus = feature["properties"].get("ADM1_NAME") == FOCUS_PROVINCE
        return {
            "color": PROVINCE_FOCUS_COLOR if is_focus else PROVINCE_LINE_COLOR,
            "weight": 3 if is_focus else 1,
            # fill:False (not just fillOpacity:0) - otherwise the invisible
            # fill still counts as "painted" for hit-testing and the whole
            # province polygon (which covers every station) swallows clicks
            # meant for the markers underneath it.
            "fill": False,
            "fillOpacity": 0,
        }

    province_layer = folium.GeoJson(
        provinces_geojson, name="Provinces", style_function=province_style,
        tooltip=folium.GeoJsonTooltip(fields=[PROVINCE_NAME_FIELD], aliases=[""]),
        show=True,
    )
    province_layer.add_to(fmap)
    province_def = {"key": "province", "label": T["province_label"], "layer": province_layer, "default_on": True}
except FileNotFoundError as e:
    st.info(str(e))

# --- Ubon districts (off by default - secondary detail) ---
district_def = None
try:
    districts_geojson = geo.load_ubon_districts()
    district_layer = folium.GeoJson(
        districts_geojson, name="Districts",
        style_function=lambda f: {"color": DISTRICT_LINE_COLOR, "weight": 1, "dashArray": "3,3",
                                   "fill": False, "fillOpacity": 0},
        tooltip=folium.GeoJsonTooltip(fields=[DISTRICT_NAME_FIELD], aliases=[""]),
        show=False,
    )
    district_layer.add_to(fmap)
    district_def = {"key": "district", "label": T["district_label"], "layer": district_layer, "default_on": False}
except FileNotFoundError:
    pass

overlay_rgba = turbidity_overlay_rgba(turbidity_map, valid_mask)
turbidity_layer = folium.raster_layers.ImageOverlay(
    image=overlay_rgba,
    bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
    opacity=0.9, name="Turbidity", show=True,
)
turbidity_layer.add_to(fmap)
turbidity_def = {"key": "turbidity", "label": T["turbidity_label"], "layer": turbidity_layer, "default_on": True}

station_layer.add_to(fmap)

overlay_defs = [d for d in [stations_def, province_def, district_def, turbidity_def] if d is not None]
map_controls.add_layer_rail(
    fmap, basemap_tile_layers, DEFAULT_BASEMAP, overlay_defs, build_legend_html(),
    legend_label=T["legend_label"], basemap_label=T["basemap_label"],
    font_stack=FONT_STACK,
    info_html=build_info_html(), info_label=T["info_label"],
)
map_controls.add_view_persistence(fmap, [[b_miny, b_minx], [b_maxy, b_maxx]])
map_controls.add_zoom_control(fmap)
map_controls.compact_attribution(fmap)

# height is generously large; the CSS rule on this iframe (see <style> above)
# clips/fills it to the actual viewport, so this just needs to cover the tallest
# realistic screen and avoid leaving blank space below a shorter fixed render.
# returned_objects=[]: panning/zooming is handled entirely client-side (see
# add_view_persistence) precisely so it does NOT feed back into a Streamlit
# return value - that would rerun the whole script on every pan/zoom tick.
st_folium(fmap, use_container_width=True, height=1400, returned_objects=[])

idx = dates.index(picked_date)

with st.container(key="timeline_bar"):
    # calendar badge | slider | prev/next | language
    nav_cal, nav_slider, nav_arrows, nav_lang = st.columns([1.1, 18, 1.7, 2.2])
    with nav_cal:
        years = sorted({d.year for d in dates})
        year_label = str(years[0]) if len(years) == 1 else f"{years[0]}-{years[-1]}"
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
    st.caption(T["situation_overview"])

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

    # Fetched with a 30-day run-up so the widest averaging window is already
    # full at the first plotted day - see load_level_history(lead_days=).
    levels = load_level_history(RANGE_START, RANGE_END, lead_days=SMOOTH_WINDOWS[-1])
    if levels.empty:
        st.markdown(
            f'<div class="sb-sub">{T["streamflow_unavailable"]}</div>', unsafe_allow_html=True,
        )
    else:
        # required=True: without it a segmented control is clearable, and
        # clicking a choice could leave *nothing* selected - the buttons all
        # went unlit while the chart silently fell back to the full range.
        # The `or` fallbacks stay as a guard for the very first render.
        month_choices = {T["month_all"]: None, T["month_nov"]: 11, T["month_dec"]: 12}
        sel_month = st.segmented_control(
            T["month_label"], list(month_choices), default=T["month_all"], key="flow_month",
            required=True,
        ) or T["month_all"]
        sel_window = st.segmented_control(
            T["smooth_label"], SMOOTH_WINDOWS, default=SMOOTH_WINDOWS[0], key="flow_window",
            required=True, format_func=lambda n: T["smooth_days"].format(n=n),
        ) or SMOOTH_WINDOWS[0]

        # Average over the full fetched series (run-up included) and only then
        # cut to the month on show, so switching month changes the view, never
        # the arithmetic behind a given day's point.
        levels = levels.sort_values("Date")
        levels["Smooth"] = levels.groupby("Gauge")["Level"].transform(
            lambda s: s.rolling(sel_window, min_periods=1).mean()
        )
        shown = levels[levels["Date"] >= pd.Timestamp(RANGE_START)]
        month = month_choices[sel_month]
        if month is not None:
            shown = shown[shown["Date"].dt.month == month]

        gauges = sorted(shown["Gauge"].unique())
        colour = alt.Color(
            "Gauge:N",
            scale=alt.Scale(domain=gauges, range=GAUGE_COLORS[: len(gauges)]),
            legend=alt.Legend(title=None, orient="bottom"),
        )
        base = alt.Chart(shown).encode(
            x=alt.X("Date:T", title=None, axis=alt.Axis(format="%d %b")), color=colour,
        )
        y_axis = alt.Y("Level:Q", title=T["level_m"], scale=alt.Scale(zero=False))
        # Raw daily kept underneath at low opacity: the averaged line is the
        # trend, but without the reading behind it there is no way to see how
        # much smoothing the chosen window is doing.
        daily = base.mark_line(strokeWidth=1, opacity=0.28).encode(y=y_axis)
        smooth = base.mark_line(strokeWidth=2.5).encode(
            y=alt.Y("Smooth:Q", title=T["level_m"], scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("Gauge:N", title=T["gauge"]),
                alt.Tooltip("Date:T", title="Date", format="%d %b %Y"),
                alt.Tooltip("Level:Q", title=T["daily_reading"], format=".2f"),
                alt.Tooltip("Smooth:Q",
                            title=T["avg_reading"].format(n=sel_window), format=".2f"),
            ],
        )
        flow_chart = (
            (daily + smooth)
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
    st.markdown(f'<div class="sf-note">{T["streamflow_note"]}</div>', unsafe_allow_html=True)

    # --- District ranking by turbidity ---
    st.markdown(f"#### {T['district_ranking']}")
    districts = district_ntu(picked_path)
    if districts.empty:
        st.caption(T["no_districts"])
    else:
        st.caption(f'{T["district_ranking_note"]} &middot; {picked_date:%d %b %Y}')
        for i, row in enumerate(districts.itertuples(), start=1):
            cls = style.classify(row.NTU)
            st.markdown(
                f'<div class="rank-row"><span class="rank-num">{i}</span>'
                f'<span class="rank-name">{row.District}</span>'
                f'<span class="rank-ntu">{row.NTU:.1f} NTU</span>'
                f'<span class="rank-risk" style="background:{cls["color"]}">{cls["label"]}</span></div>',
                unsafe_allow_html=True,
            )

    # --- Ranked station list ---
    st.markdown(f"#### {T['stations_heading']}")
    for _, r in station_now.iterrows():
        cls = style.classify(r["Predicted_NTU"])
        st.markdown(
            f'<div class="risk-row-wrap"><div class="risk-row"><span>{r["Code"]}</span>'
            f'<span class="risk-pill" style="background:{cls["color"]}">{r["Predicted_NTU"]:.1f} NTU &middot; {cls["label"]}</span></div>'
            f'<div class="risk-location">{station_geo.get(r["Code"], "")}</div></div>',
            unsafe_allow_html=True,
        )


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
