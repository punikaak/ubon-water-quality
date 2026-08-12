"""Custom Leaflet controls for the map: a pill-shaped icon rail, top-right,
styled after the control rail on ADPC's Air4Laos dashboard
(https://air4laos.adpc.net) - one colored circular icon per layer (PCD
stations, province, district, turbidity), each a direct on/off toggle, plus
a Base Map icon that opens a fly-out style picker and a Legend icon that
opens the color/shape key.

(The sidebar show/hide "Ranking" toggle used to live here too, as a custom
icon inside this same iframe - it worked, but real mouse clicks routing
through a nested iframe into a cross-frame `window.parent` call turned out
to be unreliable in practice. It's now a plain st.button in dashboard.py's
main page instead, which doesn't have that problem.)

Folium's built-in LayerControl is a plain, small Leaflet widget that doesn't
match the rest of this dashboard's card-based look, so this replaces it
entirely with a hand-built control injected as raw JS/CSS/HTML. It has to be
assembled this way (rather than as normal Streamlit widgets) because the map
lives inside streamlit-folium's iframe - only JS running inside that iframe
can add/remove Leaflet layers on the actual map object.
"""
import json

import folium
from jinja2 import Template


class _RawScript(folium.MacroElement):
    """A map child whose only job is to emit a literal <script> body.

    folium.Element doesn't work here: streamlit_folium regenerates the page's
    JS by calling `element._template.module.script(element)` on every child
    of the Map (see generate_leaflet_string in streamlit_folium), expecting a
    Jinja template that defines a `script` macro - which is how every real
    folium layer (TileLayer, GeoJson, ...) is built. A plain Element has no
    such macro, so that lookup raises AttributeError, which the caller's
    `contextlib.suppress(UndefinedError, AttributeError)` swallows silently,
    dropping the content with no error. Defining the macro ourselves is what
    makes the injected JS actually reach the page.
    """

    def __init__(self, js: str):
        super().__init__()
        self._name = "RawScript"
        self._template = Template("{% macro script(this, kwargs) %}\n" + js + "\n{% endmacro %}")


_BASEMAP_ICON = (
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round">'
    '<path d="M9 4 4 6.5v13L9 17l6 2 5-2.5v-13L15 6l-6-2Z"/><path d="M9 4v13M15 6v13"/></svg>'
)
# A key/list glyph rather than the circled "i" this used to be: the Information
# button below is the one that means "i", and two identical glyphs three slots
# apart on the same rail is not a distinction anyone can act on.
_LEGEND_ICON = (
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round">'
    '<circle cx="5.2" cy="6.5" r="1.7" fill="currentColor" stroke="none"/>'
    '<line x1="10" y1="6.5" x2="20" y2="6.5"/>'
    '<circle cx="5.2" cy="12" r="1.7" fill="currentColor" stroke="none"/>'
    '<line x1="10" y1="12" x2="20" y2="12"/>'
    '<circle cx="5.2" cy="17.5" r="1.7" fill="currentColor" stroke="none"/>'
    '<line x1="10" y1="17.5" x2="20" y2="17.5"/></svg>'
)
_LEGEND_COLOR = "#5b6b7c"
_INFO_ICON = (
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="1.8" stroke-linecap="round">'
    '<circle cx="12" cy="12" r="9"/><line x1="12" y1="11" x2="12" y2="16"/>'
    '<circle cx="12" cy="7.6" r="0.6" fill="currentColor"/></svg>'
)
# The badge inside the modal header is a solid disc with a knocked-out "i"
# (rather than the rail's outline glyph), which is what gives the header its
# weight at the larger size.
_INFO_BADGE = (
    '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10.5" fill="currentColor"/>'
    '<circle cx="12" cy="7.2" r="1.45" fill="#fff"/>'
    '<rect x="10.6" y="10.2" width="2.8" height="7.2" rx="1.4" fill="#fff"/></svg>'
)
_INFO_COLOR = "#1e3a4a"

# Fixed per-layer icon glyph + accent color, keyed by the same "key" used in
# overlay_defs - this app only ever has these four toggleable overlays, so
# the mapping lives here rather than being threaded through from dashboard.py.
_OVERLAY_STYLE = {
    "stations": (
        "#4d7ea8",
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">'
        '<path d="M12 2C8.1 2 5 5.1 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.9-3.1-7-7-7Z"/></svg>',
    ),
    "province": (
        "#e05a2b",
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linejoin="round"><path d="M4 6c2 1.2 2.5-1 4.5-.3S10 7.5 12 6.5s3-2 4.5-.8S20 6 20 6v12'
        'c-2-1.2-2.5 1-4.5.3S13 16.5 12 17.5s-3 2-4.5.8S4 18 4 18Z"/></svg>',
    ),
    "district": (
        "#8b6bb0",
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linejoin="round"><path d="M4 6c2 1.2 2.5-1 4.5-.3S10 7.5 12 6.5s3-2 4.5-.8S20 6 20 6v12'
        'c-2-1.2-2.5 1-4.5.3S13 16.5 12 17.5s-3 2-4.5.8S4 18 4 18Z"/>'
        '<path d="M12 6.5v11" stroke-dasharray="2.5,2.5"/></svg>',
    ),
    "turbidity": (
        "#3ed99b",
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">'
        '<path d="M12 2C8 8 5 11.8 5 15.2A7 7 0 0 0 12 22a7 7 0 0 0 7-6.8C19 11.8 16 8 12 2Z"/></svg>',
    ),
    # Stacked waves. A droplet would repeat the turbidity glyph three slots
    # up; waves read as a body of water, which is what this layer is.
    "water": (
        "#1a4e8a",
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M2.5 7.5c1.7 0 1.7 1.5 3.4 1.5s1.7-1.5 3.4-1.5 1.7 1.5 3.4 1.5 '
        '1.7-1.5 3.4-1.5 1.7 1.5 3.4 1.5"/>'
        '<path d="M2.5 12.5c1.7 0 1.7 1.5 3.4 1.5s1.7-1.5 3.4-1.5 1.7 1.5 3.4 1.5 '
        '1.7-1.5 3.4-1.5 1.7 1.5 3.4 1.5"/>'
        '<path d="M2.5 17.5c1.7 0 1.7 1.5 3.4 1.5s1.7-1.5 3.4-1.5 1.7 1.5 3.4 1.5 '
        '1.7-1.5 3.4-1.5 1.7 1.5 3.4 1.5"/></svg>',
    ),
}
_BASEMAP_COLOR = "#c99a5b"

BASEMAP_SWATCHES = {
    "Light": "#eef1f4",
    "Dark": "#2b2f3a",
    "Classic": "linear-gradient(135deg,#cfe3d8,#f5eeda)",
    "Terrain": "linear-gradient(135deg,#b9cf9a,#d8c398)",
    "Satellite": "linear-gradient(135deg,#3f5c42,#283f52)",
}
BASEMAP_DESCRIPTIONS = {
    "Light": "Bright, minimal map for daytime use",
    "Dark": "Low-glare map for dark backgrounds",
    "Classic": "Familiar OpenStreetMap road view",
    "Terrain": "Elevation-shaded topographic view",
    "Satellite": "High-resolution aerial imagery",
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Noto+Sans+Thai:wght@400;500;600;700&display=swap');
/* position:fixed (not absolute): st_folium renders the map container at a
   tall fixed pixel height (see the height= comment at the call site), while
   the CSS on the outer <iframe> clips it to the actual viewport - anchoring
   to that oversized container would push the control outside the visible
   area. Fixed anchors to the iframe's own viewport instead, which always
   matches what's actually visible.
   top:75px (not 16px): the page title is now a full-width bar across the
   top of the map (see dashboard.py's .page-header), so this rail has to
   clear its height instead of sitting underneath/behind it. */
.wq-rail { position:fixed; top:75px; right:16px; z-index:1000;
    display:flex; flex-direction:column; align-items:center; gap:8px; background:#fff;
    border-radius:999px; box-shadow:0 2px 14px rgba(0,0,0,0.22); padding:10px 6px;
    font-family:__WQ_FONT__; }
.wq-icon-btn { display:flex; flex-direction:column; align-items:center; gap:3px; width:48px;
    border:none; background:none; padding:0; cursor:pointer; }
.wq-icon-circle { width:31px; height:31px; border-radius:50%; display:flex; align-items:center;
    justify-content:center; color:#fff; background:var(--wq-color); transition:background .15s,opacity .15s; }
.wq-icon-circle svg { width:17px; height:17px; }
.wq-icon-btn:not(.wq-on) .wq-icon-circle { background:#c7ccd2; }
.wq-icon-btn.wq-active .wq-icon-circle { box-shadow:0 0 0 2px rgba(30,58,74,0.25); }
.wq-icon-label { font-size:0.70rem; font-weight:600; color:#3a4450; text-align:center; line-height:1.1; }
.wq-icon-btn:not(.wq-on) .wq-icon-label { color:#a9b1ba; }
/* right:76px = the rail's 60px footprint (48px button + 6px padding each
   side) plus its own 16px inset. It was 60px for the narrower rail; leaving
   it there would open every fly-out underneath the rail it belongs to. */
.wq-panel { position:fixed; top:75px; right:76px; z-index:1000;
    width:230px; background:#fff; border-radius:12px; box-shadow:0 2px 14px rgba(0,0,0,0.2);
    padding:14px; display:none; font-family:__WQ_FONT__; max-height:70vh; overflow-y:auto; }
/* Leaflet's own zoom control, top of the bottom-left stack. The other two
   controls in that stack (timeline bar, ranking button) live in the outer
   page, not this iframe, so the offsets are coordinated by hand and were
   set from measured viewport positions rather than derived: this iframe's
   own bottom edge does not line up with the page's, so a value that looks
   like it should clear the ranking button lands ~35px lower than expected.
   At 190px the control sits just above the ranking button (which occupies
   roughly 856-908px down a 1000px-tall viewport).
   position:fixed !important: Leaflet's own leaflet.css anchors this corner
   with position:absolute relative to the map's own (oversized - see the
   .wq-rail comment above) container, not the visible viewport, so without
   this override the bottom offset measures from that container's real
   bottom edge, far below what's actually visible. */
.leaflet-bottom.leaflet-left { position:fixed !important; bottom:190px !important; left:16px !important; }
.leaflet-control-zoom { border-radius:12px !important; overflow:hidden;
    box-shadow:0 2px 14px rgba(0,0,0,0.22) !important; border:none !important; }
/* Sized to match the layer rail's buttons rather than left at Leaflet's
   default 30px. The rail's icon circles are 31px, but its buttons carry a
   label underneath and so read as a ~45px block; a bare 30px square beside
   that looks like a smaller class of control, which is exactly the
   complaint. 38px sits between the two and matches by weight. */
.leaflet-control-zoom a { font-family:__WQ_FONT__ !important;
    width:38px !important; height:38px !important; line-height:38px !important;
    font-size:26px !important; }
/* The OSM credit stays - ODbL requires attribution - but it does not need to
   be a full-contrast bar competing with the data, so it is toned down.
   position:fixed for the same reason as the zoom corner above: Leaflet
   anchors this to the map container, whose bottom edge sits ~28px below the
   visible viewport (the iframe starts 32px down the page but is sized from
   the full viewport height), so the credit rendered off-screen entirely -
   which does not satisfy the licence. Fixed re-anchors it to what is
   actually visible; the offset lifts it clear of the timeline bar. */
.leaflet-bottom.leaflet-right { position:fixed !important;
    bottom:132px !important; right:0 !important; }
.leaflet-control-attribution { font-family:__WQ_FONT__ !important;
    font-size:9px !important; background:rgba(255,255,255,0.55) !important;
    padding:1px 6px !important; border-radius:6px 0 0 6px !important;
    color:#7b8794 !important; }
.leaflet-control-attribution a { color:#66707c !important; text-decoration:none !important; }
.wq-panel.wq-open { display:block; }
.wq-panel-head { display:flex; align-items:center; gap:8px; font-weight:700; font-size:0.95rem;
    color:#1e3a4a; padding-bottom:8px; margin-bottom:8px; border-bottom:2px solid #eef0f2; }
.wq-row { display:flex; align-items:center; gap:10px; padding:7px 6px; border-radius:8px;
    cursor:pointer; font-size:0.82rem; color:#2b2b3a; line-height:1.3; }
.wq-row:hover { background:#f4f6f8; }
.wq-row.wq-active-row { background:#1e3a4a; color:#fff; }
.wq-row.wq-active-row small { color:#c7d0db; }
.wq-row small { display:block; color:#8a95a3; font-size:0.72rem; margin-top:1px; }
.wq-thumb { width:38px; height:32px; border-radius:6px; flex-shrink:0; }
.wq-legend-item { display:flex; align-items:center; gap:8px; padding:4px 0; font-size:0.82rem; color:#2b2b3a; }
.wq-legend-swatch { width:12px; height:12px; border-radius:3px; flex-shrink:0; }
.wq-legend-circle { width:12px; height:12px; border-radius:50%; flex-shrink:0; box-sizing:border-box; }
.wq-legend-line { width:18px; height:3px; border-radius:2px; flex-shrink:0; }
.wq-legend-line-dashed { width:18px; height:0; flex-shrink:0; }
.wq-legend-heading { font-weight:700; font-size:0.72rem; text-transform:uppercase; letter-spacing:.03em;
    color:#8a95a3; margin:10px 0 4px 0; }
.wq-legend-heading:first-child { margin-top:0; }
.wq-legend-caption { font-size:0.7rem; color:#8a95a3; margin-top:6px; line-height:1.35; }
.wq-legend-range { color:#8a95a3; font-size:0.72rem; }

/* -------------------------------------------------- station popups ---
   Leaflet's default popup is a tight 13px box with a small blue close link.
   These build it out into a card: station code as a heading, then labelled
   rows. The !importants are unavoidable - leaflet.css sets each of these
   directly on the same elements. */
.leaflet-popup-content-wrapper { border-radius:14px !important;
    box-shadow:0 4px 20px rgba(0,0,0,0.18) !important; padding:2px !important; }
.leaflet-popup-content { margin:17px 20px !important; font-family:__WQ_FONT__ !important;
    line-height:1.5 !important; font-size:0.86rem !important; color:#2b2b3a !important; }
.leaflet-popup-close-button { top:11px !important; right:13px !important;
    width:22px !important; height:22px !important; font:400 21px/22px Arial,sans-serif !important;
    color:#9aa3ad !important; }
.leaflet-popup-close-button:hover { color:#2b2b3a !important; background:none !important; }
.wq-pop-title { font-size:1.3rem; font-weight:700; color:#12161c; line-height:1.2;
    /* Room for the close button, which overlaps this line. */
    padding-right:26px; margin-bottom:13px; }
.wq-pop-group { margin-top:13px; }
.wq-pop-group:first-of-type { margin-top:0; }
/* A popup with no title starts straight into a row, which then runs under the
   close button - the clearance used to come from .wq-pop-title's padding.
   `> .wq-pop-group:first-child` matches only when the group really is the
   first thing in the popup, so titled popups are untouched. */
.leaflet-popup-content > .wq-pop-group:first-child .wq-pop-row:first-child {
    padding-right:26px; }
.wq-pop-row b { color:#12161c; font-weight:700; }
/* The level, as a filled pill in its turbidity class colour. Dark text on
   every class rather than a per-class text colour: the ramp runs from pale
   blue to dark red, and #12161c is the one foreground that stays legible
   across all of it. */
.wq-pop-pill { display:inline-block; padding:2px 10px; border-radius:999px;
    font-size:0.72rem; font-weight:700; color:#12161c; white-space:nowrap;
    vertical-align:1px; }
/* The pixel readout puts the pill on its own line under its label rather than
   beside it, so the number is not competing with the wording for the line. */
.wq-pop-pill-row { margin-top:6px; }
.wq-pop-pill-row .wq-pop-pill { font-size:0.8rem; padding:3px 12px; }
.wq-pop-note { margin-top:12px; font-size:0.7rem; color:#9aa3ad; line-height:1.4; }

/* ------------------------------------------------- information button ---
   Bottom-left, one control above the zoom buttons, and part of the same
   hand-coordinated stack as .leaflet-bottom.leaflet-left below - so the same
   caveat applies: position:fixed anchors to the iframe's visible viewport,
   and the offsets were read off the rendered zoom control rather than
   derived. Measured at 1500x950: the zoom box spans 26-56px from the left
   and its top edge sits 260px up from the bottom, so 270px clears it by 10
   and 24px centres a 34px circle on its 30px column. */
/* left:25px centres a 40px circle on the zoom column, which now spans 26-64px
   (38px wide at the corner's 26px inset). bottom:280px, not 270: the zoom
   pair grew to 76px tall and its top edge is now 266px up, so the old offset
   left only 4px of gap. */
.wq-info-fab { position:fixed; left:25px; bottom:290px; z-index:1000;
    width:40px; height:40px; padding:0; border:none; border-radius:50%;
    cursor:pointer; display:flex; align-items:center; justify-content:center;
    background:#fff; color:#12161c; box-shadow:0 2px 14px rgba(0,0,0,0.22);
    transition:background .15s; }
.wq-info-fab svg { width:22px; height:22px; }
.wq-info-fab:hover { background:#eef1f4; }

/* --------------------------------------------------------- info modal ---
   A centred modal rather than another fly-out panel: the data-source text
   runs to several paragraphs, which in a 230px rail panel would be a column
   ~30 characters wide. Same fixed-positioning reasoning as .wq-rail above -
   inset:0 covers the iframe's visible viewport, which is what the reader
   sees, not the oversized map container. */
/* The oversized bottom padding is clearance, not taste: the timeline bar is
   a page element painted on top of this iframe, so a card centred on the
   iframe's own viewport has its lower third covered by it. Padding is what
   bounds the card rather than a vh figure, because an inset:0 fixed element
   resolves its content box to the viewport minus its own padding - so the
   card's max-height:100% below is already "what's left once the timeline
   bar's strip is excluded", with no second number to keep in sync. */
.wq-modal { position:fixed; inset:0; z-index:2000; display:none;
    align-items:center; justify-content:center; padding:24px 24px 100px 24px;
    background:rgba(15,23,32,0.45); font-family:__WQ_FONT__; }
.wq-modal.wq-open { display:flex; }
.wq-modal-card { background:#fff; border-radius:16px; width:min(640px,100%);
    max-height:100%; display:flex; flex-direction:column; overflow:hidden;
    box-shadow:0 18px 50px rgba(0,0,0,0.32); }
.wq-modal-head { display:flex; align-items:center; gap:12px; flex-shrink:0;
    padding:20px 22px 14px 22px; border-bottom:1px solid #eef0f2; }
.wq-modal-badge { width:30px; height:30px; flex-shrink:0; color:#12161c; }
.wq-modal-badge svg { width:30px; height:30px; display:block; }
.wq-modal-title { flex:1; font-size:1.3rem; font-weight:700; color:#1e3a4a; }
.wq-modal-close { border:none; background:none; cursor:pointer; color:#5b6b7c;
    font-size:1.5rem; line-height:1; padding:2px 8px; border-radius:8px; }
.wq-modal-close:hover { background:#f1f3f6; color:#1e3a4a; }
.wq-modal-body { padding:16px 22px 22px 22px; overflow-y:auto; }
.wq-info-section { font-weight:700; font-size:0.95rem; color:#1e2a36; margin:0 0 10px 0; }
.wq-info-section:not(:first-child) { margin-top:18px; }
.wq-info-box { border:1px solid #e7eaf0; border-radius:10px; padding:0 14px; }
.wq-info-row { font-size:0.84rem; line-height:1.55; color:#2b2b3a; margin:0;
    padding:11px 0; border-bottom:1px solid #eef0f2; }
.wq-info-row:last-child { border-bottom:none; }
.wq-info-row b { color:#12161c; }
.wq-info-row a { color:#2a78d6; text-decoration:none; }
.wq-info-row a:hover { text-decoration:underline; }
.wq-info-note { font-size:0.74rem; color:#8a95a3; line-height:1.5; margin:12px 2px 0 2px; }

/* ------------------------------------------------------------- phones ---
   This stylesheet lives inside the map iframe, so it needs its own media
   query - the page's one (see dashboard.py) cannot reach in here. At 390px
   the rail ran 281px down the screen and the zoom control ended up buried
   under the timeline bar, which is much taller on a phone. Panels are also
   pinned to both edges rather than a fixed 230px, which would otherwise
   leave them wider than the gap they open into. */
@media (max-width: 640px) {
  .wq-rail { top:58px; right:8px; padding:7px 5px; gap:5px; }
  .wq-icon-btn { width:39px; }
  .wq-icon-circle { width:27px; height:27px; }
  .wq-icon-circle svg { width:14px; height:14px; }
  .wq-icon-label { font-size:0.59rem; }
  /* right:46px was the gap the old 31px-wide rail left; the rail is 39px
     now, so a panel opening at the old inset would sit under it. */
  .wq-panel { top:58px; right:56px; left:8px; width:auto; max-height:52vh; padding:11px; }
  .wq-row { font-size:0.78rem; padding:6px 5px; }
  .wq-legend-item { font-size:0.76rem; }
  /* Sits above the sidebar button (page-side, bottom:126px), which in turn
     sits above the two-row timeline bar. Offsets are larger than the page's
     because this anchors to the iframe's own viewport, which starts ~32px
     down the page and runs past its bottom edge. */
  .leaflet-bottom.leaflet-left { bottom:218px !important; left:10px !important; }
  /* Same relationship to the zoom control as on desktop, re-measured for
     this breakpoint: zoom spans 20-50px from the left with its top 288px up. */
  .wq-info-fab { left:19px; bottom:310px; width:36px; height:36px; }
  .wq-info-fab svg { width:20px; height:20px; }
  /* Zoom matched to the phone rail (27px circles) the same way the desktop
     pair is matched to its 31px ones. */
  .leaflet-control-zoom a { width:34px !important; height:34px !important;
      line-height:34px !important; font-size:24px !important; }
  /* Taller timeline bar here (two rows), so the credit still needs lifting
     further than on desktop even though the bar now sits at the same margin. */
  .leaflet-bottom.leaflet-right { bottom:170px !important; }
  /* Both page-level strips are deeper here than on desktop: the title header
     wraps to two lines at the top, and the timeline bar gains a second row at
     the bottom. */
  .wq-modal { padding:60px 12px 170px 12px; }
  .wq-modal-card { border-radius:13px; }
  .wq-modal-head { padding:15px 16px 11px 16px; gap:9px; }
  .wq-modal-badge, .wq-modal-badge svg { width:25px; height:25px; }
  .wq-modal-title { font-size:1.08rem; }
  .wq-modal-body { padding:13px 16px 17px 16px; }
  .wq-info-row { font-size:0.79rem; }
}

/* Landscape phone: the page chrome does not shrink with the viewport, so the
   clear band between the header and the timeline bar is only ~190px of 386.
   The portrait paddings would spend two thirds of that on margin; these are
   measured to the actual gap so the card gets all of it. */
/* Keyed on height alone, deliberately. The page chrome this clears - header
   above, timeline bar below - costs the same vertical space whatever the
   width, and gating on width as well made the rule depend on whether the
   sidebar happened to be open: open, this iframe is ~544px wide and the rule
   applied; closed, it is ~844px and the card fell back to the desktop
   paddings and overlapped both strips. */
@media (max-height: 480px) {
  /* No room above the zoom control here - stacking one higher puts the button
     behind the page header, which the zoom control is already close to. Sit
     beside it instead, bottom edges aligned (258px corner offset + the
     corner's own 10px padding). */
  .wq-info-fab { left:70px; bottom:268px; }
  .wq-modal { padding:78px 12px 128px 12px; }
  .wq-modal-head { padding:10px 14px 8px 14px; }
  .wq-modal-body { padding:10px 14px 14px 14px; }
  .wq-modal-title { font-size:1rem; }
  .wq-modal-badge, .wq-modal-badge svg { width:21px; height:21px; }
}
</style>
"""

_BEHAVIOR_JS = """
mapEl.querySelectorAll('[data-wq-overlay]').forEach(function (btn) {
    btn.addEventListener('click', function () {
        var key = btn.getAttribute('data-wq-overlay');
        var layer = overlays[key];
        var turningOn = !btn.classList.contains('wq-on');
        if (turningOn) { layer.addTo(map); } else { map.removeLayer(layer); }
        btn.classList.toggle('wq-on', turningOn);
    });
});

mapEl.querySelectorAll('[data-wq-toggle]').forEach(function (btn) {
    btn.addEventListener('click', function () {
        var key = btn.getAttribute('data-wq-toggle');
        var panel = mapEl.querySelector('[data-wq-panel="' + key + '"]');
        var wasOpen = panel.classList.contains('wq-open');
        mapEl.querySelectorAll('.wq-panel').forEach(function (p) { p.classList.remove('wq-open'); });
        mapEl.querySelectorAll('[data-wq-toggle]').forEach(function (b) { b.classList.remove('wq-active'); });
        if (!wasOpen) { panel.classList.add('wq-open'); btn.classList.add('wq-active'); }
    });
});

mapEl.querySelectorAll('[data-wq-basemap]').forEach(function (row) {
    row.addEventListener('click', function () {
        var name = row.getAttribute('data-wq-basemap');
        if (name === currentBase) return;
        map.removeLayer(basemaps[currentBase]);
        basemaps[name].addTo(map);
        currentBase = name;
        mapEl.querySelectorAll('[data-wq-basemap]').forEach(function (r) {
            r.classList.toggle('wq-active-row', r === row);
        });
    });
});

document.addEventListener('click', function (e) {
    if (e.target.closest && (e.target.closest('.wq-rail') || e.target.closest('.wq-panel'))) return;
    mapEl.querySelectorAll('.wq-panel').forEach(function (p) { p.classList.remove('wq-open'); });
    mapEl.querySelectorAll('[data-wq-toggle]').forEach(function (b) { b.classList.remove('wq-active'); });
});

/* ---------------------------------------------------------- info modal ---
   Opening it closes any rail fly-out first, so the two never stack. */
var infoModal = mapEl.querySelector('.wq-modal');
/* A Streamlit rerun replaces this whole iframe, taking any open modal with
   it - but the flag we set on the *parent* body survives that, and would
   leave the page's controls hidden for good. This runs on every render, so
   a stale flag never outlives the modal that set it. */
try { window.parent.document.body.classList.remove('wq-modal-open'); } catch (e) {}
if (infoModal) {
    var setInfo = function (open) {
        if (open) {
            mapEl.querySelectorAll('.wq-panel').forEach(function (p) { p.classList.remove('wq-open'); });
            mapEl.querySelectorAll('[data-wq-toggle]').forEach(function (b) { b.classList.remove('wq-active'); });
        }
        infoModal.classList.toggle('wq-open', open);
        /* The page's own floating controls are painted above this iframe and
           cannot be covered by a backdrop drawn inside it - they would sit on
           top of the card, still clickable. Flag the state on the parent body
           and let the page decide which of its controls to stand down; that
           keeps this module from having to know the page's selectors. */
        try { window.parent.document.body.classList.toggle('wq-modal-open', open); } catch (e) {}
    };
    mapEl.querySelectorAll('[data-wq-modal-open]').forEach(function (btn) {
        btn.addEventListener('click', function () { setInfo(!infoModal.classList.contains('wq-open')); });
    });
    mapEl.querySelectorAll('[data-wq-modal-close]').forEach(function (btn) {
        btn.addEventListener('click', function () { setInfo(false); });
    });
    /* Backdrop only - a click that started on the card must not dismiss it,
       which is why this tests the target rather than using a bubbled click. */
    infoModal.addEventListener('click', function (e) { if (e.target === infoModal) setInfo(false); });
    var onEsc = function (e) { if (e.key === 'Escape') setInfo(false); };
    document.addEventListener('keydown', onEsc);
    /* A keydown only reaches the document that has focus, and the reader may
       well have last clicked the sidebar rather than the map - in which case
       this iframe never sees the key at all. The parent page is same-origin
       (srcdoc), so listen there too; guarded because that stops being true if
       the map is ever embedded cross-origin. */
    try { window.parent.document.addEventListener('keydown', onEsc); } catch (e) {}
}

/* These overlays are children of the Leaflet container, so without this a
   wheel scroll inside the modal or a rail panel zooms the map underneath it,
   and a drag across one pans the map. */
if (window.L && L.DomEvent) {
    mapEl.querySelectorAll('.wq-modal, .wq-panel, .wq-rail, .wq-info-fab').forEach(function (el) {
        L.DomEvent.disableClickPropagation(el);
        L.DomEvent.disableScrollPropagation(el);
    });
}
"""


def _basemap_row(name, active):
    cls = " wq-active-row" if active else ""
    return (
        f'<div class="wq-row{cls}" data-wq-basemap="{name}">'
        f'<span class="wq-thumb" style="background:{BASEMAP_SWATCHES[name]}"></span>'
        f'<span><b>{name}</b><small>{BASEMAP_DESCRIPTIONS[name]}</small></span></div>'
    )


def _overlay_button(key, label, default_on, title=""):
    color, icon_html = _OVERLAY_STYLE[key]
    cls = " wq-on" if default_on else ""
    title_attr = f' title="{title}"' if title else ""
    return (
        f'<button class="wq-icon-btn{cls}" data-wq-overlay="{key}" style="--wq-color:{color}"{title_attr}>'
        f'<span class="wq-icon-circle">{icon_html}</span>'
        f'<span class="wq-icon-label">{label}</span></button>'
    )


def add_layer_rail(fmap, basemap_layers, default_basemap, overlay_defs, legend_html,
                    legend_label="Legend", basemap_label="Base Map",
                    font_stack="'Poppins', 'Noto Sans Thai', sans-serif",
                    info_html=None, info_label="Information"):
    """Attach the pill-shaped icon rail (top-right) to a folium map.

    basemap_layers: {display_name: folium.TileLayer}, already added to fmap.
    default_basemap: name of the initially-active entry in basemap_layers.
    overlay_defs: list of dicts, each {"key", "label", "layer", "default_on",
        "title" (optional tooltip)}, where "key" is one of _OVERLAY_STYLE's
        keys and "layer" is a folium object already added to fmap.
    legend_html: inner HTML for the Legend fly-out panel (color/shape key -
        built by the caller, which owns the actual style constants).
    legend_label/basemap_label: current-language labels for those two rail
        buttons (overlay_defs already carries its own labels per-entry).
    info_html: inner HTML for the Information modal (data sources, caveats -
        again built by the caller, which knows what the app actually shows).
        Omit it and no Information button is added at all.
    info_label: current-language label for that button and the modal heading.
    font_stack: CSS font-family for this iframe's own chrome. Passed in
        because the caller owns the language, and the stack is ordered by it -
        this stylesheet lives inside the map iframe and inherits nothing from
        the page. Substituted rather than formatted: the CSS below is full of
        literal braces that str.format would choke on.
    """
    rail_html = (
        _CSS.replace("__WQ_FONT__", font_stack)
        + '<div class="wq-rail">'
        + f'<button class="wq-icon-btn wq-on" data-wq-toggle="legend" style="--wq-color:{_LEGEND_COLOR}">'
        + f'<span class="wq-icon-circle">{_LEGEND_ICON}</span><span class="wq-icon-label">{legend_label}</span></button>'
        + "".join(
            _overlay_button(o["key"], o["label"], o["default_on"], o.get("title", ""))
            for o in overlay_defs
        )
        + f'<button class="wq-icon-btn wq-on" data-wq-toggle="basemap" style="--wq-color:{_BASEMAP_COLOR}">'
        + f'<span class="wq-icon-circle">{_BASEMAP_ICON}</span><span class="wq-icon-label">{basemap_label}</span></button>'
        + '</div>'
        # Not a rail entry: it controls nothing on the map, it explains what
        # is already on it. Sits on the bottom-left stack above the zoom
        # buttons instead (see .wq-info-fab).
        + (f'<button class="wq-info-fab" data-wq-modal-open="info" '
           f'title="{info_label}" aria-label="{info_label}">{_INFO_ICON}</button>'
           if info_html else '')
        + '<div class="wq-panel" data-wq-panel="legend">'
        + f'<div class="wq-panel-head">{_LEGEND_ICON}{legend_label}</div>'
        + legend_html
        + '</div>'
        + '<div class="wq-panel" data-wq-panel="basemap">'
        + f'<div class="wq-panel-head">{_BASEMAP_ICON}{basemap_label}</div>'
        + "".join(_basemap_row(name, name == default_basemap) for name in basemap_layers)
        + '</div>'
        + (('<div class="wq-modal"><div class="wq-modal-card">'
            '<div class="wq-modal-head">'
            f'<span class="wq-modal-badge">{_INFO_BADGE}</span>'
            f'<span class="wq-modal-title">{info_label}</span>'
            f'<button class="wq-modal-close" data-wq-modal-close aria-label="Close">&times;</button>'
            '</div>'
            f'<div class="wq-modal-body">{info_html}</div>'
            '</div></div>') if info_html else '')
    )

    basemap_js_map = ",".join(f'"{name}":{layer.get_name()}' for name, layer in basemap_layers.items())
    overlay_js_map = ",".join(f'"{o["key"]}":{o["layer"].get_name()}' for o in overlay_defs)

    setup_js = (
        "var map = " + fmap.get_name() + ";\n"
        "var mapEl = document.getElementById('" + fmap.get_name() + "');\n"
        "mapEl.insertAdjacentHTML('beforeend', " + json.dumps(rail_html) + ");\n"
        "var basemaps = {" + basemap_js_map + "};\n"
        "var overlays = {" + overlay_js_map + "};\n"
        "var currentBase = " + json.dumps(default_basemap) + ";\n"
    )
    js = "(function () {\n" + setup_js + _BEHAVIOR_JS + "\n})();"
    # Must be a child of the Map itself, not fmap.get_root(): streamlit_folium
    # regenerates the page's JS by walking the Map object's own child tree
    # (see generate_leaflet_string in streamlit_folium), not the Figure's
    # header/html/script collections - a script added to get_root() is
    # silently dropped.
    _RawScript(js).add_to(fmap)


def add_pixel_readout(fmap, value_png, bounds, config, district_layer=None):
    """Tap anywhere on the raster and get that cell's reading, with no rerun.

    Everything needed is already in the browser, so nothing goes back to
    Python:

      - the VALUE of the cell comes from `value_png`, a hidden image sampled
        with a canvas (see dashboard.value_png_data_uri for the encoding and
        for why this is not a polygon layer),
      - the DISTRICT comes from the boundary layer already drawn on the map -
        the geometry is there, it just was not being asked this question.

    The alternative was st_folium's click return, which is a full Streamlit
    rerun: measured at 6.5s per tap once the double-rerun was fixed, because
    a rerun re-serialises every layer including 4,661 water polygons.

    `bounds` is the raster's own extent and the sampling is linear in
    latitude, matching how the array is indexed in Python - NOT the Web
    Mercator remap the colour overlay gets. That remap exists to fix where
    the image is DRAWN; the values are still stored on the plain lat/lon grid.
    """
    cfg = dict(config)
    cfg.update({
        "west": bounds.left, "east": bounds.right,
        "south": bounds.bottom, "north": bounds.top,
    })
    js = f"""
(function () {{
  var map = {fmap.get_name()};
  var CFG = {json.dumps(cfg, ensure_ascii=False)};
  var districts = {district_layer.get_name() if district_layer else "null"};

  // Sample the hidden value image through a canvas. Drawn once, at natural
  // size, so a tap is one getImageData of a single pixel.
  var canvas = document.createElement('canvas'), ctx = null;
  var img = new Image();
  img.onload = function () {{
    canvas.width = img.naturalWidth; canvas.height = img.naturalHeight;
    ctx = canvas.getContext('2d', {{willReadFrequently: true}});
    ctx.drawImage(img, 0, 0);
  }};
  img.src = {json.dumps(value_png)};

  function ring(pt, coords) {{
    // Ray casting. coords is one linear ring as [lon, lat] pairs.
    var inside = false;
    for (var i = 0, j = coords.length - 1; i < coords.length; j = i++) {{
      var xi = coords[i][0], yi = coords[i][1];
      var xj = coords[j][0], yj = coords[j][1];
      if (((yi > pt.lat) !== (yj > pt.lat)) &&
          (pt.lng < (xj - xi) * (pt.lat - yi) / (yj - yi) + xi)) inside = !inside;
    }}
    return inside;
  }}

  function inGeometry(pt, geom) {{
    // Outer ring counts, holes subtract - a lake-shaped hole in an amphoe
    // should not report that amphoe.
    var polys = geom.type === 'Polygon' ? [geom.coordinates] : geom.coordinates;
    for (var p = 0; p < polys.length; p++) {{
      if (!ring(pt, polys[p][0])) continue;
      var hole = false;
      for (var h = 1; h < polys[p].length; h++) {{
        if (ring(pt, polys[p][h])) {{ hole = true; break; }}
      }}
      if (!hole) return true;
    }}
    return false;
  }}

  function findPlace(latlng) {{
    if (!districts) return null;
    var hit = null;
    // Bounding box first: 930 amphoe, some with thousands of vertices, so the
    // exact test only runs on the handful whose box contains the point.
    (function walk(layer) {{
      if (hit) return;
      if (layer.eachLayer) layer.eachLayer(walk);
      if (!layer.feature || !layer.getBounds) return;
      if (!layer.getBounds().contains(latlng)) return;
      if (inGeometry(latlng, layer.feature.geometry)) hit = layer.feature.properties;
    }})(districts);
    return hit;
  }}

  function classify(v) {{
    for (var i = 0; i < CFG.classes.length; i++) {{
      if (v <= CFG.classes[i].max) return CFG.classes[i];
    }}
    return CFG.classes[CFG.classes.length - 1];
  }}

  function esc(s) {{
    return String(s).replace(/[&<>"]/g, function (c) {{
      return {{'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}}[c];
    }});
  }}

  map.on('click', function (e) {{
    if (!ctx) return;
    var lat = e.latlng.lat, lng = e.latlng.lng;
    if (lat < CFG.south || lat > CFG.north || lng < CFG.west || lng > CFG.east) return;
    var col = Math.floor((lng - CFG.west) / (CFG.east - CFG.west) * canvas.width);
    var row = Math.floor((CFG.north - lat) / (CFG.north - CFG.south) * canvas.height);
    col = Math.min(Math.max(col, 0), canvas.width - 1);
    row = Math.min(Math.max(row, 0), canvas.height - 1);
    var px = ctx.getImageData(col, row, 1, 1).data;   // [grey, grey, grey, alpha]

    var place = findPlace(e.latlng);
    var rows = '';
    if (place) {{
      if (place[CFG.districtField]) rows += '<div class="wq-pop-row"><b>' +
        CFG.districtLabel + ':</b> ' + esc(place[CFG.districtField]) + '</div>';
      if (place[CFG.provinceField]) rows += '<div class="wq-pop-row"><b>' +
        CFG.provinceLabel + ':</b> ' + esc(place[CFG.provinceField]) + '</div>';
    }}
    var placeBlock = rows ? '<div class="wq-pop-group">' + rows + '</div>' : '';

    var body;
    if (px[3] > 0) {{
      var frac = px[0] / 255;
      var v = frac * frac * CFG.maxNtu;          // inverse of the sqrt encoding
      var c = classify(v);
      // One statement, not two. The value and its class were separate rows
      // saying the same number twice; the pill carries both.
      body = '<div class="wq-pop-group">' +
        '<div class="wq-pop-row"><b>' + CFG.predictedLabel + ':</b></div>' +
        '<div class="wq-pop-pill-row">' +
        '<span class="wq-pop-pill" style="background:' + c.color + '">' +
        v.toFixed(1) + ' NTU &middot; ' + esc(c.label) + '</span></div></div>';
    }} else {{
      body = '<div class="wq-pop-group"><div class="wq-pop-row">' +
        CFG.noWater + '</div></div>';
    }}

    L.popup({{maxWidth: 300, minWidth: 230}})
      .setLatLng(e.latlng)
      .setContent(placeBlock + body +
                  '<div class="wq-pop-note">' + CFG.note + '<br>' +
                  lat.toFixed(5) + ', ' + lng.toFixed(5) + '</div>')
      .openOn(map);
  }});
}})();
"""
    _RawScript(js).add_to(fmap)


def add_zoom_control(fmap):
    """Leaflet's own +/- zoom control, positioned bottom-left (see the
    .leaflet-bottom.leaflet-left override in _CSS) rather than Leaflet's
    default top-left corner. Folium's own zoom_control=True option can't be
    repositioned, so this adds it directly via the Leaflet JS API instead."""
    map_var = fmap.get_name()
    js = f"(function () {{ L.control.zoom({{position: 'bottomleft'}}).addTo({map_var}); }})();"
    _RawScript(js).add_to(fmap)


def declutter_labels(fmap, selector=".wq-dlabel", pad=3):
    """Hide map text labels that would overlap each other at the current zoom.

    Zoom out and the districts converge on screen while their names stay the
    same pixel size, so the cluster around Ubon city turns into overlapping
    text. Leaflet has no built-in collision handling for markers.

    Greedy, in the order the caller ranked them (data-wq-rank ascending, 0
    kept first): keep a label if its box misses everything kept so far,
    otherwise hide it. Whatever survives is guaranteed non-overlapping, and
    because the ranking is stable a label never flickers between two zooms
    that resolve the same way.

    visibility rather than display: a hidden element keeps its box, so the
    next pass can measure it without having to show it first and reflow.

    Runs on zoomend, not moveend - all labels are in one projected plane, so
    panning moves them together and cannot change which ones collide. It also
    runs on layeradd, because toggling the layer off and on rebuilds the
    label elements with their inline visibility gone.
    """
    map_var = fmap.get_name()
    js = f"""
    (function () {{
        var map = {map_var};
        var SEL = {json.dumps(selector)}, PAD = {json.dumps(pad)};
        var pending = null;

        function declutter() {{
            var el = map.getContainer();
            var labels = Array.prototype.slice.call(el.querySelectorAll(SEL));
            if (!labels.length) return;
            labels.forEach(function (l) {{ l.style.visibility = ''; }});
            labels.sort(function (a, b) {{
                return (+a.dataset.wqRank || 0) - (+b.dataset.wqRank || 0);
            }});
            var kept = [];
            labels.forEach(function (l) {{
                var r = l.getBoundingClientRect();
                var hit = kept.some(function (k) {{
                    return r.left - PAD < k.right && r.right + PAD > k.left &&
                           r.top - PAD < k.bottom && r.bottom + PAD > k.top;
                }});
                if (hit) {{ l.style.visibility = 'hidden'; }} else {{ kept.push(r); }}
            }});
        }}

        function schedule() {{
            // Coalesce the burst of layeradd events a rerun fires, and let
            // Leaflet finish placing markers before anything is measured.
            if (pending) {{ clearTimeout(pending); }}
            pending = setTimeout(declutter, 60);
        }}

        map.on('zoomend', schedule);
        map.on('layeradd', schedule);
        map.whenReady(schedule);
    }})();
    """
    _RawScript(js).add_to(fmap)


def compact_attribution(fmap):
    """Trim the map credit to just what the tile data's licence requires.

    The OpenStreetMap credit itself has to stay: OSM data is ODbL-licensed
    and attribution is a condition of use, not a default we can switch off.
    Leaflet's own "Leaflet" prefix is a courtesy link with no such condition,
    so it goes, along with the separator it brought with it. What's left is
    styled down in _CSS to a small, quiet line instead of the default
    high-contrast bar that was colliding with the timeline on a phone.
    """
    map_var = fmap.get_name()
    js = (f"(function () {{ if ({map_var}.attributionControl) "
          f"{{ {map_var}.attributionControl.setPrefix(''); }} }})();")
    _RawScript(js).add_to(fmap)


def add_view_persistence(fmap, default_bounds):
    """Keep the map's pan/zoom stable across Streamlit reruns (picking a new
    date shouldn't reset the view), entirely client-side via localStorage -
    NOT by feeding center/zoom back through st_folium's return value. That
    was tried first and worked, but st_folium triggers a full Streamlit
    rerun every time its return value changes, so every single pan/zoom tick
    re-ran the whole script - the page visibly "refreshed" while exploring
    the map. Saving to localStorage on moveend/zoomend and reading it back
    on init is pure client-side JS: no rerun, ever, for panning or zooming.

    default_bounds: [[south, west], [north, east]] to fit to on the very
    first-ever visit (nothing saved in localStorage yet) or in a fresh
    browser profile. fitBounds is deferred a tick so it measures the map's
    real, CSS-settled container size instead of - if called synchronously
    at map-init time, before the page's own CSS has applied - a stale one.
    """
    map_var = fmap.get_name()
    js = f"""
    (function () {{
        var map = {map_var};
        var STORAGE_KEY = 'wq_map_view';

        /* st_folium renders the map into a fixed 1400px-tall container (see
           the height= argument at the call site) while CSS clips the iframe
           to the real viewport. Leaflet then centres on the *container's*
           middle - 700px down - which on an 844px phone is below the fold,
           so the province sat off the bottom edge of the screen with Laos
           filling the visible half. Matching the container to the iframe's
           own viewport puts Leaflet's centre back where the user's centre
           is. Re-run on resize/orientation change, since a phone rotation
           changes it and an unresized container reintroduces the offset. */
        function syncHeight() {{
            var el = map.getContainer();
            var h = window.innerHeight;
            var changed = false;
            if (h > 0 && Math.abs(el.clientHeight - h) > 2) {{
                el.style.height = h + 'px';
                changed = true;
            }}
            /* Width too, not just height: the page CSS constrains the iframe
               to its container, and Leaflet keeps serving tiles for the old
               wider box until told otherwise. */
            if (Math.abs(el.clientWidth - window.innerWidth) > 2) {{
                changed = true;
            }}
            if (changed) {{ map.invalidateSize({{animate: false}}); }}
            return changed;
        }}
        syncHeight();
        window.addEventListener('resize', syncHeight);
        window.addEventListener('orientationchange', function () {{
            setTimeout(syncHeight, 250);
        }});

        var saved = null;
        try {{ saved = JSON.parse(localStorage.getItem(STORAGE_KEY)); }} catch (e) {{}}
        if (saved && typeof saved.lat === 'number') {{
            map.setView([saved.lat, saved.lng], saved.zoom, {{animate: false}});
        }} else {{
            setTimeout(function () {{
                syncHeight();
                map.invalidateSize();
                map.fitBounds({json.dumps(default_bounds)}, {{padding: [20, 20]}});
            }}, 200);
        }}
        map.on('moveend zoomend', function () {{
            var c = map.getCenter();
            try {{
                localStorage.setItem(STORAGE_KEY, JSON.stringify({{lat: c.lat, lng: c.lng, zoom: map.getZoom()}}));
            }} catch (e) {{}}
        }});
    }})();
    """
    _RawScript(js).add_to(fmap)
