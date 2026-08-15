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
_TOUR_ICON = (
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M9.1 9a3 3 0 1 1 4.2 2.8c-.8.4-1.3 1.1-1.3 2v.6"/>'
    '<circle cx="12" cy="18" r="0.9" fill="currentColor" stroke="none"/></svg>'
)
_TOUR_COLOR = "#7a5cc4"

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


def control_icon(key: str) -> tuple[str, str]:
    """(accent colour, SVG glyph) for one control, keyed as the rail keys it.

    Public so the Information modal's how-to list can draw the *same* glyphs
    the rail draws. A second, hand-copied set of icons in the help text would
    be correct exactly until the first time an icon changed, and a help screen
    showing a symbol the map no longer uses is worse than no help screen.
    """
    if key in _OVERLAY_STYLE:
        return _OVERLAY_STYLE[key]
    return {
        "legend": (_LEGEND_COLOR, _LEGEND_ICON),
        "basemap": (_BASEMAP_COLOR, _BASEMAP_ICON),
        "info": (_INFO_COLOR, _INFO_ICON),
    }[key]


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
/* top:107px is a starting value only - fitRailToHeader() below replaces it
   with a figure measured off the real title card as soon as the map is up.
   It is set to the common case rather than something arbitrary so the first
   paint is already right and the rail does not visibly jump. */
.wq-rail { position:fixed; top:107px; right:16px; z-index:1000;
    display:flex; flex-direction:column; align-items:center; gap:6px; background:#fff;
    border-radius:999px; box-shadow:0 2px 14px rgba(0,0,0,0.22); padding:8px 5px;
    font-family:__WQ_FONT__;
    /* The rail grows with the number of layers, and Thai labels wrap to three
       lines, so on a short viewport the last entry used to be clipped by the
       timeline bar with no way to reach it. Bounded and scrollable instead;
       the scrollbar is hidden because a visible one inside a pill reads as
       damage rather than as an affordance. */
    max-height:calc(100vh - 225px); overflow-y:auto; overflow-x:hidden;
    scrollbar-width:none; -ms-overflow-style:none; }
.wq-rail::-webkit-scrollbar { display:none; }
.wq-icon-btn { display:flex; flex-direction:column; align-items:center; gap:2px; width:42px;
    border:none; background:none; padding:0; cursor:pointer; flex-shrink:0; }
.wq-icon-circle { width:27px; height:27px; border-radius:50%; display:flex; align-items:center;
    justify-content:center; color:#fff; background:var(--wq-color); transition:background .15s,opacity .15s; }
.wq-icon-circle svg { width:14px; height:14px; }
.wq-icon-btn:not(.wq-on) .wq-icon-circle { background:#c7ccd2; }
.wq-icon-btn.wq-active .wq-icon-circle { box-shadow:0 0 0 2px rgba(30,58,74,0.25); }
.wq-icon-label { font-size:0.65rem; font-weight:600; color:#3a4450; text-align:center; line-height:1.1; }
.wq-icon-btn:not(.wq-on) .wq-icon-label { color:#a9b1ba; }
/* right:68px = the rail's 52px footprint (42px button + 5px padding each
   side) plus its own 16px inset. Track this whenever the rail is resized, or
   every fly-out opens underneath the rail it belongs to. */
.wq-panel { position:fixed; top:107px; right:68px; z-index:1000;
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
.leaflet-bottom.leaflet-left { position:fixed !important; bottom:184px !important; left:16px !important; }
.leaflet-control-zoom { border-radius:12px !important; overflow:hidden;
    box-shadow:0 2px 14px rgba(0,0,0,0.22) !important; border:none !important; }
/* Sized to match the layer rail's buttons rather than left at Leaflet's
   default 30px: the rail's circles are 25px but each button carries a label
   under it, so the block reads larger, and a bare 25px square beside that
   looks like a lesser class of control. 30px sits between the two. */
.leaflet-control-zoom a { font-family:__WQ_FONT__ !important;
    width:30px !important; height:30px !important; line-height:30px !important;
    font-size:19px !important; }
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
/* The date a ground reading was taken, set beside the value it qualifies.
   Recessive on purpose - it is the caveat on the number, not a second number
   competing with it - but present, because that reading can be months away
   from the date the map is showing. */
.wq-pop-when { color:#9aa3ad; font-size:0.78em; white-space:nowrap; }
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
/* bottom:268px: the zoom pair is 60px tall sitting 194px up (its corner's
   184px plus Leaflet's own 10px control margin), so its top edge is at 254px
   and this clears it by 14px - the same gap the stack had at the old sizes.
   left:25px keeps it centred on that pair, which starts 26px in once the
   same 10px margin is counted. */
.wq-info-fab { position:fixed; left:25px; bottom:268px; z-index:1000;
    width:32px; height:32px; padding:0; border:none; border-radius:50%;
    cursor:pointer; display:flex; align-items:center; justify-content:center;
    background:#fff; color:#12161c; box-shadow:0 2px 14px rgba(0,0,0,0.22);
    transition:background .15s; }
.wq-info-fab svg { width:17px; height:17px; }
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
/* Top padding matches the bottom for the same reason: the floating title card
   sits over the map's upper strip and is outside this iframe, so a card
   centred on the raw viewport rides underneath it. 100px clears the title. */
.wq-modal { position:fixed; inset:0; z-index:2000; display:none;
    align-items:center; justify-content:center; padding:100px 24px 100px 24px;
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
/* ------------------------------------------------------ how-to list ---
   One row per control: the control's own icon on the left, what it does on
   the right. The icons are the rail's real glyphs (see control_icon), drawn
   on the same coloured disc the rail uses, so a reader can match what they
   see here to what they see on the map without translating between two
   visual languages. */
.wq-howto-row { display:flex; align-items:flex-start; gap:11px; font-size:0.84rem;
    line-height:1.55; color:#2b2b3a; padding:9px 0; border-bottom:1px solid #eef1f5; }
.wq-howto-row:last-child { border-bottom:none; }
.wq-howto-icon { flex:0 0 26px; width:26px; height:26px; border-radius:50%;
    display:flex; align-items:center; justify-content:center; color:#fff; margin-top:1px; }
.wq-howto-icon svg { width:15px; height:15px; }
/* Controls whose face is an image or a word rather than a rail glyph - the
   timeline arrows, the panel button, the +/- and the EN/TH pill. Same disc at
   the same size, so the icon column stays a column. */
.wq-howto-icon img { width:16px; height:16px; display:block; }
.wq-howto-icon.wq-howto-word { font-size:0.6rem; font-weight:700; letter-spacing:0.02em; }
.wq-howto-row b { color:#12161c; }
/* The tour button sits at the foot of the rail, under a hairline, because it
   is not a layer toggle like everything above it. */
.wq-tour-btn { border-top:1px solid #e6e9ee; padding-top:9px; margin-top:3px; }

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
/* ------------------------------------------------- short viewports ---
   The rail is eight entries tall and its labels wrap - "สถานีคุณภาพน้ำ" takes
   three lines - so it needs roughly 410px in English and 550px in Thai. Add
   the 75px it starts down and the ~150px of timeline bar it must clear, and a
   viewport under ~780px cannot show all of it. It then scrolls (see the rail's
   max-height), which keeps every button reachable but leaves the last one
   looking cut in half.
   So: tighten first, and only drop the labels when tightening is not enough.
   Height alone is the right key, for the reason given at the 480px rule
   below - the chrome being cleared costs the same whatever the width. */
@media (max-height: 820px) {
  .wq-rail { gap:4px; padding:6px 4px; }
  .wq-icon-btn { width:38px; }
  .wq-icon-circle { width:24px; height:24px; }
  .wq-icon-circle svg { width:13px; height:13px; }
  .wq-icon-label { font-size:0.58rem; }
  .wq-panel { right:62px; }
}
/* Icons only, and only when the labels genuinely cannot fit. This threshold
   was 680px, which was wrong: a laptop at browser zoom lands in the 600-700
   range routinely, and stripping the names there cost far more than the
   height it saved. At the compacted sizes above, eight labelled buttons need
   about 360px, so labels survive down to here. Each button keeps its title
   attribute, so the name is still one hover away. */
@media (max-height: 520px) {
  .wq-icon-label { display:none; }
  .wq-icon-btn { width:24px; }
  .wq-rail { gap:7px; padding:7px 5px; }
  .wq-panel { right:50px; }
}
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

/* The rail has to start below the floating title card, and that card is not in
   this document - it is page chrome drawn over the map (see .page-header) - so
   a fixed offset here is a guess about someone else's box. It was wrong twice:
   once when the title grew into a full-width bar, and again when the 32px gap
   above the map was removed and every top offset in here shifted up with it,
   leaving the rail 2px *under* the header.
   Measure it instead. Same-origin, so the card's real bottom edge is readable;
   subtract this iframe's own top so the result is in local coordinates. Guarded
   because that stops being true if the map is ever embedded cross-origin, in
   which case the CSS value stands. */
function fitRailToHeader() {
    try {
        var pdoc = window.parent.document;
        var header = pdoc.querySelector('.page-header');
        if (!header) return;
        var frame = null, frames = pdoc.querySelectorAll('iframe');
        for (var i = 0; i < frames.length; i++) {
            if (frames[i].contentWindow === window) { frame = frames[i]; break; }
        }
        if (!frame) return;
        var top = Math.round(header.getBoundingClientRect().bottom
                             - frame.getBoundingClientRect().top + 14);
        if (!(top > 0)) return;
        mapEl.querySelectorAll('.wq-rail, .wq-panel').forEach(function (el) {
            el.style.top = top + 'px';
        });
    } catch (e) {}
}
fitRailToHeader();
/* The card's height changes with the interface language and with the viewport
   (it wraps on narrow screens), so this is not a one-time measurement. */
window.addEventListener('resize', fitRailToHeader);
try { window.parent.addEventListener('resize', fitRailToHeader); } catch (e) {}

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
                    info_html=None, info_label="Information", tour_label=None):
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
    tour_label: label for the "?" button that replays the guided tour, added
        at the foot of the rail. Omit it and no such button appears - the tour
        itself is attached separately by add_guided_tour(), and this is only
        the handle that starts it.
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
        # Foot of the rail, below a hairline: it toggles nothing on the map,
        # it walks the reader through what the buttons above it do. Always
        # drawn "on" because there is no off state to be in.
        + (f'<button class="wq-icon-btn wq-on wq-tour-btn" data-wq-tour '
           f'style="--wq-color:{_TOUR_COLOR}" title="{tour_label}" aria-label="{tour_label}">'
           f'<span class="wq-icon-circle">{_TOUR_ICON}</span>'
           f'<span class="wq-icon-label">{tour_label}</span></button>' if tour_label else '')
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


def add_geojson_layer(fmap, target, cache_key, geojson, style, pane=None,
                      tooltip_field=None, focus_field=None, focus_value=None,
                      focus_style=None):
    """Fill `target` with GeoJSON, sending it at most once per session.

    This is a payload fix. Streamlit pushes each rerun down a websocket and
    streamlit_folium re-serialises the whole map every time, so the boundary
    and water geometry - which does not change when the date does - was being
    re-sent on every date change. One date step measured 50.2MB.

    The data is parked on the TOP window, which outlives this iframe:
    Streamlit rebuilds the map frame on each rerun but never reloads the page
    around it. So the caller passes the GeoJSON on the first render of a
    session and None afterwards, and later reruns carry a few hundred bytes of
    JS instead of megabytes of coordinates.

    Not served as a static file and fetched, which was the previous approach:
    that depended on Streamlit's static serving being reachable at a fixed
    /app/static/ path, and on the deployed site it was not - every layer 404'd
    and the map came up empty. Embedding once per session costs the same bytes
    per page load (Chrome refuses to cache those responses anyway: they carry
    no Cache-Control, no Expires and no Last-Modified) while depending on
    nothing outside the page.

    `target` is an empty FeatureGroup already added to the map, so everything
    that needs a layer object - the rail's toggle, the pixel readout's
    district lookup - keeps working against the same handle either way.

    `focus_*` singles out one feature to draw differently AND last, which is
    how Ubon's highlight stays on top of the other 76 provinces' lines.
    Leaflet paints in insertion order, so the split is the ordering.
    """
    tooltip_js = f"""
      layer.eachLayer(function (l) {{
        var v = l.feature && l.feature.properties
              ? l.feature.properties[{json.dumps(tooltip_field)}] : null;
        if (v) l.bindTooltip(String(v), {{sticky: true}});
      }});""" if tooltip_field else ""

    js = f"""
(function () {{
  var target = {target.get_name()};
  var PANE = {json.dumps(pane)};
  var KEY = {json.dumps(cache_key)};
  var INLINE = {json.dumps(geojson) if geojson is not None else "null"};

  var store;
  try {{
    store = (window.top.__wqGeoCache = window.top.__wqGeoCache || {{}});
  }} catch (e) {{
    store = {{}};   // cross-origin parent: this render must carry the data
  }}
  if (INLINE) store[KEY] = INLINE;

  function build(features, css) {{
    var opts = {{style: function () {{ return css; }}}};
    if (PANE) opts.pane = PANE;
    var layer = L.geoJSON({{type: 'FeatureCollection', features: features}}, opts);
    {tooltip_js}
    layer.addTo(target);
  }}

  function render(gj) {{
    var feats = (gj && gj.features) || [];
    var focusField = {json.dumps(focus_field)};
    if (!focusField) {{ build(feats, {json.dumps(style)}); return; }}
    var focus = [], others = [];
    feats.forEach(function (f) {{
      var p = f.properties || {{}};
      (p[focusField] === {json.dumps(focus_value)} ? focus : others).push(f);
    }});
    build(others, {json.dumps(style)});
    build(focus, {json.dumps(focus_style or style)});
  }}

  // Deferred by a tick even when the data is already here. The layer
  // variables this assigns into are declared later in the same generated
  // script, so running inline threw and took every statement after it down
  // with it - the whole map came back empty on the second render.
  setTimeout(function () {{
    if (store[KEY]) render(store[KEY]);
    else console.error('layer missing from cache and not supplied', KEY);
  }}, 0);
}})();
"""
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
  // Inline data: URI, not a fetched URL. The static-file route this used to
  // take 404'd on the deployed site and took the whole readout with it; this
  // image is ~1MB per date and only changes when the date does.
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


_TOUR_CSS = """
/* Effectively the top of the stack. 100000 was not enough: Streamlit's own
   sidebar outranked it, so the guide card was painted underneath the panel and
   its left half became unreadable. Nothing on this page should ever sit over
   the walkthrough. */
.wqt-overlay { position:fixed; inset:0; z-index:2147483000; display:none; font-family:__WQ_FONT__; }
.wqt-overlay.wqt-open { display:block; }
/* Four panels around the target rather than one box with a cut-out: an SVG
   mask or a giant box-shadow both work, but only panels leave the highlighted
   control genuinely uncovered, so it still reads at full contrast. */
.wqt-mask { position:fixed; background:rgba(12,20,28,0.66); }
.wqt-ring { position:fixed; border-radius:12px; pointer-events:none;
    box-shadow:0 0 0 2px #fff, 0 0 0 5px rgba(122,92,196,0.95), 0 0 22px 6px rgba(122,92,196,0.45); }
/* Deliberately loud. It has to win against a dimmed map behind it *and*
   against whatever page chrome it is standing over, so it carries an accent
   edge and a deep shadow rather than sitting flat like the info modal. */
.wqt-card { position:fixed; width:min(330px, calc(100vw - 28px)); background:#fff;
    border-radius:14px; border-top:3px solid #7a5cc4; padding:15px 17px 13px 17px;
    box-shadow:0 22px 60px rgba(0,0,0,0.45), 0 0 0 1px rgba(122,92,196,0.3); }
.wqt-step { font-size:0.66rem; font-weight:700; letter-spacing:0.07em; text-transform:uppercase;
    color:#7a5cc4; margin-bottom:5px; }
.wqt-title { font-size:1.02rem; font-weight:700; color:#12161c; margin-bottom:6px; line-height:1.3;
    padding-right:18px; }
.wqt-body { font-size:0.83rem; line-height:1.55; color:#3a4450; }
.wqt-actions { display:flex; align-items:center; gap:8px; margin-top:14px; }
.wqt-dots { flex:1; font-size:0.72rem; color:#9aa3ad; }
.wqt-btn { border:none; border-radius:999px; padding:7px 15px; font-size:0.8rem; font-weight:600;
    cursor:pointer; font-family:inherit; }
.wqt-next { background:#7a5cc4; color:#fff; }
.wqt-back { background:#eef0f4; color:#3a4450; }
.wqt-back[hidden] { display:none; }
.wqt-skip { position:absolute; top:9px; right:11px; border:none; background:none; cursor:pointer;
    color:#9aa3ad; font-size:1.3rem; line-height:1; padding:2px 5px; }
.wqt-skip:hover { color:#3a4450; }
@media (max-width: 640px) { .wqt-card { width:calc(100vw - 24px); } }
"""

_TOUR_JS = """
var pwin, pdoc;
try { pwin = window.parent; pdoc = pwin.document; } catch (e) { return; }
if (!pdoc || !pdoc.body) return;

if (!pdoc.getElementById('wqt-style')) {
    var st = pdoc.createElement('style');
    st.id = 'wqt-style'; st.textContent = TOUR_CSS;
    pdoc.head.appendChild(st);
}

/* Which iframe am I? Needed because a step targeting a rail button measures
   that button in *this* document's coordinates, while the overlay is painted
   in the parent's - the two differ by wherever the map iframe sits. */
var frameEl = null;
var frames = pdoc.querySelectorAll('iframe');
for (var i = 0; i < frames.length; i++) {
    try { if (frames[i].contentWindow === window) { frameEl = frames[i]; break; } } catch (e) {}
}

/* Streamlit reruns rebuild this iframe but not the parent body, so an overlay
   from the previous run would still be sitting there. Replace, don't append. */
var prev = pdoc.getElementById('wqt-overlay');
if (prev) prev.remove();

var ov = pdoc.createElement('div');
ov.id = 'wqt-overlay';
ov.className = 'wqt-overlay';
ov.innerHTML =
    '<div class="wqt-mask" data-m="t"></div><div class="wqt-mask" data-m="b"></div>' +
    '<div class="wqt-mask" data-m="l"></div><div class="wqt-mask" data-m="r"></div>' +
    '<div class="wqt-ring"></div>' +
    '<div class="wqt-card"><button class="wqt-skip" aria-label="' + UI.close + '">&times;</button>' +
    '<div class="wqt-step"></div><div class="wqt-title"></div><div class="wqt-body"></div>' +
    '<div class="wqt-actions"><span class="wqt-dots"></span>' +
    '<button class="wqt-btn wqt-back"></button>' +
    '<button class="wqt-btn wqt-next"></button></div></div>';
pdoc.body.appendChild(ov);

var masks = {}, ring = ov.querySelector('.wqt-ring'), card = ov.querySelector('.wqt-card');
['t', 'b', 'l', 'r'].forEach(function (k) { masks[k] = ov.querySelector('[data-m="' + k + '"]'); });
var elStep = ov.querySelector('.wqt-step'), elTitle = ov.querySelector('.wqt-title');
var elBody = ov.querySelector('.wqt-body'), elDots = ov.querySelector('.wqt-dots');
var btnBack = ov.querySelector('.wqt-back'), btnNext = ov.querySelector('.wqt-next');

/* The first *visible* match, not simply the first. A step may list several
   selectors because which element exists depends on the state of the page -
   the panel button is one testid when the panel is folded away and another
   when it is open, and the one that does not apply is still in the DOM,
   hidden. Taking the first match blindly would hand back a 0x0 box and the
   step would be dropped as unresolvable. */
function elFor(step) {
    var doc = step.frame ? document : pdoc;
    var found = doc.querySelectorAll(step.sel);
    for (var i = 0; i < found.length; i++) {
        var r = found[i].getBoundingClientRect();
        if (r.width && r.height) return found[i];
    }
    return found.length ? found[0] : null;
}

function rectFor(step) {
    var el = elFor(step);
    if (!el) return null;
    var r = el.getBoundingClientRect();
    if (!r.width || !r.height) return null;
    var dx = 0, dy = 0;
    if (step.frame) {
        if (!frameEl) return null;
        var f = frameEl.getBoundingClientRect();
        dx = f.left; dy = f.top;
    }
    return { left: r.left + dx, top: r.top + dy, width: r.width, height: r.height };
}

function px(el, o) { for (var k in o) { el.style[k] = o[k]; } }

function paint(rc, targetEl) {
    var pad = 7;
    var VW = pwin.innerWidth, VH = pwin.innerHeight;
    var x = Math.max(0, rc.left - pad), y = Math.max(0, rc.top - pad);
    var w = rc.width + pad * 2, h = rc.height + pad * 2;
    px(masks.t, { left: '0px', top: '0px', width: VW + 'px', height: y + 'px' });
    px(masks.b, { left: '0px', top: (y + h) + 'px', width: VW + 'px', height: Math.max(0, VH - y - h) + 'px' });
    px(masks.l, { left: '0px', top: y + 'px', width: x + 'px', height: h + 'px' });
    px(masks.r, { left: (x + w) + 'px', top: y + 'px', width: Math.max(0, VW - x - w) + 'px', height: h + 'px' });
    px(ring, { left: x + 'px', top: y + 'px', width: w + 'px', height: h + 'px' });

    var cw = card.offsetWidth, ch = card.offsetHeight;
    /* Keep the card off the sidebar - a card lying half over it reads as a
       rendering fault, and it paints above the panel now so this is about
       appearance rather than legibility. The step whose subject IS the sidebar
       is exempt, or it would have nowhere to sit.
       Only dodge if the card actually fits in what is left, though. On a phone
       the sidebar covers most of the width, and insisting on clearing it
       pushed the card clean off the right edge of the screen. Staying on
       screen outranks staying off the sidebar. */
    var minX = 8;
    var sb = pdoc.querySelector('[data-testid="stSidebar"]');
    if (sb && !(targetEl && sb.contains(targetEl))) {
        var sr = sb.getBoundingClientRect();
        if (sr.width > 1 && sr.right > minX && (VW - sr.right - 18) >= cw) {
            minX = sr.right + 10;
        }
    }
    var maxX = Math.max(8, VW - cw - 8);
    var cx = Math.min(Math.max(minX, rc.left + rc.width / 2 - cw / 2), maxX);
    cx = Math.max(8, Math.min(cx, maxX));
    var below = y + h + 12, above = y - ch - 12;
    var cy = (below + ch <= VH - 8) ? below : (above >= 8 ? above : Math.max(8, (VH - ch) / 2));
    px(card, { left: cx + 'px', top: cy + 'px' });
}

var live = [], idx = 0;

function render() {
    var s = live[idx], rc = rectFor(s);
    if (!rc) { return stop(); }
    elStep.textContent = UI.step;
    elTitle.textContent = s.title;
    elBody.innerHTML = s.body;
    elDots.textContent = (idx + 1) + ' / ' + live.length;
    btnBack.textContent = UI.back;
    btnBack.hidden = (idx === 0);
    btnNext.textContent = (idx === live.length - 1) ? UI.done : UI.next;
    paint(rc, elFor(s));
}

function start() {
    /* Resolved fresh on every run: which controls exist depends on the
       viewport and on which layers the caller supplied, so a step whose
       target is not on screen is dropped rather than shown pointing at
       nothing. */
    live = STEPS.filter(function (s) { return !!rectFor(s); });
    if (!live.length) return;
    idx = 0;
    ov.classList.add('wqt-open');
    render();
}

/* ...but only once the page has finished arriving, because that snapshot is
   taken exactly once and a target that is merely late looks identical to one
   that is absent.
   This bit them on Streamlit Cloud and nowhere else: the deployed app lays
   out slower than a local run, the button that opens the side panel had not
   been positioned when the tour looked for it, and the live site quietly ran
   12 steps instead of 13. Waiting a fixed longer time would just move the
   race; waiting for the count to stop growing ends it. */
function startWhenSettled(budgetMs) {
    var last = -1, waited = 0, gap = 250;
    (function tick() {
        var n = 0;
        for (var i = 0; i < STEPS.length; i++) { if (rectFor(STEPS[i])) n++; }
        if (n === last || waited >= budgetMs) { start(); return; }
        last = n; waited += gap;
        pwin.setTimeout(tick, gap);
    })();
}

function stop() {
    ov.classList.remove('wqt-open');
}

btnNext.addEventListener('click', function () {
    if (idx >= live.length - 1) { stop(); } else { idx++; render(); }
});
btnBack.addEventListener('click', function () { if (idx > 0) { idx--; render(); } });
ov.querySelector('.wqt-skip').addEventListener('click', stop);
['t', 'b', 'l', 'r'].forEach(function (k) { masks[k].addEventListener('click', stop); });

var onKey = function (e) {
    if (!ov.classList.contains('wqt-open')) return;
    if (e.key === 'Escape') { stop(); }
    else if (e.key === 'ArrowRight') { btnNext.click(); }
    else if (e.key === 'ArrowLeft') { btnBack.click(); }
};
pdoc.addEventListener('keydown', onKey);
document.addEventListener('keydown', onKey);
pwin.addEventListener('resize', function () { if (ov.classList.contains('wqt-open')) render(); });

var trigger = mapEl.querySelector('[data-wq-tour]');
if (trigger) {
    // Settled here too: pressing "?" the moment the page appears hits the
    // same race, and the wait costs one 250ms tick when everything is ready.
    trigger.addEventListener('click', function () { startWhenSettled(1500); });
}

/* Every visit runs it unprompted, and the "?" button replays it on demand.

   "Visit" has to mean a page load rather than a script run, because Streamlit
   rebuilds this iframe on every rerun - once per date change - and a naive
   flag would restart the walkthrough each time the reader moved the timeline.
   The marker therefore lives on the parent window object: created fresh by a
   real load or reload, and surviving every rerun in between. Deliberately not
   localStorage, which would remember across visits and so only ever show it
   once, and not sessionStorage, which survives a reload.

   It is claimed when the tour actually opens, not when this script runs, so
   that a rerun landing during start-up does not consume the one showing. */
try {
    if (pwin.__wqTourShown !== true) {
        pwin.setTimeout(function () {
            if (pwin.__wqTourShown === true) return;
            pwin.__wqTourShown = true;
            startWhenSettled(4000);
        }, 1400);
    }
} catch (e) {}
"""


def add_guided_tour(fmap, steps, ui, font_stack="'Poppins', 'Noto Sans Thai', sans-serif"):
    """Attach the step-by-step walkthrough that the rail's "?" button replays.

    steps: list of {"sel", "frame", "title", "body"}. `sel` is a CSS selector
        and `frame` says which document to run it against - True for controls
        drawn inside the map iframe (the rail, the zoom buttons), False for
        the ones Streamlit renders around it (the timeline, the language
        toggle, the sidebar).
    ui: {"step", "next", "back", "done", "close"} - the walkthrough's own
        chrome, in the caller's language.

    Why the overlay is built in the parent document rather than here: half the
    things worth pointing at are not in this iframe. An overlay rendered
    inside it is clipped to it, so it could never dim the page around the
    timeline bar or put a ring on the EN/TH toggle. The parent is same-origin,
    so the overlay is created there and in-frame targets have this iframe's
    offset added to their coordinates (see rectFor).
    """
    js = (
        "(function () {\n"
        "var mapEl = document.getElementById('" + fmap.get_name() + "');\n"
        "if (!mapEl) return;\n"
        "var STEPS = " + json.dumps(steps) + ";\n"
        "var UI = " + json.dumps(ui) + ";\n"
        "var TOUR_CSS = " + json.dumps(_TOUR_CSS.replace("__WQ_FONT__", font_stack)) + ";\n"
        + _TOUR_JS +
        "\n})();"
    )
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
