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
_LEGEND_ICON = (
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="1.8" stroke-linecap="round">'
    '<circle cx="12" cy="12" r="9"/><line x1="12" y1="11" x2="12" y2="16"/>'
    '<circle cx="12" cy="7.6" r="0.6" fill="currentColor"/></svg>'
)
_LEGEND_COLOR = "#5b6b7c"

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
    display:flex; flex-direction:column; align-items:center; gap:6px; background:#fff;
    border-radius:999px; box-shadow:0 2px 14px rgba(0,0,0,0.22); padding:8px 4px;
    font-family:'Poppins',sans-serif; }
.wq-icon-btn { display:flex; flex-direction:column; align-items:center; gap:2px; width:38px;
    border:none; background:none; padding:0; cursor:pointer; }
.wq-icon-circle { width:24px; height:24px; border-radius:50%; display:flex; align-items:center;
    justify-content:center; color:#fff; background:var(--wq-color); transition:background .15s,opacity .15s; }
.wq-icon-circle svg { width:13px; height:13px; }
.wq-icon-btn:not(.wq-on) .wq-icon-circle { background:#c7ccd2; }
.wq-icon-btn.wq-active .wq-icon-circle { box-shadow:0 0 0 2px rgba(30,58,74,0.25); }
.wq-icon-label { font-size:0.56rem; font-weight:600; color:#3a4450; text-align:center; line-height:1.1; }
.wq-icon-btn:not(.wq-on) .wq-icon-label { color:#a9b1ba; }
.wq-panel { position:fixed; top:75px; right:60px; z-index:1000;
    width:230px; background:#fff; border-radius:12px; box-shadow:0 2px 14px rgba(0,0,0,0.2);
    padding:14px; display:none; font-family:'Poppins',sans-serif; max-height:70vh; overflow-y:auto; }
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
.leaflet-control-zoom a { font-family:'Poppins',sans-serif !important; }
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
.leaflet-control-attribution { font-family:'Poppins',sans-serif !important;
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

/* ------------------------------------------------------------- phones ---
   This stylesheet lives inside the map iframe, so it needs its own media
   query - the page's one (see dashboard.py) cannot reach in here. At 390px
   the rail ran 281px down the screen and the zoom control ended up buried
   under the timeline bar, which is much taller on a phone. Panels are also
   pinned to both edges rather than a fixed 230px, which would otherwise
   leave them wider than the gap they open into. */
@media (max-width: 640px) {
  .wq-rail { top:58px; right:8px; padding:5px 3px; gap:3px; }
  .wq-icon-btn { width:31px; }
  .wq-icon-circle { width:21px; height:21px; }
  .wq-icon-circle svg { width:11px; height:11px; }
  .wq-icon-label { font-size:0.47rem; }
  .wq-panel { top:58px; right:46px; left:8px; width:auto; max-height:52vh; padding:11px; }
  .wq-row { font-size:0.78rem; padding:6px 5px; }
  .wq-legend-item { font-size:0.76rem; }
  /* Sits above the sidebar button (page-side, bottom:126px), which in turn
     sits above the two-row timeline bar. Offsets are larger than the page's
     because this anchors to the iframe's own viewport, which starts ~32px
     down the page and runs past its bottom edge. */
  .leaflet-bottom.leaflet-left { bottom:240px !important; left:10px !important; }
  /* Taller timeline bar here (two rows), and the bar itself sits higher to
     clear the Cloud badge, so the credit needs lifting further still. */
  .leaflet-bottom.leaflet-right { bottom:192px !important; }
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
                    legend_label="Legend", basemap_label="Base Map"):
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
    """
    rail_html = (
        _CSS
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
        + '<div class="wq-panel" data-wq-panel="legend">'
        + f'<div class="wq-panel-head">{_LEGEND_ICON}{legend_label}</div>'
        + legend_html
        + '</div>'
        + '<div class="wq-panel" data-wq-panel="basemap">'
        + f'<div class="wq-panel-head">{_BASEMAP_ICON}{basemap_label}</div>'
        + "".join(_basemap_row(name, name == default_basemap) for name in basemap_layers)
        + '</div>'
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


def add_zoom_control(fmap):
    """Leaflet's own +/- zoom control, positioned bottom-left (see the
    .leaflet-bottom.leaflet-left override in _CSS) rather than Leaflet's
    default top-left corner. Folium's own zoom_control=True option can't be
    repositioned, so this adds it directly via the Leaflet JS API instead."""
    map_var = fmap.get_name()
    js = f"(function () {{ L.control.zoom({{position: 'bottomleft'}}).addTo({map_var}); }})();"
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
