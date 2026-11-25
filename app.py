"""
GeoChange AI - interactive LULC classification & change-detection workspace.
A proposal by Coarchitive Pvt. Ltd. for NESFIC 2026 (NESFIC-D-21, ASSAC / DST Assam).

Run locally:   streamlit run app.py
"""

import hashlib
import io
import json
import os
import uuid
import zipfile
from datetime import date
from pathlib import Path

import folium
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import Draw, Fullscreen, MousePosition
from streamlit_folium import st_folium

from config import (AOI_BBOX, BHUVAN_LULC_LAYER, BHUVAN_WMS_URL, LULC_CLASSES,
                    MIN_CHANGE_PIXELS, MIN_CONFIDENCE, STATS_DIR)
from modules.models import MODELS
from modules.pipeline import CHANGE_COLORS, list_runs

st.set_page_config(page_title="NESFIC-D-21  LULC change intelligence", page_icon="assets/favicon.png"
                   if Path("assets/favicon.png").exists() else None,
                   layout="wide", initial_sidebar_state="collapsed")

INK, TEAL, GOLD, RAIL = "#13232B", "#0F6B5C", "#B08D57", "#13232B"
# Streamlit Community Cloud gives ~1 GB RAM. Live fetches above this size can crash the app.
# Raise it only if your host has more memory (or set the MAX_AREA_KM2 environment variable).
MAX_AREA_KM2 = int(os.environ.get("MAX_AREA_KM2", 100))
DEMO_NAME = "Guwahati (demo area)"
# Only used to centre the map / fill the coordinate boxes before the user picks an area
PLACEHOLDER_BBOX = [92.65, 26.30, 92.70, 26.35]
MAP_H = 600

# -------------------------------------------------
# STYLE
# -------------------------------------------------

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Manrope:wght@400;500;600;700&display=swap');
html, body, .stApp, p, li, label, input, textarea, select, button, h5, h6,
div[data-testid="stMarkdownContainer"], div[data-baseweb="select"] {{ font-family: 'Manrope', system-ui, sans-serif; }}
/* keep Streamlit's vector icon font intact (fixes text overlapping in buttons) */
span[data-testid="stIconMaterial"], .material-symbols-rounded {{
  font-family: 'Material Symbols Rounded' !important; font-weight: normal !important; letter-spacing: normal !important; }}

/* fit to window - no page scroll */
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main {{ overflow: hidden !important; height: 100vh; }}
.stApp {{ background: #F4F2EE; }}
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display: none !important; }}
.block-container {{ padding: 12px 14px 0 14px !important; max-width: 100% !important; }}
h1, h2, h3, h4 {{ font-family: 'Fraunces', Georgia, serif !important; color: {INK}; font-weight: 600 !important; letter-spacing: -0.01em; }}
p, li, label {{ color: #2B3A42; }}

/* icon rail */
.st-key-rail {{ background:#FFFFFF; border:1px solid #E4DFD6; border-radius:16px; padding:10px 6px;
               height: calc(100vh - 24px); box-shadow: 0 2px 10px rgba(19,35,43,.05); gap: 6px !important; }}
.st-key-rail button {{ background: transparent !important; border: none !important; color:#5D7A80 !important;
                       height: 44px; min-height:44px; border-radius: 12px !important; padding:0 !important; }}
.st-key-rail button p, .st-key-rail span[data-testid="stIconMaterial"] {{ font-size: 21px !important; color: inherit !important; }}
.st-key-rail button:hover {{ background:#EAF3F0 !important; color:{TEAL} !important; }}
.st-key-rail button[kind="primary"] {{ background:{TEAL} !important; color:#FFFFFF !important;
                                       box-shadow: 0 4px 12px rgba(15,107,92,.28); }}
.gc-code {{ font-size:10.5px; font-weight:700; letter-spacing:.12em; color:{TEAL}; text-transform:uppercase;
           margin: 0 0 6px 0; }}

/* work panel (scrolls on its own) */
.st-key-panel {{ background:#FFFFFF; border:1px solid #E4DFD6; border-radius:16px; padding:16px 16px 12px 16px;
                min-height: calc(100vh - 24px); box-shadow: 0 2px 10px rgba(19,35,43,.05); }}
                [data-testid="stColumn"]:has(.st-key-panel), [data-testid="column"]:has(.st-key-panel) {{
                    height: calc(100vh - 24px); overflow-y: auto; overflow-x: hidden; padding-right: 4px;
                    scrollbar-width: thin; scrollbar-color: #D9D2C5 transparent; }}
                [data-testid="stColumn"]:has(.st-key-panel)::-webkit-scrollbar {{ width: 6px; }}
                [data-testid="stColumn"]:has(.st-key-panel)::-webkit-scrollbar-thumb {{ background:#D9D2C5; border-radius:6px; }}
.st-key-panel::-webkit-scrollbar {{ width: 6px; }}
.st-key-panel::-webkit-scrollbar-thumb {{ background:#D9D2C5; border-radius:6px; }}
.gc-ph {{ font-family:'Fraunces',serif; font-size:22px; font-weight:600; color:{INK}; margin:0 0 2px 0; }}
.gc-pd {{ font-size:13px; color:#62717A; margin:0 0 12px 0; line-height:1.45; }}
.gc-step {{ font-size:10.5px; font-weight:700; letter-spacing:0.1em; color:{GOLD}; text-transform:uppercase; margin:12px 0 4px 0; }}
.gc-card {{ border:1px solid #ECE7DE; border-radius:12px; padding:10px 12px; margin-bottom:8px; background:#FBFAF7; }}
.gc-card .h {{ font-weight:600; color:{INK}; font-size:13.5px; }}
.gc-card .d {{ font-size:12px; color:#6B7A82; }}
.gc-kpis {{ display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:8px; margin:4px 0 10px 0; }}
.gc-kpi {{ border:1px solid #ECE7DE; border-radius:12px; padding:10px 12px; background:#FBFAF7; }}
.gc-kpi .v {{ font-family:'Fraunces',serif; font-size:23px; font-weight:600; color:{INK}; line-height:1.05; }}
.gc-kpi .l {{ font-size:11.5px; color:#6B7A82; margin-top:3px; }}
.gc-leg {{ display:flex; flex-wrap:wrap; gap:6px 14px; font-size:12px; color:#3A4A52; margin:4px 0 8px 0; }}
.gc-leg i {{ width:11px; height:11px; border-radius:3px; display:inline-block; margin-right:6px; vertical-align:-1px; }}
.gc-status {{ display:flex; justify-content:space-between; font-size:11.5px; color:#6B7A82; padding:0px 4px ; margin-top:-14px; }}
.gc-status b {{ color:{INK}; font-weight:600; }}
.gc-empty {{ border:1px dashed #D8D1C4; border-radius:12px; padding:14px; font-size:13px; color:#6B7A82; background:#FBFAF7; }}

/* map fills the window */
.st-key-mapbox iframe {{ height: calc(100vh - 64px) !important; border-radius: 16px;
                         box-shadow: 0 2px 10px rgba(19,35,43,.08); }}
.st-key-mapbox {{ position: relative; }}
.st-key-mapbox [data-stale="true"], .st-key-mapbox .stale-element {{
    opacity: 0.35 !important; filter: grayscale(0.6); pointer-events: none; transition: opacity .3s; }}
.st-key-mapbox:has([data-stale="true"])::after, .st-key-mapbox:has(.stale-element)::after {{
    content: "Fetching and processing satellite data…"; position: absolute; top: 45%; left: 50%;
    transform: translate(-50%, -50%); z-index: 50; background: #13232B; color: #FFFFFF;
    font: 600 14px Manrope, sans-serif; padding: 12px 20px; border-radius: 12px;
    box-shadow: 0 8px 24px rgba(0,0,0,.25); }}
/* widgets: soft edges */
.stButton > button, .stDownloadButton > button {{ border-radius:10px !important; font-weight:600; }}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {{ background:{TEAL}; border-color:{TEAL}; color:#fff; }}
div[data-baseweb="select"] > div, div[data-baseweb="input"], .stNumberInput input, .stDateInput input {{ border-radius:10px !important; }}
div[data-testid="stTabs"] button p {{ font-weight:600; font-size:13px; }}
[data-testid="stExpander"], [data-testid="stStatusWidget"] {{ border-radius:12px !important; }}

/* all hover text: white on ink */
div[data-testid="stTooltipContent"], div[data-baseweb="tooltip"], div[data-baseweb="tooltip"] > div {{
  background:#13232B !important; color:#FFFFFF !important; border-radius:10px !important; }}
div[data-testid="stTooltipContent"] *, div[data-baseweb="tooltip"] * {{ color:#FFFFFF !important; font-size:12.5px; }}
[data-tip] {{ position: relative; }}
[data-tip]:hover::after {{ content: attr(data-tip); position:absolute; left:0; top:calc(100% + 6px); z-index:9999;
  background:#13232B; color:#FFFFFF; font: 500 12px/1.4 Manrope, sans-serif; padding:7px 10px; border-radius:10px;
  width:max-content; max-width:260px; white-space:normal; box-shadow:0 6px 18px rgba(0,0,0,.18); pointer-events:none; }}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# STATE
# -------------------------------------------------

ss = st.session_state
ss.setdefault("tool", "area")
ss.setdefault("sid", uuid.uuid4().hex[:10])
ss.setdefault("bbox", list(PLACEHOLDER_BBOX))
ss.setdefault("aoi_set", False)   # False until the user picks / draws / types an area
ss.setdefault("area_name", "No area selected")
ss.setdefault("fetched", {})        # label -> {"refl","geo","info","layer","period"}
ss.setdefault("run_id", None)
ss.setdefault("basemap", "Satellite")
ss.setdefault("vis", {})            # layer key -> bool
ss.setdefault("last_draw", None)
ss.setdefault("log", [])

TOOLS = [
    ("area", ":material/crop_free:", "Area of interest",
     "Step 1 - Area of interest. Choose a preset, draw a rectangle on the map or type coordinates."),
    ("fetch", ":material/satellite_alt:", "Fetch imagery",
     "Step 2 - Fetch imagery. Pick sensor, years and season; cloud-free composites are added as map layers."),
    ("layers", ":material/layers:", "Layers",
     "Layers - basemaps, fetched imagery and AI outputs. Switch layers on or off and read the legend."),
    ("classify", ":material/model_training:", "Classify",
     "Step 3 - Classify. Choose an AI model; both periods are classified and change is detected."),
    ("results", ":material/monitoring:", "Results",
     "Step 4 - Results. Area statistics, cross-matrix, significant change and the priority list."),
    ("report", ":material/description:", "Report",
     "Report and downloads - PDF report, statistics tables and GeoTIFF layers."),
    ("scale", ":material/grid_on:", "Scale-up",
     "Scale-up - Assam processing grid (one tile per Survey of India 1:50k sheet)."),
    ("about", ":material/info:", "Method",
     "About the method - how the platform automates interpretation."),
]

SEASONS = {
    "Dry season (Jan–Mar)": ("01-01", "03-31"),
    "Post-monsoon (Oct–Dec)": ("10-01", "12-31"),
    "Pre-monsoon (Mar–May)": ("03-01", "05-31"),
    "Full year": ("01-01", "12-31"),
    "Custom dates": None,
}
PRESETS = {
    "Guwahati (demo area)": list(AOI_BBOX),
    "Guwahati city core": [91.70, 26.11, 91.82, 26.20],
    "Deepor Beel wetland": [91.60, 26.09, 91.70, 26.15],
    "Custom (draw or type)": None,
}
BASEMAPS = {
    "Streets": ("OpenStreetMap", None),
    "Satellite": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                  "Esri World Imagery"),
    "Dark": ("CartoDB dark_matter", None),
    "Light": ("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
              "© OpenStreetMap contributors © CARTO"),
}


def area_km2(b):
    import math
    w, s, e, n = b
    return abs(e - w) * 111.32 * math.cos(math.radians((s + n) / 2)) * abs(n - s) * 110.57


def is_demo_area():
    return bool(ss.aoi_set) and [round(x, 4) for x in ss.bbox] == [round(x, 4) for x in AOI_BBOX]


def current_run():
    if not ss.run_id:
        return None
    for r in list_runs():
        if r["run_id"] == ss.run_id:
            return r
    return None


def legend(items):
    return '<div class="gc-leg">' + "".join(
        f'<span data-tip="{t}"><i style="background:{c}"></i>{n}</span>' for n, c, t in items) + "</div>"


CLASS_HELP = {
    "Built-up": "Buildings, roads, paved and impervious surfaces.",
    "Vegetation": "Forest, tree cover and scrub on hills and plains.",
    "Agriculture & open land": "Cropland, fallow, bare soil and exposed sandbars (char).",
    "Water": "Rivers, wetlands (beels), ponds and reservoirs.",
}


@st.cache_resource(show_spinner=False)
def get_model(key):
    from modules.models import train
    return train(key)


def panel_head(title, desc):
    st.markdown(f'<div class="gc-ph">{title}</div><div class="gc-pd">{desc}</div>', unsafe_allow_html=True)


# -------------------------------------------------
# ANALYSIS (inside the Results panel)
# -------------------------------------------------

def render_analysis(run):
    rd = Path(run["path"])
    a, b = run["labels"][0], run["labels"][-1]
    tag = f"{a}_{b}"
    stats = rd / "stats"
    cc = pd.read_csv(stats / f"class_change_{tag}.csv")
    tm = pd.read_csv(stats / f"transition_matrix_{tag}.csv", index_col=0)
    prp = stats / f"priority_changes_{tag}.csv"
    pr = pd.read_csv(prp) if prp.exists() else pd.DataFrame()
    font = dict(family="Manrope, sans-serif", color="#2B3A42", size=12)
    colors = [v["color"] for v in LULC_CLASSES.values()]

    t1, t2, t3, t4 = st.tabs([":material/bar_chart: Area statistics", ":material/grid_view: Cross-matrix",
                              ":material/flag: Priority list", ":material/verified: Model accuracy"])
    with t1:
        c1, c2 = st.columns(2)
        fig = go.Figure()
        for lab, op in ((a, 0.45), (b, 1.0)):
            fig.add_bar(name=lab, x=cc["class_name"], y=cc[f"area_{lab}_km2"], marker_color=colors,
                        marker_opacity=op, hovertemplate="%{x}<br>%{y:.1f} km²<extra>" + lab + "</extra>")
        fig.update_layout(title=f"Class area, {a} (light) and {b} (solid), km²", barmode="group",
                          height=300, font=font, plot_bgcolor="#fff", paper_bgcolor="rgba(0,0,0,0)",
                          showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
        c1.plotly_chart(fig, use_container_width=True)
        fig2 = go.Figure()
        fig2.add_bar(name="Gain", y=cc["class_name"], x=cc["gain_km2"], orientation="h", marker_color=TEAL,
                     hovertemplate="%{y}<br>gain %{x:.2f} km²<extra></extra>")
        fig2.add_bar(name="Loss", y=cc["class_name"], x=-cc["loss_km2"], orientation="h", marker_color="#B42318",
                     hovertemplate="%{y}<br>loss %{x:.2f} km²<extra></extra>")
        fig2.update_layout(title="Gains and losses, km²", barmode="relative", height=300, font=font,
                           plot_bgcolor="#fff", paper_bgcolor="rgba(0,0,0,0)",
                           legend=dict(orientation="h", y=1.12), margin=dict(l=10, r=10, t=50, b=10))
        c2.plotly_chart(fig2, use_container_width=True)
        show = cc.drop(columns="class_id").rename(columns={
            "class_name": "Class", f"area_{a}_km2": f"{a} km²", f"area_{b}_km2": f"{b} km²",
            "gain_km2": "Gain km²", "loss_km2": "Loss km²", "net_km2": "Net km²", "net_percent": "Net %",
            "significant_gain_km2": "Significant gain", "significant_loss_km2": "Significant loss"})
        st.dataframe(show, hide_index=True, use_container_width=True)
    with t2:
        fig = go.Figure(go.Heatmap(
            z=tm.values, x=[f"{c} ({b})" for c in tm.columns], y=[f"{r} ({a})" for r in tm.index],
            colorscale=[[0, "#F7F5F0"], [1, TEAL]], text=tm.values.round(2), texttemplate="%{text}",
            hovertemplate="%{y} → %{x}<br>%{z:.2f} km²<extra></extra>", showscale=False))
        fig.update_layout(title=f"Cross-matrix: area moving between classes, km² (rows {a}, columns {b})",
                          height=340, font=font, yaxis=dict(autorange="reversed"),
                          paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Diagonal = unchanged area; off-diagonal = transitions.")
    with t3:
        if pr.empty:
            st.info("No significant change patches.")
        else:
            c1, c2, c3 = st.columns(3)
            lvl = c1.multiselect("Priority", ["High", "Medium", "Low"], ["High", "Medium"],
                                 help="Priority = transition importance × patch size × AI confidence.")
            ctx = c2.multiselect("Context", sorted(pr["context"].unique()) if "context" in pr else [],
                                 ["Land conversion"] if "context" in pr else [],
                                 help="River channel / char dynamics are separated from land conversion.")
            trn = c3.multiselect("Transition", sorted(pr["transition"].unique()),
                                 help="Filter by type of change.")
            v = pr[pr["priority"].isin(lvl)]
            if ctx and "context" in v:
                v = v[v["context"].isin(ctx)]
            if trn:
                v = v[v["transition"].isin(trn)]
            cols = [c for c in ["rank", "transition", "area_ha", "confidence", "priority", "context", "lat",
                                "lon", "field_verification"] if c in v]
            st.dataframe(v[cols], hide_index=True, use_container_width=True, height=320)
    with t4:
        bench = stats / "algorithm_benchmark.csv"
        bench = bench if bench.exists() else STATS_DIR / "algorithm_benchmark.csv"
        acc = stats / "accuracy_v2.json"
        acc = acc if acc.exists() else STATS_DIR / "accuracy_v2.json"
        c1, c2 = st.columns([1.3, 1])
        if bench.exists():
            bdf = pd.read_csv(bench).sort_values("kappa")
            fig = go.Figure()
            fig.add_bar(y=bdf["algorithm"], x=bdf["overall_accuracy"], orientation="h", name="Overall accuracy",
                        marker_color="#C9BFAE", text=(bdf["overall_accuracy"] * 100).round(1).astype(str) + "%",
                        textposition="outside")
            fig.add_bar(y=bdf["algorithm"], x=bdf["kappa"], orientation="h", name="Kappa", marker_color=TEAL,
                        text=bdf["kappa"].round(2).astype(str), textposition="outside")
            fig.update_layout(title="Algorithm benchmark (polygon-held-out validation)", barmode="group",
                              height=320, font=font, xaxis=dict(range=[0, 1.08]), plot_bgcolor="#fff",
                              paper_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.12),
                              margin=dict(l=10, r=30, t=60, b=10))
            c1.plotly_chart(fig, use_container_width=True)
        if acc.exists():
            mm = json.loads(acc.read_text())
            cm = pd.DataFrame(mm["confusion_matrix"], index=mm["classes"], columns=mm["classes"])
            fig = go.Figure(go.Heatmap(z=cm.values, x=mm["classes"], y=mm["classes"], showscale=False,
                                       colorscale=[[0, "#F7F5F0"], [1, INK]], text=cm.values,
                                       texttemplate="%{text}"))
            fig.update_layout(title="Confusion matrix, production model (rows = reference)", height=320,
                              font=font, yaxis=dict(autorange="reversed"), paper_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=10, r=10, t=60, b=10))
            c2.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------
# LAYOUT
# -------------------------------------------------

wide_panel = ss.tool in ("results", "report")
rail_col, panel_col, map_col = st.columns([0.42, 5.4, 6.6] if wide_panel else [0.42, 3.3, 8.8], gap="small")

with rail_col:
    with st.container(key="rail"):
        for key, icon, name, desc in TOOLS:
            if st.button(icon, key=f"tool_{key}", help=desc, use_container_width=True,
                         type="primary" if ss.tool == key else "secondary"):
                ss.tool = key
                st.rerun()

run = current_run()

with panel_col:
    with st.container(key="panel"):
        st.markdown('<div class="gc-code" data-tip="North East Seva First Innovation Challenge 2026 · ASSAC, DST Assam">NESFIC-D-21 · LULC change intelligence</div>', unsafe_allow_html=True)

        # ---------------- AREA ----------------
        if ss.tool == "area":
            panel_head("Area of interest",
                       "Draw a small area with the pencil tool on the map, pick a preset or type coordinates. "
                       "The Guwahati demo loads from files bundled with the app (no live download).")
            if st.button("Load Guwahati demo", icon=":material/play_circle:", use_container_width=True,
                         help="Opens the precomputed Guwahati 2020 → 2025 sample from local files. "
                              "Nothing is downloaded and no heavy processing runs."):
                ss.bbox, ss.area_name, ss.aoi_set = list(AOI_BBOX), DEMO_NAME, True
                ss.fetched = {}
                if any(r["run_id"] == "demo_guwahati" for r in list_runs()):
                    ss.run_id = "demo_guwahati"
                    ss.vis.update({"lulc_2025": True, "priority": True})
                    ss.tool = "results"
                else:
                    ss.run_id, ss.tool = None, "fetch"
                st.rerun()
            preset = st.selectbox("Preset area", list(PRESETS), index=list(PRESETS).index(
                ss.area_name) if ss.area_name in PRESETS else 3,
                help="Ready-made study areas. Pick 'Custom' to draw a rectangle on the map "
                     "(pencil tool, top left of the map) or type coordinates below.")
            if PRESETS[preset] and (PRESETS[preset] != ss.bbox or not ss.aoi_set):
                ss.bbox, ss.area_name, ss.aoi_set = PRESETS[preset], preset, True
                ss.fetched, ss.run_id = {}, None
                st.rerun()
            st.markdown('<div class="gc-step">Bounding box (WGS84)</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            w = c1.number_input("West", value=float(ss.bbox[0]), format="%.4f", step=0.01,
                                help="Western longitude in decimal degrees.")
            e = c2.number_input("East", value=float(ss.bbox[2]), format="%.4f", step=0.01,
                                help="Eastern longitude in decimal degrees.")
            s = c1.number_input("South", value=float(ss.bbox[1]), format="%.4f", step=0.01,
                                help="Southern latitude in decimal degrees.")
            n = c2.number_input("North", value=float(ss.bbox[3]), format="%.4f", step=0.01,
                                help="Northern latitude in decimal degrees.")
            if st.button("Apply coordinates", icon=":material/check:", use_container_width=True,
                         help="Use the typed bounding box as the area of interest."):
                ss.bbox, ss.area_name, ss.aoi_set = [w, s, e, n], "Custom (draw or type)", True
                ss.fetched, ss.run_id = {}, None
                st.rerun()
            a = area_km2(ss.bbox)
            if ss.aoi_set:
                st.markdown(f'<div class="gc-card" data-tip="Area of the current bounding box"><div class="h">'
                            f'{a:,.0f} km²</div><div class="d">{ss.area_name} · '
                            f'{ss.bbox[0]:.3f}, {ss.bbox[1]:.3f} → {ss.bbox[2]:.3f}, {ss.bbox[3]:.3f}</div></div>',
                            unsafe_allow_html=True)
                if a > MAX_AREA_KM2 and not is_demo_area():
                    st.warning(f"Live runs are limited to {MAX_AREA_KM2} km². Draw a smaller area, or use the "
                               "Guwahati demo. Larger areas run tile-by-tile in the state-wide engine (see Scale-up).")
            else:
                st.markdown('<div class="gc-empty">No area selected yet. Draw one with the pencil tool on the '
                            'map, pick a preset, or load the Guwahati demo.</div>', unsafe_allow_html=True)
            st.caption("Tip: use the pencil tool on the map to draw a new area.")
            if st.button("Next: fetch imagery", icon=":material/arrow_forward:", type="primary",
                         use_container_width=True, disabled=not ss.aoi_set,
                         help="Go to Step 2 - fetch satellite imagery."):
                ss.tool = "fetch"
                st.rerun()

        # ---------------- FETCH ----------------
        elif ss.tool == "fetch":
            panel_head("Fetch imagery",
                       "Cloud-free median composites are built automatically from every clear "
                       "scene in the window you choose.")
            source = st.segmented_control(
                "Sensor", ["sentinel-2", "landsat"], default="sentinel-2",
                format_func=lambda x: "Sentinel-2 · 10 m" if x == "sentinel-2" else "Landsat 8/9 · 30 m",
                help="Sentinel-2: 10 m, from 2017. Landsat 8/9: 30 m resampled to 10 m, from 2013. "
                     "Both are free (Microsoft Planetary Computer).") or "sentinel-2"
            years = list(range(date.today().year, 2012, -1))
            periods = {}
            for tag, default_year, title in (("A", 2020, "Period A · before"), ("B", 2025, "Period B · after")):
                st.markdown(f'<div class="gc-step">{title}</div>', unsafe_allow_html=True)
                c1, c2 = st.columns([1, 1.6])
                y = c1.selectbox("Year", years, index=years.index(default_year), key=f"year{tag}",
                                 help="Observation year.")
                sea = c2.selectbox("Season", list(SEASONS), key=f"season{tag}",
                                   help="Dry season (Jan–Mar) has the clearest skies over Assam and "
                                        "is recommended for change detection.")
                if SEASONS[sea]:
                    a_, b_ = SEASONS[sea]
                    start, end = date.fromisoformat(f"{y}-{a_}"), min(date.fromisoformat(f"{y}-{b_}"), date.today())
                else:
                    d1, d2 = st.columns(2)
                    start = d1.date_input("From", date(y, 1, 1), key=f"from{tag}", help="First acquisition date.")
                    end = d2.date_input("To", date(y, 3, 31), key=f"to{tag}", help="Last acquisition date.")
                periods[str(y)] = (start.isoformat(), end.isoformat())
            st.markdown('<div class="gc-step">Scene filter</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            cloud = c1.slider("Max cloud %", 5, 90, 40, 5,
                              help="Scenes with more cloud than this are skipped. Remaining clouds and "
                                   "shadows are masked pixel by pixel. Use 70-80 for monsoon months.")
            scenes = c2.slider("Scenes", 3, 20, 8,
                               help="Number of least-cloudy scenes combined into each median composite. "
                                    "Fewer scenes = faster download.")
            use_cache = st.toggle("Use bundled demo imagery (offline)", value=is_demo_area(),
                                  help="On: Guwahati 2020/2025 dry-season composites load instantly from the "
                                       "app. Off: always download live from the Sentinel-2 / Landsat archives.")
            demo_offline = use_cache and is_demo_area()
            too_big = area_km2(ss.bbox) > MAX_AREA_KM2 and not demo_offline
            ok = ss.aoi_set and len(periods) == 2 and not too_big
            if not ss.aoi_set:
                st.info("Select an area first (step 1).")
            elif too_big:
                st.warning(f"Area is {area_km2(ss.bbox):,.0f} km². Live fetches are limited to "
                           f"{MAX_AREA_KM2} km² on this server. Draw a smaller area.")
            if len(periods) < 2:
                st.warning("Choose two different years.")
            if st.button("Fetch imagery", icon=":material/cloud_download:", type="primary",
                         use_container_width=True, disabled=not ok,
                         help="Search the catalogue, mask clouds, build composites and add them to the map."):
                ss.fetched, ss.run_id = {}, None
                with st.status("Fetching satellite imagery...", expanded=True) as stt:
                    try:
                        from modules.workspace import fetch_period, rgb_layer
                        epsg = None
                        def show(msg):
                            st.write(msg)
                            st.toast(str(msg).replace("**", ""))
                        for lab in sorted(periods):
                            st.write(f"**{lab}** · {periods[lab][0]} → {periods[lab][1]}")
                            refl, geo, info = fetch_period(ss.bbox, *periods[lab], source, cloud, scenes,
                                                           epsg=epsg, log=show, use_cache=use_cache)
                            epsg = geo["epsg"]
                            layer = rgb_layer(refl, geo, lab, ss.sid)
                            ss.fetched[lab] = {"refl": refl, "geo": geo, "info": info, "layer": layer,
                                               "period": periods[lab], "source": source}
                            ss.vis[f"rgb_{lab}"] = True
                        stt.update(label="Imagery added to the map as layers", state="complete")
                        ss.tool = "layers"
                        st.rerun()
                    except Exception as err:  # noqa: BLE001
                        stt.update(label="Fetch failed", state="error")
                        st.error(str(err))
                        st.toast(f"Fetch failed: {err}")
                        import traceback; print(traceback.format_exc())
            if ss.fetched:
                for lab, f in sorted(ss.fetched.items()):
                    i = f["info"]
                    st.markdown(f'<div class="gc-card" data-tip="Composite summary"><div class="h">{lab} · '
                                f'{f["source"].title()}</div><div class="d">{f["period"][0]} → {f["period"][1]} · '
                                f'{i["clear_percent"]}% clear pixels'
                                + (" · bundled demo composite" if i.get("cached") else
                                   f' · {i["scenes_used"]} of {i["scenes_found"]} scenes') + "</div></div>",
                                unsafe_allow_html=True)

        # ---------------- LAYERS ----------------
        elif ss.tool == "layers":
            panel_head("Layers", "Everything on the map. Hover a name for what it shows; the layer "
                                 "switcher on the map toggles them instantly.")
            ss.basemap = st.radio("Basemap", list(BASEMAPS), index=list(BASEMAPS).index(ss.basemap),
                                  horizontal=True, help="Background map. Satellite uses Esri World Imagery.")
            st.markdown('<div class="gc-step">Reference</div>', unsafe_allow_html=True)
            ss.vis["bhuvan"] = st.checkbox("Bhuvan LULC 1:50k (NRSC, 2015-16)", ss.vis.get("bhuvan", False),
                                           help="Official NRSC/ISRO land-use map for Assam, streamed live from "
                                                "Bhuvan when the service is available. Useful to compare with "
                                                "the AI result.")
            st.markdown('<div class="gc-step">Fetched imagery</div>', unsafe_allow_html=True)
            if not ss.fetched:
                st.markdown('<div class="gc-empty">No imagery yet. Use <b>Fetch imagery</b> (step 2).</div>',
                            unsafe_allow_html=True)
            for lab in sorted(ss.fetched):
                k = f"rgb_{lab}"
                ss.vis[k] = st.checkbox(f"True colour composite {lab}", ss.vis.get(k, True),
                                        help=f"Cloud-free median of the {lab} scenes, red-green-blue bands "
                                             "(how the ground looks to the eye).")
            st.markdown('<div class="gc-step">AI outputs</div>', unsafe_allow_html=True)
            if run:
                a, b = run["labels"][0], run["labels"][-1]
                for k, name, desc in (
                        (f"lulc_{a}", f"Land use / land cover {a}", "AI classification of period A."),
                        (f"lulc_{b}", f"Land use / land cover {b}", "AI classification of period B."),
                        ("change", "Significant change", "Only changes confident in both years and larger "
                                                         "than the minimum mapping unit."),
                        ("intensity", "Spectral change intensity", "How strongly the satellite signal changed; "
                                                                   "drives the automatic change threshold."),
                        (f"confidence_{b}", f"AI confidence {b}", "Green = confident, red = uncertain "
                                                                  "(candidate for field verification)."),
                        ("priority", "Priority change markers", "Top 30 change patches ranked for action.")):
                    ss.vis[k] = st.checkbox(name, ss.vis.get(k, k == f"lulc_{b}"), help=desc, key=f"v_{k}")
                st.markdown(legend([(n, v["color"], CLASS_HELP.get(v["name"], "")) for n, v in
                                    ((v["name"], v) for v in LULC_CLASSES.values())]), unsafe_allow_html=True)
            else:
                st.markdown('<div class="gc-empty">Classification layers appear here after '
                            '<b>Classify</b> (step 3).</div>', unsafe_allow_html=True)
            if len(ss.fetched) == 2 and not run:
                if st.button("Next: classify", icon=":material/arrow_forward:", type="primary",
                             use_container_width=True, help="Go to Step 3 - choose a model and classify."):
                    ss.tool = "classify"
                    st.rerun()

        # ---------------- CLASSIFY ----------------
        elif ss.tool == "classify":
            panel_head("Classify",
                       "Choose the AI model. Both periods are classified together so that a label "
                       "can only change where the ground really changed.")
            key = st.radio("Model", list(MODELS), format_func=lambda k: MODELS[k]["label"],
                           captions=[MODELS[k]["short"] for k in MODELS],
                           help="All models use the same 14 features and the same cleaned training "
                                "data; accuracies are from polygon-held-out validation.")
            st.markdown(f'<div class="gc-card" data-tip="About this model"><div class="d">{MODELS[key]["help"]}</div></div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="gc-step">Output classes</div>', unsafe_allow_html=True)
            st.markdown(legend([(v["name"], v["color"], CLASS_HELP.get(v["name"], ""))
                                for v in LULC_CLASSES.values()]), unsafe_allow_html=True)
            st.markdown('<div class="gc-step">Change rules</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="gc-card"><div class="d">Significant change = confidence ≥ '
                        f'{MIN_CONFIDENCE:.0%} in both years and patch ≥ {MIN_CHANGE_PIXELS} pixels '
                        f'({MIN_CHANGE_PIXELS * 100 / 10000:.2f} ha). Threshold for spectral change is '
                        f'set automatically (Otsu).</div></div>', unsafe_allow_html=True)
            ready = len(ss.fetched) == 2
            if not ready:
                st.markdown('<div class="gc-empty">Fetch two periods first (step 2).</div>',
                            unsafe_allow_html=True)
            if st.button("Classify and detect change", icon=":material/play_arrow:", type="primary",
                         use_container_width=True, disabled=not ready,
                         help="Train / load the model, classify both periods, detect and prioritise change."):
                with st.status(f"Running {MODELS[key]['label']}...", expanded=True) as stt:
                    try:
                        from modules.workspace import classify_run
                        st.write("Preparing model...")
                        model = get_model(key)
                        labs = sorted(ss.fetched)
                        f0 = ss.fetched[labs[0]]
                        meta = {"name": ss.area_name, "bbox": ss.bbox, "source": f0["source"],
                                "max_cloud": None,
                                "periods": {l: list(ss.fetched[l]["period"]) for l in labs},
                                "acquisition": {l: ss.fetched[l]["info"] for l in labs}}
                        rd = classify_run(model, MODELS[key]["label"],
                                          {l: ss.fetched[l]["refl"] for l in labs}, f0["geo"], meta,
                                          log=st.write)
                        ss.run_id = Path(rd).name
                        for l in labs:
                            ss.vis[f"rgb_{l}"] = False
                        ss.vis[f"lulc_{labs[-1]}"] = True
                        ss.vis["priority"] = True
                        stt.update(label="Classification and change detection complete", state="complete")
                        ss.tool = "results"
                        st.rerun()
                    except Exception as err:  # noqa: BLE001
                        stt.update(label="Classification failed", state="error")
                        st.error(str(err))

        # ---------------- RESULTS ----------------
        elif ss.tool == "results":
            panel_head("Results", "Headline numbers, area statistics, cross-matrix, priority list and model "
                                  "accuracy for this analysis. Matching layers are on the map.")
            if not run:
                st.markdown('<div class="gc-empty">No results yet. Fetch imagery and classify, '
                            'or open the sample analysis.</div>', unsafe_allow_html=True)
                if any(r["run_id"] == "demo_guwahati" for r in list_runs()):
                    if st.button("Open sample: Guwahati 2020 → 2025", icon=":material/folder_open:",
                                 use_container_width=True,
                                 help="Precomputed result for the Guwahati demo area (gradient-boosted trees)."):
                        ss.run_id = "demo_guwahati"
                        ss.vis.update({"lulc_2025": True, "priority": True})
                        st.rerun()
            else:
                s = run["summary"]
                a, b = run["labels"][0], run["labels"][-1]
                st.markdown(
                    f'<div class="gc-kpis">'
                    f'<div class="gc-kpi" data-tip="Area of the analysis that had valid pixels in both periods">'
                    f'<div class="v">{s["analysed_area_km2"]:.0f}</div><div class="l">km² analysed</div></div>'
                    f'<div class="gc-kpi" data-tip="Area of significant land-cover change"><div class="v">'
                    f'{s["significant_change_km2"]}</div><div class="l">km² significant change '
                    f'({s["significant_change_percent"]}%)</div></div>'
                    f'<div class="gc-kpi" data-tip="Patches rated High priority"><div class="v">'
                    f'{s.get("priority_counts", {}).get("High", 0)}</div><div class="l">high-priority patches</div></div>'
                    f'<div class="gc-kpi" data-tip="Low-confidence change recommended for field verification">'
                    f'<div class="v">{s.get("doubtful_change_km2", 0)}</div><div class="l">km² for field check</div></div>'
                    f'</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="gc-step">{a} → {b} · {run.get("model", "Gradient-boosted trees")}</div>',
                            unsafe_allow_html=True)
                render_analysis(run)

        # ---------------- REPORT ----------------
        elif ss.tool == "report":
            panel_head("Report and downloads", "Everything needed for planning departments and GIS teams.")
            if not run:
                st.markdown('<div class="gc-empty">Run a classification (or open the sample in '
                            'Results) to generate the report.</div>', unsafe_allow_html=True)
            else:
                rd = Path(run["path"])
                pdf = rd / "GeoChange_report.pdf"
                if pdf.exists():
                    st.download_button("PDF report", pdf.read_bytes(), f"NESFIC-D-21_{run['run_id']}.pdf",
                                       "application/pdf", icon=":material/picture_as_pdf:", type="primary",
                                       use_container_width=True,
                                       help="Maps, class areas, cross-matrix, change map, priority list, "
                                            "accuracy and method.")
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for f in (rd / "stats").glob("*"):
                        z.write(f, f"statistics/{f.name}")
                st.download_button("Statistics (CSV, JSON)", buf.getvalue(),
                                   f"NESFIC-D-21_{run['run_id']}_statistics.zip", icon=":material/table_chart:",
                                   use_container_width=True,
                                   help="Class areas, gains and losses, cross-matrix, priority list, summary.")
                buf2 = io.BytesIO()
                with zipfile.ZipFile(buf2, "w", zipfile.ZIP_DEFLATED) as z:
                    for f in (rd / "rasters").glob("*.tif"):
                        z.write(f, f"geotiff/{f.name}")
                st.download_button("GIS layers (GeoTIFF)", buf2.getvalue(),
                                   f"NESFIC-D-21_{run['run_id']}_geotiff.zip", icon=":material/map:",
                                   use_container_width=True,
                                   help="LULC maps, confidence, change codes and change intensity; open in QGIS/ArcGIS.")
                st.markdown('<div class="gc-step">Run details</div>', unsafe_allow_html=True)
                st.json({k: run.get(k) for k in ("name", "bbox", "source", "periods", "model", "created")},
                        expanded=False)

        # ---------------- SCALE ----------------
        elif ss.tool == "scale":
            panel_head("Scale-up to Assam",
                       "Assam (78,438 km²) is about 112 Survey of India 1:50k sheets. Each 15′ × 15′ "
                       "sheet is one independent processing tile; the Guwahati demo is one tile.")
            for h, d in (("Queue", "Tiles queued by district or SOI sheet; parallel workers run the same pipeline."),
                         ("Store", "Per-tile GeoTIFFs (COG) and statistics in object storage / PostGIS."),
                         ("Aggregate", "District and state cross-matrices are sums of tile matrices - no re-processing."),
                         ("Monitor", "Scheduled dry-season re-runs give periodic monitoring and early alerts.")):
                st.markdown(f'<div class="gc-card"><div class="h">{h}</div><div class="d">{d}</div></div>',
                            unsafe_allow_html=True)

        # ---------------- ABOUT ----------------
        else:
            panel_head("Method", "How the platform replaces on-screen visual interpretation.")
            for h, d in (
                    ("Automated acquisition", "Sentinel-2 / Landsat for any area and dates; per-pixel cloud and "
                                              "shadow masking; median composites."),
                    ("Radiometric harmonisation", "Sentinel-2 processing-baseline offset removed per scene; periods "
                                                  "normalised to each other so one model fits all years."),
                    ("AI classification", "Five algorithms benchmarked on polygon-held-out data; label cleaning "
                                          "removes spectrally inconsistent training pixels automatically."),
                    ("Temporally consistent change", "Both dates classified together; an automatic spectral-change "
                                                     "threshold cuts false change from 27% to 5% of the area "
                                                     "versus map-by-map comparison."),
                    ("Prioritisation", "Patches ranked by transition importance × size × confidence; river and char "
                                       "dynamics separated; low-confidence change flagged for field visits."),
                    ("Roadmap", "Train on ASSAC / NRSC 1:50k LULC maps (NRSC Level-I/II classes), add ResourceSat "
                                "(Bhoonidhi) and deep-learning segmentation.")):
                st.markdown(f'<div class="gc-card"><div class="h">{h}</div><div class="d">{d}</div></div>',
                            unsafe_allow_html=True)

# -------------------------------------------------
# MAP
# -------------------------------------------------

with map_col:
    bb = ss.bbox
    scale_view = ss.tool == "scale"
    overview = scale_view or not ss.aoi_set
    m = folium.Map(location=[26.2, 92.9] if overview else [(bb[1] + bb[3]) / 2, (bb[0] + bb[2]) / 2],
                   zoom_start=7 if overview else 11, tiles=None, control_scale=True, prefer_canvas=True)
    m.get_root().header.add_child(folium.Element("""
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600&display=swap" rel="stylesheet">
<style>
.leaflet-container { font-family: Manrope, system-ui, sans-serif; }
html, body, .folium-map, .leaflet-container { height: 100vh !important; margin: 0; }
.leaflet-control-layers { font-family: Manrope, sans-serif !important; font-size: 11.5px !important; color:#2B3A42;
  border: none !important; border-radius: 14px !important; background: rgba(255,255,255,.95) !important;
  box-shadow: 0 6px 20px rgba(19,35,43,.14) !important; padding: 8px 12px 8px 10px !important; }
.leaflet-control-layers-expanded { min-width: 0; }
.leaflet-control-layers label { margin: 3px 0; font-weight: 500; line-height: 1.25; }
.leaflet-control-layers label span { vertical-align: middle; }
.leaflet-control-layers input { accent-color: #0F6B5C; margin: 0 6px 0 0; width: 12px; height: 12px; vertical-align: middle; }
.leaflet-control-layers-separator { border-top: 1px solid #ECE7DE !important; margin: 6px 0 !important; }
.leaflet-bar { border: none !important; border-radius: 12px !important; overflow: hidden;
  box-shadow: 0 4px 14px rgba(19,35,43,.14) !important; }
.leaflet-bar a { border-bottom: 1px solid #F0ECE4 !important; color: #2B3A42 !important; }
.leaflet-control-attribution { font-size: 8px !important; line-height: 1.3 !important; padding: 1px 6px !important;
  background: rgba(255,255,255,.55) !important; color: #6B7A82 !important; border-radius: 8px 0 0 0; }
.leaflet-control-attribution a { color: #6B7A82 !important; }
.leaflet-control-scale-line { font-size: 9px; border-radius: 4px; }
.leaflet-tooltip { background: #13232B !important; color: #FFFFFF !important; border: none !important;
  border-radius: 10px !important; font: 500 12px Manrope, sans-serif; padding: 6px 10px;
  box-shadow: 0 6px 18px rgba(0,0,0,.2) !important; }
.leaflet-tooltip:before { display: none; }
.leaflet-control-mouseposition { font-size: 10px !important; background: rgba(255,255,255,.75) !important;
  border-radius: 8px; padding: 2px 8px !important; }
.leaflet-draw-toolbar a { border-radius: 0 !important; }
/* pencil icon for the draw-area tool */
.leaflet-draw-toolbar a.leaflet-draw-draw-rectangle {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%2313232B' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 20h9'/%3E%3Cpath d='M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z'/%3E%3C/svg%3E") !important;
  background-size: 18px 18px !important; background-position: center !important; background-repeat: no-repeat !important; }
</style>
<script>
window.addEventListener('load', function () {
  var a = document.querySelector('.leaflet-draw-draw-rectangle');
  if (a) { a.title = 'Draw area of interest (pencil)'; }
});
</script>"""))
    for name, (tiles, attr) in BASEMAPS.items():
        folium.TileLayer(tiles, attr=attr, name=f"Basemap · {name}", overlay=False,
                         show=(name == ss.basemap)).add_to(m)
    folium.WmsTileLayer(url=BHUVAN_WMS_URL, layers=BHUVAN_LULC_LAYER, fmt="image/png", transparent=True,
                        version="1.1.1", name="Bhuvan LULC 50K (NRSC)", attr="NRSC / ISRO Bhuvan",
                        overlay=True, show=ss.vis.get("bhuvan", False)).add_to(m)

    for lab, f in sorted(ss.fetched.items()):
        lay = f["layer"]
        folium.raster_layers.ImageOverlay(image=lay["path"], bounds=lay["bounds"],
                                          name=f"Imagery {lab} (true colour)", opacity=1.0,
                                          show=ss.vis.get(f"rgb_{lab}", True)).add_to(m)
    if run:
        rd = Path(run["path"])
        ov = run["overlays"]
        a, b = run["labels"][0], run["labels"][-1]
        for k, name, op in ((f"rgb_{a}", f"Imagery {a} (analysis)", 1.0), (f"rgb_{b}", f"Imagery {b} (analysis)", 1.0),
                            (f"lulc_{a}", f"LULC {a}", 0.85), (f"lulc_{b}", f"LULC {b}", 0.85),
                            ("intensity", "Spectral change intensity", 0.9),
                            ("change", "Significant change", 0.95), (f"confidence_{b}", f"AI confidence {b}", 0.75)):
            if k in ov and not (k.startswith("rgb_") and ss.fetched):
                folium.raster_layers.ImageOverlay(image=str(rd / "overlays" / ov[k]["file"]),
                                                  bounds=ov[k]["bounds"], name=name, opacity=op,
                                                  show=ss.vis.get(k, False)).add_to(m)
        pr_path = rd / "stats" / f"priority_changes_{a}_{b}.csv"
        if pr_path.exists():
            pr = pd.read_csv(pr_path).head(30)
            fg = folium.FeatureGroup(name="Priority changes (top 30)", show=ss.vis.get("priority", True))
            pcol = {"High": "#B42318", "Medium": "#D97706", "Low": "#64748B"}
            for _, r in pr.iterrows():
                folium.CircleMarker([r["lat"], r["lon"]], radius=6, color="#ffffff", weight=1.5, fill=True,
                                    fill_color=pcol.get(r["priority"], "#64748B"), fill_opacity=0.95,
                                    tooltip=f'#{r["rank"]} · {r["transition"]} · {r["area_ha"]} ha · '
                                            f'{r["priority"]} priority').add_to(fg)
            fg.add_to(m)

    if ss.tool == "scale":
        from modules.tiling import make_tiles
        fgt = folium.FeatureGroup(name="Processing tiles (15′ sheets)")
        for t in make_tiles():
            w_, s_, e_, n_ = t["bbox"]
            demo = w_ <= 91.77 <= e_ and s_ <= 26.15 <= n_
            folium.Rectangle([[s_, w_], [n_, e_]], color=TEAL if demo else "#64748B",
                             weight=2 if demo else 0.4, fill=demo, fill_opacity=0.25,
                             tooltip=f'Tile {t["tile_id"]} · EPSG:{t["epsg"]}').add_to(fgt)
        fgt.add_to(m)

    if ss.aoi_set:
        folium.Rectangle([[bb[1], bb[0]], [bb[3], bb[2]]], color=GOLD, weight=2, dash_array="6 4",
                         fill=False, tooltip=f"Area of interest · {area_km2(bb):,.0f} km²").add_to(m)
    Draw(draw_options={"polyline": False, "polygon": False, "circle": False, "marker": False,
                       "circlemarker": False, "rectangle": {"shapeOptions": {"color": TEAL}}},
         edit_options={"edit": False, "remove": False}, position="topleft").add_to(m)
    Fullscreen(position="topleft", title="Full screen", title_cancel="Exit full screen").add_to(m)
    MousePosition(position="bottomleft", num_digits=4, prefix="Lat/Lon").add_to(m)
    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    if ss.tool != "scale" and ss.aoi_set:
        m.fit_bounds([[bb[1], bb[0]], [bb[3], bb[2]]])

    with st.container(key="mapbox"):
        out = st_folium(m, height=MAP_H, use_container_width=True,
                        returned_objects=["last_active_drawing"], key="main_map")
    drawing = (out or {}).get("last_active_drawing")
    if drawing:
        sig = hashlib.md5(json.dumps(drawing, sort_keys=True).encode()).hexdigest()
        if sig != ss.last_draw:
            ss.last_draw = sig
            coords = drawing["geometry"]["coordinates"][0]
            lons, lats = [p[0] for p in coords], [p[1] for p in coords]
            ss.bbox = [round(min(lons), 4), round(min(lats), 4), round(max(lons), 4), round(max(lats), 4)]
            ss.area_name = "Custom (draw or type)"
            ss.aoi_set = True
            ss.fetched, ss.run_id, ss.tool = {}, None, "area"
            st.rerun()

    area_txt = f"{area_km2(bb):,.0f} km²" if ss.aoi_set else "draw an area to begin"
    n_layers = len(ss.fetched) + (len(run["overlays"]) if run else 0)
    st.markdown(f'<div class="gc-status"><span data-tip="Current area of interest"><b>{ss.area_name}</b> · '
                f'{area_txt}</span><span data-tip="Layers available in the layer switcher">'
                f'{n_layers} data layers</span><span data-tip="Prototype prepared by CoArchitive Pvt. Ltd.">NESFIC-D-21 prototype · by '
                f'<b>CoArchitive Pvt. Ltd.</b></span></div>', unsafe_allow_html=True)
