"""
Interactive workspace steps used by the app:
  fetch_period()   -> imagery for one period (live, or bundled Guwahati demo cache)
  rgb_layer()      -> true-colour map layer for a fetched period
  classify_run()   -> chosen model, temporally consistent classification,
                      change detection, prioritisation, overlays, report
"""

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np

from config import AOI_BBOX, BASE_DIR, STATS_DIR
from modules.features import build_feature_stack, match_to_reference
from modules.pipeline import RUNS_DIR, _finish_run, rgb_preview, save_overlay
from modules.temporal import classify_multitemporal

DEMO_DIR = BASE_DIR / "data" / "demo_cache"
WORK_DIR = BASE_DIR / "outputs" / "workspace"
BANDS = ["B02", "B03", "B04", "B08", "B11"]


def is_demo_request(bbox, start, end, source):
    """Guwahati demo area, Sentinel-2, dry season 2020 or 2025 -> bundled imagery."""
    same_area = all(abs(a - b) < 0.02 for a, b in zip(bbox, AOI_BBOX))
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    return (same_area and source == "sentinel-2" and s.year == e.year
            and s.year in (2020, 2025) and s.month <= 3 and e.month <= 4
            and (DEMO_DIR / f"guwahati_{s.year}_B02.npz").exists())


def load_demo(year):
    grid = json.loads((DEMO_DIR / "guwahati_grid.json").read_text())
    arr = np.stack([np.load(DEMO_DIR / f"guwahati_{year}_{b}.npz")["band"] for b in BANDS])
    refl = arr.astype("float32") / 10000.0
    refl[:, arr.min(axis=0) == 0] = np.nan
    geo = {k: grid[k] for k in ("x0", "y0", "res", "epsg")}
    geo["nodata"] = None
    return refl, geo


def fetch_period(bbox, start, end, source, max_cloud, max_scenes, epsg=None, log=print, use_cache=False):
    if use_cache and is_demo_request(bbox, start, end, source):
        year = date.fromisoformat(start).year
        log(f"Guwahati demo: loading bundled Sentinel-2 composite for {year} "
            "(Jan–Mar median, cloud-masked, offset-corrected).")
        refl, geo = load_demo(year)
        clear = float(np.isfinite(refl).all(axis=0).mean() * 100)
        return refl, geo, {"scenes_found": None, "scenes_used": None, "dates": [start, end],
                           "clear_percent": round(clear, 1), "cached": True}
    from modules.fetch import fetch_composite
    refl, geo, info = fetch_composite(bbox, start, end, source, max_cloud, max_scenes,
                                      epsg=epsg, log=log)
    info["cached"] = False
    return refl, geo, info


def rgb_layer(refl, geo, label, session_id):
    out_dir = WORK_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return save_overlay(rgb_preview(refl), geo, out_dir / f"rgb_{label}.jpg", 1400) | {
        "path": str(out_dir / f"rgb_{label}.jpg")}


def classify_run(model, model_label, refls, geo, meta, log=print):
    labels = sorted(refls)
    log("Harmonising radiometry between periods...")
    harmonised = {labels[0]: refls[labels[0]]}
    for lab in labels[1:]:
        harmonised[lab] = match_to_reference(refls[lab], refls[labels[0]])

    log(f"Classifying both periods with {model_label} (temporally consistent)...")
    cl, cf, mag, thr = classify_multitemporal(
        model, [build_feature_stack(harmonised[l]) for l in labels])
    log(f"Automatic spectral-change threshold (Otsu): {thr:.2f}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    (run_dir / "stats").mkdir(parents=True, exist_ok=True)
    for f in ("accuracy_v2.json", "algorithm_benchmark.csv"):
        if (STATS_DIR / f).exists():
            (run_dir / "stats" / f).write_bytes((STATS_DIR / f).read_bytes())
    meta = meta | {"run_id": run_id, "model": model_label,
                   "created": datetime.now().isoformat(timespec="seconds")}
    _finish_run(run_dir, meta, labels, harmonised, geo, dict(zip(labels, cl)),
                dict(zip(labels, cf)), log, mag)
    return run_dir
