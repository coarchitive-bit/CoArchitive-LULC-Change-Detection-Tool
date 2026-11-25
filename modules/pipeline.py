"""
End-to-end analysis for any area and any two observation periods.

run_analysis(name, bbox, periods, source, max_cloud)
  1. fetch cloud-free composites for each period (modules.fetch)
  2. build features, classify with the pretrained model (models/lulc_hgb_v2.joblib)
  3. change detection, cross-matrix, significant change, prioritisation
  4. map overlays (PNG + bounds) and a PDF report

Everything for one run lives in outputs/runs/<run_id>/ so many runs (or many
SOI sheets / districts) can sit side by side.

build_run_from_existing() packages the Guwahati demo (already-downloaded
2020 / 2025 composites) in the same structure, so the app opens with results.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BASE_DIR, LULC_CLASSES, OUTPUT_DIR  # noqa: E402
from modules.change_detection import detect_change, prioritise  # noqa: E402
from modules.classification import area_table, load_model  # noqa: E402
from modules.temporal import classify_multitemporal  # noqa: E402
from modules.features import build_feature_stack, match_to_reference  # noqa: E402
from modules.geo_utils import warp_to_latlon  # noqa: E402
from modules.raster_io import read_raster, write_raster  # noqa: E402

RUNS_DIR = OUTPUT_DIR / "runs"

CHANGE_COLORS = {
    (2, 1): "#B5179E", (4, 1): "#7209B7", (3, 1): "#F72585",
    (2, 3): "#FB8500", (4, 3): "#FFB703", (3, 4): "#219EBC",
}


def _hex(h):
    return [int(h[i:i + 2], 16) for i in (1, 3, 5)]


def rgb_preview(refl):
    rgb = np.stack([refl[2], refl[1], refl[0]], axis=-1).astype("float32")
    valid = np.isfinite(rgb).all(axis=2)
    out = np.zeros(rgb.shape, dtype="uint8")
    for b in range(3):
        v = rgb[..., b][valid]
        if v.size:
            lo, hi = np.percentile(v, [2, 98])
            out[..., b] = (np.clip((rgb[..., b] - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype("uint8")
    alpha = np.where(valid, 255, 0).astype("uint8")
    return np.dstack([out, alpha])


def lulc_rgba(cls):
    pal = np.zeros((256, 4), dtype="uint8")
    for k, v in LULC_CLASSES.items():
        pal[k] = _hex(v["color"]) + [255]
    return pal[cls]


def change_rgba(code):
    pal = np.zeros((256, 4), dtype="uint8")
    for a in LULC_CLASSES:
        for b in LULC_CLASSES:
            if a != b:
                pal[a * 10 + b] = _hex(CHANGE_COLORS.get((a, b), "#6C757D")) + [255]
    return pal[code]


def confidence_rgba(conf):
    c = conf.astype("float32") / 100
    rgba = np.zeros(conf.shape + (4,), dtype="uint8")
    rgba[..., 0] = (255 * (1 - c)).astype("uint8")
    rgba[..., 1] = (200 * c).astype("uint8")
    rgba[..., 2] = 60
    rgba[..., 3] = np.where(conf > 0, 200, 0)
    return rgba


def save_overlay(rgba, geo, path, max_size=1200):
    warped, bounds = warp_to_latlon(rgba, geo, max_size=max_size)
    if path.suffix == ".jpg":
        Image.fromarray(warped[..., :3], "RGB").save(path, quality=82, optimize=True)
    else:
        img = Image.fromarray(warped, "RGBA")
        if len(np.unique(warped.reshape(-1, 4)[::97], axis=0)) <= 64:
            img = img.quantize(colors=64, method=Image.Quantize.FASTOCTREE)
        img.save(path, optimize=True)
    return {"file": path.name, "bounds": bounds}


def change_intensity_rgba(mag):
    m = np.nan_to_num(mag, nan=0.0)
    v = np.clip(m / max(np.percentile(m[m > 0], 99) if (m > 0).any() else 1, 1e-6), 0, 1)
    rgba = np.zeros(m.shape + (4,), dtype="uint8")
    rgba[..., 0] = 255
    rgba[..., 1] = (220 * (1 - v)).astype("uint8")
    rgba[..., 2] = 0
    rgba[..., 3] = (v ** 1.5 * 230).astype("uint8")
    return rgba


def _finish_run(run_dir, meta, labels, refl_by_label, geo, cls_by_label, conf_by_label, log,
                mag=None):
    cls_dir = run_dir / "rasters"
    stats_dir = run_dir / "stats"
    ov_dir = run_dir / "overlays"
    for d in (cls_dir, stats_dir, ov_dir):
        d.mkdir(parents=True, exist_ok=True)

    overlays = {}
    for lab in labels:
        write_raster(cls_dir / f"lulc_{lab}.tif", cls_by_label[lab], geo, nodata=0)
        write_raster(cls_dir / f"confidence_{lab}.tif", conf_by_label[lab], geo, nodata=0)
        area_table(cls_by_label[lab], geo["res"]).to_csv(stats_dir / f"lulc_area_{lab}.csv", index=False)
        overlays[f"rgb_{lab}"] = save_overlay(rgb_preview(refl_by_label[lab]), geo, ov_dir / f"rgb_{lab}.jpg", 1400)
        overlays[f"lulc_{lab}"] = save_overlay(lulc_rgba(cls_by_label[lab]), geo, ov_dir / f"lulc_{lab}.png")
        overlays[f"confidence_{lab}"] = save_overlay(confidence_rgba(conf_by_label[lab]), geo,
                                                     ov_dir / f"confidence_{lab}.png", 800)

    if mag is not None:
        write_raster(cls_dir / f"spectral_change_{labels[0]}_{labels[-1]}.tif",
                     np.clip(np.nan_to_num(mag) * 100, 0, 65535).astype("uint16"), geo)
        overlays["intensity"] = save_overlay(change_intensity_rgba(mag), geo, ov_dir / "intensity.png", 800)

    log("Detecting change and building the cross-matrix...")
    a, b = labels[0], labels[-1]
    summary = detect_change(a, b, cls_dir, stats_dir)
    log("Prioritising significant changes...")
    prioritise(a, b, cls_dir=cls_dir, stats_dir=stats_dir)
    code, _ = read_raster(cls_dir / f"change_{a}_{b}.tif")
    overlays["change"] = save_overlay(change_rgba(code[0]), geo, ov_dir / "change.png")

    meta.update({"labels": labels, "overlays": overlays, "epsg": geo["epsg"],
                 "summary": json.loads((stats_dir / f"change_summary_{a}_{b}.json").read_text())})
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2))

    log("Writing PDF report...")
    from modules.reporting import build_report
    build_report(run_dir)
    log("Done.")
    return run_dir


def run_analysis(name, bbox, periods, source="sentinel-2", max_cloud=40,
                 max_scenes=12, log=print):
    """periods: {"2020": ("2020-01-01", "2020-03-31"), "2025": (...)}"""
    from modules.fetch import fetch_composite

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model = load_model()

    labels = list(periods)
    refl, cls, conf, acq = {}, {}, {}, {}
    geo = None
    t0 = time.time()
    for lab in labels:
        start, end = periods[lab]
        log(f"[{lab}] Fetching {source} imagery {start} to {end}...")
        r, g, info = fetch_composite(bbox, start, end, source, max_cloud, max_scenes,
                                     epsg=geo["epsg"] if geo else None, log=log)
        if geo and (r.shape[1:] != refl[labels[0]].shape[1:]):
            raise ValueError("Periods returned different grids; try again.")
        geo = geo or g
        refl[lab], acq[lab] = r, info

    log("Harmonising radiometry between periods...")
    for lab in labels[1:]:
        refl[lab] = match_to_reference(refl[lab], refl[labels[0]])
    log("AI classification (temporally consistent, both periods together)...")
    cl, cf, mag, thr = classify_multitemporal(model, [build_feature_stack(refl[l]) for l in labels])
    cls, conf = dict(zip(labels, cl)), dict(zip(labels, cf))
    log(f"Spectral change threshold (automatic, Otsu): {thr:.2f}")

    meta = {"run_id": run_id, "name": name, "bbox": bbox, "source": source,
            "max_cloud": max_cloud, "periods": {k: list(v) for k, v in periods.items()},
            "acquisition": acq, "created": datetime.now().isoformat(timespec="seconds")}
    _finish_run(run_dir, meta, labels, refl, geo, cls, conf, log, mag)
    log(f"Total time {time.time() - t0:.0f} s")
    return run_dir


def build_run_from_existing(name="Guwahati demo (2020 vs 2025)", log=print):
    """Package the demo outputs produced by classification.py as a run."""
    from config import AOI_BBOX, CLASSIFIED_DIR, PERIODS, STATS_DIR
    from modules.classification import load_reflectance

    labels = sorted(PERIODS)
    refl_geo = load_reflectance(labels)
    refl = {k: v[0] for k, v in refl_geo.items()}
    geo = refl_geo[labels[0]][1]
    cls = {k: read_raster(CLASSIFIED_DIR / f"lulc_{k}.tif")[0][0] for k in labels}
    conf = {k: read_raster(CLASSIFIED_DIR / f"confidence_{k}.tif")[0][0] for k in labels}
    mpath = CLASSIFIED_DIR / f"spectral_change_{labels[0]}_{labels[-1]}.tif"
    mag = read_raster(mpath)[0][0] if mpath.exists() else None

    run_dir = RUNS_DIR / "demo_guwahati"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": "demo_guwahati", "name": name, "bbox": AOI_BBOX, "source": "sentinel-2",
            "max_cloud": 60, "periods": {k: PERIODS[k].split("/") for k in labels},
            "acquisition": {}, "created": datetime.now().isoformat(timespec="seconds"),
            "demo": True}
    for f in ("accuracy_v2.json", "algorithm_benchmark.csv", "polygon_qa.csv"):
        src = STATS_DIR / f
        if src.exists():
            (run_dir / "stats").mkdir(parents=True, exist_ok=True)
            (run_dir / "stats" / f).write_bytes(src.read_bytes())
    return _finish_run(run_dir, meta, labels, refl, geo, cls, conf, log, mag)


def list_runs():
    runs = []
    if RUNS_DIR.exists():
        for p in sorted(RUNS_DIR.iterdir(), reverse=True):
            if (p / "run.json").exists():
                runs.append(json.loads((p / "run.json").read_text()) | {"path": str(p)})
    return runs


if __name__ == "__main__":
    build_run_from_existing()
