"""
Multi-temporal LULC change detection.

Compares two classified rasters on the same 10 m grid and produces:
  data/classified/change_<y1>_<y2>.tif     uint8 transition code
                                           0 = nodata, 1 = no change,
                                           from*10 + to for changes (e.g. 21 =
                                           Vegetation -> Built-up)
  outputs/statistics/transition_matrix_<y1>_<y2>.csv   km2, rows = from
  outputs/statistics/class_change_<y1>_<y2>.csv        gains / losses / net
  outputs/statistics/change_summary_<y1>_<y2>.json     headline numbers

"Significant" change = changed pixel that
  * is confident in both years (>= MIN_CONFIDENCE), and
  * belongs to a connected change patch of >= MIN_CHANGE_PIXELS pixels.
This suppresses edge noise and single-pixel misclassification, which
otherwise dominate change maps.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    CLASSIFIED_DIR, STATS_DIR, LULC_CLASSES,
    MIN_CHANGE_PIXELS, MIN_CONFIDENCE,
)
from modules.raster_io import read_raster, write_raster  # noqa: E402


def load(year, cls_dir=None):
    cls_dir = Path(cls_dir or CLASSIFIED_DIR)
    cls, geo = read_raster(cls_dir / f"lulc_{year}.tif")
    conf_path = cls_dir / f"confidence_{year}.tif"
    conf = read_raster(conf_path)[0][0] if conf_path.exists() else np.full(cls[0].shape, 100, "uint8")
    return cls[0], conf, geo


def detect_change(year_from, year_to, cls_dir=None, stats_dir=None):
    cls_dir = Path(cls_dir or CLASSIFIED_DIR)
    stats_dir = Path(stats_dir or STATS_DIR)
    c1, p1, geo = load(year_from, cls_dir)
    c2, p2, geo2 = load(year_to, cls_dir)
    if c1.shape != c2.shape or abs(geo["x0"] - geo2["x0"]) > 1e-6 or abs(geo["y0"] - geo2["y0"]) > 1e-6:
        raise ValueError("Classified rasters are not on a common grid.")

    px_km2 = geo["res"] ** 2 / 1e6
    valid = (c1 > 0) & (c2 > 0)
    changed = valid & (c1 != c2)

    confident = (p1 >= MIN_CONFIDENCE * 100) & (p2 >= MIN_CONFIDENCE * 100)
    candidate = changed & confident
    labels, n = ndi.label(candidate, structure=np.ones((3, 3)))
    sizes = np.bincount(labels.ravel())
    keep = sizes >= MIN_CHANGE_PIXELS
    keep[0] = False
    significant = keep[labels]

    code = np.zeros(c1.shape, dtype="uint8")
    code[valid] = 1
    code[significant] = (c1[significant] * 10 + c2[significant]).astype("uint8")

    tag = f"{year_from}_{year_to}"
    write_raster(cls_dir / f"change_{tag}.tif", code, geo, nodata=0)

    ids = sorted(LULC_CLASSES)
    names = [LULC_CLASSES[k]["name"] for k in ids]

    # Transition matrix on all valid pixels (standard cross-tabulation)
    n_cls = max(ids) + 1
    flat = c1[valid].astype(int) * n_cls + c2[valid].astype(int)
    counts = np.bincount(flat, minlength=n_cls * n_cls).reshape(n_cls, n_cls)
    matrix = counts[np.ix_(ids, ids)] * px_km2
    tm = pd.DataFrame(matrix.round(3), index=names, columns=names)
    tm.index.name = f"{year_from} \\ {year_to}"
    stats_dir.mkdir(parents=True, exist_ok=True)
    tm.to_csv(stats_dir / f"transition_matrix_{tag}.csv")

    # Significant-change matrix
    sflat = c1[significant].astype(int) * n_cls + c2[significant].astype(int)
    scounts = np.bincount(sflat, minlength=n_cls * n_cls).reshape(n_cls, n_cls)
    smatrix = scounts[np.ix_(ids, ids)] * px_km2

    rows = []
    for i, k in enumerate(ids):
        a1 = float(matrix[i, :].sum())
        a2 = float(matrix[:, i].sum())
        gain = float(matrix[:, i].sum() - matrix[i, i])
        loss = float(matrix[i, :].sum() - matrix[i, i])
        rows.append({
            "class_id": k, "class_name": names[i],
            f"area_{year_from}_km2": round(a1, 3),
            f"area_{year_to}_km2": round(a2, 3),
            "gain_km2": round(gain, 3), "loss_km2": round(loss, 3),
            "net_km2": round(a2 - a1, 3),
            "net_percent": round(100 * (a2 - a1) / a1, 2) if a1 else None,
            "significant_gain_km2": round(float(smatrix[:, i].sum() - smatrix[i, i]), 3),
            "significant_loss_km2": round(float(smatrix[i, :].sum() - smatrix[i, i]), 3),
        })
    cc = pd.DataFrame(rows)
    cc.to_csv(stats_dir / f"class_change_{tag}.csv", index=False)

    transitions = []
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if i != j and smatrix[i, j] > 0:
                transitions.append({"from": names[i], "to": names[j],
                                    "code": int(a * 10 + b),
                                    "area_km2": round(float(smatrix[i, j]), 3)})
    transitions.sort(key=lambda t: -t["area_km2"])

    total = float(valid.sum() * px_km2)
    summary = {
        "years": [str(year_from), str(year_to)],
        "analysed_area_km2": round(total, 2),
        "changed_area_km2_all": round(float(changed.sum() * px_km2), 2),
        "changed_percent_all": round(100 * changed.sum() / valid.sum(), 2),
        "significant_change_km2": round(float(significant.sum() * px_km2), 2),
        "significant_change_percent": round(100 * significant.sum() / valid.sum(), 2),
        "significant_patches": int(keep.sum()),
        "rules": {"min_confidence": MIN_CONFIDENCE,
                  "min_patch_pixels": MIN_CHANGE_PIXELS,
                  "min_patch_ha": MIN_CHANGE_PIXELS * geo["res"] ** 2 / 1e4},
        "top_transitions": transitions[:10],
    }
    (stats_dir / f"change_summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    from config import PERIODS
    years = sorted(PERIODS)
    print(json.dumps(detect_change(years[0], years[-1]), indent=2))
    print(prioritise(years[0], years[-1]).head(15).to_string(index=False))


# -------------------------------------------------
# PRIORITISATION AND FIELD-VERIFICATION FLAGS
# -------------------------------------------------

# Planning importance of each transition (1 = act first).
# Keys are (from, to) class ids from config.LULC_CLASSES.
TRANSITION_WEIGHT = {
    (2, 1): 1.00,   # Vegetation -> Built-up   (forest / hill / green loss to urbanisation)
    (4, 1): 1.00,   # Water -> Built-up        (wetland / water-body encroachment)
    (3, 1): 0.80,   # Agriculture -> Built-up  (urban sprawl on farmland)
    (2, 3): 0.70,   # Vegetation -> Open land  (clearing, hill cutting)
    (4, 3): 0.60,   # Water -> Open land       (reclamation; can be river dynamics)
    (3, 4): 0.40,   # Open land -> Water       (flooding / erosion)
    (2, 4): 0.40,
    (3, 2): 0.30,
    (1, 2): 0.20, (1, 3): 0.20, (1, 4): 0.20,  # loss of built-up is unusual -> likely error
    (4, 2): 0.30,
}


RIVER_MIN_KM2 = 2.0     # water bodies this large define the river corridor
RIVER_BUFFER_PX = 20    # ~200 m at 10 m
RIVER_WEIGHT = 0.15


def prioritise(year_from, year_to, top_n=200, doubtful_conf=0.6, cls_dir=None, stats_dir=None):
    """Rank significant change patches and flag doubtful areas.

    priority = transition weight x size factor x mean confidence
    size factor = log10(area_ha + 1) / log10(11), capped at 1 (10 ha or more = 1)
    """
    from modules.geo_utils import utm_to_latlon

    cls_dir = Path(cls_dir or CLASSIFIED_DIR)
    stats_dir = Path(stats_dir or STATS_DIR)
    tag = f"{year_from}_{year_to}"
    code, geo = read_raster(cls_dir / f"change_{tag}.tif")
    code = code[0]
    _, p1, _ = load(year_from, cls_dir)
    _, p2, _ = load(year_to, cls_dir)
    conf = np.minimum(p1, p2).astype("float32") / 100
    px_ha = geo["res"] ** 2 / 1e4
    names = {k: v["name"] for k, v in LULC_CLASSES.items()}

    # River corridor: large water bodies in either year, buffered ~200 m.
    # Change involving water inside it is mostly channel migration and
    # sandbar (char) dynamics, not land conversion -> separate context, low weight.
    c1_, _, _ = load(year_from, cls_dir)
    c2_, _, _ = load(year_to, cls_dir)
    water = (c1_ == 4) | (c2_ == 4)
    wlab, wn = ndi.label(water)
    big = np.zeros(wn + 1, bool)
    if wn:
        wsize = np.bincount(wlab.ravel())
        big = wsize * px_ha / 100 >= RIVER_MIN_KM2
        big[0] = False
    river = ndi.binary_dilation(big[wlab], iterations=RIVER_BUFFER_PX).astype("float32")

    rows = []
    for tcode in np.unique(code[code > 1]):
        a, b = divmod(int(tcode), 10)
        lab, n = ndi.label(code == tcode, structure=np.ones((3, 3)))
        if n == 0:
            continue
        idx = np.arange(1, n + 1)
        size = ndi.sum(np.ones_like(lab), lab, idx)
        mconf = ndi.mean(conf, lab, idx)
        cents = ndi.center_of_mass(np.ones_like(lab), lab, idx)
        in_river = ndi.mean(river, lab, idx)
        for pid, sz, mc, (r, c), rv in zip(idx, size, mconf, cents, in_river):
            area = sz * px_ha
            x = geo["x0"] + (c + 0.5) * geo["res"]
            y = geo["y0"] - (r + 0.5) * geo["res"]
            lat, lon = utm_to_latlon(x, y, geo["epsg"])
            river_dyn = (a == 4 or b == 4 or rv > 0.5) and rv > 0.5
            w = RIVER_WEIGHT if river_dyn else TRANSITION_WEIGHT.get((a, b), 0.3)
            size_f = min(1.0, np.log10(area + 1) / np.log10(11))
            rows.append({
                "transition": f"{names[a]} → {names[b]}",
                "from_class": names[a], "to_class": names[b],
                "area_ha": round(float(area), 2),
                "confidence": round(float(mc), 2),
                "lat": round(float(lat), 5), "lon": round(float(lon), 5),
                "priority_score": round(float(w * size_f * mc), 3),
                "context": "River channel / char dynamics" if river_dyn else "Land conversion",
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)
    df["priority"] = pd.cut(df["priority_score"], [-1, 0.25, 0.5, 2],
                            labels=["Low", "Medium", "High"]).astype(str)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    df["field_verification"] = np.where(df["confidence"] < doubtful_conf, "Recommended", "")
    df.head(top_n).to_csv(stats_dir / f"priority_changes_{tag}.csv", index=False)

    # Doubtful areas: changed pixels the model is unsure about (dropped from
    # "significant change" but worth a field visit, as in ASSAC's workflow)
    c1, _, _ = load(year_from, cls_dir)
    c2, _, _ = load(year_to, cls_dir)
    changed = (c1 > 0) & (c2 > 0) & (c1 != c2)
    doubtful = changed & (conf < MIN_CONFIDENCE)
    summary_path = stats_dir / f"change_summary_{tag}.json"
    summary = json.loads(summary_path.read_text())
    summary["doubtful_change_km2"] = round(float(doubtful.sum() * px_ha / 100), 2)
    summary["priority_counts"] = df["priority"].value_counts().to_dict()
    summary_path.write_text(json.dumps(summary, indent=2))
    return df
