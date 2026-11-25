"""
Automated LULC classification (v2).

Pipeline per run:
  1. Load each year's multispectral composite, remove the Sentinel-2
     BOA offset (2022+), normalise every year to REFERENCE_YEAR.
  2. Build 14 features: 5 bands, 5 indices, 4 texture measures.
  3. Extract training pixels from the QGIS polygons (data/boundaries).
  4. Evaluate with polygon-grouped 5-fold cross-validation, so no pixel
     from a validation polygon is ever seen in training (honest accuracy).
  5. Train a gradient-boosted tree model on all samples.
  6. Classify every year: class map, per-pixel confidence, 3x3 majority
     filter to remove salt-and-pepper noise.

Outputs
  data/classified/lulc_<year>.tif          uint8, 0 = nodata
  data/classified/confidence_<year>.tif    uint8, 0-100 (%)
  outputs/statistics/accuracy_v2.json      metrics + confusion matrix
  outputs/statistics/polygon_qa.csv        training polygons to re-check
  outputs/statistics/lulc_area_<year>.csv  class areas
  models/lulc_hgb_v2.joblib
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath
from scipy import ndimage as ndi
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, f1_score, precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (  # noqa: E402
    BASE_DIR, CLASSIFIED_DIR, PROCESSED_DIR, STATS_DIR, PERIODS, IMAGERY_SOURCE,
    CLASS_MERGE, LULC_CLASSES, REFERENCE_YEAR,
)
from modules.features import (  # noqa: E402
    FEATURE_NAMES, build_feature_stack, harmonise_reflectance,
    match_to_reference,
)
from modules.gpkg_reader import read_training_polygons  # noqa: E402
from modules.raster_io import read_raster, write_raster  # noqa: E402

MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "lulc_hgb_v2.joblib"
TRAINING_GPKG = BASE_DIR / "data" / "boundaries" / "lulc_training.gpkg"
TRAINING_FEATURES = PROCESSED_DIR / "training_features_v2.csv"
MAX_PIXELS_PER_POLYGON = 500
RANDOM_SEED = 42


# -------------------------------------------------
# 1-2. LOAD, HARMONISE, FEATURES
# -------------------------------------------------

def raster_path(year):
    if IMAGERY_SOURCE == "landsat":
        return PROCESSED_DIR / f"guwahati_landsat_{year}.tif"
    return PROCESSED_DIR / f"guwahati_multispectral_{year}.tif"


def load_reflectance(years, offset_corrected=None):
    """Return {year: (reflectance, geo)} harmonised to REFERENCE_YEAR.

    offset_corrected: set of years whose GeoTIFF was produced by the new
    satellite_data.py (offset already removed). Defaults to reading the
    'offset_corrected' flag file written by satellite_data.py.
    """
    if offset_corrected is None:
        flag = PROCESSED_DIR / "offset_corrected_years.json"
        offset_corrected = set(json.loads(flag.read_text())) if flag.exists() else set()

    out = {}
    for year in years:
        raw, geo = read_raster(raster_path(year))
        already = IMAGERY_SOURCE == "landsat" or str(year) in offset_corrected
        refl = harmonise_reflectance(raw, year, already)
        out[str(year)] = (refl, geo)

    ref = out.get(str(REFERENCE_YEAR))
    if ref is not None:
        for year in out:
            if year != str(REFERENCE_YEAR):
                out[year] = (match_to_reference(out[year][0], ref[0]), out[year][1])
    return out


# -------------------------------------------------
# 3. TRAINING SAMPLES FROM POLYGONS
# -------------------------------------------------

def polygon_pixels(poly, geo, shape):
    exterior = poly[0]
    res = geo["res"]
    rows, cols = shape
    c0 = max(int((exterior[:, 0].min() - geo["x0"]) // res), 0)
    c1 = min(int((exterior[:, 0].max() - geo["x0"]) // res) + 1, cols)
    r0 = max(int((geo["y0"] - exterior[:, 1].max()) // res), 0)
    r1 = min(int((geo["y0"] - exterior[:, 1].min()) // res) + 1, rows)
    if c1 <= c0 or r1 <= r0:
        return np.empty(0, int), np.empty(0, int)

    cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
    pts = np.column_stack([
        geo["x0"] + (cc.ravel() + 0.5) * res,
        geo["y0"] - (rr.ravel() + 0.5) * res,
    ])
    inside = MPath(exterior).contains_points(pts)
    for hole in poly[1:]:
        inside &= ~MPath(hole).contains_points(pts)
    return rr.ravel()[inside], cc.ravel()[inside]


def extract_samples(stacks):
    rng = np.random.default_rng(RANDOM_SEED)
    records = []
    for feat in read_training_polygons(TRAINING_GPKG):
        year = str(feat["source_year"])
        if year not in stacks:
            continue
        stack, geo = stacks[year]
        for poly in feat["polygons"]:
            r, c = polygon_pixels(poly, geo, stack.shape[1:])
            values = stack[:, r, c].T
            values = values[np.isfinite(values).all(axis=1)]
            if len(values) > MAX_PIXELS_PER_POLYGON:
                values = values[rng.choice(len(values), MAX_PIXELS_PER_POLYGON, replace=False)]
            for v in values:
                records.append((*v, feat["class_id"], int(year), feat["fid"]))

    df = pd.DataFrame(records, columns=FEATURE_NAMES + ["train_class", "year", "fid"])
    df["class_id"] = df["train_class"].map(CLASS_MERGE)
    return clean_samples(df)


def clean_samples(df):
    """Spectral consistency check on hand-drawn polygons.

    Polygons drawn over the Brahmaputra also cover exposed sandbars
    (char land), and some land polygons clip open water. Removing pixels
    whose spectra contradict their label stops the model learning
    'bright sand = water'.
    """
    water = df["class_id"] == 4
    bad = (water & (df["MNDWI"] <= 0.0)) | (~water & (df["MNDWI"] > 0.3))
    removed = df[bad].groupby("class_id").size().to_dict()
    if removed:
        print(f"  removed {int(bad.sum())} spectrally inconsistent pixels {removed}")
    return df[~bad].reset_index(drop=True)


# -------------------------------------------------
# 4-5. MODEL
# -------------------------------------------------

def make_model():
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )


def evaluate(df):
    X = df[FEATURE_NAMES].values
    y = df["class_id"].values
    groups = df["fid"].values
    pred = np.zeros_like(y)

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for train_idx, test_idx in splitter.split(X, y, groups):
        model = make_model().fit(X[train_idx], y[train_idx])
        pred[test_idx] = model.predict(X[test_idx])

    labels = sorted(LULC_CLASSES)
    names = [LULC_CLASSES[k]["name"] for k in labels]
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)

    metrics = {
        "validation": "5-fold cross-validation grouped by training polygon",
        "n_samples": int(len(y)),
        "n_polygons": int(df["fid"].nunique()),
        "overall_accuracy": round(float(accuracy_score(y, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "kappa": round(float(cohen_kappa_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "classes": names,
        "per_class": [
            {"class": n, "precision": round(float(pp), 3), "recall": round(float(rr), 3),
             "f1": round(float(ff), 3), "support": int(ss)}
            for n, pp, rr, ff, ss in zip(names, p, r, f, s)
        ],
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
    }

    df = df.assign(pred=pred)
    qa = (
        df.groupby(["fid", "train_class", "year"])
        .apply(lambda t: pd.Series({
            "pixels": len(t),
            "agreement": round(float((t["pred"] == t["class_id"]).mean()), 3),
            "model_says": LULC_CLASSES[int(t["pred"].mode()[0])]["name"],
        }), include_groups=False)
        .reset_index()
        .sort_values("agreement")
    )
    qa["label"] = qa["train_class"].map({
        1: "Built-up", 2: "Tree cover", 3: "Agriculture",
        4: "Water", 5: "Scrub", 6: "Barren"})
    return metrics, qa


def train(df):
    model = make_model().fit(df[FEATURE_NAMES].values, df["class_id"].values)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURE_NAMES,
                 "classes": LULC_CLASSES, "class_merge": CLASS_MERGE}, MODEL_FILE)
    return model


def load_model():
    """Load the saved model; if it cannot be unpickled (different scikit-learn
    version on the server), retrain in ~10 s from the saved training table."""
    try:
        return joblib.load(MODEL_FILE)["model"]
    except Exception as err:  # noqa: BLE001
        print(f"Model load failed ({err}); retraining from {TRAINING_FEATURES.name}")
        return train(pd.read_csv(TRAINING_FEATURES))


# -------------------------------------------------
# 6. CLASSIFY FULL RASTERS
# -------------------------------------------------

def majority_filter(classes, valid, n_classes, size=3):
    counts = np.stack([
        ndi.uniform_filter(((classes == k) & valid).astype("float32"), size)
        for k in range(1, n_classes + 1)
    ])
    smoothed = counts.argmax(axis=0).astype("uint8") + 1
    smoothed[~valid] = 0
    return smoothed


def predict_proba_stack(model, stack, chunk=500_000):
    """Per-pixel class probabilities (rows*cols, n_classes) and validity mask."""
    bands, rows, cols = stack.shape
    flat = stack.reshape(bands, -1).T
    valid = np.isfinite(flat).all(axis=1)
    proba = np.zeros((flat.shape[0], len(model.classes_)), dtype="float32")
    idx = np.flatnonzero(valid)
    for start in range(0, len(idx), chunk):
        sel = idx[start:start + chunk]
        proba[sel] = model.predict_proba(flat[sel])
    return proba, valid


def proba_to_maps(model, proba, valid, shape):
    rows, cols = shape
    cls = np.zeros(proba.shape[0], dtype="uint8")
    cls[valid] = model.classes_[proba[valid].argmax(axis=1)]
    conf = np.zeros(proba.shape[0], dtype="uint8")
    conf[valid] = np.round(proba[valid].max(axis=1) * 100).astype("uint8")
    valid2d = valid.reshape(rows, cols)
    cls = majority_filter(cls.reshape(rows, cols), valid2d, len(LULC_CLASSES))
    return cls, conf.reshape(rows, cols)


def classify_stack(model, stack):
    proba, valid = predict_proba_stack(model, stack)
    return proba_to_maps(model, proba, valid, stack.shape[1:])


def area_table(cls, res=10.0):
    px_km2 = res * res / 1e6
    total = int((cls > 0).sum())
    rows = []
    for k, meta in LULC_CLASSES.items():
        n = int((cls == k).sum())
        rows.append({"class_id": k, "class_name": meta["name"], "pixels": n,
                     "area_km2": round(n * px_km2, 3),
                     "percent": round(100 * n / total, 2) if total else 0})
    return pd.DataFrame(rows)


def main(years=None):
    years = [str(y) for y in (years or PERIODS.keys())]
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    CLASSIFIED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and harmonising imagery...")
    refl = load_reflectance(years)
    stacks = {y: (build_feature_stack(a), g) for y, (a, g) in refl.items()}

    print("Extracting training samples...")
    df = extract_samples(stacks)
    print(f"  {len(df)} pixels from {df['fid'].nunique()} polygons")
    df.to_csv(TRAINING_FEATURES, index=False, float_format="%.5f")

    print("Evaluating (polygon-grouped 5-fold CV)...")
    metrics, qa = evaluate(df)
    (STATS_DIR / "accuracy_v2.json").write_text(json.dumps(metrics, indent=2))
    qa.to_csv(STATS_DIR / "polygon_qa.csv", index=False)
    print(f"  OA={metrics['overall_accuracy']}  kappa={metrics['kappa']}  "
          f"macroF1={metrics['macro_f1']}")

    print("Training final model...")
    model = train(df)

    from modules.temporal import classify_multitemporal
    labels = sorted(stacks)
    geo = stacks[labels[0]][1]
    cls, conf, mag, thr = classify_multitemporal(model, [stacks[y][0] for y in labels])
    outputs = {}
    for i, year in enumerate(labels):
        write_raster(CLASSIFIED_DIR / f"lulc_{year}.tif", cls[i], geo, nodata=0)
        write_raster(CLASSIFIED_DIR / f"confidence_{year}.tif", conf[i], geo, nodata=0)
        area_table(cls[i], geo["res"]).to_csv(STATS_DIR / f"lulc_area_{year}.csv", index=False)
        outputs[year] = CLASSIFIED_DIR / f"lulc_{year}.tif"
    write_raster(CLASSIFIED_DIR / f"spectral_change_{labels[0]}_{labels[-1]}.tif",
                 np.nan_to_num(mag).astype("float32"), geo)
    print(f"  spectral change threshold (Otsu) = {thr:.2f}")
    print("Classification complete.")
    return outputs, metrics


if __name__ == "__main__":
    main()
