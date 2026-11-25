
import numpy as np
import rasterio
import pandas as pd
import joblib

from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from config import (
    PROCESSED_DIR,
    CLASSIFIED_DIR,
    STATS_DIR
)

BANDS = ["B02", "B03", "B04", "B08", "B11"]

N_CLUSTERS = 6
RANDOM_STATE = 42
MAX_TRAIN_PIXELS = 100000


def calculate_features(data):
    """
    Input shape: (bands, height, width)
    Output shape: (height, width, 9)
    """

    blue = data[0]
    green = data[1]
    red = data[2]
    nir = data[3]
    swir = data[4]

    def safe_index(a, b):
        denominator = a + b
        return np.divide(
            a - b,
            denominator,
            out=np.zeros_like(a, dtype=np.float32),
            where=np.abs(denominator) > 1e-8
        )

    ndvi = safe_index(nir, red)
    ndwi = safe_index(green, nir)
    ndbi = safe_index(swir, nir)

    denominator = swir + red + nir + blue

    bsi = np.divide(
        (swir + red) - (nir + blue),
        denominator,
        out=np.zeros_like(red, dtype=np.float32),
        where=np.abs(denominator) > 1e-8
    )

    features = np.stack(
        [blue, green, red, nir, swir,
         ndvi, ndwi, ndbi, bsi],
        axis=-1
    )

    return features.astype(np.float32)


def classify_year(year):

    input_file = (
        PROCESSED_DIR /
        f"guwahati_multispectral_{year}.tif"
    )

    output_file = (
        CLASSIFIED_DIR /
        f"guwahati_kmeans_{year}.tif"
    )

    model_file = (
        CLASSIFIED_DIR /
        f"kmeans_scaler_model_{year}.joblib"
    )

    print(f"\nProcessing {year}...")

    with rasterio.open(input_file) as src:

        data = src.read().astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

    # Valid pixels: all five bands must be valid
    valid = np.all(np.isfinite(data), axis=0)

    if nodata is not None:
        valid &= np.all(data != nodata, axis=0)

    valid &= np.all(data >= 0, axis=0)

    if valid.sum() < N_CLUSTERS:
        raise ValueError("Not enough valid pixels for clustering.")

    features = calculate_features(data)

    X = features[valid]

    print("Valid pixels:", len(X))

    # Sample pixels for fitting
    rng = np.random.default_rng(RANDOM_STATE)

    sample_size = min(MAX_TRAIN_PIXELS, len(X))

    sample_indices = rng.choice(
        len(X),
        size=sample_size,
        replace=False
    )

    X_sample = X[sample_indices]

    print("Fitting feature scaler...")

    scaler = StandardScaler()

    X_sample_scaled = scaler.fit_transform(X_sample)

    print("Running MiniBatch K-Means...")

    model = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        batch_size=4096,
        n_init=10,
        max_iter=200
    )

    model.fit(X_sample_scaled)

    print("Predicting all valid pixels...")

    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled) + 1

    # Output raster: 0 = NoData, 1-6 = clusters
    output = np.zeros(
        valid.shape,
        dtype=np.uint8
    )

    output[valid] = predictions.astype(np.uint8)

    profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="lzw"
    )

    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(output, 1)

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": BANDS + ["NDVI", "NDWI", "NDBI", "BSI"],
            "year": year
        },
        model_file
    )

    unique, counts = np.unique(
        predictions,
        return_counts=True
    )

    total = counts.sum()

    statistics = pd.DataFrame({
        "cluster_id": unique,
        "pixel_count": counts,
        "area_sq_km": counts * 0.0001,
        "percentage": counts / total * 100
    })

    statistics.to_csv(
        STATS_DIR / f"kmeans_statistics_{year}.csv",
        index=False
    )

    print("\nCluster distribution:")
    print(statistics.to_string(index=False))

    print("\nSaved:", output_file)

    return output_file


if __name__ == "__main__":

    for year in ["2020", "2025"]:
        classify_year(year)

    print("\nUnsupervised classification completed.")
