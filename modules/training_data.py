
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from rasterio.mask import mask

from config import BASE_DIR, PROCESSED_DIR


# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

TRAINING_FILE = (
    BASE_DIR / "data" / "boundaries" /
    "lulc_training.gpkg"
)

OUTPUT_FILE = (
    PROCESSED_DIR / "lulc_training_samples.csv"
)

RASTER_FILES = {
    2020: PROCESSED_DIR / "guwahati_multispectral_2020.tif",
    2025: PROCESSED_DIR / "guwahati_multispectral_2025.tif"
}

BAND_NAMES = ["B02", "B03", "B04", "B08", "B11"]

FEATURES = BAND_NAMES + ["NDVI", "NDWI", "NDBI", "BSI"]

MAX_PIXELS_PER_POLYGON = 500
RANDOM_SEED = 42
NODATA_VALUE = -9999.0


# -------------------------------------------------
# SPECTRAL INDICES
# -------------------------------------------------

def safe_index(numerator, denominator):

    result = np.full(
        denominator.shape,
        np.nan,
        dtype=np.float32
    )

    valid = np.abs(denominator) > 1e-8

    result[valid] = (
        numerator[valid] / denominator[valid]
    )

    return result


def calculate_features(pixels):

    # Input order: B02, B03, B04, B08, B11

    blue = pixels[:, 0]
    green = pixels[:, 1]
    red = pixels[:, 2]
    nir = pixels[:, 3]
    swir = pixels[:, 4]

    ndvi = safe_index(
        nir - red,
        nir + red
    )

    ndwi = safe_index(
        green - nir,
        green + nir
    )

    ndbi = safe_index(
        swir - nir,
        swir + nir
    )

    bsi = safe_index(
        (swir + red) - (nir + blue),
        (swir + red) + (nir + blue)
    )

    return np.column_stack([
        pixels,
        ndvi,
        ndwi,
        ndbi,
        bsi
    ])


# -------------------------------------------------
# EXTRACT PIXELS FROM POLYGON
# -------------------------------------------------

def extract_polygon_pixels(src, geometry, polygon_id):

    try:
        clipped, _ = mask(
            src,
            [geometry],
            crop=True,
            filled=False,
            all_touched=False
        )

    except ValueError:
        print(f"Polygon {polygon_id}: outside raster.")
        return np.empty((0, len(FEATURES)))

    masked = np.ma.asarray(clipped)

    invalid = np.ma.getmaskarray(masked).any(axis=0)

    values = np.asarray(
        masked.filled(np.nan),
        dtype=np.float32
    )

    invalid |= ~np.isfinite(values).all(axis=0)
    invalid |= (values <= -9990).any(axis=0)

    pixels = values[:, ~invalid].T

    pixels = pixels[
        np.isfinite(pixels).all(axis=1)
    ]

    pixels = pixels[
        (pixels >= 0).all(axis=1) &
        (pixels <= 1.5).all(axis=1)
    ]

    if len(pixels) == 0:
        return np.empty((0, len(FEATURES)))

    # Calculate four additional indices
    feature_pixels = calculate_features(pixels)

    feature_pixels = feature_pixels[
        np.isfinite(feature_pixels).all(axis=1)
    ]

    return feature_pixels


# -------------------------------------------------
# MAIN EXTRACTION
# -------------------------------------------------

def extract_training_data():

    print("\nStarting nine-feature extraction...")

    if not TRAINING_FILE.exists():
        raise FileNotFoundError(TRAINING_FILE)

    training = gpd.read_file(TRAINING_FILE)

    required = [
        "class_id",
        "class_name",
        "source_year"
    ]

    for field in required:
        if field not in training.columns:
            raise ValueError(f"Missing field: {field}")

    training = training.dropna(
        subset=required + ["geometry"]
    ).copy()

    training["class_id"] = pd.to_numeric(
        training["class_id"],
        errors="coerce"
    )

    training["source_year"] = pd.to_numeric(
        training["source_year"],
        errors="coerce"
    )

    training = training.dropna(
        subset=["class_id", "source_year"]
    )

    training["class_id"] = training["class_id"].astype(int)
    training["source_year"] = training["source_year"].astype(int)

    training = training[
        training["class_id"].isin([1, 2, 3, 4, 5, 6]) &
        training["source_year"].isin([2020, 2025])
    ].copy()

    if training.crs is None:
        raise ValueError("Training layer has no CRS.")

    if training.empty:
        raise ValueError("No valid training polygons.")

    rng = np.random.default_rng(RANDOM_SEED)

    extracted_rows = []

    for year, raster_path in RASTER_FILES.items():

        year_training = training[
            training["source_year"] == year
        ].copy()

        if year_training.empty:
            print(f"No samples for {year}.")
            continue

        if not raster_path.exists():
            raise FileNotFoundError(raster_path)

        print(f"\nProcessing {year}...")

        with rasterio.open(raster_path) as src:

            if src.count != 5:
                raise ValueError(
                    f"{year} raster must have five bands."
                )

            year_training = year_training.to_crs(src.crs)

            for polygon_id, row in year_training.iterrows():

                if row.geometry is None or row.geometry.is_empty:
                    continue

                pixels = extract_polygon_pixels(
                    src,
                    row.geometry,
                    polygon_id
                )

                if len(pixels) == 0:
                    continue

                if len(pixels) > MAX_PIXELS_PER_POLYGON:

                    selected = rng.choice(
                        len(pixels),
                        size=MAX_PIXELS_PER_POLYGON,
                        replace=False
                    )

                    pixels = pixels[selected]

                for pixel in pixels:

                    record = dict(
                        zip(FEATURES, pixel)
                    )

                    record.update({
                        "class_id": int(row["class_id"]),
                        "class_name": str(row["class_name"]),
                        "source_year": year,
                        "polygon_id": str(polygon_id)
                    })

                    extracted_rows.append(record)

                print(
                    f"Polygon {polygon_id}: "
                    f"{len(pixels)} pixels"
                )

    if not extracted_rows:
        raise ValueError("No valid pixels extracted.")

    result = pd.DataFrame(extracted_rows)

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n--------------------------------")
    print("NINE-FEATURE EXTRACTION COMPLETE")
    print("--------------------------------")

    print(f"Total pixels: {len(result)}")
    print(f"Features: {FEATURES}")
    print(f"Saved: {OUTPUT_FILE}")

    print("\nClass distribution:")

    print(
        result.groupby(
            ["source_year", "class_name"]
        ).size()
    )

    return result


if __name__ == "__main__":

    extract_training_data()
