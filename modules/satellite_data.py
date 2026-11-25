
import numpy as np
import rasterio
import planetary_computer
import stackstac

from pystac_client import Client
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from PIL import Image

from config import (
    AOI_BBOX,
    TARGET_CRS,
    RESOLUTION,
    STAC_URL,
    COLLECTION,
    PERIODS,
    PROCESSED_DIR
)

NODATA_VALUE = -9999.0
MAX_SCENES = 20


# -------------------------------------------------
# 1. CONNECT TO SATELLITE CATALOGUE
# -------------------------------------------------

def connect_catalog():

    return Client.open(
        STAC_URL,
        modifier=planetary_computer.sign_inplace
    )


# -------------------------------------------------
# 2. SEARCH SENTINEL-2 IMAGERY
# -------------------------------------------------

def search_satellite_data(year):

    catalog = connect_catalog()

    search = catalog.search(
        collections=[COLLECTION],
        bbox=AOI_BBOX,
        datetime=PERIODS[str(year)],
        query={
            "eo:cloud_cover": {"lt": 60}
        },
        max_items=100
    )

    items = list(search.items())

    if not items:
        raise ValueError(
            f"No Sentinel-2 scenes found for {year}"
        )

    items.sort(
        key=lambda item: item.properties.get(
            "eo:cloud_cover", 100
        )
    )

    selected_items = items[:MAX_SCENES]

    print(f"\n{year}: {len(items)} candidate scenes found.")
    print(f"Selected {len(selected_items)} scenes.")

    for item in selected_items:

        print(
            item.id,
            "| Cloud:",
            item.properties.get("eo:cloud_cover"),
            "| Date:",
            item.properties.get("datetime")
        )

    return selected_items


# -------------------------------------------------
# 2b. RADIOMETRIC HARMONISATION
# -------------------------------------------------

BOA_OFFSET_START = np.datetime64("2022-01-25")


def remove_boa_offset(spectral, items):
    """Subtract the 1000 DN offset from scenes with processing baseline >= 4.0."""
    baselines = []
    for item in items:
        pb = item.properties.get("s2:processing_baseline")
        if pb is not None:
            try:
                baselines.append(float(pb) >= 4.0)
                continue
            except ValueError:
                pass
        baselines.append(np.datetime64(item.datetime.replace(tzinfo=None)) >= BOA_OFFSET_START)

    # stackstac keeps the item order along "time"; match by item id to be safe
    flag_by_id = {item.id: flag for item, flag in zip(items, baselines)}
    ids = [str(i) for i in spectral["id"].values]
    offsets = np.array([1000.0 if flag_by_id.get(i, False) else 0.0 for i in ids])

    print(f"BOA offset removed from {int((offsets > 0).sum())} of {len(offsets)} scenes.")
    import xarray as xr
    return spectral - xr.DataArray(offsets, dims="time", coords={"time": spectral["time"]})


def mark_offset_corrected(year):
    """Record that this year's GeoTIFF already has the offset removed."""
    import json
    flag = PROCESSED_DIR / "offset_corrected_years.json"
    years = set(json.loads(flag.read_text())) if flag.exists() else set()
    years.add(str(year))
    flag.write_text(json.dumps(sorted(years)))


# -------------------------------------------------
# 3. CREATE MULTISPECTRAL COMPOSITE
# -------------------------------------------------

def create_composite(year):

    items = search_satellite_data(year)

    print(f"\nProcessing multispectral imagery: {year}")

    spectral_bands = [
        "B02",
        "B03",
        "B04",
        "B08",
        "B11"
    ]

    # Acquire spectral bands on a common 10 m grid
    spectral = stackstac.stack(
        items,
        assets=spectral_bands,
        bounds_latlon=AOI_BBOX,
        epsg=32646,
        resolution=RESOLUTION,
        chunksize=512,
        dtype="int64",
        fill_value=0,
        rescale=False,
        resampling=Resampling.bilinear
    )

    # Acquire SCL separately
    scl_data = stackstac.stack(
        items,
        assets=["SCL"],
        bounds_latlon=AOI_BBOX,
        epsg=32646,
        resolution=RESOLUTION,
        chunksize=512,
        dtype="int64",
        fill_value=0,
        rescale=False,
        resampling=Resampling.nearest
    )

    scl = scl_data.sel(band="SCL")

    # Invalid SCL classes
    invalid_classes = [
        0, 1, 3, 7, 8, 9, 10, 11
    ]

    valid_pixels = ~scl.isin(invalid_classes)

    # Apply mask to every scene before temporal compositing
    spectral = spectral.where(valid_pixels)

    print("Cloud and shadow masking completed.")

    # -------------------------------------------------
    # FIX: Sentinel-2 processing baseline 04.00 BOA offset
    # From 25 Jan 2022 ESA adds +1000 to every L2A DN.
    # Reflectance = (DN - 1000) / 10000 for those scenes.
    # Without this, 2022+ imagery is ~0.10 too bright in every band.
    # -------------------------------------------------
    spectral = remove_boa_offset(spectral, items)

    # Temporal median composite
    composite = spectral.median(
        dim="time",
        skipna=True
    ).compute()

    print("Temporal median composite generated.")

    # Convert raw Sentinel-2 values to reflectance
    array = (
        composite.values.astype(np.float32) / 10000.0
    )

    # Preserve missing pixels as explicit NoData
    array = np.where(
        np.isfinite(array),
        array,
        NODATA_VALUE
    ).astype(np.float32)

    # Get raster coordinates
    x = composite.x.values
    y = composite.y.values

    transform = from_origin(
        float(x[0] - RESOLUTION / 2),
        float(y[0] + RESOLUTION / 2),
        RESOLUTION,
        RESOLUTION
    )

    # -------------------------------------------------
    # 4. SAVE MULTISPECTRAL GEOTIFF
    # -------------------------------------------------

    geotiff_path = (
        PROCESSED_DIR /
        f"guwahati_multispectral_{year}.tif"
    )

    with rasterio.open(
        geotiff_path,
        "w",
        driver="GTiff",
        height=array.shape[1],
        width=array.shape[2],
        count=5,
        dtype="float32",
        crs=TARGET_CRS,
        transform=transform,
        nodata=NODATA_VALUE,
        compress="deflate"
    ) as dst:

        dst.write(array)

        for index, band_name in enumerate(
            spectral_bands, start=1
        ):

            dst.set_band_description(
                index,
                band_name
            )

    print(f"Multispectral GeoTIFF saved: {geotiff_path}")

    mark_offset_corrected(year)

    # -------------------------------------------------
    # 5. GENERATE RGB PREVIEW
    # -------------------------------------------------

    # B04 = Red, B03 = Green, B02 = Blue
    rgb = np.stack(
        [
            array[2],
            array[1],
            array[0]
        ],
        axis=-1
    )

    valid = np.all(
        np.isfinite(rgb) & (rgb != NODATA_VALUE),
        axis=2
    )

    preview = np.zeros_like(
        rgb,
        dtype=np.uint8
    )

    for band in range(3):

        values = rgb[:, :, band][valid]

        if values.size == 0:
            continue

        low, high = np.percentile(
            values,
            [2, 98]
        )

        if high > low:

            stretched = (
                (rgb[:, :, band] - low)
                / (high - low)
            )

            preview[:, :, band] = (
                np.clip(stretched, 0, 1) * 255
            ).astype(np.uint8)

    # Keep missing pixels black in the preview
    preview[~valid] = 0

    image_path = (
        PROCESSED_DIR /
        f"guwahati_rgb_{year}.png"
    )

    Image.fromarray(preview).save(image_path)

    print(f"RGB preview saved: {image_path}")

    valid_percentage = (
        np.count_nonzero(valid) / valid.size
    ) * 100

    print(
        f"Valid RGB coverage: {valid_percentage:.2f}%"
    )

    return str(geotiff_path), str(image_path)


# -------------------------------------------------
# 6. RUN STANDALONE
# -------------------------------------------------

if __name__ == "__main__":

    for year in ["2020", "2025"]:

        create_composite(year)
