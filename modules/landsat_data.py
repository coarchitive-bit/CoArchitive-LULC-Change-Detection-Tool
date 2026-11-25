"""
Alternative automated imagery source: Landsat 8/9 Collection 2 Level-2.

Same catalogue (Microsoft Planetary Computer STAC), same AOI and output
format as satellite_data.py, so the rest of the pipeline does not change.
Use it when:
  * you need years before Sentinel-2 (Landsat 8 starts in 2013),
  * Sentinel-2 has too few clear scenes in a season (Assam monsoon).

Bands are written in the Sentinel-2 order the model expects:
  B02=blue, B03=green, B04=red, B08=nir08, B11=swir16
Resolution is 30 m natively; it is resampled to the 10 m project grid so
classified maps from both sensors line up for change detection.

Set IMAGERY_SOURCE = "landsat" in config.py to use these files.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import AOI_BBOX, PERIODS, PROCESSED_DIR, RESOLUTION, STAC_URL  # noqa: E402

LANDSAT_COLLECTION = "landsat-c2-l2"
LANDSAT_ASSETS = ["blue", "green", "red", "nir08", "swir16"]
OUTPUT_NAMES = ["B02", "B03", "B04", "B08", "B11"]
SCALE, OFFSET = 0.0000275, -0.2           # USGS C2 L2 surface reflectance
NODATA_VALUE = -9999.0
MAX_SCENES = 12


def search_landsat(year, max_cloud=50):
    import planetary_computer
    from pystac_client import Client

    catalog = Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[LANDSAT_COLLECTION],
        bbox=AOI_BBOX,
        datetime=PERIODS[str(year)],
        query={
            "eo:cloud_cover": {"lt": max_cloud},
            "platform": {"in": ["landsat-8", "landsat-9"]},
        },
        max_items=100,
    )
    items = sorted(search.items(), key=lambda i: i.properties.get("eo:cloud_cover", 100))
    if not items:
        raise ValueError(f"No Landsat 8/9 scenes found for {year}")
    print(f"{year}: {len(items)} Landsat scenes, using {min(len(items), MAX_SCENES)}")
    return items[:MAX_SCENES]


def create_landsat_composite(year):
    import stackstac
    from rasterio.enums import Resampling
    from modules.raster_io import write_raster

    items = search_landsat(year)

    common = dict(bounds_latlon=AOI_BBOX, epsg=32646, resolution=RESOLUTION,
                  chunksize=1024, fill_value=0, rescale=False)

    spectral = stackstac.stack(items, assets=LANDSAT_ASSETS, dtype="float32",
                               resampling=Resampling.bilinear, **common)
    qa = stackstac.stack(items, assets=["qa_pixel"], dtype="uint16",
                         resampling=Resampling.nearest, **common).sel(band="qa_pixel")

    # QA_PIXEL bits: 0 fill, 1 dilated cloud, 3 cloud, 4 cloud shadow, 5 snow
    bad_bits = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4) | (1 << 5)
    clear = (qa & bad_bits) == 0

    reflectance = (spectral * SCALE + OFFSET).where(clear & (spectral > 0))
    composite = reflectance.median(dim="time", skipna=True).compute()

    array = composite.values.astype("float32")
    array = np.where(np.isfinite(array), np.clip(array, 0, 1.5), NODATA_VALUE).astype("float32")

    x, y = composite.x.values, composite.y.values
    geo = {"x0": float(x[0] - RESOLUTION / 2), "y0": float(y[0] + RESOLUTION / 2),
           "res": float(RESOLUTION), "epsg": 32646}

    path = PROCESSED_DIR / f"guwahati_landsat_{year}.tif"
    write_raster(path, array, geo, nodata=NODATA_VALUE, descriptions=OUTPUT_NAMES)
    print(f"Landsat composite saved: {path}")
    return path


if __name__ == "__main__":
    for yr in PERIODS:
        create_landsat_composite(yr)
