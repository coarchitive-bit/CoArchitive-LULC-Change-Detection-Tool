"""
Automated satellite data acquisition for any area and any date window.

fetch_composite(bbox, start, end, source="sentinel-2", max_cloud=40)
    -> (reflectance (5, rows, cols) float32 with NaN nodata, geo dict)

Sources (Microsoft Planetary Computer STAC, free, no account needed):
  sentinel-2 : Sentinel-2 L2A, 10 m, 2017 onwards. SCL cloud/shadow mask,
               processing-baseline 04.00 offset removed per scene.
  landsat    : Landsat 8/9 Collection 2 L2, 30 m resampled to 10 m, 2013
               onwards. QA_PIXEL cloud/shadow mask.

Every period of one analysis is fetched on the same UTM grid (same bbox,
EPSG and resolution), so classified maps are pixel-aligned for change
detection.
"""

import numpy as np

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_BANDS = ["B02", "B03", "B04", "B08", "B11"]
LS_BANDS = ["blue", "green", "red", "nir08", "swir16"]
S2_INVALID_SCL = [0, 1, 3, 7, 8, 9, 10, 11]   # nodata, saturated, shadow, cloud, cirrus, snow
BOA_OFFSET_START = np.datetime64("2022-01-25")


def _catalog():
    import planetary_computer
    from pystac_client import Client
    return Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def search_scenes(bbox, start, end, source="sentinel-2", max_cloud=40, max_scenes=12):
    query = {"eo:cloud_cover": {"lt": max_cloud}}
    if source == "landsat":
        collection = "landsat-c2-l2"
        query["platform"] = {"in": ["landsat-8", "landsat-9"]}
    else:
        collection = "sentinel-2-l2a"

    items = list(_catalog().search(
        collections=[collection], bbox=bbox, datetime=f"{start}/{end}",
        query=query, max_items=200).items())
    items.sort(key=lambda i: i.properties.get("eo:cloud_cover", 100))
    return items[:max_scenes], len(items)


def _offset_flags(items):
    flags = {}
    for item in items:
        pb = item.properties.get("s2:processing_baseline")
        try:
            flags[item.id] = float(pb) >= 4.0
        except (TypeError, ValueError):
            flags[item.id] = np.datetime64(item.datetime.replace(tzinfo=None)) >= BOA_OFFSET_START
    return flags


def fetch_composite(bbox, start, end, source="sentinel-2", max_cloud=40,
                    max_scenes=12, resolution=10, epsg=None, log=print):
    import stackstac
    import xarray as xr
    from rasterio.enums import Resampling
    from modules.geo_utils import utm_epsg_for

    log("fetch module v3.1 (NaN fill) loaded from " + __file__)
    epsg = epsg or utm_epsg_for((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    items, found = search_scenes(bbox, start, end, source, max_cloud, max_scenes)
    if not items:
        raise ValueError(
            f"No {source} scenes with <{max_cloud}% cloud between {start} and {end}. "
            "Widen the date window or raise the cloud limit.")
    log(f"{found} scenes found, using the {len(items)} least cloudy "
        f"({', '.join(sorted({str(i.datetime.date()) for i in items})[:6])}...)")

    common = dict(bounds_latlon=bbox, epsg=epsg, resolution=resolution,
                  chunksize=1024, rescale=False)
    flt = dict(dtype="float32", fill_value=np.float32(np.nan))   # spectral bands
    msk = dict(dtype="uint16", fill_value=np.uint16(0))          # SCL / QA mask bands

    if source == "landsat":
        data = stackstac.stack(items, assets=LS_BANDS, **flt,
                               resampling=Resampling.bilinear, **common)
        qa = stackstac.stack(items, assets=["qa_pixel"], **msk,
                             resampling=Resampling.nearest, **common).sel(band="qa_pixel")
        bad = (1 << 0) | (1 << 1) | (1 << 3) | (1 << 4) | (1 << 5)
        refl = (data * 0.0000275 - 0.2).where(((qa & bad) == 0) & (data > 0))
    else:
        data = stackstac.stack(items, assets=S2_BANDS, **flt,
                               resampling=Resampling.bilinear, **common)
        scl = stackstac.stack(items, assets=["SCL"], **msk,
                              resampling=Resampling.nearest, **common).sel(band="SCL")
        flags = _offset_flags(items)
        offsets = xr.DataArray(
            [1000.0 if flags.get(str(i), False) else 0.0 for i in data["id"].values],
            dims="time", coords={"time": data["time"]})
        log(f"BOA offset removed from {int((offsets > 0).sum())} of {len(items)} scenes")
        refl = ((data - offsets) / 10000.0).where(~scl.isin(S2_INVALID_SCL) & (data > 0))

    log("Building cloud-free median composite...")
    comp = refl.median(dim="time", skipna=True).compute()
    arr = comp.values.astype("float32")
    arr[~np.isfinite(arr)] = np.nan

    x, y = comp.x.values, comp.y.values
    geo = {"x0": float(x[0] - resolution / 2), "y0": float(y[0] + resolution / 2),
           "res": float(resolution), "epsg": int(epsg), "nodata": None}
    valid = np.isfinite(arr).all(axis=0).mean() * 100
    log(f"Composite ready: {arr.shape[2]} x {arr.shape[1]} px, {valid:.1f}% clear pixels")
    return arr, geo, {"scenes_found": found, "scenes_used": len(items),
                      "dates": sorted({str(i.datetime.date()) for i in items}),
                      "clear_percent": round(float(valid), 1)}