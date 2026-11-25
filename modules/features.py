"""
Radiometric harmonisation and feature engineering.

Why this module exists
----------------------
Since 25 Jan 2022 (processing baseline 04.00) ESA adds BOA_ADD_OFFSET = -1000
to Sentinel-2 L2A digital numbers. Planetary Computer serves the raw values,
so reflectance must be computed as (DN - 1000) / 10000 for those scenes.
The original pipeline divided by 10000 only, which made every 2025 band
~0.10 too bright and the 2020 and 2025 classes spectrally inconsistent.

harmonise_reflectance()  -> fixes rasters already written with the old code
match_to_reference()     -> optional relative normalisation (haze, sun angle)
build_feature_stack()    -> bands + spectral indices + 5x5 texture
"""

import numpy as np
from scipy import ndimage as ndi

BAND_NAMES = ["B02", "B03", "B04", "B08", "B11"]

FEATURE_NAMES = BAND_NAMES + [
    "NDVI", "NDWI", "NDBI", "BSI", "MNDWI",
    "B08_mean5", "B08_std5", "NDVI_mean5", "NDVI_std5",
]

BOA_OFFSET = 0.1          # 1000 DN / 10000
NODATA_VALUE = -9999.0


def harmonise_reflectance(array, year, already_offset_corrected=False):
    """Return float32 reflectance with NaN for nodata.

    array: (5, rows, cols) reflectance written by the old pipeline.
    For acquisitions on/after 2022-01-25 the BOA offset is removed.
    """
    a = array.astype("float32").copy()
    nodata = (a <= NODATA_VALUE + 10).any(axis=0) | ~np.isfinite(a).all(axis=0)
    if int(year) >= 2022 and not already_offset_corrected:
        a -= BOA_OFFSET
    a[:, nodata] = np.nan
    return a


def match_to_reference(src, ref, n_quantiles=201):
    """Per-band quantile mapping of src onto ref (relative normalisation).

    Removes residual haze / illumination differences between dates so that
    one model can be applied consistently to every year.
    """
    out = src.copy()
    q = np.linspace(0, 100, n_quantiles)
    for b in range(src.shape[0]):
        s, r = src[b], ref[b]
        valid = np.isfinite(s)
        if not valid.any() or not np.isfinite(r).any():
            continue
        sq = np.nanpercentile(s, q)
        rq = np.nanpercentile(r, q)
        out[b][valid] = np.interp(s[valid], sq, rq)
    return out


def _ratio(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (a + b)


def _local_mean_std(arr, size=5):
    x = np.nan_to_num(arr, nan=0.0)
    m = ndi.uniform_filter(x, size)
    m2 = ndi.uniform_filter(x * x, size)
    return m, np.sqrt(np.maximum(m2 - m * m, 0))


def build_feature_stack(reflectance):
    """reflectance: (5, rows, cols) B02,B03,B04,B08,B11 -> (14, rows, cols)."""
    blue, green, red, nir, swir = reflectance

    ndvi = _ratio(nir, red)
    ndwi = _ratio(green, nir)
    ndbi = _ratio(swir, nir)
    bsi = _ratio(swir + red, nir + blue)
    mndwi = _ratio(green, swir)

    nir_m, nir_s = _local_mean_std(nir)
    ndvi_m, ndvi_s = _local_mean_std(ndvi)

    stack = np.stack([
        blue, green, red, nir, swir,
        ndvi, ndwi, ndbi, bsi, mndwi,
        nir_m, nir_s, ndvi_m, ndvi_s,
    ]).astype("float32")

    invalid = ~np.isfinite(reflectance).all(axis=0)
    stack[:, invalid] = np.nan
    return stack
