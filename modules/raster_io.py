"""
Small raster I/O layer.

Uses rasterio when it is installed (normal setup). Falls back to tifffile
for plain north-up GeoTIFFs in a projected CRS, so the pipeline can also
run on machines without GDAL.

Arrays are always returned band-first: (bands, rows, cols).
Georeferencing is returned as a dict: {"x0", "y0", "res", "epsg", "nodata"}
where (x0, y0) is the top-left corner of the top-left pixel.
"""

from pathlib import Path
import numpy as np

try:
    import rasterio
    from rasterio.transform import from_origin
    HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    HAS_RASTERIO = False
    import tifffile


def read_raster(path):
    path = Path(path)

    if HAS_RASTERIO:
        with rasterio.open(path) as src:
            array = src.read()
            t = src.transform
            geo = {
                "x0": t.c,
                "y0": t.f,
                "res": t.a,
                "epsg": src.crs.to_epsg() if src.crs else 32646,
                "nodata": src.nodata,
                "descriptions": list(src.descriptions),
            }
        return array, geo

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        array = page.asarray()
        tags = {tag.code: tag.value for tag in page.tags}

    if array.ndim == 2:
        array = array[np.newaxis]
    elif array.ndim == 3 and array.shape[2] < array.shape[0]:
        array = np.moveaxis(array, -1, 0)

    scale = tags[33550]
    tie = tags[33922]
    geokeys = tags.get(34735, ())
    epsg = 32646
    for i in range(4, len(geokeys), 4):
        if geokeys[i] == 3072:
            epsg = geokeys[i + 3]

    nodata = tags.get(42113)
    if nodata is not None:
        nodata = float(str(nodata).strip("\x00 "))

    geo = {
        "x0": float(tie[3]),
        "y0": float(tie[4]),
        "res": float(scale[0]),
        "epsg": int(epsg),
        "nodata": nodata,
        "descriptions": [None] * array.shape[0],
    }
    return array, geo


def write_raster(path, array, geo, nodata=None, descriptions=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if array.ndim == 2:
        array = array[np.newaxis]

    bands, rows, cols = array.shape

    if HAS_RASTERIO:
        profile = dict(
            driver="GTiff",
            height=rows,
            width=cols,
            count=bands,
            dtype=array.dtype.name,
            crs=f"EPSG:{geo['epsg']}",
            transform=from_origin(geo["x0"], geo["y0"], geo["res"], geo["res"]),
            compress="deflate",
            tiled=True,
        )
        if nodata is not None:
            profile["nodata"] = nodata
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array)
            if descriptions:
                for i, d in enumerate(descriptions, start=1):
                    dst.set_band_description(i, d)
        return path

    geokeys = (
        1, 1, 0, 4,
        1024, 0, 1, 1,            # projected model
        1025, 0, 1, 1,            # pixel is area
        3072, 0, 1, int(geo["epsg"]),
        3076, 0, 1, 9001,         # metres
    )
    extratags = [
        (33550, "d", 3, (geo["res"], geo["res"], 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, geo["x0"], geo["y0"], 0.0), False),
        (34735, "H", len(geokeys), geokeys, False),
    ]
    if nodata is not None:
        extratags.append((42113, "s", 0, f"{nodata:g}", False))

    data = np.moveaxis(array, 0, -1) if bands > 1 else array[0]
    tifffile.imwrite(
        path,
        data,
        photometric="minisblack",
        planarconfig="contig",
        compression="zlib",
        extratags=extratags,
    )
    return path
