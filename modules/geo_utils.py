"""
Coordinate helpers without GDAL/pyproj (WGS84 <-> UTM north, Snyder formulas),
plus warping of project rasters to lat/lon PNG overlays for web maps.
"""

import numpy as np

A = 6378137.0
F = 1 / 298.257223563
E2 = F * (2 - F)
EP2 = E2 / (1 - E2)
K0 = 0.9996


def utm_epsg_for(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _zone(epsg):
    return epsg % 100, epsg < 32700


def latlon_to_utm(lat, lon, epsg):
    zone, north = _zone(epsg)
    lat = np.radians(np.asarray(lat, dtype="float64"))
    lon = np.radians(np.asarray(lon, dtype="float64"))
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    t = np.tan(lat) ** 2
    c = EP2 * np.cos(lat) ** 2
    a = np.cos(lat) * (lon - lon0)
    m = A * ((1 - E2 / 4 - 3 * E2**2 / 64 - 5 * E2**3 / 256) * lat
             - (3 * E2 / 8 + 3 * E2**2 / 32 + 45 * E2**3 / 1024) * np.sin(2 * lat)
             + (15 * E2**2 / 256 + 45 * E2**3 / 1024) * np.sin(4 * lat)
             - (35 * E2**3 / 3072) * np.sin(6 * lat))
    x = K0 * n * (a + (1 - t + c) * a**3 / 6 + (5 - 18 * t + t**2 + 72 * c - 58 * EP2) * a**5 / 120) + 500000
    y = K0 * (m + n * np.tan(lat) * (a**2 / 2 + (5 - t + 9 * c + 4 * c**2) * a**4 / 24
                                     + (61 - 58 * t + t**2 + 600 * c - 330 * EP2) * a**6 / 720))
    if not north:
        y = y + 10000000
    return x, y


def utm_to_latlon(x, y, epsg):
    zone, north = _zone(epsg)
    x = np.asarray(x, dtype="float64") - 500000
    y = np.asarray(y, dtype="float64")
    if not north:
        y = y - 10000000
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)
    m = y / K0
    mu = m / (A * (1 - E2 / 4 - 3 * E2**2 / 64 - 5 * E2**3 / 256))
    e1 = (1 - np.sqrt(1 - E2)) / (1 + np.sqrt(1 - E2))
    p = (mu + (3 * e1 / 2 - 27 * e1**3 / 32) * np.sin(2 * mu)
         + (21 * e1**2 / 16 - 55 * e1**4 / 32) * np.sin(4 * mu)
         + (151 * e1**3 / 96) * np.sin(6 * mu))
    n1 = A / np.sqrt(1 - E2 * np.sin(p) ** 2)
    t1 = np.tan(p) ** 2
    c1 = EP2 * np.cos(p) ** 2
    r1 = A * (1 - E2) / (1 - E2 * np.sin(p) ** 2) ** 1.5
    d = x / (n1 * K0)
    lat = p - (n1 * np.tan(p) / r1) * (d**2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * EP2) * d**4 / 24
                                       + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * EP2 - 3 * c1**2) * d**6 / 720)
    lon = lon0 + (d - (1 + 2 * t1 + c1) * d**3 / 6
                  + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * EP2 + 24 * t1**2) * d**5 / 120) / np.cos(p)
    return np.degrees(lat), np.degrees(lon)


def raster_latlon_bounds(geo, shape):
    rows, cols = shape
    xs = [geo["x0"], geo["x0"] + cols * geo["res"]]
    ys = [geo["y0"] - rows * geo["res"], geo["y0"]]
    X, Y = np.meshgrid(xs, ys)
    lat, lon = utm_to_latlon(X.ravel(), Y.ravel(), geo["epsg"])
    return [[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]]


def warp_to_latlon(image, geo, max_size=1600):
    """Nearest-neighbour warp of a (rows, cols[, bands]) UTM image to a regular
    lat/lon grid so it overlays exactly on Web-Mercator maps.

    Returns (warped_image, [[south, west], [north, east]]).
    """
    rows, cols = image.shape[:2]
    (s, w), (n, e) = raster_latlon_bounds(geo, (rows, cols))
    scale = min(1.0, max_size / max(rows, cols))
    out_h, out_w = int(rows * scale), int(cols * scale)

    lats = np.linspace(n, s, out_h)
    lons = np.linspace(w, e, out_w)
    LON, LAT = np.meshgrid(lons, lats)
    X, Y = latlon_to_utm(LAT, LON, geo["epsg"])
    c = np.floor((X - geo["x0"]) / geo["res"]).astype(int)
    r = np.floor((geo["y0"] - Y) / geo["res"]).astype(int)
    inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)

    out = np.zeros((out_h, out_w) + image.shape[2:], dtype=image.dtype)
    out[inside] = image[r[inside], c[inside]]
    return out, [[s, w], [n, e]]
