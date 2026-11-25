"""
Minimal GeoPackage polygon reader (no GDAL / geopandas needed).

Reads the training polygons drawn in QGIS and returns plain Python
structures: [{"fid", "class_id", "class_name", "source_year", "rings"}]
where rings is a list of polygons, each a list of (N, 2) numpy arrays
(exterior ring first, then holes), in the layer's CRS (EPSG:32646).
"""

import sqlite3
import struct
import numpy as np

_ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


def _read_wkb(buf, offset=0):
    byte_order = buf[offset]
    end = "<" if byte_order == 1 else ">"
    geom_type = struct.unpack_from(end + "I", buf, offset + 1)[0]
    offset += 5

    has_z = (geom_type // 1000) in (1, 3) or bool(geom_type & 0x80000000)
    has_m = (geom_type // 1000) in (2, 3) or bool(geom_type & 0x40000000)
    base = (geom_type & 0x0FFFFFFF) % 1000
    dims = 2 + int(has_z) + int(has_m)

    def read_ring(off):
        n = struct.unpack_from(end + "I", buf, off)[0]
        off += 4
        coords = np.frombuffer(buf, dtype=end + "f8", count=n * dims, offset=off)
        off += n * dims * 8
        return coords.reshape(n, dims)[:, :2].copy(), off

    if base == 3:  # Polygon
        n_rings = struct.unpack_from(end + "I", buf, offset)[0]
        offset += 4
        rings = []
        for _ in range(n_rings):
            ring, offset = read_ring(offset)
            rings.append(ring)
        return [rings], offset

    if base == 6:  # MultiPolygon
        n_polys = struct.unpack_from(end + "I", buf, offset)[0]
        offset += 4
        polys = []
        for _ in range(n_polys):
            sub, offset = _read_wkb(buf, offset)
            polys.extend(sub)
        return polys, offset

    raise ValueError(f"Unsupported geometry type {geom_type}")


def parse_gpkg_geometry(blob):
    flags = blob[3]
    envelope = _ENVELOPE_BYTES[(flags >> 1) & 0b111]
    polygons, _ = _read_wkb(blob, 8 + envelope)
    return polygons


def read_training_polygons(path, layer="lulc_training"):
    con = sqlite3.connect(str(path))
    rows = con.execute(
        f"SELECT fid, geometry, class_id, class_name, source_year FROM '{layer}'"
    ).fetchall()
    con.close()

    features = []
    for fid, blob, class_id, class_name, year in rows:
        if blob is None or year is None or class_id is None:
            continue
        features.append({
            "fid": int(fid),
            "class_id": int(class_id),
            "class_name": class_name,
            "source_year": int(year),
            "polygons": parse_gpkg_geometry(blob),
        })
    return features
