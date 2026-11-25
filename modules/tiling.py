"""
State-wide scale-up: split Assam into processing tiles.

Each tile is processed independently with exactly the same pipeline as the
Guwahati demo (fetch -> harmonise -> classify -> change), so the work can be
spread across machines or run tile-by-tile overnight. Results are written per
tile and mosaicked / summarised afterwards (district-level statistics can be
produced by zonal sums of the per-tile change rasters).

Usage:
    python modules/tiling.py              # writes outputs/assam_tiles.geojson
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import ASSAM_BBOX, OUTPUT_DIR, TILE_SIZE_DEG  # noqa: E402


def make_tiles(bbox=ASSAM_BBOX, size=TILE_SIZE_DEG):
    lon0, lat0, lon1, lat1 = bbox
    nx = math.ceil((lon1 - lon0) / size)
    ny = math.ceil((lat1 - lat0) / size)
    tiles = []
    for j in range(ny):
        for i in range(nx):
            w = round(lon0 + i * size, 4)
            s = round(lat0 + j * size, 4)
            tiles.append({
                "tile_id": f"AS_{j:02d}_{i:02d}",
                "bbox": [w, s, round(min(w + size, lon1), 4), round(min(s + size, lat1), 4)],
                # UTM zone 45N west of 90E, 46N for the rest of Assam
                "epsg": 32645 if w + size / 2 < 90 else 32646,
            })
    return tiles


def tiles_geojson(tiles):
    feats = []
    for t in tiles:
        w, s, e, n = t["bbox"]
        feats.append({
            "type": "Feature",
            "properties": {"tile_id": t["tile_id"], "epsg": t["epsg"], "status": "pending"},
            "geometry": {"type": "Polygon",
                         "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]},
        })
    return {"type": "FeatureCollection", "features": feats}


if __name__ == "__main__":
    tiles = make_tiles()
    out = OUTPUT_DIR / "assam_tiles.geojson"
    out.write_text(json.dumps(tiles_geojson(tiles)))
    print(f"{len(tiles)} tiles of {TILE_SIZE_DEG} deg written to {out}")
