
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CLASSIFIED_DIR = BASE_DIR / "data" / "classified"

OUTPUT_DIR = BASE_DIR / "outputs"
MAP_DIR = OUTPUT_DIR / "maps"
REPORT_DIR = OUTPUT_DIR / "reports"
STATS_DIR = OUTPUT_DIR / "statistics"

for folder in [
    RAW_DIR,
    PROCESSED_DIR,
    CLASSIFIED_DIR,
    MAP_DIR,
    REPORT_DIR,
    STATS_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

AOI_BBOX = [
    91.65,
    26.05,
    91.90,
    26.25
]

TARGET_CRS = "EPSG:32646"
RESOLUTION = 10

STAC_URL = (
    "https://planetarycomputer.microsoft.com/api/stac/v1"
)

COLLECTION = "sentinel-2-l2a"

# "sentinel-2" (10 m, 2017+) or "landsat" (Landsat 8/9, 30 m -> 10 m, 2013+)
IMAGERY_SOURCE = "sentinel-2"

PERIODS = {
    "2020": "2020-01-01/2020-03-31",
    "2025": "2025-01-01/2025-03-31"
}


# -------------------------------------------------
# STATE-WIDE SCALE-UP (Assam) - tiled processing
# -------------------------------------------------

ASSAM_BBOX = [89.69, 24.13, 96.03, 28.00]   # lon_min, lat_min, lon_max, lat_max
TILE_SIZE_DEG = 0.25                        # ~27 km tiles, same size as the Guwahati AOI

# -------------------------------------------------
# LULC CLASS SCHEME
# -------------------------------------------------
# Training polygons are labelled with 6 classes (1-6). Tree cover and Scrub,
# and Barren and Agriculture, cannot be separated reliably from a single
# dry-season composite (polygon-held-out F1 of 0.47 and 0.02), so they are
# merged for mapping. Switch back by editing CLASS_MERGE.

TRAINING_CLASSES = {
    1: "Built-up", 2: "Tree cover", 3: "Agriculture",
    4: "Water", 5: "Scrub", 6: "Barren",
}

CLASS_MERGE = {1: 1, 2: 2, 3: 3, 4: 4, 5: 2, 6: 3}

LULC_CLASSES = {
    1: {"name": "Built-up", "color": "#E0457B"},
    2: {"name": "Vegetation", "color": "#1F7A4D"},
    3: {"name": "Agriculture & open land", "color": "#C9961A"},
    4: {"name": "Water", "color": "#2B7FD4"},
}

# Year whose radiometry other years are normalised to
REFERENCE_YEAR = "2020"

# Change detection
MIN_CHANGE_PIXELS = 9          # minimum mapping unit: 9 px at 10 m = 0.09 ha
MIN_CONFIDENCE = 0.55          # both dates must be at least this confident

# -------------------------------------------------
# REFERENCE LAYERS FOR THE DASHBOARD
# -------------------------------------------------

BHUVAN_WMS_URL = "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms"
BHUVAN_LULC_LAYER = "lulc:AS_LULC50K_1516"   # Assam LULC 1:50k, 2015-16

DASHBOARD_FILE = OUTPUT_DIR / "dashboard" / "geochange_dashboard.html"
