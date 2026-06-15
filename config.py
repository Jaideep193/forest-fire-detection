import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SYNTHETIC_DIR = os.path.join(DATA_DIR, 'synthetic')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
PREDICTION_DIR = os.path.join(OUTPUT_DIR, 'prediction_maps')
SPREAD_DIR = os.path.join(OUTPUT_DIR, 'spread_maps')
ANIMATION_DIR = os.path.join(OUTPUT_DIR, 'animations')

RAW_FIRE_CSV = os.path.join(BASE_DIR, 'fire_archive_J1V-C2_762685.csv')

# Study area: hottest fire-density sub-region of Uttarakhand
# 512×512 cells × 30 m ≈ 15.36 km × 15.36 km
STUDY_AREA = {
    'lat_min': 29.481,
    'lat_max': 29.619,
    'lon_min': 79.470,
    'lon_max': 79.630,
}

# GeoTIFF config — UTM Zone 44N (EPSG:32644), top-left origin
GEO_CONFIG = {
    'crs': 'EPSG:32644',
    # Approximate UTM easting/northing for NW corner (29.619°N, 79.470°E)
    'origin_easting': 338500.0,
    'origin_northing': 3279000.0,
    'pixel_size': 30.0,
    'grid_size': (512, 512),
}

# VIIRS fire footprint: 375 m native → expand to half footprint at 30 m
VIIRS_FOOTPRINT_PIXELS = 6   # radius in 30-m pixels

# Feature channels fed to U-Net
FEATURE_NAMES = [
    'elevation', 'slope', 'aspect_sin', 'aspect_cos',
    'temperature', 'humidity', 'wind_speed', 'wind_dir_sin', 'wind_dir_cos',
    'rainfall', 'lulc_norm',
]
N_FEATURES = len(FEATURE_NAMES)

# LULC classes and CA fuel weights
# 0=water  1=dense forest  2=open forest/scrub  3=grassland
# 4=agriculture  5=urban  6=snow/barren
FUEL_WEIGHTS = {0: 0.0, 1: 0.95, 2: 0.75, 3: 0.60, 4: 0.20, 5: 0.05, 6: 0.0}

MODEL_CONFIG = {
    'in_channels': N_FEATURES,
    'features': [32, 64, 128, 256],
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'batch_size': 2,
    'num_epochs': 40,
    'tile_size': 128,
    'stride': 64,
    'pos_weight': 6.0,   # upweight fire pixels (severe class imbalance)
}

CA_CONFIG = {
    'time_steps_hours': [1, 2, 3, 6, 12],
    'ca_steps_per_hour': 10,
    'base_ignition_prob': 0.38,
    'wind_weight': 0.40,
    'slope_weight': 0.25,
    'moisture_weight': 0.20,
    'fire_prob_threshold': 0.55,  # threshold to seed initial fire
    'fire_seed_fraction': 0.02,   # fraction of high-risk cells seeded
}
