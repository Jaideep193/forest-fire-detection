"""
Feature stacking and normalisation.

Reads the static terrain rasters and per-day weather rasters, assembles
the 11-channel feature cube, normalises each channel, and saves compact
numpy (.npy) arrays to data/processed/.
"""

import os, sys, csv
import numpy as np
import rasterio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SYNTHETIC_DIR, PROCESSED_DIR, FEATURE_NAMES, GEO_CONFIG


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _load(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def _normalise(arr, lo=None, hi=None):
    lo = lo if lo is not None else float(arr.min())
    hi = hi if hi is not None else float(arr.max())
    if hi == lo:
        return np.zeros_like(arr)
    return np.clip((arr - lo) / (hi - lo), 0, 1).astype(np.float32)


# ----------------------------------------------------------------------------
# Build per-day feature stacks
# ----------------------------------------------------------------------------

def build_feature_stacks(syn_dir=SYNTHETIC_DIR, out_dir=PROCESSED_DIR):
    os.makedirs(out_dir, exist_ok=True)

    # -- Static layers --------------------------------------------------------
    dem = _load(os.path.join(syn_dir, 'dem.tif'))
    slope = _load(os.path.join(syn_dir, 'slope.tif'))
    aspect = _load(os.path.join(syn_dir, 'aspect.tif'))
    lulc = _load(os.path.join(syn_dir, 'lulc.tif'))

    elev_norm = _normalise(dem, 0, 5000)
    slope_norm = _normalise(slope, 0, 60)
    asp_sin = ((np.sin(np.radians(aspect)) + 1) / 2).astype(np.float32)
    asp_cos = ((np.cos(np.radians(aspect)) + 1) / 2).astype(np.float32)
    lulc_norm = (lulc / 6.0).astype(np.float32)

    static = np.stack([elev_norm, slope_norm, asp_sin, asp_cos, lulc_norm], axis=0)  # (5, H, W)

    # -- Normalisation stats (from static) -----------------------------------
    stats = {
        'temperature': (0, 45),
        'humidity': (10, 100),
        'rainfall': (0, 25),
        'wind_speed': (0, 30),
    }

    # -- Day loop -------------------------------------------------------------
    meta_path = os.path.join(syn_dir, 'days_meta.csv')
    with open(meta_path) as f:
        days = list(csv.DictReader(f))

    print(f"[Preprocess] Building feature stacks for {len(days)} days ...")
    for day_info in days:
        idx = int(day_info['idx'])
        day_dir = os.path.join(syn_dir, f'day_{idx:03d}')

        # Dynamic weather layers
        temp = _load(os.path.join(day_dir, 'temperature.tif'))
        hum  = _load(os.path.join(day_dir, 'humidity.tif'))
        rain = _load(os.path.join(day_dir, 'rainfall.tif'))
        ws   = _load(os.path.join(day_dir, 'wind_speed.tif'))
        wd   = _load(os.path.join(day_dir, 'wind_dir.tif'))

        temp_n = _normalise(temp, *stats['temperature'])
        hum_n  = _normalise(hum,  *stats['humidity'])
        rain_n = _normalise(rain, *stats['rainfall'])
        ws_n   = _normalise(ws,   *stats['wind_speed'])
        wd_sin = ((np.sin(np.radians(wd)) + 1) / 2).astype(np.float32)
        wd_cos = ((np.cos(np.radians(wd)) + 1) / 2).astype(np.float32)

        dynamic = np.stack([temp_n, hum_n, ws_n, wd_sin, wd_cos, rain_n], axis=0)  # (6, H, W)
        features = np.concatenate([static, dynamic], axis=0)                         # (11, H, W)

        label = (rasterio.open(os.path.join(day_dir, 'fire_labels.tif')).read(1) > 0).astype(np.uint8)

        np.save(os.path.join(out_dir, f'features_{idx:03d}.npy'), features)
        np.save(os.path.join(out_dir, f'labels_{idx:03d}.npy'),   label)

    print("[Preprocess] Done.")

    # -- Also save static layers for later (CA, visualisation) ---------------
    np.save(os.path.join(out_dir, 'dem.npy'),    dem)
    np.save(os.path.join(out_dir, 'slope.npy'),  slope)
    np.save(os.path.join(out_dir, 'aspect.npy'), aspect)
    np.save(os.path.join(out_dir, 'lulc.npy'),   lulc)

    return len(days)


def train_val_split(processed_dir=PROCESSED_DIR, val_frac=0.2, seed=0):
    """Return (train_indices, val_indices) based on chronological split."""
    meta_path = os.path.join(SYNTHETIC_DIR, 'days_meta.csv')
    with open(meta_path) as f:
        days = list(csv.DictReader(f))
    n = len(days)
    n_val = max(1, int(n * val_frac))
    # Chronological split: last val_frac days go to validation
    train_idx = list(range(n - n_val))
    val_idx   = list(range(n - n_val, n))
    return train_idx, val_idx


if __name__ == '__main__':
    build_feature_stacks()
