"""
Data generation module.

Synthetic terrain (DEM -> slope/aspect, LULC) and weather are generated for
the study tile.  Real VIIRS fire points are read from the CSV and rasterised
to the 30 m grid to produce daily binary fire-label rasters.

All outputs are saved as GeoTIFF files under data/synthetic/.
"""

import os, sys, csv, gc
import numpy as np
from scipy.ndimage import gaussian_filter, zoom as _zoom, distance_transform_edt
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (SYNTHETIC_DIR, GEO_CONFIG, STUDY_AREA,
                    FUEL_WEIGHTS, RAW_FIRE_CSV, VIIRS_FOOTPRINT_PIXELS)


# ----------------------------------------------------------------------------
# GeoTIFF helpers
# ----------------------------------------------------------------------------

def _transform(geo):
    return from_origin(geo['origin_easting'], geo['origin_northing'],
                       geo['pixel_size'], geo['pixel_size'])


def save_raster(path, array, dtype, geo, nodata=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = array if array.ndim == 3 else array[np.newaxis]
    t = _transform(geo)
    with rasterio.open(path, 'w', driver='GTiff',
                       height=arr.shape[1], width=arr.shape[2],
                       count=arr.shape[0], dtype=dtype,
                       crs=geo['crs'], transform=t, nodata=nodata) as dst:
        for i in range(arr.shape[0]):
            dst.write(arr[i].astype(dtype), i + 1)


def load_raster(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


# ----------------------------------------------------------------------------
# Terrain generation
# ----------------------------------------------------------------------------

def generate_dem(grid_size=(512, 512), seed=42):
    """Multi-scale layered noise -> Himalayan DEM (500-3 500 m)."""
    np.random.seed(seed)
    H, W = grid_size
    dem = np.zeros((H, W), dtype=np.float32)

    for sz, amp in [(4, 2200), (8, 900), (16, 450), (32, 200), (64, 80), (128, 35)]:
        raw = np.random.randn(sz, sz).astype(np.float32)
        up = _zoom(raw, (H / sz, W / sz), order=1)[:H, :W]
        dem += amp * up

    y, x = np.ogrid[:H, :W]
    dem += (np.exp(-((y - x * 0.6 - 60) ** 2) / (2 * 110 ** 2)) * 1600).astype(np.float32)
    dem -= (np.exp(-((y - H * 0.70) ** 2) / (2 * 55 ** 2)) * 700).astype(np.float32)

    dem -= dem.min()
    dem = dem / dem.max() * 3000 + 500
    return dem


def compute_slope_aspect(dem, pixel_size=30.0):
    dy_g, dx_g = np.gradient(dem.astype(np.float64), pixel_size)
    slope = np.degrees(np.arctan(np.sqrt(dx_g ** 2 + dy_g ** 2))).astype(np.float32)
    aspect = (np.degrees(np.arctan2(dx_g, -dy_g)) % 360).astype(np.float32)
    return slope, aspect


def generate_lulc(dem, seed=42):
    """
    Elevation-based LULC:
    0=water  1=dense forest  2=open forest  3=grassland  4=agri  5=urban  6=snow
    """
    np.random.seed(seed + 1)
    H, W = dem.shape
    noise = gaussian_filter(np.random.randn(H, W).astype(np.float32), sigma=18) * 160
    eff = dem + noise

    lulc = np.zeros((H, W), dtype=np.int32)
    lulc[eff < 700] = 4
    lulc[(eff >= 700) & (eff < 1750)] = 1
    lulc[(eff >= 1750) & (eff < 2450)] = 2
    lulc[(eff >= 2450) & (eff < 3150)] = 3
    lulc[eff >= 3150] = 6

    y, x = np.ogrid[:H, :W]
    lulc[np.abs(y - W * 0.70 - x * 0.10) < 5] = 0
    lulc[np.abs(x - H * 0.30 + y * 0.05) < 4] = 0

    for cy, cx in [(int(H*0.72), int(W*0.28)), (int(H*0.50), int(W*0.68)), (int(H*0.82), int(W*0.60))]:
        r = 10
        dy_p, dx_p = np.ogrid[-r:r+1, -r:r+1]
        mask = dy_p**2 + dx_p**2 <= r**2
        ys = np.clip(cy + np.arange(-r, r+1), 0, H-1)
        xs = np.clip(cx + np.arange(-r, r+1), 0, W-1)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        valid = mask & (dem[yy, xx] < 2000)
        lulc[yy[valid], xx[valid]] = 5

    return lulc


def compute_settlement_distance(lulc):
    """Euclidean distance (m) to nearest urban cell."""
    mask = (lulc == 5)
    if not mask.any():
        return np.full(lulc.shape, 9999.0, dtype=np.float32)
    return (distance_transform_edt(~mask) * 30.0).astype(np.float32)


# ----------------------------------------------------------------------------
# Weather generation (synthetic, correlated with fire activity)
# ----------------------------------------------------------------------------

def _gf(arr, sigma):
    """gaussian_filter with limited truncation to avoid FFT and keep memory low."""
    return gaussian_filter(arr, sigma=sigma, truncate=2.0)


def generate_weather(dem, n_fires_today, day_idx=0, seed=100):
    """
    Generate one day of spatially-varying synthetic weather.
    Higher n_fires_today -> hotter, drier, windier conditions.
    Uses small sigma values (direct convolution, no FFT) to keep memory use low.
    """
    np.random.seed(seed + day_idx * 7)
    H, W = dem.shape

    fire_intensity = np.clip(n_fires_today / 20.0, 0, 1)

    # Temperature: lapse rate + fire-day boost
    noise_t = _gf(np.random.randn(H, W).astype(np.float32), 8) * 3
    base_t = (28 + fire_intensity * 10 - (dem - 500) * 6.5e-3
              + float(np.random.uniform(-3, 3)))
    temp = np.clip(base_t + noise_t, -5, 45).astype(np.float32)
    del noise_t

    # Humidity
    noise_h = _gf(np.random.randn(H, W).astype(np.float32), 6) * 8
    base_h = 80 - fire_intensity * 40 + float(np.random.uniform(-10, 10))
    humidity = np.clip(base_h + noise_h, 10, 100).astype(np.float32)
    del noise_h

    # Rainfall
    rain_base = max(0.0, float(np.random.normal(1.5 - fire_intensity * 1.4, 1.5)))
    rain_sp = _gf(np.random.rand(H, W).astype(np.float32), 8)
    rainfall = np.clip(rain_base * rain_sp, 0, 25).astype(np.float32)
    del rain_sp

    # Wind speed
    noise_ws = _gf(np.random.randn(H, W).astype(np.float32), 5) * 3
    base_ws = 5 + fire_intensity * 12 + float(np.random.uniform(-2, 2))
    wind_speed = np.clip(base_ws + noise_ws, 0.5, 30).astype(np.float32)
    del noise_ws

    # Wind direction
    noise_wd = _gf(np.random.randn(H, W).astype(np.float32), 5) * 20
    base_wd = float(np.random.uniform(0, 360))
    wind_dir = ((base_wd + noise_wd) % 360).astype(np.float32)
    del noise_wd

    return {
        'temperature': temp,
        'humidity':    humidity,
        'rainfall':    rainfall,
        'wind_speed':  wind_speed,
        'wind_dir':    wind_dir,
    }


# ----------------------------------------------------------------------------
# VIIRS rasterisation
# ----------------------------------------------------------------------------

def load_viirs_by_date(csv_path=RAW_FIRE_CSV):
    """Load VIIRS CSV -> {date_str: [(lat, lon), ...]} for the study tile."""
    sa = STUDY_AREA
    by_date = {}
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            lat, lon = float(row['latitude']), float(row['longitude'])
            if sa['lat_min'] <= lat <= sa['lat_max'] and sa['lon_min'] <= lon <= sa['lon_max']:
                d = row['acq_date']
                by_date.setdefault(d, []).append((lat, lon))
    return by_date


def latlon_to_pixel(lat, lon, grid_size, study_area):
    """Convert geographic coordinates to pixel (row, col) - row=0 at top (north)."""
    H, W = grid_size
    row = int((study_area['lat_max'] - lat) / (study_area['lat_max'] - study_area['lat_min']) * H)
    col = int((lon - study_area['lon_min']) / (study_area['lon_max'] - study_area['lon_min']) * W)
    return np.clip(row, 0, H-1), np.clip(col, 0, W-1)


def rasterise_fires(fire_points, grid_size=(512, 512), radius=VIIRS_FOOTPRINT_PIXELS):
    """
    Burn VIIRS lat/lon points onto a binary grid.
    Each point expands to a disk of `radius` pixels to account for the
    ~375 m VIIRS footprint at 30 m grid resolution.
    """
    H, W = grid_size
    raster = np.zeros((H, W), dtype=np.uint8)
    y_idx, x_idx = np.ogrid[:H, :W]

    for lat, lon in fire_points:
        r, c = latlon_to_pixel(lat, lon, grid_size, STUDY_AREA)
        disk = (y_idx - r) ** 2 + (x_idx - c) ** 2 <= radius ** 2
        raster[disk] = 1

    return raster


# ----------------------------------------------------------------------------
# Master generation routine
# ----------------------------------------------------------------------------

def generate_all(n_days=60, seed=42):
    """
    Generate the complete synthetic dataset and save to SYNTHETIC_DIR.

    - Static layers: DEM, slope, aspect, LULC, settlement_dist
    - Dynamic layers (per day): weather + fire labels from real VIIRS data
    """
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)
    geo = GEO_CONFIG
    gs = geo['grid_size']

    print("[DataGen] Building terrain ...")
    dem = generate_dem(gs, seed)
    slope, aspect = compute_slope_aspect(dem, geo['pixel_size'])
    lulc = generate_lulc(dem, seed)
    settle_dist = compute_settlement_distance(lulc)

    save_raster(os.path.join(SYNTHETIC_DIR, 'dem.tif'),          dem,        'float32', geo)
    save_raster(os.path.join(SYNTHETIC_DIR, 'slope.tif'),        slope,      'float32', geo)
    save_raster(os.path.join(SYNTHETIC_DIR, 'aspect.tif'),       aspect,     'float32', geo)
    save_raster(os.path.join(SYNTHETIC_DIR, 'lulc.tif'),         lulc,       'int32',   geo)
    save_raster(os.path.join(SYNTHETIC_DIR, 'settle_dist.tif'),  settle_dist,'float32', geo)

    print("[DataGen] Loading VIIRS fire data ...")
    viirs = load_viirs_by_date()
    all_dates = sorted(viirs.keys())
    print(f"  Found {len(all_dates)} dates with >=1 fire in study tile "
          f"({sum(len(v) for v in viirs.values())} total points)")

    # Build day list: include all fire-active dates + some no-fire days
    fire_dates = all_dates[:n_days] if len(all_dates) >= n_days else all_dates
    # Pad with synthetic no-fire entries so we have n_days
    no_fire_dates = [f"2024-07-{i+1:02d}" for i in range(max(0, n_days - len(fire_dates)))]
    day_list = fire_dates + no_fire_dates
    day_list = day_list[:n_days]

    print(f"[DataGen] Generating {len(day_list)} daily feature+label stacks ...")
    days_meta = []  # [(date_str, n_fires)]

    for idx, date_str in enumerate(day_list):
        fire_pts = viirs.get(date_str, [])
        n_fires = len(fire_pts)

        weather = generate_weather(dem, n_fires, day_idx=idx, seed=seed + idx * 13)
        fire_map = rasterise_fires(fire_pts, gs)

        day_dir = os.path.join(SYNTHETIC_DIR, f'day_{idx:03d}')
        os.makedirs(day_dir, exist_ok=True)

        for name, arr in weather.items():
            save_raster(os.path.join(day_dir, f'{name}.tif'), arr, 'float32', geo)
        save_raster(os.path.join(day_dir, 'fire_labels.tif'), fire_map, 'uint8', geo)

        # Save metadata
        days_meta.append({'idx': idx, 'date': date_str, 'n_fires': n_fires})

        # Explicit memory cleanup to avoid accumulation over 60 iterations
        del weather, fire_map
        gc.collect()

        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(day_list)} - {date_str} | fires: {n_fires}")

    # Save day list as CSV for reproducibility
    import csv as _csv
    meta_path = os.path.join(SYNTHETIC_DIR, 'days_meta.csv')
    with open(meta_path, 'w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=['idx', 'date', 'n_fires'])
        w.writeheader(); w.writerows(days_meta)

    print("[DataGen] Done.")
    return dem, slope, aspect, lulc


if __name__ == '__main__':
    generate_all()
