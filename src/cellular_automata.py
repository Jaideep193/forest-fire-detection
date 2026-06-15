"""
Cellular Automata fire spread simulation.

Cell states
-----------
0  unburned
1  burning
2  burned out

Spread probability from a burning source cell to each of its 8 neighbours
is modulated by:
  * fuel  (LULC class)
  * wind  (direction alignment + speed)
  * slope (uphill spread faster)
  * moisture (high humidity reduces spread)

Simulation runs for CA_STEPS_PER_HOUR x hours_target iterations, saving
raster snapshots at each target hour.

All outputs are GeoTIFF at 30 m in data/outputs/spread_maps/.
"""

import os, sys
import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (CA_CONFIG, GEO_CONFIG, SPREAD_DIR, FUEL_WEIGHTS,
                    SYNTHETIC_DIR, PREDICTION_DIR, PROCESSED_DIR)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _make_transform(geo):
    return from_origin(geo['origin_easting'], geo['origin_northing'],
                       geo['pixel_size'], geo['pixel_size'])


def _save_state_tif(path, state, geo):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = _make_transform(geo)
    with rasterio.open(path, 'w', driver='GTiff',
                       height=state.shape[0], width=state.shape[1],
                       count=1, dtype='uint8',
                       crs=geo['crs'], transform=t) as dst:
        dst.write(state.astype(np.uint8), 1)


def _load1(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


# ----------------------------------------------------------------------------
# 8-neighbourhood offsets and their fire-spread angles
#   angle = direction FROM source cell TOWARD neighbour (met. convention)
#   0=N, 90=E, 180=S, 270=W
# ----------------------------------------------------------------------------
_NEIGHBOURS = [
    (-1, -1, 315), (-1, 0, 0),   (-1, 1, 45),
    ( 0, -1, 270),               ( 0, 1, 90),
    ( 1, -1, 225), ( 1, 0, 180), ( 1, 1, 135),
]


class FireSpreadCA:
    """
    Parameters
    ----------
    state      : (H, W) uint8  initial state (0/1/2)
    lulc       : (H, W) int32
    elevation  : (H, W) float32   metres
    wind_speed : (H, W) float32   m/s
    wind_dir   : (H, W) float32   degrees (meteorological -- direction FROM which wind comes)
    humidity   : (H, W) float32   %
    cfg        : CA_CONFIG dict
    """
    def __init__(self, state, lulc, elevation, wind_speed, wind_dir, humidity, cfg=CA_CONFIG):
        self.state     = state.copy().astype(np.uint8)
        self.lulc      = lulc.astype(np.int32)
        self.elevation = elevation.astype(np.float32)
        self.ws        = wind_speed.astype(np.float32)
        self.wd        = wind_dir.astype(np.float32)   # meteorological direction
        self.humidity  = humidity.astype(np.float32)
        self.base_prob = cfg['base_ignition_prob']
        self.ww        = cfg['wind_weight']
        self.sw        = cfg['slope_weight']
        self.mw        = cfg['moisture_weight']

        # Pre-compute fuel map
        self.fuel = np.vectorize(lambda v: FUEL_WEIGHTS.get(int(v), 0.0))(self.lulc).astype(np.float32)

        # Wind destination: where the wind is GOING (opposite of met. origin)
        self.wind_dest = (self.wd + 180) % 360   # direction wind blows toward

        self.step_count = 0

    def step(self):
        """One CA iteration -- vectorised over all 8 neighbours."""
        burning = (self.state == 1)
        if not burning.any():
            return False   # fire extinguished

        new_ignition = np.zeros_like(burning, dtype=bool)

        for dy, dx, spread_angle in _NEIGHBOURS:
            # shifted_burning[i,j] = burning[i-dy, j-dx]
            # i.e., "does position (i,j) have a burning neighbour to its (dy,dx) side"
            sb = np.roll(np.roll(burning, dy, axis=0), dx, axis=1)
            candidates = sb & (self.state == 0)
            if not candidates.any():
                continue

            # Wind at source -- shift source properties to target coordinate frame
            wd_src = np.roll(np.roll(self.wind_dest, dy, axis=0), dx, axis=1)
            ws_src = np.roll(np.roll(self.ws,       dy, axis=0), dx, axis=1)
            el_src = np.roll(np.roll(self.elevation, dy, axis=0), dx, axis=1)
            hm_src = np.roll(np.roll(self.humidity,  dy, axis=0), dx, axis=1)

            # Wind alignment: fire spreads faster downwind
            align = (1 + np.cos(np.radians(wd_src - spread_angle))) / 2   # [0,1]
            wind_f = 1 + self.ww * align * ws_src / 15.0

            # Slope: uphill (positive deltaelev) -> faster
            d_elev = self.elevation - el_src
            slope_angle = np.degrees(np.arctan(d_elev / 30.0))
            slope_f = np.clip(1 + self.sw * np.sin(np.radians(slope_angle)), 0.05, 3.0)

            # Moisture at source
            moisture_f = 1 - self.mw * hm_src / 100.0

            prob = self.base_prob * self.fuel * wind_f * slope_f * moisture_f
            prob = np.clip(prob, 0, 1)

            ignite = np.random.random(prob.shape) < prob
            new_ignition |= (candidates & ignite)

        # State transitions
        self.state[burning]     = 2   # burning -> burned
        self.state[new_ignition] = 1  # unburned -> burning
        self.step_count += 1
        return True

    def run(self, total_steps):
        for _ in range(total_steps):
            active = self.step()
            if not active:
                break


# ----------------------------------------------------------------------------
# Main simulation driver
# ----------------------------------------------------------------------------

def run_simulation(prob_map=None, cfg=CA_CONFIG):
    """
    Initialise fire from high-probability zones in the prediction map and
    simulate spread for all target time steps.

    Saves one GeoTIFF per time step to SPREAD_DIR.
    Returns a dict {hours: state_array}.
    """
    os.makedirs(SPREAD_DIR, exist_ok=True)
    geo = GEO_CONFIG

    # -- Load static rasters ----------------------------------------------
    lulc      = np.load(os.path.join(PROCESSED_DIR, 'lulc.npy')).astype(np.int32)
    elevation = np.load(os.path.join(PROCESSED_DIR, 'dem.npy')).astype(np.float32)

    # Latest day weather (use last available)
    import glob
    w_dirs = sorted(os.listdir(os.path.join(SYNTHETIC_DIR)))
    day_dirs = [d for d in w_dirs if d.startswith('day_')]
    last_day = os.path.join(SYNTHETIC_DIR, day_dirs[-1])

    def _ld(name):
        with rasterio.open(os.path.join(last_day, name)) as src:
            return src.read(1).astype(np.float32)

    wind_speed = _ld('wind_speed.tif')
    wind_dir   = _ld('wind_dir.tif')
    humidity   = _ld('humidity.tif')

    # -- Seed initial fire from prediction map -----------------------------
    if prob_map is None:
        prob_path = os.path.join(PREDICTION_DIR, 'fire_prob_nextday.tif')
        with rasterio.open(prob_path) as src:
            prob_map = src.read(1).astype(np.float32)

    threshold    = cfg['fire_prob_threshold']
    seed_frac    = cfg['fire_seed_fraction']
    high_risk    = prob_map >= threshold
    n_seed       = max(1, int(high_risk.sum() * seed_frac))
    risk_indices = np.argwhere(high_risk)
    np.random.shuffle(risk_indices)
    seed_pts     = risk_indices[:n_seed]

    initial_state = np.zeros_like(prob_map, dtype=np.uint8)
    for r, c in seed_pts:
        initial_state[r, c] = 1
    print(f"[CA] Seeded {n_seed} initial fire cells from {high_risk.sum()} high-risk pixels")

    # -- Run simulation ----------------------------------------------------
    sph     = cfg['ca_steps_per_hour']
    targets = sorted(cfg['time_steps_hours'])
    target_steps = {h: h * sph for h in targets}
    max_steps    = max(target_steps.values())

    ca = FireSpreadCA(initial_state, lulc, elevation, wind_speed, wind_dir, humidity, cfg)
    results = {0: initial_state.copy()}

    step = 0
    for h in targets:
        steps_needed = target_steps[h] - step
        ca.run(steps_needed)
        step = target_steps[h]
        results[h] = ca.state.copy()

        path = os.path.join(SPREAD_DIR, f'spread_{h:02d}h.tif')
        _save_state_tif(path, ca.state, geo)

        burned  = (ca.state == 2).sum()
        burning = (ca.state == 1).sum()
        area_ha = (burned + burning) * 30 * 30 / 10000
        print(f"  t={h:2d}h | burning={burning}  burned={burned} | "
              f"affected area  {area_ha:.1f} ha")

    return results


if __name__ == '__main__':
    run_simulation()
