"""
Forest Fire Detection & Spread Simulation -- Main Pipeline
=========================================================

Stages
------
1. data_gen   -- Generate synthetic terrain + weather; rasterise real VIIRS fire labels
2. preprocess -- Stack and normalise feature channels
3. train      -- Train U-Net (BCE + Dice loss, 40 epochs)
4. predict    -- Sliding-window inference -> fire probability GeoTIFF
5. simulate   -- Cellular Automata spread for 1/2/3/6/12 h
6. visualise  -- Maps, snapshots, animation, dashboard

Usage
-----
  python main.py                    # run all stages
  python main.py --skip data_gen    # skip data generation if already done
  python main.py --only simulate    # run only CA simulation
  python main.py --days 60          # number of synthetic days
"""

import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (MODEL_DIR, PROCESSED_DIR, SYNTHETIC_DIR,
                    PREDICTION_DIR, OUTPUT_DIR)


# ----------------------------------------------------------------------------
# Stage runners
# ----------------------------------------------------------------------------

def stage_data_gen(n_days):
    from src.data_generator import generate_all
    print("\n" + "="*60)
    print("STAGE 1 -- Data Generation")
    print("="*60)
    generate_all(n_days=n_days)


def stage_preprocess():
    from src.preprocessing import build_feature_stacks
    print("\n" + "="*60)
    print("STAGE 2 -- Preprocessing")
    print("="*60)
    build_feature_stacks()


def stage_train():
    from src.sklearn_model import train as rf_train
    print("\n" + "="*60)
    print("STAGE 3 -- Training Random Forest (sklearn)")
    print("="*60)
    model = rf_train()
    return model


def stage_predict():
    from src.sklearn_model import predict as rf_predict
    print("\n" + "="*60)
    print("STAGE 4 -- Prediction")
    print("="*60)
    prob_map, binary = rf_predict(day_idx='last')
    return prob_map, binary


def stage_simulate(prob_map=None):
    from src.cellular_automata import run_simulation
    print("\n" + "="*60)
    print("STAGE 5 -- Fire Spread Simulation (CA)")
    print("="*60)
    results = run_simulation(prob_map=prob_map)
    return results


def stage_visualise(prob_map, binary, results):
    from src.visualization import (plot_fire_probability, plot_binary_map,
                                   plot_spread_snapshots, create_animation,
                                   plot_training_curve, create_dashboard)
    print("\n" + "="*60)
    print("STAGE 6 -- Visualisation")
    print("="*60)

    dem = None
    dem_path = os.path.join(PROCESSED_DIR, 'dem.npy')
    if os.path.exists(dem_path):
        dem = np.load(dem_path)

    plot_fire_probability(prob_map, dem)
    plot_binary_map(binary, dem)
    plot_spread_snapshots(results, dem)
    create_animation(results, dem)
    plot_training_curve()
    create_dashboard(prob_map, binary, results, dem)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Forest Fire Detection & Simulation Pipeline')
    parser.add_argument('--skip', nargs='+', default=[],
                        choices=['data_gen', 'preprocess', 'train', 'predict', 'simulate', 'visualise'],
                        help='Skip specific stages')
    parser.add_argument('--only', type=str, default=None,
                        choices=['data_gen', 'preprocess', 'train', 'predict', 'simulate', 'visualise'],
                        help='Run only this stage (and prerequisites in memory)')
    parser.add_argument('--days', type=int, default=60,
                        help='Number of synthetic daily datasets to generate (default 60)')
    args = parser.parse_args()

    skip = set(args.skip)

    print("\n" + "="*62)
    print("  Forest Fire Detection & Spread Simulation")
    print("  Region: Uttarakhand, India  |  Resolution: 30 m")
    print("="*62)

    if args.only:
        # Jump directly to the requested stage (assumes earlier artefacts exist)
        prob_map = binary = results = None
        if args.only == 'data_gen':   stage_data_gen(args.days)
        elif args.only == 'preprocess': stage_preprocess()
        elif args.only == 'train':    stage_train()
        elif args.only == 'predict':  stage_predict()
        elif args.only == 'simulate': stage_simulate()
        elif args.only == 'visualise':
            prob_map, binary = stage_predict()
            results          = stage_simulate(prob_map)
            stage_visualise(prob_map, binary, results)
        return

    prob_map = binary = results = None

    if 'data_gen' not in skip:
        stage_data_gen(args.days)

    if 'preprocess' not in skip:
        stage_preprocess()

    if 'train' not in skip:
        stage_train()

    if 'predict' not in skip:
        prob_map, binary = stage_predict()

    if 'simulate' not in skip:
        results = stage_simulate(prob_map)

    if 'visualise' not in skip:
        if prob_map is None:
            import rasterio
            with rasterio.open(os.path.join(PREDICTION_DIR, 'fire_prob_nextday.tif')) as src:
                prob_map = src.read(1).astype('float32')
            with rasterio.open(os.path.join(PREDICTION_DIR, 'fire_binary_nextday.tif')) as src:
                binary = src.read(1).astype('uint8')
        if results is None:
            from src.cellular_automata import run_simulation
            results = run_simulation(prob_map)
        stage_visualise(prob_map, binary, results)

    print("\n" + "="*62)
    print("  Pipeline complete!  Outputs:")
    print(f"    Prediction maps  -> {PREDICTION_DIR}")
    print(f"    Spread maps      -> outputs/spread_maps/")
    print(f"    Animation        -> outputs/animations/")
    print(f"    Dashboard        -> outputs/dashboard.png")
    print("="*62 + "\n")


if __name__ == '__main__':
    main()
