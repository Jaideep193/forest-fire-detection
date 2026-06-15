"""
Sliding-window inference on the full 512x512 feature raster.

Loads the best U-Net checkpoint and runs patch-wise prediction with
overlap-averaging to eliminate tile boundary artefacts.

Outputs
-------
outputs/prediction_maps/fire_prob_nextday.tif   -- float32 probability [0,1]
outputs/prediction_maps/fire_binary_nextday.tif -- uint8  binary map
"""

import os, sys
import numpy as np
import torch
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (MODEL_DIR, PROCESSED_DIR, PREDICTION_DIR,
                    MODEL_CONFIG, GEO_CONFIG, SYNTHETIC_DIR)
from src.unet import UNet


def load_model(device, cfg=MODEL_CONFIG):
    ckpt = torch.load(os.path.join(MODEL_DIR, 'unet_best.pth'),
                      map_location=device)
    model = UNet(cfg['in_channels'], cfg['features']).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


def _make_transform(geo):
    return from_origin(geo['origin_easting'], geo['origin_northing'],
                       geo['pixel_size'], geo['pixel_size'])


def _save_tif(path, array, dtype, geo):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = _make_transform(geo)
    with rasterio.open(path, 'w', driver='GTiff',
                       height=array.shape[0], width=array.shape[1],
                       count=1, dtype=dtype,
                       crs=geo['crs'], transform=t) as dst:
        dst.write(array.astype(dtype), 1)


def predict_full_raster(day_idx='last', threshold=0.40, cfg=MODEL_CONFIG):
    """
    Run inference on a feature stack.

    Parameters
    ----------
    day_idx : int or 'last'   which day's features to use
    threshold : float         probability threshold for binary map
    """
    os.makedirs(PREDICTION_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Load features ------------------------------------------------------
    if day_idx == 'last':
        import glob
        paths = sorted(glob.glob(os.path.join(PROCESSED_DIR, 'features_*.npy')))
        feat_path = paths[-1]
    else:
        feat_path = os.path.join(PROCESSED_DIR, f'features_{day_idx:03d}.npy')

    features = np.load(feat_path)                     # (C, H, W)
    C, H, W  = features.shape

    # -- Sliding-window prediction ------------------------------------------
    T   = cfg['tile_size']
    S   = cfg['stride']
    model = load_model(device, cfg)

    prob_sum   = np.zeros((H, W), dtype=np.float32)
    count_map  = np.zeros((H, W), dtype=np.float32)

    rows = list(range(0, H - T + 1, S))
    cols = list(range(0, W - T + 1, S))
    if rows[-1] + T < H: rows.append(H - T)
    if cols[-1] + T < W: cols.append(W - T)

    with torch.no_grad():
        for r in rows:
            for c in cols:
                patch = torch.from_numpy(
                    features[:, r:r+T, c:c+T]).float().unsqueeze(0).to(device)
                pred = model(patch).squeeze().cpu().numpy()     # (T, T)
                prob_sum[r:r+T, c:c+T]  += pred
                count_map[r:r+T, c:c+T] += 1.0

    prob_map = prob_sum / np.maximum(count_map, 1)
    binary   = (prob_map >= threshold).astype(np.uint8)

    geo = GEO_CONFIG
    prob_path   = os.path.join(PREDICTION_DIR, 'fire_prob_nextday.tif')
    binary_path = os.path.join(PREDICTION_DIR, 'fire_binary_nextday.tif')
    _save_tif(prob_path,   prob_map, 'float32', geo)
    _save_tif(binary_path, binary,   'uint8',   geo)

    fire_pct = 100 * binary.mean()
    print(f"[Predict] Fire probability map saved  -> {prob_path}")
    print(f"[Predict] Binary fire map saved       -> {binary_path}")
    print(f"[Predict] Predicted fire coverage: {fire_pct:.2f}%")

    return prob_map, binary


if __name__ == '__main__':
    predict_full_raster()
