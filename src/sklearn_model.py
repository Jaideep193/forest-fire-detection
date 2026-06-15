"""
Scikit-learn Random Forest model for fire probability prediction.

Replaces the PyTorch U-Net when system memory/paging-file limits prevent
loading torch.  Produces identical GeoTIFF outputs.

Training approach
-----------------
* Flatten all daily feature stacks -> pixel-level feature matrix  (N, 11)
* Balance classes by undersampling the majority (no-fire) class
* Train sklearn RandomForestClassifier with calibrated probabilities
* Save model to models/rf_model.pkl

Prediction
----------
* Load feature stack for the target day
* Predict fire probability per pixel -> (H, W) float32 map
* Save fire_prob_nextday.tif + fire_binary_nextday.tif
"""

import os, sys, csv, pickle, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (MODEL_DIR, PROCESSED_DIR, PREDICTION_DIR,
                    SYNTHETIC_DIR, GEO_CONFIG, N_FEATURES)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_day(idx, processed_dir=PROCESSED_DIR):
    feat  = np.load(os.path.join(processed_dir, f'features_{idx:03d}.npy'))  # (C,H,W)
    label = np.load(os.path.join(processed_dir, f'labels_{idx:03d}.npy'))    # (H,W)
    C, H, W = feat.shape
    X = feat.reshape(C, -1).T          # (H*W, C)
    y = label.reshape(-1).astype(int)  # (H*W,)
    return X, y


def build_dataset(day_indices, undersample_ratio=10, seed=0):
    """
    Sample per-day to keep memory bounded, then concatenate the small samples.

    For each day: keep ALL fire pixels + up to (n_fire * undersample_ratio) no-fire pixels.
    This avoids loading all days into memory at once.
    """
    rng = np.random.default_rng(seed)
    Xs_fire, Xs_nofire = [], []
    n_days_used = 0

    for idx in day_indices:
        fp = os.path.join(PROCESSED_DIR, f'features_{idx:03d}.npy')
        if not os.path.exists(fp):
            continue
        X, y = _load_day(idx)

        fire_mask   = y == 1
        nofire_mask = y == 0
        fire_px     = X[fire_mask]
        nofire_px   = X[nofire_mask]

        if len(fire_px) > 0:
            Xs_fire.append(fire_px)

        n_keep = min(len(nofire_px), max(len(fire_px) * undersample_ratio, 500))
        if n_keep > 0:
            sel = rng.choice(len(nofire_px), size=n_keep, replace=False)
            Xs_nofire.append(nofire_px[sel])

        del X, y, fire_px, nofire_px
        n_days_used += 1

    X_fire   = np.vstack(Xs_fire).astype(np.float32)   if Xs_fire   else np.zeros((0, N_FEATURES), np.float32)
    X_nofire = np.vstack(Xs_nofire).astype(np.float32) if Xs_nofire else np.zeros((0, N_FEATURES), np.float32)
    y_fire   = np.ones(len(X_fire),   dtype=np.int32)
    y_nofire = np.zeros(len(X_nofire), dtype=np.int32)

    X_all = np.vstack([X_fire, X_nofire])
    y_all = np.concatenate([y_fire, y_nofire])
    # shuffle
    perm = rng.permutation(len(y_all))
    X_all, y_all = X_all[perm], y_all[perm]

    print(f"  Dataset: {len(X_fire)} fire + {len(X_nofire)} no-fire = {len(y_all)} pixels "
          f"(from {n_days_used} days)")
    return X_all, y_all


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(seed=42):
    os.makedirs(MODEL_DIR, exist_ok=True)

    meta_path = os.path.join(SYNTHETIC_DIR, 'days_meta.csv')
    with open(meta_path) as f:
        days = list(csv.DictReader(f))
    n = len(days)
    n_val  = max(1, int(n * 0.2))
    tr_idx = list(range(n - n_val))
    va_idx = list(range(n - n_val, n))

    print("[Train-RF] Building training set ...")
    X_tr, y_tr = build_dataset(tr_idx, undersample_ratio=10, seed=seed)
    print("[Train-RF] Building validation set ...")
    X_va, y_va = build_dataset(va_idx, undersample_ratio=10, seed=seed + 1)

    print("[Train-RF] Fitting Random Forest ...")
    t0 = time.time()
    base = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=20,
        max_features='sqrt',
        class_weight='balanced',
        oob_score=True,
        n_jobs=1,          # single job avoids shared-memory issues on Windows
        random_state=seed,
    )
    base.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s")

    # Validation
    y_pred = base.predict(X_va)
    y_prob = base.predict_proba(X_va)[:, 1]
    print("[Train-RF] Validation results:")
    print(classification_report(y_va, y_pred, target_names=['no-fire', 'fire']))
    try:
        auc = roc_auc_score(y_va, y_prob)
        print(f"  ROC-AUC: {auc:.4f}")
    except Exception:
        pass

    # Feature importances
    fi = base.feature_importances_
    from config import FEATURE_NAMES
    print("[Train-RF] Feature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, fi), key=lambda x: -x[1]):
        print(f"  {name:20s}: {imp:.4f}")

    # Save
    model_path = os.path.join(MODEL_DIR, 'rf_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(base, f)
    print(f"[Train-RF] Model saved -> {model_path}")

    # Save a simple history dict compatible with visualisation
    history = {'train': [float(getattr(base, 'oob_score_', 0.0))],
               'val':   [float((y_pred == y_va).mean())]}
    np.save(os.path.join(MODEL_DIR, 'history.npy'), history)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

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


def predict(day_idx='last', threshold=0.55):
    os.makedirs(PREDICTION_DIR, exist_ok=True)

    # Load model
    model_path = os.path.join(MODEL_DIR, 'rf_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Load features
    if day_idx == 'last':
        import glob
        paths = sorted(glob.glob(os.path.join(PROCESSED_DIR, 'features_*.npy')))
        feat_path = paths[-1]
    else:
        feat_path = os.path.join(PROCESSED_DIR, f'features_{day_idx:03d}.npy')

    feat = np.load(feat_path)            # (C, H, W)
    C, H, W = feat.shape
    X = feat.reshape(C, -1).T           # (H*W, C)

    print(f"[Predict-RF] Running inference on {H}x{W} grid ...")
    prob_flat = model.predict_proba(X)[:, 1].astype(np.float32)
    prob_map  = prob_flat.reshape(H, W)
    binary    = (prob_map >= threshold).astype(np.uint8)

    geo = GEO_CONFIG
    prob_path   = os.path.join(PREDICTION_DIR, 'fire_prob_nextday.tif')
    binary_path = os.path.join(PREDICTION_DIR, 'fire_binary_nextday.tif')
    _save_tif(prob_path,   prob_map, 'float32', geo)
    _save_tif(binary_path, binary,   'uint8',   geo)

    fire_pct = 100 * binary.mean()
    print(f"[Predict-RF] Probability map -> {prob_path}")
    print(f"[Predict-RF] Binary map      -> {binary_path}")
    print(f"[Predict-RF] Predicted fire coverage: {fire_pct:.2f}%")
    return prob_map, binary


if __name__ == '__main__':
    train()
    predict()
