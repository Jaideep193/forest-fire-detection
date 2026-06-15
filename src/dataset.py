"""
PyTorch Dataset for tile-based training on the pre-processed feature stacks.

Each sample is a (tile_size x tile_size) spatial patch with matching label.
Fire-containing patches are oversampled to combat class imbalance.
"""

import os, sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DIR, MODEL_CONFIG


class FirePatchDataset(Dataset):
    """
    Parameters
    ----------
    day_indices : list[int]   which day indices to include
    tile_size   : int         spatial patch size (default 128)
    stride      : int         sliding-window stride for patch extraction
    augment     : bool        random horizontal/vertical flips
    oversample  : float       replicate fire-containing patches this many times
    """
    def __init__(self, day_indices, tile_size=128, stride=64,
                 augment=False, oversample=3.0, processed_dir=PROCESSED_DIR):
        self.tile  = tile_size
        self.aug   = augment
        self.patches = []   # list of (features_path, label_path, row, col)

        for idx in day_indices:
            feat_path  = os.path.join(processed_dir, f'features_{idx:03d}.npy')
            label_path = os.path.join(processed_dir, f'labels_{idx:03d}.npy')
            if not (os.path.exists(feat_path) and os.path.exists(label_path)):
                continue
            label = np.load(label_path, mmap_mode='r')
            H, W  = label.shape
            for r in range(0, H - tile_size + 1, stride):
                for c in range(0, W - tile_size + 1, stride):
                    patch_lbl = label[r:r+tile_size, c:c+tile_size]
                    has_fire  = patch_lbl.any()
                    entry     = (feat_path, label_path, r, c)
                    self.patches.append(entry)
                    # Oversample fire patches
                    if has_fire:
                        for _ in range(int(oversample) - 1):
                            self.patches.append(entry)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, i):
        feat_path, label_path, r, c = self.patches[i]
        T = self.tile

        feat  = np.load(feat_path,  mmap_mode='r')[:, r:r+T, c:c+T].copy()
        label = np.load(label_path, mmap_mode='r')[r:r+T, c:c+T].copy()

        # Augmentation: random flips
        if self.aug:
            if np.random.rand() > 0.5:
                feat  = feat[:, :, ::-1].copy()
                label = label[:, ::-1].copy()
            if np.random.rand() > 0.5:
                feat  = feat[:, ::-1, :].copy()
                label = label[::-1, :].copy()

        return (torch.from_numpy(feat).float(),
                torch.from_numpy(label).float().unsqueeze(0))


def make_loaders(train_idx, val_idx, cfg=MODEL_CONFIG):
    train_ds = FirePatchDataset(train_idx, cfg['tile_size'], cfg['stride'], augment=True)
    val_ds   = FirePatchDataset(val_idx,   cfg['tile_size'], cfg['stride'], augment=False)
    train_dl = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=cfg['batch_size'], shuffle=False, num_workers=0, pin_memory=False)
    print(f"[Dataset] train patches={len(train_ds)}  val patches={len(val_ds)}")
    return train_dl, val_dl
