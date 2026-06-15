"""
Training pipeline for the U-Net fire prediction model.

Loss : BCE (pos_weight) + Dice  -- handles severe class imbalance
Saves best model checkpoint to models/unet_best.pth
"""

import os, sys, time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_DIR, MODEL_CONFIG
from src.unet import UNet, combined_loss
from src.dataset import make_loaders
from src.preprocessing import train_val_split


def train(cfg=MODEL_CONFIG):
    os.makedirs(MODEL_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Device: {device}")

    train_idx, val_idx = train_val_split()
    train_dl, val_dl  = make_loaders(train_idx, val_idx, cfg)

    model = UNet(cfg['in_channels'], cfg['features']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'],
                           weight_decay=cfg['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val = float('inf')
    history  = {'train': [], 'val': []}
    ckpt_path = os.path.join(MODEL_DIR, 'unet_best.pth')

    print(f"[Train] Starting -- {cfg['num_epochs']} epochs")
    for epoch in range(1, cfg['num_epochs'] + 1):
        t0 = time.time()

        # -- Training ------------------------------------------------------
        model.train()
        train_loss = 0.0
        for feat, lbl in train_dl:
            feat, lbl = feat.to(device), lbl.to(device)
            optimizer.zero_grad()
            pred = model(feat)
            loss = combined_loss(pred, lbl, cfg['pos_weight'])
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        # -- Validation ----------------------------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feat, lbl in val_dl:
                feat, lbl = feat.to(device), lbl.to(device)
                pred = model(feat)
                val_loss += combined_loss(pred, lbl, cfg['pos_weight']).item()
        val_loss /= max(len(val_dl), 1)

        scheduler.step(val_loss)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{cfg['num_epochs']} | "
              f"train={train_loss:.4f}  val={val_loss:.4f}  [{elapsed:.1f}s]")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_loss': best_val, 'cfg': cfg}, ckpt_path)
            print(f"    OK Saved best model (val={best_val:.4f})")

    # Save loss history
    np.save(os.path.join(MODEL_DIR, 'history.npy'), history)
    print(f"[Train] Done. Best val loss: {best_val:.4f}")
    return model, history


if __name__ == '__main__':
    train()
