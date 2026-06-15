"""
U-Net architecture for spatial fire probability prediction.

Input  : (B, C, H, W)  -- multi-channel feature raster patches
Output : (B, 1, H, W)  -- fire probability in [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    """Two x (Conv 3x3 -> BN -> ReLU)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    4-level U-Net.

    Parameters
    ----------
    in_channels : int   number of input feature channels
    features    : list  channel counts at each encoder level, e.g. [32, 64, 128, 256]
    """
    def __init__(self, in_channels=11, features=(32, 64, 128, 256)):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder
        self.enc = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.enc.append(_ConvBlock(ch, f))
            ch = f

        # Bottleneck
        self.bottleneck = _ConvBlock(ch, ch * 2)
        ch = ch * 2

        # Decoder
        self.up_convs = nn.ModuleList()
        self.dec      = nn.ModuleList()
        for f in reversed(features):
            self.up_convs.append(nn.ConvTranspose2d(ch, f, 2, stride=2))
            self.dec.append(_ConvBlock(f * 2, f))
            ch = f

        self.final = nn.Conv2d(ch, 1, 1)

    def forward(self, x):
        skips = []
        for enc in self.enc:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.up_convs, self.dec, reversed(skips)):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return torch.sigmoid(self.final(x))


# ----------------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------------

def dice_loss(pred, target, smooth=1.0):
    pred_f   = pred.view(-1)
    target_f = target.view(-1).float()
    inter    = (pred_f * target_f).sum()
    return 1 - (2 * inter + smooth) / (pred_f.sum() + target_f.sum() + smooth)


def combined_loss(pred, target, pos_weight=6.0):
    """BCE (with pos_weight) + Dice -- handles severe class imbalance."""
    bce = F.binary_cross_entropy(
        pred, target.float(),
        weight=(target.float() * (pos_weight - 1) + 1)
    )
    return bce + dice_loss(pred, target)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, '..')
    from config import MODEL_CONFIG
    m = UNet(MODEL_CONFIG['in_channels'], MODEL_CONFIG['features'])
    x = torch.randn(2, MODEL_CONFIG['in_channels'], 128, 128)
    out = m(x)
    print('Output shape:', out.shape)   # expected (2, 1, 128, 128)
