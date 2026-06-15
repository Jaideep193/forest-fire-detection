"""
Visualization module -- colour scheme matches reference Figure 6.8.

Risk probability map  : 5-class discrete legend (Very Less -> Very High)
Fire spread maps      : risk-map background  +  burned/burning overlay
Animation             : same compositing, animated per hour
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.ndimage import binary_dilation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (PREDICTION_DIR, SPREAD_DIR, ANIMATION_DIR,
                    OUTPUT_DIR, MODEL_DIR, PROCESSED_DIR)


# ---------------------------------------------------------------------------
# 5-class risk colour scheme  (matches reference legend exactly)
# ---------------------------------------------------------------------------

_RISK_BOUNDS  = [0.00, 0.20, 0.35, 0.45, 0.55, 1.01]
_RISK_COLORS  = ['#1a9850',   # Very Less  -- dark green
                 '#74c476',   # Less       -- medium green
                 '#fed976',   # Moderate   -- yellow
                 '#fd8d3c',   # High       -- orange
                 '#e31a1c']   # Very High  -- red
_RISK_LABELS  = ['Very Less', 'Less', 'Moderate', 'High', 'Very High']

_RISK_CMAP = mcolors.ListedColormap(_RISK_COLORS)
_RISK_NORM = mcolors.BoundaryNorm(_RISK_BOUNDS, len(_RISK_COLORS))

# Fire-state overlay colours (applied on top of the risk background)
_BURNED_RGBA  = np.array([0.20, 0.10, 0.05, 0.70], dtype=np.float32)   # dark brown
_BURNING_RGBA = np.array([1.00, 0.30, 0.00, 0.85], dtype=np.float32)   # bright orange-red


def _risk_legend_patches():
    return [mpatches.Patch(facecolor=c, edgecolor='grey', linewidth=0.5, label=l)
            for c, l in zip(_RISK_COLORS, _RISK_LABELS)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hillshade(dem):
    dy, dx = np.gradient(dem.astype(np.float64), 30)
    az_r, al_r = np.radians(315), np.radians(45)
    slope  = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(dx, -dy)
    hs = np.cos(al_r)*np.cos(slope) + np.sin(al_r)*np.sin(slope)*np.cos(az_r - aspect)
    hs = np.clip(hs, 0, 1)
    return ((hs - hs.min()) / (hs.max() - hs.min() + 1e-8)).astype(np.float32)


def _load_prob_map():
    """Load saved probability map (used as background in spread figures)."""
    import rasterio
    p = os.path.join(PREDICTION_DIR, 'fire_prob_nextday.tif')
    if not os.path.exists(p):
        return None
    with rasterio.open(p) as src:
        return src.read(1).astype(np.float32)


def _draw_risk_background(ax, prob_map, dem=None, alpha=0.90):
    """Draw hillshade + 5-class risk map."""
    if dem is not None:
        hs = _hillshade(dem)
        ax.imshow(hs, cmap='gray', vmin=0, vmax=1, alpha=0.30, interpolation='nearest')
    ax.imshow(prob_map, cmap=_RISK_CMAP, norm=_RISK_NORM,
              alpha=alpha, interpolation='nearest')


def _fire_overlay(ax, state):
    """Overlay burned (dark brown) and burning (orange-red) cells on current axis."""
    H, W = state.shape
    overlay = np.zeros((H, W, 4), dtype=np.float32)

    burned  = state == 2
    burning = state == 1

    if burned.any():
        overlay[burned] = _BURNED_RGBA

    if burning.any():
        # Draw a 1-pixel glow halo around burning perimeter
        halo = binary_dilation(burning, iterations=2) & ~burning
        overlay[halo, 0] = 1.0
        overlay[halo, 1] = 0.55
        overlay[halo, 2] = 0.0
        overlay[halo, 3] = 0.45
        overlay[burning] = _BURNING_RGBA

    ax.imshow(overlay, interpolation='nearest')


def _fire_legend():
    return [
        mpatches.Patch(facecolor='#320D02', edgecolor='grey', lw=0.5, label='Burned'),
        mpatches.Patch(facecolor='#FF4D00', edgecolor='grey', lw=0.5, label='Burning'),
    ]


def _style_ax(ax, title):
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)


# ---------------------------------------------------------------------------
# 1. Fire probability map  (5-class)
# ---------------------------------------------------------------------------

def plot_fire_probability(prob_map, dem=None, save_dir=PREDICTION_DIR):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))

    _draw_risk_background(ax, prob_map, dem, alpha=0.92)

    ax.legend(handles=_risk_legend_patches(), loc='lower right',
              fontsize=9, framealpha=0.85, title='Fire Risk', title_fontsize=9)
    _style_ax(ax, 'Next-Day Forest Fire Probability Map\n'
                  'Uttarakhand  |  30 m resolution  |  EPSG:32644')

    plt.tight_layout()
    out = os.path.join(save_dir, 'fire_prob_nextday.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Viz] Probability map  -> {out}")


# ---------------------------------------------------------------------------
# 2. Binary fire map
# ---------------------------------------------------------------------------

def plot_binary_map(binary, dem=None, save_dir=PREDICTION_DIR):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    if dem is not None:
        ax.imshow(_hillshade(dem), cmap='gray', vmin=0, vmax=1,
                  alpha=0.35, interpolation='nearest')

    cmap2 = mcolors.ListedColormap(['#1a9850', '#e31a1c'])
    ax.imshow(binary, cmap=cmap2, vmin=0, vmax=1,
              alpha=0.82, interpolation='nearest')

    patches = [mpatches.Patch(facecolor='#1a9850', label='No Fire'),
               mpatches.Patch(facecolor='#e31a1c', label='Fire')]
    ax.legend(handles=patches, loc='lower right', fontsize=10,
              framealpha=0.85)
    _style_ax(ax, 'Binary Fire/No-Fire Prediction  (Next Day)\n'
                  'Uttarakhand  |  30 m  |  EPSG:32644')

    plt.tight_layout()
    out = os.path.join(save_dir, 'fire_binary_nextday.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Viz] Binary map       -> {out}")


# ---------------------------------------------------------------------------
# 3. Spread snapshot maps  (risk background + fire overlay)
# ---------------------------------------------------------------------------

def plot_spread_snapshots(results: dict, dem=None, save_dir=SPREAD_DIR):
    os.makedirs(save_dir, exist_ok=True)
    prob_map = _load_prob_map()
    hours = sorted(results.keys())

    for h in hours:
        state = results[h]
        H, W  = state.shape

        fig, ax = plt.subplots(figsize=(8, 7))

        # Background: 5-class risk map (or hillshade if prob_map unavailable)
        if prob_map is not None:
            _draw_risk_background(ax, prob_map, dem, alpha=0.88)
        elif dem is not None:
            ax.imshow(_hillshade(dem), cmap='gray', alpha=0.60,
                      interpolation='nearest')

        # Fire state overlay
        _fire_overlay(ax, state)

        # Legend
        combined = _risk_legend_patches() + _fire_legend()
        ax.legend(handles=combined, loc='lower right', fontsize=8,
                  framealpha=0.88, ncol=2,
                  title='Risk / Fire State', title_fontsize=8)

        label    = f'Hour-{h:02d}' if h > 0 else 'Initial Ignition'
        affected = (state > 0).sum() * 900 / 1e4   # ha
        _style_ax(ax, f'Fire Spread Simulation -- {label}\n'
                      f'Affected area: {affected:.1f} ha  |  30 m grid  |  Uttarakhand')

        plt.tight_layout()
        fname = f'spread_{h:02d}h.png'
        plt.savefig(os.path.join(save_dir, fname), dpi=130, bbox_inches='tight')
        plt.close()

    print(f"[Viz] Spread snapshots -> {save_dir}")


# ---------------------------------------------------------------------------
# 4. Animated GIF  (risk background + animated fire overlay)
# ---------------------------------------------------------------------------

def create_animation(results: dict, dem=None, save_dir=ANIMATION_DIR, fps=1):
    os.makedirs(save_dir, exist_ok=True)
    prob_map = _load_prob_map()
    hours    = sorted(results.keys())
    H, W     = results[hours[0]].shape

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_xticks([]); ax.set_yticks([])

    # Static background layers
    if dem is not None:
        ax.imshow(_hillshade(dem), cmap='gray', vmin=0, vmax=1,
                  alpha=0.28, interpolation='nearest')
    if prob_map is not None:
        ax.imshow(prob_map, cmap=_RISK_CMAP, norm=_RISK_NORM,
                  alpha=0.88, interpolation='nearest')

    # Dynamic overlay (starts transparent)
    overlay_data = np.zeros((H, W, 4), dtype=np.float32)
    overlay_im   = ax.imshow(overlay_data, interpolation='nearest')
    title_obj    = ax.set_title('', fontsize=12, fontweight='bold', pad=8)

    # Legend (static)
    combined = _risk_legend_patches() + _fire_legend()
    ax.legend(handles=combined, loc='lower right', fontsize=8,
              framealpha=0.88, ncol=2,
              title='Risk / Fire State', title_fontsize=8)

    def _build_overlay(state):
        ov = np.zeros((H, W, 4), dtype=np.float32)
        burned  = state == 2
        burning = state == 1
        if burned.any():
            ov[burned] = _BURNED_RGBA
        if burning.any():
            halo = binary_dilation(burning, iterations=2) & ~burning
            ov[halo, 0]=1.0; ov[halo, 1]=0.55; ov[halo, 2]=0.0; ov[halo, 3]=0.40
            ov[burning] = _BURNING_RGBA
        return ov

    def update(frame):
        h  = hours[frame]
        st = results[h]
        overlay_im.set_data(_build_overlay(st))
        ha  = (st > 0).sum() * 900 / 1e4
        lbl = f'Hour-{h:02d}' if h > 0 else 'Initial Ignition (t=0)'
        title_obj.set_text(
            f'Forest Fire Spread  --  {lbl}\n'
            f'Affected: {ha:.1f} ha  |  Uttarakhand  |  30 m grid')
        return overlay_im, title_obj

    ani = FuncAnimation(fig, update, frames=len(hours),
                        interval=int(1000 / fps), blit=False)
    out = os.path.join(save_dir, 'fire_spread_animation.gif')
    ani.save(out, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"[Viz] Animation        -> {out}")


# ---------------------------------------------------------------------------
# 5. Feature importance bar chart
# ---------------------------------------------------------------------------

def plot_feature_importance(save_dir=OUTPUT_DIR):
    pkl_path = os.path.join(MODEL_DIR, 'rf_model.pkl')
    if not os.path.exists(pkl_path):
        return
    try:
        import pickle
        with open(pkl_path, 'rb') as f:
            model = pickle.load(f)
        from config import FEATURE_NAMES
        fi  = model.feature_importances_
        idx = np.argsort(fi)[::-1]
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = [_RISK_COLORS[-1] if fi[i] > np.median(fi) else _RISK_COLORS[0]
                  for i in idx]
        ax.bar(range(len(fi)), fi[idx], color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(fi)))
        ax.set_xticklabels([FEATURE_NAMES[i] for i in idx],
                           rotation=40, ha='right', fontsize=9)
        ax.set_ylabel('Importance')
        ax.set_title('Random Forest Feature Importances  --  Fire Risk Prediction',
                     fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, 'feature_importance.png')
        plt.savefig(out, dpi=130)
        plt.close()
        print(f"[Viz] Feature importance -> {out}")
    except Exception as e:
        print(f"[Viz] Feature importance skipped: {e}")


def plot_training_curve(save_dir=OUTPUT_DIR):
    plot_feature_importance(save_dir)


# ---------------------------------------------------------------------------
# 6. Dashboard  (2x3 grid summary)
# ---------------------------------------------------------------------------

def create_dashboard(prob_map, binary, results, dem=None, save_dir=OUTPUT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    hours = sorted(h for h in results if h > 0)[:4]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()

    # Panel 0 -- Probability map
    ax = axes[0]
    _draw_risk_background(ax, prob_map, dem, alpha=0.92)
    ax.legend(handles=_risk_legend_patches(), loc='lower right',
              fontsize=7, framealpha=0.85, title='Risk', title_fontsize=7)
    _style_ax(ax, 'Fire Probability (Next Day)')

    # Panel 1 -- Binary map
    ax = axes[1]
    if dem is not None:
        ax.imshow(_hillshade(dem), cmap='gray', alpha=0.30, interpolation='nearest')
    cmap2 = mcolors.ListedColormap(['#1a9850', '#e31a1c'])
    ax.imshow(binary, cmap=cmap2, vmin=0, vmax=1, alpha=0.82, interpolation='nearest')
    ax.legend(handles=[mpatches.Patch(facecolor='#1a9850', label='No Fire'),
                        mpatches.Patch(facecolor='#e31a1c', label='Fire')],
              loc='lower right', fontsize=7, framealpha=0.85)
    _style_ax(ax, 'Binary Fire Map (>= threshold)')

    # Panels 2-5 -- Spread at each hour
    for pi, h in enumerate(hours):
        ax = axes[2 + pi]
        _draw_risk_background(ax, prob_map, dem, alpha=0.86)
        _fire_overlay(ax, results[h])
        ha = (results[h] > 0).sum() * 900 / 1e4
        ax.legend(handles=_fire_legend(), loc='lower right',
                  fontsize=7, framealpha=0.85)
        _style_ax(ax, f'Spread  Hour-{h:02d}  ({ha:.0f} ha)')

    for ax in axes[2 + len(hours):]:
        ax.axis('off')

    fig.suptitle(
        'Forest Fire Detection & Spread Simulation  --  Uttarakhand, India  |  30 m GeoTIFF',
        fontsize=14, fontweight='bold', y=1.005)
    plt.tight_layout()
    out = os.path.join(save_dir, 'dashboard.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Viz] Dashboard        -> {out}")
