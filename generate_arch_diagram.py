"""
Generate a clean vertical flowchart for the README.
Outputs: outputs/architecture.png
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

os.makedirs('outputs', exist_ok=True)

# ── Canvas ────────────────────────────────────────────────────────────────────
FW, FH = 14, 20
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.axis('off')
fig.patch.set_facecolor('#FAFAFA')
ax.set_facecolor('#FAFAFA')

# ── Colour palette (stage → fill, border) ────────────────────────────────────
CLR = {
    'title':   ('#1A237E', '#283593'),   # navy
    'input':   ('#E3F2FD', '#1565C0'),   # blue
    'prep':    ('#E8F5E9', '#2E7D32'),   # green
    'model':   ('#FFF3E0', '#E65100'),   # orange
    'pred':    ('#FCE4EC', '#880E4F'),   # pink
    'ca':      ('#EDE7F6', '#4527A0'),   # purple
    'out':     ('#E0F2F1', '#00695C'),   # teal
    'arrow':   '#546E7A',
    'label_bg':'#ECEFF1',
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, style,
        title='', lines=(), title_size=10, body_size=8.5, radius=0.4):
    fill, border = CLR[style]
    # shadow
    sh = FancyBboxPatch((x+0.07, y-0.07), w, h,
                        boxstyle=f'round,pad=0,rounding_size={radius}',
                        facecolor='#BDBDBD', edgecolor='none', zorder=2, alpha=0.45)
    ax.add_patch(sh)
    # body
    bd = FancyBboxPatch((x, y), w, h,
                        boxstyle=f'round,pad=0,rounding_size={radius}',
                        linewidth=2, edgecolor=border, facecolor=fill, zorder=3)
    ax.add_patch(bd)
    # left accent bar
    bar = FancyBboxPatch((x, y), 0.18, h,
                         boxstyle=f'round,pad=0,rounding_size=0.15',
                         linewidth=0, facecolor=border, zorder=4, alpha=0.85)
    ax.add_patch(bar)
    # title
    ax.text(x + 0.38, y + h - 0.30, title,
            fontsize=title_size, fontweight='bold', color=border,
            va='center', zorder=5)
    # body lines
    for i, ln in enumerate(lines):
        ax.text(x + 0.38, y + h - 0.62 - i * 0.30, ln,
                fontsize=body_size, color='#37474F', va='center', zorder=5)
    return (x + w/2, y + h), (x + w/2, y)   # top-center, bottom-center

def small_box(ax, x, y, w, h, style, title, subtitle='', radius=0.3):
    fill, border = CLR[style]
    sh = FancyBboxPatch((x+0.05, y-0.05), w, h,
                        boxstyle=f'round,pad=0,rounding_size={radius}',
                        facecolor='#BDBDBD', edgecolor='none', zorder=2, alpha=0.40)
    ax.add_patch(sh)
    bd = FancyBboxPatch((x, y), w, h,
                        boxstyle=f'round,pad=0,rounding_size={radius}',
                        linewidth=1.8, edgecolor=border, facecolor=fill, zorder=3)
    ax.add_patch(bd)
    bar = FancyBboxPatch((x, y+h-0.28), w, 0.28,
                         boxstyle=f'round,pad=0,rounding_size={radius}',
                         linewidth=0, facecolor=border, zorder=4)
    ax.add_patch(bar)
    ax.text(x + w/2, y+h-0.14, title,
            fontsize=8, fontweight='bold', color='white', ha='center', va='center', zorder=5)
    if subtitle:
        ax.text(x + w/2, y+h*0.38, subtitle,
                fontsize=7, color='#37474F', ha='center', va='center', zorder=5)
    return (x + w/2, y + h), (x + w/2, y), (x, y + h/2), (x + w, y + h/2)

def arrow_v(ax, x, y1, y2, label=''):
    """Straight vertical arrow from (x,y1) down to (x,y2)."""
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=CLR['arrow'],
                                lw=2.0, connectionstyle='arc3,rad=0'),
                zorder=6)
    if label:
        ax.text(x + 0.18, (y1+y2)/2, label, fontsize=7.5,
                color=CLR['arrow'], va='center', zorder=7)

def arrow_diag(ax, x1, y1, x2, y2, rad=0.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=CLR['arrow'],
                                lw=1.8, connectionstyle=f'arc3,rad={rad}'),
                zorder=6)

def arrow_h(ax, x1, x2, y):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=CLR['arrow'],
                                lw=1.8, connectionstyle='arc3,rad=0'),
                zorder=6)

def stage_chip(ax, x, y, label, style):
    _, border = CLR[style]
    chip = FancyBboxPatch((x, y), 1.5, 0.38,
                           boxstyle='round,pad=0,rounding_size=0.12',
                           linewidth=1.5, edgecolor=border, facecolor=border,
                           zorder=7, alpha=0.90)
    ax.add_patch(chip)
    ax.text(x + 0.75, y + 0.19, label,
            fontsize=7.5, fontweight='bold', color='white',
            ha='center', va='center', zorder=8)

def connector_dot(ax, x, y):
    ax.plot(x, y, 'o', color=CLR['arrow'], ms=6, zorder=7)

# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT  (y increases upward in matplotlib)
# We place from TOP (y=19) downward.
# ─────────────────────────────────────────────────────────────────────────────

MX  = 0.55    # left margin
RMX = FW - 0.55  # right bound
CW  = RMX - MX   # full content width

# ── TITLE BANNER ──────────────────────────────────────────────────────────────
_, border = CLR['title']
title_bg = FancyBboxPatch((0, 19.3), FW, 0.65,
                           boxstyle='round,pad=0,rounding_size=0',
                           linewidth=0, facecolor=border, zorder=3)
ax.add_patch(title_bg)
ax.text(FW/2, 19.63,
        'Forest Fire Detection & Spread Simulation  —  Pipeline Flowchart',
        ha='center', va='center', fontsize=13, fontweight='bold',
        color='white', zorder=4)
ax.text(FW/2, 19.35,
        'Uttarakhand, India   ·   512 × 512 grid   ·   30 m GeoTIFF   ·   EPSG:32644',
        ha='center', va='center', fontsize=8, color='#B0BEC5', zorder=4)

# ── STAGE 1 — DATA INPUTS (two side-by-side boxes) ───────────────────────────
stage_chip(ax, MX, 18.70, 'STAGE 1', 'input')

LW = (CW - 0.35) / 2   # width of each side box

# Left: VIIRS
box(ax, MX, 17.25, LW, 1.35, 'input',
    'VIIRS Fire Archive  (NASA FIRMS)',
    ('NOAA-20 / Suomi-NPP  ·  Collection 2',
     'fire_archive_J1V-C2_762685.csv  (~47 MB)',
     'Fields: lat · lon · acq_date · FRP · confidence'))

# Right: Synthetic
box(ax, MX + LW + 0.35, 17.25, LW, 1.35, 'input',
    'Synthetic Data Generator',
    ('Multi-scale Gaussian noise DEM  (500–3500 m)',
     'LULC (7 classes, elevation-correlated)',
     'Weather: temp · humidity · wind · rainfall'))

# merge arrows → downward to Stage 2
cx = FW / 2   # centre x
arrow_diag(ax, MX + LW/2,            17.25, cx, 16.90)
arrow_diag(ax, MX + LW + 0.35 + LW/2, 17.25, cx, 16.90)
connector_dot(ax, cx, 16.92)

# ── STAGE 2 — PREPROCESSING ───────────────────────────────────────────────────
arrow_v(ax, cx, 16.90, 16.65)
stage_chip(ax, MX, 16.60, 'STAGE 2', 'prep')

box(ax, MX, 15.15, CW, 1.35, 'prep',
    'Preprocessing  —  Feature Stack Builder',
    ('VIIRS detections rasterised to 6-pixel disks (radius 180 m = VIIRS footprint)',
     'Static  (5 ch): elevation · slope · aspect_sin · aspect_cos · lulc_norm',
     'Dynamic (6 ch): temperature · humidity · wind_speed · wind_dir_sin/cos · rainfall',
     'All channels normalised [0,1] · Saved as .npy · Chronological 80/20 split'))

arrow_v(ax, cx, 15.15, 14.88)

# ── STAGE 3 — MODEL TRAINING ──────────────────────────────────────────────────
stage_chip(ax, MX, 14.83, 'STAGE 3', 'model')

box(ax, MX, 13.38, CW, 1.35, 'model',
    'Random Forest Classifier  (scikit-learn)',
    ('Per-day balanced sampling: ALL fire pixels  +  10× undersampled no-fire pixels',
     'n_estimators=100  ·  max_depth=10  ·  max_features=sqrt  ·  class_weight=balanced',
     'OOB validation: ROC-AUC ≈ 0.91  ·  OOB score ≈ 0.96',
     'Saved to  models/rf_model.pkl'))

arrow_v(ax, cx, 13.38, 13.10)

# ── STAGE 4 — PREDICTION (two output boxes) ───────────────────────────────────
stage_chip(ax, MX, 13.05, 'STAGE 4', 'pred')

PW = (CW - 0.35) / 2
# left: prob map
box(ax, MX, 11.60, PW, 1.35, 'pred',
    'Fire Probability Map',
    ('Per-pixel P(fire) in [0.0, 1.0]',
     'fire_prob_nextday.tif  (float32)',
     '512 × 512  ·  30 m  ·  EPSG:32644'))

# right: binary map
box(ax, MX + PW + 0.35, 11.60, PW, 1.35, 'pred',
    'Binary Fire Map',
    ('Threshold: P(fire) >= 0.55',
     'fire_binary_nextday.tif  (uint8)',
     'Fire coverage: 3.09 % of tile'))

# branch arrows from centre
arrow_diag(ax, cx, 13.10, MX + PW/2,              11.95, rad=0.15)
arrow_diag(ax, cx, 13.10, MX + PW + 0.35 + PW/2, 11.95, rad=-0.15)

# left box feeds into CA
arrow_diag(ax, MX + PW/2, 11.60, cx, 11.27, rad=-0.10)
connector_dot(ax, cx, 11.28)

# ── STAGE 5 — CA SIMULATION ───────────────────────────────────────────────────
arrow_v(ax, cx, 11.27, 11.02)
stage_chip(ax, MX, 10.97, 'STAGE 5', 'ca')

box(ax, MX, 9.52, CW, 1.35, 'ca',
    'Cellular Automata Fire Spread Engine',
    ('Seed: P >= 0.55 high-risk cells, 2% fraction  ->  ~161 ignition points',
     'P_spread = base_prob x fuel(LULC) x wind_align x slope_factor x moisture_factor',
     '8-neighbourhood via np.roll  (vectorised, no Python pixel loops)',
     '10 CA steps / simulated hour  ->  snapshots at 1h, 2h, 3h, 6h, 12h'))

arrow_v(ax, cx, 9.52, 9.25)

# ── STAGE 5 — SPREAD OUTPUTS (five small boxes) ───────────────────────────────
SW = (CW - 4*0.18) / 5
SY = 8.05
sy_top = SY + 1.05

hours = ['01 h', '02 h', '03 h', '06 h', '12 h']
areas = ['662 ha', '1 464 ha', '2 310 ha', '4 380 ha', '6 845 ha']

for i, (h, a) in enumerate(zip(hours, areas)):
    sx = MX + i * (SW + 0.18)
    # fan arrow from centre
    bx = sx + SW/2
    arrow_diag(ax, cx, 9.25, bx, sy_top, rad=0.0)
    small_box(ax, sx, SY, SW, 1.05, 'ca',
              f'Hour {h}', f'{a}')

# ── STAGE 6 — VISUALISATION ────────────────────────────────────────────────────
arrow_v(ax, cx, SY, 7.72)
stage_chip(ax, MX, 7.68, 'STAGE 6', 'out')

box(ax, MX, 6.95, CW, 0.65, 'out',
    'Visualisation Engine  (5-class discrete risk colormap)',
    ('Risk background: Very Less(green) / Less / Moderate / High / Very High(red)',))

# ── OUTPUT ARTEFACTS (6 boxes) ────────────────────────────────────────────────
OW = (CW - 5*0.14) / 6
outputs = [
    ('Probability\nMap PNG', 'fire_prob_\nnextday.png'),
    ('Binary\nMap PNG', 'fire_binary_\nnextday.png'),
    ('Spread\nSnapshots', '5x tif+png\n01-12 h'),
    ('Animated\nGIF', 'fire_spread_\nanimation.gif'),
    ('Dashboard\nPNG', '6-panel\nsummary'),
    ('Feature\nImportance', 'bar chart\nPNG'),
]
for i, (title, sub) in enumerate(outputs):
    ox = MX + i * (OW + 0.14)
    arrow_diag(ax, cx, 6.95, ox + OW/2, 6.45)
    small_box(ax, ox, 5.10, OW, 1.30, 'out', title, sub)

# ── PHYSICS CALLOUT (right margin note) ───────────────────────────────────────
note_x, note_y = 0.18, 3.60
note_h = 2.30
note_w = 3.40
_, border = CLR['ca']
note_bg = FancyBboxPatch((note_x, note_y), note_w, note_h,
                          boxstyle='round,pad=0,rounding_size=0.3',
                          linewidth=1.5, edgecolor=border,
                          facecolor='#EDE7F6', zorder=3)
ax.add_patch(note_bg)
ax.text(note_x + note_w/2, note_y + note_h - 0.26,
        'CA Physics Detail',
        fontsize=8.5, fontweight='bold', color=border,
        ha='center', va='center', zorder=5)
physics = [
    'Wind factor:',
    '  align = (1 + cos(wind_dest - θ)) / 2',
    '  boost = 1 + w_wind x speed/10 x align',
    '',
    'Slope factor:',
    '  fac = 1 + w_slope x tanh(Δz / 100)',
    '',
    'Moisture factor:',
    '  fac = 1 - w_moist x (humidity / 100)',
    '',
    'Fuel weights (LULC):',
    '  Dense forest=0.95 · Grass=0.60',
    '  Agriculture=0.20 · Urban=0.05',
]
for i, line in enumerate(physics):
    ax.text(note_x + 0.18, note_y + note_h - 0.55 - i*0.155,
            line, fontsize=6.8, color='#1A237E',
            va='center', fontfamily='monospace', zorder=5)

# ── LEGEND ────────────────────────────────────────────────────────────────────
legend_items = [
    ('input', 'Input / External Data'),
    ('prep',  'Preprocessing'),
    ('model', 'ML Model'),
    ('pred',  'Prediction Outputs'),
    ('ca',    'CA Simulation'),
    ('out',   'Visualisation / Export'),
]
lx, ly = note_x, note_y - 1.25
ax.text(lx + note_w/2, ly + 1.15,
        'Legend', fontsize=8, fontweight='bold',
        color='#37474F', ha='center', va='center', zorder=7)
lg_bg = FancyBboxPatch((lx, ly), note_w, 1.10,
                         boxstyle='round,pad=0,rounding_size=0.2',
                         linewidth=1, edgecolor='#B0BEC5',
                         facecolor='white', zorder=3, alpha=0.9)
ax.add_patch(lg_bg)
for i, (key, label) in enumerate(legend_items):
    _, bdr = CLR[key]
    row = i % 3
    col = i // 3
    ix = lx + 0.20 + col * 1.65
    iy = ly + 0.80 - row * 0.30
    dot = FancyBboxPatch((ix, iy - 0.08), 0.22, 0.18,
                          boxstyle='round,pad=0,rounding_size=0.05',
                          facecolor=bdr, edgecolor='none', zorder=6)
    ax.add_patch(dot)
    ax.text(ix + 0.28, iy + 0.01, label,
            fontsize=6.8, color='#37474F', va='center', zorder=7)

# ── GEO CONFIG NOTE ───────────────────────────────────────────────────────────
geo_x = FW - 3.60
geo_y = note_y
geo_w = 3.08
geo_h = 2.30
_, gbdr = CLR['input']
geo_bg = FancyBboxPatch((geo_x, geo_y), geo_w, geo_h,
                          boxstyle='round,pad=0,rounding_size=0.3',
                          linewidth=1.5, edgecolor=gbdr,
                          facecolor='#E3F2FD', zorder=3)
ax.add_patch(geo_bg)
ax.text(geo_x + geo_w/2, geo_y + geo_h - 0.26,
        'Spatial Reference',
        fontsize=8.5, fontweight='bold', color=gbdr,
        ha='center', va='center', zorder=5)
geo_lines = [
    'CRS  :  EPSG:32644',
    '         WGS 84 / UTM Zone 44N',
    '',
    'Grid  :  512 x 512 pixels',
    'Pixel :  30 m x 30 m',
    'Area  :  15.36 x 15.36 km',
    '',
    'Origin:  338500 E',
    '         3 279 000 N',
    '',
    'Study :  29.48-29.62 N',
    '         79.47-79.63 E',
]
for i, line in enumerate(geo_lines):
    ax.text(geo_x + 0.18, geo_y + geo_h - 0.55 - i*0.155,
            line, fontsize=6.8, color='#0D47A1',
            va='center', fontfamily='monospace', zorder=5)

plt.tight_layout(pad=0)
out = 'outputs/architecture.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='#FAFAFA')
plt.close()
print(f'Saved: {out}')
