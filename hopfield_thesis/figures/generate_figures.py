"""
Generate all thesis figures for the Work Methodology and Results sections.

Run from hopfield_thesis/:
    .venv/Scripts/python figures/generate_figures.py
"""

from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse, Polygon
import numpy as np

ROOT    = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"
EXP     = ROOT / "experiments"
DATA    = ROOT / "data"
sys.path.insert(0, str(ROOT))

# ── palette ──────────────────────────────────────────────────────────────────
C_BOX   = "#F2F3F4"   # light gray — process boxes
C_DIAM  = "#FEF9E7"   # light yellow — decision diamonds
C_DATA  = "#D6EAF8"   # light blue — data / input
C_ATT   = "#FDEBD0"   # light orange — attack / adversarial
C_OK    = "#D5F5E3"   # light green — success
C_FAIL  = "#FADBD8"   # light pink — failure
C_PURP  = "#E8DAEF"   # light purple
C_EB    = "#1A5276"   # dark blue edge
C_EG    = "#1E6B3C"   # dark green edge
C_ER    = "#B03A2E"   # dark red edge
C_EO    = "#A04000"   # dark orange edge
C_EP    = "#6C3483"   # dark purple edge
C_GRAY  = "#2C3E50"   # dark gray text
C_MID   = "#5D6D7E"   # mid gray

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

CREATED: list[tuple[str, str]] = []   # (filename, description)


# ─────────────────────────────────────────────────────────────────────────────
# Flowchart drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rect(ax, cx, cy, bw, bh, text, fc=C_BOX, ec=C_GRAY, fs=9.5, lw=1.4,
          bold=False, color=C_GRAY):
    p = FancyBboxPatch((cx - bw/2, cy - bh/2), bw, bh,
                       boxstyle="round,pad=0.04", linewidth=lw,
                       edgecolor=ec, facecolor=fc, zorder=3, clip_on=False)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=color, weight="bold" if bold else "normal",
            zorder=4, multialignment="center", clip_on=False)


def _oval(ax, cx, cy, w, h, text, fc=C_DATA, ec=C_EB, fs=9.5, lw=1.4,
          bold=False, color=C_GRAY):
    e = Ellipse((cx, cy), w, h, facecolor=fc, edgecolor=ec, linewidth=lw,
                zorder=3, clip_on=False)
    ax.add_patch(e)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=color, weight="bold" if bold else "normal",
            zorder=4, multialignment="center", clip_on=False)


def _diamond(ax, cx, cy, dw, dh, text, fc=C_DIAM, ec=C_EO, fs=9.5, lw=1.4,
             color=C_GRAY):
    verts = [(cx+dw, cy), (cx, cy+dh), (cx-dw, cy), (cx, cy-dh)]
    p = Polygon(verts, closed=True, facecolor=fc, edgecolor=ec, linewidth=lw,
                zorder=3, clip_on=False)
    ax.add_patch(p)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=color,
            zorder=4, multialignment="center", clip_on=False)


def _route(ax, points, color=C_GRAY, lw=1.5):
    """Draw a multi-segment path with arrowhead at the final point."""
    for i in range(len(points) - 1):
        style = "-|>" if i == len(points) - 2 else "-"
        ax.annotate("", xy=points[i+1], xytext=points[i],
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                   mutation_scale=12,
                                   connectionstyle="arc3,rad=0"))


def _label(ax, x, y, text, fs=8.5, color=C_GRAY, bold=True, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color,
            weight="bold" if bold else "normal", clip_on=False)


# ─────────────────────────────────────────────────────────────────────────────
# CORRECTION 1 — Figure M2: White-Box One-Pixel Attack Protocol
# ─────────────────────────────────────────────────────────────────────────────

def fig_m2_whitebox_protocol():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Figure M2: White-Box One-Pixel Attack Protocol",
                 fontsize=12, fontweight="bold", color=C_GRAY, pad=8)

    # ── rail heights ──────────────────────────────────────────────────────────
    YM = 2.5    # main flow
    YU = 4.0    # upper branch (Yes / Stage 1)
    YD = 1.0    # lower branch (No  / Stage 2 fallback)

    # ── node x-centres ───────────────────────────────────────────────────────
    x1 = 0.65
    x2 = 2.35
    x3 = 4.10
    x4 = 5.85
    x5 = 7.55    # diamond 1
    x6a = 9.35   # argmin  (YES, upper)
    x6b = 9.35   # sensitivity (NO, lower)
    x7 = 10.90   # set pixel  (merge)
    x8 = 12.50   # diamond 2
    x9a = 12.50  # success badge (YES, upper)
    x9b = 12.50  # fail badge    (NO,  lower)

    # ── node dimensions ───────────────────────────────────────────────────────
    OW, OH    = 1.15, 0.70   # oval
    RBW, RBH1 = 1.55, 1.05  # tall rect (3-line)
    RBH2      = 0.80         # medium rect (2-line)
    RBH3      = 0.65         # short rect  (1-line)
    DW1, DH1  = 0.95, 0.65  # diamond 1
    DW2, DH2  = 0.80, 0.60  # diamond 2
    BW,  BH   = 1.35, 0.80  # badge oval

    # ── nodes ─────────────────────────────────────────────────────────────────
    _oval(ax, x1, YM, OW, OH,
          "Query  q\n(d-dim)",
          fc=C_DATA, ec=C_EB, bold=True)

    _rect(ax, x2, YM, RBW, RBH1,
          "Enumerate candidates\nd × 5 values\n3,920 / 5,120 total",
          fc=C_BOX, ec=C_GRAY)

    _rect(ax, x3, YM, RBW, RBH2,
          "Batch retrieve\nnet.retrieve(candidates)",
          fc=C_DATA, ec=C_EB)

    _rect(ax, x4, YM, RBW, RBH2,
          "Cosine to  xₜ\nfor each candidate",
          fc=C_PURP, ec=C_EP)

    _diamond(ax, x5, YM, DW1, DH1,
             "Δcos < −1e−4?",
             fc=C_DIAM, ec=C_EO)

    _rect(ax, x6a, YU, RBW, RBH2,
          "argmin  Δcos\n→  pixel  k*",
          fc=C_OK, ec=C_EG)

    _rect(ax, x6b, YD, RBW, RBH2,
          "1st-order sensitivity\n→  pixel  k*",
          fc=C_ATT, ec=C_EO)

    _rect(ax, x7, YM, RBW, RBH3,
          "Set pixel  k*  in  q  →  q*",
          fc=C_BOX, ec=C_GRAY)

    _diamond(ax, x8, YM, DW2, DH2,
             "retrieved(q*)\n≠  xₜ ?",
             fc=C_DIAM, ec=C_EO)

    _oval(ax, x9a, YU, BW, BH,
          "✓  ATTACK\nSUCCESS",
          fc=C_OK, ec=C_EG, bold=True, color=C_EG)

    _oval(ax, x9b, YD, BW, BH,
          "✗  ATTACK\nFAILS",
          fc=C_FAIL, ec=C_ER, bold=True, color=C_ER)

    # ── straight arrows on main rail ─────────────────────────────────────────
    gaps = [
        (x1 + OW/2,        x2 - RBW/2),   # 1 → 2
        (x2 + RBW/2,       x3 - RBW/2),   # 2 → 3
        (x3 + RBW/2,       x4 - RBW/2),   # 3 → 4
        (x4 + RBW/2,       x5 - DW1),     # 4 → 5 (diamond left)
        (x7 + RBW/2,       x8 - DW2),     # 7 → 8 (diamond left)
    ]
    for x_from, x_to in gaps:
        _route(ax, [(x_from, YM), (x_to, YM)])

    # ── D1 top → 6a (YES, upper) ──────────────────────────────────────────────
    _route(ax, [(x5, YM + DH1), (x5, YU), (x6a - RBW/2, YU)], color=C_EG)
    _label(ax, x5 - 0.35, YM + DH1 + 0.28, "Yes", color=C_EG, fs=8.5)
    ax.text(x6a, YU + BH/2 + 0.22, "Stage 1",
            ha="center", va="bottom", fontsize=8, color=C_EG, style="italic")

    # ── D1 bottom → 6b (NO, lower) ───────────────────────────────────────────
    _route(ax, [(x5, YM - DH1), (x5, YD), (x6b - RBW/2, YD)], color=C_EO)
    _label(ax, x5 - 0.35, YM - DH1 - 0.28, "No", color=C_EO, fs=8.5)
    ax.text(x6b, YD - BH/2 - 0.22, "Stage 2 fallback",
            ha="center", va="top", fontsize=8, color=C_EO, style="italic")

    # ── 6a right → x7 top (merge from upper) ─────────────────────────────────
    _route(ax, [(x6a + RBW/2, YU), (x7, YU), (x7, YM + RBH3/2)], color=C_EG)

    # ── 6b right → x7 bottom (merge from lower) ──────────────────────────────
    _route(ax, [(x6b + RBW/2, YD), (x7, YD), (x7, YM - RBH3/2)], color=C_EO)

    # ── D2 top → 9a (YES) ────────────────────────────────────────────────────
    _route(ax, [(x8, YM + DH2), (x8, YU - BH/2)], color=C_EG)
    _label(ax, x8 + 0.35, YM + DH2 + 0.25, "Yes", color=C_EG, fs=8.5)

    # ── D2 bottom → 9b (NO) ──────────────────────────────────────────────────
    _route(ax, [(x8, YM - DH2), (x8, YD + BH/2)], color=C_ER)
    _label(ax, x8 + 0.32, YM - DH2 - 0.25, "No", color=C_ER, fs=8.5)

    out = FIGURES / "methodology_fig2_whitebox_protocol.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    CREATED.append((out.name, "White-box one-pixel attack protocol flowchart (corrected)"))
    print(f"  saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# CORRECTION 2 — Figure M3: Two-Stage Experimental Protocol
# ─────────────────────────────────────────────────────────────────────────────

def fig_m3_twostage_protocol():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Figure M3: Two-Stage Experimental Protocol — Conditional Success",
                 fontsize=12, fontweight="bold", color=C_GRAY, pad=8)

    YM = 2.0    # main flow
    YU = 3.2    # upper branch (excluded / success)
    YD = 0.80   # lower branch (fail)

    x1   = 0.60
    x2   = 2.05
    x3   = 3.65    # diamond 1
    xEX  = 5.20    # excluded (upper, from D1)
    x4   = 5.20    # stage 2 (main, from D1)
    x5   = 7.10    # diamond 2
    xSUC = 10.10   # success badge (upper, from D2)
    xFAL = 9.00    # fail badge    (lower, from D2)

    OW, OH    = 1.10, 0.75
    RBW       = 1.45
    DW1, DH1  = 0.85, 0.60
    DW2, DH2  = 0.90, 0.60

    # ── nodes ─────────────────────────────────────────────────────────────────
    _oval(ax, x1, YM, OW, OH,
          "N probes\n5 seeds",
          fc=C_DATA, ec=C_EB, bold=True)

    _rect(ax, x2, YM, RBW, 1.00,
          "Stage 1\nBaseline retrieval\n(no attack)",
          fc=C_BOX, ec=C_GRAY)

    _diamond(ax, x3, YM, DW1, DH1,
             "Baseline\nincorrect?",
             fc=C_DIAM, ec=C_EO)

    _rect(ax, xEX, YU, 1.80, 1.05,
          "Excluded\nMNIST: 59/500\nFMNIST: 408/500",
          fc=C_FAIL, ec=C_ER, color=C_ER)

    _rect(ax, x4, YM, RBW, 1.00,
          "Stage 2\nApply one-pixel\nattack → q*",
          fc=C_BOX, ec=C_GRAY)

    _diamond(ax, x5, YM, DW2, DH2,
             "Retrieval\nunchanged?",
             fc=C_DIAM, ec=C_EO)

    # Success badge — compact 4-line layout
    _rect(ax, xSUC, YU, 2.40, 1.20,
          "✓ CONDITIONAL SUCCESS\n"
          "MNIST: 3.63%  (16/441)\n"
          "FMNIST: 8.70%  (8/92)\n"
          "CIFAR N=500: 2.01%  (4/199)",
          fc=C_OK, ec=C_EG, bold=True, color=C_EG, fs=9.0)

    _rect(ax, xFAL, YD, 1.40, 0.68,
          "✗ Attack\nfails",
          fc=C_FAIL, ec=C_ER, color=C_ER)

    # ── straight arrows on main rail ─────────────────────────────────────────
    _route(ax, [(x1 + OW/2, YM), (x2 - RBW/2, YM)])       # 1 → 2
    _route(ax, [(x2 + RBW/2, YM), (x3 - DW1, YM)])         # 2 → 3
    _route(ax, [(x3 + DW1, YM), (x4 - RBW/2, YM)])         # D1 right → 4
    _route(ax, [(x4 + RBW/2, YM), (x5 - DW2, YM)])         # 4 → 5

    # ── D1 top → Excluded (NO — baseline WAS incorrect → excluded) ────────────
    _route(ax, [(x3, YM + DH1), (x3, YU), (xEX - 0.90, YU)], color=C_ER)
    _label(ax, x3 - 0.38, YM + DH1 + 0.20, "Yes", color=C_ER, fs=8.5)

    # ── D1 right label (YES = baseline was correct → stage 2) ────────────────
    _label(ax, (x3 + DW1 + x4 - RBW/2) / 2, YM + 0.18,
           "No  ✓", color=C_EG, fs=8.5)

    # ── D2 top → Success (YES) ────────────────────────────────────────────────
    _route(ax, [(x5, YM + DH2), (x5, YU), (xSUC - 1.20, YU)], color=C_EG)
    _label(ax, x5 - 0.38, YM + DH2 + 0.20, "Yes  ✓", color=C_EG, fs=8.5)

    # ── D2 bottom → Fail (NO) ────────────────────────────────────────────────
    _route(ax, [(x5, YM - DH2), (x5, YD), (xFAL - 0.70, YD)], color=C_ER)
    _label(ax, x5 - 0.35, YM - DH2 - 0.22, "No", color=C_ER, fs=8.5)

    out = FIGURES / "methodology_fig3_twostage_protocol.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    CREATED.append((out.name, "Two-stage conditional success protocol flowchart (corrected)"))
    print(f"  saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# NEW FIGURE 1 — MNIST Sample Grid
# ─────────────────────────────────────────────────────────────────────────────

def fig_mnist_samples():
    import torch, torchvision
    ds = torchvision.datasets.MNIST(root=str(DATA), train=True, download=False)
    images, labels = ds.data.numpy(), ds.targets.numpy()

    fig, axes = plt.subplots(2, 5, figsize=(8, 4))
    fig.suptitle("MNIST Sample Patterns (one per class)",
                 fontsize=12, fontweight="bold", color=C_GRAY, y=1.02)

    for cls in range(10):
        idx = int(np.where(labels == cls)[0][0])
        ax = axes[cls // 5][cls % 5]
        ax.imshow(images[idx], cmap="gray_r", vmin=0, vmax=255)
        ax.set_title(str(cls), fontsize=11, fontweight="bold", color=C_GRAY)
        ax.axis("off")

    plt.tight_layout()
    out = FIGURES / "dataset_samples_mnist.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    CREATED.append((out.name, "MNIST sample images, one per digit class 0–9"))
    print(f"  saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# NEW FIGURE 2 — Fashion-MNIST Sample Grid
# ─────────────────────────────────────────────────────────────────────────────

FMNIST_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def fig_fmnist_samples():
    import torch, torchvision
    ds = torchvision.datasets.FashionMNIST(root=str(DATA), train=True, download=False)
    images, labels = ds.data.numpy(), ds.targets.numpy()

    fig, axes = plt.subplots(2, 5, figsize=(8, 4))
    fig.suptitle("Fashion-MNIST Sample Patterns (one per class)",
                 fontsize=12, fontweight="bold", color=C_GRAY, y=1.02)

    for cls in range(10):
        idx = int(np.where(labels == cls)[0][0])
        ax = axes[cls // 5][cls % 5]
        ax.imshow(images[idx], cmap="gray_r", vmin=0, vmax=255)
        ax.set_title(FMNIST_NAMES[cls], fontsize=8.5, fontweight="bold", color=C_GRAY)
        ax.axis("off")

    plt.tight_layout()
    out = FIGURES / "dataset_samples_fmnist.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    CREATED.append((out.name, "Fashion-MNIST sample images with class names"))
    print(f"  saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# NEW FIGURE 3 — Raw vs Centered Grayscale CIFAR-10
# ─────────────────────────────────────────────────────────────────────────────

CIFAR_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def fig_cifar_raw_vs_centered():
    import torch, torchvision
    ds  = torchvision.datasets.CIFAR10(root=str(DATA), train=True, download=False)
    raw = ds.data                                       # (50000, 32, 32, 3) uint8
    targets = np.array(ds.targets)
    data_f  = raw.astype(np.float32) / 255.0

    # luminance-weighted grayscale (matches grayscale_cifar_two_stage_centered.py)
    gray = (0.2989 * data_f[:, :, :, 0]
            + 0.5870 * data_f[:, :, :, 1]
            + 0.1140 * data_f[:, :, :, 2])             # (50000, 32, 32)

    # Pick first image per class for classes 0–4
    n_show = 5
    indices = [int(np.where(targets == c)[0][0]) for c in range(n_show)]
    raws = np.stack([gray[i] for i in indices])         # (5, 32, 32)

    # Center each image: subtract per-pixel mean across the 5 images,
    # then unit-normalise — mirrors center_and_normalise() on this small batch.
    X = raws.reshape(n_show, -1).T                     # (1024, 5)
    mu = X.mean(axis=1, keepdims=True)
    Xc = X - mu
    nrm = np.linalg.norm(Xc, axis=0, keepdims=True).clip(min=1e-8)
    Xn  = Xc / nrm                                     # (1024, 5)
    centered = Xn.T.reshape(n_show, 32, 32)

    abs_max = np.abs(centered).max()

    fig, axes = plt.subplots(2, n_show, figsize=(10, 5))
    fig.suptitle("Raw vs Centered Grayscale CIFAR-10 Patterns",
                 fontsize=12, fontweight="bold", color=C_GRAY)

    for i in range(n_show):
        # top row: raw
        axes[0, i].imshow(raws[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(CIFAR_NAMES[i], fontsize=9, fontweight="bold", color=C_GRAY)
        axes[0, i].axis("off")

        # bottom row: centered (signed values → diverging colormap)
        im = axes[1, i].imshow(centered[i], cmap="RdBu_r",
                               vmin=-abs_max, vmax=abs_max)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("centered", fontsize=8, color=C_MID)

    # colorbar for the bottom row
    cbar = fig.colorbar(im, ax=axes[1, :], fraction=0.025, pad=0.04)
    cbar.set_label("Centred value (warm=+, cool=−, white=0)",
                   fontsize=8.5, color=C_GRAY)

    # row labels
    axes[0, 0].annotate("Raw", xy=(-0.05, 0.5), xycoords="axes fraction",
                        rotation=90, ha="right", va="center",
                        fontsize=9, fontweight="bold", color=C_GRAY)
    axes[1, 0].annotate("Centered", xy=(-0.05, 0.5), xycoords="axes fraction",
                        rotation=90, ha="right", va="center",
                        fontsize=9, fontweight="bold", color=C_MID)

    plt.tight_layout()
    out = FIGURES / "dataset_cifar_raw_vs_centered.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    CREATED.append((out.name,
                    "Raw vs centered grayscale CIFAR-10 (classes 0–4); bottom row uses RdBu_r"))
    print(f"  saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# NEW FIGURE 4 — Successful Attack Visualizations (MNIST)
# ─────────────────────────────────────────────────────────────────────────────

def fig_attack_examples_mnist():
    import torch, torchvision
    sys.path.insert(0, str(ROOT))
    from hopfield.sampling import sample_class_balanced

    # Load MNIST
    ds = torchvision.datasets.MNIST(root=str(DATA), train=True, download=False)
    mnist_images = ds.data.float().view(-1, 784) / 255.0
    mnist_labels = ds.targets

    # Load successful attack rows
    with open(EXP / "mnist_two_stage_results.csv") as f:
        rows = list(csv.DictReader(f))

    successes = [r for r in rows
                 if r["baseline_correct"] == "1" and r["a_success"] == "1"][:4]

    if not successes:
        print("  WARNING: no successful MNIST attacks found — skipping Fig 4")
        return

    fig, axes = plt.subplots(len(successes), 3, figsize=(12, 8))
    if len(successes) == 1:
        axes = axes[np.newaxis, :]

    headers = ["ORIGINAL", "ATTACKED", "RETRIEVED (WRONG)"]
    for col, h in enumerate(headers):
        axes[0, col].set_title(h, fontsize=10, fontweight="bold", color=C_GRAY, pad=6)

    for row_i, rec in enumerate(successes):
        seed       = int(rec["seed"])
        probe_idx  = int(rec["probe_idx"])
        cls_name   = rec["class"]
        pixel_i    = int(rec["a_pixel_i"])
        pixel_j    = int(rec["a_pixel_j"])
        pixel_val  = float(rec["a_pixel_val"])
        ret_idx    = int(rec["a_retrieved"])

        # Reconstruct storage matrix (exact same sampling)
        X_gray, _  = sample_class_balanced((mnist_images, mnist_labels), 100, seed=seed)
        stored     = X_gray.T.contiguous()           # (100, 784)

        probe      = stored[probe_idx].numpy().reshape(28, 28)
        attacked   = stored[probe_idx].clone()
        attacked[pixel_i * 28 + pixel_j] = pixel_val
        attacked   = attacked.numpy().reshape(28, 28)

        retrieved  = X_gray[:, ret_idx].numpy().reshape(28, 28)

        # Determine retrieved class label from index
        ret_cls = ret_idx // 10

        # Column 0: original
        axes[row_i, 0].imshow(probe, cmap="gray_r", vmin=0, vmax=1)
        axes[row_i, 0].set_xlabel(f"Original  (class {cls_name})",
                                  fontsize=9, color=C_GRAY)
        axes[row_i, 0].axis("off")

        # Column 1: attacked + red marker
        axes[row_i, 1].imshow(attacked, cmap="gray_r", vmin=0, vmax=1)
        rect = mpatches.Rectangle(
            (pixel_j - 0.5, pixel_i - 0.5), 1, 1,
            linewidth=1.8, edgecolor="red", facecolor="none"
        )
        axes[row_i, 1].add_patch(rect)
        axes[row_i, 1].set_xlabel(
            f"Attacked  (row {pixel_i}, col {pixel_j}, val {pixel_val:.2f})",
            fontsize=9, color=C_ER)
        axes[row_i, 1].axis("off")

        # Column 2: retrieved (wrong) pattern
        axes[row_i, 2].imshow(retrieved, cmap="gray_r", vmin=0, vmax=1)
        axes[row_i, 2].set_xlabel(f"Retrieved as class {ret_cls}",
                                  fontsize=9, color=C_EO)
        axes[row_i, 2].axis("off")

    fig.suptitle("Successful One-Pixel Attacks on MNIST  (N=100, class-balanced)",
                 fontsize=12, fontweight="bold", color=C_GRAY, y=1.01)
    plt.tight_layout()
    out = FIGURES / "attack_examples_mnist.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    CREATED.append((out.name, "4 successful MNIST attacks: original / attacked / retrieved"))
    print(f"  saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# NEW FIGURE 5 — Vulnerability Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def fig_vulnerability_heatmap():
    with open(EXP / "mnist_two_stage_results.csv") as f:
        rows = list(csv.DictReader(f))

    heatmap = np.zeros((28, 28), dtype=float)
    for r in rows:
        if r["baseline_correct"] == "1" and r["a_success"] == "1":
            heatmap[int(r["a_pixel_i"]), int(r["a_pixel_j"])] += 1

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(heatmap, cmap="YlOrRd", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Number of successful attacks at this pixel",
                   fontsize=9, color=C_GRAY)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        "Spatial Distribution of Successful Attack Locations\n"
        "(MNIST N=100 class-balanced, 5 seeds aggregated)",
        fontsize=11, fontweight="bold", color=C_GRAY)

    plt.tight_layout()
    out = FIGURES / "vulnerability_heatmap_mnist.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    CREATED.append((out.name, "28×28 heatmap of successful attack pixel locations on MNIST"))
    print(f"  saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# NEW FIGURE 6 — Capacity Curve
# ─────────────────────────────────────────────────────────────────────────────

def fig_capacity_curve():
    Ns     = [10,   50,   100,  500,  1000]
    means  = [0.0,  4.0,  9.2,  37.2, 47.6]
    stds   = [0.0,  2.0,  3.6,  6.2,  5.0]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xscale("log")

    ax.errorbar(Ns, means, yerr=stds,
                fmt="o-", color=C_EB, lw=2.0, ms=7,
                ecolor=C_MID, elinewidth=1.4, capsize=4,
                label="Baseline failure rate (%)")

    for n, m in zip(Ns, means):
        ax.annotate(f"{m:.1f}%",
                    xy=(n, m), xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", fontsize=9, color=C_GRAY)

    ax.set_xlabel("Storage size  N  (log scale)", fontsize=11, color=C_GRAY)
    ax.set_ylabel("Baseline failure rate  (%)", fontsize=11, color=C_GRAY)
    ax.set_title("MNIST Baseline Retrieval Failure vs Storage Size N",
                 fontsize=12, fontweight="bold", color=C_GRAY)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(n) for n in Ns])
    ax.set_ylim(-2, 60)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=9)
    plt.tight_layout()

    out = FIGURES / "results_fig1_capacity_curve.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    CREATED.append((out.name,
                    "Baseline failure rate vs N on log scale with error bars (5 seeds)"))
    print(f"  saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# NEW FIGURE 7 — Cross-Dataset Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def fig_cross_dataset():
    configs = [
        "MNIST\nN=100 raw",
        "FMNIST\nN=100 raw",
        "Centered\nCIFAR N=100",
        "Centered\nCIFAR N=500",
    ]
    baseline   = [11.8, 81.6,  0.0, 20.4]
    cond_att   = [3.63,  8.70, 0.0,  2.01]
    cos_vals   = [0.396, 0.595, -0.009, -0.001]
    cos_scaled = [v * 100 for v in cos_vals]   # scale to % for same axis

    x   = np.arange(len(configs))
    w   = 0.26

    fig, ax = plt.subplots(figsize=(10, 6))

    b1 = ax.bar(x - w,     baseline,   w, label="Baseline failure (%)",
                color=C_DATA,  edgecolor=C_EB, linewidth=1.2)
    b2 = ax.bar(x,          cond_att,  w, label="Conditional attack success (%)",
                color=C_ATT,   edgecolor=C_EO, linewidth=1.2)
    b3 = ax.bar(x + w,     cos_scaled, w, label="Mean pairwise cosine × 100",
                color=C_BOX,   edgecolor=C_MID, linewidth=1.2, hatch="//")

    # Annotate bar 3 with actual cosine values in italics
    for rect, cv in zip(b3, cos_vals):
        height = rect.get_height()
        ypos   = height + 0.8 if height >= 0 else height - 2.5
        ax.text(rect.get_x() + rect.get_width() / 2, ypos,
                f"{cv:+.3f}", ha="center", va="bottom", fontsize=7.5,
                color=C_MID, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10, color=C_GRAY)
    ax.set_ylabel("Percentage  (%)", fontsize=11, color=C_GRAY)
    ax.set_ylim(-10, 95)
    ax.axhline(0, color=C_GRAY, lw=0.8, ls="-")
    ax.set_title(
        "Cross-Dataset Comparison: Baseline Failure,\n"
        "Conditional Attack Success, and Mean Pairwise Cosine",
        fontsize=12, fontweight="bold", color=C_GRAY)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", ls="--", alpha=0.4)
    plt.tight_layout()

    out = FIGURES / "results_fig10_cross_dataset.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    CREATED.append((out.name,
                    "Cross-dataset grouped bar: baseline failure, conditional attack, cosine"))
    print(f"  saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# NEW FIGURE 8 — Matched-Sigma Amplification
# ─────────────────────────────────────────────────────────────────────────────

def fig_matched_sigma():
    with open(EXP / "phase3_diag_c_matched_sigma.csv") as f:
        rows = list(csv.DictReader(f))

    sigmas = sorted(set(float(r["sigma"]) for r in rows))
    means_pct, stds_pct = [], []
    for s in sigmas:
        vals = [float(r["conditional_failure_rate"]) * 100
                for r in rows if float(r["sigma"]) == s]
        means_pct.append(st.mean(vals))
        stds_pct.append(st.stdev(vals) if len(vals) > 1 else 0.0)

    WB_SUCCESS = 3.10   # conditional WB attack success rate (%)
    BEST_SIGMA = 0.005
    RANDOM_AT_BEST = 0.86

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xscale("log")

    ax.errorbar(sigmas, means_pct, yerr=stds_pct,
                fmt="s-", color=C_EB, lw=2.0, ms=6,
                ecolor=C_MID, elinewidth=1.4, capsize=4,
                label="Random noise conditional failure rate (%)")

    ax.axhline(WB_SUCCESS, color=C_ER, lw=1.8, ls="--",
               label=f"Conditional WB attack success = {WB_SUCCESS}%")
    ax.axvline(BEST_SIGMA, color=C_EO, lw=1.5, ls=":",
               label=f"Best-matched σ ≈ {BEST_SIGMA}  (L2 ≈ 0.11)")

    # Annotate intersection
    ax.annotate(f"Random failure ≈ {RANDOM_AT_BEST}%",
                xy=(BEST_SIGMA, RANDOM_AT_BEST),
                xytext=(BEST_SIGMA * 2.5, RANDOM_AT_BEST + 0.5),
                fontsize=9, color=C_GRAY,
                arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.2))

    # Amplification box
    ax.text(0.97, 0.97,
            "Amplification =\n3.10% / 0.86%\n≈ 3.6×",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color=C_EG, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=C_OK, ec=C_EG, lw=1.4))

    ax.set_xlabel("σ (noise std, log scale)", fontsize=11, color=C_GRAY)
    ax.set_ylabel("Conditional random failure rate  (%)", fontsize=11, color=C_GRAY)
    ax.set_title("Adversarial Amplification at Matched Perturbation Magnitude",
                 fontsize=12, fontweight="bold", color=C_GRAY)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, which="both", ls="--", alpha=0.35)
    ax.set_ylim(-0.1, WB_SUCCESS + 1.5)
    plt.tight_layout()

    out = FIGURES / "results_fig5_matched_sigma.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    CREATED.append((out.name,
                    "Random noise failure vs sigma; amplification=3.6× annotation"))
    print(f"  saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# NEW FIGURE 9 — Per-Class Vulnerability Distribution
# ─────────────────────────────────────────────────────────────────────────────

def fig_vulnerability_classes():
    # From phase3_diag_d_summary.csv (5 seeds × 50 probes = 250, 30 successes)
    counts   = [1, 11, 0, 3, 2, 1, 1, 4, 2, 5]
    classes  = list(range(10))
    expected = 3.0   # 30 successes / 10 classes

    # Highlight over-represented classes (> expected + 1 std)
    threshold = expected + 2.0
    colors = [C_ATT if c > threshold else C_DATA for c in counts]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(classes, counts, color=colors,
                  edgecolor=[C_EO if c > threshold else C_EB for c in counts],
                  linewidth=1.3)

    ax.axhline(expected, color=C_EG, lw=1.8, ls="--",
               label=f"Expected under uniform distribution ({expected:.1f})")

    # Annotate χ² result
    ax.text(0.98, 0.97,
            "χ² = 30.67\np = 0.0004",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color=C_ER, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=C_FAIL, ec=C_ER, lw=1.4))

    ax.set_xticks(classes)
    ax.set_xticklabels([str(c) for c in classes], fontsize=11)
    ax.set_xlabel("Digit class", fontsize=11, color=C_GRAY)
    ax.set_ylabel("Number of vulnerable probes", fontsize=11, color=C_GRAY)
    ax.set_title(
        "Class Distribution of Vulnerable Probes\n"
        "(MNIST N=100 class-balanced, 5 seeds aggregated)",
        fontsize=12, fontweight="bold", color=C_GRAY)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(counts) + 2)
    ax.grid(axis="y", ls="--", alpha=0.4)

    # Value labels above bars
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 0.15, str(c),
                ha="center", va="bottom", fontsize=10, color=C_GRAY)

    plt.tight_layout()
    out = FIGURES / "results_fig6_vulnerability_classes.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    CREATED.append((out.name,
                    "Per-class vulnerable probe counts; class 1 and 9 highlighted; chi2=30.67"))
    print(f"  saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def write_index():
    lines = [
        "# Figures Index\n",
        "Generated by `figures/generate_figures.py`.\n",
        "| Filename | Description |",
        "| --- | --- |",
    ]
    for fname, desc in CREATED:
        lines.append(f"| `{fname}` | {desc} |")
    text = "\n".join(lines) + "\n"

    idx = FIGURES / "FIGURES_INDEX.md"
    idx.write_text(text, encoding="utf-8")
    print(f"\n  index saved: {idx.name}")

    print("\n" + "=" * 70)
    print(f"{'Filename':<48}  Description")
    print("-" * 70)
    for fname, desc in CREATED:
        print(f"  {fname:<46}  {desc}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating thesis figures …\n")
    fig_m2_whitebox_protocol()
    fig_m3_twostage_protocol()
    fig_mnist_samples()
    fig_fmnist_samples()
    fig_cifar_raw_vs_centered()
    fig_attack_examples_mnist()
    fig_vulnerability_heatmap()
    fig_capacity_curve()
    fig_cross_dataset()
    fig_matched_sigma()
    fig_vulnerability_classes()
    write_index()
