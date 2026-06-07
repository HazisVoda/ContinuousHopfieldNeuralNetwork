"""
Post-attack visualization: CIFAR two-stage and MNIST one-pixel attack images.

Generates 4 figures:
  1. attack_cifar_image_grid.png      — clean vs 3 attack variants, all 10 CIFAR classes
  2. attack_cifar_conditions.png      — 2×2 condition comparison bar chart (null result)
  3. attack_mnist_examples.png        — MNIST successful one-pixel attack examples
  4. attack_cross_dataset.png         — MNIST / FMNIST / CIFAR side-by-side

Run: python -m experiments.attack_visualization
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import torch
import torchvision

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hopfield.network  import ContinuousHopfield
from hopfield.sampling import sample_class_balanced

DATA_DIR    = ROOT / "data"
EXP_DIR     = ROOT / "experiments"
FIG_DIR     = ROOT / "figures"
ONE_PIX_DIR = ROOT / "one_pixel_test"

CIFAR_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
]
MNIST_CLASSES  = [str(d) for d in range(10)]
FMNIST_CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",  "Shirt",   "Sneaker",  "Bag",   "Ankle-boot",
]
CHAN_LABEL = ["R", "G", "B"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar_two_stage_results(seed: int = 42) -> list[dict]:
    path = EXP_DIR / "grayscale_cifar_two_stage_results.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if int(r["seed"]) == seed]


def load_cifar_two_stage_summary() -> list[dict]:
    path = EXP_DIR / "grayscale_cifar_two_stage_summary.csv"
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_mnist() -> tuple[torch.Tensor, torch.Tensor]:
    ds      = torchvision.datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
    images  = torch.tensor(
        ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0,
        dtype=torch.float32,
    )
    labels  = torch.tensor(ds.targets.numpy(), dtype=torch.long)
    return images, labels


def load_cifar_gray() -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (gray_images, labels) matching the grayscale_cifar_two_stage_attack format."""
    ds  = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False)
    raw = ds.data.astype(np.float32) / 255.0      # (50000, 32, 32, 3)
    gray = (0.2989 * raw[:, :, :, 0]
            + 0.5870 * raw[:, :, :, 1]
            + 0.1140 * raw[:, :, :, 2])           # (50000, 32, 32)
    gray_t   = torch.tensor(gray.reshape(-1, 1024), dtype=torch.float32)
    labels_t = torch.tensor(ds.targets, dtype=torch.long)
    return gray_t, labels_t


def load_phase3_vulnerable(n: int = 100, strategy: str = "class_balanced",
                            max_examples: int = 6) -> list[dict]:
    path = EXP_DIR / "phase3_whitebox_results.csv"
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (int(r["N"]) == n and r["strategy"] == strategy
                    and r["attack_success"] == "1"):
                rows.append({
                    "seed":           int(r["seed"]),
                    "true_index":     int(r["true_index"]),
                    "pixel_i":        int(r["pixel_i"]),
                    "pixel_j":        int(r["pixel_j"]),
                    "pixel_value":    float(r["pixel_value"]),
                    "original_value": float(r["original_value"]),
                    "retrieved_index": int(r["retrieved_index"]),
                })
                if len(rows) == max_examples:
                    break
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load a saved PNG from one_pixel_test/
# ─────────────────────────────────────────────────────────────────────────────

def _png_gray(path: Path) -> np.ndarray:
    """Read grayscale PNG (saved with cmap='gray') → (32, 32) float in [0,1]."""
    arr = mpimg.imread(str(path))   # (32, 32, 4) RGBA or (32, 32, 3)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    return arr[:, :, 0]             # R=G=B for gray colormap


def _png_rgb(path: Path) -> np.ndarray:
    """Read color PNG → (32, 32, 3) float in [0,1]."""
    arr = mpimg.imread(str(path))
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    return np.clip(arr[:, :, :3], 0, 1)


def _highlight_pixel(ax, row: int, col: int, color: str = "red",
                     lw: float = 1.5, sz: int = 1) -> None:
    """Draw a rectangle outline around (row, col) in a displayed image."""
    rect = mpatches.Rectangle(
        (col - sz / 2, row - sz / 2), sz, sz,
        linewidth=lw, edgecolor=color, facecolor="none",
    )
    ax.add_patch(rect)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: CIFAR two-stage image grid (seed 42, all 10 classes)
# ─────────────────────────────────────────────────────────────────────────────

def fig_cifar_image_grid(X_stored: torch.Tensor) -> None:
    """
    X_stored: (1024, 10) stored patterns for seed=42, N=10 — used to display
              the network's retrieved pattern in the 5th column.
    """
    seed     = 42
    seed_dir = ONE_PIX_DIR / f"seed_{seed}"
    rows     = load_cifar_two_stage_results(seed)

    # columns: clean | A1 gray attack | A2 RGB→gray | A2 RGB color | Retrieved (A1)
    n_cols   = 5
    n_rows   = 10
    col_titles = [
        "Clean stored\npattern",
        "A1: Gray-space\nattack (5,120 cands)",
        "A2: RGB channel\nattack — gray",
        "A2: RGB channel\nattack — color",
        "Network output\n(A1 retrieved pattern)",
    ]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5 + 0.8))
    fig.suptitle(
        "Grayscale CIFAR-10 — Two-Stage One-Pixel Attack\n"
        "Seed 42 · N=10 class-balanced · β=8.0 · All 10 classes",
        fontsize=11, fontweight="bold", y=1.01,
    )

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=8, pad=4)

    for row in rows:
        i      = int(row["probe_idx"])
        cname  = CIFAR_CLASSES[i]
        prefix = f"probe{i:02d}_{cname}"

        clean  = _png_gray(seed_dir / f"{prefix}_clean.png")
        a1     = _png_gray(seed_dir / f"{prefix}_a1_gray_attacked.png")
        a2_g   = _png_gray(seed_dir / f"{prefix}_a2_rgb_attacked_gray.png")
        a2_c   = _png_rgb( seed_dir / f"{prefix}_a2_rgb_attacked_color.png")

        # Network's retrieved pattern for A1 — the stored pattern nearest to
        # what the network output after receiving the attacked query.
        a1_ret_idx   = int(row["a1_retrieved"])
        retrieved_img = X_stored[:, a1_ret_idx].view(32, 32).numpy()
        ret_class     = CIFAR_CLASSES[a1_ret_idx]

        a1_pi, a1_pj = int(row["a1_pixel_i"]), int(row["a1_pixel_j"])
        a2_pi, a2_pj = int(row["a2_pixel_i"]), int(row["a2_pixel_j"])
        a2_chan       = row["a2_channel"]
        bl_ok         = int(row["baseline_correct"])
        a1_success    = int(row["a1_success"])

        # Background: green=correct baseline, red=failing baseline
        bg = "#d4edda" if bl_ok else "#f8d7da"

        for col_idx, (img, cmap) in enumerate([
            (clean, "gray"), (a1, "gray"), (a2_g, "gray"),
            (a2_c, None), (retrieved_img, "gray")
        ]):
            ax = axes[i, col_idx]
            ax.set_facecolor(bg)
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1,
                      interpolation="nearest", aspect="equal")
            ax.axis("off")
            ax.set_xlim(-0.5, 31.5)
            ax.set_ylim(31.5, -0.5)

            # Highlight attacked pixel
            if col_idx == 1:
                _highlight_pixel(ax, a1_pi, a1_pj, color="red", lw=1.5, sz=3)
                ax.set_title(
                    f"px ({a1_pi},{a1_pj})  L2={float(row['a1_l2']):.3f}",
                    fontsize=6, pad=1,
                )
            elif col_idx in (2, 3):
                _highlight_pixel(ax, a2_pi, a2_pj,
                                 color="cyan" if col_idx == 3 else "red",
                                 lw=1.5, sz=3)
                if col_idx == 2:
                    ax.set_title(
                        f"ch={a2_chan} ({a2_pi},{a2_pj})  L2={float(row['a2_l2']):.3f}",
                        fontsize=6, pad=1,
                    )
            elif col_idx == 4:
                # Label retrieved class; highlight if different from true class
                match = "=" if ret_class == cname else "≠"
                color = "green" if ret_class == cname else "red"
                ax.set_title(f"→ {ret_class} {match}", fontsize=7,
                             pad=2, color=color, fontweight="bold")

        # Row label
        status = "CORRECT" if bl_ok else "FAIL"
        axes[i, 0].set_ylabel(
            f"{i}: {cname}\n[{status}]",
            fontsize=7, rotation=0, labelpad=55, va="center",
        )

    # Legend
    correct_patch = mpatches.Patch(color="#d4edda", label="Baseline-correct probe")
    fail_patch    = mpatches.Patch(color="#f8d7da", label="Baseline-failing probe")
    fig.legend(handles=[correct_patch, fail_patch],
               loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    out = FIG_DIR / "attack_cifar_image_grid.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: CIFAR two-stage condition comparison (null result chart)
# ─────────────────────────────────────────────────────────────────────────────

def fig_cifar_conditions() -> None:
    summary = load_cifar_two_stage_summary()

    cond_keys = [
        ("Baseline\n(no attack)",  "bl_fail",   None),
        ("A1\ngray attack\nas query", "a1_raw", "a1_cond"),
        ("A2\nRGB attack\nas query",  "a2_raw", "a2_cond"),
        ("B1\ngray-attacked\nstored",  "b1_raw", "b1_cond"),
        ("B2\nRGB-attacked\nstored",   "b2_raw", "b2_cond"),
    ]

    # Build arrays
    N = 10
    bl_vals   = [1.0 - int(r["n_baseline_correct"]) / N for r in summary]

    def _vals(key: str) -> list[float]:
        return [float(r[key]) if r.get(key, "nan") != "nan" else 0.0
                for r in summary]

    a1r = _vals("a1_raw");  a1c = _vals("a1_cond")
    a2r = _vals("a2_raw");  a2c = _vals("a2_cond")
    b1r = _vals("b1_raw");  b1c = _vals("b1_cond")
    b2r = _vals("b2_raw");  b2c = _vals("b2_cond")

    groups = [
        ("Baseline\nfailure",         bl_vals, None),
        ("A1 raw\nsuccess",            a1r,    None),
        ("A1 cond.\nsuccess",          a1c,    None),
        ("A2 raw\nsuccess",            a2r,    None),
        ("A2 cond.\nsuccess",          a2c,    None),
        ("B1 raw\nfailure",            b1r,    None),
        ("B1 cond.\nfailure",          b1c,    None),
        ("B2 raw\nfailure",            b2r,    None),
        ("B2 cond.\nfailure",          b2c,    None),
    ]

    colors = (["#6c757d"]
              + ["#1f77b4", "#aec7e8"] * 2
              + ["#d62728", "#f7b6b6"] * 2)

    means  = [float(np.mean(vals)) for _, vals, _ in groups]
    stds   = [float(np.std(vals, ddof=1)) for _, vals, _ in groups]
    labels = [lbl for lbl, _, _ in groups]
    x      = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.6,
                  color=colors, edgecolor="white", linewidth=0.8)

    # Scatter individual seed points
    for xi, (_, vals, _) in enumerate(groups):
        jitter = np.random.RandomState(xi).uniform(-0.12, 0.12, len(vals))
        ax.scatter(xi + jitter, vals, color="black", s=18, zorder=5, alpha=0.7)

    # Annotate mean values
    for xi, (mean, std) in enumerate(zip(means, stds)):
        ax.text(xi, mean + std + 0.02, f"{mean*100:.0f}%",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    # Baseline reference line
    ax.axhline(float(np.mean(bl_vals)), color="#6c757d",
               linestyle="--", lw=1.2, alpha=0.6, label="Baseline failure (80%)")
    ax.axhline(0.0, color="green", linestyle="-", lw=1.0, alpha=0.4,
               label="Zero attack success")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(-0.08, 1.1)
    ax.set_ylabel("Rate (mean ± std, 5 seeds)", fontsize=9)
    ax.set_title(
        "Grayscale CIFAR-10 — 2×2 Two-Stage Attack: All Conditions (N=10, β=8.0)\n"
        "Conditional success = 0% across all conditions → "
        "Null result extends to all attack configurations",
        fontsize=9, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    # Condition group dividers
    for xv in [0.5, 2.5, 4.5, 6.5]:
        ax.axvline(xv, color="gray", lw=0.8, alpha=0.4, linestyle=":")

    # Group labels at top
    for label, xc in [("Baseline", 0), ("A1 (query)", 1.5),
                       ("A2 (query)", 3.5), ("B1 (stored)", 5.5), ("B2 (stored)", 7.5)]:
        ax.text(xc, 1.05, label, ha="center", fontsize=8, color="dimgray",
                style="italic")

    plt.tight_layout()
    out = FIG_DIR / "attack_cifar_conditions.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: MNIST successful attack examples
# ─────────────────────────────────────────────────────────────────────────────

def fig_mnist_examples(mnist_images: torch.Tensor, mnist_labels: torch.Tensor) -> None:
    examples = load_phase3_vulnerable(n=100, strategy="class_balanced", max_examples=6)
    if not examples:
        print("  No MNIST attack examples found — skipping Figure 3")
        return

    # Pre-load stored patterns for each unique seed required
    N_STORE = 100
    cache: dict[int, torch.Tensor] = {}
    for ex in examples:
        s = ex["seed"]
        if s not in cache:
            X, _ = sample_class_balanced((mnist_images, mnist_labels), N_STORE, seed=s)
            cache[s] = X   # (784, 100)

    n_ex    = len(examples)
    fig, axes = plt.subplots(n_ex, 3, figsize=(7, n_ex * 1.8 + 1.2))
    if n_ex == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        "MNIST One-Pixel White-Box Attack — Successful Examples\n"
        "N=100 class-balanced · β=8.0 · 3,920 candidates (784 × 5)",
        fontsize=10, fontweight="bold",
    )
    col_titles = ["Clean stored pattern", "After one-pixel attack\n(pixel highlighted)",
                  "Retrieved false pattern"]
    for ci, t in enumerate(col_titles):
        axes[0, ci].set_title(t, fontsize=8, pad=4)

    for ei, ex in enumerate(examples):
        X         = cache[ex["seed"]]                         # (784, 100)
        ti        = ex["true_index"]
        ri        = ex["retrieved_index"]
        pi, pj    = ex["pixel_i"], ex["pixel_j"]
        pv        = ex["pixel_value"]
        ov        = ex["original_value"]

        q_clean   = X[:, ti].clone()                          # (784,)
        q_attacked = q_clean.clone()
        q_attacked[pi * 28 + pj] = pv

        true_class = ti // 10
        ret_class  = ri // 10

        for ci, (vec, label, highlight) in enumerate([
            (q_clean,    f"Digit {true_class}", False),
            (q_attacked, f"Digit {true_class}\n→ pixel ({pi},{pj}): {ov:.2f}→{pv:.2f}", True),
            (X[:, ri],   f"Retrieved: digit {ret_class}", False),
        ]):
            ax  = axes[ei, ci]
            img = vec.view(28, 28).numpy()
            ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.axis("off")
            ax.set_title(label, fontsize=6.5, pad=2)

            if highlight:
                ax.set_xlim(-0.5, 27.5)
                ax.set_ylim(27.5, -0.5)
                _highlight_pixel(ax, pi, pj, color="red", lw=2.0, sz=2)

        # Row label
        axes[ei, 0].set_ylabel(f"Seed {ex['seed']}\nprobe {ti}", fontsize=6,
                                rotation=0, labelpad=50, va="center")

    plt.tight_layout()
    out = FIG_DIR / "attack_mnist_examples.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Cross-dataset comparison (updated with CIFAR two-stage)
# ─────────────────────────────────────────────────────────────────────────────

def fig_cross_dataset() -> None:
    # ── Values from prior experiments ────────────────────────────────────────
    datasets = ["MNIST\nN=100", "Fashion-MNIST\nN=100", "CIFAR-10\nN=10"]
    colors   = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Baseline failure rate
    bl_means = [0.092, 0.800, 0.800]
    bl_stds  = [0.021, 0.0,   0.173]

    # Conditional white-box attack success (among baseline-correct probes)
    wb_means = [0.031, 0.076, 0.000]
    wb_stds  = [0.012, 0.051, 0.000]

    # Mean pairwise cosine at tested N
    cos_vals = [0.397, 0.648, 0.832]

    # CIFAR two-stage cond success for A1/A2/B1/B2 (all 0%)
    cifar_cond_keys = ["A1\ngray\nquery", "A2\nRGB\nquery",
                       "B1\ngray\nstored", "B2\nRGB\nstored"]
    cifar_cond_vals = [0.0, 0.0, 0.0, 0.0]

    fig = plt.figure(figsize=(14, 10))
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.38)

    # ── Top-left: baseline failure rate ──────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    xp = np.arange(len(datasets))
    bars = ax.bar(xp, bl_means, yerr=bl_stds, capsize=5, width=0.5,
                  color=colors, edgecolor="white")
    for xi, (m, s) in enumerate(zip(bl_means, bl_stds)):
        ax.annotate(f"{m*100:.1f}%", xy=(xi, m + s + 0.02),
                    ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(xp); ax.set_xticklabels(datasets, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Baseline failure rate", fontsize=9)
    ax.set_title("Clean retrieval baseline failure\n(network regime diagnostic)",
                 fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(0.2, color="gray", ls=":", lw=1, alpha=0.5)
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5)

    # ── Top-right: conditional WB attack success ──────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    bars = ax.bar(xp, wb_means, yerr=wb_stds, capsize=5, width=0.5,
                  color=colors, edgecolor="white")
    for xi, (m, s) in enumerate(zip(wb_means, wb_stds)):
        ax.annotate(f"{m*100:.1f}%", xy=(xi, m + s + 0.003),
                    ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(xp); ax.set_xticklabels(datasets, fontsize=8)
    ax.set_ylim(0, 0.20)
    ax.set_ylabel("Conditional WB attack success", fontsize=9)
    ax.set_title("One-pixel attack success\n(conditional on baseline-correct probes)",
                 fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(0.0, color="green", ls="-", lw=1, alpha=0.4)

    # ── Bottom-left: mean pairwise cosine ─────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    bars = ax.bar(xp, cos_vals, width=0.5, color=colors, edgecolor="white")
    for xi, m in enumerate(cos_vals):
        ax.annotate(f"{m:.3f}", xy=(xi, m + 0.01), ha="center",
                    fontsize=8.5, fontweight="bold")
    ax.set_xticks(xp); ax.set_xticklabels(datasets, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean off-diagonal pairwise cosine", fontsize=9)
    ax.set_title("Pattern crowding\n(root cause of capacity saturation)",
                 fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5,
               label="High-crowding threshold")
    ax.legend(fontsize=7)

    # ── Bottom-right: CIFAR two-stage null result breakdown ───────────────────
    ax = fig.add_subplot(gs[1, 1])
    xc = np.arange(len(cifar_cond_keys))
    ax.bar(xc, [0.80]*4, width=0.5, color="#aec7e8", edgecolor="white",
           label="Raw success (= baseline failure)")
    ax.bar(xc, cifar_cond_vals, width=0.5, color="#d62728", edgecolor="white",
           label="Conditional success")
    ax.axhline(0.80, color="#1f77b4", ls="--", lw=1.5, alpha=0.7,
               label="Baseline failure rate (80%)")
    ax.set_xticks(xc); ax.set_xticklabels(cifar_cond_keys, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate", fontsize=9)
    ax.set_title("CIFAR-10 two-stage: all 4 conditions\n"
                 "Conditional success = 0% across all configurations",
                 fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=7)
    ax.text(1.5, 0.15, "NULL\nRESULT", ha="center", va="center",
            fontsize=22, fontweight="bold", color="#d62728", alpha=0.25,
            rotation=0)

    fig.suptitle(
        "Cross-Dataset One-Pixel Attack Summary\n"
        "Continuous Hopfield Network · β=8.0 · White-box exhaustive attacker",
        fontsize=11, fontweight="bold",
    )
    out = FIG_DIR / "attack_cross_dataset.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 & 6: MNIST / FMNIST two-stage image grids (4 columns)
# ─────────────────────────────────────────────────────────────────────────────

def _load_two_stage_results(dataset: str, seed: int) -> list[dict]:
    path = EXP_DIR / f"{dataset}_two_stage_results.csv"
    out: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["seed"]) == seed:
                out.append({
                    "probe_idx":        int(r["probe_idx"]),
                    "class":            r["class"],
                    "baseline_correct": int(r["baseline_correct"]),
                    "a_success":        int(r["a_success"]),
                    "a_retrieved":      int(r["a_retrieved"]),
                    "a_pixel_i":        int(r["a_pixel_i"]),
                    "a_pixel_j":        int(r["a_pixel_j"]),
                    "a_pixel_val":      float(r["a_pixel_val"]),
                    "a_l2":             float(r["a_l2"]),
                    "a_cosine_delta":   float(r["a_cosine_delta"]),
                    "b_success":        int(r["b_success"]),
                    "b_retrieved":      int(r["b_retrieved"]),
                })
    return out


def _probe_prefix(dataset: str, probe_idx: int, cls_name: str) -> str:
    """Reconstruct the filename prefix used when the experiment saved PNGs."""
    if dataset == "mnist":
        return f"probe{probe_idx:03d}_dig{cls_name}"
    return f"probe{probe_idx:03d}_{cls_name}"


def fig_dataset_image_grid(
    dataset:     str,              # "mnist" or "fmnist"
    class_names: list[str],
    img_sz:      int,              # 28
    seed:        int,
    images:      torch.Tensor,     # full dataset (N_total, d)
    labels:      torch.Tensor,
    N:           int = 100,
    beta:        float = 8.0,
) -> None:
    """
    4-column grid matching the CIFAR attack_cifar_image_grid style.

    Columns:
      1. Clean stored pattern
      2. Attacked query (A condition) — pixel highlighted; green border if success
      3. A: Retrieved stored pattern   — green "=" or red "≠"
      4. B: Retrieved stored pattern   — green "=" or red "≠" (clean query, poisoned store)

    Row selection: one probe per class (10 rows), preferring attack successes.
    """
    n_per_class = N // 10
    rows_all    = _load_two_stage_results(dataset, seed)
    if not rows_all:
        print(f"  No data for {dataset} seed={seed} — skipping")
        return

    X, _ = sample_class_balanced((images, labels), N, seed=seed)  # (d, N)
    seed_dir = ONE_PIX_DIR / f"{dataset}_seed_{seed}"

    # One probe per class — baseline_correct=1 first, then attack success,
    # then lowest probe_idx for a deterministic result.
    # This guarantees green backgrounds wherever the network retrieves correctly.
    selected: list[dict] = []
    for cls in range(10):
        all_cls = [r for r in rows_all if r["probe_idx"] // n_per_class == cls]
        if not all_cls:
            continue
        # Prefer correct-baseline probes; fall back to any if none exist
        correct = [r for r in all_cls if r["baseline_correct"] == 1]
        pool    = correct if correct else all_cls
        pool.sort(key=lambda r: (
            -(r["a_success"] or r["b_success"]),  # attack success preferred
            r["probe_idx"],                        # deterministic tiebreak
        ))
        selected.append(pool[0])

    n_rows = len(selected)
    col_titles = [
        "Clean stored\npattern",
        f"Attacked query\n(A: {N * len([0.0, 0.25, 0.5, 0.75, 1.0]) * img_sz * img_sz // img_sz // img_sz:,} cands)",
        "A: Network output\n(retrieved pattern)",
        "B: Network output\n(clean query, poisoned store)",
    ]

    fig, axes = plt.subplots(n_rows, 4, figsize=(4 * 2.5, n_rows * 2.5 + 0.8))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    dname = dataset.upper().replace("FMNIST", "Fashion-MNIST")
    fig.suptitle(
        f"{dname} — Two-Stage One-Pixel Attack\n"
        f"Seed {seed} · N={N} class-balanced · β={beta} · One row per class",
        fontsize=11, fontweight="bold", y=1.01,
    )

    # Correct column title for A candidates
    n_cands = img_sz * img_sz * 5   # 784 × 5 = 3920
    axes[0, 1].set_title(f"Attacked query\n(A: {n_cands:,} cands)", fontsize=8, pad=4)
    for ci, title in enumerate(col_titles):
        if ci != 1:
            axes[0, ci].set_title(title, fontsize=8, pad=4)

    for ri, row in enumerate(selected):
        i         = row["probe_idx"]
        cls       = i // n_per_class
        cname     = class_names[cls]
        bl_ok     = row["baseline_correct"]
        a_suc     = row["a_success"]
        b_suc     = row["b_success"]
        a_pi      = row["a_pixel_i"]
        a_pj      = row["a_pixel_j"]
        a_ret     = row["a_retrieved"]
        b_ret     = row["b_retrieved"]

        prefix     = _probe_prefix(dataset, i, cname)
        clean      = _png_gray(seed_dir / f"{prefix}_clean.png")
        attacked_a = _png_gray(seed_dir / f"{prefix}_attacked.png")

        a_ret_cls   = a_ret // n_per_class
        a_ret_cname = class_names[a_ret_cls]
        a_output    = X[:, a_ret].view(img_sz, img_sz).numpy()

        b_ret_cls   = b_ret // n_per_class
        b_ret_cname = class_names[b_ret_cls]
        b_output    = X[:, b_ret].view(img_sz, img_sz).numpy()

        bg = "#d4edda" if bl_ok else "#f8d7da"

        for ci, img in enumerate([clean, attacked_a, a_output, b_output]):
            ax = axes[ri, ci]
            ax.set_facecolor(bg)
            ax.imshow(img, cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest", aspect="equal")
            ax.axis("off")
            ax.set_xlim(-0.5, img_sz - 0.5)
            ax.set_ylim(img_sz - 0.5, -0.5)

            if ci == 1:
                _highlight_pixel(ax, a_pi, a_pj, color="red", lw=1.5, sz=2)
                ax.set_title(
                    f"px ({a_pi},{a_pj})  L2={row['a_l2']:.3f}",
                    fontsize=6, pad=1,
                    color="red" if a_suc else "black",
                )
                if a_suc:
                    for sp in ax.spines.values():
                        sp.set_edgecolor("limegreen"); sp.set_linewidth(3)
            elif ci == 2:
                match = "=" if a_ret_cls == cls else "≠"
                color = "green" if a_ret_cls == cls else "red"
                ax.set_title(f"→ {a_ret_cname} {match}",
                             fontsize=7, pad=2, color=color, fontweight="bold")
            elif ci == 3:
                match = "=" if b_ret_cls == cls else "≠"
                color = "green" if b_ret_cls == cls else "red"
                ax.set_title(f"→ {b_ret_cname} {match}",
                             fontsize=7, pad=2, color=color, fontweight="bold")
                if b_suc:
                    for sp in ax.spines.values():
                        sp.set_edgecolor("limegreen"); sp.set_linewidth(3)

        status  = "CORRECT" if bl_ok else "FAIL"
        atk_tag = " ★" if (a_suc or b_suc) else ""
        axes[ri, 0].set_ylabel(
            f"{cname}\n[{status}]{atk_tag}",
            fontsize=7, rotation=0, labelpad=58, va="center",
        )

    correct_patch = mpatches.Patch(color="#d4edda", label="Baseline-correct probe")
    fail_patch    = mpatches.Patch(color="#f8d7da", label="Baseline-failing probe")
    star_patch    = mpatches.Patch(facecolor="white", edgecolor="limegreen",
                                   linewidth=2, label="Attack succeeded (★ / green border)")
    fig.legend(handles=[correct_patch, fail_patch, star_patch],
               loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    out = FIG_DIR / f"attack_{dataset}_image_grid.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 & 8: CIFAR centered+norm image grids (N=100 and N=500)
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar_centered_results(N: int, seed: int) -> list[dict]:
    path = EXP_DIR / "cifar_centered_two_stage_results.csv"
    out: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["N"]) == N and int(r["seed"]) == seed:
                out.append({
                    "N":                int(r["N"]),
                    "seed":             int(r["seed"]),
                    "probe_idx":        int(r["probe_idx"]),
                    "class":            r["class"],
                    "baseline_correct": int(r["baseline_correct"]),
                    "a1_success":       int(r["a1_success"]),
                    "a1_retrieved":     int(r["a1_retrieved"]),
                    "a1_pixel_i":       int(r["a1_pixel_i"]),
                    "a1_pixel_j":       int(r["a1_pixel_j"]),
                    "a1_pixel_val":     float(r["a1_pixel_val"]),
                    "a1_l2":            float(r["a1_l2"]),
                    "a1_cosine_delta":  float(r["a1_cosine_delta"]),
                    "a2_success":       int(r["a2_success"]),
                    "a2_retrieved":     int(r["a2_retrieved"]),
                    "a2_pixel_i":       int(r["a2_pixel_i"]),
                    "a2_pixel_j":       int(r["a2_pixel_j"]),
                    "a2_channel":       r["a2_channel"],
                    "a2_pixel_val":     float(r["a2_pixel_val"]),
                    "a2_l2":            float(r["a2_l2"]),
                    "b1_success":       int(r["b1_success"]),
                    "b2_success":       int(r["b2_success"]),
                })
    return out


def fig_cifar_centered_image_grid(N: int, seed: int,
                                   X_raw: torch.Tensor) -> None:
    """
    5-column grid matching attack_cifar_image_grid.png format, for the
    centered+normalised CIFAR model.

    X_raw: (1024, N) raw pixel stored patterns for the given (N, seed) —
           used to display the retrieved pattern in the 5th column.

    Row selection: one probe per class (10 rows), prioritising attack successes
    over baseline-correct probes so interesting cases appear first.
    """
    n_per_class = N // 10
    rows_all    = load_cifar_centered_results(N, seed)
    if not rows_all:
        print(f"  No data for N={N} seed={seed} — skipping")
        return

    # One row per class: prefer attack success, then any baseline-correct, then any
    selected: list[dict] = []
    for cls in range(10):
        cls_rows = [r for r in rows_all if r["probe_idx"] // n_per_class == cls]
        if not cls_rows:
            continue
        # Sort: attack success first, then baseline correct, then anything
        cls_rows.sort(key=lambda r: (
            -(r["a1_success"] or r["a2_success"]),
            -r["baseline_correct"],
        ))
        selected.append(cls_rows[0])

    n_rows  = len(selected)
    n_cols  = 5
    col_titles = [
        "Clean stored\npattern",
        "A1: Gray-space\nattack (5,120 cands)",
        "A2: RGB channel\nattack — gray",
        "A2: RGB channel\nattack — color",
        "Network output\n(A1 retrieved pattern)",
    ]

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.5, n_rows * 2.5 + 0.8))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    preprocessing = "Centered+Normalised"
    fig.suptitle(
        f"Grayscale CIFAR-10 — Two-Stage One-Pixel Attack ({preprocessing})\n"
        f"Seed {seed} · N={N} class-balanced · β=8.0 · One row per class",
        fontsize=11, fontweight="bold", y=1.01,
    )
    for ci, title in enumerate(col_titles):
        axes[0, ci].set_title(title, fontsize=8, pad=4)

    seed_dir = ONE_PIX_DIR / f"cifar_centered_N{N}" / f"seed_{seed}"

    for ri, row in enumerate(selected):
        i     = row["probe_idx"]
        cls   = i // n_per_class
        cname = CIFAR_CLASSES[cls]
        prefix = f"probe{i:03d}_{cname}"

        clean = _png_gray(seed_dir / f"{prefix}_clean.png")
        a1    = _png_gray(seed_dir / f"{prefix}_a1_gray.png")
        a2_g  = _png_gray(seed_dir / f"{prefix}_a2_rgb_gray.png")
        a2_c  = _png_rgb( seed_dir / f"{prefix}_a2_rgb_color.png")

        # 5th column: nearest stored pattern retrieved after A1 attack
        a1_ret_idx    = row["a1_retrieved"]
        retrieved_img = X_raw[:, a1_ret_idx].view(32, 32).numpy()
        ret_cls       = a1_ret_idx // n_per_class
        ret_cname     = CIFAR_CLASSES[ret_cls]

        a1_pi, a1_pj = row["a1_pixel_i"], row["a1_pixel_j"]
        a2_pi, a2_pj = row["a2_pixel_i"], row["a2_pixel_j"]
        a2_chan       = row["a2_channel"]
        bl_ok         = row["baseline_correct"]
        a1_suc        = row["a1_success"]
        a2_suc        = row["a2_success"]

        bg = "#d4edda" if bl_ok else "#f8d7da"

        for ci, (img, cmap) in enumerate([
            (clean,         "gray"),
            (a1,            "gray"),
            (a2_g,          "gray"),
            (a2_c,          None),
            (retrieved_img, "gray"),
        ]):
            ax = axes[ri, ci]
            ax.set_facecolor(bg)
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1,
                      interpolation="nearest", aspect="equal")
            ax.axis("off")
            ax.set_xlim(-0.5, 31.5)
            ax.set_ylim(31.5, -0.5)

            if ci == 1:
                _highlight_pixel(ax, a1_pi, a1_pj, color="red", lw=1.5, sz=3)
                ax.set_title(
                    f"px ({a1_pi},{a1_pj})  L2={row['a1_l2']:.3f}",
                    fontsize=6, pad=1,
                    color="red" if a1_suc else "black",
                )
                if a1_suc:
                    for sp in ax.spines.values():
                        sp.set_edgecolor("limegreen"); sp.set_linewidth(3)
            elif ci == 2:
                _highlight_pixel(ax, a2_pi, a2_pj, color="red", lw=1.5, sz=3)
                ax.set_title(
                    f"ch={a2_chan} ({a2_pi},{a2_pj})  L2={row['a2_l2']:.3f}",
                    fontsize=6, pad=1,
                    color="red" if a2_suc else "black",
                )
                if a2_suc:
                    for sp in ax.spines.values():
                        sp.set_edgecolor("limegreen"); sp.set_linewidth(3)
            elif ci == 3:
                _highlight_pixel(ax, a2_pi, a2_pj, color="cyan", lw=1.5, sz=3)
                if a2_suc:
                    for sp in ax.spines.values():
                        sp.set_edgecolor("limegreen"); sp.set_linewidth(3)
            elif ci == 4:
                match = "=" if ret_cls == cls else "≠"
                color = "green" if ret_cls == cls else "red"
                ax.set_title(f"→ {ret_cname} {match}",
                             fontsize=7, pad=2, color=color, fontweight="bold")

        status = "CORRECT" if bl_ok else "FAIL"
        atk_tag = " ★" if (a1_suc or a2_suc) else ""
        axes[ri, 0].set_ylabel(
            f"{cls}: {cname}\n[{status}]{atk_tag}",
            fontsize=7, rotation=0, labelpad=60, va="center",
        )

    # Legend
    correct_patch = mpatches.Patch(color="#d4edda", label="Baseline-correct probe")
    fail_patch    = mpatches.Patch(color="#f8d7da", label="Baseline-failing probe")
    star_patch    = mpatches.Patch(facecolor="white", edgecolor="limegreen",
                                   linewidth=2, label="Attack succeeded (★ / green border)")
    fig.legend(handles=[correct_patch, fail_patch, star_patch],
               loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    out = FIG_DIR / f"attack_cifar_centered_N{N}_grid.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Post-Attack Visualization")
    print("=" * 60)

    # Check prerequisites
    two_stage_csv = EXP_DIR / "grayscale_cifar_two_stage_results.csv"
    if not two_stage_csv.exists():
        print(f"  ERROR: {two_stage_csv.name} not found.")
        print("  Run: python -m experiments.grayscale_cifar_two_stage_attack  first.")
        return

    if not (ONE_PIX_DIR / "seed_42").exists():
        print("  ERROR: one_pixel_test/seed_42/ not found.")
        print("  Run: python -m experiments.grayscale_cifar_two_stage_attack  first.")
        return

    print("\nLoading MNIST ...")
    mnist_images, mnist_labels = load_mnist()
    print(f"  ok  {len(mnist_images)} samples")

    print("Loading CIFAR-10 (for network output column) ...")
    cifar_gray, cifar_labels = load_cifar_gray()
    X_cifar, _ = sample_class_balanced((cifar_gray, cifar_labels), 10, seed=42)
    print(f"  ok  stored patterns: {X_cifar.shape}")

    print("\nGenerating figures ...")

    print("Figure 1: CIFAR two-stage image grid (raw N=10) ...")
    fig_cifar_image_grid(X_cifar)

    print("Figure 2: CIFAR two-stage conditions comparison ...")
    fig_cifar_conditions()

    print("Figure 3: MNIST successful attack examples ...")
    fig_mnist_examples(mnist_images, mnist_labels)

    print("Figure 4: Cross-dataset comparison ...")
    fig_cross_dataset()

    # ── MNIST / FMNIST 4-column grids ────────────────────────────────────────
    # MNIST seed 42: all classes have baseline-correct probes.
    # FMNIST seed 44: the only seed where ALL 10 classes have at least one
    #   baseline-correct probe (seed 42 has 0 correct probes for Pullover,
    #   Dress, Coat, and Sandal, producing red-background rows).
    mnist_seed  = 42
    fmnist_seed = 44

    for ds, cls_names, ds_images, ds_labels, vis_seed in [
        ("mnist",  MNIST_CLASSES,  mnist_images, mnist_labels, mnist_seed),
        ("fmnist", FMNIST_CLASSES, None,         None,         fmnist_seed),
    ]:
        csv_path = EXP_DIR / f"{ds}_two_stage_results.csv"
        if not csv_path.exists():
            print(f"  (Skipping {ds} grid: {csv_path.name} not found)")
            continue
        if ds == "fmnist":
            print("Loading Fashion-MNIST ...")
            fmnist_ds  = torchvision.datasets.FashionMNIST(
                root=str(DATA_DIR), train=True, download=True)
            ds_images  = fmnist_ds.data.float().view(-1, 784) / 255.0
            ds_labels  = fmnist_ds.targets
        fig_name = "MNIST" if ds == "mnist" else "Fashion-MNIST"
        print(f"Figure 5/6: {fig_name} 4-column image grid (seed={vis_seed}) ...")
        fig_dataset_image_grid(ds, cls_names, 28, seed=vis_seed,
                               images=ds_images, labels=ds_labels)

    # ── Centered CIFAR grids ─────────────────────────────────────────────────
    centered_csv = EXP_DIR / "cifar_centered_two_stage_results.csv"
    if centered_csv.exists():
        print("Figure 7: CIFAR centered+norm grid — N=100 (perfect retrieval) ...")
        X_c100, _ = sample_class_balanced((cifar_gray, cifar_labels), 100, seed=42)
        fig_cifar_centered_image_grid(N=100, seed=42, X_raw=X_c100)

        print("Figure 8: CIFAR centered+norm grid — N=500 (attack regime) ...")
        X_c500, _ = sample_class_balanced((cifar_gray, cifar_labels), 500, seed=45)
        fig_cifar_centered_image_grid(N=500, seed=45, X_raw=X_c500)
    else:
        print("  (Skipping Figures 7-8: run grayscale_cifar_two_stage_centered.py first)")

    print("\nDone. All figures saved to figures/")


if __name__ == "__main__":
    main()
