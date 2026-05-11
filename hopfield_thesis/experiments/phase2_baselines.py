"""
Phase 2: Random-noise robustness baseline for the continuous Hopfield network.

Measures retrieval quality across a grid of:
  - N stored patterns  : {10, 50, 100, 500, 1000}
  - Selection strategy : {random, class_balanced}
  - Noise type         : Gaussian, pixel-flip, half-occlusion
  - Magnitude levels   : 5 levels per noise type (single level for occlusion)

All results saved to experiments/phase2_results.csv.
Five figures saved to figures/phase2_*.png.

Run from hopfield_thesis/ as:
    python -m experiments.phase2_baselines
"""

import csv
import sys
import time
import random
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hopfield.network import ContinuousHopfield
from hopfield.corruption import add_gaussian_noise, flip_pixels, mask_bottom_half
from hopfield.metrics import mse as metric_mse, cosine_similarity, retrieval_accuracy
from hopfield.sampling import sample_random, sample_class_balanced

# ── config ────────────────────────────────────────────────────────────────────
SEED             = 42
BETA             = 8.0
N_VALUES         = [10, 50, 100, 500, 1000]
STRATEGIES       = ["random", "class_balanced"]
GAUSSIAN_SIGMAS  = [0.05, 0.10, 0.20, 0.30, 0.50]
FLIP_RATES       = [0.05, 0.10, 0.20, 0.30, 0.50]
N_PROBE          = 50
# Representative magnitudes used for heatmap / summary
REP_MAG = {"gaussian": 0.20, "flip": 0.20, "occlusion": 0.50}

DATA_DIR = ROOT / "data"
FIG_DIR  = ROOT / "figures"
EXP_DIR  = ROOT / "experiments"
FIG_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_mnist_train() -> tuple[torch.Tensor, torch.Tensor]:
    ds = torchvision.datasets.MNIST(
        root=str(DATA_DIR), train=True, download=True, transform=T.ToTensor()
    )
    images = ds.data.float() / 255.0   # (60000, 28, 28)
    images = images.view(-1, 784)      # (60000, 784)
    return images, ds.targets


# ─────────────────────────────────────────────────────────────────────────────
# Core retrieval loop
# ─────────────────────────────────────────────────────────────────────────────

def run_retrievals(
    hop: ContinuousHopfield,
    X: torch.Tensor,
    stored: torch.Tensor,        # (N, 784) — X.T
    probe_indices: list[int],
    noise_type: str,
    magnitude: float,
) -> dict:
    """
    Corrupt each probe, retrieve, and return mean MSE / accuracy / cosine.

    Noise seed for probe j: SEED + j  (consistent across all cells so noise
    realisations are comparable between conditions).
    """
    batch_mse = []
    batch_acc = []
    batch_cos = []

    for j, true_idx in enumerate(probe_indices):
        original = stored[true_idx]   # (784,)

        if noise_type == "gaussian":
            corrupted = add_gaussian_noise(original, sigma=magnitude, seed=SEED + j)
        elif noise_type == "flip":
            corrupted = flip_pixels(original, flip_rate=magnitude, seed=SEED + j)
        else:   # occlusion
            corrupted = mask_bottom_half(original)

        retrieved = hop.retrieve(corrupted, steps=1)
        batch_mse.append(metric_mse(retrieved, original))
        batch_acc.append(float(retrieval_accuracy(retrieved, X, true_idx)))
        batch_cos.append(cosine_similarity(retrieved, original))

    return {
        "mse":      float(np.mean(batch_mse)),
        "accuracy": float(np.mean(batch_acc)),
        "cosine":   float(np.mean(batch_cos)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _show(ax: plt.Axes, vec: torch.Tensor, title: str = "") -> None:
    img = vec.detach().cpu().float().clamp(0, 1).view(28, 28).numpy()
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=6)
    ax.axis("off")


def _strategy_label(s: str) -> str:
    return "Random" if s == "random" else "Class-balanced"


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 & 2 — sweep grids (Gaussian / pixel-flip)
# ─────────────────────────────────────────────────────────────────────────────

def save_sweep_figure(
    results: dict,
    noise_type: str,
    magnitudes: list[float],
    x_label: str,
    fig_title: str,
    fig_path: str,
) -> None:
    """2x5 subplot grid: rows=strategy, cols=N, dual y-axis (accuracy + MSE)."""
    COLOR_ACC = "steelblue"
    COLOR_MSE = "firebrick"

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    legend_handles: list | None = None

    for s_idx, strategy in enumerate(STRATEGIES):
        for n_idx, N in enumerate(N_VALUES):
            ax = axes[s_idx, n_idx]

            accs = [results[(N, strategy, noise_type, m)]["accuracy"] for m in magnitudes]
            mses = [results[(N, strategy, noise_type, m)]["mse"]      for m in magnitudes]

            l1, = ax.plot(magnitudes, accs, "o-",  color=COLOR_ACC, lw=1.5, ms=4, label="Accuracy")
            ax.set_ylim(-0.05, 1.05)
            ax.tick_params(axis="y", labelcolor=COLOR_ACC, labelsize=7)

            ax2 = ax.twinx()
            l2, = ax2.plot(magnitudes, mses, "s--", color=COLOR_MSE, lw=1.5, ms=4, label="MSE")
            ax2.tick_params(axis="y", labelcolor=COLOR_MSE, labelsize=7)
            # Only expose right y-axis label on the rightmost column
            if n_idx < len(N_VALUES) - 1:
                ax2.set_yticks([])

            # Column titles (top row only)
            if s_idx == 0:
                ax.set_title(f"N={N}", fontsize=9)

            # X labels (bottom row only)
            if s_idx == 1:
                ax.set_xlabel(x_label, fontsize=8)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)

            # Left y-axis label (leftmost column only)
            if n_idx == 0:
                ax.set_ylabel(f"Accuracy\n({_strategy_label(strategy)})", fontsize=8, color=COLOR_ACC)

            if legend_handles is None:
                legend_handles = [l1, l2]

    fig.legend(
        legend_handles, ["Accuracy (left y)", "MSE (right y)"],
        loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(fig_title, fontsize=11, y=1.01)
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — occlusion bar chart
# ─────────────────────────────────────────────────────────────────────────────

def save_occlusion_figure(results: dict, fig_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    x = np.arange(len(N_VALUES))
    width = 0.55

    for s_idx, (strategy, ax) in enumerate(zip(STRATEGIES, axes)):
        accs = [results[(N, strategy, "occlusion", 0.50)]["accuracy"] for N in N_VALUES]
        mses = [results[(N, strategy, "occlusion", 0.50)]["mse"]      for N in N_VALUES]

        bars = ax.bar(x, accs, width, color="steelblue", alpha=0.82, edgecolor="white")

        for bar, m in zip(bars, mses):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"MSE\n{m:.3f}",
                ha="center", va="bottom", fontsize=7.5, color="#333333",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([str(n) for n in N_VALUES], fontsize=9)
        ax.set_xlabel("N (stored patterns)", fontsize=9)
        ax.set_ylabel("Retrieval accuracy", fontsize=9)
        ax.set_ylim(0, 1.25)
        ax.set_title(_strategy_label(strategy), fontsize=10)
        ax.axhline(0.5, color="gray", linestyle=":", lw=1, alpha=0.6)

    fig.suptitle("Half-occlusion retrieval (bottom 50% masked)", fontsize=11)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — summary heatmap
# ─────────────────────────────────────────────────────────────────────────────

def save_heatmap_figure(results: dict, fig_path: str) -> None:
    noise_configs = [
        ("gaussian",  0.20, "Gaussian  (sigma=0.20)"),
        ("flip",      0.20, "Pixel-flip  (rate=0.20)"),
        ("occlusion", 0.50, "Half-occlusion"),
    ]
    strategy_labels = [_strategy_label(s) for s in STRATEGIES]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    for ax_idx, (noise_type, rep_mag, title) in enumerate(noise_configs):
        ax = axes[ax_idx]
        data = np.zeros((len(N_VALUES), len(STRATEGIES)))
        for n_idx, N in enumerate(N_VALUES):
            for s_idx, strategy in enumerate(STRATEGIES):
                data[n_idx, s_idx] = results[(N, strategy, noise_type, rep_mag)]["accuracy"]

        im = ax.imshow(data, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(STRATEGIES)))
        ax.set_xticklabels(strategy_labels, fontsize=9)
        ax.set_yticks(range(len(N_VALUES)))
        ax.set_yticklabels([str(n) for n in N_VALUES], fontsize=9)
        if ax_idx == 0:
            ax.set_ylabel("N (stored patterns)", fontsize=9)
        ax.set_title(title, fontsize=9)

        for i in range(len(N_VALUES)):
            for j in range(len(STRATEGIES)):
                val = data[i, j]
                color = "white" if val < 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)

        plt.colorbar(im, ax=ax, label="Retrieval accuracy", fraction=0.046, pad=0.04)

    fig.suptitle(
        "Phase 2 Summary: Retrieval accuracy at representative magnitudes",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — qualitative examples (N=100 class-balanced)
# ─────────────────────────────────────────────────────────────────────────────

def save_qualitative_figure(qual: dict, fig_path: str) -> None:
    """
    3x6 logical grid (rows=noise type, cols=digits 0-5).
    Each logical cell is rendered as 3 stacked image panels:
        original | corrupted | retrieved
    Total actual subplot grid: 9 rows x 6 cols.
    """
    X      = qual["X"]       # (784, 100)
    stored = qual["stored"]  # (100, 784)
    hop    = qual["hop"]

    # class-balanced N=100: 10 per class, ordered class 0..9
    # => digit d lives at stored indices [d*10 .. d*10+9]; pick first of each
    digit_true_indices = [d * 10 for d in range(6)]   # digits 0-5

    noise_configs = [
        ("Gaussian  sigma=0.2",   "gaussian",  0.20),
        ("Pixel-flip  rate=0.2",  "flip",      0.20),
        ("Half-occlusion",        "occlusion", 0.50),
    ]
    row_sublabels = ["Original", "Corrupted", "Retrieved"]

    fig, axes = plt.subplots(9, 6, figsize=(12, 14))
    fig.patch.set_facecolor("white")

    for noise_idx, (noise_label, noise_type, magnitude) in enumerate(noise_configs):
        row_base = noise_idx * 3

        for col_idx, true_idx in enumerate(digit_true_indices):
            original = stored[true_idx]

            if noise_type == "gaussian":
                corrupted = add_gaussian_noise(original, sigma=magnitude,  seed=SEED + true_idx)
            elif noise_type == "flip":
                corrupted = flip_pixels(original, flip_rate=magnitude, seed=SEED + true_idx)
            else:
                corrupted = mask_bottom_half(original)

            retrieved = hop.retrieve(corrupted, steps=1)

            top_title = f"digit {col_idx}" if noise_idx == 0 else ""
            _show(axes[row_base + 0, col_idx], original,  top_title)
            _show(axes[row_base + 1, col_idx], corrupted, "")
            _show(axes[row_base + 2, col_idx], retrieved, "")

        # Row sub-labels (left of each image row)
        for sub_row, sub_lbl in enumerate(row_sublabels):
            axes[row_base + sub_row, 0].set_ylabel(sub_lbl, fontsize=7)

        # Noise-type label: annotate on leftmost col, middle image row
        axes[row_base + 1, 0].set_ylabel(
            f"{noise_label}\n{row_sublabels[1]}",
            fontsize=7, color="#555555",
        )

    # Horizontal separators between noise groups
    for divider_row in [3, 6]:
        for col_idx in range(6):
            axes[divider_row, col_idx].spines["top"].set_linewidth(2)
            axes[divider_row, col_idx].spines["top"].set_color("#aaaaaa")

    fig.suptitle(
        "Phase 2 Qualitative: N=100 class-balanced, beta=8.0\n"
        "Rows: Gaussian (top) | pixel-flip (middle) | occlusion (bottom)",
        fontsize=9, y=1.005,
    )
    plt.tight_layout(h_pad=0.15, w_pad=0.1)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(rows: list[dict]) -> Path:
    csv_path = EXP_DIR / "phase2_results.csv"
    fieldnames = ["N", "strategy", "noise_type", "magnitude", "mse", "accuracy", "cosine"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict, total_retrievals: int, elapsed: float) -> None:
    noise_configs = [
        ("gaussian",  "Gaussian (sigma=0.20)"),
        ("flip",      "Pixel-flip (rate=0.20)"),
        ("occlusion", "Half-occlusion"),
    ]
    strat_labels = [_strategy_label(s) for s in STRATEGIES]

    for noise_type, noise_label in noise_configs:
        rep_mag = REP_MAG[noise_type]
        print(f"\n--- {noise_label} ---")
        header = f"{'N':>8}  " + "  ".join(f"{l:>16}" for l in strat_labels)
        print(header)
        print("-" * len(header))
        for N in N_VALUES:
            row = f"{N:>8}  "
            for strategy in STRATEGIES:
                acc = results[(N, strategy, noise_type, rep_mag)]["accuracy"]
                row += f"{acc:>16.3f}  "
            print(row)

    total_cells = len(N_VALUES) * len(STRATEGIES) * (len(GAUSSIAN_SIGMAS) + len(FLIP_RATES) + 1)
    print()
    print("=== PHASE 2 SUMMARY ===")
    print(f"Total experimental cells: {total_cells}")
    print(f"Total retrievals performed: {total_retrievals}")
    print(f"Wall-clock time: {elapsed:.1f}s")
    print(f"Results CSV: experiments/phase2_results.csv")
    print(f"Figures: figures/phase2_*.png (5 files)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()

    print("=" * 62)
    print("Phase 2: Random-Noise Robustness Baseline")
    print("=" * 62)

    print("\nLoading MNIST ...")
    images, labels = load_mnist_train()
    dataset = (images, labels)
    print(f"  {images.shape[0]} training images, {images.shape[1]}-dim each")

    results:    dict      = {}
    csv_rows:   list[dict] = []
    qual_data:  dict | None = None
    total_retrievals = 0

    for n_idx, N in enumerate(N_VALUES):
        for s_idx, strategy in enumerate(STRATEGIES):
            print(f"\n[N={N:>5}, strategy={strategy}]")

            # Build storage
            if strategy == "random":
                X, _ = sample_random(dataset, N, seed=SEED + n_idx)
            else:
                X, _ = sample_class_balanced(dataset, N, seed=SEED + n_idx)

            stored = X.T.contiguous()   # (N, 784)
            hop = ContinuousHopfield(X, beta=BETA)

            # Select probe indices (deterministic per cell)
            n_probes = min(N_PROBE, N)
            probe_seed = SEED * 1000 + n_idx * 100 + s_idx * 10
            rng_probe = torch.Generator()
            rng_probe.manual_seed(probe_seed)
            probe_indices = torch.randperm(N, generator=rng_probe)[:n_probes].tolist()

            # ── Gaussian sweep ────────────────────────────────────────────
            for sigma in GAUSSIAN_SIGMAS:
                m = run_retrievals(hop, X, stored, probe_indices, "gaussian", sigma)
                results[(N, strategy, "gaussian", sigma)] = m
                csv_rows.append({"N": N, "strategy": strategy, "noise_type": "gaussian",
                                 "magnitude": sigma, **m})
                total_retrievals += n_probes

            # ── Pixel-flip sweep ─────────────────────────────────────────
            for rate in FLIP_RATES:
                m = run_retrievals(hop, X, stored, probe_indices, "flip", rate)
                results[(N, strategy, "flip", rate)] = m
                csv_rows.append({"N": N, "strategy": strategy, "noise_type": "flip",
                                 "magnitude": rate, **m})
                total_retrievals += n_probes

            # ── Half-occlusion ───────────────────────────────────────────
            m = run_retrievals(hop, X, stored, probe_indices, "occlusion", 0.50)
            results[(N, strategy, "occlusion", 0.50)] = m
            csv_rows.append({"N": N, "strategy": strategy, "noise_type": "occlusion",
                             "magnitude": 0.50, **m})
            total_retrievals += n_probes

            # Quick per-cell report at representative magnitudes
            g_acc = results[(N, strategy, "gaussian",  0.20)]["accuracy"]
            f_acc = results[(N, strategy, "flip",      0.20)]["accuracy"]
            o_acc = results[(N, strategy, "occlusion", 0.50)]["accuracy"]
            print(f"  acc@gauss0.2={g_acc:.3f}  acc@flip0.2={f_acc:.3f}  acc@occ={o_acc:.3f}")

            # Save qual data for N=100, class-balanced
            if N == 100 and strategy == "class_balanced":
                qual_data = {"X": X, "stored": stored, "hop": hop}

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = save_csv(csv_rows)
    print(f"\nSaved CSV: {csv_path}  ({len(csv_rows)} rows)")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures ...")

    save_sweep_figure(
        results, "gaussian", GAUSSIAN_SIGMAS,
        x_label="Gaussian sigma",
        fig_title="Phase 2: Gaussian noise — Accuracy (left) and MSE (right) vs sigma",
        fig_path=str(FIG_DIR / "phase2_gaussian.png"),
    )
    print("  Saved: phase2_gaussian.png")

    save_sweep_figure(
        results, "flip", FLIP_RATES,
        x_label="Pixel-flip rate",
        fig_title="Phase 2: Pixel-flip noise — Accuracy (left) and MSE (right) vs flip rate",
        fig_path=str(FIG_DIR / "phase2_pixelflip.png"),
    )
    print("  Saved: phase2_pixelflip.png")

    save_occlusion_figure(results, str(FIG_DIR / "phase2_occlusion.png"))
    print("  Saved: phase2_occlusion.png")

    save_heatmap_figure(results, str(FIG_DIR / "phase2_summary_heatmap.png"))
    print("  Saved: phase2_summary_heatmap.png")

    if qual_data is not None:
        save_qualitative_figure(qual_data, str(FIG_DIR / "phase2_qualitative.png"))
        print("  Saved: phase2_qualitative.png")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print_summary(results, total_retrievals, elapsed)


if __name__ == "__main__":
    main()
