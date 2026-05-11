"""
Phase 2 stability verification: re-run the Phase 2 grid across 5 seeds to
quantify variability in headline retrieval accuracy numbers.

Only the representative magnitudes are swept (sigma=0.20 / rate=0.20 / occlusion),
keeping runtime equivalent to a single Phase 2 run.

Each seed controls pattern selection, probe selection, and noise sampling so that
seed=42 exactly reproduces the Phase 2 single-seed results.

Run from hopfield_thesis/ as:
    python -m experiments.phase2_stability
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
SEEDS      = [42, 43, 44, 45, 46]
BETA       = 8.0
N_VALUES   = [10, 50, 100, 500, 1000]
STRATEGIES = ["random", "class_balanced"]
N_PROBE    = 50

# Representative magnitudes only (the Phase 2 summary magnitudes)
NOISE_CONDITIONS = [
    ("gaussian",  0.20),
    ("flip",      0.20),
    ("occlusion", 0.50),
]

UNSTABLE_THRESHOLD = 0.10   # std(accuracy) above this → unstable flag

DATA_DIR = ROOT / "data"
FIG_DIR  = ROOT / "figures"
EXP_DIR  = ROOT / "experiments"
FIG_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_mnist_train() -> tuple[torch.Tensor, torch.Tensor]:
    ds = torchvision.datasets.MNIST(
        root=str(DATA_DIR), train=True, download=True, transform=T.ToTensor()
    )
    images = ds.data.float() / 255.0
    images = images.view(-1, 784)
    return images, ds.targets


# ─────────────────────────────────────────────────────────────────────────────
# Single-cell retrieval
# ─────────────────────────────────────────────────────────────────────────────

def run_cell(
    images: torch.Tensor,
    labels: torch.Tensor,
    N: int,
    n_idx: int,
    s_idx: int,
    strategy: str,
    noise_type: str,
    magnitude: float,
    seed: int,
) -> dict:
    """
    Run one (N, strategy, noise_type, magnitude) cell for a given seed.

    Seeding convention (matches Phase 2 exactly when seed=42):
      - pattern selection : seed + n_idx
      - probe selection   : seed * 1000 + n_idx * 100 + s_idx * 10
      - noise for probe j : seed + j
    """
    dataset = (images, labels)

    if strategy == "random":
        X, _ = sample_random(dataset, N, seed=seed + n_idx)
    else:
        X, _ = sample_class_balanced(dataset, N, seed=seed + n_idx)

    stored = X.T.contiguous()   # (N, 784)
    hop = ContinuousHopfield(X, beta=BETA)

    n_probes = min(N_PROBE, N)
    probe_seed = seed * 1000 + n_idx * 100 + s_idx * 10
    rng_probe = torch.Generator()
    rng_probe.manual_seed(probe_seed)
    probe_indices = torch.randperm(N, generator=rng_probe)[:n_probes].tolist()

    batch_mse = []
    batch_acc = []
    batch_cos = []

    for j, true_idx in enumerate(probe_indices):
        original = stored[true_idx]

        if noise_type == "gaussian":
            corrupted = add_gaussian_noise(original, sigma=magnitude,  seed=seed + j)
        elif noise_type == "flip":
            corrupted = flip_pixels(original, flip_rate=magnitude, seed=seed + j)
        else:
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
# Phase 2 comparison baseline
# ─────────────────────────────────────────────────────────────────────────────

def load_phase2_comparison() -> dict:
    """
    Read the Phase 2 single-seed CSV and return accuracy for the three
    representative (noise_type, magnitude) conditions.
    """
    csv_path = EXP_DIR / "phase2_results.csv"
    if not csv_path.exists():
        return {}

    rep_keys = {("gaussian", 0.20), ("flip", 0.20), ("occlusion", 0.50)}
    comparison: dict = {}

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            nt  = row["noise_type"]
            mag = float(row["magnitude"])
            if (nt, mag) in rep_keys:
                key = (int(row["N"]), row["strategy"], nt, mag)
                comparison[key] = float(row["accuracy"])

    return comparison


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(per_seed: dict) -> dict:
    """
    Given per_seed[(seed, N, strategy, nt, mag)] = {mse, accuracy, cosine},
    return agg[(N, strategy, nt, mag)] = {mse_mean, mse_std, accuracy_mean,
                                           accuracy_std, cosine_mean, cosine_std}.
    """
    from collections import defaultdict
    buckets: dict = defaultdict(lambda: {"mse": [], "accuracy": [], "cosine": []})

    for (seed, N, strategy, nt, mag), m in per_seed.items():
        k = (N, strategy, nt, mag)
        buckets[k]["mse"].append(m["mse"])
        buckets[k]["accuracy"].append(m["accuracy"])
        buckets[k]["cosine"].append(m["cosine"])

    agg = {}
    for k, vals in buckets.items():
        agg[k] = {
            "mse_mean":      float(np.mean(vals["mse"])),
            "mse_std":       float(np.std(vals["mse"],  ddof=1)),
            "accuracy_mean": float(np.mean(vals["accuracy"])),
            "accuracy_std":  float(np.std(vals["accuracy"], ddof=1)),
            "cosine_mean":   float(np.mean(vals["cosine"])),
            "cosine_std":    float(np.std(vals["cosine"], ddof=1)),
            "n_seeds":       len(vals["mse"]),
            "unstable_flag": float(np.std(vals["accuracy"], ddof=1)) > UNSTABLE_THRESHOLD,
        }
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def save_stability_figure(agg: dict, fig_path: str) -> None:
    noise_labels = {
        ("gaussian",  0.20): "Gaussian  (sigma=0.20)",
        ("flip",      0.20): "Pixel-flip  (rate=0.20)",
        ("occlusion", 0.50): "Half-occlusion",
    }
    colors = {"random": "steelblue", "class_balanced": "darkorange"}
    slabels = {"random": "Random", "class_balanced": "Class-balanced"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax_idx, (nt, mag) in enumerate(NOISE_CONDITIONS):
        ax = axes[ax_idx]

        for strategy in STRATEGIES:
            means = np.array([agg[(N, strategy, nt, mag)]["accuracy_mean"] for N in N_VALUES])
            stds  = np.array([agg[(N, strategy, nt, mag)]["accuracy_std"]  for N in N_VALUES])

            ax.plot(N_VALUES, means, "o-", color=colors[strategy],
                    label=slabels[strategy], lw=1.8, ms=5, zorder=3)
            ax.errorbar(N_VALUES, means, yerr=stds, fmt="none",
                        ecolor=colors[strategy], capsize=5, elinewidth=1.5, zorder=2)
            ax.fill_between(N_VALUES,
                            np.clip(means - stds, 0, 1),
                            np.clip(means + stds, 0, 1),
                            color=colors[strategy], alpha=0.12, zorder=1)

        ax.set_xscale("log")
        ax.set_xticks(N_VALUES)
        ax.set_xticklabels([str(n) for n in N_VALUES], fontsize=8)
        ax.set_xlabel("N (stored patterns)", fontsize=9)
        if ax_idx == 0:
            ax.set_ylabel("Retrieval accuracy", fontsize=9)
        ax.set_ylim(-0.05, 1.10)
        ax.set_title(noise_labels[(nt, mag)], fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.axhline(0.5, color="gray", linestyle=":", lw=1, alpha=0.5)

    fig.suptitle(
        "Phase 2 Stability: Retrieval accuracy mean +/- std  (5 seeds: 42-46)",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CSV output
# ─────────────────────────────────────────────────────────────────────────────

def save_per_seed_csv(per_seed: dict) -> Path:
    path = EXP_DIR / "phase2_stability_results.csv"
    fields = ["seed", "N", "strategy", "noise_type", "magnitude",
              "mse", "accuracy", "cosine"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (seed, N, strategy, nt, mag), m in sorted(per_seed.items()):
            w.writerow({"seed": seed, "N": N, "strategy": strategy,
                        "noise_type": nt, "magnitude": mag, **m})
    return path


def save_summary_csv(agg: dict) -> Path:
    path = EXP_DIR / "phase2_stability_summary.csv"
    fields = ["N", "strategy", "noise_type", "magnitude",
              "mse_mean", "mse_std", "accuracy_mean", "accuracy_std",
              "cosine_mean", "cosine_std", "n_seeds", "unstable_flag"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (N, strategy, nt, mag), v in sorted(agg.items()):
            w.writerow({"N": N, "strategy": strategy,
                        "noise_type": nt, "magnitude": mag, **v})
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Stdout summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(agg: dict, p2: dict) -> None:
    strategy_labels = {"random": "Random", "class_balanced": "Class-balanced"}

    # ── per-noise mini tables ─────────────────────────────────────────────────
    for nt, mag in NOISE_CONDITIONS:
        label = {
            ("gaussian",  0.20): "Gaussian (sigma=0.20)",
            ("flip",      0.20): "Pixel-flip (rate=0.20)",
            ("occlusion", 0.50): "Half-occlusion",
        }[(nt, mag)]

        print(f"\n--- {label} ---")
        col_heads = "  ".join(f"{strategy_labels[s]:>20}" for s in STRATEGIES)
        print(f"{'N':>8}  {col_heads}")
        print("-" * (8 + 2 + len(col_heads) + 2))
        for N in N_VALUES:
            row = f"{N:>8}  "
            for strategy in STRATEGIES:
                v = agg[(N, strategy, nt, mag)]
                row += f"{v['accuracy_mean']:>7.3f} +/- {v['accuracy_std']:>5.3f}  "
            print(row)

    # ── unstable cells ────────────────────────────────────────────────────────
    unstable = [
        (N, strategy, nt, mag)
        for (N, strategy, nt, mag), v in agg.items()
        if v["unstable_flag"]
    ]

    # ── comparison to Phase 2 ─────────────────────────────────────────────────
    print()
    print("=== STABILITY REPORT ===")
    if unstable:
        print(f"Cells with std(accuracy) > {UNSTABLE_THRESHOLD}: {len(unstable)} found")
        for cell in sorted(unstable):
            N, strategy, nt, mag = cell
            print(f"  N={N}, {strategy}, {nt}, mag={mag}")
    else:
        print(f"Cells with std(accuracy) > {UNSTABLE_THRESHOLD}: none")

    # Comparison
    print()
    print("Comparison to single-seed Phase 2 results:")
    if not p2:
        print("  (phase2_results.csv not found — skipping comparison)")
    else:
        n_within = 0
        n_total  = 0
        fmt_nt = {"gaussian": "gauss", "flip": "flip", "occlusion": "occ"}
        for N in N_VALUES:
            for strategy in STRATEGIES:
                for nt, mag in NOISE_CONDITIONS:
                    key = (N, strategy, nt, mag)
                    if key not in p2:
                        continue
                    p2_val  = p2[key]
                    v       = agg[key]
                    mu      = v["accuracy_mean"]
                    sigma   = v["accuracy_std"]
                    within  = abs(p2_val - mu) <= sigma
                    marker  = "OK" if within else "OUTSIDE"
                    n_within += within
                    n_total  += 1
                    print(
                        f"  N={N:>5} {strategy:<16} {fmt_nt[nt]:<7}"
                        f"  p2={p2_val:.3f}  mean+/-std={mu:.3f}+/-{sigma:.3f}"
                        f"  [{marker}]"
                    )
        print(f"\n  {n_within}/{n_total} original Phase 2 values within 1 std of the multi-seed mean.")

    # Verdict
    n_unstable = len(unstable)
    all_within = (p2 and n_within == n_total)

    print()
    print("Overall verdict:")
    if n_unstable == 0 and all_within:
        print("  Phase 2 results are stable. Proceed to Phase 3.")
    elif 1 <= n_unstable <= 3:
        print(
            f"  Mostly stable ({n_unstable} unstable cell(s)). "
            "Note unstable cells in thesis but Phase 3 can proceed."
        )
    else:
        print(
            f"  Significant instability ({n_unstable} unstable cells). "
            "Consider raising probe count from 50 before Phase 3."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()

    print("=" * 62)
    print("Phase 2 Stability Verification")
    print(f"Seeds: {SEEDS}   Beta: {BETA}   N_probe: {N_PROBE}")
    print("=" * 62)

    print("\nLoading MNIST ...")
    images, labels = load_mnist_train()

    per_seed: dict = {}
    total_retrievals = 0

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        for n_idx, N in enumerate(N_VALUES):
            for s_idx, strategy in enumerate(STRATEGIES):
                for nt, mag in NOISE_CONDITIONS:
                    m = run_cell(images, labels, N, n_idx, s_idx,
                                 strategy, nt, mag, seed)
                    per_seed[(seed, N, strategy, nt, mag)] = m
                    total_retrievals += min(N_PROBE, N)
            # Progress line per N within seed
            g = per_seed[(seed, N, "random",        "gaussian",  0.20)]["accuracy"]
            c = per_seed[(seed, N, "class_balanced", "gaussian",  0.20)]["accuracy"]
            print(f"  N={N:>5}  gauss acc: random={g:.3f}  class_bal={c:.3f}")

    # Aggregate across seeds
    agg = aggregate(per_seed)

    # Load Phase 2 single-seed comparison
    p2 = load_phase2_comparison()

    # Save CSVs
    path_raw = save_per_seed_csv(per_seed)
    path_agg = save_summary_csv(agg)
    print(f"\nSaved per-seed CSV:   {path_raw}  ({len(per_seed)} rows)")
    print(f"Saved summary CSV:    {path_agg}  ({len(agg)} rows)")

    # Figure
    fig_path = str(FIG_DIR / "phase2_stability.png")
    save_stability_figure(agg, fig_path)
    print(f"Saved figure:         {fig_path}")

    # Summary
    elapsed = time.time() - t_start
    print(f"\nTotal retrievals: {total_retrievals}   Wall-clock time: {elapsed:.1f}s")
    print_summary(agg, p2)


if __name__ == "__main__":
    main()
