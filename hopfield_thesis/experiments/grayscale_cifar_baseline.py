"""
Grayscale CIFAR-10 baseline characterization.

Capacity-only experiment — NO adversarial attacks.
Converts 32×32 RGB CIFAR-10 images to 1024-dim grayscale vectors and measures
clean retrieval baseline failure rate vs N for the continuous Hopfield network.

Run: python -m experiments.grayscale_cifar_baseline
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import torch
import torchvision
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hopfield.network  import ContinuousHopfield
from hopfield.metrics  import retrieval_accuracy
from hopfield.sampling import sample_class_balanced

# ── config ─────────────────────────────────────────────────────────────────────
SEEDS   = [42, 43, 44, 45, 46]
BETA    = 8.0
N_PROBE = 50
N_CIFAR = [10, 20, 30, 50, 100]

DATA_DIR = ROOT / "data"
EXP_DIR  = ROOT / "experiments"
FIG_DIR  = ROOT / "figures"
OUT_DIR  = ROOT / "excel_exports"
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar10_gray() -> tuple[torch.Tensor, torch.Tensor]:
    """Load CIFAR-10, convert to grayscale 1024-dim, normalised to [0,1]."""
    ds = torchvision.datasets.CIFAR10(
        root=str(DATA_DIR), train=True, download=True,
    )
    data = ds.data.astype(np.float32) / 255.0      # (50000, 32, 32, 3)
    # Standard luminance weights
    gray = (0.2989 * data[:, :, :, 0]
            + 0.5870 * data[:, :, :, 1]
            + 0.1140 * data[:, :, :, 2])            # (50000, 32, 32)
    flat = gray.reshape(-1, 1024)                   # (50000, 1024)
    images  = torch.tensor(flat,     dtype=torch.float32)
    targets = torch.tensor(ds.targets, dtype=torch.long)
    return images, targets


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise cosine helper
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_cosine_stats(X: torch.Tensor) -> tuple[float, float]:
    """Return (mean, max) of off-diagonal pairwise cosine similarities."""
    Xn  = X.numpy()
    nrm = np.linalg.norm(Xn, axis=0, keepdims=True)
    Xnn = Xn / (nrm + 1e-8)
    C   = Xnn.T @ Xnn      # (N, N)
    np.fill_diagonal(C, -1.0)
    return float(C.mean()), float(C.max())


# ─────────────────────────────────────────────────────────────────────────────
# MNIST reference — read from phase3_diag_b_baseline_corrected.csv
# ─────────────────────────────────────────────────────────────────────────────

def load_mnist_baseline_ref() -> dict[int, tuple[float, float]]:
    """Return {N: (mean_baseline_failure, std)} for MNIST class-balanced."""
    path = EXP_DIR / "phase3_diag_b_baseline_corrected.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    ref: dict[int, list[float]] = {}
    for r in rows:
        if r["strategy"] == "class_balanced":
            n = int(r["N"])
            ref.setdefault(n, []).append(float(r["baseline_failure_rate"]))
    return {n: (float(np.mean(v)), float(np.std(v, ddof=1) if len(v) > 1 else 0.0))
            for n, v in ref.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    images: torch.Tensor, labels: torch.Tensor,
) -> list[dict]:
    rows: list[dict] = []
    for N in N_CIFAR:
        for seed in SEEDS:
            X, _ = sample_class_balanced((images, labels), N, seed=seed)
            stored  = X.T.contiguous()   # (N, 1024)
            hop     = ContinuousHopfield(X, beta=BETA)
            n_probe = min(N_PROBE, N)

            rng = torch.Generator()
            rng.manual_seed(seed * 1000 + N)
            probe_indices = torch.randperm(N, generator=rng)[:n_probe].tolist()

            failures = 0
            for true_idx in probe_indices:
                ret  = hop.retrieve(stored[true_idx], steps=1)
                fail = not retrieval_accuracy(ret, X, true_idx)
                if fail:
                    failures += 1

            bl_fail = failures / n_probe
            cos_mean, cos_max = pairwise_cosine_stats(X)

            rows.append({
                "seed":                  seed,
                "N":                     N,
                "baseline_failure_rate": round(bl_fail, 4),
                "mean_pairwise_cosine":  round(cos_mean, 5),
                "max_pairwise_cosine":   round(cos_max,  5),
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Save CSVs
# ─────────────────────────────────────────────────────────────────────────────

def save_results_csv(rows: list[dict]) -> None:
    fields = ["seed", "N", "baseline_failure_rate",
              "mean_pairwise_cosine", "max_pairwise_cosine"]
    path = EXP_DIR / "grayscale_cifar_baseline_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})
    print(f"  Saved: {path.name}  ({len(rows)} rows)")


def save_excel_csv(rows: list[dict]) -> None:
    fields = ["N",
              "baseline_failure_mean", "baseline_failure_std",
              "mean_pairwise_cosine_mean", "mean_pairwise_cosine_std"]
    path   = OUT_DIR / "fig_grayscale_cifar_baseline.csv"
    out_rows = []
    for N in N_CIFAR:
        sub = [r for r in rows if r["N"] == N]
        bl_vals  = [r["baseline_failure_rate"] for r in sub]
        cos_vals = [r["mean_pairwise_cosine"]  for r in sub]
        out_rows.append({
            "N":                          N,
            "baseline_failure_mean":      round(float(np.mean(bl_vals)), 4),
            "baseline_failure_std":       round(float(np.std(bl_vals, ddof=1)), 4),
            "mean_pairwise_cosine_mean":  round(float(np.mean(cos_vals)), 5),
            "mean_pairwise_cosine_std":   round(float(np.std(cos_vals, ddof=1)), 5),
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)
    print(f"  Saved: {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def save_figure(rows: list[dict], mnist_ref: dict[int, tuple[float, float]]) -> None:
    cifar_agg: dict[int, dict[str, list]] = {N: {"bl": [], "cos": []} for N in N_CIFAR}
    for r in rows:
        cifar_agg[r["N"]]["bl"].append(r["baseline_failure_rate"])
        cifar_agg[r["N"]]["cos"].append(r["mean_pairwise_cosine"])

    bl_means  = [np.mean(cifar_agg[N]["bl"])             for N in N_CIFAR]
    bl_stds   = [np.std(cifar_agg[N]["bl"], ddof=1)      for N in N_CIFAR]
    cos_means = [np.mean(cifar_agg[N]["cos"])             for N in N_CIFAR]
    cos_stds  = [np.std(cifar_agg[N]["cos"], ddof=1)     for N in N_CIFAR]

    # MNIST reference points (only at N values present in both datasets)
    mnist_Ns  = sorted(n for n in mnist_ref if n in N_CIFAR)
    mnist_bl  = [mnist_ref[n][0] for n in mnist_Ns]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: baseline failure rate vs N
    ax = axes[0]
    ax.errorbar(N_CIFAR, bl_means, yerr=bl_stds, fmt="o-",
                color="darkorange", lw=2, ms=6, capsize=5,
                label="CIFAR-10 grayscale (5-seed mean ± std)")
    if mnist_Ns:
        ax.plot(mnist_Ns, mnist_bl, "s--", color="steelblue", lw=1.5, ms=7,
                label="MNIST class-balanced reference (5-seed mean)")
    ax.set_xscale("log")
    ax.set_xticks(N_CIFAR)
    ax.set_xticklabels([str(n) for n in N_CIFAR])
    ax.set_xlabel("N (stored patterns, log scale)", fontsize=10)
    ax.set_ylabel("Baseline failure rate", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Clean retrieval baseline failure rate vs N\n"
                 "Grayscale CIFAR-10 vs MNIST (class-balanced, β=8.0)", fontsize=9)
    ax.axhline(0.2, color="gray", linestyle=":", lw=1.2, alpha=0.6, label="20% threshold")
    ax.axhline(0.5, color="gray", linestyle="--", lw=1.2, alpha=0.6, label="50% threshold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    for n, m, s in zip(N_CIFAR, bl_means, bl_stds):
        ax.annotate(f"{m:.2f}", xy=(n, m), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8)

    # Right: mean pairwise cosine vs N
    ax = axes[1]
    ax.errorbar(N_CIFAR, cos_means, yerr=cos_stds, fmt="o-",
                color="darkorange", lw=2, ms=6, capsize=5,
                label="CIFAR-10 grayscale")
    ax.set_xscale("log")
    ax.set_xticks(N_CIFAR)
    ax.set_xticklabels([str(n) for n in N_CIFAR])
    ax.set_xlabel("N (stored patterns, log scale)", fontsize=10)
    ax.set_ylabel("Mean off-diagonal pairwise cosine", fontsize=10)
    ax.set_title("Pattern crowding: mean pairwise cosine vs N\n"
                 "Higher cosine = more similar patterns = harder retrieval", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    for n, m in zip(N_CIFAR, cos_means):
        ax.annotate(f"{m:.3f}", xy=(n, m), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=8)

    fig.suptitle("Grayscale CIFAR-10 — Continuous Hopfield Network Baseline Characterization\n"
                 "Class-balanced storage, 5 seeds × 50 probes, no noise or attacks",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = FIG_DIR / "grayscale_cifar_baseline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Stdout summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(rows: list[dict], mnist_ref: dict[int, tuple[float, float]]) -> None:
    print("\n=== GRAYSCALE CIFAR BASELINE CHARACTERIZATION ===\n")
    print(f"{'N':>6}  {'Baseline Failure':>22}  {'Mean Pairwise Cosine':>22}")
    print("-" * 56)
    agg: dict[int, dict] = {}
    for N in N_CIFAR:
        sub  = [r for r in rows if r["N"] == N]
        bl   = [r["baseline_failure_rate"] for r in sub]
        cos  = [r["mean_pairwise_cosine"]  for r in sub]
        bl_m, bl_s   = float(np.mean(bl)), float(np.std(bl, ddof=1))
        cos_m, cos_s = float(np.mean(cos)), float(np.std(cos, ddof=1))
        agg[N] = {"bl_m": bl_m, "bl_s": bl_s}
        print(f"{N:>6}  {bl_m:.3f} +/- {bl_s:.3f}          {cos_m:.4f} +/- {cos_s:.4f}")

    below20 = [N for N in N_CIFAR if agg[N]["bl_m"] < 0.20]
    below50 = [N for N in N_CIFAR if agg[N]["bl_m"] < 0.50]
    print(f"\nLargest N with baseline failure < 20%: "
          f"{max(below20) if below20 else 'none'}")
    print(f"Largest N with baseline failure < 50%: "
          f"{max(below50) if below50 else 'none'}")

    cifar_bl_100 = agg.get(100, {}).get("bl_m", float("nan"))
    print(f"\nComparison to MNIST and Fashion-MNIST at N=100:")
    print(f"  MNIST baseline failure:         9.2%")
    print(f"  Fashion-MNIST baseline failure: 80.0%")
    print(f"  Grayscale CIFAR baseline failure: {cifar_bl_100*100:.1f}%")

    # Verdict
    if cifar_bl_100 > 0.50:
        verdict = (
            f"Grayscale CIFAR baseline failure of {cifar_bl_100*100:.1f}% at N=100 "
            f"confirms the network operates outside its reliable retrieval regime on this "
            f"dataset at this scale, consistent with the Fashion-MNIST finding and the "
            f"broader pattern-crowding mechanism characterised in Phase 2."
        )
    elif cifar_bl_100 > 0.20:
        verdict = (
            f"Grayscale CIFAR baseline failure of {cifar_bl_100*100:.1f}% at N=100 "
            f"is substantially higher than MNIST (9.2%) but lower than Fashion-MNIST (80.0%), "
            f"placing CIFAR in an intermediate difficulty regime for this network architecture."
        )
    else:
        verdict = (
            f"Grayscale CIFAR baseline failure of {cifar_bl_100*100:.1f}% at N=100 "
            f"is comparable to MNIST (9.2%), suggesting grayscale CIFAR is within the "
            f"network's reliable retrieval regime at this capacity."
        )
    print(f"\nVerdict:\n  {verdict}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print("Grayscale CIFAR-10 Baseline Characterization")
    print("=" * 60)

    print("\nLoading CIFAR-10 (downloading if needed) ...")
    images, labels = load_cifar10_gray()
    print(f"  ok  {len(images)} samples, dim=1024, "
          f"value range [{images.min():.3f}, {images.max():.3f}]")

    print(f"\nRunning clean retrieval: N = {N_CIFAR}, seeds = {SEEDS} ...")
    rows = run_experiment(images, labels)

    print("\nSaving results ...")
    save_results_csv(rows)
    save_excel_csv(rows)

    mnist_ref = load_mnist_baseline_ref()
    print("Generating figure ...")
    save_figure(rows, mnist_ref)

    print_summary(rows, mnist_ref)
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
