"""
Beta calibration diagnostic for the continuous Hopfield network.

Motivation: Phase 1 Check 4 showed retrieval accuracy collapsing at N=1000/5000
under sigma=0.2 Gaussian noise.  This script determines whether the bottleneck
is the beta hyperparameter (fixable by tuning) or intrinsic pattern similarity
in MNIST (a fundamental capacity constraint).

Run from hopfield_thesis/ as:
    python -m experiments.phase1_beta_diagnostic
"""

import sys
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
from hopfield.corruption import add_gaussian_noise
from hopfield.metrics import mse, cosine_similarity, retrieval_accuracy

# ── config ────────────────────────────────────────────────────────────────────
SEED      = 42
N_STORE   = 1000
N_PROBE   = 50
SIGMA     = 0.2
BETAS     = [1, 2, 4, 8, 16, 32, 64, 128]
DATA_DIR  = ROOT / "data"
FIG_DIR   = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_mnist_train() -> tuple[torch.Tensor, torch.Tensor]:
    ds = torchvision.datasets.MNIST(
        root=str(DATA_DIR), train=True, download=True, transform=T.ToTensor()
    )
    images = ds.data.float() / 255.0
    images = images.view(-1, 784)
    return images, ds.targets


# ─────────────────────────────────────────────────────────────────────────────
# Pattern similarity baseline (independent of beta)
# ─────────────────────────────────────────────────────────────────────────────

def pattern_similarity_stats(X: torch.Tensor) -> dict:
    """
    Compute off-diagonal pairwise cosine similarity for all N columns of X.

    Args:
        X: shape (d, N)

    Returns:
        dict with keys: mean, median, p95, max, off_diag_flat (tensor)
    """
    # Normalize columns to unit norm
    norms = X.norm(dim=0, keepdim=True).clamp(min=1e-8)   # (1, N)
    X_n = X / norms                                         # (d, N)

    # Full cosine similarity matrix (N, N)
    C = X_n.T @ X_n                                        # (N, N)

    # Extract strictly off-diagonal elements
    N = X.shape[1]
    mask = ~torch.eye(N, dtype=torch.bool)
    off_diag = C[mask]                                      # N*(N-1) elements

    return {
        "mean":   float(off_diag.mean()),
        "median": float(off_diag.median()),
        "p95":    float(torch.quantile(off_diag.float(), 0.95)),
        "max":    float(off_diag.max()),
        "off_diag_flat": off_diag,
    }


def save_similarity_histogram(off_diag: torch.Tensor) -> None:
    vals = off_diag.cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=100, color="steelblue", edgecolor="none", alpha=0.85)
    ax.axvline(float(np.mean(vals)), color="crimson",  linestyle="--", label=f"mean={np.mean(vals):.3f}")
    ax.axvline(float(np.median(vals)), color="orange", linestyle=":",  label=f"median={np.median(vals):.3f}")
    ax.set_xlabel("Pairwise cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title(f"Off-diagonal pairwise cosine similarity — N={len(off_diag) // (int(len(off_diag)**0.5)) + 1} patterns")
    ax.legend()
    plt.tight_layout()
    fig.savefig(str(FIG_DIR / "05b_pattern_similarity.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Beta sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_beta_sweep(
    X: torch.Tensor,
    stored: torch.Tensor,
    probe_indices: list[int],
    noisy_probes: list[torch.Tensor],
) -> dict[float, dict]:
    """
    For each beta in BETAS, retrieve all noisy probes and compute metrics.

    Returns:
        dict mapping beta -> {"mse": float, "accuracy": float, "cosine": float}
    """
    results = {}
    for beta in BETAS:
        hop = ContinuousHopfield(X, beta=float(beta))
        batch_mse  = []
        batch_acc  = []
        batch_cos  = []
        for j, (noisy, true_idx) in enumerate(zip(noisy_probes, probe_indices)):
            retrieved = hop.retrieve(noisy, steps=1)
            original  = stored[true_idx]
            batch_mse.append(mse(retrieved, original))
            batch_acc.append(float(retrieval_accuracy(retrieved, X, true_idx)))
            batch_cos.append(cosine_similarity(retrieved, original))
        results[beta] = {
            "mse":      float(np.mean(batch_mse)),
            "accuracy": float(np.mean(batch_acc)),
            "cosine":   float(np.mean(batch_cos)),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def save_beta_sweep_figure(results: dict) -> None:
    betas    = [float(b) for b in BETAS]
    mses     = [results[b]["mse"]      for b in BETAS]
    accs     = [results[b]["accuracy"] for b in BETAS]
    cosines  = [results[b]["cosine"]   for b in BETAS]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].semilogx(betas, mses, "o-", color="steelblue", base=2)
    axes[0].set_xlabel("beta (log2 scale)")
    axes[0].set_ylabel("Mean MSE")
    axes[0].set_title("MSE vs beta")
    axes[0].set_xticks(betas)
    axes[0].set_xticklabels([str(b) for b in BETAS], rotation=45, fontsize=8)

    axes[1].semilogx(betas, accs, "s-", color="darkorange", base=2)
    axes[1].set_xlabel("beta (log2 scale)")
    axes[1].set_ylabel("Retrieval accuracy")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_title("Accuracy vs beta")
    axes[1].set_xticks(betas)
    axes[1].set_xticklabels([str(b) for b in BETAS], rotation=45, fontsize=8)
    # Mark β=8 baseline
    axes[1].axvline(8, color="gray", linestyle=":", alpha=0.7, label="β=8 baseline")
    axes[1].legend(fontsize=8)

    axes[2].semilogx(betas, cosines, "^-", color="seagreen", base=2)
    axes[2].set_xlabel("beta (log2 scale)")
    axes[2].set_ylabel("Mean cosine similarity")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_title("Cosine similarity vs beta")
    axes[2].set_xticks(betas)
    axes[2].set_xticklabels([str(b) for b in BETAS], rotation=45, fontsize=8)

    fig.suptitle(
        f"Beta sweep: N={N_STORE} patterns, sigma={SIGMA}, {N_PROBE} probes (seed={SEED})",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(str(FIG_DIR / "05_beta_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Interpretation
# ─────────────────────────────────────────────────────────────────────────────

def interpret(results: dict, sim_stats: dict) -> None:
    best_beta = max(BETAS, key=lambda b: results[b]["accuracy"])
    best_acc  = results[best_beta]["accuracy"]
    base_acc  = results[8]["accuracy"]
    max_acc   = best_acc

    print()
    print("=== INTERPRETATION ===")
    print(f"Best beta: {best_beta}  (accuracy = {best_acc:.3f})")
    print(f"beta=8 baseline: accuracy = {base_acc:.3f}")
    print(
        f"Pattern similarity (off-diagonal): "
        f"mean={sim_stats['mean']:.3f}, "
        f"median={sim_stats['median']:.3f}, "
        f"95th={sim_stats['p95']:.3f}, "
        f"max={sim_stats['max']:.3f}"
    )
    print()
    print("Diagnosis:")
    if max_acc > 0.7:
        print(f"  beta was the bottleneck -- recalibrate to beta={best_beta} for later phases.")
    elif max_acc < 0.5:
        print(
            "  Pattern similarity is the dominant bottleneck. "
            "beta cannot fix overlapping patterns."
        )
    else:
        print(
            "  Mixed regime -- both beta tuning and pattern selection matter."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 62)
    print("Beta Calibration Diagnostic -- Continuous Hopfield on MNIST")
    print("=" * 62)

    # Load data
    print(f"\nLoading MNIST ...")
    images, _ = load_mnist_train()

    # Sample N_STORE patterns
    rng_store = torch.Generator()
    rng_store.manual_seed(SEED)
    perm    = torch.randperm(images.shape[0], generator=rng_store)[:N_STORE]
    stored  = images[perm]          # (N_STORE, 784)
    X       = stored.T.contiguous() # (784, N_STORE)
    print(f"  Stored {N_STORE} patterns.  X shape: {X.shape}")

    # Select probe indices from the 1000 stored patterns
    rng_probe = torch.Generator()
    rng_probe.manual_seed(SEED)
    probe_perm    = torch.randperm(N_STORE, generator=rng_probe)[:N_PROBE]
    probe_indices = probe_perm.tolist()
    print(f"  Probe indices (first 5): {probe_indices[:5]}")

    # Generate noisy probes (fixed corruption, same across all beta)
    noisy_probes = [
        add_gaussian_noise(stored[idx], sigma=SIGMA, seed=SEED + j)
        for j, idx in enumerate(probe_indices)
    ]
    print(f"  {N_PROBE} noisy probes prepared (sigma={SIGMA})")

    # Pattern similarity baseline
    print(f"\nComputing pairwise cosine similarity for {N_STORE} patterns ...")
    sim_stats = pattern_similarity_stats(X)
    print(
        f"  Off-diagonal cosine sim: "
        f"mean={sim_stats['mean']:.4f}, "
        f"median={sim_stats['median']:.4f}, "
        f"95th={sim_stats['p95']:.4f}, "
        f"max={sim_stats['max']:.4f}"
    )
    save_similarity_histogram(sim_stats["off_diag_flat"])
    print(f"  Saved: figures/05b_pattern_similarity.png")

    # Beta sweep
    print(f"\nRunning beta sweep: {BETAS} ...")
    results = run_beta_sweep(X, stored, probe_indices, noisy_probes)

    # Print table
    print()
    print(f"{'beta':>8}  {'mean MSE':>10}  {'accuracy':>10}  {'mean cosine':>13}")
    print("-" * 48)
    for beta in BETAS:
        r = results[beta]
        marker = "  <-- baseline" if beta == 8 else ""
        print(f"{beta:>8}  {r['mse']:>10.4f}  {r['accuracy']:>10.3f}  {r['cosine']:>13.4f}{marker}")

    # Save figure
    save_beta_sweep_figure(results)
    print(f"\nSaved: figures/05_beta_sweep.png")

    # Interpretation
    interpret(results, sim_stats)


if __name__ == "__main__":
    main()
