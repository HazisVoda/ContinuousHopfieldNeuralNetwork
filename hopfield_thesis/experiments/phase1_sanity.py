"""
Phase 1 sanity checks for the continuous Hopfield network on MNIST.

Run from hopfield_thesis/ as:
    python -m experiments.phase1_sanity

Four checks:
    1. Clean retrieval          — query with stored pattern itself
    2. Half-occlusion retrieval — mask bottom 14 rows, then retrieve
    3. Gaussian noise robustness — sweep sigma, measure degradation
    4. Capacity stress test      — vary N, measure retrieval quality
"""

import sys
import os
import random
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── make imports work whether called as a module or a script ──────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hopfield.network import ContinuousHopfield
from hopfield.corruption import mask_bottom_half, add_gaussian_noise
from hopfield.metrics import mse, cosine_similarity, retrieval_accuracy

# ── global config ─────────────────────────────────────────────────────────────
SEED = 42
BETA = 8.0
DATA_DIR = ROOT / "data"
FIG_DIR  = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_mnist_train() -> tuple[torch.Tensor, torch.Tensor]:
    """Return (images, labels) — images shape (N, 784) in [0,1] float32."""
    ds = torchvision.datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=T.ToTensor(),
    )
    images = ds.data.float() / 255.0          # (60000, 28, 28)
    images = images.view(-1, 784)             # (60000, 784)
    labels = ds.targets                       # (60000,)
    return images, labels


def pick_one_per_class(
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, list[int]]:
    """
    Deterministically pick the FIRST example of each digit class 0-9.

    Returns:
        patterns: shape (10, 784)
        indices:  list of 10 dataset indices (for tracing)
    """
    patterns = []
    indices  = []
    for cls in range(10):
        idx = int((labels == cls).nonzero(as_tuple=True)[0][0].item())
        patterns.append(images[idx])
        indices.append(idx)
    return torch.stack(patterns, dim=0), indices  # (10, 784)


def patterns_to_X(patterns: torch.Tensor) -> torch.Tensor:
    """Convert (N, 784) row-matrix to (784, N) column-matrix for the network."""
    return patterns.T.contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _show(ax: plt.Axes, vec: torch.Tensor, title: str = "") -> None:
    img = vec.detach().cpu().float().clamp(0, 1).view(28, 28).numpy()
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check 1 — clean retrieval
# ─────────────────────────────────────────────────────────────────────────────

def check1_clean_retrieval(
    hop: ContinuousHopfield,
    patterns: torch.Tensor,
) -> list[float]:
    """Query each stored pattern with itself.  Return per-pattern MSE."""
    X = hop.X  # (784, 10)
    mses = []
    for i in range(patterns.shape[0]):
        query     = patterns[i]
        retrieved = hop.retrieve(query, steps=1)
        mses.append(mse(retrieved, query))

    # figure
    fig, axes = plt.subplots(2, 10, figsize=(14, 3.5))
    for i in range(10):
        query     = patterns[i]
        retrieved = hop.retrieve(query, steps=1)
        _show(axes[0, i], query,     f"orig {i}")
        _show(axes[1, i], retrieved, f"retr {i}\nMSE={mses[i]:.4f}")
    fig.suptitle("Sanity Check 1: Clean Retrieval", fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(str(FIG_DIR / "01_clean_retrieval.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return mses


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check 2 — half-occlusion retrieval
# ─────────────────────────────────────────────────────────────────────────────

def check2_half_occlusion(
    hop: ContinuousHopfield,
    patterns: torch.Tensor,
) -> tuple[list[float], list[bool]]:
    """Mask bottom half of each pattern and retrieve."""
    X = hop.X
    mses  = []
    accs  = []
    queries = []
    retrieveds = []

    for i in range(patterns.shape[0]):
        query     = mask_bottom_half(patterns[i])
        retrieved = hop.retrieve(query, steps=1)
        mses.append(mse(retrieved, patterns[i]))
        accs.append(retrieval_accuracy(retrieved, X, i))
        queries.append(query)
        retrieveds.append(retrieved)

    # figure — 3 cols per digit: original | masked | retrieved
    fig, axes = plt.subplots(3, 10, figsize=(14, 5))
    row_labels = ["Original", "Masked query", "Retrieved"]
    for i in range(10):
        _show(axes[0, i], patterns[i],   f"orig {i}")
        _show(axes[1, i], queries[i],    "masked")
        _show(axes[2, i], retrieveds[i], f"retr\nMSE={mses[i]:.3f}")
    for ax, lbl in zip(axes[:, 0], row_labels):
        ax.set_ylabel(lbl, fontsize=8)
    fig.suptitle("Sanity Check 2: Half-Occlusion Retrieval", fontsize=10, y=1.01)
    plt.tight_layout()
    fig.savefig(str(FIG_DIR / "02_half_occlusion.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return mses, accs


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check 3 — Gaussian noise robustness
# ─────────────────────────────────────────────────────────────────────────────

def check3_gaussian_robustness(
    hop: ContinuousHopfield,
    patterns: torch.Tensor,
) -> tuple[list[float], list[float], list[float]]:
    """
    Sweep sigma.  Return (sigmas, mean_mses, mean_accs).
    """
    X = hop.X
    sigmas = [0.05, 0.1, 0.2, 0.3, 0.5]
    mean_mses = []
    mean_accs = []

    for sigma in sigmas:
        batch_mse  = []
        batch_acc  = []
        for i in range(patterns.shape[0]):
            query     = add_gaussian_noise(patterns[i], sigma=sigma, seed=SEED + i)
            retrieved = hop.retrieve(query, steps=1)
            batch_mse.append(mse(retrieved, patterns[i]))
            batch_acc.append(float(retrieval_accuracy(retrieved, X, i)))
        mean_mses.append(float(np.mean(batch_mse)))
        mean_accs.append(float(np.mean(batch_acc)))

    # Panel plot: MSE vs sigma | accuracy vs sigma
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.plot(sigmas, mean_mses, "o-", color="steelblue")
    ax1.set_xlabel("Noise sigma"); ax1.set_ylabel("Mean MSE")
    ax1.set_title("MSE vs Gaussian sigma")
    ax2.plot(sigmas, mean_accs, "s-", color="darkorange")
    ax2.set_xlabel("Noise sigma"); ax2.set_ylabel("Retrieval accuracy")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Accuracy vs Gaussian sigma")
    fig.suptitle("Sanity Check 3: Gaussian Noise Robustness", fontsize=10)
    plt.tight_layout()
    fig.savefig(str(FIG_DIR / "03_gaussian_robustness.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Qualitative examples at sigma=0.2
    sigma_qual = 0.2
    fig2, axes = plt.subplots(3, 10, figsize=(14, 5))
    for i in range(10):
        query     = add_gaussian_noise(patterns[i], sigma=sigma_qual, seed=SEED + i)
        retrieved = hop.retrieve(query, steps=1)
        _show(axes[0, i], patterns[i], f"orig {i}")
        _show(axes[1, i], query,       f"σ={sigma_qual}")
        _show(axes[2, i], retrieved,   "retrieved")
    for ax, lbl in zip(axes[:, 0], ["Original", "Corrupted", "Retrieved"]):
        ax.set_ylabel(lbl, fontsize=8)
    fig2.suptitle("Sanity Check 3b: Examples at sigma=0.2", fontsize=10, y=1.01)
    plt.tight_layout()
    fig2.savefig(str(FIG_DIR / "03b_gaussian_examples.png"), dpi=150, bbox_inches="tight")
    plt.close(fig2)

    return sigmas, mean_mses, mean_accs


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check 4 — capacity stress test
# ─────────────────────────────────────────────────────────────────────────────

def check4_capacity(
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[list[int], list[float], list[float]]:
    """
    Vary N (number of stored patterns).  For each N, take 50 stored patterns,
    corrupt with sigma=0.2, retrieve, and report mean MSE + accuracy.
    """
    Ns = [10, 100, 1000, 5000]
    SIGMA_CAP = 0.2
    N_PROBE   = 50   # how many of the stored patterns we probe

    rng = torch.Generator()
    rng.manual_seed(SEED)

    cap_mses = []
    cap_accs = []

    for N in Ns:
        # Sample N unique indices
        perm = torch.randperm(images.shape[0], generator=rng)[:N]
        stored = images[perm]      # (N, 784)
        X = patterns_to_X(stored)  # (784, N)
        hop = ContinuousHopfield(X, beta=BETA)

        n_probe = min(N_PROBE, N)
        batch_mse = []
        batch_acc = []
        for j in range(n_probe):
            query     = add_gaussian_noise(stored[j], sigma=SIGMA_CAP, seed=SEED + j)
            retrieved = hop.retrieve(query, steps=1)
            batch_mse.append(mse(retrieved, stored[j]))
            batch_acc.append(float(retrieval_accuracy(retrieved, X, j)))

        cap_mses.append(float(np.mean(batch_mse)))
        cap_accs.append(float(np.mean(batch_acc)))
        print(f"  N={N:5d}: MSE={cap_mses[-1]:.4f}  acc={cap_accs[-1]:.3f}")

    # 2-panel plot with log-x
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.semilogx(Ns, cap_mses, "o-", color="steelblue")
    ax1.set_xlabel("N (stored patterns)"); ax1.set_ylabel("Mean MSE")
    ax1.set_title("MSE vs Capacity")
    ax2.semilogx(Ns, cap_accs, "s-", color="darkorange")
    ax2.set_xlabel("N (stored patterns)"); ax2.set_ylabel("Retrieval accuracy")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Accuracy vs Capacity")
    fig.suptitle("Sanity Check 4: Capacity Stress Test", fontsize=10)
    plt.tight_layout()
    fig.savefig(str(FIG_DIR / "04_capacity.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return Ns, cap_mses, cap_accs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Phase 1 Sanity Checks — Continuous Hopfield Network on MNIST")
    print("=" * 60)

    # Load data
    print("\nLoading MNIST …")
    images, labels = load_mnist_train()
    print(f"  Train set: {images.shape[0]} images, shape per image: {images.shape[1]}")

    # Build base network with 10 patterns (one per class)
    patterns, ds_indices = pick_one_per_class(images, labels)
    X10 = patterns_to_X(patterns)    # (784, 10)
    hop10 = ContinuousHopfield(X10, beta=BETA)
    print(f"  Pattern matrix X shape: {X10.shape}  (beta={BETA})")

    # ── Check 1 ──────────────────────────────────────────────────────────────
    print("\n[1/4] Clean retrieval …")
    mses1 = check1_clean_retrieval(hop10, patterns)
    print(f"  Per-pattern MSE: {[f'{v:.4f}' for v in mses1]}")
    print(f"  Mean MSE: {np.mean(mses1):.6f}  (expected ~= 0)")

    # ── Check 2 ──────────────────────────────────────────────────────────────
    print("\n[2/4] Half-occlusion retrieval …")
    mses2, accs2 = check2_half_occlusion(hop10, patterns)
    n_correct2 = sum(accs2)
    print(f"  Per-pattern MSE:      {[f'{v:.4f}' for v in mses2]}")
    print(f"  Mean MSE:             {np.mean(mses2):.4f}")
    print(f"  Retrieval accuracy:   {n_correct2}/{len(accs2)}")

    # ── Check 3 ──────────────────────────────────────────────────────────────
    print("\n[3/4] Gaussian noise robustness …")
    sigmas, mses3, accs3 = check3_gaussian_robustness(hop10, patterns)
    print(f"  {'sigma':>8}  {'MSE':>8}  {'accuracy':>10}")
    for s, m, a in zip(sigmas, mses3, accs3):
        print(f"  {s:>8.2f}  {m:>8.4f}  {a:>10.3f}")

    # ── Check 4 ──────────────────────────────────────────────────────────────
    print("\n[4/4] Capacity stress test …")
    Ns, mses4, accs4 = check4_capacity(images, labels)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n--- Check 1: Clean Retrieval ---")
    print(f"  Mean MSE (all 10 patterns): {np.mean(mses1):.6f}")
    print(f"  Pass (MSE < 1e-3 per pattern): {all(v < 1e-3 for v in mses1)}")

    print("\n--- Check 2: Half-Occlusion ---")
    print(f"  Mean MSE:             {np.mean(mses2):.4f}")
    print(f"  Retrieval accuracy:   {n_correct2}/{len(accs2)}")
    print(f"  Pass (≥9/10 correct): {n_correct2 >= 9}")

    print("\n--- Check 3: Gaussian Noise Robustness ---")
    print(f"  {'sigma':>8}  {'mean MSE':>10}  {'accuracy':>10}")
    for s, m, a in zip(sigmas, mses3, accs3):
        print(f"  {s:>8.2f}  {m:>10.4f}  {a:>10.3f}")

    print("\n--- Check 4: Capacity ---")
    print(f"  {'N':>8}  {'mean MSE':>10}  {'accuracy':>10}")
    for n, m, a in zip(Ns, mses4, accs4):
        print(f"  {n:>8d}  {m:>10.4f}  {a:>10.3f}")

    print("\nFigures saved to:", FIG_DIR)
    figs = sorted(FIG_DIR.glob("*.png"))
    for f in figs:
        print(f"  {f.name}")
    print("\nDone.")


if __name__ == "__main__":
    main()
