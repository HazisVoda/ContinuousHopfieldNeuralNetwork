"""
Presentation visuals for Fashion-MNIST experiments.

Generates four figures:
  fmnist_stored_patterns.png   — 10x10 grid of stored patterns (one per class slot)
  fmnist_retrieval_demo.png    — clean / Gaussian / occlusion retrieval examples
  fmnist_attack_demo.png       — one-pixel adversarial attack on each vulnerable probe
  fmnist_vs_mnist.png          — side-by-side similarity comparison (why FMNIST is harder)

Run: python -m experiments.fmnist_visualization
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hopfield.network    import ContinuousHopfield
from hopfield.corruption import add_gaussian_noise, mask_bottom_half
from hopfield.metrics    import retrieval_accuracy
from hopfield.sampling   import sample_class_balanced, sample_random
from hopfield.attacks    import WhiteBoxOnePixelAttacker

# ── config (must match Phase 3) ───────────────────────────────────────────────
SEED       = 42
BETA       = 8.0
N          = 100
N_VALUES   = [10, 50, 100, 500, 1000]
STRATEGIES = ["random", "class_balanced"]
N_IDX      = N_VALUES.index(N)
S_IDX      = STRATEGIES.index("class_balanced")

FMNIST_CLASSES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",  "Shirt",   "Sneaker",  "Bag",   "Ankle boot",
]
MNIST_CLASSES = [str(i) for i in range(10)]

DATA_DIR = ROOT / "data"
FIG_DIR  = ROOT / "figures"
EXP_DIR  = ROOT / "experiments"
FIG_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_fmnist() -> tuple[torch.Tensor, torch.Tensor]:
    ds = torchvision.datasets.FashionMNIST(
        root=str(DATA_DIR), train=True, download=False, transform=T.ToTensor()
    )
    return ds.data.float().view(-1, 784) / 255.0, ds.targets


def load_mnist() -> tuple[torch.Tensor, torch.Tensor]:
    ds = torchvision.datasets.MNIST(
        root=str(DATA_DIR), train=True, download=False, transform=T.ToTensor()
    )
    return ds.data.float().view(-1, 784) / 255.0, ds.targets


# ─────────────────────────────────────────────────────────────────────────────
# Cell builder (identical seeding to Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

def build_cell(
    images: torch.Tensor, labels: torch.Tensor, seed: int = SEED,
) -> tuple[torch.Tensor, torch.Tensor, ContinuousHopfield, list[int]]:
    X, _ = sample_class_balanced((images, labels), N, seed=seed + N_IDX)
    stored = X.T.contiguous()
    hop    = ContinuousHopfield(X, beta=BETA)
    probe_seed = seed * 1000 + N_IDX * 100 + S_IDX * 10
    rng = torch.Generator()
    rng.manual_seed(probe_seed)
    probe_indices = torch.randperm(N, generator=rng)[:50].tolist()
    return X, stored, hop, probe_indices


def read_vulnerable_probes() -> list[dict]:
    path = EXP_DIR / "phase3_fashion_mnist_results.csv"
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if int(r["clean_ok"]) == 1 and int(r["wb_attack_success"]) == 1]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: stored pattern grid
# ─────────────────────────────────────────────────────────────────────────────

def fig_stored_patterns(images: torch.Tensor, labels: torch.Tensor) -> None:
    X, stored, _, _ = build_cell(images, labels)

    fig, axes = plt.subplots(10, 10, figsize=(12, 12))
    for cls in range(10):
        for k in range(10):
            pat_idx = cls * 10 + k
            ax = axes[cls][k]
            ax.imshow(stored[pat_idx].view(28, 28).numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if k == 0:
                ax.set_title(FMNIST_CLASSES[cls], fontsize=8, loc="left", pad=2)

    fig.suptitle(
        "Fashion-MNIST stored patterns (N=100, class-balanced, seed=42)\n"
        "Each row = one class, each column = one stored exemplar",
        fontsize=11, fontweight="bold",
    )
    plt.subplots_adjust(wspace=0.05, hspace=0.25)
    out = FIG_DIR / "fmnist_stored_patterns.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: retrieval demo (clean / gaussian / occlusion)
# ─────────────────────────────────────────────────────────────────────────────

def fig_retrieval_demo(images: torch.Tensor, labels: torch.Tensor) -> None:
    X, stored, hop, probe_indices = build_cell(images, labels)

    # Pick 4 probes that retrieve correctly when clean
    examples: list[int] = []
    for ti in probe_indices:
        ret = hop.retrieve(stored[ti], steps=1)
        if retrieval_accuracy(ret, X, ti) and len(examples) < 4:
            examples.append(ti)
    if not examples:
        print("  [retrieval demo] No clean-retrievable probes found for seed 42.")
        return

    cols   = ["Original", "Gaussian\nquery (σ=0.3)", "Retrieved\n(Gaussian)", "Occluded\nquery", "Retrieved\n(Occluded)"]
    n_rows = len(examples)
    fig, axes = plt.subplots(n_rows, 5, figsize=(13, 3.0 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, ti in enumerate(examples):
        pat  = stored[ti]
        cls  = ti // 10

        noisy  = add_gaussian_noise(pat, sigma=0.3, seed=SEED + ti)
        ret_g  = hop.retrieve(noisy, steps=1)
        ok_g   = retrieval_accuracy(ret_g, X, ti)

        occl   = mask_bottom_half(pat)
        ret_o  = hop.retrieve(occl, steps=1)
        ok_o   = retrieval_accuracy(ret_o, X, ti)

        panels = [pat, noisy, ret_g, occl, ret_o]
        ok_flags = [True, None, ok_g, None, ok_o]

        for col, (img, ok) in enumerate(zip(panels, ok_flags)):
            ax = axes[row, col]
            ax.imshow(img.view(28, 28).numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(cols[col], fontsize=9)
            if ok is not None:
                color  = "#2ca02c" if ok else "#d62728"
                label  = "Correct" if ok else "Wrong"
                ax.set_xlabel(label, color=color, fontsize=8, labelpad=2)

        axes[row, 0].set_ylabel(FMNIST_CLASSES[cls], fontsize=9, rotation=0,
                                labelpad=60, va="center")

    fig.suptitle(
        "Fashion-MNIST retrieval demo: N=100 class-balanced Hopfield network",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out = FIG_DIR / "fmnist_retrieval_demo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: one-pixel attack demo on each vulnerable probe
# ─────────────────────────────────────────────────────────────────────────────

def fig_attack_demo(images: torch.Tensor, labels: torch.Tensor) -> None:
    vuln_rows = read_vulnerable_probes()
    if not vuln_rows:
        print("  [attack demo] No vulnerable FMNIST probes found.")
        return

    attacker = WhiteBoxOnePixelAttacker()
    records: list[dict] = []

    for vr in vuln_rows:
        seed      = int(vr["seed"])
        probe_idx = int(vr["probe_idx"])

        X, stored, hop, probe_indices = build_cell(images, labels, seed)
        true_idx  = probe_indices[probe_idx]
        q         = stored[true_idx]

        res = attacker.attack(q, true_idx, hop)
        if not res["success"]:
            continue

        adv = q.clone()
        adv[res["pixel_i"] * 28 + res["pixel_j"]] = res["pixel_value"]
        ret = hop.retrieve(adv, steps=1)

        records.append({
            "seed":       seed,
            "true_idx":   true_idx,
            "cls":        true_idx // 10,
            "ret_idx":    res["retrieved_index"],
            "ret_cls":    res["retrieved_index"] // 10,
            "q":          q,
            "adv":        adv,
            "ret_pat":    stored[res["retrieved_index"]],
            "pi":         res["pixel_i"],
            "pj":         res["pixel_j"],
            "pv":         res["pixel_value"],
            "orig_pv":    res["original_value"],
        })

    if not records:
        print("  [attack demo] Attacker did not succeed on any re-run.")
        return

    n_rows = len(records)
    fig, axes = plt.subplots(n_rows, 4, figsize=(11, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original query", "Adversarial query\n(one pixel changed)",
                  "Difference\n(×50)", "Retrieved pattern\n(wrong)"]

    for row, rec in enumerate(records):
        q_img   = rec["q"].view(28, 28).numpy()
        adv_img = rec["adv"].view(28, 28).numpy()
        diff    = np.abs(adv_img - q_img) * 50
        ret_img = rec["ret_pat"].view(28, 28).numpy()

        panels = [q_img, adv_img, diff, ret_img]
        cmaps  = ["gray", "gray", "hot", "gray"]

        for col, (img, cmap) in enumerate(zip(panels, cmaps)):
            ax = axes[row, col]
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9)

            # Highlight the attacked pixel on the adversarial image
            if col == 1:
                rect = patches.Rectangle(
                    (rec["pj"] - 0.5, rec["pi"] - 0.5), 1, 1,
                    linewidth=2, edgecolor="red", facecolor="none",
                )
                ax.add_patch(rect)

        # Row label
        axes[row, 0].set_ylabel(
            f"seed={rec['seed']}\n"
            f"True: {FMNIST_CLASSES[rec['cls']]} (idx {rec['true_idx']})\n"
            f"Got:  {FMNIST_CLASSES[rec['ret_cls']]} (idx {rec['ret_idx']})\n"
            f"Pixel ({rec['pi']},{rec['pj']}): "
            f"{rec['orig_pv']:.2f}→{rec['pv']:.2f}",
            fontsize=7.5, rotation=0, labelpad=120, va="center",
        )

    fig.suptitle(
        "One-pixel adversarial attack on Fashion-MNIST\n"
        "N=100 class-balanced Hopfield network — all conditionally vulnerable probes",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out = FIG_DIR / "fmnist_attack_demo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: MNIST vs FMNIST — why Fashion-MNIST is harder
# ─────────────────────────────────────────────────────────────────────────────

def fig_vs_mnist(
    fmnist_images: torch.Tensor, fmnist_labels: torch.Tensor,
    mnist_images:  torch.Tensor, mnist_labels:  torch.Tensor,
) -> None:
    Xf, _, _, _ = build_cell(fmnist_images, fmnist_labels)
    Xm, _, _, _ = build_cell(mnist_images,  mnist_labels)

    def pairwise_off_diag_cosine(X: torch.Tensor) -> np.ndarray:
        Xn  = X.numpy()
        nrm = np.linalg.norm(Xn, axis=0, keepdims=True)
        Xnn = Xn / (nrm + 1e-8)
        C   = Xnn.T @ Xnn
        mask = ~np.eye(C.shape[0], dtype=bool)
        return C[mask]

    cos_mnist  = pairwise_off_diag_cosine(Xm)
    cos_fmnist = pairwise_off_diag_cosine(Xf)

    fig = plt.figure(figsize=(14, 9))
    gs  = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    # ── Top row: sample stored patterns ──────────────────────────────────────
    ax_m = fig.add_subplot(gs[0, :2])
    ax_f = fig.add_subplot(gs[0, 2])

    # MNIST: one exemplar per class (indices 0,10,20,...,90 in class-balanced X)
    mnist_strip = np.concatenate(
        [Xm[:, cls * 10].view(28, 28).numpy() for cls in range(10)], axis=1
    )
    ax_m.imshow(mnist_strip, cmap="gray", vmin=0, vmax=1)
    ax_m.axis("off")
    ax_m.set_title("MNIST stored exemplars (one per class)", fontsize=9)
    for cls in range(10):
        ax_m.text(cls * 28 + 14, 30, MNIST_CLASSES[cls],
                  ha="center", va="top", fontsize=8, color="white",
                  bbox=dict(boxstyle="round,pad=0.1", fc="#333", alpha=0.6))

    fmnist_strip = np.concatenate(
        [Xf[:, cls * 10].view(28, 28).numpy() for cls in range(10)], axis=1
    )
    # Show only 5 (won't fit 10 in 1/3 width) — use a 2×5 mini grid
    mini = np.zeros((56, 5 * 28), dtype=np.float32)
    for cls in range(5):
        mini[:28,  cls * 28:(cls + 1) * 28] = Xf[:, cls * 10].view(28, 28).numpy()
        mini[28:, cls * 28:(cls + 1) * 28]  = Xf[:, (cls + 5) * 10].view(28, 28).numpy()
    ax_f.imshow(mini, cmap="gray", vmin=0, vmax=1)
    ax_f.axis("off")
    ax_f.set_title("F-MNIST exemplars\n(2 rows × 5 classes)", fontsize=9)

    # ── Bottom left: pairwise cosine histograms ───────────────────────────────
    ax_hist = fig.add_subplot(gs[1, :2])
    bins = np.linspace(-0.2, 1.0, 60)
    ax_hist.hist(cos_mnist,  bins=bins, alpha=0.65, color="steelblue",
                 label=f"MNIST   (mean={cos_mnist.mean():.3f}, median={np.median(cos_mnist):.3f})")
    ax_hist.hist(cos_fmnist, bins=bins, alpha=0.65, color="darkorange",
                 label=f"F-MNIST (mean={cos_fmnist.mean():.3f}, median={np.median(cos_fmnist):.3f})")
    ax_hist.axvline(cos_mnist.mean(),  color="steelblue",  linestyle="--", lw=1.5)
    ax_hist.axvline(cos_fmnist.mean(), color="darkorange", linestyle="--", lw=1.5)
    ax_hist.set_xlabel("Pairwise cosine similarity between stored patterns", fontsize=10)
    ax_hist.set_ylabel("Count", fontsize=10)
    ax_hist.set_title("MNIST vs Fashion-MNIST: pairwise pattern similarity\n"
                      "N=100 class-balanced — higher similarity → harder retrieval", fontsize=9)
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.2)

    # ── Bottom right: key metrics bar chart ───────────────────────────────────
    ax_bar = fig.add_subplot(gs[1, 2])
    metrics = ["Baseline\nfailure", "Raw WB\nsuccess"]
    m_vals  = [0.092, 0.120]
    f_vals  = [0.800, 0.816]
    x  = np.arange(2);  bw = 0.35
    ax_bar.bar(x - bw/2, m_vals, bw, label="MNIST",   color="steelblue",  alpha=0.8)
    ax_bar.bar(x + bw/2, f_vals, bw, label="F-MNIST", color="darkorange", alpha=0.8)
    for xi, mv, fv in zip(x, m_vals, f_vals):
        ax_bar.text(xi - bw/2, mv + 0.02, f"{mv:.1%}", ha="center", fontsize=8)
        ax_bar.text(xi + bw/2, fv + 0.02, f"{fv:.1%}", ha="center", fontsize=8)
    ax_bar.set_xticks(x);  ax_bar.set_xticklabels(metrics, fontsize=9)
    ax_bar.set_ylabel("Rate", fontsize=9);  ax_bar.set_ylim(0, 1.05)
    ax_bar.set_title("Key metrics\nMNIST vs F-MNIST", fontsize=9)
    ax_bar.legend(fontsize=8);  ax_bar.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Why Fashion-MNIST is harder for the Hopfield network than MNIST\n"
        "Higher inter-pattern similarity → crowded energy landscape → retrieval failures",
        fontsize=11, fontweight="bold",
    )

    out = FIG_DIR / "fmnist_vs_mnist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Fashion-MNIST visualization")
    print("=" * 40)

    print("Loading datasets ...")
    fmnist_images, fmnist_labels = load_fmnist()
    mnist_images,  mnist_labels  = load_mnist()
    print("  ok")

    print("\nFigure 1: stored pattern grid ...")
    fig_stored_patterns(fmnist_images, fmnist_labels)

    print("Figure 2: retrieval demo ...")
    fig_retrieval_demo(fmnist_images, fmnist_labels)

    print("Figure 3: one-pixel attack demo ...")
    fig_attack_demo(fmnist_images, fmnist_labels)

    print("Figure 4: MNIST vs Fashion-MNIST comparison ...")
    fig_vs_mnist(fmnist_images, fmnist_labels, mnist_images, mnist_labels)

    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
