"""
Grayscale CIFAR-10 experiments and presentation visualization.

Runs white-box one-pixel attack experiments at N=10 class-balanced
(the lowest-failure operating point found in the baseline characterization)
and generates four figures analogous to fmnist_visualization.py.

Inline helpers replace the 28x28-specific corruption.py and the
784-hardcoded WhiteBoxOnePixelAttacker — no existing modules are modified.

Run: python -m experiments.grayscale_cifar_visualization
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
import matplotlib.patches as patches

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hopfield.network  import ContinuousHopfield
from hopfield.metrics  import retrieval_accuracy
from hopfield.sampling import sample_class_balanced

# ── config ─────────────────────────────────────────────────────────────────────
SEEDS    = [42, 43, 44, 45, 46]
BETA     = 8.0
N        = 10
N_PROBE  = 50          # capped to N when N < 50
IMG_DIM  = 1024        # 32×32
IMG_SIZE = 32
MATCHED_SIGMA = 0.005  # magnitude-matched from Phase 3 Diagnostic C

CIFAR_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog",      "Frog",       "Horse", "Ship", "Truck",
]

DATA_DIR = ROOT / "data"
EXP_DIR  = ROOT / "experiments"
FIG_DIR  = ROOT / "figures"
OUT_DIR  = ROOT / "excel_exports"
FIG_DIR.mkdir(exist_ok=True)

CANDIDATE_VALUES = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar10_gray() -> tuple[torch.Tensor, torch.Tensor]:
    ds   = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False)
    data = ds.data.astype(np.float32) / 255.0          # (50000, 32, 32, 3)
    gray = (0.2989 * data[:, :, :, 0]
            + 0.5870 * data[:, :, :, 1]
            + 0.1140 * data[:, :, :, 2])               # (50000, 32, 32)
    images  = torch.tensor(gray.reshape(-1, IMG_DIM), dtype=torch.float32)
    targets = torch.tensor(ds.targets, dtype=torch.long)
    return images, targets


def load_mnist() -> tuple[torch.Tensor, torch.Tensor]:
    ds = torchvision.datasets.MNIST(
        root=str(DATA_DIR), train=True, download=False,
        transform=torchvision.transforms.ToTensor(),
    )
    return ds.data.float().view(-1, 784) / 255.0, ds.targets


# ─────────────────────────────────────────────────────────────────────────────
# Cell builder
# ─────────────────────────────────────────────────────────────────────────────

def build_cell(
    images: torch.Tensor, labels: torch.Tensor, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, ContinuousHopfield, list[int]]:
    X, _   = sample_class_balanced((images, labels), N, seed=seed)
    stored = X.T.contiguous()           # (N, 1024)
    hop    = ContinuousHopfield(X, beta=BETA)
    n_p    = min(N_PROBE, N)
    rng    = torch.Generator()
    rng.manual_seed(seed * 1000 + N)
    probe_indices = torch.randperm(N, generator=rng)[:n_p].tolist()
    return X, stored, hop, probe_indices


# ─────────────────────────────────────────────────────────────────────────────
# Inline corruption helpers (32×32 / 1024-dim)
# ─────────────────────────────────────────────────────────────────────────────

def gaussian_noise(q: torch.Tensor, sigma: float, seed: int) -> torch.Tensor:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return (q + torch.zeros_like(q).normal_(0.0, sigma, generator=rng)).clamp(0.0, 1.0)


def mask_bottom_half(q: torch.Tensor) -> torch.Tensor:
    img = q.clone().view(IMG_SIZE, IMG_SIZE)
    img[IMG_SIZE // 2:, :] = 0.0
    return img.view(IMG_DIM)


# ─────────────────────────────────────────────────────────────────────────────
# Inline exhaustive one-pixel attacker (1024-dim CIFAR)
# ─────────────────────────────────────────────────────────────────────────────

def wb_attack_cifar(
    query: torch.Tensor, true_index: int, hop: ContinuousHopfield,
) -> dict:
    X       = hop.X                                   # (1024, N)
    cands   = CANDIDATE_VALUES
    n_cands = IMG_DIM * len(cands)                    # 1024 × 5 = 5120

    locs = torch.arange(IMG_DIM).repeat_interleave(len(cands))
    vals = cands.repeat(IMG_DIM)

    queries = query.unsqueeze(1).expand(-1, n_cands).clone()
    queries[locs, torch.arange(n_cands)] = vals

    retrieved = hop.retrieve(queries, steps=1)        # (1024, 5120)
    true_pat  = X[:, true_index]
    dots      = retrieved.T @ true_pat
    r_norms   = retrieved.norm(dim=0)
    t_norm    = true_pat.norm()
    cos_sims  = dots / (r_norms * t_norm).clamp(min=1e-8)

    worst_k   = int(cos_sims.argmin().item())
    worst_loc = int(locs[worst_k].item())
    worst_val = float(vals[worst_k].item())

    worst_ret   = retrieved[:, worst_k]
    X_norms     = X.norm(dim=0)
    cos_all     = (X.T @ worst_ret) / (X_norms * worst_ret.norm()).clamp(min=1e-8)
    ret_idx     = int(cos_all.argmax().item())
    orig_val    = float(query[worst_loc].item())

    return {
        "success":         ret_idx != true_index,
        "pixel_i":         worst_loc // IMG_SIZE,
        "pixel_j":         worst_loc % IMG_SIZE,
        "pixel_value":     worst_val,
        "original_value":  orig_val,
        "perturbation_l2": abs(worst_val - orig_val),
        "cosine_to_true":  float(cos_sims[worst_k].item()),
        "retrieved_index": ret_idx,
        "evaluations":     n_cands,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise cosine stats
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_off_diag_cosine(X: torch.Tensor) -> np.ndarray:
    Xn  = X.numpy()
    nrm = np.linalg.norm(Xn, axis=0, keepdims=True)
    Xnn = Xn / (nrm + 1e-8)
    C   = Xnn.T @ Xnn
    mask = ~np.eye(C.shape[0], dtype=bool)
    return C[mask]


# ─────────────────────────────────────────────────────────────────────────────
# Run experiments across 5 seeds
# ─────────────────────────────────────────────────────────────────────────────

def run_attacks(
    images: torch.Tensor, labels: torch.Tensor,
) -> tuple[list[dict], list[dict]]:
    """Return (probe_rows, per_seed_rows)."""
    probe_rows: list[dict] = []
    per_seed:   list[dict] = []

    for seed in SEEDS:
        print(f"  Seed {seed} ...")
        X, stored, hop, probe_indices = build_cell(images, labels, seed)

        clean_ok_list: list[bool] = []
        wb_ok_list:    list[int]  = []
        rne_fail_list: list[int]  = []

        for j, true_idx in enumerate(probe_indices):
            q = stored[true_idx]

            ret_clean = hop.retrieve(q, steps=1)
            is_clean  = retrieval_accuracy(ret_clean, X, true_idx)
            clean_ok_list.append(is_clean)

            res = wb_attack_cifar(q, true_idx, hop)
            wb_ok_list.append(int(res["success"]))

            noisy    = gaussian_noise(q, MATCHED_SIGMA, seed=seed + j)
            ret_noisy = hop.retrieve(noisy, steps=1)
            rne_fail_list.append(int(not retrieval_accuracy(ret_noisy, X, true_idx)))

            probe_rows.append({
                "seed":             seed,
                "probe_idx":        j,
                "true_index":       true_idx,
                "cifar_class":      true_idx,    # N=10 class_balanced: true_idx == class
                "clean_ok":         int(is_clean),
                "wb_attack_success": int(res["success"]),
                "pixel_i":          res["pixel_i"],
                "pixel_j":          res["pixel_j"],
                "pixel_value":      round(res["pixel_value"], 4),
                "original_value":   round(res["original_value"], 4),
                "perturbation_l2":  round(res["perturbation_l2"], 5),
                "retrieved_index":  res["retrieved_index"],
                "rne_fail":         rne_fail_list[-1],
            })

        n_total  = len(probe_indices)
        n_clean  = sum(clean_ok_list)
        cond_wb  = sum(int(wb_ok_list[j] and clean_ok_list[j]) for j in range(n_total))
        cond_rne = sum(int(rne_fail_list[j] and clean_ok_list[j]) for j in range(n_total))

        per_seed.append({
            "seed":                  seed,
            "n_total":               n_total,
            "n_clean_ok":            n_clean,
            "baseline_failure_rate": round(1 - n_clean / n_total, 4),
            "raw_wb_success":        round(sum(wb_ok_list) / n_total, 4),
            "cond_wb_success":       round(cond_wb / max(n_clean, 1), 4),
            "cond_rne_fail":         round(cond_rne / max(n_clean, 1), 4),
        })

    return probe_rows, per_seed


def save_attack_csv(probe_rows: list[dict], per_seed: list[dict]) -> None:
    fields = ["seed", "probe_idx", "true_index", "cifar_class", "clean_ok",
              "wb_attack_success", "pixel_i", "pixel_j", "pixel_value",
              "original_value", "perturbation_l2", "retrieved_index", "rne_fail"]
    path = EXP_DIR / "grayscale_cifar_attack_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in probe_rows:
            w.writerow({k: row[k] for k in fields})
    print(f"  Saved: {path.name}  ({len(probe_rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: stored pattern grid (seed=42, all 10 classes)
# ─────────────────────────────────────────────────────────────────────────────

def fig_stored_patterns(images: torch.Tensor, labels: torch.Tensor) -> None:
    X, stored, _, _ = build_cell(images, labels, SEEDS[0])
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for cls in range(10):
        ax = axes[cls // 5][cls % 5]
        ax.imshow(stored[cls].view(IMG_SIZE, IMG_SIZE).numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(CIFAR_CLASSES[cls], fontsize=9)

    fig.suptitle(
        f"Grayscale CIFAR-10 stored patterns (N={N}, class-balanced, seed={SEEDS[0]})\n"
        "One pattern per class — 32×32 grayscale, normalised [0,1]",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out = FIG_DIR / "grayscale_cifar_stored_patterns.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: retrieval demo — aggregate across seeds to find clean examples
# ─────────────────────────────────────────────────────────────────────────────

def fig_retrieval_demo(images: torch.Tensor, labels: torch.Tensor) -> None:
    examples: list[dict] = []
    for seed in SEEDS:
        if len(examples) >= 4:
            break
        X, stored, hop, probe_indices = build_cell(images, labels, seed)
        for ti in probe_indices:
            if len(examples) >= 4:
                break
            ret_clean = hop.retrieve(stored[ti], steps=1)
            if retrieval_accuracy(ret_clean, X, ti):
                noisy = gaussian_noise(stored[ti], sigma=0.3, seed=seed + ti)
                ret_g = hop.retrieve(noisy, steps=1)
                ok_g  = retrieval_accuracy(ret_g, X, ti)
                occl  = mask_bottom_half(stored[ti])
                ret_o = hop.retrieve(occl, steps=1)
                ok_o  = retrieval_accuracy(ret_o, X, ti)
                examples.append({
                    "seed": seed, "ti": ti, "cls": ti,
                    "pat": stored[ti], "noisy": noisy, "ret_g": ret_g,
                    "occl": occl, "ret_o": ret_o, "ok_g": ok_g, "ok_o": ok_o,
                    "X": X,
                })

    if not examples:
        print("  [retrieval demo] No clean-retrievable probes found across all seeds.")
        return

    n_rows = len(examples)
    cols   = ["Original\npattern", "Gaussian\nquery (σ=0.3)", "Retrieved\n(Gaussian)",
              "Occluded\nquery", "Retrieved\n(Occluded)"]
    fig, axes = plt.subplots(n_rows, 5, figsize=(13, 3.0 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, ex in enumerate(examples):
        panels   = [ex["pat"], ex["noisy"], ex["ret_g"], ex["occl"], ex["ret_o"]]
        ok_flags = [None, None, ex["ok_g"], None, ex["ok_o"]]
        for col, (img, ok) in enumerate(zip(panels, ok_flags)):
            ax = axes[row, col]
            ax.imshow(img.view(IMG_SIZE, IMG_SIZE).numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(cols[col], fontsize=9)
            if ok is not None:
                color = "#2ca02c" if ok else "#d62728"
                ax.set_xlabel("Correct" if ok else "Wrong",
                              color=color, fontsize=8, labelpad=2)
        axes[row, 0].set_ylabel(
            f"seed={ex['seed']}\n{CIFAR_CLASSES[ex['cls']]}",
            fontsize=9, rotation=0, labelpad=70, va="center",
        )

    fig.suptitle(
        f"Grayscale CIFAR-10 retrieval demo: N={N} class-balanced Hopfield network\n"
        "Showing only clean-retrievable probes (found across all seeds)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out = FIG_DIR / "grayscale_cifar_retrieval_demo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: one-pixel attack demo
# ─────────────────────────────────────────────────────────────────────────────

def fig_attack_demo(probe_rows: list[dict], images: torch.Tensor, labels: torch.Tensor) -> None:
    vuln_rows = [r for r in probe_rows if r["clean_ok"] == 1 and r["wb_attack_success"] == 1]

    if not vuln_rows:
        # Show all clean-ok probes with attack attempt if no successes
        clean_rows = [r for r in probe_rows if r["clean_ok"] == 1][:4]
        n_rows     = max(len(clean_rows), 1)
        fig, axes  = plt.subplots(1, 1, figsize=(8, 4))
        axes.text(0.5, 0.5,
                  f"No successful one-pixel attacks found at N={N}.\n"
                  f"({len(clean_rows)} clean-retrievable probes attacked;\n"
                  f"high pattern crowding (mean cosine ≈0.65) prevents\n"
                  f"single-pixel perturbations from causing mis-retrieval.)",
                  ha="center", va="center", fontsize=12,
                  transform=axes.transAxes,
                  bbox=dict(boxstyle="round", fc="#fff3cd", ec="#ffc107", lw=1.5))
        axes.axis("off")
        fig.suptitle(f"Grayscale CIFAR-10: One-pixel attack at N={N} class-balanced",
                     fontsize=11, fontweight="bold")
        out = FIG_DIR / "grayscale_cifar_attack_demo.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out.name}  (no successful attacks — informational panel)")
        return

    records: list[dict] = []
    for vr in vuln_rows[:6]:   # cap at 6 rows
        seed     = vr["seed"]
        probe_j  = vr["probe_idx"]
        true_idx = vr["true_index"]
        ret_idx  = vr["retrieved_index"]
        X, stored, hop, probe_indices = build_cell(images, labels, seed)
        q   = stored[true_idx]
        adv = q.clone()
        adv[vr["pixel_i"] * IMG_SIZE + vr["pixel_j"]] = vr["pixel_value"]
        records.append({
            "seed": seed, "true_idx": true_idx, "ret_idx": ret_idx,
            "cls": CIFAR_CLASSES[vr["cifar_class"]],
            "ret_cls": CIFAR_CLASSES[vr["retrieved_index"]],
            "q": q, "adv": adv,
            "ret_pat": stored[ret_idx],
            "pi": vr["pixel_i"], "pj": vr["pixel_j"],
            "pv": vr["pixel_value"], "orig_pv": vr["original_value"],
        })

    col_titles = ["Original query", "Adversarial query\n(one pixel changed)",
                  "Difference ×50", "Retrieved pattern\n(wrong)"]
    n_rows = len(records)
    fig, axes = plt.subplots(n_rows, 4, figsize=(11, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, rec in enumerate(records):
        q_img   = rec["q"].view(IMG_SIZE, IMG_SIZE).numpy()
        adv_img = rec["adv"].view(IMG_SIZE, IMG_SIZE).numpy()
        diff    = np.abs(adv_img - q_img) * 50
        ret_img = rec["ret_pat"].view(IMG_SIZE, IMG_SIZE).numpy()
        panels  = [q_img, adv_img, diff, ret_img]
        cmaps   = ["gray", "gray", "hot", "gray"]

        for col, (img, cmap) in enumerate(zip(panels, cmaps)):
            ax = axes[row, col]
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=9)
            if col == 1:
                rect = patches.Rectangle(
                    (rec["pj"] - 0.5, rec["pi"] - 0.5), 1, 1,
                    linewidth=2, edgecolor="red", facecolor="none",
                )
                ax.add_patch(rect)

        axes[row, 0].set_ylabel(
            f"seed={rec['seed']}\n"
            f"True:  {rec['cls']} (idx {rec['true_idx']})\n"
            f"Got:   {rec['ret_cls']} (idx {rec['ret_idx']})\n"
            f"Pixel ({rec['pi']},{rec['pj']}): "
            f"{rec['orig_pv']:.2f}→{rec['pv']:.2f}",
            fontsize=7.5, rotation=0, labelpad=125, va="center",
        )

    fig.suptitle(
        f"One-pixel adversarial attack on grayscale CIFAR-10\n"
        f"N={N} class-balanced Hopfield network — all conditionally vulnerable probes",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out = FIG_DIR / "grayscale_cifar_attack_demo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}  ({len(records)} vulnerable probes)")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: CIFAR vs MNIST comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig_vs_mnist(
    cifar_images: torch.Tensor, cifar_labels: torch.Tensor,
    mnist_images: torch.Tensor, mnist_labels: torch.Tensor,
    per_seed: list[dict],
) -> None:
    Xc, _, _, _ = build_cell(cifar_images, cifar_labels, SEEDS[0])
    Xm, _ = sample_class_balanced((mnist_images, mnist_labels), N, seed=SEEDS[0])

    cos_cifar = pairwise_off_diag_cosine(Xc)
    cos_mnist = pairwise_off_diag_cosine(Xm)

    # Metrics from existing data
    cifar_bl_mean = float(np.mean([r["baseline_failure_rate"] for r in per_seed]))
    cifar_bl_std  = float(np.std( [r["baseline_failure_rate"] for r in per_seed], ddof=1))
    cifar_cw_mean = float(np.mean([r["cond_wb_success"]        for r in per_seed]))
    cifar_cw_std  = float(np.std( [r["cond_wb_success"]        for r in per_seed], ddof=1))
    fmnist_bl, fmnist_cw = 0.800, 0.076   # from phase3_final_diagnostics
    mnist_bl,  mnist_cw  = 0.092, 0.031   # from phase3_diag_b

    fig = plt.figure(figsize=(14, 9))
    gs  = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35)

    # ── Top-left: sample stored patterns (CIFAR 2×5) ─────────────────────────
    ax = fig.add_subplot(gs[0, :2])
    strip = np.concatenate(
        [Xc[:, cls].view(IMG_SIZE, IMG_SIZE).numpy() for cls in range(10)], axis=1
    )
    ax.imshow(strip, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title("Grayscale CIFAR-10 stored patterns (one per class, seed=42)", fontsize=9)
    for cls in range(10):
        ax.text(cls * IMG_SIZE + IMG_SIZE // 2, IMG_SIZE + 2,
                CIFAR_CLASSES[cls], ha="center", va="top", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.1", fc="#333", alpha=0.6), color="white")

    # ── Top-right: 5 MNIST exemplars (28×28, padded to 32×32) ───────────────
    ax = fig.add_subplot(gs[0, 2])
    mnist_strip = np.concatenate(
        [np.pad(Xm[:, cls].view(28, 28).numpy(), 2) for cls in range(5)], axis=1
    )
    ax.imshow(mnist_strip, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title("MNIST patterns\n(first 5 classes, for scale)", fontsize=9)

    # ── Bottom-left: pairwise cosine histograms ───────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    bins = np.linspace(-0.2, 1.0, 60)
    ax.hist(cos_mnist, bins=bins, alpha=0.65, color="steelblue",
            label=f"MNIST N={N}  (mean={cos_mnist.mean():.3f})")
    ax.hist(cos_cifar, bins=bins, alpha=0.65, color="darkorange",
            label=f"CIFAR-10 N={N}  (mean={cos_cifar.mean():.3f})")
    ax.axvline(cos_mnist.mean(), color="steelblue",  linestyle="--", lw=1.5)
    ax.axvline(cos_cifar.mean(), color="darkorange", linestyle="--", lw=1.5)
    ax.set_xlabel("Pairwise cosine similarity between stored patterns", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Pattern similarity: MNIST vs Grayscale CIFAR-10\n"
                 f"N={N} class-balanced — higher similarity → harder retrieval", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # ── Bottom-right: three-dataset baseline + attack bar chart ──────────────
    ax = fig.add_subplot(gs[1, 2])
    datasets = ["MNIST", "F-MNIST", "CIFAR-10\n(gray)"]
    bl_vals  = [mnist_bl,  fmnist_bl,  cifar_bl_mean]
    bl_errs  = [0.0,       0.0,        cifar_bl_std]
    cw_vals  = [mnist_cw,  fmnist_cw,  cifar_cw_mean]
    cw_errs  = [0.0,       0.0,        cifar_cw_std]
    x   = np.arange(3);  bw = 0.35
    bars_bl = ax.bar(x - bw/2, bl_vals, bw, label="Baseline failure",
                     color=["steelblue", "#ff7f0e", "darkorange"], alpha=0.8)
    bars_cw = ax.bar(x + bw/2, cw_vals, bw, label="Cond. WB attack",
                     color=["#2ca02c",   "#d62728",  "#9467bd"],   alpha=0.8)
    ax.errorbar(x - bw/2, bl_vals, yerr=bl_errs, fmt="none", color="black", capsize=4, lw=1.5)
    ax.errorbar(x + bw/2, cw_vals, yerr=cw_errs, fmt="none", color="black", capsize=4, lw=1.5)
    for xi, bv, cv in zip(x, bl_vals, cw_vals):
        ax.text(xi - bw/2, bv + 0.02, f"{bv:.0%}", ha="center", fontsize=8)
        ax.text(xi + bw/2, cv + 0.02, f"{cv:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x);  ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylim(0, 1.15);  ax.set_ylabel("Rate", fontsize=9)
    ax.set_title(f"MNIST / F-MNIST / CIFAR comparison\n"
                 f"N={N} class-balanced, 5-seed mean", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "Why Grayscale CIFAR-10 is harder than MNIST and Fashion-MNIST\n"
        "Higher pairwise cosine → crowded energy landscape → near-total retrieval failure",
        fontsize=11, fontweight="bold",
    )
    out = FIG_DIR / "grayscale_cifar_vs_mnist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Print summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(probe_rows: list[dict], per_seed: list[dict]) -> None:
    print("\n=== GRAYSCALE CIFAR-10 ATTACK RESULTS ===\n")
    print(f"{'Seed':>6}  {'BL fail':>8}  {'Raw WB':>7}  {'Cond WB':>8}  {'Cond RNE':>9}  {'#Clean':>6}")
    print("-" * 55)
    for r in per_seed:
        print(f"{r['seed']:>6}  {r['baseline_failure_rate']:>8.4f}  "
              f"{r['raw_wb_success']:>7.4f}  {r['cond_wb_success']:>8.4f}  "
              f"{r['cond_rne_fail']:>9.4f}  {r['n_clean_ok']:>6}")
    print("-" * 55)
    bl_m  = float(np.mean([r["baseline_failure_rate"] for r in per_seed]))
    cw_m  = float(np.mean([r["cond_wb_success"]       for r in per_seed]))
    rne_m = float(np.mean([r["cond_rne_fail"]          for r in per_seed]))
    bl_s  = float(np.std( [r["baseline_failure_rate"] for r in per_seed], ddof=1))
    cw_s  = float(np.std( [r["cond_wb_success"]       for r in per_seed], ddof=1))
    print(f"{'Mean':>6}  {bl_m:>8.4f}  {'':>7}  {cw_m:>8.4f}  {rne_m:>9.4f}")

    n_vuln  = sum(1 for r in probe_rows if r["clean_ok"] == 1 and r["wb_attack_success"] == 1)
    n_clean = sum(1 for r in probe_rows if r["clean_ok"] == 1)
    print(f"\nVulnerable (clean+attacked): {n_vuln} / {n_clean} clean-ok probes "
          f"/ {len(probe_rows)} total")

    print(f"\nComparison at N={N}:")
    print(f"  MNIST     — baseline fail:  9.2%   cond WB:  3.1%")
    print(f"  F-MNIST   — baseline fail: 80.0%   cond WB:  7.6%")
    print(f"  CIFAR-10  — baseline fail: {bl_m*100:.1f}%   cond WB: {cw_m*100:.1f}%")

    amp = cw_m / max(rne_m, 1e-9)
    if rne_m < 1e-6:
        amp_str = "undefined (0% conditional RNE failures)"
    else:
        amp_str = f"{amp:.1f}x"
    print(f"\nAmplification (cond WB / cond RNE): {amp_str}")

    print(f"\nVerdict:")
    if bl_m > 0.7:
        print(f"  Grayscale CIFAR-10 baseline failure of {bl_m*100:.1f}% at N={N} "
              f"places the network entirely outside its reliable retrieval regime. "
              f"Attack analysis is confounded by near-total baseline failure; "
              f"CIFAR pattern crowding (mean cosine ≈0.65) exceeds what this "
              f"architecture can separate at any tested N.")
    else:
        print(f"  CIFAR-10 at N={N} shows {bl_m*100:.1f}% baseline failure with "
              f"{cw_m*100:.1f}% conditional WB success.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    print("=" * 60)
    print("Grayscale CIFAR-10 Visualization & Attack Experiments")
    print("=" * 60)

    print("\nLoading datasets ...")
    cifar_images, cifar_labels = load_cifar10_gray()
    mnist_images,  mnist_labels  = load_mnist()
    print(f"  CIFAR: {len(cifar_images)} samples, dim=1024")
    print(f"  MNIST: {len(mnist_images)} samples, dim=784")

    print(f"\nRunning WB attack experiments (N={N}, {len(SEEDS)} seeds) ...")
    probe_rows, per_seed = run_attacks(cifar_images, cifar_labels)
    save_attack_csv(probe_rows, per_seed)

    print("\nGenerating figures ...")
    print("Figure 1: stored pattern grid ...")
    fig_stored_patterns(cifar_images, cifar_labels)

    print("Figure 2: retrieval demo ...")
    fig_retrieval_demo(cifar_images, cifar_labels)

    print("Figure 3: one-pixel attack demo ...")
    fig_attack_demo(probe_rows, cifar_images, cifar_labels)

    print("Figure 4: CIFAR vs MNIST comparison ...")
    fig_vs_mnist(cifar_images, cifar_labels, mnist_images, mnist_labels, per_seed)

    print_summary(probe_rows, per_seed)
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
