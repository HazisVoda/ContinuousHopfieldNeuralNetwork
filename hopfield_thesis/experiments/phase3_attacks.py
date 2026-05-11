"""
Phase 3: One-pixel adversarial attacks on the continuous Hopfield network.

Four stages:
  1. White-box exhaustive attack across the full N × strategy × seed grid.
  2. Magnitude-equivalent random-noise baseline (same probes, same cells).
  3. DE black-box attack at the headline cell (N=100, class-balanced).
  4. Pixel vulnerability maps for N ∈ {10, 100, 500} class-balanced.

Run from hopfield_thesis/ as:
    python -m experiments.phase3_attacks
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
import matplotlib.patches as patches
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hopfield.network      import ContinuousHopfield
from hopfield.corruption   import add_gaussian_noise
from hopfield.metrics      import retrieval_accuracy
from hopfield.sampling     import sample_random, sample_class_balanced
from hopfield.attacks      import WhiteBoxOnePixelAttacker, DEBlackBoxOnePixelAttacker
from hopfield.vulnerability import compute_vulnerability_map

# ── config ────────────────────────────────────────────────────────────────────
SEEDS      = [42, 43, 44, 45, 46]
BETA       = 8.0
N_VALUES   = [10, 50, 100, 500, 1000]
STRATEGIES = ["random", "class_balanced"]
N_PROBE    = 50
RNE_SIGMAS = [0.01, 0.02, 0.05, 0.10]     # random-noise-equivalent sweep

DATA_DIR = ROOT / "data"
FIG_DIR  = ROOT / "figures"
EXP_DIR  = ROOT / "experiments"
FIG_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ─────────────────────────────────────────────────────────────────────────────
# Data & cell helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_mnist_train() -> tuple[torch.Tensor, torch.Tensor]:
    ds = torchvision.datasets.MNIST(
        root=str(DATA_DIR), train=True, download=True, transform=T.ToTensor()
    )
    images = ds.data.float() / 255.0
    images = images.view(-1, 784)
    return images, ds.targets


def build_cell(
    images: torch.Tensor,
    labels: torch.Tensor,
    N: int,
    n_idx: int,
    s_idx: int,
    strategy: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, ContinuousHopfield, list[int]]:
    """Build (X, stored, hop, probe_indices) deterministically from (N,strategy,seed)."""
    dataset = (images, labels)
    if strategy == "random":
        X, _ = sample_random(dataset, N, seed=seed + n_idx)
    else:
        X, _ = sample_class_balanced(dataset, N, seed=seed + n_idx)

    stored = X.T.contiguous()          # (N, 784)
    hop    = ContinuousHopfield(X, beta=BETA)

    n_probes   = min(N_PROBE, N)
    probe_seed = seed * 1000 + n_idx * 100 + s_idx * 10
    rng_probe  = torch.Generator()
    rng_probe.manual_seed(probe_seed)
    probe_indices = torch.randperm(N, generator=rng_probe)[:n_probes].tolist()

    return X, stored, hop, probe_indices


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: White-box attack grid
# ─────────────────────────────────────────────────────────────────────────────

def stage1_whitebox(
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[list[dict], dict, dict]:
    """
    Returns
    -------
    wb_rows      : per-probe CSV rows
    wb_cell_data : (N, strategy, seed) -> {success_rate, mean_l2, mean_damage}
    headline     : data for N=100 class-balanced seed=42 (figures 3,5)
    """
    print("\n=== Stage 1: White-box attacks ===")
    attacker   = WhiteBoxOnePixelAttacker()
    wb_rows    = []
    wb_cell_data: dict = {}
    headline   = {"X": None, "stored": None, "hop": None, "probes": []}

    for n_idx, N in enumerate(N_VALUES):
        for s_idx, strategy in enumerate(STRATEGIES):
            for seed in SEEDS:
                X, stored, hop, probe_indices = build_cell(
                    images, labels, N, n_idx, s_idx, strategy, seed
                )
                cell_succ, cell_l2, cell_dmg = [], [], []

                for j, true_idx in enumerate(probe_indices):
                    query  = stored[true_idx]
                    result = attacker.attack(query, true_idx, hop)

                    wb_rows.append({
                        "seed":           seed,
                        "N":              N,
                        "strategy":       strategy,
                        "probe_idx":      j,
                        "true_index":     true_idx,
                        "attack_success": int(result["success"]),
                        "pixel_i":        result["pixel_i"],
                        "pixel_j":        result["pixel_j"],
                        "pixel_value":    round(result["pixel_value"],    4),
                        "original_value": round(result["original_value"], 4),
                        "perturbation_l2": round(result["perturbation_l2"], 4),
                        "retrieved_index": result["retrieved_index"],
                        "evaluations":    result["evaluations"],
                    })
                    cell_succ.append(float(result["success"]))
                    cell_l2.append(result["perturbation_l2"])
                    cell_dmg.append(1.0 - result["cosine_to_true"])

                    # Collect headline examples (N=100, class_balanced, seed=42)
                    if N == 100 and strategy == "class_balanced" and seed == 42:
                        headline["probes"].append({
                            "true_idx": true_idx,
                            "result":   result,
                        })

                wb_cell_data[(N, strategy, seed)] = {
                    "success_rate": float(np.mean(cell_succ)),
                    "mean_l2":      float(np.mean(cell_l2)),
                    "mean_damage":  float(np.mean(cell_dmg)),
                }

                # Save X/stored/hop for headline cell
                if N == 100 and strategy == "class_balanced" and seed == 42:
                    headline["X"]      = X
                    headline["stored"] = stored
                    headline["hop"]    = hop

                sr = wb_cell_data[(N, strategy, seed)]["success_rate"]
                print(
                    f"  N={N:>5} {strategy:<16} seed={seed}  "
                    f"success={sr:.2f}  "
                    f"mean_L2={wb_cell_data[(N, strategy, seed)]['mean_l2']:.3f}"
                )

    return wb_rows, wb_cell_data, headline


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Random-noise-equivalent baseline
# ─────────────────────────────────────────────────────────────────────────────

def stage2_rne(
    images: torch.Tensor,
    labels: torch.Tensor,
    wb_cell_data: dict,
) -> list[dict]:
    """Gaussian noise at σ ∈ RNE_SIGMAS on the same probes as stage 1."""
    print("\n=== Stage 2: Random-noise-equivalent baseline ===")
    rne_rows: list[dict] = []

    for n_idx, N in enumerate(N_VALUES):
        for s_idx, strategy in enumerate(STRATEGIES):
            for seed in SEEDS:
                X, stored, hop, probe_indices = build_cell(
                    images, labels, N, n_idx, s_idx, strategy, seed
                )
                mean_attack_l2 = wb_cell_data[(N, strategy, seed)]["mean_l2"]

                for sigma in RNE_SIGMAS:
                    expected_l2 = sigma * (784 ** 0.5)   # ~28σ
                    failures = 0
                    for j, true_idx in enumerate(probe_indices):
                        noisy     = add_gaussian_noise(stored[true_idx], sigma=sigma, seed=seed + j)
                        retrieved = hop.retrieve(noisy, steps=1)
                        if not retrieval_accuracy(retrieved, X, true_idx):
                            failures += 1

                    rne_rows.append({
                        "seed":            seed,
                        "N":               N,
                        "strategy":        strategy,
                        "sigma":           sigma,
                        "mean_l2":         round(expected_l2, 4),
                        "attack_mean_l2":  round(mean_attack_l2, 4),
                        "failure_rate":    failures / len(probe_indices),
                    })

            # Quick progress line per (N, strategy)
            r = [r["failure_rate"] for r in rne_rows
                 if r["N"] == N and r["strategy"] == strategy and r["sigma"] == 0.05]
            if r:
                print(f"  N={N:>5} {strategy:<16} sigma=0.05 fail={np.mean(r):.3f}")

    return rne_rows


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: DE black-box at headline cell
# ─────────────────────────────────────────────────────────────────────────────

def stage3_de(
    images: torch.Tensor,
    labels: torch.Tensor,
) -> list[dict]:
    """DE black-box attack at N=100, class-balanced across 5 seeds."""
    print("\n=== Stage 3: DE black-box attacks (N=100 class-balanced) ===")
    bb_rows: list[dict] = []
    n_idx = N_VALUES.index(100)
    s_idx = STRATEGIES.index("class_balanced")

    for seed in SEEDS:
        X, stored, hop, probe_indices = build_cell(
            images, labels, 100, n_idx, s_idx, "class_balanced", seed
        )
        print(f"  seed={seed}: {len(probe_indices)} probes")

        for j, true_idx in enumerate(probe_indices):
            if j % 10 == 0:
                print(f"    probe {j}/{len(probe_indices)} ...")

            de     = DEBlackBoxOnePixelAttacker(seed=seed * 1000 + j)
            query  = stored[true_idx]
            result = de.attack(query, true_idx, hop)

            bb_rows.append({
                "seed":            seed,
                "N":               100,
                "strategy":        "class_balanced",
                "probe_idx":       j,
                "true_index":      true_idx,
                "attack_success":  int(result["success"]),
                "pixel_i":         result["pixel_i"],
                "pixel_j":         result["pixel_j"],
                "pixel_value":     round(result["pixel_value"],    4),
                "original_value":  round(result["original_value"], 4),
                "perturbation_l2": round(result["perturbation_l2"], 4),
                "retrieved_index": result["retrieved_index"],
                "evaluations":     result["evaluations"],
            })

        seed_sr = np.mean([r["attack_success"] for r in bb_rows if r["seed"] == seed])
        print(f"    seed={seed} done  success_rate={seed_sr:.2f}")

    return bb_rows


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Vulnerability maps
# ─────────────────────────────────────────────────────────────────────────────

def stage4_vuln(
    images: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """Compute 28×28 heatmaps for N ∈ {10, 100, 500}, class-balanced, seed=42."""
    print("\n=== Stage 4: Vulnerability maps ===")
    attacker = WhiteBoxOnePixelAttacker()
    vuln_maps: dict = {}
    vuln_ns   = [10, 100, 500]
    s_idx     = STRATEGIES.index("class_balanced")

    for N in vuln_ns:
        n_idx = N_VALUES.index(N)
        X, stored, hop, probe_indices = build_cell(
            images, labels, N, n_idx, s_idx, "class_balanced", 42
        )
        heatmap, per_pat = compute_vulnerability_map(X, hop, attacker, probe_indices)
        vuln_maps[N] = {"heatmap": heatmap, "per_pattern": per_pat}
        print(f"  N={N}: max pixel freq={float(heatmap.max()):.3f}")

    return vuln_maps


# ─────────────────────────────────────────────────────────────────────────────
# Figure helpers
# ─────────────────────────────────────────────────────────────────────────────

def _agg(wb_cell_data: dict, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (means_random, stds_random, means_cb, stds_cb) over N_VALUES."""
    mr, sr, mc, sc = [], [], [], []
    for N in N_VALUES:
        vals_r = [wb_cell_data[(N, "random",         seed)][metric] for seed in SEEDS]
        vals_c = [wb_cell_data[(N, "class_balanced", seed)][metric] for seed in SEEDS]
        mr.append(np.mean(vals_r)); sr.append(np.std(vals_r, ddof=1))
        mc.append(np.mean(vals_c)); sc.append(np.std(vals_c, ddof=1))
    return np.array(mr), np.array(sr), np.array(mc), np.array(sc)


def _show_img(ax: plt.Axes, vec: torch.Tensor, title: str = "") -> None:
    img = vec.detach().cpu().float().clamp(0, 1).view(28, 28).numpy()
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=6)
    ax.axis("off")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: attack effectiveness grid
# ─────────────────────────────────────────────────────────────────────────────

def save_attack_grid(wb_cell_data: dict, fig_path: str) -> None:
    metrics = [
        ("success_rate", "Attack success rate",   "steelblue"),
        ("mean_l2",      "Mean perturbation L2",  "darkorange"),
        ("mean_damage",  "Cosine damage",         "seagreen"),
    ]
    colors   = {"random": "steelblue", "class_balanced": "darkorange"}
    slabels  = {"random": "Random",    "class_balanced": "Class-balanced"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax_idx, (metric, ylabel, _) in enumerate(metrics):
        ax = axes[ax_idx]
        mr, sr, mc, sc = _agg(wb_cell_data, metric)

        ax.errorbar(N_VALUES, mr, yerr=sr, fmt="o-", color=colors["random"],
                    label=slabels["random"], lw=1.8, ms=5, capsize=4)
        ax.errorbar(N_VALUES, mc, yerr=sc, fmt="s-", color=colors["class_balanced"],
                    label=slabels["class_balanced"], lw=1.8, ms=5, capsize=4)

        ax.set_xscale("log")
        ax.set_xticks(N_VALUES)
        ax.set_xticklabels([str(n) for n in N_VALUES], fontsize=8)
        ax.set_xlabel("N (stored patterns)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        if metric == "success_rate":
            ax.set_ylim(-0.05, 1.05)

    fig.suptitle(
        "White-box one-pixel attack effectiveness across operating regimes\n"
        "(mean +/- std, 5 seeds 42-46)",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: attack vs random noise at equal L2
# ─────────────────────────────────────────────────────────────────────────────

def save_attack_vs_random(
    wb_cell_data: dict,
    rne_rows: list[dict],
    fig_path: str,
) -> None:
    # Headline cell: N=100, class-balanced
    attack_sr   = np.mean([wb_cell_data[(100, "class_balanced", s)]["success_rate"] for s in SEEDS])
    attack_l2   = np.mean([wb_cell_data[(100, "class_balanced", s)]["mean_l2"]      for s in SEEDS])
    attack_l2_std = np.std([wb_cell_data[(100, "class_balanced", s)]["mean_l2"]     for s in SEEDS], ddof=1)

    # Random noise: average failure rate across seeds for headline cell
    rne_hc = [r for r in rne_rows if r["N"] == 100 and r["strategy"] == "class_balanced"]
    sigma_to_fr: dict = {}
    for sigma in RNE_SIGMAS:
        frs = [r["failure_rate"] for r in rne_hc if r["sigma"] == sigma]
        sigma_to_fr[sigma] = float(np.mean(frs))

    rne_l2s = [sigma * (784 ** 0.5) for sigma in RNE_SIGMAS]
    rne_frs = [sigma_to_fr[s] for s in RNE_SIGMAS]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(rne_l2s, rne_frs, "o-", color="steelblue", lw=2, ms=7,
            label="Gaussian noise (random)")
    ax.annotate(
        f"σ={RNE_SIGMAS[0]}", xy=(rne_l2s[0], rne_frs[0]),
        xytext=(rne_l2s[0] * 1.15, rne_frs[0] + 0.03), fontsize=7, color="steelblue",
    )
    ax.annotate(
        f"σ={RNE_SIGMAS[-1]}", xy=(rne_l2s[-1], rne_frs[-1]),
        xytext=(rne_l2s[-1] * 0.7, rne_frs[-1] + 0.03), fontsize=7, color="steelblue",
    )

    ax.scatter([attack_l2], [attack_sr], color="crimson", s=120, zorder=5,
               label="One-pixel attack (adversarial)")
    ax.annotate(
        f"Attack\nL2={attack_l2:.2f}\nSR={attack_sr:.2f}",
        xy=(attack_l2, attack_sr),
        xytext=(attack_l2 * 1.1, attack_sr - 0.08),
        fontsize=8, color="crimson",
        arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2),
    )
    ax.axvline(attack_l2, color="crimson", linestyle=":", lw=1, alpha=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("L2 perturbation magnitude (log scale)", fontsize=10)
    ax.set_ylabel("Retrieval failure rate", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "One-pixel attack vs. equivalent-magnitude random noise\n"
        "N=100 class-balanced, beta=8.0, 5-seed mean",
        fontsize=9,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: white-box vs DE comparison
# ─────────────────────────────────────────────────────────────────────────────

def save_wb_vs_de(
    wb_cell_data: dict,
    bb_rows: list[dict],
    fig_path: str,
) -> None:
    wb_sr   = np.mean([wb_cell_data[(100, "class_balanced", s)]["success_rate"] for s in SEEDS])
    wb_l2   = np.mean([wb_cell_data[(100, "class_balanced", s)]["mean_l2"]      for s in SEEDS])
    wb_sr_std = np.std([wb_cell_data[(100, "class_balanced", s)]["success_rate"] for s in SEEDS], ddof=1)
    wb_l2_std = np.std([wb_cell_data[(100, "class_balanced", s)]["mean_l2"]      for s in SEEDS], ddof=1)

    # DE aggregated over seeds
    de_by_seed = {}
    for seed in SEEDS:
        seed_rows = [r for r in bb_rows if r["seed"] == seed]
        de_by_seed[seed] = {
            "success_rate": np.mean([r["attack_success"] for r in seed_rows]),
            "mean_l2":      np.mean([r["perturbation_l2"] for r in seed_rows]),
        }
    de_sr     = np.mean([de_by_seed[s]["success_rate"] for s in SEEDS])
    de_l2     = np.mean([de_by_seed[s]["mean_l2"]      for s in SEEDS])
    de_sr_std = np.std( [de_by_seed[s]["success_rate"] for s in SEEDS], ddof=1)
    de_l2_std = np.std( [de_by_seed[s]["mean_l2"]      for s in SEEDS], ddof=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.array([0.0, 1.0, 3.0, 4.0])
    heights = [wb_sr, de_sr, wb_l2, de_l2]
    errs    = [wb_sr_std, de_sr_std, wb_l2_std, de_l2_std]
    colors_b = ["#2166ac", "#4dac26", "#2166ac", "#4dac26"]
    labels_b = ["White-box", "DE black-box", "White-box", "DE black-box"]

    bars = ax.bar(x, heights, width=0.7, color=colors_b, alpha=0.85, edgecolor="white")
    ax.errorbar(x, heights, yerr=errs, fmt="none", ecolor="black", capsize=5, elinewidth=1.5)

    for bar, h in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_b, fontsize=9)
    ax.set_ylabel("Value", fontsize=10)
    ax.set_title(
        "White-box (analytical) vs. black-box (Differential Evolution)\n"
        "one-pixel attacks at N=100 class-balanced  (mean +/- std, 5 seeds)",
        fontsize=9,
    )
    # Group labels
    ax.text(0.5, -0.12, "Success rate", ha="center", transform=ax.get_xaxis_transform(),
            fontsize=9, color="#555")
    ax.text(3.5, -0.12, "Mean L2",      ha="center", transform=ax.get_xaxis_transform(),
            fontsize=9, color="#555")
    ax.axvline(2.0, color="gray", linestyle="--", lw=1, alpha=0.4)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elems = [Patch(color="#2166ac", label="White-box"), Patch(color="#4dac26", label="DE black-box")]
    ax.legend(handles=legend_elems, fontsize=9)

    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: vulnerability maps
# ─────────────────────────────────────────────────────────────────────────────

def save_vuln_maps(vuln_maps: dict, fig_path: str) -> None:
    vuln_ns = [10, 100, 500]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax_idx, N in enumerate(vuln_ns):
        ax  = axes[ax_idx]
        hm  = vuln_maps[N]["heatmap"]  # (28, 28) tensor
        hm_np = hm.cpu().numpy()
        im    = ax.imshow(hm_np, cmap="magma", vmin=0, vmax=hm_np.max())
        ax.set_title(f"N={N}, class-balanced\nmax={hm_np.max():.3f}", fontsize=9)
        ax.set_xlabel("Pixel column", fontsize=8)
        if ax_idx == 0:
            ax.set_ylabel("Pixel row", fontsize=8)
        ax.tick_params(labelsize=7)
        plt.colorbar(im, ax=ax, label="Attack frequency", fraction=0.046, pad=0.04)

    fig.suptitle(
        "Per-pixel vulnerability under white-box one-pixel attack\n"
        "(seed=42, probe fraction of probes selecting each pixel)",
        fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: attack examples
# ─────────────────────────────────────────────────────────────────────────────

def save_attack_examples(headline: dict, fig_path: str) -> None:
    """6×4 grid: original | attacked query | retrieved | false stored pattern."""
    stored = headline["stored"]   # (100, 784)
    hop    = headline["hop"]
    probes = headline["probes"]   # list of {true_idx, result}

    # Select 6 successful attacks with diverse digit classes (class = true_idx // 10)
    selected: list[dict] = []
    used_classes: set    = set()
    for p in probes:
        if not p["result"]["success"]:
            continue
        digit_class = p["true_idx"] // 10
        if digit_class not in used_classes:
            selected.append(p)
            used_classes.add(digit_class)
        if len(selected) == 6:
            break

    # Fall back: take any 6 successful attacks
    if len(selected) < 6:
        selected = [p for p in probes if p["result"]["success"]][:6]

    if not selected:
        print("  WARNING: no successful attacks to show in figure 5.")
        return

    n_ex = len(selected)
    fig, axes = plt.subplots(4, n_ex, figsize=(2.5 * n_ex, 11))

    col_titles = ["Original", "Attacked query", "Retrieved", "False pattern"]
    for row in range(4):
        axes[row, 0].set_ylabel(col_titles[row], fontsize=8)

    for col_idx, p in enumerate(selected):
        true_idx   = p["true_idx"]
        result     = p["result"]
        digit_cls  = true_idx // 10

        original   = stored[true_idx]                      # (784,)
        pi, pj     = result["pixel_i"], result["pixel_j"]
        pix_val    = result["pixel_value"]

        # Build attacked query
        attacked = original.clone()
        attacked[pi * 28 + pj] = pix_val

        # Retrieved pattern from network
        retrieved_vec = hop.retrieve(attacked, steps=1)

        # False stored pattern
        false_pat = stored[result["retrieved_index"]]

        # Top title: digit class
        axes[0, col_idx].set_title(f"digit {digit_cls}", fontsize=7)

        _show_img(axes[0, col_idx], original,      "")
        _show_img(axes[1, col_idx], attacked,      "")
        _show_img(axes[2, col_idx], retrieved_vec, "")
        _show_img(axes[3, col_idx], false_pat,     "")

        # Red box around attacked pixel on row 1
        rect = patches.Rectangle(
            (pj - 0.5, pi - 0.5), 1, 1,
            linewidth=1.5, edgecolor="red", facecolor="none",
        )
        axes[1, col_idx].add_patch(rect)

        # Annotation: pixel location and L2
        axes[1, col_idx].set_xlabel(
            f"({pi},{pj}) L2={result['perturbation_l2']:.2f}", fontsize=6
        )

    fig.suptitle(
        "Phase 3: White-box one-pixel attack examples\n"
        "N=100 class-balanced, seed=42  (red box = attacked pixel)",
        fontsize=9,
    )
    plt.tight_layout(h_pad=0.3)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fieldnames})


def save_summary_csv(wb_cell_data: dict, bb_rows: list[dict]) -> None:
    summary_rows = []

    # White-box: aggregate over seeds per (N, strategy)
    for N in N_VALUES:
        for strategy in STRATEGIES:
            srs = [wb_cell_data[(N, strategy, s)]["success_rate"] for s in SEEDS]
            l2s = [wb_cell_data[(N, strategy, s)]["mean_l2"]      for s in SEEDS]
            summary_rows.append({
                "N": N, "strategy": strategy, "attacker_type": "whitebox",
                "attack_success_rate_mean": round(float(np.mean(srs)), 4),
                "attack_success_rate_std":  round(float(np.std(srs, ddof=1)), 4),
                "mean_l2_mean": round(float(np.mean(l2s)), 4),
                "mean_l2_std":  round(float(np.std(l2s, ddof=1)), 4),
                "n_seeds": len(SEEDS),
            })

    # DE black-box: headline cell only
    if bb_rows:
        de_by_seed = {}
        for seed in SEEDS:
            sr = [r for r in bb_rows if r["seed"] == seed]
            de_by_seed[seed] = {
                "success_rate": np.mean([r["attack_success"] for r in sr]),
                "mean_l2":      np.mean([r["perturbation_l2"] for r in sr]),
            }
        srs = [de_by_seed[s]["success_rate"] for s in SEEDS]
        l2s = [de_by_seed[s]["mean_l2"]      for s in SEEDS]
        summary_rows.append({
            "N": 100, "strategy": "class_balanced", "attacker_type": "de_blackbox",
            "attack_success_rate_mean": round(float(np.mean(srs)), 4),
            "attack_success_rate_std":  round(float(np.std(srs, ddof=1)), 4),
            "mean_l2_mean": round(float(np.mean(l2s)), 4),
            "mean_l2_std":  round(float(np.std(l2s, ddof=1)), 4),
            "n_seeds": len(SEEDS),
        })

    save_csv(
        summary_rows,
        EXP_DIR / "phase3_summary.csv",
        ["N", "strategy", "attacker_type",
         "attack_success_rate_mean", "attack_success_rate_std",
         "mean_l2_mean", "mean_l2_std", "n_seeds"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stdout summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(
    wb_cell_data: dict,
    bb_rows: list[dict],
    rne_rows: list[dict],
    elapsed: float,
) -> None:
    print()
    print("=== PHASE 3 SUMMARY ===")

    # White-box success rate table
    print("\nWhite-box attack success rate (mean +/- std across seeds):")
    print(f"{'N':>8}  {'Random':>20}  {'Class-balanced':>20}")
    print("-" * 54)
    for N in N_VALUES:
        r_vals = [wb_cell_data[(N, "random",         s)]["success_rate"] for s in SEEDS]
        c_vals = [wb_cell_data[(N, "class_balanced", s)]["success_rate"] for s in SEEDS]
        print(f"{N:>8}  {np.mean(r_vals):>7.3f} +/- {np.std(r_vals,ddof=1):>5.3f}  "
              f"{np.mean(c_vals):>7.3f} +/- {np.std(c_vals,ddof=1):>5.3f}")

    # Mean L2 table
    print("\nMean perturbation L2 (mean +/- std):")
    print(f"{'N':>8}  {'Random':>20}  {'Class-balanced':>20}")
    print("-" * 54)
    for N in N_VALUES:
        r_vals = [wb_cell_data[(N, "random",         s)]["mean_l2"] for s in SEEDS]
        c_vals = [wb_cell_data[(N, "class_balanced", s)]["mean_l2"] for s in SEEDS]
        print(f"{N:>8}  {np.mean(r_vals):>7.3f} +/- {np.std(r_vals,ddof=1):>5.3f}  "
              f"{np.mean(c_vals):>7.3f} +/- {np.std(c_vals,ddof=1):>5.3f}")

    # Headline cell
    hl_wb_sr = np.mean([wb_cell_data[(100, "class_balanced", s)]["success_rate"] for s in SEEDS])
    hl_wb_l2 = np.mean([wb_cell_data[(100, "class_balanced", s)]["mean_l2"]      for s in SEEDS])
    hl_wb_ev = 3920.0   # fixed for white-box

    print("\n=== HEADLINE CELL: N=100 class-balanced ===")
    print(f"White-box:    success_rate={hl_wb_sr:.3f}  mean_L2={hl_wb_l2:.3f}  "
          f"evaluations={hl_wb_ev:.0f}")

    if bb_rows:
        hl_de_sr = np.mean([
            np.mean([r["attack_success"] for r in bb_rows if r["seed"] == s])
            for s in SEEDS
        ])
        hl_de_l2 = np.mean([
            np.mean([r["perturbation_l2"] for r in bb_rows if r["seed"] == s])
            for s in SEEDS
        ])
        hl_de_ev = np.mean([r["evaluations"] for r in bb_rows])
        print(f"DE black-box: success_rate={hl_de_sr:.3f}  mean_L2={hl_de_l2:.3f}  "
              f"mean_evals={hl_de_ev:.0f}")

        # Random noise at closest L2 to attack L2
        rne_hc = [r for r in rne_rows if r["N"] == 100 and r["strategy"] == "class_balanced"]
        closest_sigma = min(RNE_SIGMAS, key=lambda s: abs(s * (784**0.5) - hl_wb_l2))
        rne_fr = np.mean([r["failure_rate"] for r in rne_hc if r["sigma"] == closest_sigma])
        print(f"Random noise at sigma={closest_sigma} (L2={closest_sigma*784**0.5:.2f}, "
              f"closest to attack L2={hl_wb_l2:.3f}): failure_rate={rne_fr:.3f}")

        print("\n=== INTERPRETATION ===")
        amplif = hl_wb_sr / max(rne_fr, 1e-6)
        print(f"Headline result: white-box one-pixel attack succeeds {hl_wb_sr*100:.1f}% of "
              f"the time at the headline cell, while equivalent-magnitude random noise "
              f"causes failure only {rne_fr*100:.1f}% of the time. "
              f"This represents a {amplif:.1f}x amplification of damage from adversarial "
              f"pixel selection.")
        wb_eff = hl_de_sr / max(hl_wb_sr, 1e-6)
        print(f"DE black-box attacker achieves {wb_eff*100:.1f}% of white-box effectiveness, "
              f"confirming the threat is realizable without storage-matrix access.")

    print(f"\nTotal runtime: {elapsed:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()

    print("=" * 62)
    print("Phase 3: Adversarial One-Pixel Attacks")
    print(f"Seeds: {SEEDS}  Beta: {BETA}  N_probe: {N_PROBE}")
    print("=" * 62)

    print("\nLoading MNIST ...")
    images, labels = load_mnist_train()

    # ── stages ────────────────────────────────────────────────────────────────
    wb_rows, wb_cell_data, headline = stage1_whitebox(images, labels)

    rne_rows = stage2_rne(images, labels, wb_cell_data)

    bb_rows = stage3_de(images, labels)

    vuln_maps = stage4_vuln(images, labels)

    # ── save CSVs ─────────────────────────────────────────────────────────────
    wb_fields = [
        "seed", "N", "strategy", "probe_idx", "true_index",
        "attack_success", "pixel_i", "pixel_j",
        "pixel_value", "original_value", "perturbation_l2",
        "retrieved_index", "evaluations",
    ]
    rne_fields = ["seed", "N", "strategy", "sigma", "mean_l2", "attack_mean_l2", "failure_rate"]
    bb_fields  = wb_fields

    save_csv(wb_rows,  EXP_DIR / "phase3_whitebox_results.csv",  wb_fields)
    save_csv(bb_rows,  EXP_DIR / "phase3_blackbox_results.csv",  bb_fields)
    save_csv(rne_rows, EXP_DIR / "phase3_random_noise_equivalent.csv", rne_fields)
    save_summary_csv(wb_cell_data, bb_rows)

    n_wb  = len(wb_rows)
    n_bb  = len(bb_rows)
    n_rne = len(rne_rows)
    print(f"\nSaved CSVs: whitebox={n_wb} rows, blackbox={n_bb} rows, rne={n_rne} rows")

    # ── figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures ...")
    save_attack_grid(wb_cell_data, str(FIG_DIR / "phase3_attack_grid.png"))
    print("  Saved: phase3_attack_grid.png")

    save_attack_vs_random(wb_cell_data, rne_rows, str(FIG_DIR / "phase3_attack_vs_random.png"))
    print("  Saved: phase3_attack_vs_random.png")

    if bb_rows:
        save_wb_vs_de(wb_cell_data, bb_rows, str(FIG_DIR / "phase3_whitebox_vs_de.png"))
        print("  Saved: phase3_whitebox_vs_de.png")

    save_vuln_maps(vuln_maps, str(FIG_DIR / "phase3_vulnerability_maps.png"))
    print("  Saved: phase3_vulnerability_maps.png")

    save_attack_examples(headline, str(FIG_DIR / "phase3_attack_examples.png"))
    print("  Saved: phase3_attack_examples.png")

    # ── summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print_summary(wb_cell_data, bb_rows, rne_rows, elapsed)


if __name__ == "__main__":
    main()
