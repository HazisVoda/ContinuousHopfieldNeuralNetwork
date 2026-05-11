"""
Phase 3 closing diagnostics.

Diagnostic C: σ-sweep to find the exact magnitude-matched random-noise comparison.
Diagnostic D: Characterization of the 30 vulnerable probes at the headline cell.

No attack re-runs. Reads existing Phase 3 CSVs + runs clean retrievals only.
Run: python -m experiments.phase3_closing_diagnostics
"""

from __future__ import annotations

import csv
import math
import sys
import time
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

from hopfield.network    import ContinuousHopfield
from hopfield.corruption import add_gaussian_noise
from hopfield.metrics    import retrieval_accuracy
from hopfield.sampling   import sample_random, sample_class_balanced

# ── config — must exactly match Phase 3 ──────────────────────────────────────
SEEDS      = [42, 43, 44, 45, 46]
BETA       = 8.0
N_VALUES   = [10, 50, 100, 500, 1000]
STRATEGIES = ["random", "class_balanced"]
N_PROBE    = 50

WB_MEAN_L2      = 0.113   # headline-cell mean attack L2 (Phase 3)
COND_WB_SUCCESS = 0.031   # conditional WB attack success (Diagnostic B)

SIGMAS_C = [0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.015, 0.020]

DATA_DIR = ROOT / "data"
FIG_DIR  = ROOT / "figures"
EXP_DIR  = ROOT / "experiments"
FIG_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_mnist_train() -> tuple[torch.Tensor, torch.Tensor]:
    ds = torchvision.datasets.MNIST(
        root=str(DATA_DIR), train=True, download=False, transform=T.ToTensor()
    )
    return ds.data.float().view(-1, 784) / 255.0, ds.targets


def build_cell(
    images: torch.Tensor, labels: torch.Tensor,
    N: int, n_idx: int, s_idx: int, strategy: str, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, ContinuousHopfield, list[int]]:
    dataset = (images, labels)
    sampler = sample_random if strategy == "random" else sample_class_balanced
    X, _    = sampler(dataset, N, seed=seed + n_idx)
    stored  = X.T.contiguous()
    hop     = ContinuousHopfield(X, beta=BETA)
    n_p         = min(N_PROBE, N)
    probe_seed  = seed * 1000 + n_idx * 100 + s_idx * 10
    rng         = torch.Generator()
    rng.manual_seed(probe_seed)
    probe_indices = torch.randperm(N, generator=rng)[:n_p].tolist()
    return X, stored, hop, probe_indices


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers (no scipy dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _norm_sf(z: float) -> float:
    """P(Z > z) for standard normal via math.erfc."""
    return math.erfc(z / math.sqrt(2)) / 2


def chi2_pvalue(stat: float, df: int) -> float:
    """Upper-tail p-value for chi-square, Wilson-Hilferty normal approximation."""
    if stat <= 0 or df <= 0:
        return 1.0
    z = ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return max(0.0, min(1.0, _norm_sf(z)))


def mannwhitney_u(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sided Mann-Whitney U, normal approximation with tie averaging."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float("nan"), float("nan")
    combined = np.concatenate([x, y])
    n = len(combined)
    order = np.argsort(combined, kind="stable")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and combined[order[j]] == combined[order[j + 1]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    Rx = ranks[:nx].sum()
    U  = Rx - nx * (nx + 1) / 2.0
    mu    = nx * ny / 2.0
    sigma = math.sqrt(nx * ny * (nx + ny + 1) / 12.0)
    z  = abs(U - mu) / sigma if sigma > 0 else 0.0
    p  = min(1.0, 2.0 * _norm_sf(z))
    return float(U), float(p)


# ─────────────────────────────────────────────────────────────────────────────
# Logistic regression with balanced class weights, 5-fold CV
# Returns balanced accuracy = (TPR + TNR) / 2  (50% baseline, not 88%)
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _logistic_fit(
    X: np.ndarray, y: np.ndarray, sw: np.ndarray,
    n_iter: int = 2000, lr: float = 0.05, reg: float = 0.1,
) -> np.ndarray:
    n, d = X.shape
    Xa = np.column_stack([X, np.ones(n)])
    w  = np.zeros(d + 1)
    for _ in range(n_iter):
        pred = _sigmoid(Xa @ w)
        grad = Xa.T @ ((pred - y) * sw) / n
        grad[:-1] += reg * w[:-1]
        w -= lr * grad
    return w


def logistic_cv(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> tuple[float, float]:
    """5-fold CV with balanced class weights; returns mean ± std balanced accuracy."""
    n   = len(y)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    fold_size = n // n_folds
    bal_accs  = []

    for fold in range(n_folds):
        te = idx[fold * fold_size:(fold + 1) * fold_size]
        tr = np.concatenate([idx[:fold * fold_size], idx[(fold + 1) * fold_size:]])
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]

        mu = Xtr.mean(0);  sd = Xtr.std(0) + 1e-8
        Xtr = (Xtr - mu) / sd;  Xte = (Xte - mu) / sd

        n_pos = ytr.sum();  n_neg = len(ytr) - n_pos
        w_pos = len(ytr) / (2.0 * max(n_pos, 1))
        w_neg = len(ytr) / (2.0 * max(n_neg, 1))
        sw = np.where(ytr == 1, w_pos, w_neg)

        w    = _logistic_fit(Xtr, ytr, sw)
        Xta  = np.column_stack([Xte, np.ones(len(Xte))])
        pred = (_sigmoid(Xta @ w) >= 0.5).astype(int)

        pos = yte == 1;  neg = yte == 0
        tpr = (pred[pos] == 1).mean() if pos.any() else 0.0
        tnr = (pred[neg] == 0).mean() if neg.any() else 0.0
        bal_accs.append((tpr + tnr) / 2.0)

    return float(np.mean(bal_accs)), float(np.std(bal_accs, ddof=1))


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic C: magnitude-matched σ sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostic_c(
    images: torch.Tensor, labels: torch.Tensor, exp_dir: Path
) -> list[dict]:
    n_idx = N_VALUES.index(100)
    s_idx = STRATEGIES.index("class_balanced")
    rows: list[dict] = []

    for seed in SEEDS:
        X, stored, hop, probe_indices = build_cell(
            images, labels, 100, n_idx, s_idx, "class_balanced", seed
        )
        # Clean baseline for this seed
        clean_ok = []
        for j, true_idx in enumerate(probe_indices):
            ret = hop.retrieve(stored[true_idx], steps=1)
            clean_ok.append(retrieval_accuracy(ret, X, true_idx))
        n_clean = sum(clean_ok)
        n_excl  = len(probe_indices) - n_clean

        for sigma in SIGMAS_C:
            noise_l2s, raw_fails, cond_fails = [], [], []
            for j, true_idx in enumerate(probe_indices):
                q     = stored[true_idx]
                noisy = add_gaussian_noise(q, sigma=sigma, seed=seed + j)
                noise_l2s.append(float((noisy - q).norm()))
                ret  = hop.retrieve(noisy, steps=1)
                fail = not retrieval_accuracy(ret, X, true_idx)
                raw_fails.append(int(fail))
                if clean_ok[j]:
                    cond_fails.append(int(fail))

            n_p   = len(probe_indices)
            rows.append({
                "seed":                   seed,
                "sigma":                  sigma,
                "mean_noise_l2":          round(float(np.mean(noise_l2s)), 5),
                "raw_failure_rate":       round(sum(raw_fails) / n_p, 4),
                "conditional_failure_rate": round(
                    sum(cond_fails) / n_clean if n_clean > 0 else 0.0, 4),
                "n_baseline_excluded":    n_excl,
            })

    fields = ["seed", "sigma", "mean_noise_l2",
              "raw_failure_rate", "conditional_failure_rate", "n_baseline_excluded"]
    write_csv(rows, exp_dir / "phase3_diag_c_matched_sigma.csv", fields)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic D: vulnerable probe characterization
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostic_d(
    images: torch.Tensor, labels: torch.Tensor, exp_dir: Path
) -> dict:
    paired_rows = read_csv_rows(exp_dir / "phase3_diag_a_paired.csv")
    vulnerable_set = {
        (int(r["seed"]), int(r["probe_idx"]))
        for r in paired_rows if int(r["wb_success"]) == 1
    }

    n_idx = N_VALUES.index(100)
    s_idx = STRATEGIES.index("class_balanced")
    feature_rows: list[dict] = []

    for seed in SEEDS:
        X, stored, hop, probe_indices = build_cell(
            images, labels, 100, n_idx, s_idx, "class_balanced", seed
        )
        # Pairwise cosine matrix (100×100), diagonal zeroed out
        Xn = X.numpy()
        norms = np.linalg.norm(Xn, axis=0, keepdims=True)
        Xnn   = Xn / (norms + 1e-8)
        cos_mat = Xnn.T @ Xnn               # (100, 100)
        np.fill_diagonal(cos_mat, -1.0)     # exclude self

        for j, true_idx in enumerate(probe_indices):
            pat  = stored[true_idx].numpy()
            feature_rows.append({
                "seed":               seed,
                "probe_idx":          j,
                "digit_class":        true_idx // 10,
                "vulnerable":         int((seed, j) in vulnerable_set),
                "max_neighbor_cosine": round(float(cos_mat[true_idx].max()), 5),
                "mean_intensity":     round(float(pat.mean()), 5),
                "intensity_std":      round(float(pat.std()),  5),
            })

    feat_fields = ["seed", "probe_idx", "digit_class", "vulnerable",
                   "max_neighbor_cosine", "mean_intensity", "intensity_std"]
    write_csv(feature_rows, exp_dir / "phase3_diag_d_probe_features.csv", feat_fields)

    v_rows  = [r for r in feature_rows if r["vulnerable"] == 1]
    nv_rows = [r for r in feature_rows if r["vulnerable"] == 0]
    n_vuln  = len(v_rows)

    # ── Chi-square: class distribution ───────────────────────────────────────
    class_counts = np.zeros(10, dtype=float)
    for r in v_rows:
        class_counts[r["digit_class"]] += 1
    exp_cls  = n_vuln / 10.0
    chi2_s   = float(np.sum((class_counts - exp_cls) ** 2 / exp_cls))
    chi2_p   = chi2_pvalue(chi2_s, df=9)
    over_rep  = [i for i in range(10) if class_counts[i] > 1.5 * exp_cls]
    under_rep = [i for i in range(10) if class_counts[i] < 0.5 * exp_cls and exp_cls > 0]

    # ── Mann-Whitney U: max neighbor cosine ───────────────────────────────────
    vc = np.array([r["max_neighbor_cosine"] for r in v_rows])
    nc = np.array([r["max_neighbor_cosine"] for r in nv_rows])
    u_cos, p_cos = mannwhitney_u(vc, nc)
    cos_dir = "higher" if np.median(vc) > np.median(nc) else "lower"

    # ── Mann-Whitney U: mean intensity ────────────────────────────────────────
    vm = np.array([r["mean_intensity"] for r in v_rows])
    nm = np.array([r["mean_intensity"] for r in nv_rows])
    u_mi, p_mi = mannwhitney_u(vm, nm)

    # ── Mann-Whitney U: intensity std ─────────────────────────────────────────
    vs = np.array([r["intensity_std"] for r in v_rows])
    ns = np.array([r["intensity_std"] for r in nv_rows])
    u_sd, p_sd = mannwhitney_u(vs, ns)

    # ── Logistic regression (balanced accuracy) ───────────────────────────────
    X_lr = np.array([[r["max_neighbor_cosine"], r["mean_intensity"], r["intensity_std"]]
                     for r in feature_rows], dtype=float)
    y_lr = np.array([r["vulnerable"] for r in feature_rows], dtype=float)
    lr_mean, lr_std = logistic_cv(X_lr, y_lr)

    # ── Summary CSV ───────────────────────────────────────────────────────────
    sum_rows = [
        {"analysis_type": "chi2_class_distribution",
         "vulnerable_stat": str(class_counts.tolist()),
         "nonvulnerable_stat": f"expected_per_class={exp_cls:.1f}",
         "test_statistic": round(chi2_s, 4), "p_value": round(chi2_p, 4),
         "interpretation": ("non-uniform class bias" if chi2_p < 0.05
                            else "no significant class bias")},
        {"analysis_type": "mannwhitney_neighbor_cosine",
         "vulnerable_stat": round(float(np.median(vc)), 4),
         "nonvulnerable_stat": round(float(np.median(nc)), 4),
         "test_statistic": round(u_cos, 2), "p_value": round(p_cos, 4),
         "interpretation": (f"vulnerable {cos_dir} (p<0.05)" if p_cos < 0.05
                            else "no significant difference")},
        {"analysis_type": "mannwhitney_mean_intensity",
         "vulnerable_stat": round(float(np.median(vm)), 4),
         "nonvulnerable_stat": round(float(np.median(nm)), 4),
         "test_statistic": round(u_mi, 2), "p_value": round(p_mi, 4),
         "interpretation": ("sig. diff. (p<0.05)" if p_mi < 0.05
                            else "no significant difference")},
        {"analysis_type": "mannwhitney_intensity_std",
         "vulnerable_stat": round(float(np.median(vs)), 4),
         "nonvulnerable_stat": round(float(np.median(ns)), 4),
         "test_statistic": round(u_sd, 2), "p_value": round(p_sd, 4),
         "interpretation": ("sig. diff. (p<0.05)" if p_sd < 0.05
                            else "no significant difference")},
        {"analysis_type": "logistic_cv_balanced_accuracy",
         "vulnerable_stat": round(lr_mean, 4),
         "nonvulnerable_stat": round(lr_std, 4),
         "test_statistic": float("nan"), "p_value": float("nan"),
         "interpretation": (
             "predictable (bal.acc>0.70)" if lr_mean > 0.70 else
             "partially predictable (0.55<bal.acc<=0.70)" if lr_mean > 0.55 else
             "not predictable (bal.acc<=0.55)"
         )},
    ]
    sum_fields = ["analysis_type", "vulnerable_stat", "nonvulnerable_stat",
                  "test_statistic", "p_value", "interpretation"]
    write_csv(sum_rows, exp_dir / "phase3_diag_d_summary.csv", sum_fields)

    return {
        "feature_rows": feature_rows, "n_vuln": n_vuln,
        "class_counts": class_counts, "exp_cls": exp_cls,
        "chi2_s": chi2_s, "chi2_p": chi2_p,
        "over_rep": over_rep, "under_rep": under_rep,
        "vc": vc, "nc": nc, "u_cos": u_cos, "p_cos": p_cos, "cos_dir": cos_dir,
        "vm": vm, "nm": nm, "u_mi":  u_mi,  "p_mi":  p_mi,
        "vs": vs, "ns": ns, "u_sd":  u_sd,  "p_sd":  p_sd,
        "lr_mean": lr_mean, "lr_std": lr_std,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figure C: σ sweep
# ─────────────────────────────────────────────────────────────────────────────

def save_figure_c(c_rows: list[dict], best_sigma: float, fig_path: str) -> None:
    sigma_vals  = SIGMAS_C
    cond_means, cond_stds, mean_l2s = [], [], []
    for sigma in sigma_vals:
        sub = [r for r in c_rows if r["sigma"] == sigma]
        cond_means.append(np.mean([r["conditional_failure_rate"] for r in sub]))
        cond_stds.append(np.std( [r["conditional_failure_rate"] for r in sub], ddof=1))
        mean_l2s.append(np.mean([r["mean_noise_l2"] for r in sub]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(sigma_vals, cond_means, yerr=cond_stds, fmt="o-",
                color="steelblue", lw=2, ms=6, capsize=4,
                label="Conditional random-noise failure (5-seed mean +/- std)")
    ax.axhline(COND_WB_SUCCESS, color="crimson", linestyle="--", lw=1.8,
               label=f"Conditional WB attack success ({COND_WB_SUCCESS*100:.1f}%)")
    ax.axvline(best_sigma, color="darkorange", linestyle=":", lw=1.8,
               label=f"Best-matched σ = {best_sigma:.3f} (L2 ≈ {best_sigma*28:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("σ (Gaussian noise std, log scale)", fontsize=10)
    ax.set_ylabel("Conditional retrieval failure rate", fontsize=10)
    ax.set_ylim(-0.01, max(cond_means) * 1.5 + 0.05)
    ax.set_title(
        "Conditional random-noise failure vs σ, with magnitude-matched WB attack reference\n"
        "N=100 class-balanced, conditional on clean-retrievable probes",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    # Annotate each point with mean L2
    for s, cm, ml2 in zip(sigma_vals, cond_means, mean_l2s):
        ax.annotate(f"L2={ml2:.3f}", xy=(s, cm),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=6, color="#555")
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure D-classes: vulnerable probe class distribution
# ─────────────────────────────────────────────────────────────────────────────

def save_figure_d_classes(d: dict, fig_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(10)
    bars = ax.bar(x, d["class_counts"], color="steelblue", alpha=0.8, edgecolor="white")
    ax.axhline(d["exp_cls"], color="crimson", linestyle="--", lw=1.8,
               label=f"Expected (uniform): {d['exp_cls']:.1f}")
    for i in d["over_rep"]:
        bars[i].set_color("darkorange")
    for i in d["under_rep"]:
        bars[i].set_color("lightgray")
        bars[i].set_edgecolor("gray")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(10)], fontsize=10)
    ax.set_xlabel("Digit class", fontsize=10)
    ax.set_ylabel("Vulnerable probe count", fontsize=10)
    p_str = f"{d['chi2_p']:.3f}" if d["chi2_p"] >= 0.001 else "<0.001"
    ax.set_title(
        f"Vulnerable probe distribution by digit class (n={d['n_vuln']})\n"
        f"chi-sq={d['chi2_s']:.2f}, p={p_str}  "
        f"| orange=over-represented, gray=under-represented",
        fontsize=9,
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure D-features: violin/box plots of probe features
# ─────────────────────────────────────────────────────────────────────────────

def save_figure_d_features(d: dict, fig_path: str) -> None:
    panels = [
        ("max_neighbor_cosine", "Max neighbor cosine",   d["vc"], d["nc"], d["p_cos"]),
        ("mean_intensity",      "Mean pixel intensity",  d["vm"], d["nm"], d["p_mi"]),
        ("intensity_std",       "Pixel intensity std",   d["vs"], d["ns"], d["p_sd"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    colors = {"Vulnerable": "#d6604d", "Non-vulnerable": "#4393c3"}

    for ax, (_, ylabel, vdata, ndata, pval) in zip(axes, panels):
        bp = ax.boxplot(
            [vdata, ndata],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", lw=2),
        )
        bp["boxes"][0].set_facecolor(colors["Vulnerable"])
        bp["boxes"][0].set_alpha(0.75)
        bp["boxes"][1].set_facecolor(colors["Non-vulnerable"])
        bp["boxes"][1].set_alpha(0.75)
        # Overlay jitter for vulnerable (small n)
        xj = np.random.RandomState(0).uniform(0.75, 1.25, len(vdata))
        ax.scatter(xj, vdata, color=colors["Vulnerable"], alpha=0.5, s=15, zorder=3)

        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Vulnerable\n(n=30)", f"Non-vuln.\n(n={len(ndata)})"],
                           fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(ylabel, fontsize=9)
        p_str = f"p={pval:.3f}" if pval >= 0.001 else "p<0.001"
        ax.annotate(p_str, xy=(0.5, 0.97), xycoords="axes fraction",
                    ha="center", va="top", fontsize=9,
                    color="darkred" if pval < 0.05 else "#555")
        ax.grid(True, axis="y", alpha=0.2)

    # Legend
    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(color=colors["Vulnerable"],     label="Vulnerable"),
                 Patch(color=colors["Non-vulnerable"], label="Non-vulnerable")],
        loc="upper right", fontsize=9,
    )
    fig.suptitle(
        "Probe feature distributions: vulnerable vs non-vulnerable\n"
        "N=100 class-balanced, 250 probes (30 vulnerable), 5 seeds",
        fontsize=9,
    )
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Stdout summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(c_rows: list[dict], d: dict) -> None:
    # ── Aggregate C ───────────────────────────────────────────────────────────
    sigma_agg: dict = {}
    for sigma in SIGMAS_C:
        sub = [r for r in c_rows if r["sigma"] == sigma]
        sigma_agg[sigma] = {
            "mean_l2":  float(np.mean([r["mean_noise_l2"]           for r in sub])),
            "cond_m":   float(np.mean([r["conditional_failure_rate"] for r in sub])),
            "cond_s":   float(np.std( [r["conditional_failure_rate"] for r in sub], ddof=1)),
        }
    best_sigma = min(SIGMAS_C, key=lambda s: abs(sigma_agg[s]["mean_l2"] - WB_MEAN_L2))
    matched_fr_m = sigma_agg[best_sigma]["cond_m"]
    matched_fr_s = sigma_agg[best_sigma]["cond_s"]
    sharpened    = COND_WB_SUCCESS / max(matched_fr_m, 1e-6)

    print("\n=== DIAGNOSTIC C: Magnitude-matched random noise ===\n")
    print(f"Best-matched σ: {best_sigma:.3f}  "
          f"(target L2 = {WB_MEAN_L2:.3f}, actual mean L2 = {sigma_agg[best_sigma]['mean_l2']:.4f})")
    print(f"Conditional random-noise failure at matched σ: "
          f"{matched_fr_m:.3f} +/- {matched_fr_s:.3f}")
    print(f"Conditional WB attack success: {COND_WB_SUCCESS*100:.1f}%")
    print(f"Sharpened amplification factor: {sharpened:.1f}x\n")
    print("Full sweep (conditional failure rate, mean +/- std across seeds):")
    print(f"{'σ':>8}  {'Mean L2':>10}  {'Cond. failure':>18}")
    print("-" * 42)
    for sigma in SIGMAS_C:
        a  = sigma_agg[sigma]
        mk = " <--" if sigma == best_sigma else ""
        print(f"{sigma:>8.3f}  {a['mean_l2']:>10.4f}  "
              f"{a['cond_m']:>7.4f} +/- {a['cond_s']:>5.4f}{mk}")

    # ── Diagnostic D ──────────────────────────────────────────────────────────
    print("\n=== DIAGNOSTIC D: Vulnerable probe characterization ===\n")

    print("Class distribution of vulnerable probes:")
    for cls in range(10):
        cnt = int(d["class_counts"][cls])
        pct = 100.0 * cnt / d["n_vuln"] if d["n_vuln"] > 0 else 0
        exp_pct = 100.0 / 10
        print(f"  Digit {cls}: {cnt}  ({pct:.1f}%, expected {exp_pct:.1f}%)")
    p_str = f"{d['chi2_p']:.3f}" if d["chi2_p"] >= 0.001 else "<0.001"
    print(f"  Chi-square: {d['chi2_s']:.3f}, p = {p_str}")
    if d["chi2_p"] < 0.05:
        over  = ", ".join(str(i) for i in d["over_rep"])  or "none"
        under = ", ".join(str(i) for i in d["under_rep"]) or "none"
        print(f"  Verdict: significantly non-uniform class distribution.")
        print(f"  Over-represented: {over}.  Under-represented: {under}.")
    else:
        print("  Verdict: no significant class bias.")

    print("\nNearest-neighbor cosine (vulnerable vs non-vulnerable):")
    print(f"  Vulnerable median:     {np.median(d['vc']):.4f}")
    print(f"  Non-vulnerable median: {np.median(d['nc']):.4f}")
    cp = f"{d['p_cos']:.4f}" if d["p_cos"] >= 0.0001 else "<0.0001"
    print(f"  Mann-Whitney U = {d['u_cos']:.1f}, p = {cp}")
    if d["p_cos"] < 0.05:
        print(f"  Verdict: vulnerable probes have significantly {d['cos_dir']} neighbor cosine.")
    else:
        print("  Verdict: no significant difference in neighbor cosine.")

    print("\nPixel intensity (vulnerable vs non-vulnerable):")
    mp = f"{d['p_mi']:.4f}" if d["p_mi"] >= 0.0001 else "<0.0001"
    sp = f"{d['p_sd']:.4f}" if d["p_sd"] >= 0.0001 else "<0.0001"
    print(f"  Mean intensity  — vulnerable median: {np.median(d['vm']):.4f}, "
          f"non-vulnerable median: {np.median(d['nm']):.4f}, p = {mp}")
    print(f"  Intensity std   — vulnerable median: {np.median(d['vs']):.4f}, "
          f"non-vulnerable median: {np.median(d['ns']):.4f}, p = {sp}")

    print("\nLogistic regression prediction (5-fold CV, balanced accuracy, 50% baseline):")
    print(f"  Mean balanced accuracy: {d['lr_mean']:.3f} +/- {d['lr_std']:.3f}")
    if d["lr_mean"] > 0.70:
        lr_verdict = "Vulnerability is predictable from probe geometry alone."
    elif d["lr_mean"] > 0.55:
        lr_verdict = "Probe features partially predict vulnerability."
    else:
        lr_verdict = ("Vulnerability is NOT predictable from simple probe statistics; "
                      "deeper structure (e.g., specific pattern interactions) drives it.")
    print(f"  Verdict: {lr_verdict}")

    # ── Overall closing ───────────────────────────────────────────────────────
    vuln_pred = (
        "are not predictable from pairwise pattern geometry"
        if d["lr_mean"] <= 0.55 else
        "are partially predictable from pairwise pattern geometry"
        if d["lr_mean"] <= 0.70 else
        "are predictable from pairwise pattern geometry"
    )
    print("\n=== OVERALL PHASE 3 CLOSING ===")
    print(
        f"Headline thesis claim (sharpened): \"At N=100 class-balanced storage, "
        f"white-box one-pixel attacks succeed on {COND_WB_SUCCESS*100:.1f}% of "
        f"clean-retrievable probes, representing a {sharpened:.1f}x amplification "
        f"over equivalent-magnitude random noise ({matched_fr_m*100:.1f}% conditional "
        f"failure at σ={best_sigma:.3f}). Vulnerable probes {vuln_pred}.\""
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    print("=" * 62)
    print("Phase 3 Closing Diagnostics")
    print("=" * 62)

    print("\nLoading MNIST ...")
    images, labels = load_mnist_train()

    print("\n--- Diagnostic C: σ sweep (headline cell) ---")
    c_rows = run_diagnostic_c(images, labels, EXP_DIR)
    print(f"  Saved phase3_diag_c_matched_sigma.csv  ({len(c_rows)} rows)")

    print("\n--- Diagnostic D: Vulnerable probe characterization ---")
    d = run_diagnostic_d(images, labels, EXP_DIR)
    print(f"  Saved phase3_diag_d_probe_features.csv  ({len(d['feature_rows'])} rows)")
    print(f"  Saved phase3_diag_d_summary.csv")

    # Best-matched sigma (for figure)
    sigma_mean_l2 = {}
    for sigma in SIGMAS_C:
        sub = [r for r in c_rows if r["sigma"] == sigma]
        sigma_mean_l2[sigma] = float(np.mean([r["mean_noise_l2"] for r in sub]))
    best_sigma = min(SIGMAS_C, key=lambda s: abs(sigma_mean_l2[s] - WB_MEAN_L2))

    print("\nGenerating figures ...")
    save_figure_c(c_rows, best_sigma, str(FIG_DIR / "phase3_diag_c_sigma_sweep.png"))
    print("  Saved: phase3_diag_c_sigma_sweep.png")
    save_figure_d_classes(d, str(FIG_DIR / "phase3_diag_d_classes.png"))
    print("  Saved: phase3_diag_d_classes.png")
    save_figure_d_features(d, str(FIG_DIR / "phase3_diag_d_features.png"))
    print("  Saved: phase3_diag_d_features.png")

    print_summary(c_rows, d)
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
