"""
CIFAR-10 pattern crowding fix experiment.

Tests three preprocessing strategies against the raw baseline:
  1. Raw            — no preprocessing (original grayscale pixels in [0,1])
  2. Centered       — subtract the mean of the N stored patterns from X and query
  3. Centered+Norm  — centered, then L2-normalise each pattern to unit norm

Also measures:
  • β sweep on the centered method at N=10 (where crowding is worst)
  • One-pixel white-box attack success: raw vs centered at N=10
    (direct comparison: does fixing retrieval also expose vulnerability?)

Run: python -m experiments.grayscale_cifar_fix_experiment
"""

from __future__ import annotations

import csv
import io
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hopfield.network  import ContinuousHopfield
from hopfield.metrics  import retrieval_accuracy
from hopfield.sampling import sample_class_balanced

# ── config ──────────────────────────────────────────────────────────────────
SEEDS      = [42, 43, 44, 45, 46]
N_VALUES   = [10, 50, 100]
BETA       = 8.0
BETA_SWEEP = [4.0, 8.0, 16.0, 32.0, 64.0]
N_BETA     = 10          # N used for β sweep
N_ATK      = 10          # N used for attack comparison
METHODS    = ["raw", "centered", "centered_norm"]

_CANDS   = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
IMG_DIM  = 1024
IMG_SZ   = 32
N_CANDS  = IMG_DIM * len(_CANDS)   # 5120

CIFAR_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
]

DATA_DIR = ROOT / "data"
EXP_DIR  = ROOT / "experiments"
FIG_DIR  = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar10_gray() -> tuple[torch.Tensor, torch.Tensor]:
    ds   = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True)
    raw  = ds.data.astype(np.float32) / 255.0
    gray = (0.2989 * raw[:, :, :, 0]
            + 0.5870 * raw[:, :, :, 1]
            + 0.1140 * raw[:, :, :, 2])
    images  = torch.tensor(gray.reshape(-1, 1024), dtype=torch.float32)
    labels  = torch.tensor(ds.targets, dtype=torch.long)
    return images, labels


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(X: torch.Tensor, method: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (X_proc, mu) where:
      X_proc : preprocessed storage matrix  (d, N)
      mu     : column vector to subtract from queries  (d, 1)
               (all-zeros for 'raw', mean for the other two)
    """
    if method == "raw":
        return X, torch.zeros(X.shape[0], 1)

    mu = X.mean(dim=1, keepdim=True)    # (d, 1) — mean across the N stored patterns
    Xc = X - mu                          # (d, N) centred

    if method == "centered":
        return Xc, mu

    # centered_norm: additionally normalise each centred pattern to unit L2 norm
    nrm  = Xc.norm(dim=0, keepdim=True).clamp(min=1e-8)
    Xcn  = Xc / nrm
    return Xcn, mu


def preprocess_query(q: torch.Tensor, mu: torch.Tensor, method: str) -> torch.Tensor:
    """Apply the same preprocessing to a single query vector."""
    qc = q - mu.squeeze()
    if method == "centered_norm":
        qc = qc / qc.norm().clamp(min=1e-8)
    return qc


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_cosine_mean(X: torch.Tensor) -> float:
    Xn  = X.numpy()
    nrm = np.linalg.norm(Xn, axis=0, keepdims=True)
    Xnn = Xn / (nrm + 1e-8)
    C   = Xnn.T @ Xnn
    np.fill_diagonal(C, 0.0)
    n = C.shape[0]
    return float(C.sum() / (n * (n - 1)))


def _batch_cosine(retrieved: torch.Tensor, true_pat: torch.Tensor) -> torch.Tensor:
    dots    = retrieved.T @ true_pat
    r_norms = retrieved.norm(dim=0)
    t_norm  = true_pat.norm()
    return dots / (r_norms * t_norm).clamp(min=1e-8)


def _nearest_stored(ret: torch.Tensor, X: torch.Tensor) -> int:
    cos_all = (X.T @ ret) / (X.norm(dim=0) * ret.norm()).clamp(min=1e-8)
    return int(cos_all.argmax().item())


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 1: Preprocessing comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing_comparison(
    images: torch.Tensor,
    labels: torch.Tensor,
) -> list[dict]:
    rows: list[dict] = []
    total = len(METHODS) * len(N_VALUES) * len(SEEDS)
    done  = 0

    for method in METHODS:
        for N in N_VALUES:
            for seed in SEEDS:
                X_raw, _ = sample_class_balanced((images, labels), N, seed=seed)
                X_proc, mu = preprocess(X_raw, method)
                hop = ContinuousHopfield(X_proc, beta=BETA)
                stored_raw = X_raw.T.contiguous()   # (N, 1024)

                failures = 0
                for i in range(N):
                    q   = preprocess_query(stored_raw[i], mu, method)
                    ret = hop.retrieve(q, steps=1)
                    if not retrieval_accuracy(ret, X_proc, i):
                        failures += 1

                rows.append({
                    "method":   method,
                    "N":        N,
                    "seed":     seed,
                    "bl_fail":  round(failures / N, 4),
                    "cos_mean": round(pairwise_cosine_mean(X_proc), 5),
                })
                done += 1
                if done % 10 == 0:
                    print(f"  [{done}/{total}] {method} N={N} seed={seed}  "
                          f"fail={failures/N:.2f}")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 2: β sweep (centered, N=N_BETA)
# ─────────────────────────────────────────────────────────────────────────────

def run_beta_sweep(
    images: torch.Tensor,
    labels: torch.Tensor,
) -> list[dict]:
    rows: list[dict] = []

    for beta in BETA_SWEEP:
        for seed in SEEDS:
            X_raw, _ = sample_class_balanced((images, labels), N_BETA, seed=seed)
            X_proc, mu = preprocess(X_raw, "centered")
            hop = ContinuousHopfield(X_proc, beta=beta)
            stored_raw = X_raw.T.contiguous()

            failures = 0
            for i in range(N_BETA):
                q   = preprocess_query(stored_raw[i], mu, "centered")
                ret = hop.retrieve(q, steps=1)
                if not retrieval_accuracy(ret, X_proc, i):
                    failures += 1

            rows.append({
                "beta":    beta,
                "seed":    seed,
                "bl_fail": round(failures / N_BETA, 4),
            })
        bl_m = np.mean([r["bl_fail"] for r in rows if r["beta"] == beta])
        print(f"  β={beta:<5.0f}  N={N_BETA}  mean_fail={bl_m:.3f}")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Inline white-box attacker (centered or raw)
# ─────────────────────────────────────────────────────────────────────────────

def wb_attack(
    q_orig:    torch.Tensor,          # (1024,) original [0,1] grayscale query
    mu:        torch.Tensor,          # (1024, 1) mean to subtract (zeros for raw)
    true_index: int,
    X_proc:    torch.Tensor,          # (1024, N) preprocessed storage
    hop:       ContinuousHopfield,
) -> dict:
    """
    Perturbs one pixel in the original [0,1] space, then applies centering
    (mu subtraction) before retrieval.  Uses cosine-delta selection with
    first-order sensitivity fallback for the degenerate regime.
    """
    locs = torch.arange(IMG_DIM).repeat_interleave(len(_CANDS))
    vals = _CANDS.repeat(IMG_DIM)

    # Build all candidate queries in original space, then centre
    queries_orig = q_orig.unsqueeze(1).expand(-1, N_CANDS).clone()
    queries_orig[locs, torch.arange(N_CANDS)] = vals
    queries_proc = queries_orig - mu             # (1024, 5120) — centred

    retrieved = hop.retrieve(queries_proc, steps=1)
    cos_sims  = _batch_cosine(retrieved, X_proc[:, true_index])

    # Cosine delta vs clean retrieval
    q_proc    = q_orig - mu.squeeze()
    ret_clean = hop.retrieve(q_proc, steps=1)
    clean_cos = float(_batch_cosine(ret_clean.unsqueeze(1), X_proc[:, true_index])[0])
    delta     = cos_sims - clean_cos

    if float(delta.min().item()) < -1e-4:
        k = int(delta.argmin().item())
    else:
        true_at_locs = X_proc[:, true_index][locs]
        q_proc_at    = queries_proc[locs, torch.arange(N_CANDS)]
        fo_damage    = true_at_locs * (q_proc[locs] - q_proc_at)
        noop_mask    = (q_proc[locs] - q_proc_at).abs() < 1e-6
        fo_damage    = fo_damage.clone()
        fo_damage[noop_mask] = -1e9
        k = int(fo_damage.argmax().item())

    loc          = int(locs[k].item())
    val          = float(vals[k].item())
    cosine_delta = float(delta[k].item())

    return {
        "success":         _nearest_stored(retrieved[:, k], X_proc) != true_index,
        "pixel_i":         loc // IMG_SZ,
        "pixel_j":         loc % IMG_SZ,
        "pixel_value":     val,
        "original_value":  float(q_orig[loc].item()),
        "perturbation_l2": abs(val - float(q_orig[loc].item())),
        "cosine_delta":    cosine_delta,
        "retrieved_index": _nearest_stored(retrieved[:, k], X_proc),
        "evaluations":     N_CANDS,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Experiment 3: Attack comparison (raw vs centered, N=N_ATK)
# ─────────────────────────────────────────────────────────────────────────────

def run_attack_comparison(
    images: torch.Tensor,
    labels: torch.Tensor,
) -> list[dict]:
    rows: list[dict] = []

    for method in ["raw", "centered"]:
        for seed in SEEDS:
            X_raw, _ = sample_class_balanced((images, labels), N_ATK, seed=seed)
            X_proc, mu = preprocess(X_raw, method)
            hop = ContinuousHopfield(X_proc, beta=BETA)
            stored_raw = X_raw.T.contiguous()

            for i in range(N_ATK):
                q_proc = preprocess_query(stored_raw[i], mu, method)
                ret_bl = hop.retrieve(q_proc, steps=1)
                bl_ok  = int(retrieval_accuracy(ret_bl, X_proc, i))

                atk = wb_attack(stored_raw[i], mu, i, X_proc, hop)

                rows.append({
                    "method":   method,
                    "seed":     seed,
                    "probe":    i,
                    "class":    CIFAR_CLASSES[i],
                    "baseline_correct": bl_ok,
                    "attack_success":   int(atk["success"]),
                    "cosine_delta":     round(atk["cosine_delta"], 6),
                    "pixel_i":          atk["pixel_i"],
                    "pixel_j":          atk["pixel_j"],
                    "l2":               round(atk["perturbation_l2"], 4),
                })

        bl_all  = [r["baseline_correct"] for r in rows if r["method"] == method]
        atk_bl  = [r["attack_success"]   for r in rows
                   if r["method"] == method and r["baseline_correct"] == 1]
        print(f"  {method:<14} N={N_ATK}  "
              f"baseline_correct={sum(bl_all)}/{len(bl_all)}  "
              f"conditional_attack={sum(atk_bl)}/{len(atk_bl)} "
              f"({sum(atk_bl)/max(len(atk_bl),1)*100:.1f}%)")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

METHOD_LABELS = {
    "raw":           "Raw pixels",
    "centered":      "Mean-centered",
    "centered_norm": "Centered + normalised",
}
METHOD_COLORS = {
    "raw":           "#d62728",
    "centered":      "#1f77b4",
    "centered_norm": "#2ca02c",
}

def save_figure(
    prep_rows:  list[dict],
    beta_rows:  list[dict],
    atk_rows:   list[dict],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "CIFAR-10 Pattern Crowding Fix: Preprocessing & β Sweep\n"
        f"Class-balanced storage · 5 seeds · β={BETA} (except β sweep panel)",
        fontsize=11, fontweight="bold",
    )

    # ── Panel 1: Baseline failure rate vs N ──────────────────────────────────
    ax = axes[0]
    for method in METHODS:
        sub = [r for r in prep_rows if r["method"] == method]
        means = [np.mean([r["bl_fail"] for r in sub if r["N"] == N]) for N in N_VALUES]
        stds  = [np.std( [r["bl_fail"] for r in sub if r["N"] == N], ddof=1)
                 for N in N_VALUES]
        ax.errorbar(N_VALUES, means, yerr=stds, fmt="o-",
                    color=METHOD_COLORS[method], label=METHOD_LABELS[method],
                    lw=2, ms=6, capsize=5)
        for n, m in zip(N_VALUES, means):
            ax.annotate(f"{m:.2f}", xy=(n, m), xytext=(0, 7),
                        textcoords="offset points", ha="center",
                        fontsize=7, color=METHOD_COLORS[method])

    ax.set_xscale("log")
    ax.set_xticks(N_VALUES)
    ax.set_xticklabels([str(n) for n in N_VALUES])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("N (stored patterns)", fontsize=9)
    ax.set_ylabel("Baseline failure rate", fontsize=9)
    ax.set_title("Retrieval failure rate vs N", fontsize=9)
    ax.axhline(0.2, color="gray", ls=":", lw=1, alpha=0.5, label="20% threshold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # ── Panel 2: Mean pairwise cosine vs N ───────────────────────────────────
    ax = axes[1]
    for method in METHODS:
        sub = [r for r in prep_rows if r["method"] == method]
        means = [np.mean([r["cos_mean"] for r in sub if r["N"] == N]) for N in N_VALUES]
        stds  = [np.std( [r["cos_mean"] for r in sub if r["N"] == N], ddof=1)
                 for N in N_VALUES]
        ax.errorbar(N_VALUES, means, yerr=stds, fmt="s--",
                    color=METHOD_COLORS[method], label=METHOD_LABELS[method],
                    lw=2, ms=6, capsize=5)
        for n, m in zip(N_VALUES, means):
            ax.annotate(f"{m:.3f}", xy=(n, m), xytext=(0, 7),
                        textcoords="offset points", ha="center",
                        fontsize=7, color=METHOD_COLORS[method])

    ax.set_xscale("log")
    ax.set_xticks(N_VALUES)
    ax.set_xticklabels([str(n) for n in N_VALUES])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("N (stored patterns)", fontsize=9)
    ax.set_ylabel("Mean pairwise cosine", fontsize=9)
    ax.set_title("Pattern crowding (lower = more separable)", fontsize=9)
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5,
               label="High-crowding threshold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # ── Panel 3: β sweep (centered, N=N_BETA) ────────────────────────────────
    ax = axes[2]
    means = [np.mean([r["bl_fail"] for r in beta_rows if r["beta"] == b])
             for b in BETA_SWEEP]
    stds  = [np.std( [r["bl_fail"] for r in beta_rows if r["beta"] == b], ddof=1)
             for b in BETA_SWEEP]
    ax.errorbar(BETA_SWEEP, means, yerr=stds, fmt="D-",
                color="#9467bd", lw=2, ms=7, capsize=5)
    for b, m in zip(BETA_SWEEP, means):
        ax.annotate(f"{m:.2f}", xy=(b, m), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8)

    # Also show attack conditional success for raw vs centered
    if atk_rows:
        for method, mk, clr in [("raw", "^", "#d62728"), ("centered", "v", "#1f77b4")]:
            sub = [r for r in atk_rows if r["method"] == method]
            bl_corr  = [r for r in sub if r["baseline_correct"] == 1]
            cond_sr  = sum(r["attack_success"] for r in bl_corr) / max(len(bl_corr), 1)
            raw_fail = 1 - sum(r["baseline_correct"] for r in sub) / len(sub)
            label = (f"{METHOD_LABELS[method]}\n"
                     f"baseline fail={raw_fail:.0%}, "
                     f"cond. attack={cond_sr:.0%}")
            ax.axhline(raw_fail, color=clr, ls=":", lw=1.5, alpha=0.7)
            ax.annotate(label, xy=(BETA_SWEEP[-1], raw_fail),
                        xytext=(-80, 8 if method == "raw" else -25),
                        textcoords="offset points", fontsize=6.5, color=clr,
                        arrowprops=dict(arrowstyle="-", color=clr, lw=0.8))

    ax.set_xlabel("β (inverse temperature)", fontsize=9)
    ax.set_ylabel("Baseline failure rate", fontsize=9)
    ax.set_title(
        f"β sweep — mean-centered, N={N_BETA}\n"
        "(dashed lines = raw/centered failure at β=8)",
        fontsize=9,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = FIG_DIR / "grayscale_cifar_fix_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def build_report(
    prep_rows: list[dict],
    beta_rows: list[dict],
    atk_rows:  list[dict],
) -> str:
    buf = io.StringIO()

    def p(line: str = "") -> None:
        buf.write(line + "\n")

    p("=" * 68)
    p("GRAYSCALE CIFAR-10 PATTERN CROWDING FIX EXPERIMENT")
    p("=" * 68)
    p()
    p(f"Seeds: {SEEDS}  |  β={BETA}  |  N values: {N_VALUES}")
    p(f"β sweep: {BETA_SWEEP} (centered, N={N_BETA})")
    p(f"Attack comparison: raw vs centered, N={N_ATK}")
    p()

    # ── Preprocessing comparison ──────────────────────────────────────────────
    p("=" * 68)
    p("EXPERIMENT 1: PREPROCESSING COMPARISON")
    p("=" * 68)
    p()
    p(f"  {'Method':<20}  {'N':>5}  {'Baseline fail':>15}  {'Mean cosine':>13}")
    p("  " + "-" * 58)
    for method in METHODS:
        for N in N_VALUES:
            sub = [r for r in prep_rows if r["method"] == method and r["N"] == N]
            bls = [r["bl_fail"]  for r in sub]
            cos = [r["cos_mean"] for r in sub]
            bm, bs = float(np.mean(bls)), float(np.std(bls, ddof=1))
            cm, cs = float(np.mean(cos)), float(np.std(cos, ddof=1))
            p(f"  {method:<20}  {N:>5}  "
              f"{bm*100:>6.1f}% +/- {bs*100:>4.1f}%  "
              f"{cm:>7.4f} +/- {cs:.4f}")
        p()

    # Best improvement summary
    for N in N_VALUES:
        raw_bl  = np.mean([r["bl_fail"] for r in prep_rows
                           if r["method"] == "raw" and r["N"] == N])
        cen_bl  = np.mean([r["bl_fail"] for r in prep_rows
                           if r["method"] == "centered" and r["N"] == N])
        cnorm_bl = np.mean([r["bl_fail"] for r in prep_rows
                            if r["method"] == "centered_norm" and r["N"] == N])
        raw_cos  = np.mean([r["cos_mean"] for r in prep_rows
                            if r["method"] == "raw" and r["N"] == N])
        cen_cos  = np.mean([r["cos_mean"] for r in prep_rows
                            if r["method"] == "centered" and r["N"] == N])
        p(f"  N={N}: Raw failure {raw_bl*100:.1f}% → Centered {cen_bl*100:.1f}%"
          f" → Centered+Norm {cnorm_bl*100:.1f}%  "
          f"(cosine {raw_cos:.3f} → {cen_cos:.3f})")
    p()

    # ── β sweep ───────────────────────────────────────────────────────────────
    p("=" * 68)
    p(f"EXPERIMENT 2: β SWEEP (centered, N={N_BETA})")
    p("=" * 68)
    p()
    p(f"  {'β':>6}  {'Baseline fail':>20}")
    p("  " + "-" * 28)
    for beta in BETA_SWEEP:
        sub = [r["bl_fail"] for r in beta_rows if r["beta"] == beta]
        bm  = float(np.mean(sub))
        bs  = float(np.std(sub, ddof=1))
        p(f"  {beta:>6.1f}  {bm*100:>8.1f}% +/- {bs*100:.1f}%")
    p()
    best_beta = min(BETA_SWEEP, key=lambda b:
                    np.mean([r["bl_fail"] for r in beta_rows if r["beta"] == b]))
    p(f"  Best β: {best_beta:.0f}")
    p()

    # ── Attack comparison ─────────────────────────────────────────────────────
    p("=" * 68)
    p(f"EXPERIMENT 3: ONE-PIXEL ATTACK (N={N_ATK}, β={BETA})")
    p("=" * 68)
    p()
    for method in ["raw", "centered"]:
        sub     = [r for r in atk_rows if r["method"] == method]
        total   = len(sub)
        n_bl    = sum(r["baseline_correct"] for r in sub)
        bl_corr = [r for r in sub if r["baseline_correct"] == 1]
        n_atk   = sum(r["attack_success"] for r in bl_corr)
        cond_sr = n_atk / max(len(bl_corr), 1)
        raw_sr  = sum(r["attack_success"] for r in sub) / total

        # How many probes had a real cosine delta (not fallback)?
        real_atk = sum(1 for r in sub if float(r["cosine_delta"]) < -1e-4)

        p(f"  {METHOD_LABELS[method]}:")
        p(f"    Baseline correct:      {n_bl}/{total}  "
          f"({(1-n_bl/total)*100:.1f}% failure)")
        p(f"    Raw attack success:    {sum(r['attack_success'] for r in sub)}/{total}")
        p(f"    Conditional attack:    {n_atk}/{len(bl_corr)} = {cond_sr*100:.1f}%")
        p(f"    Probes with real Δcos: {real_atk}/{total}")
        p()

    p("=" * 68)
    p("INTERPRETATION")
    p("=" * 68)
    p()
    raw_bl_10  = np.mean([r["bl_fail"] for r in prep_rows
                           if r["method"] == "raw" and r["N"] == N_ATK])
    cen_bl_10  = np.mean([r["bl_fail"] for r in prep_rows
                           if r["method"] == "centered" and r["N"] == N_ATK])
    raw_cos_10 = np.mean([r["cos_mean"] for r in prep_rows
                           if r["method"] == "raw" and r["N"] == N_ATK])
    cen_cos_10 = np.mean([r["cos_mean"] for r in prep_rows
                           if r["method"] == "centered" and r["N"] == N_ATK])

    sub_cen   = [r for r in atk_rows if r["method"] == "centered"]
    bl_cen    = [r for r in sub_cen if r["baseline_correct"] == 1]
    cond_cen  = sum(r["attack_success"] for r in bl_cen) / max(len(bl_cen), 1)
    sub_raw   = [r for r in atk_rows if r["method"] == "raw"]
    bl_raw_atk = [r for r in sub_raw if r["baseline_correct"] == 1]
    cond_raw  = sum(r["attack_success"] for r in bl_raw_atk) / max(len(bl_raw_atk), 1)

    p(f"  Mean-centering at N={N_ATK}:")
    p(f"    Pairwise cosine:  {raw_cos_10:.3f} (raw)  →  {cen_cos_10:.3f} (centered)")
    p(f"    Baseline failure: {raw_bl_10*100:.1f}% (raw)  →  {cen_bl_10*100:.1f}% (centered)")
    p(f"    Cond. attack:     {cond_raw*100:.1f}% (raw)  →  {cond_cen*100:.1f}% (centered)")
    p()

    if cond_cen > cond_raw + 0.01:
        p("  Verdict: Mean-centering successfully reduces pattern crowding AND")
        p("  makes the network vulnerable to one-pixel attacks — confirming that")
        p("  crowding was the root cause of the null result, not network robustness.")
    elif cen_bl_10 < raw_bl_10 - 0.1:
        p("  Verdict: Mean-centering improves baseline retrieval but one-pixel")
        p("  attacks still fail (or partially succeed). The remaining crowding")
        p("  or the network's attractor structure prevents single-pixel redirection.")
    else:
        p("  Verdict: Mean-centering has limited effect at this N. The similarity")
        p("  between CIFAR patterns is intrinsic to image content, not just DC bias.")
    p()
    p("=" * 68)

    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# CSV saving
# ─────────────────────────────────────────────────────────────────────────────

def save_csvs(
    prep_rows: list[dict],
    beta_rows: list[dict],
    atk_rows:  list[dict],
) -> None:
    def _write(rows, path, fields):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow({k: row[k] for k in fields})
        print(f"  Saved: {Path(path).name}  ({len(rows)} rows)")

    _write(prep_rows, EXP_DIR / "cifar_fix_preprocessing.csv",
           ["method", "N", "seed", "bl_fail", "cos_mean"])

    _write(beta_rows, EXP_DIR / "cifar_fix_beta_sweep.csv",
           ["beta", "seed", "bl_fail"])

    _write(atk_rows, EXP_DIR / "cifar_fix_attack.csv",
           ["method", "seed", "probe", "class", "baseline_correct",
            "attack_success", "cosine_delta", "pixel_i", "pixel_j", "l2"])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    print("=" * 68)
    print("CIFAR-10 Pattern Crowding Fix Experiment")
    print("=" * 68)

    print("\nLoading CIFAR-10 ...")
    images, labels = load_cifar10_gray()
    print(f"  {images.shape}  range [{images.min():.3f}, {images.max():.3f}]")

    print(f"\nExperiment 1: Preprocessing comparison "
          f"(methods={METHODS}, N={N_VALUES}, seeds={SEEDS}) ...")
    prep_rows = run_preprocessing_comparison(images, labels)

    print(f"\nExperiment 2: β sweep (centered, N={N_BETA}, β={BETA_SWEEP}) ...")
    beta_rows = run_beta_sweep(images, labels)

    print(f"\nExperiment 3: Attack comparison "
          f"(raw vs centered, N={N_ATK}, seeds={SEEDS}) ...")
    atk_rows = run_attack_comparison(images, labels)

    print("\nSaving CSVs ...")
    save_csvs(prep_rows, beta_rows, atk_rows)

    print("Generating figure ...")
    save_figure(prep_rows, beta_rows, atk_rows)

    report = build_report(prep_rows, beta_rows, atk_rows)
    print()
    print(report)

    report_path = EXP_DIR / "cifar_fix_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {report_path.name}")
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
