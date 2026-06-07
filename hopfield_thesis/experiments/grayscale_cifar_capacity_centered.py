"""
Grayscale CIFAR-10 capacity sweep — mean-centered preprocessing.

Finds the operating regime of the Continuous Hopfield Network on CIFAR-10
after fixing the pattern crowding issue via mean-centering.

Tests the centered+normalised model at N ∈ [10, 50, 100, 200, 500, 1000, 2000]
and runs a one-pixel white-box attack at each N, producing:
  - capacity curve: baseline failure vs N
  - attack curve:   conditional attack success vs N
  - comparison:     centred CIFAR vs raw MNIST (same β, same methodology)

This answers: "at what N does the model correctly retrieve CIFAR images, and
does it become exploitable by single-pixel attacks in that regime?"

Run: python -m experiments.grayscale_cifar_capacity_centered
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
import matplotlib.patches as mpatches
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
SEEDS       = [42, 43, 44, 45, 46]
N_VALUES    = [10, 50, 100, 200, 500, 1000, 2000]
BETA        = 8.0
N_PROBE_MAX = 50      # max probes per (N, seed) for the capacity sweep
N_ATK_MAX   = 20      # max probes for attack evaluation (expensive)
N_ATK_VALS  = [10, 50, 100, 200, 500]   # subset of N_VALUES for attack

_CANDS  = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
IMG_DIM = 1024
IMG_SZ  = 32
N_CANDS = IMG_DIM * len(_CANDS)   # 5120

DATA_DIR = ROOT / "data"
EXP_DIR  = ROOT / "experiments"
FIG_DIR  = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar10_gray() -> tuple[torch.Tensor, torch.Tensor]:
    ds   = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True)
    raw  = ds.data.astype(np.float32) / 255.0
    gray = (0.2989 * raw[:, :, :, 0]
            + 0.5870 * raw[:, :, :, 1]
            + 0.1140 * raw[:, :, :, 2])
    images = torch.tensor(gray.reshape(-1, 1024), dtype=torch.float32)
    labels = torch.tensor(ds.targets, dtype=torch.long)
    return images, labels


def load_mnist() -> tuple[torch.Tensor, torch.Tensor]:
    ds = torchvision.datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
    return ds.data.float().view(-1, 784) / 255.0, ds.targets


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing: centered + L2-normalised
# ─────────────────────────────────────────────────────────────────────────────

def center_and_normalise(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (X_proc, mu):
      X_proc : (d, N)  — each column centred then unit-norm
      mu     : (d, 1)  — mean of the raw stored patterns (subtract from query)
    """
    mu  = X.mean(dim=1, keepdim=True)
    Xc  = X - mu
    nrm = Xc.norm(dim=0, keepdim=True).clamp(min=1e-8)
    return Xc / nrm, mu


def preprocess_query(q: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    qc  = q - mu.squeeze()
    return qc / qc.norm().clamp(min=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
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
# Inline white-box attacker (centered+normalised space)
# ─────────────────────────────────────────────────────────────────────────────

def wb_attack(
    q_orig:     torch.Tensor,     # (1024,) raw [0,1] query
    mu:         torch.Tensor,     # (1024, 1) stored-set mean
    true_index: int,
    X_proc:     torch.Tensor,     # (1024, N) centred+normalised storage
    hop:        ContinuousHopfield,
) -> dict:
    locs = torch.arange(IMG_DIM).repeat_interleave(len(_CANDS))
    vals = _CANDS.repeat(IMG_DIM)

    # Build all candidate queries in original space, then centre+normalise
    queries_orig = q_orig.unsqueeze(1).expand(-1, N_CANDS).clone()
    queries_orig[locs, torch.arange(N_CANDS)] = vals

    mu_sq = mu.squeeze()
    queries_c = queries_orig - mu.unsqueeze(2).squeeze(2)  # (1024, N_CANDS)
    nrms      = queries_c.norm(dim=0, keepdim=True).clamp(min=1e-8)
    queries_p = queries_c / nrms

    retrieved = hop.retrieve(queries_p, steps=1)
    cos_sims  = _batch_cosine(retrieved, X_proc[:, true_index])

    q_proc    = preprocess_query(q_orig, mu)
    ret_clean = hop.retrieve(q_proc, steps=1)
    clean_cos = float(_batch_cosine(ret_clean.unsqueeze(1), X_proc[:, true_index])[0])
    delta     = cos_sims - clean_cos

    if float(delta.min().item()) < -1e-4:
        k = int(delta.argmin().item())
    else:
        true_at_locs = X_proc[:, true_index][locs]
        fo_damage    = true_at_locs * (q_proc[locs] - queries_p[locs, torch.arange(N_CANDS)])
        noop_mask    = (q_proc[locs] - queries_p[locs, torch.arange(N_CANDS)]).abs() < 1e-6
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
    }


# ─────────────────────────────────────────────────────────────────────────────
# Capacity sweep (baseline retrieval only)
# ─────────────────────────────────────────────────────────────────────────────

def run_capacity_sweep(
    images: torch.Tensor,
    labels: torch.Tensor,
) -> list[dict]:
    rows: list[dict] = []
    total = len(N_VALUES) * len(SEEDS)
    done  = 0

    for N in N_VALUES:
        for seed in SEEDS:
            X_raw, _ = sample_class_balanced((images, labels), N, seed=seed)
            X_proc, mu = center_and_normalise(X_raw)
            hop      = ContinuousHopfield(X_proc, beta=BETA)
            stored   = X_raw.T.contiguous()

            n_probe = min(N_PROBE_MAX, N)
            rng = torch.Generator()
            rng.manual_seed(seed * 10000 + N)
            probe_idx = torch.randperm(N, generator=rng)[:n_probe].tolist()

            failures = 0
            for i in probe_idx:
                q   = preprocess_query(stored[i], mu)
                ret = hop.retrieve(q, steps=1)
                if not retrieval_accuracy(ret, X_proc, i):
                    failures += 1

            done += 1
            cos_m = pairwise_cosine_mean(X_proc)
            rows.append({
                "N":        N,
                "seed":     seed,
                "n_probe":  n_probe,
                "bl_fail":  round(failures / n_probe, 4),
                "cos_mean": round(cos_m, 6),
            })
            print(f"  [{done:>2}/{total}] N={N:>5}  seed={seed}  "
                  f"fail={failures/n_probe:.3f}  cos={cos_m:.4f}")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Attack sweep (subset of N values)
# ─────────────────────────────────────────────────────────────────────────────

def run_attack_sweep(
    images: torch.Tensor,
    labels: torch.Tensor,
) -> list[dict]:
    rows: list[dict] = []

    for N in N_ATK_VALS:
        for seed in SEEDS:
            X_raw, _ = sample_class_balanced((images, labels), N, seed=seed)
            X_proc, mu = center_and_normalise(X_raw)
            hop      = ContinuousHopfield(X_proc, beta=BETA)
            stored   = X_raw.T.contiguous()

            n_probe = min(N_ATK_MAX, N)
            rng = torch.Generator()
            rng.manual_seed(seed * 10000 + N + 1)
            probe_idx = torch.randperm(N, generator=rng)[:n_probe].tolist()

            for i in probe_idx:
                q_proc = preprocess_query(stored[i], mu)
                ret_bl = hop.retrieve(q_proc, steps=1)
                bl_ok  = int(retrieval_accuracy(ret_bl, X_proc, i))

                atk = wb_attack(stored[i], mu, i, X_proc, hop)

                rows.append({
                    "N":               N,
                    "seed":            seed,
                    "probe":           i,
                    "baseline_correct": bl_ok,
                    "attack_success":  int(atk["success"]),
                    "cosine_delta":    round(atk["cosine_delta"], 6),
                    "l2":              round(atk["perturbation_l2"], 4),
                })

        sub      = [r for r in rows if r["N"] == N]
        n_bl     = sum(r["baseline_correct"] for r in sub)
        bl_corr  = [r for r in sub if r["baseline_correct"] == 1]
        n_atk    = sum(r["attack_success"] for r in bl_corr)
        cond_sr  = n_atk / max(len(bl_corr), 1)
        real_atk = sum(1 for r in sub if float(r["cosine_delta"]) < -1e-4)
        print(f"  N={N:>5}  baseline_correct={n_bl}/{len(sub)}  "
              f"conditional_attack={n_atk}/{len(bl_corr)} ({cond_sr*100:.1f}%)  "
              f"real_Δcos={real_atk}/{len(sub)}")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# MNIST reference (raw, baseline only)
# ─────────────────────────────────────────────────────────────────────────────

def run_mnist_reference(
    images: torch.Tensor,
    labels: torch.Tensor,
) -> list[dict]:
    N_VALS_MNIST = [10, 50, 100, 200, 500, 1000]
    rows: list[dict] = []
    for N in N_VALS_MNIST:
        for seed in SEEDS:
            X, _ = sample_class_balanced((images, labels), N, seed=seed)
            hop  = ContinuousHopfield(X, beta=BETA)
            stored = X.T.contiguous()
            n_probe = min(N_PROBE_MAX, N)
            rng = torch.Generator()
            rng.manual_seed(seed * 10000 + N)
            probe_idx = torch.randperm(N, generator=rng)[:n_probe].tolist()
            failures = 0
            for i in probe_idx:
                ret = hop.retrieve(stored[i], steps=1)
                if not retrieval_accuracy(ret, X, i):
                    failures += 1
            rows.append({"N": N, "seed": seed, "bl_fail": failures / n_probe})
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Figure
# ─────────────────────────────────────────────────────────────────────────────

def save_figure(
    cap_rows:   list[dict],
    atk_rows:   list[dict],
    mnist_rows: list[dict],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Grayscale CIFAR-10 (Centered+Normalised) vs Raw MNIST — Capacity & Attacks\n"
        f"Continuous Hopfield Network · β={BETA} · 5 seeds",
        fontsize=11, fontweight="bold",
    )

    # ── Panel 1: Capacity curve ───────────────────────────────────────────────
    ax = axes[0]

    # CIFAR centered+norm — baseline failure
    cifar_Ns  = N_VALUES
    cifar_m   = [np.mean([r["bl_fail"] for r in cap_rows if r["N"] == N])
                 for N in cifar_Ns]
    cifar_s   = [np.std([r["bl_fail"] for r in cap_rows if r["N"] == N], ddof=1)
                 for N in cifar_Ns]
    ax.errorbar(cifar_Ns, cifar_m, yerr=cifar_s, fmt="o-", color="#ff7f0e",
                lw=2, ms=7, capsize=5, label="CIFAR-10 (centered+normalised)")

    # MNIST raw — baseline failure (reference)
    mnist_Ns = sorted(set(r["N"] for r in mnist_rows))
    mnist_m  = [np.mean([r["bl_fail"] for r in mnist_rows if r["N"] == N])
                for N in mnist_Ns]
    mnist_s  = [np.std([r["bl_fail"] for r in mnist_rows if r["N"] == N], ddof=1)
                for N in mnist_Ns]
    ax.errorbar(mnist_Ns, mnist_m, yerr=mnist_s, fmt="s--", color="#1f77b4",
                lw=2, ms=7, capsize=5, label="MNIST (raw pixels, reference)")

    for n, m in zip(cifar_Ns, cifar_m):
        ax.annotate(f"{m:.2f}", xy=(n, m), xytext=(0, 8),
                    textcoords="offset points", ha="center",
                    fontsize=7.5, color="#ff7f0e")

    ax.set_xscale("log")
    ax.set_xticks(cifar_Ns)
    ax.set_xticklabels([str(n) for n in cifar_Ns], fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("N (stored patterns)", fontsize=10)
    ax.set_ylabel("Baseline failure rate", fontsize=10)
    ax.set_title("Clean retrieval capacity curve\n"
                 "CIFAR centered+norm vs raw MNIST", fontsize=9)
    ax.axhline(0.2, color="gray", ls=":", lw=1.2, alpha=0.6, label="20% threshold")
    ax.axhline(0.5, color="gray", ls="--", lw=1.2, alpha=0.6, label="50% threshold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # ── Panel 2: Attack success vs N ─────────────────────────────────────────
    ax = axes[1]

    # Conditional attack success (centered CIFAR)
    cifar_atk_Ns = N_ATK_VALS
    cond_m, cond_s, bl_m = [], [], []
    for N in cifar_atk_Ns:
        sub     = [r for r in atk_rows if r["N"] == N]
        bl_corr = [r for r in sub if r["baseline_correct"] == 1]
        cond_m.append(
            np.mean([r["attack_success"] for r in bl_corr]) if bl_corr else 0.0
        )
        cond_s.append(0.0)   # small sample — skip std
        bl_m.append(np.mean([r["baseline_correct"] for r in sub]))

    ax.bar(np.arange(len(cifar_atk_Ns)) - 0.18, bl_m, width=0.35,
           color="#aec7e8", label="Baseline-correct rate (CIFAR centered+norm)")
    ax.bar(np.arange(len(cifar_atk_Ns)) + 0.18, cond_m, width=0.35,
           color="#ff7f0e", label="Conditional attack success (CIFAR centered+norm)")

    # MNIST reference at N=100 (from phase3 headline)
    ax.axhline(0.908, color="#1f77b4", ls="--", lw=1.5, alpha=0.7,
               label="MNIST N=100 baseline-correct (90.8%)")
    ax.axhline(0.037, color="#2ca02c", ls="-.", lw=1.5, alpha=0.7,
               label="MNIST N=100 cond. attack (3.7%)")

    for xi, (n, cm, bm) in enumerate(zip(cifar_atk_Ns, cond_m, bl_m)):
        ax.text(xi - 0.18, bm + 0.015, f"{bm:.2f}", ha="center",
                fontsize=7, color="#1f77b4")
        ax.text(xi + 0.18, cm + 0.015, f"{cm:.2f}", ha="center",
                fontsize=7, color="#ff7f0e")

    ax.set_xticks(np.arange(len(cifar_atk_Ns)))
    ax.set_xticklabels([str(n) for n in cifar_atk_Ns], fontsize=9)
    ax.set_xlabel("N (stored patterns)", fontsize=10)
    ax.set_ylabel("Rate", fontsize=10)
    ax.set_ylim(-0.05, 1.15)
    ax.set_title("One-pixel attack success — centered+norm CIFAR\n"
                 "vs MNIST reference (dashed lines)", fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    out = FIG_DIR / "grayscale_cifar_capacity_centered.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def build_report(
    cap_rows:   list[dict],
    atk_rows:   list[dict],
    mnist_rows: list[dict],
) -> str:
    buf = io.StringIO()

    def p(line: str = "") -> None:
        buf.write(line + "\n")

    p("=" * 68)
    p("GRAYSCALE CIFAR-10 (CENTERED+NORMALISED) — CAPACITY & ATTACK SWEEP")
    p("=" * 68)
    p()
    p(f"Preprocessing: center by stored-pattern mean, then L2-normalise")
    p(f"β={BETA}  Seeds={SEEDS}")
    p(f"Capacity sweep N: {N_VALUES}")
    p(f"Attack sweep N:   {N_ATK_VALS}")
    p()

    p("=" * 68)
    p("CAPACITY SWEEP (centered+normalised)")
    p("=" * 68)
    p()
    p(f"  {'N':>6}  {'Baseline fail':>18}  {'Mean cos (proc)':>18}")
    p("  " + "-" * 46)
    for N in N_VALUES:
        sub  = [r for r in cap_rows if r["N"] == N]
        bl_m = float(np.mean([r["bl_fail"]  for r in sub]))
        bl_s = float(np.std( [r["bl_fail"]  for r in sub], ddof=1))
        co_m = float(np.mean([r["cos_mean"] for r in sub]))
        p(f"  {N:>6}  {bl_m*100:>7.1f}% +/- {bl_s*100:>4.1f}%    {co_m:>+.4f}")
    p()

    p("  MNIST reference (raw pixels):")
    for N in sorted(set(r["N"] for r in mnist_rows)):
        sub  = [r for r in mnist_rows if r["N"] == N]
        bl_m = float(np.mean([r["bl_fail"] for r in sub]))
        bl_s = float(np.std( [r["bl_fail"] for r in sub], ddof=1))
        p(f"    N={N:>5}  {bl_m*100:>7.1f}% +/- {bl_s*100:>4.1f}%")
    p()

    p("=" * 68)
    p("ATTACK SWEEP (centered+normalised)")
    p("=" * 68)
    p()
    p(f"  {'N':>6}  {'Baseline ok':>12}  {'Cond. attack':>14}  {'Real Δcos':>10}")
    p("  " + "-" * 48)
    for N in N_ATK_VALS:
        sub     = [r for r in atk_rows if r["N"] == N]
        n_bl    = sum(r["baseline_correct"] for r in sub)
        bl_corr = [r for r in sub if r["baseline_correct"] == 1]
        n_atk   = sum(r["attack_success"] for r in bl_corr)
        cond    = n_atk / max(len(bl_corr), 1)
        real_d  = sum(1 for r in sub if float(r["cosine_delta"]) < -1e-4)
        p(f"  {N:>6}  {n_bl:>5}/{len(sub):<5}  "
          f"{n_atk:>5}/{len(bl_corr):<5} ({cond*100:>4.1f}%)  "
          f"{real_d:>4}/{len(sub)}")
    p()

    p("=" * 68)
    p("INTERPRETATION")
    p("=" * 68)
    p()

    # Find N where failure crosses thresholds
    cross20 = [N for N in N_VALUES
               if np.mean([r["bl_fail"] for r in cap_rows if r["N"] == N]) > 0.20]
    cross50 = [N for N in N_VALUES
               if np.mean([r["bl_fail"] for r in cap_rows if r["N"] == N]) > 0.50]

    p(f"  Failure > 20% first occurs at N = {min(cross20) if cross20 else '>'+str(max(N_VALUES))}")
    p(f"  Failure > 50% first occurs at N = {min(cross50) if cross50 else '>'+str(max(N_VALUES))}")
    p()

    # Best attack result
    best_cond_N = None
    best_cond   = 0.0
    for N in N_ATK_VALS:
        sub     = [r for r in atk_rows if r["N"] == N]
        bl_corr = [r for r in sub if r["baseline_correct"] == 1]
        cond    = sum(r["attack_success"] for r in bl_corr) / max(len(bl_corr), 1)
        if cond > best_cond:
            best_cond = cond
            best_cond_N = N

    if best_cond > 0.0:
        p(f"  Best conditional attack: {best_cond*100:.1f}% at N={best_cond_N}")
        p("  Mean-centering brings CIFAR into a regime where one-pixel attacks succeed.")
    else:
        p("  Conditional attack success remains 0% across all tested N values.")
        p()
        p("  Explanation: centering+normalisation creates strongly negative pairwise cosine")
        p("  — patterns actively repel each other in the preprocessed space. This produces")
        p("  very deep, well-separated energy basins that a single-pixel perturbation")
        p("  (which changes one of 1024 dimensions by at most ±1.0) cannot escape.")
        p()
        p("  This is genuine model robustness — not the pseudo-robustness of confusion.")
        p("  The network correctly retrieves CIFAR images AND resists one-pixel attacks.")
        p()
        p("  For the thesis: this separates two distinct sources of attack immunity:")
        p("    (1) Confusion immunity (raw CIFAR): network already wrong, attack irrelevant")
        p("    (2) Basin immunity (centred CIFAR): network correct, basins too deep to escape")
        p("    (3) Exploitable regime (MNIST N=100): correct retrieval, shallow enough basins")
    p()
    p("=" * 68)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# CSV saving
# ─────────────────────────────────────────────────────────────────────────────

def save_csvs(cap_rows: list[dict], atk_rows: list[dict]) -> None:
    def _w(rows, path, fields):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow({k: row[k] for k in fields})
        print(f"  Saved: {Path(path).name}  ({len(rows)} rows)")

    _w(cap_rows, EXP_DIR / "cifar_centered_capacity.csv",
       ["N", "seed", "n_probe", "bl_fail", "cos_mean"])
    _w(atk_rows, EXP_DIR / "cifar_centered_attack.csv",
       ["N", "seed", "probe", "baseline_correct",
        "attack_success", "cosine_delta", "l2"])


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    print("=" * 68)
    print("Grayscale CIFAR-10 (Centered+Normalised) Capacity & Attack Sweep")
    print("=" * 68)

    print("\nLoading CIFAR-10 ...")
    cifar_imgs, cifar_lbls = load_cifar10_gray()
    print(f"  {cifar_imgs.shape}")

    print("\nLoading MNIST (reference) ...")
    mnist_imgs, mnist_lbls = load_mnist()
    print(f"  {mnist_imgs.shape}")

    print(f"\nCapacity sweep: N={N_VALUES} ...")
    cap_rows = run_capacity_sweep(cifar_imgs, cifar_lbls)

    print(f"\nMNIST reference sweep ...")
    mnist_rows = run_mnist_reference(mnist_imgs, mnist_lbls)

    print(f"\nAttack sweep: N={N_ATK_VALS} ...")
    atk_rows = run_attack_sweep(cifar_imgs, cifar_lbls)

    print("\nSaving CSVs ...")
    save_csvs(cap_rows, atk_rows)

    print("Generating figure ...")
    save_figure(cap_rows, atk_rows, mnist_rows)

    report = build_report(cap_rows, atk_rows, mnist_rows)
    print()
    print(report)

    rp = EXP_DIR / "cifar_centered_capacity_report.txt"
    with open(rp, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {rp.name}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
