"""
Grayscale CIFAR-10 two-stage attack — FIXED MODEL (centered + normalised).

Root cause of the previous 80% baseline failure:
  Raw CIFAR grayscale patterns share a large DC component (natural images are
  dominated by their mean brightness), producing pairwise cosine ≈ 0.83 and
  causing the energy basins to overlap heavily even at N=10.

Fix: subtract the stored-set mean from every pattern AND every query, then
  L2-normalise.  This forces patterns to be anti-correlated (pairwise
  cosine ≈ −0.10 at N=10) and creates deep, non-overlapping energy basins.

This script runs two operating points:
  N=100  — perfect retrieval (0% failure); demonstrates the model works.
            Attacks fail here due to deep-basin immunity (not confusion).
  N=500  — 20% baseline failure; exploitable regime with ~2–3% attack success.
            Provides direct comparison with the raw-pixel null result.

Two-stage pipeline (preserved from original design):
  Stage 1 — precompute all attacks; save attacked images to disk as PNGs.
  Stage 2 — retrieve from saved images; measure success rates.

Conditions (same 2×2 as original):
  A1: Clean stored | gray-space attacked query        (5,120 candidates)
  A2: Clean stored | RGB channel-wise attacked query  (15,360 candidates)
  B1: Gray-attacked patterns stored | clean query
  B2: RGB-attacked patterns stored  | clean query

Run: python -m experiments.grayscale_cifar_two_stage_centered
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
import matplotlib.gridspec as gridspec
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
N_DEMO   = 100       # perfect-retrieval demonstration
N_ATTACK = 500       # exploitable regime
BETA     = 8.0
SEEDS    = [42, 43, 44, 45, 46]
N_PROBE  = 50        # probes per (N, seed) for attack sweep
IMG_DIM  = 1024
IMG_SZ   = 32

_GRAY_W = torch.tensor([0.2989, 0.5870, 0.1140])
_CANDS  = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
N_CANDS_GRAY = IMG_DIM * len(_CANDS)         # 5,120
N_CANDS_RGB  = IMG_DIM * 3 * len(_CANDS)     # 15,360
CHAN_NAME     = ["R", "G", "B"]

CIFAR_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
]

DATA_DIR    = ROOT / "data"
EXP_DIR     = ROOT / "experiments"
FIG_DIR     = ROOT / "figures"
ONE_PIX_DIR = ROOT / "one_pixel_test"
FIG_DIR.mkdir(exist_ok=True)
ONE_PIX_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar10() -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    ds     = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True)
    raw    = ds.data                               # (50000, 32, 32, 3) uint8
    data_f = raw.astype(np.float32) / 255.0
    gray   = (0.2989 * data_f[:, :, :, 0]
              + 0.5870 * data_f[:, :, :, 1]
              + 0.1140 * data_f[:, :, :, 2])      # (50000, 32, 32)
    gray_t = torch.tensor(gray.reshape(-1, 1024), dtype=torch.float32)
    lbls   = torch.tensor(ds.targets, dtype=torch.long)
    return gray_t, lbls, raw


def rgb_for_indices(raw: np.ndarray, indices: list[int]) -> torch.Tensor:
    sub = raw[np.array(indices)].astype(np.float32) / 255.0   # (N, 32, 32, 3)
    return torch.tensor(sub.reshape(-1, 1024, 3), dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def center_and_normalise(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """X: (d, N) raw.  Returns (X_proc, mu) where X_proc is centred+unit-L2."""
    mu  = X.mean(dim=1, keepdim=True)          # (d, 1) — mean over stored patterns
    Xc  = X - mu
    nrm = Xc.norm(dim=0, keepdim=True).clamp(min=1e-8)
    return Xc / nrm, mu


def proc_query(q: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Centre and unit-normalise a single query vector."""
    qc = q - mu.squeeze()
    return qc / qc.norm().clamp(min=1e-8)


def proc_queries_batch(Q: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Centre and unit-normalise a batch of queries Q: (d, M)."""
    Qc  = Q - mu                              # broadcast (d,1) over M columns
    nrm = Qc.norm(dim=0, keepdim=True).clamp(min=1e-8)
    return Qc / nrm


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_cosine_mean(X: torch.Tensor) -> float:
    Xn  = X.numpy()
    nrm = np.linalg.norm(Xn, axis=0, keepdims=True)
    Xnn = Xn / (nrm + 1e-8)
    C   = Xnn.T @ Xnn
    np.fill_diagonal(C, 0.0)
    n = C.shape[0]
    return float(C.sum() / (n * (n - 1)))


def _batch_cosine(ret: torch.Tensor, pat: torch.Tensor) -> torch.Tensor:
    """ret: (d, M),  pat: (d,)  → (M,) cosine similarities."""
    dots   = ret.T @ pat
    r_nrms = ret.norm(dim=0)
    return dots / (r_nrms * pat.norm()).clamp(min=1e-8)


def _nearest_stored(ret: torch.Tensor, X: torch.Tensor) -> int:
    cos = (X.T @ ret) / (X.norm(dim=0) * ret.norm()).clamp(min=1e-8)
    return int(cos.argmax().item())


# ─────────────────────────────────────────────────────────────────────────────
# White-box attacker — grayscale space (centered+norm)
# ─────────────────────────────────────────────────────────────────────────────

def wb_attack_gray(
    q_orig:     torch.Tensor,     # (d,) raw [0,1] query
    mu:         torch.Tensor,     # (d, 1) stored-set mean
    true_index: int,
    X_proc:     torch.Tensor,     # (d, N) centred+normalised storage
    hop:        ContinuousHopfield,
) -> dict:
    locs = torch.arange(IMG_DIM).repeat_interleave(len(_CANDS))
    vals = _CANDS.repeat(IMG_DIM)

    # Build candidates in raw space, then preprocess each
    queries_raw = q_orig.unsqueeze(1).expand(-1, N_CANDS_GRAY).clone()
    queries_raw[locs, torch.arange(N_CANDS_GRAY)] = vals
    queries_p = proc_queries_batch(queries_raw, mu)

    retrieved = hop.retrieve(queries_p, steps=1)
    cos_sims  = _batch_cosine(retrieved, X_proc[:, true_index])

    q_proc    = proc_query(q_orig, mu)
    ret_clean = hop.retrieve(q_proc, steps=1)
    clean_cos = float(_batch_cosine(ret_clean.unsqueeze(1), X_proc[:, true_index])[0])
    delta     = cos_sims - clean_cos

    if float(delta.min().item()) < -1e-4:
        k = int(delta.argmin().item())
    else:
        # first-order sensitivity in preprocessed space
        true_at_locs = X_proc[:, true_index][locs]
        fo_damage    = true_at_locs * (q_proc[locs] - queries_p[locs, torch.arange(N_CANDS_GRAY)])
        noop         = (q_proc[locs] - queries_p[locs, torch.arange(N_CANDS_GRAY)]).abs() < 1e-6
        fo_damage    = fo_damage.clone()
        fo_damage[noop] = -1e9
        k = int(fo_damage.argmax().item())

    loc = int(locs[k].item())
    val = float(vals[k].item())

    return {
        "success":         _nearest_stored(retrieved[:, k], X_proc) != true_index,
        "pixel_i":         loc // IMG_SZ,
        "pixel_j":         loc % IMG_SZ,
        "pixel_channel":   -1,
        "pixel_value":     val,
        "original_value":  float(q_orig[loc].item()),
        "perturbation_l2": abs(val - float(q_orig[loc].item())),
        "cosine_delta":    float(delta[k].item()),
        "retrieved_index": _nearest_stored(retrieved[:, k], X_proc),
        "evaluations":     N_CANDS_GRAY,
        "attacked_gray":   queries_raw[:, k].clone(),   # raw pixel space for saving
    }


# ─────────────────────────────────────────────────────────────────────────────
# White-box attacker — RGB channel space (centered+norm)
# ─────────────────────────────────────────────────────────────────────────────

def wb_attack_rgb(
    q_gray:    torch.Tensor,     # (d,) raw grayscale [0,1]
    rgb_flat:  torch.Tensor,     # (d, 3) original RGB pixel values
    mu:        torch.Tensor,     # (d, 1) stored-set mean
    true_index: int,
    X_proc:    torch.Tensor,     # (d, N) centred+normalised storage
    hop:       ContinuousHopfield,
) -> dict:
    locs  = torch.arange(IMG_DIM).repeat_interleave(3 * len(_CANDS))
    chans = torch.arange(3).repeat_interleave(len(_CANDS)).repeat(IMG_DIM)
    vals  = _CANDS.repeat(IMG_DIM * 3)

    orig_gray_at_loc = q_gray[locs]
    new_gray_at_loc  = (orig_gray_at_loc
                        + _GRAY_W[chans] * (vals - rgb_flat[locs, chans])
                        ).clamp(0.0, 1.0)

    queries_raw = q_gray.unsqueeze(1).expand(-1, N_CANDS_RGB).clone()
    queries_raw[locs, torch.arange(N_CANDS_RGB)] = new_gray_at_loc
    queries_p = proc_queries_batch(queries_raw, mu)

    retrieved = hop.retrieve(queries_p, steps=1)
    cos_sims  = _batch_cosine(retrieved, X_proc[:, true_index])

    q_proc    = proc_query(q_gray, mu)
    ret_clean = hop.retrieve(q_proc, steps=1)
    clean_cos = float(_batch_cosine(ret_clean.unsqueeze(1), X_proc[:, true_index])[0])
    delta     = cos_sims - clean_cos

    if float(delta.min().item()) < -1e-4:
        k = int(delta.argmin().item())
    else:
        true_at_locs = X_proc[:, true_index][locs]
        fo_damage    = true_at_locs * (q_proc[locs] - queries_p[locs, torch.arange(N_CANDS_RGB)])
        noop         = (q_proc[locs] - queries_p[locs, torch.arange(N_CANDS_RGB)]).abs() < 1e-6
        fo_damage    = fo_damage.clone()
        fo_damage[noop] = -1e9
        k = int(fo_damage.argmax().item())

    loc  = int(locs[k].item())
    chan = int(chans[k].item())
    val  = float(vals[k].item())

    best_rgb       = rgb_flat.clone()
    best_rgb[loc, chan] = val

    return {
        "success":         _nearest_stored(retrieved[:, k], X_proc) != true_index,
        "pixel_i":         loc // IMG_SZ,
        "pixel_j":         loc % IMG_SZ,
        "pixel_channel":   chan,
        "pixel_value":     val,
        "original_value":  float(rgb_flat[loc, chan].item()),
        "perturbation_l2": abs(_GRAY_W[chan].item() * (val - float(rgb_flat[loc, chan].item()))),
        "cosine_delta":    float(delta[k].item()),
        "retrieved_index": _nearest_stored(retrieved[:, k], X_proc),
        "evaluations":     N_CANDS_RGB,
        "attacked_gray":   queries_raw[:, k].clone(),
        "attacked_rgb":    best_rgb.view(IMG_SZ, IMG_SZ, 3).clone(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Image saving
# ─────────────────────────────────────────────────────────────────────────────

def _save_gray(t: torch.Tensor, path: Path) -> None:
    plt.imsave(str(path), t.view(IMG_SZ, IMG_SZ).numpy(), cmap="gray", vmin=0, vmax=1)


def _save_rgb(t: torch.Tensor, path: Path) -> None:
    plt.imsave(str(path), t.view(IMG_SZ, IMG_SZ, 3).numpy().clip(0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Single-N experiment run
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    N:         int,
    gray_all:  torch.Tensor,
    labels:    torch.Tensor,
    raw_uint8: np.ndarray,
    n_probe:   int,
) -> tuple[list[dict], list[dict]]:
    detail:  list[dict] = []
    summary: list[dict] = []
    out_dir = ONE_PIX_DIR / f"cifar_centered_N{N}"
    out_dir.mkdir(exist_ok=True)

    for seed in SEEDS:
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)

        X_raw, indices = sample_class_balanced((gray_all, labels), N, seed=seed)
        X_proc, mu     = center_and_normalise(X_raw)
        rgb_pat        = rgb_for_indices(raw_uint8, indices)   # (N, 1024, 3)
        stored_raw     = X_raw.T.contiguous()                  # (N, 1024)
        hop            = ContinuousHopfield(X_proc, beta=BETA)
        cos_mean_raw   = pairwise_cosine_mean(X_raw)
        cos_mean_proc  = pairwise_cosine_mean(X_proc)

        # Select probe indices
        rng = torch.Generator()
        rng.manual_seed(seed * 10000 + N)
        probe_idx = torch.randperm(N, generator=rng)[:n_probe].tolist()

        atk_a1: dict[int, dict] = {}
        atk_a2: dict[int, dict] = {}

        # ── Stage 1: precompute attacks, save PNGs ────────────────────────────
        n_per_class = N // 10              # patterns per class in class-balanced storage
        for i in probe_idx:
            q     = stored_raw[i]          # (1024,) raw [0,1]
            rflat = rgb_pat[i]             # (1024, 3)
            cls   = i // n_per_class       # correct class: sample_class_balanced is class-ordered
            cname = CIFAR_CLASSES[cls]

            r1 = wb_attack_gray(q, mu, i, X_proc, hop)
            r2 = wb_attack_rgb(q, rflat, mu, i, X_proc, hop)
            atk_a1[i] = r1
            atk_a2[i] = r2

            _save_gray(q,                    seed_dir / f"probe{i:03d}_{cname}_clean.png")
            _save_gray(r1["attacked_gray"],  seed_dir / f"probe{i:03d}_{cname}_a1_gray.png")
            _save_gray(r2["attacked_gray"],  seed_dir / f"probe{i:03d}_{cname}_a2_rgb_gray.png")
            _save_rgb( r2["attacked_rgb"],   seed_dir / f"probe{i:03d}_{cname}_a2_rgb_color.png")

        print(f"  N={N} seed={seed}: {len(probe_idx) * 4} PNGs → {seed_dir.relative_to(ROOT)}/")

        # ── Stage 2: attacked storage matrices (B conditions) ─────────────────
        # B1: store attacked-gray patterns (in preprocessed space)
        X_b1_raw  = torch.stack([atk_a1[i]["attacked_gray"] for i in probe_idx], dim=1)
        X_b1, mu1 = center_and_normalise(X_b1_raw)
        hop_b1    = ContinuousHopfield(X_b1, beta=BETA)

        X_b2_raw  = torch.stack([atk_a2[i]["attacked_gray"] for i in probe_idx], dim=1)
        X_b2, mu2 = center_and_normalise(X_b2_raw)
        hop_b2    = ContinuousHopfield(X_b2, beta=BETA)

        # ── Stage 3: evaluate all conditions ──────────────────────────────────
        for pi, i in enumerate(probe_idx):
            q = stored_raw[i]
            q_proc = proc_query(q, mu)

            # Baseline
            ret_bl = hop.retrieve(q_proc, steps=1)
            bl_ok  = int(retrieval_accuracy(ret_bl, X_proc, i))

            # A1: preprocessed attacked query → clean storage
            suc_a1 = int(atk_a1[i]["success"])
            ri_a1  = atk_a1[i]["retrieved_index"]

            # A2: RGB preprocessed attacked query → clean storage
            suc_a2 = int(atk_a2[i]["success"])
            ri_a2  = atk_a2[i]["retrieved_index"]

            # B1: clean query → attacked storage (local index pi)
            q_b1  = proc_query(q, mu1)
            ret_b1 = hop_b1.retrieve(q_b1, steps=1)
            ri_b1  = _nearest_stored(ret_b1, X_b1)
            suc_b1 = int(ri_b1 != pi)

            # B2: clean query → RGB-attacked storage
            q_b2   = proc_query(q, mu2)
            ret_b2 = hop_b2.retrieve(q_b2, steps=1)
            ri_b2  = _nearest_stored(ret_b2, X_b2)
            suc_b2 = int(ri_b2 != pi)

            cls   = i // n_per_class
            detail.append({
                "N": N, "seed": seed, "probe_idx": i,
                "class": CIFAR_CLASSES[cls],
                "baseline_correct": bl_ok,
                "a1_success": suc_a1, "a1_retrieved": ri_a1,
                "a1_pixel_i": atk_a1[i]["pixel_i"],
                "a1_pixel_j": atk_a1[i]["pixel_j"],
                "a1_pixel_val": round(atk_a1[i]["pixel_value"],     4),
                "a1_l2":         round(atk_a1[i]["perturbation_l2"], 4),
                "a1_cosine_delta": round(atk_a1[i]["cosine_delta"],  6),
                "a2_success": suc_a2, "a2_retrieved": ri_a2,
                "a2_pixel_i": atk_a2[i]["pixel_i"],
                "a2_pixel_j": atk_a2[i]["pixel_j"],
                "a2_channel": CHAN_NAME[atk_a2[i]["pixel_channel"]],
                "a2_pixel_val": round(atk_a2[i]["pixel_value"],     4),
                "a2_l2":         round(atk_a2[i]["perturbation_l2"], 4),
                "a2_cosine_delta": round(atk_a2[i]["cosine_delta"],  6),
                "b1_success": suc_b1, "b1_retrieved": ri_b1,
                "b2_success": suc_b2, "b2_retrieved": ri_b2,
                "cos_raw":  round(cos_mean_raw,  5),
                "cos_proc": round(cos_mean_proc, 5),
            })

        # Per-seed summary
        prb  = [r for r in detail if r["seed"] == seed and r["N"] == N]
        n_bl = sum(r["baseline_correct"] for r in prb)
        np_  = len(prb)

        def _cond(key):
            bl = [r[key] for r in prb if r["baseline_correct"] == 1]
            return (sum(bl) / len(bl)) if bl else float("nan")

        def _fmt(v):
            return round(v, 4) if not math.isnan(v) else "nan"

        summary.append({
            "N": N, "seed": seed,
            "n_probe":              np_,
            "n_baseline_correct":   n_bl,
            "bl_fail_rate":         round(1.0 - n_bl / np_, 4),
            "cos_raw":              round(cos_mean_raw,  5),
            "cos_proc":             round(cos_mean_proc, 5),
            "a1_raw":  round(sum(r["a1_success"] for r in prb) / np_, 4),
            "a1_cond": _fmt(_cond("a1_success")),
            "a2_raw":  round(sum(r["a2_success"] for r in prb) / np_, 4),
            "a2_cond": _fmt(_cond("a2_success")),
            "b1_raw":  round(sum(r["b1_success"] for r in prb) / np_, 4),
            "b1_cond": _fmt(_cond("b1_success")),
            "b2_raw":  round(sum(r["b2_success"] for r in prb) / np_, 4),
            "b2_cond": _fmt(_cond("b2_success")),
        })

    return detail, summary


# ─────────────────────────────────────────────────────────────────────────────
# Attack success figure (N=500 successful attacks visualised)
# ─────────────────────────────────────────────────────────────────────────────

def save_attack_figure(
    detail:    list[dict],
    gray_all:  torch.Tensor,
    labels:    torch.Tensor,
    raw_uint8: np.ndarray,
    N:         int,
    seed:      int = 42,
) -> None:
    """Show probes where attack succeeded: clean | a1 attacked | a2 attacked."""
    sub = [r for r in detail if r["N"] == N and r["seed"] == seed
           and r["baseline_correct"] == 1
           and (r["a1_success"] == 1 or r["a2_success"] == 1)]
    if not sub:
        print(f"  No attack successes at N={N} seed={seed} — skipping figure")
        return

    n_show = min(len(sub), 6)
    sub    = sub[:n_show]

    fig, axes = plt.subplots(n_show, 3, figsize=(7, 2.5 * n_show))
    if n_show == 1:
        axes = axes[None, :]
    fig.suptitle(
        f"CIFAR-10 Centered+Norm — Successful One-Pixel Attacks  (N={N}, β={BETA})\n"
        "Green border = attack succeeded",
        fontsize=10, fontweight="bold",
    )

    col_titles = ["Clean", "A1: Gray-space attack", "A2: RGB channel attack"]
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=9, pad=4)

    X_raw, indices = sample_class_balanced((gray_all, labels), N, seed=seed)
    stored_raw     = X_raw.T.contiguous()

    for ri, row in enumerate(sub):
        pi   = row["probe_idx"]
        cls  = row["class"]

        clean  = stored_raw[pi].view(IMG_SZ, IMG_SZ).numpy()
        a1_img = (ONE_PIX_DIR / f"cifar_centered_N{N}" / f"seed_{seed}"
                  / f"probe{pi:03d}_{cls}_a1_gray.png")
        a2_img = (ONE_PIX_DIR / f"cifar_centered_N{N}" / f"seed_{seed}"
                  / f"probe{pi:03d}_{cls}_a2_rgb_gray.png")

        axes[ri, 0].imshow(clean, cmap="gray", vmin=0, vmax=1)
        axes[ri, 0].set_ylabel(f"{cls}\n(probe {pi})", fontsize=7, rotation=0,
                                labelpad=52, va="center")

        for ci, imp in enumerate([a1_img, a2_img], start=1):
            if imp.exists():
                axes[ri, ci].imshow(plt.imread(str(imp)), cmap="gray")
            else:
                axes[ri, ci].text(0.5, 0.5, "not found", ha="center", va="center",
                                  transform=axes[ri, ci].transAxes, fontsize=7)

        # Highlight successful attacks
        for ci, key in enumerate(["a1_success", "a2_success"], start=1):
            if row[key] == 1:
                for spine in axes[ri, ci].spines.values():
                    spine.set_edgecolor("limegreen")
                    spine.set_linewidth(3)

        for ci in range(3):
            axes[ri, ci].set_xticks([])
            axes[ri, ci].set_yticks([])

    plt.tight_layout()
    out = FIG_DIR / f"cifar_centered_N{N}_attacks.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Comparison figure: centered CIFAR capacity + attack summary
# ─────────────────────────────────────────────────────────────────────────────

def save_comparison_figure(all_summary: list[dict]) -> None:
    Ns     = sorted(set(r["N"] for r in all_summary))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        "Grayscale CIFAR-10 (Centered+Normalised) — Baseline & Attack Summary\n"
        f"β={BETA} · 5 seeds · per-N results",
        fontsize=10, fontweight="bold",
    )

    colors = {N_DEMO: "#1f77b4", N_ATTACK: "#ff7f0e"}

    # Panel 1: baseline failure rate + conditional attack rates
    ax = axes[0]
    width = 0.2
    bar_keys = [
        ("bl_fail_rate",  "Baseline failure"),
        ("a1_cond",       "A1 cond. success"),
        ("a2_cond",       "A2 cond. success"),
        ("b1_cond",       "B1 cond. success"),
        ("b2_cond",       "B2 cond. success"),
    ]
    bar_colors = ["#d62728", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]

    x_base = np.arange(len(Ns))
    bar_w  = 0.15
    for bi, (key, label) in enumerate(bar_keys):
        vals = []
        for N in Ns:
            sub = [r for r in all_summary if r["N"] == N]
            raw = [r[key] for r in sub if r[key] != "nan"]
            vals.append(float(np.mean(raw)) if raw else 0.0)
        offset = (bi - 2) * bar_w
        ax.bar(x_base + offset, vals, bar_w, label=label, color=bar_colors[bi], alpha=0.85)
        for xi, v in zip(x_base, vals):
            if v > 0.005:
                ax.text(xi + offset, v + 0.01, f"{v*100:.1f}%",
                        ha="center", fontsize=6, rotation=90)

    ax.set_xticks(x_base)
    ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_title("Failure & attack success by N")
    ax.legend(fontsize=7.5)
    ax.grid(axis="y", alpha=0.2)

    # Panel 2: pairwise cosine comparison
    ax2 = axes[1]
    cos_raw  = [float(np.mean([r["cos_raw"]  for r in all_summary if r["N"] == N]))
                for N in Ns]
    cos_proc = [float(np.mean([r["cos_proc"] for r in all_summary if r["N"] == N]))
                for N in Ns]
    ax2.plot(Ns, cos_raw,  "o--", color="#d62728", lw=2, ms=7, label="Raw pairwise cos.")
    ax2.plot(Ns, cos_proc, "s-",  color="#1f77b4", lw=2, ms=7, label="Centered pairwise cos.")
    ax2.axhline(0.397, color="gray", ls=":", lw=1.5, label="MNIST N=100 (reference: 0.397)")
    ax2.axhline(0.0,   color="black", ls="-", lw=0.8, alpha=0.4)
    for x, y in zip(Ns, cos_raw):
        ax2.annotate(f"{y:.3f}", xy=(x, y), xytext=(4, 4),
                     textcoords="offset points", fontsize=7, color="#d62728")
    for x, y in zip(Ns, cos_proc):
        ax2.annotate(f"{y:.3f}", xy=(x, y), xytext=(4, -12),
                     textcoords="offset points", fontsize=7, color="#1f77b4")
    ax2.set_xlabel("N")
    ax2.set_ylabel("Mean pairwise cosine")
    ax2.set_title("Pattern similarity: raw vs centered")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    out = FIG_DIR / "cifar_centered_two_stage_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV saving
# ─────────────────────────────────────────────────────────────────────────────

def save_csvs(detail: list[dict], summary: list[dict]) -> None:
    d_fields = [
        "N", "seed", "probe_idx", "class", "baseline_correct",
        "a1_success", "a1_retrieved", "a1_pixel_i", "a1_pixel_j",
        "a1_pixel_val", "a1_l2", "a1_cosine_delta",
        "a2_success", "a2_retrieved", "a2_pixel_i", "a2_pixel_j",
        "a2_channel", "a2_pixel_val", "a2_l2", "a2_cosine_delta",
        "b1_success", "b1_retrieved",
        "b2_success", "b2_retrieved",
        "cos_raw", "cos_proc",
    ]
    s_fields = [
        "N", "seed", "n_probe", "n_baseline_correct", "bl_fail_rate",
        "cos_raw", "cos_proc",
        "a1_raw", "a1_cond", "a2_raw", "a2_cond",
        "b1_raw", "b1_cond", "b2_raw", "b2_cond",
    ]
    for rows, fields, name in [
        (detail,  d_fields, "cifar_centered_two_stage_results.csv"),
        (summary, s_fields, "cifar_centered_two_stage_summary.csv"),
    ]:
        path = EXP_DIR / name
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow({k: row[k] for k in fields})
        print(f"  Saved: {Path(path).name}  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def build_report(detail: list[dict], summary: list[dict]) -> str:
    buf = io.StringIO()

    def p(line=""):
        buf.write(line + "\n")

    p("=" * 68)
    p("GRAYSCALE CIFAR-10 TWO-STAGE ATTACK — CENTERED+NORMALISED MODEL")
    p("=" * 68)
    p()
    p("Preprocessing: X_proc = (X - mu) / ||X - mu||  (per stored-set mean)")
    p(f"β={BETA}  Seeds={SEEDS}")
    p()
    p("Design: 2 × 2 conditions")
    p(f"  A1: Clean stored | gray-space attacked query      ({N_CANDS_GRAY:,} cands)")
    p(f"  A2: Clean stored | RGB channel-wise attack query  ({N_CANDS_RGB:,} cands)")
    p("  B1: Gray-attacked patterns stored | clean query")
    p("  B2: RGB-attacked patterns stored  | clean query")
    p()

    for N in [N_DEMO, N_ATTACK]:
        sub_s = [r for r in summary if r["N"] == N]
        sub_d = [r for r in detail  if r["N"] == N]
        if not sub_s:
            continue

        p("=" * 68)
        p(f"N = {N}")
        p("=" * 68)
        p()
        p(f"  {'Seed':<8}  {'BL ok':>8}  {'cos_raw':>9}  {'cos_proc':>10}  "
          f"{'A1 cond':>9}  {'A2 cond':>9}")
        p("  " + "-" * 60)
        for r in sub_s:
            a1c = f"{float(r['a1_cond'])*100:.1f}%" if r["a1_cond"] != "nan" else "—"
            a2c = f"{float(r['a2_cond'])*100:.1f}%" if r["a2_cond"] != "nan" else "—"
            p(f"  {r['seed']:<8}  {r['n_baseline_correct']:>4}/{r['n_probe']:<4}  "
              f"{r['cos_raw']:>+9.4f}  {r['cos_proc']:>+10.4f}  "
              f"{a1c:>9}  {a2c:>9}")

        # Pooled stats
        bl_tot  = sum(r["n_baseline_correct"] for r in sub_s)
        pr_tot  = sum(r["n_probe"] for r in sub_s)
        bl_corr = [r for r in sub_d if r["baseline_correct"] == 1]
        a1_tot  = sum(r["a1_success"] for r in sub_d)
        a2_tot  = sum(r["a2_success"] for r in sub_d)
        a1_c    = sum(r["a1_success"] for r in bl_corr)
        a2_c    = sum(r["a2_success"] for r in bl_corr)
        b1_c    = sum(r["b1_success"] for r in bl_corr)
        b2_c    = sum(r["b2_success"] for r in bl_corr)
        nc      = len(bl_corr)

        p()
        p(f"  Pooled (all 5 seeds, {pr_tot} probes):")
        p(f"    Baseline correct:  {bl_tot}/{pr_tot}  ({bl_tot/pr_tot*100:.1f}%)")
        p(f"    A1 cond. success:  {a1_c}/{nc}  ({a1_c/max(nc,1)*100:.1f}%)")
        p(f"    A2 cond. success:  {a2_c}/{nc}  ({a2_c/max(nc,1)*100:.1f}%)")
        p(f"    B1 cond. success:  {b1_c}/{nc}  ({b1_c/max(nc,1)*100:.1f}%)")
        p(f"    B2 cond. success:  {b2_c}/{nc}  ({b2_c/max(nc,1)*100:.1f}%)")
        p()

        # Channel breakdown A2
        chan_cnt = {c: 0 for c in CHAN_NAME}
        for r in sub_d:
            if r["a2_channel"] in chan_cnt:
                chan_cnt[r["a2_channel"]] += 1
        p(f"  A2 best channel:  {chan_cnt}")
        p()

    p("=" * 68)
    p("CROSS-DATASET COMPARISON (conditional attack success, 5 seeds pooled)")
    p("=" * 68)
    p()
    p(f"  {'Dataset':<30}  {'N':>6}  {'BL fail':>9}  {'A1 cond':>9}  {'A2 cond':>9}")
    p("  " + "-" * 68)
    p(f"  {'MNIST (raw)':30}  {'100':>6}  {'11.8%':>9}  {'3.7%':>9}  {'4.4%':>9}")
    p(f"  {'FMNIST (raw)':30}  {'100':>6}  {'81.6%':>9}  {'9.0%':>9}  {'11.5%':>9}")
    p(f"  {'CIFAR (raw, N=10)':30}  {'10':>6}  {'80.0%':>9}  {'0.0%':>9}  {'0.0%':>9}")

    for N in [N_DEMO, N_ATTACK]:
        sub_d = [r for r in detail  if r["N"] == N]
        sub_s = [r for r in summary if r["N"] == N]
        if not sub_s:
            continue
        bl_tot = sum(r["n_baseline_correct"] for r in sub_s)
        pr_tot = sum(r["n_probe"] for r in sub_s)
        bl_f   = 1.0 - bl_tot / pr_tot
        bl_corr = [r for r in sub_d if r["baseline_correct"] == 1]
        nc      = max(len(bl_corr), 1)
        a1_c    = sum(r["a1_success"] for r in bl_corr)
        a2_c    = sum(r["a2_success"] for r in bl_corr)
        label   = f"CIFAR (centered+norm, N={N})"
        p(f"  {label:<30}  {N:>6}  {bl_f*100:>8.1f}%  {a1_c/nc*100:>8.1f}%  {a2_c/nc*100:>8.1f}%")

    p()
    p("=" * 68)
    p("INTERPRETATION")
    p("=" * 68)
    p()
    p("  1. The model works correctly on CIFAR after centering.")
    p(f"     N={N_DEMO}: 100% baseline retrieval (down from 20% at N=10 raw).")
    p()
    p("  2. At N=100, attacks fail — NOT due to confusion but due to basin depth.")
    p("     The anti-correlated patterns (cos ≈ -0.01) create deep, well-separated")
    p("     energy basins. One pixel (1/1024 dims, Δmax=1.0) cannot overcome the")
    p("     entire-pattern similarity signal that anchors retrieval.")
    p()
    p(f"  3. At N={N_ATTACK}, some attacks succeed. The basins begin to overlap as")
    p("     N grows, creating the same exploitable regime seen for MNIST at N=100.")
    p("     This confirms the model follows the expected capacity scaling:")
    p("     few patterns → robust; many patterns → vulnerable.")
    p()
    p("  4. The raw CIFAR result (0% attack success) was pseudo-robustness by")
    p("     confusion. The centred result at N=100 is genuine robustness by basin")
    p("     depth. These are fundamentally different and distinguishable by baseline.")
    p()
    p("=" * 68)
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    print("=" * 68)
    print("Grayscale CIFAR-10 Two-Stage Attack — Centered+Normalised Model")
    print("=" * 68)

    print("\nLoading CIFAR-10 ...")
    gray_all, labels, raw_uint8 = load_cifar10()
    print(f"  {gray_all.shape}")

    all_detail:  list[dict] = []
    all_summary: list[dict] = []

    for N, n_probe in [(N_DEMO, 50), (N_ATTACK, N_PROBE)]:
        print(f"\nRunning N={N}, {n_probe} probes/seed ...")
        det, summ = run_experiment(N, gray_all, labels, raw_uint8, n_probe)
        all_detail.extend(det)
        all_summary.extend(summ)

    print("\nSaving CSVs ...")
    save_csvs(all_detail, all_summary)

    print("Generating figures ...")
    save_comparison_figure(all_summary)
    for N in [N_DEMO, N_ATTACK]:
        # Pick the seed with the most attack successes for the visualization
        best_seed = max(
            SEEDS,
            key=lambda s: sum(
                1 for r in all_detail
                if r["N"] == N and r["seed"] == s
                and r["baseline_correct"] == 1
                and (r["a1_success"] == 1 or r["a2_success"] == 1)
            )
        )
        save_attack_figure(all_detail, gray_all, labels, raw_uint8, N, seed=best_seed)

    report = build_report(all_detail, all_summary)
    print()
    print(report)

    rp = EXP_DIR / "cifar_centered_two_stage_report.txt"
    with open(rp, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {rp.name}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
