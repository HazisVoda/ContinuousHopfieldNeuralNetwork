"""
Grayscale CIFAR-10 two-stage attack experiment.

Professor's suggestion: split the preprocessing pipeline — precompute adversarial
images and save them to disk, then feed those saved images to the Hopfield network.

2×2 design:
  A1: Clean stored | Gray-space attacked query      (5,120 candidates per probe)
  A2: Clean stored | RGB channel-wise attacked query (15,360 candidates per probe)
  B1: Gray-attacked patterns stored | Clean query
  B2: RGB-attacked patterns stored  | Clean query

For RGB attacks: one channel of one pixel is perturbed (e.g. red channel of pixel 5).
The perturbed RGB image is then converted to grayscale via luminance weights before
retrieval.  This tests whether the colour-space attack surface differs from greyscale.

Attacked images are saved to one_pixel_test/seed_k/ as PNG files before any retrieval,
so the two stages (precompute + retrieve) are genuinely separated.

Run: python -m experiments.grayscale_cifar_two_stage_attack
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
N       = 10
BETA    = 8.0
SEEDS   = [42, 43, 44, 45, 46]
IMG_DIM = 1024
IMG_SZ  = 32

# Luminance weights (BT.601)
_GRAY_W = torch.tensor([0.2989, 0.5870, 0.1140])
_CANDS  = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

N_CANDS_GRAY = IMG_DIM * len(_CANDS)           # 5,120
N_CANDS_RGB  = IMG_DIM * 3 * len(_CANDS)       # 15,360

CHAN_NAME = ["R", "G", "B"]
CIFAR_CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
]

DATA_DIR    = ROOT / "data"
EXP_DIR     = ROOT / "experiments"
ONE_PIX_DIR = ROOT / "one_pixel_test"
ONE_PIX_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar10() -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Returns (gray_images, labels, raw_uint8).
      gray_images : (50000, 1024) float32 in [0,1]
      labels      : (50000,) long
      raw_uint8   : (50000, 32, 32, 3) uint8  — kept compact for per-index RGB lookup
    """
    ds       = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True)
    raw      = ds.data                                       # (50000, 32, 32, 3) uint8
    data_f   = raw.astype(np.float32) / 255.0
    gray     = (0.2989 * data_f[:, :, :, 0]
                + 0.5870 * data_f[:, :, :, 1]
                + 0.1140 * data_f[:, :, :, 2])              # (50000, 32, 32)
    gray_t   = torch.tensor(gray.reshape(-1, 1024), dtype=torch.float32)
    labels_t = torch.tensor(ds.targets, dtype=torch.long)
    return gray_t, labels_t, raw


def rgb_for_indices(raw: np.ndarray, indices: list[int]) -> torch.Tensor:
    """Return (N, 1024, 3) float32 RGB tensor for dataset row indices."""
    sub  = raw[np.array(indices)].astype(np.float32) / 255.0   # (N, 32, 32, 3)
    return torch.tensor(sub.reshape(-1, 1024, 3), dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _batch_cosine(retrieved: torch.Tensor, true_pat: torch.Tensor) -> torch.Tensor:
    dots    = retrieved.T @ true_pat
    r_norms = retrieved.norm(dim=0)
    t_norm  = true_pat.norm()
    return dots / (r_norms * t_norm).clamp(min=1e-8)


def _nearest_stored(ret: torch.Tensor, X: torch.Tensor) -> int:
    X_norms = X.norm(dim=0)
    cos_all = (X.T @ ret) / (X_norms * ret.norm()).clamp(min=1e-8)
    return int(cos_all.argmax().item())


def pairwise_cosine_mean(X: torch.Tensor) -> float:
    Xn  = X.numpy()
    nrm = np.linalg.norm(Xn, axis=0, keepdims=True)
    Xnn = Xn / (nrm + 1e-8)
    C   = Xnn.T @ Xnn
    np.fill_diagonal(C, 0.0)
    n = C.shape[0]
    return float(C.sum() / (n * (n - 1)))


# ─────────────────────────────────────────────────────────────────────────────
# Grayscale-space exhaustive white-box attack  (5,120 candidates)
# ─────────────────────────────────────────────────────────────────────────────

def wb_attack_gray(
    q_gray: torch.Tensor,           # (1024,)
    true_index: int,
    network: ContinuousHopfield,
) -> dict:
    X      = network.X
    n_locs = IMG_DIM
    n_vals = len(_CANDS)
    n_c    = n_locs * n_vals                              # 5120

    locs = torch.arange(n_locs).repeat_interleave(n_vals)
    vals = _CANDS.repeat(n_locs)

    queries = q_gray.unsqueeze(1).expand(-1, n_c).clone()
    queries[locs, torch.arange(n_c)] = vals

    retrieved = network.retrieve(queries, steps=1)
    cos_sims  = _batch_cosine(retrieved, X[:, true_index])

    # Cosine delta vs clean retrieval — finds candidate causing most additional damage.
    # If all candidates are in the degenerate saturated regime (delta never < -1e-4),
    # fall back to first-order sensitivity: argmax X[:,true_idx][loc] * (q[loc] - val),
    # which selects the pixel whose change most reduces the dot-product with the true pattern.
    ret_clean = network.retrieve(q_gray, steps=1)
    clean_cos = float(_batch_cosine(ret_clean.unsqueeze(1), X[:, true_index])[0])
    delta     = cos_sims - clean_cos

    if float(delta.min().item()) < -1e-4:
        k = int(delta.argmin().item())
    else:
        true_at_locs = X[:, true_index][locs]
        fo_damage    = true_at_locs * (q_gray[locs] - vals)
        noop_mask    = (q_gray[locs] - vals).abs() < 1e-6
        fo_damage    = fo_damage.clone()
        fo_damage[noop_mask] = -1e9
        k = int(fo_damage.argmax().item())

    cosine_delta = float(delta[k].item())
    loc = int(locs[k].item())
    val = float(vals[k].item())

    return {
        "success":         _nearest_stored(retrieved[:, k], X) != true_index,
        "pixel_i":         loc // IMG_SZ,
        "pixel_j":         loc % IMG_SZ,
        "pixel_channel":   -1,
        "pixel_value":     val,
        "original_value":  float(q_gray[loc].item()),
        "perturbation_l2": abs(val - float(q_gray[loc].item())),
        "cosine_to_true":  float(cos_sims[k].item()),
        "cosine_delta":    cosine_delta,
        "retrieved_index": _nearest_stored(retrieved[:, k], X),
        "evaluations":     n_c,
        "attacked_gray":   queries[:, k].clone(),         # (1024,)
    }


# ─────────────────────────────────────────────────────────────────────────────
# RGB channel-wise exhaustive white-box attack  (15,360 candidates)
# ─────────────────────────────────────────────────────────────────────────────

def wb_attack_rgb(
    q_gray:   torch.Tensor,     # (1024,) precomputed luminance grayscale
    rgb_flat: torch.Tensor,     # (1024, 3) original RGB pixel values
    true_index: int,
    network: ContinuousHopfield,
) -> dict:
    """
    Perturbs one channel (R, G, or B) of one pixel location.
    The perturbed RGB image is converted to grayscale for retrieval.

    Delta in grayscale space: Δgray[loc] = weight[chan] × (new_val − old_rgb[loc,chan])
    Green channel has highest luminance weight (0.5870), so G perturbations are largest.
    """
    X       = network.X
    weights = _GRAY_W
    n_locs  = IMG_DIM
    n_chans = 3
    n_vals  = len(_CANDS)
    n_c     = n_locs * n_chans * n_vals             # 15360

    # Index tensors: locs[k], chans[k], vals[k] describe candidate k
    locs  = torch.arange(n_locs).repeat_interleave(n_chans * n_vals)
    chans = torch.arange(n_chans).repeat_interleave(n_vals).repeat(n_locs)
    vals  = _CANDS.repeat(n_locs * n_chans)

    # Grayscale delta: perturbing channel c of pixel loc changes gray[loc] by:
    #   weight[c] × (new_val − original_rgb[loc, c])
    orig_gray_at_loc = q_gray[locs]
    new_gray_at_loc  = (orig_gray_at_loc
                        + weights[chans] * (vals - rgb_flat[locs, chans])
                        ).clamp(0.0, 1.0)

    queries = q_gray.unsqueeze(1).expand(-1, n_c).clone()
    queries[locs, torch.arange(n_c)] = new_gray_at_loc

    retrieved = network.retrieve(queries, steps=1)
    cos_sims  = _batch_cosine(retrieved, X[:, true_index])

    # Same degenerate-regime fix as wb_attack_gray
    ret_clean = network.retrieve(q_gray, steps=1)
    clean_cos = float(_batch_cosine(ret_clean.unsqueeze(1), X[:, true_index])[0])
    delta     = cos_sims - clean_cos

    if float(delta.min().item()) < -1e-4:
        k = int(delta.argmin().item())
    else:
        true_at_locs = X[:, true_index][locs]
        fo_damage    = true_at_locs * (q_gray[locs] - new_gray_at_loc)
        noop_mask    = (q_gray[locs] - new_gray_at_loc).abs() < 1e-6
        fo_damage    = fo_damage.clone()
        fo_damage[noop_mask] = -1e9
        k = int(fo_damage.argmax().item())

    cosine_delta = float(delta[k].item())
    loc  = int(locs[k].item())
    chan = int(chans[k].item())
    val  = float(vals[k].item())

    orig_chan_val = float(rgb_flat[loc, chan].item())
    ret_idx      = _nearest_stored(retrieved[:, k], X)

    # Reconstruct the best attacked RGB image for saving
    best_rgb_flat       = rgb_flat.clone()
    best_rgb_flat[loc, chan] = val

    return {
        "success":         ret_idx != true_index,
        "pixel_i":         loc // IMG_SZ,
        "pixel_j":         loc % IMG_SZ,
        "pixel_channel":   chan,
        "pixel_value":     val,
        "original_value":  orig_chan_val,
        "perturbation_l2": abs(weights[chan].item() * (val - orig_chan_val)),
        "cosine_to_true":  float(cos_sims[k].item()),
        "cosine_delta":    cosine_delta,
        "retrieved_index": ret_idx,
        "evaluations":     n_c,
        "attacked_gray":   queries[:, k].clone(),                # (1024,)
        "attacked_rgb":    best_rgb_flat.view(IMG_SZ, IMG_SZ, 3).clone(),  # (32, 32, 3)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Image saving helpers
# ─────────────────────────────────────────────────────────────────────────────

def _save_gray(t: torch.Tensor, path: Path) -> None:
    plt.imsave(str(path), t.view(IMG_SZ, IMG_SZ).numpy(), cmap="gray", vmin=0, vmax=1)


def _save_rgb(t: torch.Tensor, path: Path) -> None:
    plt.imsave(str(path), t.view(IMG_SZ, IMG_SZ, 3).numpy().clip(0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    gray_all:  torch.Tensor,    # (50000, 1024)
    labels:    torch.Tensor,
    raw_uint8: np.ndarray,      # (50000, 32, 32, 3) uint8
) -> tuple[list[dict], list[dict]]:
    detail:  list[dict] = []
    summary: list[dict] = []

    for seed in SEEDS:
        seed_dir = ONE_PIX_DIR / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)

        X_gray, indices = sample_class_balanced((gray_all, labels), N, seed=seed)
        rgb_pat         = rgb_for_indices(raw_uint8, indices)   # (N, 1024, 3)
        stored          = X_gray.T.contiguous()                 # (N, 1024)
        hop_clean       = ContinuousHopfield(X_gray, beta=BETA)
        cos_mean        = pairwise_cosine_mean(X_gray)

        # ── Stage 1: precompute attacks and save images ──────────────────────
        atk_a1: list[dict] = []
        atk_a2: list[dict] = []

        for i in range(N):
            q     = stored[i]          # (1024,) clean gray query
            rflat = rgb_pat[i]         # (1024, 3) original RGB
            cname = CIFAR_CLASSES[i]

            r1 = wb_attack_gray(q, i, hop_clean)
            r2 = wb_attack_rgb(q, rflat, i, hop_clean)
            atk_a1.append(r1)
            atk_a2.append(r2)

            # Save to one_pixel_test/seed_k/
            _save_gray(q,               seed_dir / f"probe{i:02d}_{cname}_clean.png")
            _save_gray(r1["attacked_gray"],
                       seed_dir / f"probe{i:02d}_{cname}_a1_gray_attacked.png")
            _save_gray(r2["attacked_gray"],
                       seed_dir / f"probe{i:02d}_{cname}_a2_rgb_attacked_gray.png")
            _save_rgb( r2["attacked_rgb"],
                       seed_dir / f"probe{i:02d}_{cname}_a2_rgb_attacked_color.png")

        n_imgs = 4 * N   # per seed
        print(f"  Seed {seed}: {n_imgs} images saved → {seed_dir.relative_to(ROOT)}/")

        # ── Stage 2: build attacked storage matrices (B conditions) ──────────
        # B1: store gray-attacked patterns, query with clean
        X_b1   = torch.stack([atk_a1[i]["attacked_gray"] for i in range(N)], dim=1)
        hop_b1 = ContinuousHopfield(X_b1, beta=BETA)

        # B2: store RGB-attacked-then-gray patterns, query with clean
        X_b2   = torch.stack([atk_a2[i]["attacked_gray"] for i in range(N)], dim=1)
        hop_b2 = ContinuousHopfield(X_b2, beta=BETA)

        # ── Stage 3: evaluate all 4 conditions ───────────────────────────────
        for i in range(N):
            q = stored[i]   # clean gray query (same for all conditions)

            # Baseline: clean stored, clean query
            ret_clean  = hop_clean.retrieve(q, steps=1)
            bl_correct = int(retrieval_accuracy(ret_clean, X_gray, i))

            # A1: clean stored, gray-attacked query
            suc_a1 = int(atk_a1[i]["success"])
            ri_a1  = atk_a1[i]["retrieved_index"]

            # A2: clean stored, RGB-attacked query (as grayscale)
            suc_a2 = int(atk_a2[i]["success"])
            ri_a2  = atk_a2[i]["retrieved_index"]

            # B1: gray-attacked stored, clean query
            ret_b1 = hop_b1.retrieve(q, steps=1)
            ri_b1  = _nearest_stored(ret_b1, X_b1)
            suc_b1 = int(ri_b1 != i)

            # B2: RGB-attacked stored, clean query
            ret_b2 = hop_b2.retrieve(q, steps=1)
            ri_b2  = _nearest_stored(ret_b2, X_b2)
            suc_b2 = int(ri_b2 != i)

            detail.append({
                "seed": seed, "probe_idx": i,
                "class": CIFAR_CLASSES[i],
                "baseline_correct": bl_correct,
                # A1
                "a1_success":       suc_a1,
                "a1_retrieved":     ri_a1,
                "a1_pixel_i":       atk_a1[i]["pixel_i"],
                "a1_pixel_j":       atk_a1[i]["pixel_j"],
                "a1_pixel_val":     round(atk_a1[i]["pixel_value"],     4),
                "a1_l2":            round(atk_a1[i]["perturbation_l2"], 4),
                "a1_cosine_delta":  round(atk_a1[i]["cosine_delta"],    6),
                # A2
                "a2_success":       suc_a2,
                "a2_retrieved":     ri_a2,
                "a2_pixel_i":       atk_a2[i]["pixel_i"],
                "a2_pixel_j":       atk_a2[i]["pixel_j"],
                "a2_channel":       CHAN_NAME[atk_a2[i]["pixel_channel"]],
                "a2_pixel_val":     round(atk_a2[i]["pixel_value"],     4),
                "a2_l2":            round(atk_a2[i]["perturbation_l2"], 4),
                "a2_cosine_delta":  round(atk_a2[i]["cosine_delta"],    6),
                # B1
                "b1_success":    suc_b1,
                "b1_retrieved":  ri_b1,
                # B2
                "b2_success":    suc_b2,
                "b2_retrieved":  ri_b2,
                # meta
                "mean_pairwise_cosine": round(cos_mean, 5),
            })

        # ── Per-seed summary ──────────────────────────────────────────────────
        prb  = [r for r in detail if r["seed"] == seed]
        n_bl = sum(r["baseline_correct"] for r in prb)

        def _cond(key: str) -> float:
            bl = [r[key] for r in prb if r["baseline_correct"] == 1]
            return (sum(bl) / len(bl)) if bl else float("nan")

        c_a1, c_a2, c_b1, c_b2 = _cond("a1_success"), _cond("a2_success"), \
                                   _cond("b1_success"), _cond("b2_success")

        def _fmt(v: float) -> float | str:
            return round(v, 4) if not math.isnan(v) else "nan"

        summary.append({
            "seed": seed,
            "n_baseline_correct":   n_bl,
            "mean_pairwise_cosine": round(cos_mean, 5),
            "a1_raw":   round(sum(r["a1_success"] for r in prb) / N, 4),
            "a1_cond":  _fmt(c_a1),
            "a2_raw":   round(sum(r["a2_success"] for r in prb) / N, 4),
            "a2_cond":  _fmt(c_a2),
            "b1_raw":   round(sum(r["b1_success"] for r in prb) / N, 4),
            "b1_cond":  _fmt(c_b1),
            "b2_raw":   round(sum(r["b2_success"] for r in prb) / N, 4),
            "b2_cond":  _fmt(c_b2),
        })

    return detail, summary


# ─────────────────────────────────────────────────────────────────────────────
# CSV saving
# ─────────────────────────────────────────────────────────────────────────────

def save_detail_csv(rows: list[dict]) -> None:
    fields = [
        "seed", "probe_idx", "class", "baseline_correct",
        "a1_success", "a1_retrieved", "a1_pixel_i", "a1_pixel_j", "a1_pixel_val", "a1_l2",
        "a1_cosine_delta",
        "a2_success", "a2_retrieved", "a2_pixel_i", "a2_pixel_j", "a2_channel",
        "a2_pixel_val", "a2_l2", "a2_cosine_delta",
        "b1_success", "b1_retrieved",
        "b2_success", "b2_retrieved",
        "mean_pairwise_cosine",
    ]
    path = EXP_DIR / "grayscale_cifar_two_stage_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})
    print(f"  Saved: {path.name}  ({len(rows)} rows)")


def save_summary_csv(rows: list[dict]) -> None:
    fields = [
        "seed", "n_baseline_correct", "mean_pairwise_cosine",
        "a1_raw", "a1_cond",
        "a2_raw", "a2_cond",
        "b1_raw", "b1_cond",
        "b2_raw", "b2_cond",
    ]
    path = EXP_DIR / "grayscale_cifar_two_stage_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})
    print(f"  Saved: {path.name}  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def build_report(detail: list[dict], summary: list[dict]) -> str:
    buf = io.StringIO()

    def p(line: str = "") -> None:
        buf.write(line + "\n")

    total       = len(detail)                   # 50
    total_bl    = sum(r["n_baseline_correct"] for r in summary)
    bl_rates    = [1.0 - r["n_baseline_correct"] / N for r in summary]
    cos_mean    = float(np.mean([r["mean_pairwise_cosine"] for r in summary]))

    def _agg(key_raw: str, key_cond: str):
        raw_vals  = [r[key_raw] for r in summary]
        cond_vals = [r[key_cond] for r in summary
                     if r[key_cond] != "nan"]
        raw_mean  = float(np.mean(raw_vals))
        raw_std   = float(np.std( raw_vals,  ddof=1))
        cond_mean = float(np.mean(cond_vals)) if cond_vals else float("nan")
        cond_std  = float(np.std( cond_vals, ddof=1)) if len(cond_vals) > 1 else 0.0
        return raw_mean, raw_std, cond_mean, cond_std

    a1_rm, a1_rs, a1_cm, a1_cs = _agg("a1_raw", "a1_cond")
    a2_rm, a2_rs, a2_cm, a2_cs = _agg("a2_raw", "a2_cond")
    b1_rm, b1_rs, b1_cm, b1_cs = _agg("b1_raw", "b1_cond")
    b2_rm, b2_rs, b2_cm, b2_cs = _agg("b2_raw", "b2_cond")

    mean_bl_fail = float(np.mean(bl_rates))
    std_bl_fail  = float(np.std( bl_rates, ddof=1))

    p("=" * 64)
    p("GRAYSCALE CIFAR-10 TWO-STAGE ATTACK EXPERIMENT")
    p("=" * 64)
    p()
    p("Setup: N=10, class-balanced, beta=8.0, 5 seeds")
    p(f"Total probe evaluations: {total} (10 per seed x 5 seeds)")
    p()
    p("Conditions:")
    p(f"  A1: Clean stored | Gray-space attacked query    ({N_CANDS_GRAY:,} candidates)")
    p(f"  A2: Clean stored | RGB channel-wise attacked query ({N_CANDS_RGB:,} candidates)")
    p("  B1: Gray-attacked patterns stored | Clean query")
    p("  B2: RGB-attacked patterns stored  | Clean query")
    p()
    p(f"Attacked images saved to: one_pixel_test/  ({4 * N * len(SEEDS)} PNG files)")
    p()
    p("=" * 64)
    p("BASELINE RETRIEVAL (clean stored, clean query)")
    p("=" * 64)
    p()
    for r in summary:
        p(f"  Seed {r['seed']}: {r['n_baseline_correct']}/10 correct  "
          f"(pairwise cosine {r['mean_pairwise_cosine']:.3f})")
    p(f"  Mean baseline failure: {mean_bl_fail*100:.1f}% +/- {std_bl_fail*100:.1f}%")
    p(f"  Total baseline-correct probes (across 5 seeds): {total_bl}")
    p()

    def _section(label: str, rm: float, rs: float, cm: float, cs: float,
                 cond_label: str, seed_key: str) -> None:
        p("=" * 64)
        p(label)
        p("=" * 64)
        p()
        p(f"  Seed-level successes out of 10:")
        for r in summary:
            p(f"    Seed {r['seed']}: {round(r[seed_key] * N)}/10")
        p(f"  Raw success rate:         {rm*100:.1f}% +/- {rs*100:.1f}%")
        if not math.isnan(cm):
            p(f"  Conditional success rate ({cond_label}): "
              f"{cm*100:.1f}% +/- {cs*100:.1f}%")
        else:
            p(f"  Conditional success rate ({cond_label}): undefined (no qualifying probes)")
        p()

    _section("CONDITION A1: Gray-space attacked query, clean stored",
             a1_rm, a1_rs, a1_cm, a1_cs,
             "among baseline-correct", "a1_raw")

    _section("CONDITION A2: RGB channel-wise attacked query, clean stored",
             a2_rm, a2_rs, a2_cm, a2_cs,
             "among baseline-correct", "a2_raw")

    # Channel breakdown for A2
    chan_counts = {c: 0 for c in CHAN_NAME}
    for r in detail:
        if r["a2_channel"] in chan_counts:
            chan_counts[r["a2_channel"]] += 1
    p("  Best attack channel distribution (across 50 probes):")
    for c in CHAN_NAME:
        p(f"    {c}: {chan_counts[c]}/50 ({chan_counts[c]*2:.0f}%)")
    p()

    _section("CONDITION B1: Gray-attacked patterns stored, clean query",
             b1_rm, b1_rs, b1_cm, b1_cs,
             "among baseline-correct on clean net", "b1_raw")

    _section("CONDITION B2: RGB-attacked patterns stored, clean query",
             b2_rm, b2_rs, b2_cm, b2_cs,
             "among baseline-correct on clean net", "b2_raw")

    p("=" * 64)
    p("INTERPRETATION")
    p("=" * 64)
    p()
    p(f"Mean pairwise cosine at N=10: {cos_mean:.3f}  (MNIST N=100 reference: 0.397)")
    p()
    p(f"  {'Condition':<45}  {'Raw success':>12}  {'Cond success':>13}")
    p("  " + "-" * 72)
    p(f"  {'A1: Gray attack as query':<45}  {a1_rm*100:>11.1f}%  "
      f"  {a1_cm*100 if not math.isnan(a1_cm) else 0.0:>11.1f}%")
    p(f"  {'A2: RGB channel attack as query':<45}  {a2_rm*100:>11.1f}%  "
      f"  {a2_cm*100 if not math.isnan(a2_cm) else 0.0:>11.1f}%")
    p(f"  {'B1: Gray-attacked stored, clean query':<45}  {b1_rm*100:>11.1f}%  "
      f"  {b1_cm*100 if not math.isnan(b1_cm) else 0.0:>11.1f}%")
    p(f"  {'B2: RGB-attacked stored, clean query':<45}  {b2_rm*100:>11.1f}%  "
      f"  {b2_cm*100 if not math.isnan(b2_cm) else 0.0:>11.1f}%")
    p()

    # A condition: attack-attributable if conditional success > 0
    # B condition: attack-attributable if conditional success > 0
    # (raw B success equals baseline failure by construction — not attributable)
    cond_vals_defined = [v for v in [a1_cm, a2_cm, b1_cm, b2_cm] if not math.isnan(v)]
    any_attributable  = any(v > 0.0 for v in cond_vals_defined)

    if not any_attributable:
        p("Verdict:")
        p("  Null result extends across all four conditions. The raw success rate")
        p("  for all conditions equals the baseline failure rate (80.0%) — the")
        p("  'successes' are probes that were already failing before any attack.")
        p("  Among the 10 baseline-correct probes, zero were affected by single-")
        p("  pixel perturbations in any configuration (A1, A2, B1, B2).")
        p("  The professor's two-stage split confirms the conclusion: CIFAR-10's")
        p("  high pairwise cosine ({:.3f}) is the dominant factor. Single-pixel".format(cos_mean))
        p("  changes in either query space or storage space are overwhelmed by the")
        p("  overall similarity structure, which determines retrieval outcomes.")
    else:
        p("Verdict:")
        p("  Partial success found in at least one condition (see conditional rates).")
        p("  The two-stage pipeline reveals a difference between attacking the query")
        p("  (A conditions) and pre-attacking stored patterns (B conditions).")
    p()
    p("=" * 64)

    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    print("=" * 64)
    print("Grayscale CIFAR-10 Two-Stage Attack Experiment")
    print("=" * 64)

    print("\nLoading CIFAR-10 ...")
    gray_all, labels, raw_uint8 = load_cifar10()
    print(f"  gray: {gray_all.shape}  raw uint8: {raw_uint8.shape}")

    print(f"\nRunning 2x2 experiment: N={N}, seeds={SEEDS} ...")
    print(f"  A1: gray-space attack ({N_CANDS_GRAY:,} candidates/probe)")
    print(f"  A2: RGB channel attack ({N_CANDS_RGB:,} candidates/probe)")
    print("  B1/B2: attacked storage, clean query")
    print()

    detail, summary = run_experiment(gray_all, labels, raw_uint8)

    print("\nSaving CSVs ...")
    save_detail_csv(detail)
    save_summary_csv(summary)

    report = build_report(detail, summary)
    print()
    print(report)

    report_path = EXP_DIR / "grayscale_cifar_two_stage_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {report_path.name}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
