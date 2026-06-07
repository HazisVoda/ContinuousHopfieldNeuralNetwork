"""
Grayscale CIFAR-10 one-pixel attack — null result documentation.

Formally documents that white-box exhaustive one-pixel attacks produce zero
attack-attributable failures on grayscale CIFAR-10 stored in a continuous
Hopfield network with N=10, class-balanced, β=8.0.

The WhiteBoxOnePixelAttacker in hopfield/attacks.py hardcodes n_locs=784 for
28×28 MNIST.  For 32×32 CIFAR (1024-dim) the same exhaustive logic is
implemented inline below with n_locs=1024, identical in every other respect.

Run: python -m experiments.grayscale_cifar_attack
"""

from __future__ import annotations

import csv
import io
import math
import sys
import time
from pathlib import Path

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
IMG_DIM = 1024          # 32 × 32
IMG_SZ  = 32

DATA_DIR = ROOT / "data"
EXP_DIR  = ROOT / "experiments"

_CANDIDATE_VALUES = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
N_CANDS = IMG_DIM * len(_CANDIDATE_VALUES)   # 5120


# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (same luminance conversion as grayscale_cifar_baseline.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_cifar10_gray() -> tuple[torch.Tensor, torch.Tensor]:
    ds   = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True)
    data = ds.data.astype(np.float32) / 255.0           # (50000, 32, 32, 3)
    gray = (0.2989 * data[:, :, :, 0]
            + 0.5870 * data[:, :, :, 1]
            + 0.1140 * data[:, :, :, 2])                 # (50000, 32, 32)
    flat    = gray.reshape(-1, 1024)
    images  = torch.tensor(flat,       dtype=torch.float32)
    targets = torch.tensor(ds.targets, dtype=torch.long)
    return images, targets


# ─────────────────────────────────────────────────────────────────────────────
# Inline exhaustive white-box attacker for 32×32 CIFAR (1024-dim)
# Identical to WhiteBoxOnePixelAttacker.attack() except n_locs=1024, grid=32
# ─────────────────────────────────────────────────────────────────────────────

def _wb_attack_cifar_exhaustive(
    query: torch.Tensor,
    true_index: int,
    network: ContinuousHopfield,
) -> dict:
    device  = query.device
    X       = network.X                                          # (1024, N)
    cands   = _CANDIDATE_VALUES.to(device)
    n_locs  = IMG_DIM                                            # 1024
    n_vals  = cands.shape[0]                                     # 5
    n_cands = n_locs * n_vals                                    # 5120

    locs = torch.arange(n_locs, device=device).repeat_interleave(n_vals)   # (5120,)
    vals = cands.repeat(n_locs)                                              # (5120,)

    queries = query.unsqueeze(1).expand(-1, n_cands).clone()                 # (1024, 5120)
    queries[locs, torch.arange(n_cands, device=device)] = vals

    retrieved = network.retrieve(queries, steps=1)               # (1024, 5120)

    true_pat = X[:, true_index]                                  # (1024,)
    dots     = retrieved.T @ true_pat                            # (5120,)
    r_norms  = retrieved.norm(dim=0)                             # (5120,)
    t_norm   = true_pat.norm()
    cos_sims = dots / (r_norms * t_norm).clamp(min=1e-8)        # (5120,)

    worst_k   = int(cos_sims.argmin().item())
    worst_loc = int(locs[worst_k].item())
    worst_val = float(vals[worst_k].item())
    worst_cos = float(cos_sims[worst_k].item())

    worst_ret    = retrieved[:, worst_k]                         # (1024,)
    X_norms      = X.norm(dim=0)                                 # (N,)
    cos_all      = (X.T @ worst_ret) / (X_norms * worst_ret.norm()).clamp(min=1e-8)
    ret_index    = int(cos_all.argmax().item())

    pixel_i        = worst_loc // IMG_SZ
    pixel_j        = worst_loc % IMG_SZ
    original_value = float(query[worst_loc].item())

    return {
        "success":         ret_index != true_index,
        "pixel_i":         pixel_i,
        "pixel_j":         pixel_j,
        "pixel_value":     worst_val,
        "original_value":  original_value,
        "perturbation_l2": abs(worst_val - original_value),
        "cosine_to_true":  worst_cos,
        "retrieved_index": ret_index,
        "evaluations":     n_cands,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise cosine (mean off-diagonal)
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_cosine_mean(X: torch.Tensor) -> float:
    Xn  = X.numpy()
    nrm = np.linalg.norm(Xn, axis=0, keepdims=True)
    Xnn = Xn / (nrm + 1e-8)
    C   = Xnn.T @ Xnn                             # (N, N)
    np.fill_diagonal(C, 0.0)
    n = C.shape[0]
    return float(C.sum() / (n * (n - 1)))


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[list[dict], list[dict]]:
    detail:  list[dict] = []
    summary: list[dict] = []

    for seed in SEEDS:
        X, _   = sample_class_balanced((images, labels), N, seed=seed)
        stored = X.T.contiguous()     # (N, 1024) — row i = stored pattern i
        hop    = ContinuousHopfield(X, beta=BETA)
        cos_mean = pairwise_cosine_mean(X)

        seed_bl_correct   = 0
        seed_atk_success  = 0
        l2_successes: list[float] = []

        for probe_idx in range(N):
            query = stored[probe_idx]        # (1024,)

            # Baseline retrieval
            ret_clean        = hop.retrieve(query, steps=1)
            baseline_correct = int(retrieval_accuracy(ret_clean, X, probe_idx))

            # White-box attack (run on all probes for full picture)
            res = _wb_attack_cifar_exhaustive(query, probe_idx, hop)

            if baseline_correct:
                seed_bl_correct += 1
                if res["success"]:
                    seed_atk_success += 1
                    l2_successes.append(res["perturbation_l2"])

            # class_label = probe_idx for N=10 class-balanced (one per class)
            detail.append({
                "seed":               seed,
                "probe_idx":          probe_idx,
                "class_label":        probe_idx,
                "baseline_correct":   baseline_correct,
                "attack_success":     int(res["success"]),
                "pixel_i":            res["pixel_i"],
                "pixel_j":            res["pixel_j"],
                "pixel_value":        round(res["pixel_value"],     4),
                "original_value":     round(res["original_value"],  4),
                "perturbation_l2":    round(res["perturbation_l2"], 4),
                "retrieved_index":    res["retrieved_index"],
                "true_index":         probe_idx,
                "mean_pairwise_cosine": round(cos_mean, 5),
            })

        cond_rate = (seed_atk_success / seed_bl_correct
                     if seed_bl_correct > 0 else float("nan"))
        mean_l2   = (float(np.mean(l2_successes))
                     if l2_successes else float("nan"))

        summary.append({
            "seed":                              seed,
            "n_baseline_correct":                seed_bl_correct,
            "n_attack_success":                  seed_atk_success,
            "raw_attack_success_rate":           round(seed_atk_success / N, 4),
            "conditional_attack_success_rate":   (round(cond_rate, 4)
                                                   if not math.isnan(cond_rate)
                                                   else "nan"),
            "mean_attack_l2_among_successes_or_nan": (round(mean_l2, 4)
                                                       if not math.isnan(mean_l2)
                                                       else "nan"),
            "mean_pairwise_cosine":              round(cos_mean, 5),
        })

    return detail, summary


# ─────────────────────────────────────────────────────────────────────────────
# Save CSVs
# ─────────────────────────────────────────────────────────────────────────────

def save_detail_csv(rows: list[dict]) -> None:
    fields = [
        "seed", "probe_idx", "class_label", "baseline_correct",
        "attack_success", "pixel_i", "pixel_j", "pixel_value",
        "original_value", "perturbation_l2", "retrieved_index",
        "true_index", "mean_pairwise_cosine",
    ]
    path = EXP_DIR / "grayscale_cifar_attack_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})
    print(f"  Saved: {path.name}  ({len(rows)} rows)")


def save_summary_csv(rows: list[dict]) -> None:
    fields = [
        "seed", "n_baseline_correct", "n_attack_success",
        "raw_attack_success_rate", "conditional_attack_success_rate",
        "mean_attack_l2_among_successes_or_nan", "mean_pairwise_cosine",
    ]
    path = EXP_DIR / "grayscale_cifar_attack_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})
    print(f"  Saved: {path.name}  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Report builder
# ─────────────────────────────────────────────────────────────────────────────

def build_report(detail: list[dict], summary: list[dict]) -> str:
    buf = io.StringIO()

    def p(line: str = "") -> None:
        buf.write(line + "\n")

    bl_fail_rates = [1.0 - r["n_baseline_correct"] / N for r in summary]
    raw_rates     = [r["n_attack_success"] / N          for r in summary]

    mean_bl_fail  = float(np.mean(bl_fail_rates))
    std_bl_fail   = float(np.std(bl_fail_rates,  ddof=1))
    mean_raw      = float(np.mean(raw_rates))
    std_raw       = float(np.std(raw_rates,      ddof=1))

    total_bl_correct = sum(r["n_baseline_correct"] for r in summary)
    total_attacked   = sum(r["n_attack_success"]   for r in summary)

    cond_vals: list[float] = []
    for r in summary:
        if r["n_baseline_correct"] > 0:
            cond_vals.append(r["n_attack_success"] / r["n_baseline_correct"])
    mean_cond = float(np.mean(cond_vals)) if cond_vals else float("nan")
    std_cond  = float(np.std(cond_vals, ddof=1)) if len(cond_vals) > 1 else 0.0

    cos_mean = float(np.mean([r["mean_pairwise_cosine"] for r in summary]))

    p("========================================================")
    p("GRAYSCALE CIFAR-10 ONE-PIXEL ATTACK: NULL RESULT DOCUMENTATION")
    p("========================================================")
    p()
    p(f"Setup: N={N}, class-balanced, beta=8.0, 5 seeds, white-box attacker")
    p(f"Total probes attacked: {len(detail)} (10 per seed x 5 seeds)")
    p(f"Candidates evaluated per probe: {N_CANDS} (1024 pixels x 5 values)")
    p()
    p("=== BASELINE RETRIEVAL ===")
    p()
    p("Seed-level baseline (correct retrievals out of 10):")
    for r in summary:
        p(f"  Seed {r['seed']}: {r['n_baseline_correct']}/10")
    p(f"Mean baseline failure: {mean_bl_fail*100:.1f}% +/- {std_bl_fail*100:.1f}%")
    p()
    p("=== ATTACK OUTCOMES ===")
    p()
    p("Seed-level attack outcomes (successes out of 10):")
    for r in summary:
        p(f"  Seed {r['seed']}: {r['n_attack_success']}/10")
    p(f"Raw attack success rate: {mean_raw*100:.1f}% +/- {std_raw*100:.1f}%")
    if not math.isnan(mean_cond):
        p(f"Conditional attack success rate (among baseline-correct probes): "
          f"{mean_cond*100:.1f}% +/- {std_cond*100:.1f}%")
    else:
        p("Conditional attack success rate (among baseline-correct probes): "
          "undefined (no baseline-correct probes)")
    p()
    p(f"Among the baseline-correct probes (total across all seeds: {total_bl_correct}):")
    p(f"  - Successfully attacked: {total_attacked}")
    p(f"  - Robust to all {N_CANDS} candidate single-pixel perturbations: "
      f"{total_bl_correct - total_attacked}")
    p()
    p("=== INTERPRETATION ===")
    p()
    p("Pairwise pattern similarity (mean off-diagonal cosine):")
    p(f"  N={N} grayscale CIFAR: {cos_mean:.3f}")
    p(f"  N=100 MNIST (reference): 0.397")
    p()
    p("Comparison to MNIST headline cell:")
    hdr = f"  {'Dataset':<25}  {'Baseline failure':>18}  {'Conditional attack success':>26}"
    sep = "  " + "-" * (len(hdr) - 2)
    p(hdr)
    p(sep)
    p(f"  {'MNIST N=100':<25}  {'9.2%':>18}  {'3.1%':>26}")
    bl_str   = f"{mean_bl_fail*100:.1f}%"
    cond_str = (f"{mean_cond*100:.1f}%"
                if not math.isnan(mean_cond) else "0.0% (undefined)")
    p(f"  {f'Grayscale CIFAR N={N}':<25}  {bl_str:>18}  {cond_str:>26}")
    p()
    p("Verdict:")
    if not math.isnan(mean_cond) and mean_cond == 0.0 and total_bl_correct > 0:
        p(f"  Null result confirmed: across {total_bl_correct} baseline-correct probes, the")
        p(f"  white-box one-pixel attacker found zero successful single-pixel")
        p(f"  perturbations among {N_CANDS} candidates per probe. This documents that")
        p(f"  grayscale CIFAR-10 with raw-pixel storage is not susceptible to")
        p(f"  one-pixel adversarial attacks at this storage size, consistent with")
        p(f"  the dataset operating outside the network's reliable retrieval regime.")
    elif total_bl_correct == 0:
        p(f"  No baseline-correct probes across any seed at N={N}: the network")
        p(f"  failed on all clean probes, leaving no population to attack. The")
        p(f"  null result holds by vacuity — attack analysis requires at least")
        p(f"  one baseline-correct probe.")
    else:
        p(f"  Partial result: of {total_bl_correct} baseline-correct probes, {total_attacked} were")
        p(f"  successfully attacked. Conditional attack rate: {mean_cond*100:.1f}%. While non-zero,")
        p(f"  the small sample size and high baseline failure rate limit the")
        p(f"  interpretability of this result; the headline finding remains that")
        p(f"  grayscale CIFAR-10 operates outside the network's reliable regime.")
    p()
    p("========================================================")

    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    print("Loading CIFAR-10 ...")
    images, labels = load_cifar10_gray()
    print(f"  ok  {len(images)} samples, dim={IMG_DIM}")

    print(f"Running attacks: N={N}, seeds={SEEDS}, {N_CANDS} candidates/probe ...")
    detail, summary = run_experiment(images, labels)

    print("Saving results ...")
    save_detail_csv(detail)
    save_summary_csv(summary)

    report = build_report(detail, summary)
    print()
    print(report)

    report_path = EXP_DIR / "grayscale_cifar_attack_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {report_path.name}")
    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
