"""
MNIST two-stage one-pixel attack experiment.

Applies the same two-stage pipeline as grayscale_cifar_two_stage_attack:
  1. Precompute adversarial images with the inline attacker (cosine-delta + first-order
     sensitivity fallback) and save them to one_pixel_test/mnist_seed_k/.
  2. Feed saved images to the Hopfield network and evaluate retrieval.

Two conditions:
  A: Clean stored patterns | Attacked query
  B: Attacked patterns stored | Clean query

Run: python -m experiments.mnist_two_stage_attack
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
import matplotlib.patches as patches
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
N       = 100
BETA    = 8.0
SEEDS   = [42, 43, 44, 45, 46]
IMG_DIM = 784
IMG_SZ  = 28

_CANDS  = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
N_CANDS = IMG_DIM * len(_CANDS)    # 3920

MNIST_CLASSES = [str(i) for i in range(10)]

DATA_DIR    = ROOT / "data"
EXP_DIR     = ROOT / "experiments"
FIG_DIR     = ROOT / "figures"
ONE_PIX_DIR = ROOT / "one_pixel_test"
ONE_PIX_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_mnist() -> tuple[torch.Tensor, torch.Tensor]:
    ds = torchvision.datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
    return ds.data.float().view(-1, 784) / 255.0, ds.targets


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _batch_cosine(retrieved: torch.Tensor, true_pat: torch.Tensor) -> torch.Tensor:
    dots    = retrieved.T @ true_pat
    r_norms = retrieved.norm(dim=0)
    t_norm  = true_pat.norm()
    return dots / (r_norms * t_norm).clamp(min=1e-8)


def _nearest_stored(ret: torch.Tensor, X: torch.Tensor) -> int:
    cos_all = (X.T @ ret) / (X.norm(dim=0) * ret.norm()).clamp(min=1e-8)
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
# Inline white-box attacker with cosine-delta + first-order sensitivity fallback
# ─────────────────────────────────────────────────────────────────────────────

def wb_attack(
    q: torch.Tensor,       # (784,)
    true_index: int,
    network: ContinuousHopfield,
) -> dict:
    X     = network.X
    n_c   = N_CANDS

    locs = torch.arange(IMG_DIM).repeat_interleave(len(_CANDS))
    vals = _CANDS.repeat(IMG_DIM)

    queries = q.unsqueeze(1).expand(-1, n_c).clone()
    queries[locs, torch.arange(n_c)] = vals

    retrieved = network.retrieve(queries, steps=1)
    cos_sims  = _batch_cosine(retrieved, X[:, true_index])

    # Cosine delta vs clean retrieval — avoids degenerate argmin when all
    # candidates retrieve the same wrong pattern (already-failing probes).
    ret_clean = network.retrieve(q, steps=1)
    clean_cos = float(_batch_cosine(ret_clean.unsqueeze(1), X[:, true_index])[0])
    delta     = cos_sims - clean_cos

    if float(delta.min().item()) < -1e-4:
        k = int(delta.argmin().item())
    else:
        # First-order sensitivity: select pixel maximally reducing dot-product
        # with the true stored pattern.
        true_at_locs = X[:, true_index][locs]
        fo_damage    = true_at_locs * (q[locs] - vals)
        noop_mask    = (q[locs] - vals).abs() < 1e-6
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
        "pixel_value":     val,
        "original_value":  float(q[loc].item()),
        "perturbation_l2": abs(val - float(q[loc].item())),
        "cosine_to_true":  float(cos_sims[k].item()),
        "cosine_delta":    cosine_delta,
        "retrieved_index": _nearest_stored(retrieved[:, k], X),
        "evaluations":     n_c,
        "attacked":        queries[:, k].clone(),   # (784,)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Image saving
# ─────────────────────────────────────────────────────────────────────────────

def _save_gray(t: torch.Tensor, path: Path) -> None:
    plt.imsave(str(path), t.view(IMG_SZ, IMG_SZ).numpy(), cmap="gray", vmin=0, vmax=1)


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[list[dict], list[dict], dict]:
    detail:   list[dict] = []
    summary:  list[dict] = []
    seed42_state: dict   = {}   # X, stored, hop_clean, hop_b for figure

    for seed in SEEDS:
        seed_dir = ONE_PIX_DIR / f"mnist_seed_{seed}"
        seed_dir.mkdir(exist_ok=True)

        X_gray, _   = sample_class_balanced((images, labels), N, seed=seed)
        stored      = X_gray.T.contiguous()          # (N, 784)
        hop_clean   = ContinuousHopfield(X_gray, beta=BETA)
        cos_mean    = pairwise_cosine_mean(X_gray)

        # ── Stage 1: precompute attacks and save images ──────────────────────
        attacks: list[dict] = []
        for i in range(N):
            cls_name = MNIST_CLASSES[i // (N // len(MNIST_CLASSES))]
            r = wb_attack(stored[i], i, hop_clean)
            attacks.append(r)
            _save_gray(stored[i],   seed_dir / f"probe{i:03d}_dig{cls_name}_clean.png")
            _save_gray(r["attacked"], seed_dir / f"probe{i:03d}_dig{cls_name}_attacked.png")

        print(f"  Seed {seed}: {2*N} images saved -> {seed_dir.relative_to(ROOT)}/")

        # ── Stage 2: build attacked storage matrix (B condition) ─────────────
        X_b   = torch.stack([attacks[i]["attacked"] for i in range(N)], dim=1)
        hop_b = ContinuousHopfield(X_b, beta=BETA)

        if seed == 42:
            seed42_state = {
                "X": X_gray, "stored": stored,
                "hop_clean": hop_clean, "hop_b": hop_b,
                "attacks": attacks,
            }

        # ── Stage 3: evaluate both conditions ────────────────────────────────
        for i in range(N):
            q = stored[i]

            ret_clean  = hop_clean.retrieve(q, steps=1)
            bl_correct = int(retrieval_accuracy(ret_clean, X_gray, i))

            suc_a = int(attacks[i]["success"])
            ri_a  = attacks[i]["retrieved_index"]

            ret_b = hop_b.retrieve(q, steps=1)
            ri_b  = _nearest_stored(ret_b, X_b)
            suc_b = int(ri_b != i)

            cls_name = MNIST_CLASSES[i // (N // len(MNIST_CLASSES))]
            detail.append({
                "seed": seed, "probe_idx": i, "class": cls_name,
                "baseline_correct": bl_correct,
                "a_success":      suc_a,
                "a_retrieved":    ri_a,
                "a_pixel_i":      attacks[i]["pixel_i"],
                "a_pixel_j":      attacks[i]["pixel_j"],
                "a_pixel_val":    round(attacks[i]["pixel_value"],     4),
                "a_l2":           round(attacks[i]["perturbation_l2"], 4),
                "a_cosine_delta": round(attacks[i]["cosine_delta"],    6),
                "b_success":      suc_b,
                "b_retrieved":    ri_b,
                "mean_pairwise_cosine": round(cos_mean, 5),
            })

        # ── Per-seed summary ──────────────────────────────────────────────────
        prb  = [r for r in detail if r["seed"] == seed]
        n_bl = sum(r["baseline_correct"] for r in prb)

        def _cond(key: str) -> float:
            bl = [r[key] for r in prb if r["baseline_correct"] == 1]
            return (sum(bl) / len(bl)) if bl else float("nan")

        c_a = _cond("a_success")
        c_b = _cond("b_success")

        summary.append({
            "seed": seed,
            "n_baseline_correct":   n_bl,
            "mean_pairwise_cosine": round(cos_mean, 5),
            "a_raw":  round(sum(r["a_success"] for r in prb) / N, 4),
            "a_cond": round(c_a, 4) if not math.isnan(c_a) else "nan",
            "b_raw":  round(sum(r["b_success"] for r in prb) / N, 4),
            "b_cond": round(c_b, 4) if not math.isnan(c_b) else "nan",
        })

    return detail, summary, seed42_state


# ─────────────────────────────────────────────────────────────────────────────
# Figure: 10x3 grid (one probe per class, seed=42)
# ─────────────────────────────────────────────────────────────────────────────

def save_figure(
    seed42_state: dict,
    seed42_detail: list[dict],
    fig_path: Path,
    dataset_name: str = "MNIST",
    class_names: list[str] = MNIST_CLASSES,
) -> None:
    stored    = seed42_state["stored"]
    hop_clean = seed42_state["hop_clean"]
    attacks   = seed42_state["attacks"]

    n_per_class = N // len(class_names)
    class_reps  = list(range(0, N, n_per_class))[:len(class_names)]

    probe_map = {r["probe_idx"]: r for r in seed42_detail}

    fig, axes = plt.subplots(len(class_reps), 3,
                             figsize=(7, 2.2 * len(class_reps)))

    for col_idx, title in enumerate(["Clean", "Attacked query", "Retrieved (A)"]):
        axes[0, col_idx].set_title(title, fontsize=8)

    for row, i in enumerate(class_reps):
        r   = probe_map[i]
        q   = stored[i]
        atk = attacks[i]["attacked"]

        ret_a = hop_clean.retrieve(atk, steps=1)

        imgs = [q, atk, ret_a]
        for col, vec in enumerate(imgs):
            ax = axes[row, col]
            ax.imshow(vec.view(IMG_SZ, IMG_SZ).detach().numpy(),
                      cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

        # Red rectangle on attacked pixel
        pi, pj = attacks[i]["pixel_i"], attacks[i]["pixel_j"]
        rect = patches.Rectangle((pj - 0.5, pi - 0.5), 1, 1,
                                  linewidth=1.5, edgecolor="red", facecolor="none")
        axes[row, 1].add_patch(rect)

        # Colour background by outcome
        bl = r["baseline_correct"]
        ok = r["a_success"]
        bg = "#c8e6c9" if ok else ("#fff9c4" if bl else "#ffcdd2")
        for col in range(3):
            axes[row, col].set_facecolor(bg)

        # Class label + pixel annotation
        axes[row, 0].set_ylabel(class_names[row], fontsize=7,
                                rotation=0, labelpad=30, va="center")
        axes[row, 1].set_xlabel(
            f"({pi},{pj}) Δ={attacks[i]['cosine_delta']:+.4f}",
            fontsize=6,
        )

    fig.suptitle(
        f"{dataset_name} two-stage one-pixel attack  |  seed=42, N={N}, β={BETA}\n"
        "green=attack succeeded  yellow=attack failed (baseline ok)  red=already failing",
        fontsize=8,
    )
    plt.tight_layout()
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fig_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV saving
# ─────────────────────────────────────────────────────────────────────────────

def save_detail_csv(rows: list[dict], name: str) -> None:
    fields = [
        "seed", "probe_idx", "class", "baseline_correct",
        "a_success", "a_retrieved", "a_pixel_i", "a_pixel_j",
        "a_pixel_val", "a_l2", "a_cosine_delta",
        "b_success", "b_retrieved",
        "mean_pairwise_cosine",
    ]
    path = EXP_DIR / f"{name}_two_stage_results.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})
    print(f"  Saved: {path.name}  ({len(rows)} rows)")


def save_summary_csv(rows: list[dict], name: str) -> None:
    fields = [
        "seed", "n_baseline_correct", "mean_pairwise_cosine",
        "a_raw", "a_cond", "b_raw", "b_cond",
    ]
    path = EXP_DIR / f"{name}_two_stage_summary.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fields})
    print(f"  Saved: {path.name}  ({len(rows)} rows)")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def build_report(detail: list[dict], summary: list[dict], dataset_name: str) -> str:
    buf = io.StringIO()

    def p(line: str = "") -> None:
        buf.write(line + "\n")

    total    = len(detail)
    total_bl = sum(r["n_baseline_correct"] for r in summary)
    bl_rates = [1.0 - r["n_baseline_correct"] / N for r in summary]
    cos_vals = [r["mean_pairwise_cosine"] for r in summary]

    def _agg(key_raw: str, key_cond: str):
        raw_vals  = [r[key_raw]  for r in summary]
        cond_vals = [r[key_cond] for r in summary if r[key_cond] != "nan"]
        raw_mean  = float(np.mean(raw_vals))
        raw_std   = float(np.std(raw_vals, ddof=1))
        cond_mean = float(np.mean(cond_vals)) if cond_vals else float("nan")
        cond_std  = float(np.std(cond_vals, ddof=1)) if len(cond_vals) > 1 else 0.0
        return raw_mean, raw_std, cond_mean, cond_std

    a_rm, a_rs, a_cm, a_cs = _agg("a_raw", "a_cond")
    b_rm, b_rs, b_cm, b_cs = _agg("b_raw", "b_cond")

    mean_bl_fail = float(np.mean(bl_rates))
    std_bl_fail  = float(np.std(bl_rates, ddof=1))
    mean_cos     = float(np.mean(cos_vals))

    # Count probes where cosine_delta < -1e-4 (effective attack found)
    effective = [r for r in detail if float(r["a_cosine_delta"]) < -1e-4]
    fallback  = [r for r in detail if float(r["a_cosine_delta"]) >= -1e-4]

    p("=" * 64)
    p(f"{dataset_name.upper()} TWO-STAGE ONE-PIXEL ATTACK EXPERIMENT")
    p("=" * 64)
    p()
    p(f"Setup: N={N}, class-balanced, beta={BETA}, 5 seeds")
    p(f"Total probe evaluations: {total}  ({N} per seed x 5 seeds)")
    p(f"Candidates per probe: {N_CANDS:,}  ({IMG_DIM} locs x {len(_CANDS)} values)")
    p()
    p("Conditions:")
    p("  A: Clean stored patterns | Attacked query")
    p("  B: Attacked patterns stored | Clean query")
    p()
    p(f"Images saved to: one_pixel_test/{dataset_name.lower()}_seed_k/")
    p(f"  ({2 * N * len(SEEDS)} PNG files total)")
    p()
    p("=" * 64)
    p("PIXEL SELECTION METHOD")
    p("=" * 64)
    p()
    p(f"  Cosine-delta selection (real attack found): {len(effective)}/{total} probes")
    p(f"  First-order sensitivity fallback:           {len(fallback)}/{total} probes")
    p()
    p("=" * 64)
    p("BASELINE RETRIEVAL (clean stored, clean query)")
    p("=" * 64)
    p()
    for r in summary:
        p(f"  Seed {r['seed']}: {r['n_baseline_correct']}/{N} correct  "
          f"(pairwise cosine {r['mean_pairwise_cosine']:.3f})")
    p(f"  Mean baseline failure: {mean_bl_fail*100:.1f}% +/- {std_bl_fail*100:.1f}%")
    p(f"  Total baseline-correct probes (across 5 seeds): {total_bl}")
    p()

    def _section(label: str, rm: float, rs: float, cm: float, cs: float,
                 seed_key: str) -> None:
        p("=" * 64)
        p(label)
        p("=" * 64)
        p()
        p(f"  Seed-level successes out of {N}:")
        for r in summary:
            p(f"    Seed {r['seed']}: {round(r[seed_key] * N)}/{N}")
        p(f"  Raw success rate:         {rm*100:.1f}% +/- {rs*100:.1f}%")
        if not math.isnan(cm):
            p(f"  Conditional success rate (among baseline-correct): "
              f"{cm*100:.1f}% +/- {cs*100:.1f}%")
        else:
            p("  Conditional success rate: undefined (no qualifying probes)")
        p()

    _section("CONDITION A: Attacked query, clean stored", a_rm, a_rs, a_cm, a_cs, "a_raw")
    _section("CONDITION B: Attacked stored, clean query", b_rm, b_rs, b_cm, b_cs, "b_raw")

    p("=" * 64)
    p("INTERPRETATION")
    p("=" * 64)
    p()
    p(f"Mean pairwise cosine at N={N}: {mean_cos:.3f}")
    p()
    p(f"  {'Condition':<45}  {'Raw success':>12}  {'Cond success':>13}")
    p("  " + "-" * 72)
    p(f"  {'A: Attacked query, clean stored':<45}  {a_rm*100:>11.1f}%  "
      f"  {a_cm*100 if not math.isnan(a_cm) else 0.0:>11.1f}%")
    p(f"  {'B: Attacked stored, clean query':<45}  {b_rm*100:>11.1f}%  "
      f"  {b_cm*100 if not math.isnan(b_cm) else 0.0:>11.1f}%")
    p()

    any_success = any(
        v > 0.0 for v in [a_cm, b_cm] if not math.isnan(v)
    )
    if any_success:
        p("Verdict:")
        p(f"  Attacks succeed conditionally: condition A achieves {a_cm*100:.1f}%,")
        p(f"  condition B achieves {b_cm*100:.1f}% success among baseline-correct probes.")
        p(f"  The {len(effective)}/{total} probes where cosine delta < -1e-4 represent")
        p("  genuine adversarial perturbations that redirect retrieval.")
    else:
        p("Verdict:")
        p("  Null result — no attack-attributable successes among baseline-correct probes.")
    p()
    p("=" * 64)

    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    name = "mnist"
    print("=" * 64)
    print("MNIST Two-Stage One-Pixel Attack Experiment")
    print("=" * 64)

    print("\nLoading MNIST ...")
    images, labels = load_mnist()
    print(f"  {images.shape}  labels: {labels.shape}")

    print(f"\nRunning experiment: N={N}, seeds={SEEDS} ...")
    detail, summary, seed42_state = run_experiment(images, labels)

    print("\nSaving CSVs ...")
    save_detail_csv(detail, name)
    save_summary_csv(summary, name)

    report = build_report(detail, summary, "MNIST")
    print()
    print(report)

    report_path = EXP_DIR / f"{name}_two_stage_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {report_path.name}")

    print("\nGenerating figure ...")
    seed42_detail = [r for r in detail if r["seed"] == 42]
    save_figure(
        seed42_state, seed42_detail,
        FIG_DIR / "mnist_attack_grid.png",
        dataset_name="MNIST",
        class_names=MNIST_CLASSES,
    )

    print(f"Total runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
