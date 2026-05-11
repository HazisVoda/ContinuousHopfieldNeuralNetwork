"""
Phase 3 diagnostics.

Diagnostic A: White-box vs DE attack agreement analysis at headline cell
             (N=100, class_balanced). Reads CSVs only — no attack re-runs.
Diagnostic B: Baseline-corrected attack effectiveness. Measures clean-retrieval
             failure rate per cell, then re-expresses WB and RNE rates
             conditional on clean success.

Run: python -m experiments.phase3_diagnostics
"""

from __future__ import annotations

import csv
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
    images = ds.data.float() / 255.0
    return images.view(-1, 784), ds.targets


def build_cell(
    images: torch.Tensor,
    labels: torch.Tensor,
    N: int, n_idx: int, s_idx: int,
    strategy: str, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, ContinuousHopfield, list[int]]:
    dataset = (images, labels)
    sampler = sample_random if strategy == "random" else sample_class_balanced
    X, _    = sampler(dataset, N, seed=seed + n_idx)
    stored  = X.T.contiguous()
    hop     = ContinuousHopfield(X, beta=BETA)
    n_probes   = min(N_PROBE, N)
    probe_seed = seed * 1000 + n_idx * 100 + s_idx * 10
    rng        = torch.Generator()
    rng.manual_seed(probe_seed)
    probe_indices = torch.randperm(N, generator=rng)[:n_probes].tolist()
    return X, stored, hop, probe_indices


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fieldnames})


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic A: WB vs DE agreement at N=100 class_balanced
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostic_a(exp_dir: Path) -> dict:
    wb_rows = read_csv_rows(exp_dir / "phase3_whitebox_results.csv")
    de_rows = read_csv_rows(exp_dir / "phase3_blackbox_results.csv")

    def is_hl(r: dict) -> bool:
        return int(r["N"]) == 100 and r["strategy"] == "class_balanced"

    wb_map = {(int(r["seed"]), int(r["probe_idx"])): r for r in wb_rows if is_hl(r)}
    de_map = {(int(r["seed"]), int(r["probe_idx"])): r for r in de_rows if is_hl(r)}

    paired: list[dict] = []
    for key in sorted(wb_map):
        if key not in de_map:
            continue
        wb, de = wb_map[key], de_map[key]
        seed, probe_idx = key
        wb_pi, wb_pj = int(wb["pixel_i"]), int(wb["pixel_j"])
        de_pi, de_pj = int(de["pixel_i"]), int(de["pixel_j"])
        same_pixel = int(wb_pi == de_pi and wb_pj == de_pj)
        same_ret   = int(int(wb["retrieved_index"]) == int(de["retrieved_index"]))
        paired.append({
            "seed":               seed,
            "probe_idx":          probe_idx,
            "wb_success":         int(wb["attack_success"]),
            "de_success":         int(de["attack_success"]),
            "wb_pixel_i":         wb_pi,
            "wb_pixel_j":         wb_pj,
            "de_pixel_i":         de_pi,
            "de_pixel_j":         de_pj,
            "wb_l2":              float(wb["perturbation_l2"]),
            "de_l2":              float(de["perturbation_l2"]),
            "wb_retrieved_index": int(wb["retrieved_index"]),
            "de_retrieved_index": int(de["retrieved_index"]),
            "same_pixel":         same_pixel,
            "same_retrieved":     same_ret,
        })

    fields = [
        "seed", "probe_idx", "wb_success", "de_success",
        "wb_pixel_i", "wb_pixel_j", "de_pixel_i", "de_pixel_j",
        "wb_l2", "de_l2", "wb_retrieved_index", "de_retrieved_index",
        "same_pixel", "same_retrieved",
    ]
    write_csv(paired, exp_dir / "phase3_diag_a_paired.csv", fields)

    n  = len(paired)
    ws = np.array([p["wb_success"] for p in paired])
    ds = np.array([p["de_success"] for p in paired])
    wl = np.array([p["wb_l2"]      for p in paired])
    dl = np.array([p["de_l2"]      for p in paired])

    both_succ = int(((ws == 1) & (ds == 1)).sum())
    wb_only   = int(((ws == 1) & (ds == 0)).sum())
    de_only   = int(((ws == 0) & (ds == 1)).sum())
    both_fail = int(((ws == 0) & (ds == 0)).sum())

    joint = [p for p in paired if p["wb_success"] == 1 and p["de_success"] == 1]
    n_joint = len(joint)

    same_pix      = sum(p["same_pixel"] for p in joint)
    same_row_difc = sum(
        1 for p in joint
        if p["wb_pixel_i"] == p["de_pixel_i"] and p["wb_pixel_j"] != p["de_pixel_j"]
    )
    diff_entirely = n_joint - same_pix - same_row_difc
    same_ret_j    = sum(p["same_retrieved"] for p in joint)

    r_corr = float(np.corrcoef(wl, dl)[0, 1]) if n > 1 else float("nan")

    # Verdict
    if wb_only > n * 0.10 and de_only > n * 0.10:
        verdict = (
            "WB and DE are attacking independently. The matched ~12% rates reflect "
            "a genuinely small attackable subset of probes."
        )
    elif abs(r_corr - 1.0) < 1e-5 and wb_only == 0 and de_only == 0:
        verdict = (
            "WB and DE outputs are suspiciously identical — likely code path issue. "
            "Investigate before trusting Phase 3 DE results."
        )
    else:
        verdict = (
            "WB and DE show partial agreement. The matched headline rate is plausibly "
            "real but DE is finding the same vulnerable probes via different specific perturbations."
        )

    return {
        "n":            n,
        "both_succ":    both_succ,
        "wb_only":      wb_only,
        "de_only":      de_only,
        "both_fail":    both_fail,
        "wb_rate":      float(ws.mean()),
        "de_rate":      float(ds.mean()),
        "n_joint":      n_joint,
        "same_pix":     same_pix,
        "same_row_difc": same_row_difc,
        "diff_entirely": diff_entirely,
        "same_ret_j":   same_ret_j,
        "r_corr":       r_corr,
        "verdict":      verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic B: baseline-corrected effectiveness
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostic_b(
    images: torch.Tensor,
    labels: torch.Tensor,
    exp_dir: Path,
) -> list[dict]:
    wb_rows  = read_csv_rows(exp_dir / "phase3_whitebox_results.csv")
    wb_index = {
        (int(r["N"]), r["strategy"], int(r["seed"]), int(r["probe_idx"])): r
        for r in wb_rows
    }

    diag_rows: list[dict] = []

    for n_idx, N in enumerate(N_VALUES):
        for s_idx, strategy in enumerate(STRATEGIES):
            print(f"  N={N:>5}  {strategy} ...")
            for seed in SEEDS:
                X, stored, hop, probe_indices = build_cell(
                    images, labels, N, n_idx, s_idx, strategy, seed
                )

                baseline_fail  = 0
                clean_ok_count = 0
                wb_succ_total  = 0
                cond_wb_succ   = 0
                rne_fail_001   = 0
                cond_rne_001   = 0

                for j, true_idx in enumerate(probe_indices):
                    q = stored[true_idx]

                    # Clean baseline
                    clean_ret = hop.retrieve(q, steps=1)
                    clean_ok  = retrieval_accuracy(clean_ret, X, true_idx)
                    if clean_ok:
                        clean_ok_count += 1
                    else:
                        baseline_fail += 1

                    # WB result from CSV
                    wb_rec  = wb_index.get((N, strategy, seed, j))
                    wb_succ = int(wb_rec["attack_success"]) if wb_rec else 0
                    wb_succ_total += wb_succ
                    if clean_ok and wb_succ:
                        cond_wb_succ += 1

                    # RNE at sigma=0.01  (same seed formula as Phase 3 stage2_rne)
                    noisy = add_gaussian_noise(q, sigma=0.01, seed=seed + j)
                    ret   = hop.retrieve(noisy, steps=1)
                    fail  = not retrieval_accuracy(ret, X, true_idx)
                    rne_fail_001 += int(fail)
                    if clean_ok and fail:
                        cond_rne_001 += 1

                n_p  = len(probe_indices)
                bfr  = baseline_fail  / n_p
                rwb  = wb_succ_total  / n_p
                cwb  = cond_wb_succ  / clean_ok_count if clean_ok_count > 0 else 0.0
                rr1  = rne_fail_001  / n_p
                cr1  = cond_rne_001  / clean_ok_count if clean_ok_count > 0 else 0.0

                diag_rows.append({
                    "seed":     seed,
                    "N":        N,
                    "strategy": strategy,
                    "baseline_failure_rate":             round(bfr,              4),
                    "raw_wb_success":                    round(rwb,              4),
                    "conditional_wb_success":            round(cwb,              4),
                    "attack_attributable_rate":          round(max(0.0, rwb - bfr), 4),
                    "raw_random_failure_sigma_001":      round(rr1,              4),
                    "conditional_random_failure_sigma_001": round(cr1,           4),
                })

    fields = [
        "seed", "N", "strategy",
        "baseline_failure_rate", "raw_wb_success", "conditional_wb_success",
        "attack_attributable_rate",
        "raw_random_failure_sigma_001", "conditional_random_failure_sigma_001",
    ]
    write_csv(diag_rows, exp_dir / "phase3_diag_b_baseline_corrected.csv", fields)
    return diag_rows


# ─────────────────────────────────────────────────────────────────────────────
# Figure A: contingency heatmap
# ─────────────────────────────────────────────────────────────────────────────

def save_figure_a(da: dict, fig_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5.5))

    data = np.array([
        [da["both_succ"], da["wb_only"]],
        [da["de_only"],   da["both_fail"]],
    ])
    n = da["n"]

    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=n)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["DE success", "DE fail"], fontsize=11)
    ax.set_yticklabels(["WB success", "WB fail"], fontsize=11)
    ax.set_xlabel("DE black-box outcome", fontsize=10)
    ax.set_ylabel("White-box outcome", fontsize=10)

    cell_labels = [
        ["Both succeed\n(agreement)", "WB-only success\n(disagreement)"],
        ["DE-only success\n(disagreement)", "Both fail\n(agreement)"],
    ]
    for ri in range(2):
        for ci in range(2):
            cnt = int(data[ri, ci])
            pct = 100.0 * cnt / n
            is_dark = cnt > n * 0.35
            ax.text(ci, ri,
                    f"{cell_labels[ri][ci]}\n{cnt}  ({pct:.1f}%)",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if is_dark else "black")

    plt.colorbar(im, ax=ax, label="Count", fraction=0.046, pad=0.04)
    ax.set_title(
        f"WB vs DE attack agreement — N=100 class-balanced\n"
        f"{n} paired probes (5 seeds)   "
        f"WB={da['wb_rate']*100:.1f}%   DE={da['de_rate']*100:.1f}%   "
        f"r(L2)={da['r_corr']:.3f}",
        fontsize=9,
    )

    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure B: raw vs conditional corrected view
# ─────────────────────────────────────────────────────────────────────────────

def save_figure_b(diag_rows: list[dict], fig_path: str) -> None:
    def agg(metric: str, N: int, strategy: str) -> tuple[float, float]:
        vals = [r[metric] for r in diag_rows if r["N"] == N and r["strategy"] == strategy]
        return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: headline cell bar chart ─────────────────────────────────────────
    ax = axes[0]
    rwb_m, rwb_s = agg("raw_wb_success",                       100, "class_balanced")
    cwb_m, cwb_s = agg("conditional_wb_success",               100, "class_balanced")
    rr1_m, rr1_s = agg("raw_random_failure_sigma_001",         100, "class_balanced")
    cr1_m, cr1_s = agg("conditional_random_failure_sigma_001", 100, "class_balanced")

    x    = np.array([0.0, 0.8, 2.0, 2.8])
    hts  = [rwb_m, cwb_m, rr1_m, cr1_m]
    errs = [rwb_s, cwb_s, rr1_s, cr1_s]
    clrs = ["#4393c3", "#08306b", "#f4a582", "#67001f"]
    xlbls = ["WB\n(raw)", "WB\n(conditional)", r"RNE $\sigma$=0.01" + "\n(raw)",
             r"RNE $\sigma$=0.01" + "\n(conditional)"]

    bars = ax.bar(x, hts, width=0.65, color=clrs, alpha=0.88, edgecolor="white")
    ax.errorbar(x, hts, yerr=errs, fmt="none", ecolor="black", capsize=5, elinewidth=1.5)
    for bar, h in zip(bars, hts):
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                f"{h:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_ylabel("Rate", fontsize=10)
    y_top = max(hts) * 1.40
    ax.set_ylim(0, y_top if y_top > 0.15 else 0.25)
    ax.axvline(1.4, color="gray", linestyle="--", lw=1, alpha=0.4)
    ax.text(0.4, -0.10, "Attack", ha="center",
            transform=ax.get_xaxis_transform(), fontsize=9, color="#444")
    ax.text(2.4, -0.10, "Random noise", ha="center",
            transform=ax.get_xaxis_transform(), fontsize=9, color="#444")
    ax.set_title(
        "Headline cell: N=100 class-balanced\n"
        "Raw vs baseline-corrected rates (5-seed mean +/- std)",
        fontsize=9,
    )
    ax.grid(True, axis="y", alpha=0.25)

    # ── Right: raw vs conditional across N, class_balanced ────────────────────
    ax2 = axes[1]
    raw_m,  raw_s  = [], []
    cond_m, cond_s = [], []
    for N in N_VALUES:
        rm, rs = agg("raw_wb_success",        N, "class_balanced")
        cm, cs = agg("conditional_wb_success", N, "class_balanced")
        raw_m.append(rm);  raw_s.append(rs)
        cond_m.append(cm); cond_s.append(cs)

    raw_m  = np.array(raw_m);  raw_s  = np.array(raw_s)
    cond_m = np.array(cond_m); cond_s = np.array(cond_s)
    x_ns   = np.array(N_VALUES, dtype=float)

    ax2.errorbar(x_ns, raw_m, yerr=raw_s, fmt="o-",
                 color="#4393c3", lw=2, ms=6, capsize=4, label="Raw WB success")
    ax2.errorbar(x_ns, cond_m, yerr=cond_s, fmt="s--",
                 color="#08306b", lw=2, ms=6, capsize=4, label="Conditional WB success")
    ax2.fill_between(x_ns, (raw_m - raw_s).clip(0),   raw_m + raw_s,
                     alpha=0.12, color="#4393c3")
    ax2.fill_between(x_ns, (cond_m - cond_s).clip(0), cond_m + cond_s,
                     alpha=0.12, color="#08306b")

    ax2.set_xscale("log")
    ax2.set_xticks(N_VALUES)
    ax2.set_xticklabels([str(n) for n in N_VALUES], fontsize=9)
    ax2.set_xlabel("N (stored patterns)", fontsize=10)
    ax2.set_ylabel("WB attack success rate", fontsize=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title(
        "Raw vs conditional WB success across N\n"
        "Class-balanced, 5-seed mean +/- std",
        fontsize=9,
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    fig.suptitle("Phase 3 Diagnostic B: Baseline-corrected attack effectiveness", fontsize=10)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Stdout summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(da: dict, diag_rows: list[dict]) -> None:
    n = da["n"]

    def agg(metric: str, N: int, strategy: str) -> tuple[float, float]:
        vals = [r[metric] for r in diag_rows if r["N"] == N and r["strategy"] == strategy]
        return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    # ── Diagnostic A ──────────────────────────────────────────────────────────
    print(f"\n=== DIAGNOSTIC A: WB vs DE attack agreement (N=100 class_balanced) ===")
    print(f"\nContingency table ({n} paired attacks):")
    print(f"{'':22} {'DE success':>12}  {'DE fail':>10}")
    print(f"  {'WB success':20} {da['both_succ']:>12}  {da['wb_only']:>10}")
    print(f"  {'WB fail':20} {da['de_only']:>12}  {da['both_fail']:>10}")

    agree = da["both_succ"] + da["both_fail"]
    print(f"\nMarginal success: WB={da['wb_rate']:.3f}, DE={da['de_rate']:.3f}")
    print(f"Per-probe agreement: {agree}/{n} ({100*agree/n:.1f}%) agree on success/failure.")

    if da["n_joint"] > 0:
        nj = da["n_joint"]
        print(f"\nOn {nj} jointly-successful probes:")
        print(f"  Same pixel (i,j):           {da['same_pix']:>3}  ({100*da['same_pix']/nj:.1f}%)")
        print(f"  Same row, different column: {da['same_row_difc']:>3}  ({100*da['same_row_difc']/nj:.1f}%)")
        print(f"  Different pixel entirely:   {da['diff_entirely']:>3}  ({100*da['diff_entirely']/nj:.1f}%)")
        print(f"  Same retrieved false-index: {da['same_ret_j']:>3}  ({100*da['same_ret_j']/nj:.1f}%)")
    else:
        print("\nNo jointly-successful probes.")

    print(f"\nPerturbation L2 correlation (WB vs DE, all {n} pairs): r = {da['r_corr']:.4f}")
    print(f"\nVerdict:\n  {da['verdict']}")

    # ── Diagnostic B ──────────────────────────────────────────────────────────
    print(f"\n=== DIAGNOSTIC B: Baseline-corrected attack effectiveness ===")

    print("\nBaseline failure rate (clean retrieval, no attack):")
    print(f"{'N':>8}  {'Random':>18}  {'Class-balanced':>18}")
    print("-" * 50)
    for N in N_VALUES:
        rm, rs = agg("baseline_failure_rate", N, "random")
        cm, cs = agg("baseline_failure_rate", N, "class_balanced")
        print(f"{N:>8}  {rm:>7.3f} +/- {rs:>5.3f}  {cm:>7.3f} +/- {cs:>5.3f}")

    for strategy in STRATEGIES:
        label = "Random" if strategy == "random" else "Class-balanced"
        print(f"\nRaw vs baseline-corrected white-box attack success [{label}]:")
        print(f"{'N':>8}  {'Raw':>18}  {'Conditional':>18}  {'Attributable':>18}")
        print("-" * 70)
        for N in N_VALUES:
            rm, rs  = agg("raw_wb_success",          N, strategy)
            cm, cs  = agg("conditional_wb_success",  N, strategy)
            am, as_ = agg("attack_attributable_rate", N, strategy)
            print(f"{N:>8}  {rm:>7.3f} +/- {rs:>5.3f}  "
                  f"{cm:>7.3f} +/- {cs:>5.3f}  "
                  f"{am:>7.3f} +/- {as_:>5.3f}")

    print("\nHeadline cell (N=100, class_balanced):")
    bfr_m, bfr_s = agg("baseline_failure_rate",                100, "class_balanced")
    rwb_m, rwb_s = agg("raw_wb_success",                       100, "class_balanced")
    cwb_m, cwb_s = agg("conditional_wb_success",               100, "class_balanced")
    rr1_m, rr1_s = agg("raw_random_failure_sigma_001",         100, "class_balanced")
    cr1_m, cr1_s = agg("conditional_random_failure_sigma_001", 100, "class_balanced")
    corr_amplif  = cwb_m / max(cr1_m, 1e-6)

    print(f"  Baseline retrieval failure:              {bfr_m:.3f} +/- {bfr_s:.3f}")
    print(f"  Raw WB attack success:                   {rwb_m:.3f} +/- {rwb_s:.3f}")
    print(f"  Conditional WB attack success:           {cwb_m:.3f} +/- {cwb_s:.3f}")
    print(f"  Conditional random failure (sigma=0.01): {cr1_m:.3f} +/- {cr1_s:.3f}")
    print(f"  Corrected adversarial amplification:     {corr_amplif:.2f}x")

    print("\nVerdict:")
    print(
        f"  Baseline retrieval failure at headline cell is {bfr_m*100:.1f}%. "
        f"Conditional attack success is {cwb_m*100:.1f}%, which represents a "
        f"{corr_amplif:.1f}x amplification over conditional random-noise failure of "
        f"{cr1_m*100:.1f}% at equivalent perturbation magnitude. "
        f"This is the methodologically defensible headline figure for the thesis."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    print("=" * 62)
    print("Phase 3 Diagnostics")
    print("=" * 62)

    print("\nLoading MNIST ...")
    images, labels = load_mnist_train()

    print("\n--- Diagnostic A: WB vs DE agreement ---")
    da = run_diagnostic_a(EXP_DIR)
    print(f"  Saved phase3_diag_a_paired.csv  ({da['n']} rows)")

    print("\n--- Diagnostic B: Baseline correction ---")
    diag_rows = run_diagnostic_b(images, labels, EXP_DIR)
    print(f"  Saved phase3_diag_b_baseline_corrected.csv  ({len(diag_rows)} rows)")

    print("\nGenerating figures ...")
    save_figure_a(da, str(FIG_DIR / "phase3_diag_a_contingency.png"))
    print("  Saved: phase3_diag_a_contingency.png")
    save_figure_b(diag_rows, str(FIG_DIR / "phase3_diag_b_corrected.png"))
    print("  Saved: phase3_diag_b_corrected.png")

    print_summary(da, diag_rows)
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
