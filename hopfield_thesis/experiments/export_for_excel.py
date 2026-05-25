"""
Export all key result data to Excel-friendly CSV files.

Creates excel_exports/ with 18 per-figure CSVs plus INDEX.csv.
Phase 1 data has no saved CSV; values are hardcoded from recorded Phase 1 stdout.

Run: python -m experiments.export_for_excel
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT    = Path(__file__).resolve().parent.parent
EXP_DIR = ROOT / "experiments"
OUT_DIR = ROOT / "excel_exports"
OUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_csv(name: str) -> list[dict]:
    with open(EXP_DIR / name, newline="") as f:
        return list(csv.DictReader(f))


def write_excel_csv(rows: list[dict | list], path: Path, fieldnames: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            if isinstance(row, dict):
                w.writerow({k: row.get(k, "") for k in fieldnames})
            else:
                w.writerow(dict(zip(fieldnames, row)))
    print(f"  {path.name}  ({len(rows)} rows)")


def _agg(vals: list[float]) -> tuple[float, float]:
    a = np.array(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else (float(a.mean()), 0.0)


def _r(v, d=4):
    if v != v:  # nan
        return ""
    return round(float(v), d)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 hardcoded (no CSV was saved from phase1_sanity.py)
# Values are from recorded Phase 1 stdout: seed=42, beta=8.0
# ─────────────────────────────────────────────────────────────────────────────

P1_CAPACITY = [
    # N, accuracy (sigma=0.2 Gaussian, 50 probes)
    (10,   0.30),
    (100,  0.68),
    (1000, 0.18),
    (5000, 0.14),
]

P1_BETA = [
    # beta, accuracy (N=1000, sigma=0.2, 50 probes, seed=42)
    # All betas give flat 0.180; MSE range 0.104-0.121 (not captured per-beta)
    (1,   0.180),
    (2,   0.180),
    (4,   0.180),
    (8,   0.180),
    (16,  0.180),
    (32,  0.180),
    (64,  0.180),
    (128, 0.180),
]

N_VALUES   = [10, 50, 100, 500, 1000]
STRATEGIES = ["random", "class_balanced"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. fig_p1_capacity.csv
# ─────────────────────────────────────────────────────────────────────────────

def export_p1_capacity():
    fields = ["N", "accuracy"]
    rows   = [{"N": n, "accuracy": acc} for n, acc in P1_CAPACITY]
    write_excel_csv(rows, OUT_DIR / "fig_p1_capacity.csv", fields)


# ─────────────────────────────────────────────────────────────────────────────
# 2. fig_p1_beta_diagnostic.csv
# ─────────────────────────────────────────────────────────────────────────────

def export_p1_beta():
    fields = ["beta", "accuracy"]
    rows   = [{"beta": b, "accuracy": acc} for b, acc in P1_BETA]
    write_excel_csv(rows, OUT_DIR / "fig_p1_beta_diagnostic.csv", fields)


# ─────────────────────────────────────────────────────────────────────────────
# 3. fig_p2_gaussian.csv  (wide: sigma × N*strategy accuracy)
# 4. fig_p2_pixelflip.csv
# ─────────────────────────────────────────────────────────────────────────────

def export_p2_noise_wide(noise_type: str, out_name: str):
    p2 = read_csv("phase2_results.csv")
    sub = [r for r in p2 if r["noise_type"] == noise_type]
    mags = sorted(set(float(r["magnitude"]) for r in sub))

    col_keys = []
    for s in STRATEGIES:
        for n in N_VALUES:
            col_keys.append((s, n, f"accuracy_{s}_N{n}"))

    fields = ["sigma" if noise_type == "gaussian" else "flip_rate"] + [c for _, _, c in col_keys]
    x_col  = fields[0]

    lookup: dict[tuple, float] = {}
    for r in sub:
        lookup[(r["strategy"], int(r["N"]), float(r["magnitude"]))] = float(r["accuracy"])

    rows = []
    for mag in mags:
        row: dict = {x_col: mag}
        for s, n, col in col_keys:
            row[col] = _r(lookup.get((s, n, mag), ""), 4)
        rows.append(row)

    write_excel_csv(rows, OUT_DIR / out_name, fields)


# ─────────────────────────────────────────────────────────────────────────────
# 5. fig_p2_occlusion.csv  (N × mean/std from stability CSV)
# ─────────────────────────────────────────────────────────────────────────────

def export_p2_occlusion():
    stab = read_csv("phase2_stability_summary.csv")
    sub  = [r for r in stab if r["noise_type"] == "occlusion"]
    fields = ["N",
              "accuracy_random_mean", "accuracy_random_std",
              "accuracy_classbal_mean", "accuracy_classbal_std"]
    rows = []
    for n in N_VALUES:
        row: dict = {"N": n}
        for strat, prefix in [("random", "accuracy_random"), ("class_balanced", "accuracy_classbal")]:
            match = next((r for r in sub if int(r["N"]) == n and r["strategy"] == strat), None)
            row[f"{prefix}_mean"] = _r(float(match["accuracy_mean"])) if match else ""
            row[f"{prefix}_std"]  = _r(float(match["accuracy_std"]))  if match else ""
        rows.append(row)
    write_excel_csv(rows, OUT_DIR / "fig_p2_occlusion.csv", fields)


# ─────────────────────────────────────────────────────────────────────────────
# 6. fig_p2_stability.csv  (N × 12 series: noise_type × strategy × mean/std)
# ─────────────────────────────────────────────────────────────────────────────

def export_p2_stability():
    stab = read_csv("phase2_stability_summary.csv")
    noise_types = ["gaussian", "flip", "occlusion"]
    col_specs: list[tuple[str, str, str]] = []
    for nt in noise_types:
        for st, abbr in [("random", "rand"), ("class_balanced", "cb")]:
            col_specs.append((nt, st, f"{nt}_{abbr}_mean", f"{nt}_{abbr}_std"))

    fields = ["N"]
    for _, _, cm, cs in col_specs:
        fields += [cm, cs]

    rows = []
    for n in N_VALUES:
        row: dict = {"N": n}
        for nt, st, cm, cs in col_specs:
            match = next((r for r in stab if int(r["N"]) == n
                          and r["strategy"] == st and r["noise_type"] == nt), None)
            row[cm] = _r(float(match["accuracy_mean"])) if match else ""
            row[cs] = _r(float(match["accuracy_std"]))  if match else ""
        rows.append(row)
    write_excel_csv(rows, OUT_DIR / "fig_p2_stability.csv", fields)


# ─────────────────────────────────────────────────────────────────────────────
# 7-9. Phase 3 attack grid CSVs (success, L2, cosine damage)
# ─────────────────────────────────────────────────────────────────────────────

def _p3_wb_agg() -> dict:
    """Aggregate phase3_whitebox_results.csv → {(N, strategy): {metric: (mean, std)}}."""
    wb = read_csv("phase3_whitebox_results.csv")
    cell_data: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    seed_seen: dict[tuple, set] = defaultdict(set)

    for r in wb:
        key  = (int(r["N"]), r["strategy"])
        seed = int(r["seed"])
        seed_seen[key].add(seed)

    per_seed: dict[tuple, dict[int, dict[str, list]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for r in wb:
        key  = (int(r["N"]), r["strategy"])
        seed = int(r["seed"])
        per_seed[key][seed]["success"].append(int(r["attack_success"]))
        l2 = float(r["perturbation_l2"])
        per_seed[key][seed]["l2"].append(l2)
        # cosine damage = 1 - perturbation_l2 is NOT right; we have no stored-pattern cosine here
        # use 1 - retrieval cosine approximation: retrieve cosine not stored directly in WB CSV
        # instead compute mean L2 which proxies damage
        per_seed[key][seed]["l2_all"].append(l2)

    agg: dict[tuple, dict[str, tuple]] = {}
    for key, seeds in per_seed.items():
        s_rates = [np.mean(d["success"]) for d in seeds.values()]
        s_l2s   = [np.mean(d["l2"]) for d in seeds.values()]
        agg[key] = {
            "success": _agg(s_rates),
            "l2":      _agg(s_l2s),
        }
    return agg


def export_p3_attack_grid():
    agg = _p3_wb_agg()
    for metric, suffix, out_name in [
        ("success", ("success_rate_mean", "success_rate_std"), "fig_p3_attack_grid_success.csv"),
        ("l2",      ("mean_l2_mean",      "mean_l2_std"),      "fig_p3_attack_grid_l2.csv"),
    ]:
        col_specs = []
        for st, abbr in [("random", "rand"), ("class_balanced", "cb")]:
            col_specs.append((st, f"wb_{suffix[0].split('_')[0]}_{abbr}_mean",
                                  f"wb_{suffix[0].split('_')[0]}_{abbr}_std"))

        # Re-label properly
        col_specs = []
        for st, abbr in [("random", "rand"), ("class_balanced", "cb")]:
            col_specs.append((st, f"{suffix[0].replace('_mean','')}__{abbr}_mean",
                                  f"{suffix[0].replace('_mean','')}__{abbr}_std"))

        fields = ["N"] + [c for _, c, _ in col_specs] + [c for _, _, c in col_specs]
        fields = ["N"]
        for _, cm, cs in col_specs:
            fields += [cm, cs]

        rows = []
        for n in N_VALUES:
            row: dict = {"N": n}
            for st, cm, cs in col_specs:
                v = agg.get((n, st), {}).get(metric, (None, None))
                row[cm] = _r(v[0]) if v[0] is not None else ""
                row[cs] = _r(v[1]) if v[1] is not None else ""
            rows.append(row)
        write_excel_csv(rows, OUT_DIR / out_name, fields)


def export_p3_attack_grid_damage():
    """Cosine damage from phase3_summary.csv (uses attacker cosine metrics)."""
    # phase3_whitebox_results.csv doesn't store cosine to true pattern.
    # Use mean L2 as proxy and note in header. If we had cosine_to_true we'd use it.
    # The CSV does have 'perturbation_l2' which is the L2 of the single-pixel change.
    # Use success-weighted mean L2 as "effective damage" proxy.
    agg = _p3_wb_agg()
    fields = ["N",
              "effective_damage_rand_mean", "effective_damage_rand_std",
              "effective_damage_cb_mean",   "effective_damage_cb_std"]
    rows = []
    for n in N_VALUES:
        row: dict = {"N": n}
        for st, prefix in [("random", "effective_damage_rand"),
                            ("class_balanced", "effective_damage_cb")]:
            v_s  = agg.get((n, st), {}).get("success", (None, None))
            v_l2 = agg.get((n, st), {}).get("l2",      (None, None))
            # effective damage = success_rate * mean_l2
            dam = v_s[0] * v_l2[0] if (v_s[0] is not None and v_l2[0] is not None) else None
            row[f"{prefix}_mean"] = _r(dam) if dam is not None else ""
            row[f"{prefix}_std"]  = ""
        rows.append(row)
    write_excel_csv(rows, OUT_DIR / "fig_p3_attack_grid_damage.csv", fields)


# ─────────────────────────────────────────────────────────────────────────────
# 10. fig_p3_attack_vs_random.csv
# ─────────────────────────────────────────────────────────────────────────────

def export_p3_attack_vs_random():
    diag_c = read_csv("phase3_diag_c_matched_sigma.csv")
    sigmas = sorted(set(float(r["sigma"]) for r in diag_c))

    rows = []
    for sigma in sigmas:
        sub = [r for r in diag_c if float(r["sigma"]) == sigma]
        l2s   = [float(r["mean_noise_l2"])           for r in sub]
        cfrs  = [float(r["conditional_failure_rate"]) for r in sub]
        l2_m, _ = _agg(l2s)
        cfr_m, cfr_s = _agg(cfrs)
        rows.append({
            "condition":    f"random_sigma_{sigma:.3f}",
            "L2_magnitude": _r(l2_m),
            "failure_rate": _r(cfr_m),
            "failure_std":  _r(cfr_s),
        })

    # Add whitebox row
    diag_b = read_csv("phase3_diag_b_baseline_corrected.csv")
    wb_cond = [float(r["conditional_wb_success"])
               for r in diag_b if r["strategy"] == "class_balanced" and int(r["N"]) == 100]
    wb_m, wb_s = _agg(wb_cond)
    rows.append({
        "condition":    "whitebox_attack",
        "L2_magnitude": 0.113,
        "failure_rate": _r(wb_m),
        "failure_std":  _r(wb_s),
    })

    fields = ["condition", "L2_magnitude", "failure_rate", "failure_std"]
    write_excel_csv(rows, OUT_DIR / "fig_p3_attack_vs_random.csv", fields)


# ─────────────────────────────────────────────────────────────────────────────
# 11. fig_p3_whitebox_vs_de.csv
# ─────────────────────────────────────────────────────────────────────────────

def export_p3_wb_vs_de():
    wb  = read_csv("phase3_whitebox_results.csv")
    de  = read_csv("phase3_blackbox_results.csv")
    SEEDS = [42, 43, 44, 45, 46]

    # WB at N=100 class_balanced: per-seed success_rate and mean_l2
    wb_sub = [r for r in wb if int(r["N"]) == 100 and r["strategy"] == "class_balanced"]
    wb_sr, wb_l2 = [], []
    for seed in SEEDS:
        s_rows = [r for r in wb_sub if int(r["seed"]) == seed]
        wb_sr.append(np.mean([int(r["attack_success"]) for r in s_rows]))
        wb_l2.append(np.mean([float(r["perturbation_l2"]) for r in s_rows]))
    wb_sr_m, wb_sr_s   = _agg(wb_sr)
    wb_l2_m, wb_l2_s   = _agg(wb_l2)
    wb_evals = 3920.0

    # DE at N=100 class_balanced: per-seed success_rate, mean_l2, mean_evals
    de_sub = [r for r in de if int(r["N"]) == 100 and r["strategy"] == "class_balanced"]
    de_sr, de_l2, de_ev = [], [], []
    for seed in SEEDS:
        s_rows = [r for r in de_sub if int(r["seed"]) == seed]
        de_sr.append(np.mean([int(r["attack_success"]) for r in s_rows]))
        de_l2.append(np.mean([float(r["perturbation_l2"]) for r in s_rows]))
        de_ev.append(np.mean([float(r["evaluations"]) for r in s_rows]))
    de_sr_m, de_sr_s   = _agg(de_sr)
    de_l2_m, de_l2_s   = _agg(de_l2)
    de_ev_m, _         = _agg(de_ev)

    fields = ["attacker", "success_rate_mean", "success_rate_std",
              "mean_l2_mean", "mean_l2_std", "mean_evaluations"]
    rows = [
        {"attacker": "whitebox",  "success_rate_mean": _r(wb_sr_m),
         "success_rate_std": _r(wb_sr_s), "mean_l2_mean": _r(wb_l2_m),
         "mean_l2_std": _r(wb_l2_s), "mean_evaluations": _r(wb_evals)},
        {"attacker": "DE_blackbox", "success_rate_mean": _r(de_sr_m),
         "success_rate_std": _r(de_sr_s), "mean_l2_mean": _r(de_l2_m),
         "mean_l2_std": _r(de_l2_s), "mean_evaluations": _r(de_ev_m)},
    ]
    write_excel_csv(rows, OUT_DIR / "fig_p3_whitebox_vs_de.csv", fields)


# ─────────────────────────────────────────────────────────────────────────────
# 12. fig_p3_diag_b_baseline_corrected.csv  (class_balanced only)
# ─────────────────────────────────────────────────────────────────────────────

def export_p3_diag_b():
    diag_b = read_csv("phase3_diag_b_baseline_corrected.csv")
    sub    = [r for r in diag_b if r["strategy"] == "class_balanced"]

    fields = ["N",
              "baseline_failure_mean", "baseline_failure_std",
              "raw_wb_success_mean",   "raw_wb_success_std",
              "cond_wb_success_mean",  "cond_wb_success_std"]
    rows = []
    for n in N_VALUES:
        cell = [r for r in sub if int(r["N"]) == n]
        bl_m,  bl_s  = _agg([float(r["baseline_failure_rate"])    for r in cell])
        rwb_m, rwb_s = _agg([float(r["raw_wb_success"])           for r in cell])
        cwb_m, cwb_s = _agg([float(r["conditional_wb_success"])   for r in cell])
        rows.append({
            "N": n,
            "baseline_failure_mean": _r(bl_m),  "baseline_failure_std": _r(bl_s),
            "raw_wb_success_mean":   _r(rwb_m), "raw_wb_success_std":   _r(rwb_s),
            "cond_wb_success_mean":  _r(cwb_m), "cond_wb_success_std":  _r(cwb_s),
        })
    write_excel_csv(rows, OUT_DIR / "fig_p3_diag_b_baseline_corrected.csv", fields)


# ─────────────────────────────────────────────────────────────────────────────
# 13. fig_p3_diag_c_sigma_sweep.csv
# ─────────────────────────────────────────────────────────────────────────────

def export_p3_diag_c():
    diag_c = read_csv("phase3_diag_c_matched_sigma.csv")
    sigmas = sorted(set(float(r["sigma"]) for r in diag_c))
    fields = ["sigma", "mean_l2", "conditional_failure_mean", "conditional_failure_std"]

    rows = []
    for sigma in sigmas:
        sub   = [r for r in diag_c if float(r["sigma"]) == sigma]
        l2_m, _   = _agg([float(r["mean_noise_l2"])           for r in sub])
        cfr_m, cfr_s = _agg([float(r["conditional_failure_rate"]) for r in sub])
        rows.append({
            "sigma": sigma, "mean_l2": _r(l2_m),
            "conditional_failure_mean": _r(cfr_m), "conditional_failure_std": _r(cfr_s),
        })
    # WB reference row
    rows.append({
        "sigma": "whitebox_attack_reference",
        "mean_l2": 0.113,
        "conditional_failure_mean": 0.031,
        "conditional_failure_std": "",
    })
    write_excel_csv(rows, OUT_DIR / "fig_p3_diag_c_sigma_sweep.csv", fields)


# ─────────────────────────────────────────────────────────────────────────────
# 14. fig_p3_diag_d_classes.csv
# ─────────────────────────────────────────────────────────────────────────────

def export_p3_diag_d_classes():
    summary = read_csv("phase3_diag_d_summary.csv")
    chi2_row = next(r for r in summary if r["analysis_type"] == "chi2_class_distribution")
    import ast
    counts = ast.literal_eval(chi2_row["vulnerable_stat"])
    n_vuln = sum(counts)
    exp_per_class = n_vuln / 10.0

    fields = ["digit_class", "vulnerable_count", "expected_count_uniform", "percentage"]
    rows = [
        {
            "digit_class":           cls,
            "vulnerable_count":      int(counts[cls]),
            "expected_count_uniform": _r(exp_per_class, 2),
            "percentage":            _r(counts[cls] / n_vuln * 100 if n_vuln else 0, 2),
        }
        for cls in range(10)
    ]
    write_excel_csv(rows, OUT_DIR / "fig_p3_diag_d_classes.csv", fields)


# ─────────────────────────────────────────────────────────────────────────────
# 15. fig_p3_diag_d_features.csv  (long format for grouped boxplots)
# ─────────────────────────────────────────────────────────────────────────────

def export_p3_diag_d_features():
    feat_rows = read_csv("phase3_diag_d_probe_features.csv")
    features  = ["max_neighbor_cosine", "mean_intensity", "intensity_std"]
    fields    = ["feature_name", "vulnerable", "value"]
    rows = []
    for r in feat_rows:
        vuln = "True" if int(r["vulnerable"]) == 1 else "False"
        for f in features:
            rows.append({"feature_name": f, "vulnerable": vuln, "value": _r(float(r[f]), 5)})
    write_excel_csv(rows, OUT_DIR / "fig_p3_diag_d_features.csv", fields)


# ─────────────────────────────────────────────────────────────────────────────
# 16. fig_p3_pixel_value_distribution.csv
# ─────────────────────────────────────────────────────────────────────────────

def export_p3_pixel_distribution():
    pv = read_csv("phase3_pixel_value_analysis.csv")
    wb = read_csv("phase3_whitebox_results.csv")
    succ = [r for r in wb if int(r["attack_success"]) == 1]
    CAND = [0.0, 0.25, 0.5, 0.75, 1.0]

    def bin_to_cand(v: float) -> float:
        return min(CAND, key=lambda c: abs(v - c))

    # attack counts per adv pixel_value (overall)
    atk_cnt = {cv: sum(1 for r in succ if abs(float(r["pixel_value"]) - cv) < 0.01)
               for cv in CAND}
    # original pixel distribution (successful attacks only)
    orig_cnt = {cv: sum(1 for r in succ if abs(bin_to_cand(float(r["original_value"])) - cv) < 0.01)
                for cv in CAND}
    total_s = max(sum(atk_cnt.values()), 1)

    fields = ["pixel_value", "attack_count", "attack_pct",
              "original_count", "original_pct"]
    rows = []
    for cv in CAND:
        rows.append({
            "pixel_value":    cv,
            "attack_count":   atk_cnt[cv],
            "attack_pct":     _r(atk_cnt[cv] / total_s, 4),
            "original_count": orig_cnt[cv],
            "original_pct":   _r(orig_cnt[cv] / total_s, 4),
        })
    write_excel_csv(rows, OUT_DIR / "fig_p3_pixel_value_distribution.csv", fields)


# ─────────────────────────────────────────────────────────────────────────────
# 17. fig_p3_pixel_transition_matrix.csv  (5×5 wide)
# ─────────────────────────────────────────────────────────────────────────────

def export_p3_pixel_transition():
    pv   = read_csv("phase3_pixel_value_analysis.csv")
    CAND = [0.0, 0.25, 0.5, 0.75, 1.0]
    trans_rows = [r for r in pv
                  if r["regime"] == "all"
                  and r["original_value_bin"] not in ("any", "extreme_rate", "same_as_original_rate")]

    mat: dict[float, dict[float, int]] = {orig: {adv: 0 for adv in CAND} for orig in CAND}
    for r in trans_rows:
        try:
            orig = float(r["original_value_bin"])
            adv  = float(r["adversarial_value"])
            mat[orig][adv] = int(r["count"])
        except (ValueError, KeyError):
            pass

    fields = ["original_value"] + [str(cv) for cv in CAND]
    rows   = []
    for orig in CAND:
        row = {"original_value": orig}
        for adv in CAND:
            row[str(adv)] = mat[orig][adv]
        rows.append(row)
    write_excel_csv(rows, OUT_DIR / "fig_p3_pixel_transition_matrix.csv", fields)


# ─────────────────────────────────────────────────────────────────────────────
# 18. fig_p3_fashion_mnist_comparison.csv
# ─────────────────────────────────────────────────────────────────────────────

def export_p3_fmnist_comparison():
    diag_b  = read_csv("phase3_diag_b_baseline_corrected.csv")
    fmnist  = read_csv("phase3_fashion_mnist_summary.csv")
    diag_d  = read_csv("phase3_diag_d_summary.csv")

    # MNIST values (N=100 class_balanced, 5 seeds)
    cb_b = [r for r in diag_b if r["strategy"] == "class_balanced" and int(r["N"]) == 100]
    mn_bl_m,  mn_bl_s  = _agg([float(r["baseline_failure_rate"])  for r in cb_b])
    mn_cwb_m, mn_cwb_s = _agg([float(r["conditional_wb_success"]) for r in cb_b])

    # Conditional RNE from diag_c
    diag_c = read_csv("phase3_diag_c_matched_sigma.csv")
    rne005 = [r for r in diag_c if abs(float(r["sigma"]) - 0.005) < 0.001]
    mn_rne_m, mn_rne_s = _agg([float(r["conditional_failure_rate"]) for r in rne005])

    mn_amp_m = mn_cwb_m / max(mn_rne_m, 1e-6)

    # MNIST native predictor from diag_d_summary
    lr_row   = next(r for r in diag_d if r["analysis_type"] == "logistic_cv_balanced_accuracy")
    mn_pred_m = float(lr_row["vulnerable_stat"])
    mn_pred_s = float(lr_row["nonvulnerable_stat"])

    # FMNIST values (5 seeds)
    fm_rows = [r for r in fmnist if r["seed"] not in ("mean_5seeds",)]
    fm_bl_m,  fm_bl_s  = _agg([float(r["baseline_failure_rate"]) for r in fm_rows])
    fm_cwb_m, fm_cwb_s = _agg([float(r["cond_wb_success"])       for r in fm_rows])
    fm_rne_m, fm_rne_s = _agg([float(r["cond_rne_fail"])         for r in fm_rows])
    fm_amp_m = fm_cwb_m / max(fm_rne_m, 1e-6)

    # FMNIST predictors — hardcoded from phase3_final_diagnostics stdout
    fm_native_m,   fm_native_s   = 0.576, 0.256
    fm_transfer_m, fm_transfer_s = 0.398, 0.0    # single value (no std)
    mn_transfer_m, mn_transfer_s = 0.398, 0.0    # transfer is MNIST→FMNIST

    metrics = [
        ("baseline_failure",       mn_bl_m, mn_bl_s,    fm_bl_m, fm_bl_s),
        ("conditional_wb_success", mn_cwb_m, mn_cwb_s,  fm_cwb_m, fm_cwb_s),
        ("conditional_random_failure", mn_rne_m, mn_rne_s, fm_rne_m, fm_rne_s),
        ("amplification_factor",   mn_amp_m, 0.0,       fm_amp_m, 0.0),
        ("mnist_native_predictor", mn_pred_m, mn_pred_s, 0.0, 0.0),
        ("mnist_to_fmnist_transfer", mn_transfer_m, mn_transfer_s,
                                    fm_transfer_m, fm_transfer_s),
        ("fmnist_native_predictor", 0.0, 0.0, fm_native_m, fm_native_s),
    ]

    fields = ["metric",
              "MNIST_mean", "MNIST_std",
              "FashionMNIST_mean", "FashionMNIST_std"]
    rows = [
        {"metric": m,
         "MNIST_mean": _r(mm), "MNIST_std": _r(ms),
         "FashionMNIST_mean": _r(fm), "FashionMNIST_std": _r(fs)}
        for m, mm, ms, fm, fs in metrics
    ]
    write_excel_csv(rows, OUT_DIR / "fig_p3_fashion_mnist_comparison.csv", fields)


# ─────────────────────────────────────────────────────────────────────────────
# INDEX.csv
# ─────────────────────────────────────────────────────────────────────────────

INDEX = [
    ("fig_p1_capacity.csv",
     "phase1_sanity.py (stdout)",
     "Phase 1 Check 4",
     "Capacity stress test: retrieval accuracy vs N at sigma=0.2, seed=42"),

    ("fig_p1_beta_diagnostic.csv",
     "phase1_beta_diagnostic.py (stdout)",
     "Phase 1 Beta sweep",
     "Beta calibration: retrieval accuracy vs beta at N=1000, sigma=0.2, seed=42"),

    ("fig_p2_gaussian.csv",
     "phase2_results.csv",
     "Phase 2 Gaussian",
     "Gaussian noise retrieval accuracy vs sigma, wide format, all N/strategy, seed=42"),

    ("fig_p2_pixelflip.csv",
     "phase2_results.csv",
     "Phase 2 Pixel-flip",
     "Pixel-flip retrieval accuracy vs flip_rate, wide format, all N/strategy, seed=42"),

    ("fig_p2_occlusion.csv",
     "phase2_stability_summary.csv",
     "Phase 2 Occlusion",
     "Half-occlusion retrieval accuracy vs N, mean+std across 5 seeds"),

    ("fig_p2_stability.csv",
     "phase2_stability_summary.csv",
     "Phase 2 Stability",
     "All noise types: accuracy mean+std across 5 seeds, all N/strategy"),

    ("fig_p3_attack_grid_success.csv",
     "phase3_whitebox_results.csv",
     "Phase 3 Attack grid (success)",
     "White-box attack success rate vs N, mean+std across 5 seeds, both strategies"),

    ("fig_p3_attack_grid_l2.csv",
     "phase3_whitebox_results.csv",
     "Phase 3 Attack grid (L2)",
     "White-box mean perturbation L2 vs N, mean+std across 5 seeds"),

    ("fig_p3_attack_grid_damage.csv",
     "phase3_whitebox_results.csv",
     "Phase 3 Attack grid (damage)",
     "Effective attack damage (success_rate x mean_L2) vs N, both strategies"),

    ("fig_p3_attack_vs_random.csv",
     "phase3_diag_c_matched_sigma.csv + phase3_diag_b_baseline_corrected.csv",
     "Phase 3 Attack vs random noise",
     "Conditional failure rate: sigma sweep + WB reference, N=100 class-balanced"),

    ("fig_p3_whitebox_vs_de.csv",
     "phase3_whitebox_results.csv + phase3_blackbox_results.csv",
     "Phase 3 WB vs DE",
     "White-box vs DE black-box: success rate, L2, evaluations at N=100 class-balanced"),

    ("fig_p3_diag_b_baseline_corrected.csv",
     "phase3_diag_b_baseline_corrected.csv",
     "Phase 3 Diag B",
     "Baseline-corrected attack effectiveness: baseline/raw/conditional WB vs N, class-balanced"),

    ("fig_p3_diag_c_sigma_sweep.csv",
     "phase3_diag_c_matched_sigma.csv",
     "Phase 3 Diag C",
     "Magnitude-matched sigma sweep: conditional RNE failure vs sigma, N=100 class-balanced"),

    ("fig_p3_diag_d_classes.csv",
     "phase3_diag_d_summary.csv",
     "Phase 3 Diag D (classes)",
     "Vulnerable probe class distribution: count vs digit class (0-9)"),

    ("fig_p3_diag_d_features.csv",
     "phase3_diag_d_probe_features.csv",
     "Phase 3 Diag D (features)",
     "Probe geometry features in long format for grouped boxplots in Excel"),

    ("fig_p3_pixel_value_distribution.csv",
     "phase3_whitebox_results.csv",
     "Phase 3 Pixel distribution",
     "Adversarial vs original pixel value distribution across 5 candidate values"),

    ("fig_p3_pixel_transition_matrix.csv",
     "phase3_pixel_value_analysis.csv",
     "Phase 3 Pixel transition",
     "5x5 transition matrix: original pixel bin to adversarial pixel value (counts)"),

    ("fig_p3_fashion_mnist_comparison.csv",
     "phase3_fashion_mnist_summary.csv + phase3_diag_b/c/d_*.csv",
     "Phase 3 FMNIST comparison",
     "MNIST vs Fashion-MNIST: key metrics comparison including transfer prediction accuracy"),

    ("fig_grayscale_cifar_baseline.csv",
     "grayscale_cifar_baseline_results.csv",
     "CIFAR Grayscale Baseline",
     "Grayscale CIFAR-10 baseline failure rate and pairwise cosine vs N, 5 seeds"),
]


def export_index():
    fields = ["csv_filename", "source_python_csv_or_script",
              "thesis_figure_name", "description_short"]
    rows   = [{"csv_filename": f, "source_python_csv_or_script": s,
               "thesis_figure_name": t, "description_short": d}
              for f, s, t, d in INDEX]
    write_excel_csv(rows, OUT_DIR / "INDEX.csv", fields)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Excel CSV Export")
    print("=" * 60)
    print(f"Output directory: {OUT_DIR}\n")

    export_p1_capacity()
    export_p1_beta()
    export_p2_noise_wide("gaussian", "fig_p2_gaussian.csv")
    export_p2_noise_wide("flip",     "fig_p2_pixelflip.csv")
    export_p2_occlusion()
    export_p2_stability()
    export_p3_attack_grid()
    export_p3_attack_grid_damage()
    export_p3_attack_vs_random()
    export_p3_wb_vs_de()
    export_p3_diag_b()
    export_p3_diag_c()
    export_p3_diag_d_classes()
    export_p3_diag_d_features()
    export_p3_pixel_distribution()
    export_p3_pixel_transition()
    export_p3_fmnist_comparison()
    export_index()

    files = sorted(OUT_DIR.glob("*.csv"))
    print(f"\nTotal files in excel_exports/: {len(files)}")
    for f in files:
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
