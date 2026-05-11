"""
Phase 3 final diagnostics.

Stage A: Fashion-MNIST cross-validation at the headline operating point
         (N=100, class-balanced).
Stage B: Pixel-value distribution analysis from existing
         phase3_whitebox_results.csv — no new attacks are run.

Run: python -m experiments.phase3_final_diagnostics
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
from hopfield.sampling   import sample_class_balanced
from hopfield.attacks    import WhiteBoxOnePixelAttacker

# ── config — must match Phase 3 exactly ──────────────────────────────────────
SEEDS      = [42, 43, 44, 45, 46]
BETA       = 8.0
N          = 100
STRATEGY   = "class_balanced"
N_PROBE    = 50
N_VALUES   = [10, 50, 100, 500, 1000]
STRATEGIES = ["random", "class_balanced"]
N_IDX      = N_VALUES.index(N)           # 2
S_IDX      = STRATEGIES.index(STRATEGY)  # 1

MATCHED_SIGMA = 0.005   # from Diagnostic C: best magnitude-matched sigma

# MNIST headline metrics
MNIST_RAW_WB_SUCCESS  = 0.120
MNIST_COND_WB_SUCCESS = 0.031
MNIST_COND_RNE_FAIL   = 0.009

CAND_VALS = [0.0, 0.25, 0.5, 0.75, 1.0]

DATA_DIR = ROOT / "data"
FIG_DIR  = ROOT / "figures"
EXP_DIR  = ROOT / "experiments"
FIG_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_fmnist_train() -> tuple[torch.Tensor, torch.Tensor]:
    ds = torchvision.datasets.FashionMNIST(
        root=str(DATA_DIR), train=True, download=False, transform=T.ToTensor()
    )
    return ds.data.float().view(-1, 784) / 255.0, ds.targets


def read_csv_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def bin_to_cand(v: float) -> float:
    return min(CAND_VALS, key=lambda c: abs(v - c))


# ─────────────────────────────────────────────────────────────────────────────
# Cell builder — identical seeding to Phase 3
# ─────────────────────────────────────────────────────────────────────────────

def build_cell(
    images: torch.Tensor, labels: torch.Tensor, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, ContinuousHopfield, list[int]]:
    X, _ = sample_class_balanced((images, labels), N, seed=seed + N_IDX)
    stored = X.T.contiguous()   # (N, 784)
    hop    = ContinuousHopfield(X, beta=BETA)
    probe_seed = seed * 1000 + N_IDX * 100 + S_IDX * 10
    rng = torch.Generator()
    rng.manual_seed(probe_seed)
    probe_indices = torch.randperm(N, generator=rng)[:N_PROBE].tolist()
    return X, stored, hop, probe_indices


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers (no scipy)
# ─────────────────────────────────────────────────────────────────────────────

def _norm_sf(z: float) -> float:
    return math.erfc(z / math.sqrt(2)) / 2


def chi2_pvalue(stat: float, df: int) -> float:
    if stat <= 0 or df <= 0:
        return 1.0
    z = ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return max(0.0, min(1.0, _norm_sf(z)))


def mannwhitney_u(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
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
    Rx  = ranks[:nx].sum()
    U   = Rx - nx * (nx + 1) / 2.0
    mu  = nx * ny / 2.0
    sig = math.sqrt(nx * ny * (nx + ny + 1) / 12.0)
    z   = abs(U - mu) / sig if sig > 0 else 0.0
    p   = min(1.0, 2.0 * _norm_sf(z))
    return float(U), float(p)


# ─────────────────────────────────────────────────────────────────────────────
# Logistic regression — balanced class weights, 5-fold CV
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    n_pos = y.sum();  n_neg = len(y) - n_pos
    return np.where(y == 1,
                    len(y) / (2.0 * max(n_pos, 1)),
                    len(y) / (2.0 * max(n_neg, 1)))


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
    n   = len(y)
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    fs  = n // n_folds
    bal_accs = []
    for fold in range(n_folds):
        te = idx[fold * fs:(fold + 1) * fs]
        tr = np.concatenate([idx[:fold * fs], idx[(fold + 1) * fs:]])
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]
        mu = Xtr.mean(0);  sd = Xtr.std(0) + 1e-8
        Xtr = (Xtr - mu) / sd;  Xte = (Xte - mu) / sd
        w    = _logistic_fit(Xtr, ytr, _balanced_weights(ytr))
        pred = (_sigmoid(np.column_stack([Xte, np.ones(len(Xte))]) @ w) >= 0.5).astype(int)
        pos  = yte == 1;  neg = yte == 0
        tpr  = float((pred[pos] == 1).mean()) if pos.any() else 0.0
        tnr  = float((pred[neg] == 0).mean()) if neg.any() else 0.0
        bal_accs.append((tpr + tnr) / 2.0)
    return float(np.mean(bal_accs)), float(np.std(bal_accs, ddof=1))


def logistic_fit_full(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(0);  sd = X.std(0) + 1e-8
    w  = _logistic_fit((X - mu) / sd, y, _balanced_weights(y))
    return w, mu, sd


def logistic_predict_proba(
    X: np.ndarray, w: np.ndarray, mu: np.ndarray, sd: np.ndarray,
) -> np.ndarray:
    Xa = np.column_stack([(X - mu) / sd, np.ones(len(X))])
    return _sigmoid(Xa @ w)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(
    X: torch.Tensor, stored: torch.Tensor, probe_indices: list[int],
) -> dict:
    Xn  = X.numpy()
    nrm = np.linalg.norm(Xn, axis=0, keepdims=True)
    Xnn = Xn / (nrm + 1e-8)
    cos_mat = Xnn.T @ Xnn       # (N, N)
    np.fill_diagonal(cos_mat, -1.0)
    mnc, mi, ist = [], [], []
    for true_idx in probe_indices:
        pat = stored[true_idx].numpy()
        mnc.append(float(cos_mat[true_idx].max()))
        mi.append(float(pat.mean()))
        ist.append(float(pat.std()))
    return {"max_neighbor_cosine": mnc, "mean_intensity": mi, "intensity_std": ist}


# ─────────────────────────────────────────────────────────────────────────────
# Stage A: Fashion-MNIST cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def run_stage_a(images: torch.Tensor, labels: torch.Tensor, exp_dir: Path) -> dict:
    print("\n--- Stage A: Fashion-MNIST cross-validation ---")

    # Load MNIST probe features and train transfer predictor
    mnist_feat_rows = read_csv_rows(exp_dir / "phase3_diag_d_probe_features.csv")
    X_mn = np.array([[float(r["max_neighbor_cosine"]),
                      float(r["mean_intensity"]),
                      float(r["intensity_std"])] for r in mnist_feat_rows], dtype=float)
    y_mn = np.array([int(r["vulnerable"]) for r in mnist_feat_rows], dtype=float)
    print(f"  MNIST predictor: {len(X_mn)} probes, {int(y_mn.sum())} vulnerable")
    mnist_w, mnist_mu, mnist_sd = logistic_fit_full(X_mn, y_mn)

    attacker   = WhiteBoxOnePixelAttacker()
    probe_rows: list[dict] = []
    per_seed:   list[dict] = []

    for seed in SEEDS:
        print(f"  Seed {seed} ...")
        X, stored, hop, probe_indices = build_cell(images, labels, seed)
        feats = compute_features(X, stored, probe_indices)
        feat_arr = np.array(list(zip(feats["max_neighbor_cosine"],
                                     feats["mean_intensity"],
                                     feats["intensity_std"])), dtype=float)
        mnist_scores = logistic_predict_proba(feat_arr, mnist_w, mnist_mu, mnist_sd)

        clean_ok_list: list[bool] = []
        wb_ok_list:    list[int]  = []
        rne_fail_list: list[int]  = []

        for j, true_idx in enumerate(probe_indices):
            q = stored[true_idx]

            ret_clean = hop.retrieve(q, steps=1)
            is_clean  = retrieval_accuracy(ret_clean, X, true_idx)
            clean_ok_list.append(is_clean)

            res = attacker.attack(q, true_idx, hop)
            wb_ok_list.append(int(res["success"]))

            noisy    = add_gaussian_noise(q, sigma=MATCHED_SIGMA, seed=seed + j)
            ret_noisy = hop.retrieve(noisy, steps=1)
            rne_fail_list.append(int(not retrieval_accuracy(ret_noisy, X, true_idx)))

            probe_rows.append({
                "seed":                  seed,
                "probe_idx":             j,
                "true_index":            true_idx,
                "digit_class":           true_idx // 10,
                "clean_ok":              int(is_clean),
                "wb_attack_success":     int(res["success"]),
                "rne_fail":              rne_fail_list[-1],
                "max_neighbor_cosine":   round(feats["max_neighbor_cosine"][j], 5),
                "mean_intensity":        round(feats["mean_intensity"][j], 5),
                "intensity_std":         round(feats["intensity_std"][j], 5),
                "mnist_predictor_score": round(float(mnist_scores[j]), 5),
            })

        n_total = len(probe_indices)
        n_clean = sum(clean_ok_list)
        n_wb    = sum(wb_ok_list)
        cond_wb  = sum(int(wb_ok_list[j] == 1 and clean_ok_list[j]) for j in range(n_total))
        cond_rne = sum(int(rne_fail_list[j] == 1 and clean_ok_list[j]) for j in range(n_total))

        per_seed.append({
            "seed":                  seed,
            "n_total":               n_total,
            "n_clean_ok":            n_clean,
            "baseline_failure_rate": round(1 - n_clean / n_total, 4),
            "raw_wb_success":        round(n_wb / n_total, 4),
            "cond_wb_success":       round(cond_wb / max(n_clean, 1), 4),
            "raw_rne_fail":          round(sum(rne_fail_list) / n_total, 4),
            "cond_rne_fail":         round(cond_rne / max(n_clean, 1), 4),
        })

    # Aggregate per-seed means
    raw_wb_mean   = float(np.mean([r["raw_wb_success"]        for r in per_seed]))
    cond_wb_mean  = float(np.mean([r["cond_wb_success"]       for r in per_seed]))
    bl_fail_mean  = float(np.mean([r["baseline_failure_rate"] for r in per_seed]))
    rne_cond_mean = float(np.mean([r["cond_rne_fail"]         for r in per_seed]))

    # Vulnerable = conditionally vulnerable (clean AND attacked)
    v_rows  = [r for r in probe_rows if r["wb_attack_success"] == 1 and r["clean_ok"] == 1]
    nv_rows = [r for r in probe_rows if not (r["wb_attack_success"] == 1 and r["clean_ok"] == 1)
               and r["clean_ok"] == 1]
    n_vuln = len(v_rows)

    # Chi-square: class distribution of vulnerable probes
    cls_counts = np.zeros(10, dtype=float)
    for r in v_rows:
        cls_counts[r["digit_class"]] += 1
    exp_cls = n_vuln / 10.0 if n_vuln > 0 else 1.0
    chi2_s  = float(np.sum((cls_counts - exp_cls) ** 2 / max(exp_cls, 1e-9))) if n_vuln > 0 else 0.0
    chi2_p  = chi2_pvalue(chi2_s, df=9) if n_vuln > 0 else 1.0
    over_cls  = [i for i in range(10) if cls_counts[i] > 1.5 * exp_cls]
    under_cls = [i for i in range(10) if cls_counts[i] < 0.5 * exp_cls and exp_cls > 0]

    # Mann-Whitney U on 3 features
    def _mwu_feature(key: str) -> tuple[float, float, np.ndarray, np.ndarray]:
        vf = np.array([r[key] for r in v_rows])
        nf = np.array([r[key] for r in nv_rows])
        U, p = mannwhitney_u(vf, nf)
        return U, p, vf, nf

    u_cos, p_cos, vc, nc = _mwu_feature("max_neighbor_cosine")
    u_mi,  p_mi,  vm, nm = _mwu_feature("mean_intensity")
    u_sd,  p_sd,  vs, ns = _mwu_feature("intensity_std")

    # Native FMNIST 5-fold CV
    X_lr = np.array([[r["max_neighbor_cosine"], r["mean_intensity"], r["intensity_std"]]
                     for r in probe_rows], dtype=float)
    y_lr = np.array([int(r["wb_attack_success"] == 1 and r["clean_ok"] == 1)
                     for r in probe_rows], dtype=float)
    native_lr_mean, native_lr_std = logistic_cv(X_lr, y_lr)

    # MNIST transfer balanced accuracy on FMNIST
    all_scores = np.array([r["mnist_predictor_score"] for r in probe_rows])
    all_preds  = (all_scores >= 0.5).astype(int)
    pos_mask   = y_lr == 1;  neg_mask = y_lr == 0
    tpr_t = float((all_preds[pos_mask] == 1).mean()) if pos_mask.any() else 0.0
    tnr_t = float((all_preds[neg_mask] == 0).mean()) if neg_mask.any() else 0.0
    transfer_bal_acc = (tpr_t + tnr_t) / 2.0

    # Save CSVs
    probe_fields = ["seed", "probe_idx", "true_index", "digit_class",
                    "clean_ok", "wb_attack_success", "rne_fail",
                    "max_neighbor_cosine", "mean_intensity", "intensity_std",
                    "mnist_predictor_score"]
    write_csv(probe_rows, exp_dir / "phase3_fashion_mnist_results.csv", probe_fields)
    print(f"  Saved phase3_fashion_mnist_results.csv  ({len(probe_rows)} rows)")

    sum_rows = list(per_seed) + [{
        "seed": "mean_5seeds", "n_total": N_PROBE, "n_clean_ok": "",
        "baseline_failure_rate": round(bl_fail_mean, 4),
        "raw_wb_success":        round(raw_wb_mean, 4),
        "cond_wb_success":       round(cond_wb_mean, 4),
        "raw_rne_fail":          "",
        "cond_rne_fail":         round(rne_cond_mean, 4),
    }]
    sum_fields = ["seed", "n_total", "n_clean_ok", "baseline_failure_rate",
                  "raw_wb_success", "cond_wb_success", "raw_rne_fail", "cond_rne_fail"]
    write_csv(sum_rows, exp_dir / "phase3_fashion_mnist_summary.csv", sum_fields)
    print(f"  Saved phase3_fashion_mnist_summary.csv")

    return {
        "probe_rows": probe_rows, "per_seed": per_seed,
        "n_vuln": n_vuln, "v_rows": v_rows, "nv_rows": nv_rows,
        "cls_counts": cls_counts, "exp_cls": exp_cls,
        "chi2_s": chi2_s, "chi2_p": chi2_p,
        "over_cls": over_cls, "under_cls": under_cls,
        "vc": vc, "nc": nc, "u_cos": u_cos, "p_cos": p_cos,
        "vm": vm, "nm": nm, "u_mi":  u_mi,  "p_mi":  p_mi,
        "vs": vs, "ns": ns, "u_sd":  u_sd,  "p_sd":  p_sd,
        "native_lr_mean": native_lr_mean, "native_lr_std": native_lr_std,
        "transfer_bal_acc": transfer_bal_acc,
        "raw_wb_mean":  raw_wb_mean,  "cond_wb_mean": cond_wb_mean,
        "bl_fail_mean": bl_fail_mean, "rne_cond_mean": rne_cond_mean,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stage B: Pixel-value distribution from existing CSV
# ─────────────────────────────────────────────────────────────────────────────

def run_stage_b(exp_dir: Path) -> dict:
    print("\n--- Stage B: Pixel-value distribution analysis ---")
    wb_rows = read_csv_rows(exp_dir / "phase3_whitebox_results.csv")
    succ    = [r for r in wb_rows if int(r["attack_success"]) == 1]
    print(f"  Total rows: {len(wb_rows)}, successful attacks: {len(succ)}")

    # Build 5×5 transition matrix (rows=orig bin, cols=adv val)
    trans_mat = np.zeros((5, 5), dtype=int)
    for r in succ:
        oi = CAND_VALS.index(bin_to_cand(float(r["original_value"])))
        ai = CAND_VALS.index(bin_to_cand(float(r["pixel_value"])))
        trans_mat[oi, ai] += 1

    REGIMES = [
        ("N10_random",           lambda r: int(r["N"]) == 10   and r["strategy"] == "random"),
        ("N100_class_balanced",  lambda r: int(r["N"]) == 100  and r["strategy"] == "class_balanced"),
        ("N1000_class_balanced", lambda r: int(r["N"]) == 1000 and r["strategy"] == "class_balanced"),
        ("all",                  lambda r: True),
    ]

    pv_rows: list[dict] = []
    for regime_name, filt in REGIMES:
        sub     = [r for r in succ if filt(r)]
        n_regime = len(sub)
        for cv in CAND_VALS:
            cnt = sum(1 for r in sub if abs(float(r["pixel_value"]) - cv) < 0.01)
            pv_rows.append({
                "regime":            regime_name,
                "original_value_bin": "any",
                "adversarial_value": cv,
                "count":             cnt,
                "proportion":        round(cnt / max(n_regime, 1), 4),
                "n_regime":          n_regime,
            })

    # Full transition rows for regime="all"
    for oi, orig_cv in enumerate(CAND_VALS):
        n_in_bin = int(trans_mat[oi].sum())
        for ai, adv_cv in enumerate(CAND_VALS):
            cnt = int(trans_mat[oi, ai])
            pv_rows.append({
                "regime":             "all",
                "original_value_bin":  orig_cv,
                "adversarial_value":   adv_cv,
                "count":               cnt,
                "proportion":          round(cnt / max(n_in_bin, 1), 4),
                "n_regime":            n_in_bin,
            })

    pv_fields = ["regime", "original_value_bin", "adversarial_value",
                 "count", "proportion", "n_regime"]
    write_csv(pv_rows, exp_dir / "phase3_pixel_value_analysis.csv", pv_fields)
    print(f"  Saved phase3_pixel_value_analysis.csv  ({len(pv_rows)} rows)")

    return {"succ": succ, "pv_rows": pv_rows, "trans_mat": trans_mat}


# ─────────────────────────────────────────────────────────────────────────────
# Figure A: 2×2 Fashion-MNIST comparison
# ─────────────────────────────────────────────────────────────────────────────

def save_figure_fashion_mnist(a: dict, fig_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # [0,0] MNIST vs FMNIST rates
    ax = axes[0, 0]
    cats   = ["Raw WB", "Cond. WB", "Cond. RNE"]
    m_vals = [MNIST_RAW_WB_SUCCESS, MNIST_COND_WB_SUCCESS, MNIST_COND_RNE_FAIL]
    f_vals = [a["raw_wb_mean"], a["cond_wb_mean"], a["rne_cond_mean"]]
    x = np.arange(3);  bw = 0.35
    ax.bar(x - bw/2, m_vals, bw, label="MNIST",    color="steelblue",  alpha=0.8)
    ax.bar(x + bw/2, f_vals, bw, label="F-MNIST",  color="darkorange", alpha=0.8)
    ax.set_xticks(x);  ax.set_xticklabels(cats, fontsize=9)
    ax.set_ylabel("Rate", fontsize=9)
    ax.set_title("Attack success rates: MNIST vs Fashion-MNIST\n"
                 "(N=100 class-balanced, 5-seed mean)", fontsize=9)
    ax.legend(fontsize=9);  ax.grid(True, axis="y", alpha=0.25)

    # [0,1] FMNIST vulnerable class distribution
    ax = axes[0, 1]
    x2   = np.arange(10)
    bars = ax.bar(x2, a["cls_counts"], color="steelblue", alpha=0.8, edgecolor="white")
    ax.axhline(a["exp_cls"], color="crimson", linestyle="--", lw=1.8,
               label=f"Expected {a['exp_cls']:.1f}")
    for i in a["over_cls"]:
        bars[i].set_color("darkorange")
    for i in a["under_cls"]:
        bars[i].set_color("lightgray");  bars[i].set_edgecolor("gray")
    ax.set_xticks(x2);  ax.set_xticklabels([str(i) for i in range(10)], fontsize=9)
    ax.set_xlabel("F-MNIST class", fontsize=9);  ax.set_ylabel("Count", fontsize=9)
    ps = f"{a['chi2_p']:.3f}" if a["chi2_p"] >= 0.001 else "<0.001"
    ax.set_title(f"F-MNIST vulnerable class distribution (n={a['n_vuln']})\n"
                 f"chi-sq={a['chi2_s']:.2f}, p={ps}", fontsize=9)
    ax.legend(fontsize=8);  ax.grid(True, axis="y", alpha=0.25)

    # [1,0] Max neighbor cosine boxplot
    ax = axes[1, 0]
    if len(a["vc"]) > 0:
        bp = ax.boxplot([a["vc"], a["nc"]], patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", lw=2))
        bp["boxes"][0].set_facecolor("#d6604d");  bp["boxes"][0].set_alpha(0.75)
        bp["boxes"][1].set_facecolor("#4393c3");  bp["boxes"][1].set_alpha(0.75)
        xj = np.random.RandomState(1).uniform(0.75, 1.25, len(a["vc"]))
        ax.scatter(xj, a["vc"], color="#d6604d", alpha=0.5, s=15, zorder=3)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f"Vulnerable\n(n={len(a['vc'])})",
                             f"Non-vuln.\n(n={len(a['nc'])})"], fontsize=9)
    else:
        ax.text(0.5, 0.5, "No vulnerable probes", ha="center", va="center",
                transform=ax.transAxes, fontsize=11)
    ax.set_ylabel("Max neighbor cosine", fontsize=9)
    cp = f"p={a['p_cos']:.3f}" if a["p_cos"] >= 0.001 else "p<0.001"
    ax.set_title(f"F-MNIST: Max neighbor cosine\n(MWU {cp})", fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)

    # [1,1] Transfer vs native CV
    ax = axes[1, 1]
    lr_labels = ["MNIST\ntransfer", "FMNIST\nnative CV", "Chance\n(50%)"]
    lr_vals   = [a["transfer_bal_acc"], a["native_lr_mean"], 0.5]
    lr_colors = ["steelblue", "darkorange", "lightgray"]
    ax.bar(range(3), lr_vals, color=lr_colors, alpha=0.8, edgecolor="white")
    ax.errorbar([1], [a["native_lr_mean"]], yerr=[a["native_lr_std"]],
                fmt="none", color="black", capsize=5, lw=2)
    ax.set_xticks(range(3));  ax.set_xticklabels(lr_labels, fontsize=9)
    ax.set_ylabel("Balanced accuracy", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", lw=1.2)
    ax.set_title("Vulnerability prediction: MNIST→FMNIST transfer\n"
                 "vs native FMNIST 5-fold CV", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Fashion-MNIST one-pixel attack cross-validation\n"
                 "N=100 class-balanced, 5 seeds × 50 probes", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure B: 1×2 pixel-value distribution
# ─────────────────────────────────────────────────────────────────────────────

def save_figure_pixel_values(b: dict, fig_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: overall adversarial pixel value distribution
    ax = axes[0]
    succ = b["succ"]
    pv_cnt = {cv: sum(1 for r in succ if abs(float(r["pixel_value"]) - cv) < 0.01)
              for cv in CAND_VALS}
    total  = max(sum(pv_cnt.values()), 1)
    ax.bar([str(cv) for cv in CAND_VALS],
           [pv_cnt[cv] / total for cv in CAND_VALS],
           color="steelblue", alpha=0.8, edgecolor="white")
    ax.set_xlabel("Adversarial pixel value", fontsize=10)
    ax.set_ylabel("Proportion of successful attacks", fontsize=10)
    ax.set_title(f"Adversarial pixel value distribution\n"
                 f"({total} total successful attacks, all N/strategy)", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    # Right: 5×5 transition matrix heatmap
    ax = axes[1]
    mat      = b["trans_mat"].astype(float)
    row_sums = mat.sum(axis=1, keepdims=True)
    mat_norm = mat / np.where(row_sums == 0, 1, row_sums)
    im = ax.imshow(mat_norm, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Proportion within original bin")
    ax.set_xticks(range(5));  ax.set_xticklabels([str(cv) for cv in CAND_VALS], fontsize=9)
    ax.set_yticks(range(5));  ax.set_yticklabels([str(cv) for cv in CAND_VALS], fontsize=9)
    ax.set_xlabel("Adversarial pixel value", fontsize=10)
    ax.set_ylabel("Original pixel bin", fontsize=10)
    ax.set_title("Transition matrix: original bin → adversarial value\n"
                 "(row-normalized, all successful attacks)", fontsize=9)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{int(mat[i,j])}", ha="center", va="center",
                    fontsize=8, color="white" if mat_norm[i, j] > 0.6 else "black")

    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Print summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(a: dict, b: dict) -> None:
    print("\n" + "=" * 62)
    print("STAGE A: Fashion-MNIST results")
    print("=" * 62)

    print(f"\n{'Seed':>6}  {'BL fail':>8}  {'Raw WB':>7}  {'Cond WB':>8}  {'Cond RNE':>9}")
    print("-" * 46)
    for r in a["per_seed"]:
        print(f"{r['seed']:>6}  {r['baseline_failure_rate']:>8.4f}  "
              f"{r['raw_wb_success']:>7.4f}  {r['cond_wb_success']:>8.4f}  "
              f"{r['cond_rne_fail']:>9.4f}")
    print("-" * 46)
    print(f"{'Mean':>6}  {a['bl_fail_mean']:>8.4f}  "
          f"{a['raw_wb_mean']:>7.4f}  {a['cond_wb_mean']:>8.4f}  "
          f"{a['rne_cond_mean']:>9.4f}")

    print(f"\nMNIST reference: BL={0.092:.4f}  RawWB={MNIST_RAW_WB_SUCCESS:.4f}  "
          f"CondWB={MNIST_COND_WB_SUCCESS:.4f}  CondRNE={MNIST_COND_RNE_FAIL:.4f}")

    amplif_f  = a["cond_wb_mean"] / max(a["rne_cond_mean"], 1e-6)
    amplif_mn = MNIST_COND_WB_SUCCESS / max(MNIST_COND_RNE_FAIL, 1e-6)
    print(f"\nAmplification: FMNIST={amplif_f:.2f}x  MNIST={amplif_mn:.2f}x")

    replicate = abs(a["cond_wb_mean"] - MNIST_COND_WB_SUCCESS) < 0.05
    direction = ("replicates" if replicate
                 else "exceeds" if a["cond_wb_mean"] > MNIST_COND_WB_SUCCESS
                 else "underperforms")
    print(f"Verdict: FMNIST {direction} the MNIST headline finding.")

    print(f"\nVulnerable probes: {a['n_vuln']} / {len(a['probe_rows'])} total FMNIST probes")

    ps = f"{a['chi2_p']:.3f}" if a["chi2_p"] >= 0.001 else "<0.001"
    print(f"\nChi-square class bias: chi2={a['chi2_s']:.3f}, p={ps}")
    if a["chi2_p"] < 0.05:
        print(f"  Verdict: significantly non-uniform class distribution.")
        print(f"  Over-represented:  {', '.join(str(i) for i in a['over_cls']) or 'none'}")
        print(f"  Under-represented: {', '.join(str(i) for i in a['under_cls']) or 'none'}")
    else:
        print("  Verdict: no significant class bias in FMNIST vulnerable probes.")

    print("\nMann-Whitney U (FMNIST):")
    for label, (U, p, vmed, nmed) in [
        ("Max neighbor cosine", (a["u_cos"], a["p_cos"],
                                 np.median(a["vc"]) if len(a["vc"]) > 0 else float("nan"),
                                 np.median(a["nc"]) if len(a["nc"]) > 0 else float("nan"))),
        ("Mean intensity",      (a["u_mi"],  a["p_mi"],
                                 np.median(a["vm"]) if len(a["vm"]) > 0 else float("nan"),
                                 np.median(a["nm"]) if len(a["nm"]) > 0 else float("nan"))),
        ("Intensity std",       (a["u_sd"],  a["p_sd"],
                                 np.median(a["vs"]) if len(a["vs"]) > 0 else float("nan"),
                                 np.median(a["ns"]) if len(a["ns"]) > 0 else float("nan"))),
    ]:
        pf  = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
        sig = " *" if p < 0.05 else ""
        print(f"  {label}: vuln={vmed:.4f}, non-vuln={nmed:.4f}, U={U:.1f}, p={pf}{sig}")

    print(f"\nVulnerability prediction (balanced accuracy, 50% baseline):")
    print(f"  Native FMNIST 5-fold CV: {a['native_lr_mean']:.3f} +/- {a['native_lr_std']:.3f}")
    print(f"  MNIST -> FMNIST transfer: {a['transfer_bal_acc']:.3f}")
    if a["transfer_bal_acc"] > 0.70:
        tr_v = "Transfer succeeds: MNIST vulnerability geometry predicts FMNIST."
    elif a["transfer_bal_acc"] > 0.55:
        tr_v = "Partial transfer: features partially generalise across domains."
    else:
        tr_v = "Transfer fails: MNIST geometry does not predict FMNIST vulnerability."
    print(f"  Verdict: {tr_v}")

    # ── Stage B ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("STAGE B: Pixel-value distribution")
    print("=" * 62)

    succ = b["succ"]
    pv_cnt = {cv: sum(1 for r in succ if abs(float(r["pixel_value"]) - cv) < 0.01)
              for cv in CAND_VALS}
    total_s = max(sum(pv_cnt.values()), 1)
    print(f"\nOverall adversarial pixel value distribution ({total_s} successful attacks):")
    print(f"  {'Value':>6}  {'Count':>8}  {'Proportion':>12}")
    print("-" * 32)
    for cv in CAND_VALS:
        print(f"  {cv:>6.2f}  {pv_cnt[cv]:>8}  {pv_cnt[cv]/total_s:>12.4f}")
    extreme = pv_cnt[0.0] + pv_cnt[1.0]
    most_cv = max(CAND_VALS, key=lambda cv: pv_cnt[cv])
    print(f"\n  Extreme values (0.0 or 1.0): {extreme}/{total_s} = {extreme/total_s:.4f}")
    print(f"  Most common adversarial value: {most_cv:.2f}")

    print("\nTransition matrix (counts, rows=original bin, cols=adversarial value):")
    col_hdr = "orig/adv"
    header = f"  {col_hdr:>8}  " + "  ".join(f"{cv:>6.2f}" for cv in CAND_VALS)
    print(header)
    print("-" * (len(header) + 4))
    for i, ro in enumerate(CAND_VALS):
        row_str = "  ".join(f"{b['trans_mat'][i, j]:>6}" for j in range(5))
        print(f"  {ro:>8.2f}  {row_str}")

    REGIME_NAMES = ["N10_random", "N100_class_balanced", "N1000_class_balanced"]
    regime_dist = {name: [r for r in b["pv_rows"]
                           if r["regime"] == name and r["original_value_bin"] == "any"]
                   for name in REGIME_NAMES}
    print("\nPer-regime adversarial pixel value breakdown:")
    for name in REGIME_NAMES:
        sub = regime_dist[name]
        if not sub or int(sub[0]["n_regime"]) == 0:
            print(f"  {name}: no successful attacks")
            continue
        n_reg = int(sub[0]["n_regime"])
        print(f"\n  {name}  ({n_reg} successful attacks):")
        for r in sub:
            print(f"    pv={float(r['adversarial_value']):.2f}: "
                  f"{r['count']:>4}  ({float(r['proportion'])*100:.1f}%)")

    # Overall closing paragraph
    class_bias_str = "class-biased" if a["chi2_p"] < 0.05 else "uniformly distributed"
    transfer_str   = ("transfers successfully" if a["transfer_bal_acc"] > 0.70 else
                      "partially transfers"   if a["transfer_bal_acc"] > 0.55 else
                      "does not transfer")
    extreme_str = ("extreme pixel values (0.0 or 1.0)" if extreme / total_s > 0.5
                   else "intermediate pixel values (0.25, 0.5, 0.75)")

    print("\n" + "=" * 62)
    print("OVERALL CLOSING")
    print("=" * 62)
    print(
        f"\nFashion-MNIST one-pixel attacks {direction} the MNIST headline result "
        f"(FMNIST cond. WB {a['cond_wb_mean']*100:.1f}% vs MNIST {MNIST_COND_WB_SUCCESS*100:.1f}%). "
        f"Amplification factor: FMNIST {amplif_f:.1f}x vs MNIST {amplif_mn:.1f}x. "
        f"Vulnerable FMNIST probes are {class_bias_str} across classes. "
        f"MNIST-trained vulnerability predictor {transfer_str} to FMNIST "
        f"(transfer bal.acc.={a['transfer_bal_acc']:.3f}, "
        f"native bal.acc.={a['native_lr_mean']:.3f}+/-{a['native_lr_std']:.3f}). "
        f"Pixel-value analysis ({total_s} successful MNIST attacks) shows attacks "
        f"predominantly target {extreme_str} "
        f"({extreme/total_s*100:.1f}% extreme, most common={most_cv:.2f})."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    print("=" * 62)
    print("Phase 3 Final Diagnostics")
    print("=" * 62)

    print("\nLoading Fashion-MNIST ...")
    images, labels = load_fmnist_train()
    print(f"  ok {len(images)} samples")

    a = run_stage_a(images, labels, EXP_DIR)
    b = run_stage_b(EXP_DIR)

    print("\nGenerating figures ...")
    save_figure_fashion_mnist(a, str(FIG_DIR / "phase3_fashion_mnist_comparison.png"))
    print("  Saved: phase3_fashion_mnist_comparison.png")
    save_figure_pixel_values(b, str(FIG_DIR / "phase3_pixel_value_distribution.png"))
    print("  Saved: phase3_pixel_value_distribution.png")

    print_summary(a, b)
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
