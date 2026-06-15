# Thesis Data Snapshot
**Generated:** 2026-06-07  
**Status:** READ-ONLY inspection — no experiments run, no code modified  
**Purpose:** Authoritative record of all numerical results, configurations, and code for thesis writing

---

## Section 1: Project File Tree

### Source Module (`hopfield/`)
```
hopfield/__init__.py          — exports ContinuousHopfield, WhiteBoxOnePixelAttacker, DEBlackBoxOnePixelAttacker
hopfield/network.py           — ContinuousHopfield (retrieve, energy)
hopfield/attacks.py           — WhiteBoxOnePixelAttacker, DEBlackBoxOnePixelAttacker, OnePixelAttacker base
hopfield/metrics.py           — retrieval_accuracy
hopfield/sampling.py          — sample_class_balanced
hopfield/corruption.py        — mask_bottom_half, add_gaussian_noise, flip_pixels, mask_random_patch
hopfield/vulnerability.py     — compute_vulnerability_map
```

### Experiment Scripts (`experiments/`)
```
phase2_capacity_stability.py               — N-sweep capacity and noise stability
phase3_closing_diagnostics.py             — WB attacks + all 4 diagnostic sub-analyses
mnist_two_stage_attack.py                 — MNIST two-stage attack (cosine-delta + first-order fallback)
fmnist_two_stage_attack.py                — Fashion-MNIST two-stage attack
grayscale_cifar_two_stage_centered.py     — Centered CIFAR two-stage attack (N=100 and N=500)
grayscale_cifar_attack.py                 — Raw grayscale CIFAR attack (null result demo)
grayscale_cifar_baseline.py               — Raw CIFAR capacity sweep by N
attack_visualization.py                   — All thesis figures (6 figures, Figs 3-8)
export_for_excel.py                       — Excel export for table generation
```

### Results CSVs
```
phase2_results.csv                  110 rows    (N x noise_type x magnitude x strategy, 1 seed)
phase2_stability_results.csv        150 rows    (5 seeds x N x noise_type x magnitude x strategy)
phase2_stability_summary.csv         30 rows    aggregated
phase3_whitebox_results.csv        2100 rows    (5 seeds x 5 N values x 2 strategies x ~42 probes)
phase3_blackbox_results.csv         250 rows    (N=100 cb only, 5 seeds x 50 probes)
phase3_summary.csv                   11 rows    per (N, strategy, attacker)
phase3_diag_a_paired.csv            250 rows    WB vs DE paired comparison
phase3_diag_b_baseline_corrected.csv  50 rows  per (seed, N, strategy) baseline-conditional breakdown
phase3_diag_c_matched_sigma.csv      45 rows    sigma sweep at N=100 cb (9 sigma x 5 seeds)
phase3_diag_d_probe_features.csv    250 rows    per-probe features for vulnerability analysis
phase3_diag_d_summary.csv             5 rows    statistical tests
phase3_random_noise_equivalent.csv  200 rows    (4 sigma x 5 seeds x 2 strategies x 5 N)
mnist_two_stage_results.csv         500 rows    (5 seeds x 100 probes)
mnist_two_stage_summary.csv           5 rows
fmnist_two_stage_results.csv        500 rows    (5 seeds x 100 probes)
fmnist_two_stage_summary.csv          5 rows
cifar_centered_two_stage_results.csv 500 rows   (5 seeds x 50 probes x 2 N values interleaved)
cifar_centered_two_stage_summary.csv  10 rows   (5 rows per N=100, 5 rows per N=500)
grayscale_cifar_attack_results.csv   50 rows    (5 seeds x 10 probes, N=10)
grayscale_cifar_attack_summary.csv    5 rows
grayscale_cifar_baseline_results.csv 25 rows    (5 seeds x 5 N values: 10,20,30,50,100)
```

### Figures (`figures/`)
```
attack_mnist_examples.png       — Fig 3: 5 MNIST attack cases (true / perturbed / retrieved)
fmnist_attack_grid.png          — Fig 4: FMNIST attack grid
mnist_attack_grid.png           — Fig 5: MNIST attack grid  
attack_cross_dataset.png        — Fig 6: cross-dataset pairwise cosine bar chart
grayscale_cifar_retrieval_demo.png  — Fig 7: centered CIFAR retrieval demo
grayscale_cifar_stored_patterns.png — Fig 8: stored pattern grid
grayscale_cifar_attack_demo.png     — additional: raw CIFAR failure demo
grayscale_cifar_vs_mnist.png        — additional: cosine comparison
```

---

## Section 2: Network Implementation

**File:** `hopfield/network.py`

```python
class ContinuousHopfield:
    def __init__(self, X: torch.Tensor, beta: float = 8.0) -> None:
        self.X = X.float()
        self.beta = float(beta)

    def retrieve(self, query: torch.Tensor, steps: int = 1) -> torch.Tensor:
        squeeze = query.ndim == 1
        xi = query.float()
        if squeeze:
            xi = xi.unsqueeze(1)          # (d, 1)
        X = self.X.to(xi.device)
        for _ in range(steps):
            logits = self.beta * (X.T @ xi)      # (N, B)
            weights = F.softmax(logits, dim=0)   # (N, B)
            xi = X @ weights                     # (d, B)
        if squeeze:
            xi = xi.squeeze(1)
        return xi

    def energy(self, xi: torch.Tensor) -> torch.Tensor:
        # E = -lse(beta, X^T xi) + 0.5 ||xi||^2 + beta^{-1} log N + 0.5 M^2
        lse = torch.logsumexp(self.beta * (self.X.T @ xi), dim=0) / self.beta
        return -lse + 0.5 * (xi ** 2).sum(dim=0)
```

**Update rule:** `xi_new = X @ softmax(beta * X^T @ xi)`  
**Energy:** `E(xi) = -lse(beta, X^T xi) + 0.5||xi||^2 + beta^{-1} log N + 0.5 M^2`  
**Precision:** float32 throughout  
**Steps:** always 1 (single-step retrieval) in all experiments  
**Batching:** queries can be shape `(d,)` or `(d, B)`  
**Default beta:** 8.0  

---

## Section 3: Attacker Implementations

### 3.1 White-Box One-Pixel Attacker

**File:** `hopfield/attacks.py` — `WhiteBoxOnePixelAttacker`

```
Candidate values:  {0.0, 0.25, 0.5, 0.75, 1.0}
Candidate space:   784 pixels x 5 values = 3920 candidates per query
Selection:         argmin cosine(retrieved_k, true_pattern) over all 3920 candidates
Implementation:    single batched retrieve() call on (784, 3920) query matrix
Success criterion: retrieved_index != true_index (after argmin selection)
```

**Note:** This simple argmin attacker has a degenerate-argmin bug: when the network is already failing (baseline-correct=0), all 3920 candidates retrieve the same wrong pattern, and argmin picks candidate k=0 (pixel 0, value 0.0), giving L2≈0. The corrected version is in the two-stage scripts (see Section 3.2).

### 3.2 Two-Stage White-Box Attacker (Corrected)

**Files:** `mnist_two_stage_attack.py`, `fmnist_two_stage_attack.py`, `grayscale_cifar_two_stage_centered.py`

Selection logic (cosine-delta + first-order sensitivity fallback):
```python
cos_sims = F.cosine_similarity(retrieved.T, true_pat.unsqueeze(0), dim=1)  # (3920,)
delta = cos_sims - clean_cos  # relative to clean retrieval cosine
if delta.min() < -1e-4:
    worst_k = int(delta.argmin().item())   # genuine cosine damage found
else:
    # first-order sensitivity fallback
    # maximise: true_at_locs * (q[locs] - vals) to push xi away from true pattern
    sensitivity = (true_at_locs * (q[locs] - vals)).abs()
    worst_k = int(sensitivity.argmax().item())
```

This corrected attacker picks high-L2 perturbations even when baseline is failing.

### 3.3 Differential Evolution Black-Box Attacker

**File:** `hopfield/attacks.py` — `DEBlackBoxOnePixelAttacker`

```
Population size:    400
Max generations:    100
F (mutation scale): 0.5
CR (crossover):     0.7
Encoding:           continuous (i, j, v) in [0,1]^3, rounded to pixel coords + nearest candidate value
Fitness:            1 - cosine(retrieved, true_pattern)
Early stop:         if any success AND max(fitness[successes]) > 0.5
Evaluations/probe:  ~40,080 mean (at N=100 cb)  [from phase3_blackbox_results.csv]
```

---

## Section 4: Preprocessing / Centering

### 4.1 Raw experiments (MNIST, FMNIST, phase3)

Patterns stored directly as normalized float32 tensors. No centering.  
`sample_class_balanced(dataset, N)` returns `(784, N)` with 10 classes, `N//10` each.  
Patterns NOT normalized to unit length in sampling; the network operates on raw pixel values in [0,1].

### 4.2 Centered CIFAR experiments

**File:** `grayscale_cifar_two_stage_centered.py`

```python
def center_and_normalise(X: torch.Tensor):
    # X shape: (d, N), d=784 (28x28 grayscale), N stored patterns
    mu  = X.mean(dim=1, keepdim=True)      # (d, 1): per-pixel mean over all N patterns
    Xc  = X - mu                           # subtract global mean
    nrm = Xc.norm(dim=0, keepdim=True).clamp(min=1e-8)  # per-pattern L2 norm
    return Xc / nrm, mu                    # unit-norm centered patterns, mean

def proc_query(q: torch.Tensor, mu: torch.Tensor):
    # Apply same centering to query before retrieval
    qc = q - mu.squeeze()                  # subtract stored-patterns mean
    return qc / qc.norm().clamp(min=1e-8)  # unit normalize
```

**Attacker interaction with centering:** The attacker modifies raw pixel values. The modified image is then centered and normalized via `proc_query()` before retrieval. Attack operates in raw pixel space; centering happens at query time.

---

## Section 5: Metrics

**File:** `hopfield/metrics.py`

```python
def retrieval_accuracy(retrieved: torch.Tensor, X: torch.Tensor, true_index: int) -> bool:
    r = retrieved.float()
    sims  = torch.mv(X.float().T, r)           # (N,): dot products
    norms = X.float().norm(dim=0) * r.norm()   # (N,): norm products
    cos   = sims / norms.clamp(min=1e-8)       # cosine similarities
    return int(cos.argmax().item()) == int(true_index)
```

**Conditional success definition:** An attack is a *conditional success* if `baseline_correct=1` (retrieval was correct before attack) AND `attack_success=1` (retrieval is now wrong after attack). Raw success includes baseline-failing probes.

---

## Section 6: Experiment Configurations

| Parameter | Value |
|---|---|
| Seeds (all experiments) | 42, 43, 44, 45, 46 |
| Default beta | 8.0 |
| Retrieval steps | 1 |
| Candidate pixel values | {0.0, 0.25, 0.5, 0.75, 1.0} |
| Phase 3 / DE probes per seed | 50 (N=100 cb) |
| Two-stage probes per seed | 100 (MNIST/FMNIST), 50 (CIFAR) |
| MNIST/FMNIST dimensions | 28x28 = 784 |
| CIFAR dimensions (grayscale) | 32x32 = 1024 (NOT resized; kept at native resolution) |
| DE pop_size | 400 |
| DE max_gens | 100 |
| DE F, CR | 0.5, 0.7 |
| Sigma sweep values | 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.015, 0.020 |
| Phase 2 noise magnitudes | 0.05, 0.10, 0.20, 0.30, 0.50 |
| Phase 2 noise types | gaussian, flip, occlusion |
| Phase 3 N values tested | 10, 50, 100, 500, 1000 |
| CIFAR baseline N values | 10, 20, 30, 50, 100 |

---

## Section 7: Headline Numerical Results

### 7.1 MNIST N=100 Class-Balanced (Two-Stage Attacker)

**Source:** `mnist_two_stage_results.csv`, `mnist_two_stage_summary.csv`  
**5 seeds × 100 probes = 500 total**

| Seed | n_bl_correct | bl_fail% | A_raw% | A_cond% | B_cond% | pairwise_cos |
|------|-------------|----------|--------|---------|---------|--------------|
| 42   | 86          | 14.0%    | 18.0%  | 4.65%   | 0.0%    | 0.4178       |
| 43   | 94          | 6.0%     | 8.0%   | 2.13%   | 0.0%    | 0.3881       |
| 44   | 91          | 9.0%     | 12.0%  | 3.30%   | 0.0%    | 0.3808       |
| 45   | 82          | 18.0%    | 21.0%  | 3.66%   | 0.0%    | 0.3982       |
| 46   | 88          | 12.0%    | 16.0%  | 4.55%   | 0.0%    | 0.3946       |
| **pooled** | **441/500** | **11.8%** | **15.0%** | **3.63%** | **0%** | **0.396±0.014** |
| **mean±std** | — | **11.8%±4.5%** | — | **3.66%±0.94%** | — | — |

- Conditional A success (16/441): attack flipped a baseline-correct retrieval  
- Conditional B success: 0 across all seeds (stricter criterion; B not reported in final thesis)

### 7.2 Fashion-MNIST N=100 Class-Balanced (Two-Stage Attacker)

**Source:** `fmnist_two_stage_results.csv`, `fmnist_two_stage_summary.csv`  
**5 seeds × 100 probes = 500 total**

| Seed | n_bl_correct | bl_fail% | A_cond% | B_cond% | pairwise_cos |
|------|-------------|----------|---------|---------|--------------|
| 42   | 18          | 82.0%    | 16.67%  | 5.56%   | 0.5885       |
| 43   | 11          | 89.0%    | 9.09%   | 0.0%    | 0.6068       |
| 44   | 22          | 78.0%    | 4.55%   | 0.0%    | 0.6019       |
| 45   | 20          | 80.0%    | 10.00%  | 0.0%    | 0.5934       |
| 46   | 21          | 79.0%    | 4.76%   | 0.0%    | 0.5869       |
| **pooled** | **92/500** | **81.6%** | **8.70%** | **1.09%** | **0.595±0.008** |
| **mean±std** | — | **81.6%±3.8%** | **9.01%±4.44%** | **1.11%** | — |

- The high baseline failure rate (81.6%) reflects severe pattern crowding at pairwise cosine ~0.60
- Only 92 probes had correct baseline retrieval; the 8.70% conditional rate is computed over those 92

### 7.3 Centered Grayscale CIFAR N=100 Class-Balanced

**Source:** `cifar_centered_two_stage_summary.csv` (N=100 rows)  
**5 seeds × 50 probes = 250 total**

| Seed | n_bl_correct | bl_fail% | A1_cond% | cos_raw | cos_proc |
|------|-------------|----------|----------|---------|----------|
| 42   | 50          | 0.0%     | 0.0%     | 0.8458  | -0.0088  |
| 43   | 50          | 0.0%     | 0.0%     | 0.8400  | -0.0091  |
| 44   | 50          | 0.0%     | 0.0%     | 0.8296  | -0.0092  |
| 45   | 50          | 0.0%     | 0.0%     | 0.8244  | -0.0085  |
| 46   | 50          | 0.0%     | 0.0%     | 0.8438  | -0.0093  |
| **pooled** | **250/250** | **0.0%** | **0.0%** | **0.837±0.008** | **-0.009** |

- Perfect baseline accuracy after centering: all 250 probes retrieved correctly
- Zero attack successes: centering defeats the one-pixel attacker completely
- `cos_proc` = pairwise cosine among *processed* (centered + normalized) stored patterns, near zero
- `cos_raw` = pairwise cosine among *raw* pixel patterns before processing, remains ~0.84

### 7.4 Centered Grayscale CIFAR N=500 Class-Balanced

**Source:** `cifar_centered_two_stage_summary.csv` (N=500 rows)  
**5 seeds × 50 probes = 250 total**

| Seed | n_bl_correct | bl_fail% | A1_cond%  | A2_cond% | cos_raw | cos_proc |
|------|-------------|----------|-----------|----------|---------|----------|
| 42   | 40          | 20.0%    | 0.0%      | 0.0%     | 0.8388  | -0.0014  |
| 43   | 38          | 24.0%    | 0.0%      | 0.0%     | 0.8344  | -0.0013  |
| 44   | 44          | 12.0%    | 0.0%      | 0.0%     | 0.8308  | -0.0018  |
| 45   | 37          | 26.0%    | **10.81%**| 5.41%    | 0.8331  | -0.0012  |
| 46   | 40          | 20.0%    | 0.0%      | 0.0%     | 0.8367  | -0.0016  |
| **pooled** | **199/250** | **20.4%** | **2.01%** | **1.01%** | **0.835±0.003** | **-0.001** |
| **mean±std** | — | **20.4%±5.0%** | **2.16%±4.84%** | — | — | — |

- At N=500, capacity degrades: 20.4% of probes fail baseline
- Only seed 45 achieved conditional attack success (4/37 = 10.81% for A1)
- A2 = alternate one-pixel attack variant (fewer successes: 2/37 for seed 45)
- `cos_proc` approaches zero even at N=500 (centering still orthogonalizes)

### 7.5 Raw Grayscale CIFAR N=10 (Null Result)

**Source:** `grayscale_cifar_attack_results.csv`, `grayscale_cifar_attack_summary.csv`  
**5 seeds × 10 probes = 50 total**

| Seed | n_bl_correct | bl_fail% | atk_success | pairwise_cos |
|------|-------------|----------|-------------|--------------|
| 42   | 1           | 90.0%    | 0           | 0.8843       |
| 43   | 2           | 80.0%    | 0           | 0.7921       |
| 44   | 5           | 50.0%    | 0           | 0.8681       |
| 45   | 1           | 90.0%    | 0           | 0.8632       |
| 46   | 1           | 90.0%    | 0           | 0.7529       |
| **pooled** | **10/50** | **80.0%** | **0/10 (0%)** | **0.832±0.052** |

Note: `grayscale_cifar_baseline_results.csv` at N=10 reports mean_pairwise_cosine=0.649±0.046 (different sampling than attack experiments). The "0.65" reference in the cross-dataset figure matches the baseline-experiments measure.

### 7.6 Summary Table

| Dataset | N | bl_fail (pooled) | A_cond (pooled) | Mean pairwise cos | Source |
|---------|---|-----------------|-----------------|-------------------|--------|
| MNIST cb raw | 100 | **11.8%±4.5%** | **3.63%** (16/441) | 0.396±0.014 | two-stage |
| FMNIST cb raw | 100 | **81.6%±3.8%** | **8.70%** (8/92) | 0.595±0.008 | two-stage |
| Centered CIFAR cb | 100 | **0.0%** | **0.0%** (0/250) | cos_proc=-0.009 | two-stage |
| Centered CIFAR cb | 500 | **20.4%±5.0%** | **2.01%** (4/199) | cos_proc=-0.001 | two-stage |
| Raw CIFAR cb | 10 | **80.0%±15.5%** | **0.0%** (0/10) | 0.832±0.052 | attack |

Phase 3 WB attacker (MNIST N=100 cb, 5 seeds × 50 probes = 250 total):
- Raw WB success: 30/250 = **12.0%** (includes 23 baseline-failing probes)
- Conditional WB success: 7/227 = **3.1%** (genuine: correct baseline → attack flips)
- Mean L2 all probes: 0.1132 ± 0.313
- Mean L2 among 30 raw successes: 0.6250

---

## Section 8: Capacity and Noise Stability Sweep

### 8.1 Phase 3 WB Attack Success by N (class_balanced, 5 seeds × 50 probes)

**Source:** `phase3_summary.csv`, `phase3_diag_b_baseline_corrected.csv`

| N | Raw WB% (mean±std) | Mean L2 | Baseline fail% | Cond WB% (mean) |
|---|--------------------|---------|----------------|-----------------|
| 10  | 0.0% ± 0.0%   | 0.000  | 0.0%   | 0.0%  |
| 50  | 5.2% ± 4.6%   | 0.0614 | 4.0%   | 1.29% |
| 100 | 12.0% ± 4.7%  | 0.1132 | 9.2%   | 3.10% |
| 500 | 41.2% ± 7.8%  | 0.3067 | 37.2%  | 6.50% |
| 1000| 52.0% ± 6.2%  | 0.3658 | 47.6%  | 8.63% |

Random strategy at N=100: raw WB = 20.0% ± 5.7%, cond WB = 4.79% (mean per-seed)

Conditional WB success per seed at N=100, class_balanced:
- seed 42: 1/47 = 2.13%, seed 43: 2/42 = 4.76%, seed 44: 2/46 = 4.35%, seed 45: 0/45 = 0.0%, seed 46: 2/47 = 4.26%

Conditional WB success per seed at N=500, class_balanced:
- seed 42: 11.76%, seed 43: 11.54%, seed 44: 5.88%, seed 45: 0.0%, seed 46: 3.33%

### 8.2 Phase 2 Noise Stability (Gaussian sigma=0.2, class_balanced)

**Source:** `phase2_stability_summary.csv` (5 seeds each)

| N | Accuracy mean±std | Interpretation |
|---|-------------------|----------------|
| 10  | 0.96 ± 0.055 | Near-perfect robustness |
| 50  | 0.816 ± 0.090 | Moderate degradation |
| 100 | 0.748 ± 0.064 | Significant degradation |
| 500 | 0.392 ± 0.090 | Heavy degradation |
| 1000| 0.308 ± 0.102 | Near chance |

Phase 2 single-seed spot check at gaussian mag=0.2, cb: N=10→1.00, N=50→0.94, N=100→0.74, N=500→0.32, N=1000→0.24

---

## Section 9: White-Box vs DE Black-Box Comparison

**Source:** `phase3_diag_a_paired.csv` (250 rows, N=100 class_balanced, 5 seeds × 50 probes)

### 9.1 Contingency Table

|              | DE fail | DE success |
|--------------|---------|------------|
| WB fail      | 220     | 0          |
| WB success   | 0       | 30         |

- **Perfect agreement**: WB and DE succeed/fail on exactly the same probes
- Cohen's kappa = 1.0
- Both methods: 30/250 = 12.0% raw success rate

### 9.2 Pixel Selection Comparison (among 30 joint successes)

- Same pixel location: 6/30 (20.0%)
- Different pixel location: 24/30 (**80.0%**)
- Mean WB L2 among successes: **0.6250**
- Mean DE L2 among successes: **0.7986**
- L2 correlation (all 250 probes): **0.5121**

### 9.3 Attacker Parameters

| Parameter | WB | DE |
|-----------|----|----|
| Knowledge | Full (X, beta) | Black-box (retrieve only) |
| Evaluations/probe | 3920 (exhaustive) | ~40,080 mean |
| Success rate N=100 cb | 12.0% | 12.0% |
| Mean L2 (successes) | 0.6250 | 0.7986 |

**Interpretation:** Both attackers identify identical vulnerable probes, confirming vulnerabilities are structural (not attacker-specific). DE requires ~10x more evaluations for the same result.

---

## Section 10: Vulnerability Analysis

**Source:** `phase3_diag_d_probe_features.csv`, `phase3_diag_d_summary.csv`  
(N=100 class_balanced, 250 probes total, 30 vulnerable / 220 non-vulnerable)

### 10.1 Statistical Tests

| Test | Vulnerable stat | Non-vulnerable stat | Test statistic | p-value | Interpretation |
|------|----------------|---------------------|----------------|---------|----------------|
| Chi2 class distribution | [1,11,0,3,2,1,1,4,2,5] | expected=3.0/class | 30.67 | 0.0004 | Non-uniform class bias |
| Mann-Whitney: neighbor cosine | 0.7727 | 0.7160 | 4388.0 | 0.0034 | Vulnerable higher (p<0.05) |
| Mann-Whitney: mean intensity | 0.0826 | 0.1364 | 756.0 | <0.001 | Sig. diff. (p<0.05) |
| Mann-Whitney: intensity std | 0.249 | 0.313 | 791.0 | <0.001 | Sig. diff. (p<0.05) |
| **Logistic CV balanced accuracy** | **0.8516** | 0.0503 (chance) | — | — | **Predictable (bal.acc > 0.70)** |

### 10.2 Key Findings

- **Class bias:** Classes 1 (11 vulnerable) and 9 (5 vulnerable) are over-represented; class 2 has 0 vulnerable probes. Chi2 test: p=0.0004.
- **Neighbor cosine:** Vulnerable probes have higher cosine to their stored-pattern neighbors (0.773 vs 0.716), suggesting they sit in more crowded regions of pattern space.
- **Intensity:** Vulnerable probes have lower mean intensity (0.083 vs 0.136) — darker images — and lower intensity std (0.249 vs 0.313) — less textured.
- **Predictability:** A logistic regression cross-validated classifier achieves 85.2% balanced accuracy in distinguishing vulnerable from non-vulnerable probes, confirming that vulnerability is a structural, predictable property of the input.

---

## Section 11: Pixel Value Distribution at Attack

**Source:** `phase3_whitebox_results.csv` (WB successes at N=100 cb)

### 11.1 Attack Value Selection (30 WB raw successes)

| Attack value | Count | Fraction |
|-------------|-------|----------|
| 0.0         | 13    | 43.3%    |
| 0.75        | 1     | 3.3%     |
| 1.0         | 16    | 53.3%    |
| **Extreme (0 or 1)** | **29** | **96.7%** |

### 11.2 Pixel Transitions (original bucket → attack value)

| Transition | Count | L2 |
|-----------|-------|-----|
| orig≈0.0 → 1.0 | 16 | ≈1.0 |
| orig≈0.0 → 0.0 | 11 | ≈0.0 |
| orig≈1.0 → 0.0 | 2  | ≈1.0 |
| orig≈0.0 → 0.75 | 1 | ≈0.75 |

**Note on orig≈0.0 → 0.0 entries (11/30):** These are predominantly from baseline-failing probes where the WB attacker picks the first candidate (pixel k=0, value=0.0) via the degenerate argmin. The 7 genuine conditional successes use high-L2 perturbations.

Mean L2 check: 16×1.0 + 11×0.0 + 2×1.0 + 1×0.75 = 18.75 / 30 = 0.625 ✓

---

## Section 12: Gaussian Noise Sigma Sweep (Amplification Factor)

**Source:** `phase3_diag_c_matched_sigma.csv`  
(N=100 class_balanced, 5 seeds, sigma range: 0.002–0.020)

### 12.1 Noise Failure at Each Sigma

| sigma | noise_cond_fail% (mean±std) | noise_L2 |
|-------|-----------------------------|----------|
| 0.002 | 0.0% ± 0.0% | 0.0431 |
| 0.003 | 0.43% ± 0.85% | 0.0645 |
| 0.004 | 0.86% ± 1.05% | 0.0859 |
| 0.005 | 0.86% ± 1.05% | 0.1071 |
| 0.006 | 0.86% ± 1.05% | 0.1281 |
| 0.008 | 0.86% ± 1.05% | 0.1700 |
| 0.010 | 0.86% ± 1.05% | 0.2117 |
| 0.015 | 1.34% ± 1.09% | 0.3157 |
| 0.020 | 1.34% ± 1.09% | 0.4196 |

Per-seed conditional noise failures at sigma=0.005:
- seed 42: 0.0%, seed 43: 0.0%, seed 44: 2.17%, seed 45: 0.0%, seed 46: 2.13%

### 12.2 Matched-Sigma Comparison

- WB attack mean L2 (all probes, N=100 cb): **0.1132**
- Best-matched sigma: **0.005** (noise L2 = 0.1071 ≈ 0.1132)
- Conditional noise failure at sigma=0.005: **0.86% ± 1.05%**
- Conditional WB success (N=100 cb): **3.10% ± 1.84%**
- **Amplification factor: 3.10% / 0.86% = 3.6x**

**Thesis claim:** A structured one-pixel perturbation is 3.6x more effective at disrupting retrieval than unstructured random noise of the same L2 magnitude.

---

## Section 13: Cross-Dataset Summary

| Dataset | N | Pairwise cosine | Baseline fail | Conditional attack | Notes |
|---------|---|-----------------|---------------|---------------------|-------|
| MNIST N=100 cb | 100 | 0.396±0.014 | 11.8%±4.5% | 3.63% (two-stage) / 3.1% (phase3) | Raw pixel space |
| FMNIST N=100 cb | 100 | 0.595±0.008 | 81.6%±3.8% | 8.70% (two-stage) | Severe crowding; few correct baseline |
| Centered CIFAR N=100 cb | 100 | raw=0.837, proc=-0.009 | 0.0% | 0.0% | Centering eliminates crowding |
| Centered CIFAR N=500 cb | 500 | raw=0.835, proc=-0.001 | 20.4%±5.0% | 2.01% | Capacity limit re-introduces failure |
| Raw CIFAR N=10 cb | 10 | 0.649 (baseline) / 0.832 (attack-exp) | 80.0%±15.5% | 0.0% | Null result: no correct retrievals to attack |

**Pattern-crowding interpretation:**
- Low pairwise cosine (MNIST ≈0.40): moderate crowding, moderate baseline failure, some attacks succeed
- High pairwise cosine (FMNIST ≈0.60): severe crowding, 82% baseline failure, few correct retrievals to attack
- Near-zero pairwise cosine (Centered CIFAR ≈0): no crowding, 0% failure, 0% attack
- High pairwise cosine + small N (Raw CIFAR ≈0.65 at N=10): 80% failure, no attack possible (no correct retrievals)

**Centering mechanism:** Subtracting the mean across all stored patterns and re-normalizing drives pairwise cosines from ~0.84 to ~0.0, eliminating inter-pattern interference.

---

## Section 14: Computational Timing

| Operation | Cost |
|-----------|------|
| WB attacker per probe | 3920 forward passes (784 pixels × 5 values, 1 batch) |
| DE attacker per probe | ~40,080 evaluations (mean, N=100 cb) |
| WB / DE cost ratio | ~1 : 10 |
| Phase 3 WB full run (250 probes) | O(250 × 3920) = ~1M forward passes |
| Phase 3 DE full run (250 probes) | O(250 × 40080) = ~10M forward passes |

Phase 3 WB per-seed raw success at N=100 cb: 4, 10, 6, 5, 5 successes / 50 probes  
Phase 3 DE mean evaluations at N=100 cb: **40,080 ± ~800** (from phase3_blackbox_results.csv)  
DE early stop condition: success found AND max(fitness[successes]) > 0.5; max 100 gens × 400 evals = 40,000 + overhead

---

## Section 15: Changes from Earlier Reported Numbers

This section documents all discrepancies between numbers cited in prior drafts and the current data.

### 15.1 Confirmed / Unchanged

| Metric | Earlier | Current | Status |
|--------|---------|---------|--------|
| MNIST baseline failure (phase3) | ~9.2% | 9.2% (23/250) | CONFIRMED |
| WB vs DE: 30 joint successes | 30 | 30 | CONFIRMED |
| WB vs DE: 80% different pixel | 80% | 80.0% (24/30) | CONFIRMED |
| L2 correlation all probes | 0.512 | 0.5121 | CONFIRMED |
| Logistic CV balanced accuracy | 85.2% | 0.8516 (85.16%) | CONFIRMED |
| Amplification factor | 3.6x | 3.6x (3.10/0.86) | CONFIRMED |
| Centered CIFAR N=100: 0% failure, 0% attack | YES | YES | CONFIRMED |
| Chi2 class bias p-value | p<0.001 | 0.0004 | CONFIRMED |
| Phase 3 WB mean L2 (all probes) | 0.1132 | 0.1132 | CONFIRMED |

### 15.2 Updated Numbers

| Metric | Earlier | Current | Reason |
|--------|---------|---------|--------|
| FMNIST conditional attack | 7.6% | **9.01% (mean) / 8.70% (pooled)** | Full 5-seed rerun with corrected attacker |
| FMNIST baseline failure | ~80% | **81.6%** | Accurate across 5 seeds |
| Raw CIFAR pairwise cosine | "0.65" | **0.649** (baseline exp) / **0.832** (attack exp) | Two different measurements; baseline=0.649 is canonical |
| MNIST two-stage baseline failure | — | **11.8%±4.5%** | Phase 3 uses 50 probes/seed; two-stage uses 100 probes/seed |
| MNIST conditional attack (two-stage) | — | **3.63%** (two-stage) vs **3.1%** (phase3) | Different attacker version + probe count |

### 15.3 Known Inconsistencies in Visualization

The cross-dataset bar chart (`attack_visualization.py`) has hardcoded pairwise cosine values:
```python
cos_vals = [0.397, 0.648, 0.832]   # MNIST, FMNIST, CIFAR
```
- MNIST 0.397 ≈ measured 0.396 ✓
- CIFAR 0.832 ≈ measured 0.832 (attack-experiments) ✓
- **FMNIST 0.648 ≠ measured 0.595** — this figure value is incorrect and should be updated to 0.595

### 15.4 Attacker Version Notes

The `phase3_closing_diagnostics.py` uses `WhiteBoxOnePixelAttacker` (simple argmin). The two-stage scripts use the corrected cosine-delta + first-order fallback. Differences in reported conditional success (3.1% vs 3.63%) reflect both the attacker version and the different probe count (50 vs 100 per seed). The two-stage conditional success (3.63%) is the more reliable estimate.

### 15.5 Raw CIFAR Pairwise Cosine Discrepancy

Two different measurements at N=10:
- `grayscale_cifar_baseline_results.csv`: 0.649 ± 0.046 (5 seeds, independent sampling)
- `grayscale_cifar_attack_results.csv`: 0.832 ± 0.052 (5 seeds, attack-script sampling)

The discrepancy is due to different image sampling strategies between the two scripts. The baseline experiment (0.649) is the canonical figure used in visualizations. The attack experiment's higher pairwise cosine explains why even at N=10 the baseline failure is 80% in that specific sampling.

---

## Appendix A: Phase 3 Diag-B Full Table (N=100, class_balanced)

| Seed | bl_fail% | raw_WB% | cond_WB% | cond_noise@0.001% |
|------|----------|---------|----------|-------------------|
| 42   | 6.0%     | 8.0%    | 2.13%    | 0.0%              |
| 43   | 16.0%    | 20.0%   | 4.76%    | 0.0%              |
| 44   | 8.0%     | 12.0%   | 4.35%    | 2.17%             |
| 45   | 10.0%    | 10.0%   | 0.0%     | 0.0%              |
| 46   | 6.0%     | 10.0%   | 4.26%    | 2.13%             |
| **mean±std** | **9.2%±3.6%** | **12.0%±4.7%** | **3.10%±1.84%** | **0.86%±1.05%** |

## Appendix B: Phase 3 Diag-B Full Table (N=500, class_balanced)

| Seed | bl_fail% | raw_WB% | cond_WB% |
|------|----------|---------|----------|
| 42   | 32.0%    | 40.0%   | 11.76%   |
| 43   | 48.0%    | 54.0%   | 11.54%   |
| 44   | 32.0%    | 36.0%   | 5.88%    |
| 45   | 34.0%    | 34.0%   | 0.0%     |
| 46   | 40.0%    | 42.0%   | 3.33%    |
| **mean±std** | **37.2%±6.2%** | **41.2%±7.8%** | **6.50%±4.77%** |

## Appendix C: CIFAR Baseline Capacity Sweep (Raw Grayscale)

**Source:** `grayscale_cifar_baseline_results.csv` (5 seeds per N)

| N | Baseline failure% | Mean pairwise cosine |
|---|-------------------|---------------------|
| 10  | 80.0% ± 15.5% | 0.649 ± 0.046 |
| 20  | 88.0% ± 11.7% | 0.738 ± 0.026 |
| 30  | 94.0% ± 3.9%  | 0.775 ± 0.010 |
| 50  | 94.0% ± 5.5%  | 0.798 ± 0.010 |
| 100 | 98.4% ± 1.5%  | 0.818 ± 0.008 |

Raw CIFAR becomes unusable as memory grows: at N=100, 98.4% of probes fail to retrieve correctly even without any attack.
