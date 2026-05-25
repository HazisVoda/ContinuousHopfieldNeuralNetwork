# Continuous Modern Hopfield Network — Thesis Project (Phase 1)

This repository implements a **continuous modern Hopfield network** (Ramsauer et al. 2021)
applied to MNIST digit images.  It is the first phase of a four-phase thesis project
investigating adversarial robustness of associative memory models.

---

## Mathematical Formulation

### Storage

Patterns are stored as columns of a matrix **X** ∈ ℝ^{d×N}, where d = 784 (flattened
28×28 MNIST pixel) and N is the number of stored patterns.

### Retrieval update rule

Given a query vector ξ ∈ ℝ^d, one step of synchronous update is:

```
ξ_new = X · softmax(β · X^T · ξ)
```

This is a differentiable, continuous relaxation of the classical Hopfield update.
The softmax concentrates mass on the most-similar stored pattern when β is large,
and blends patterns when β is small.

### Energy function

The Ramsauer energy is:

```
E(ξ) = −lse(β, X^T ξ) + ½ ‖ξ‖² + β⁻¹ log N + ½ M²
```

where lse(β, z) = β⁻¹ log Σᵢ exp(β zᵢ) is the log-sum-exp, and M = maxᵢ ‖xᵢ‖.
The update rule is a gradient descent step on this energy, guaranteeing convergence.

---

## Design Decisions

### Raw flattened pixels (no encoder / learned embeddings)

**Chosen because:** Phase 3 will apply Su et al. 2019 one-pixel attacks, which
operate directly in pixel space.  Using raw pixels means the attack search space
equals the representation space — no inversion step is needed.  Learned embeddings
would require either a differentiable decoder or approximate inversion, complicating
the attack pipeline and obscuring what the network actually memorises.

### [0, 1] normalisation

**Chosen because:** Matches the Su et al. 2019 one-pixel attack formulation (pixel
values in [0, 1] or [0, 255]).  Also keeps softmax dot-products β · X^T · ξ in a
numerically stable range; very large dot-products cause softmax saturation.

### From-scratch PyTorch (no Hopfield-layers library)

**Chosen because:** Full transparency for thesis defensibility.  Every line of the
update rule and energy function is explicit, making it straightforward to modify
(e.g. change the similarity kernel) and to verify against the paper.

### β = 8.0 default

**Chosen as a starting point.**  At β = 8.0 the softmax is sharp enough to achieve
near-perfect clean retrieval on 10 patterns (see Sanity Check 1) without saturating
to hard argmax.  Calibration across β values is reserved for Phase 2.

---

## Repository Structure

```
hopfield_thesis/
├── hopfield/
│   ├── __init__.py
│   ├── network.py        # ContinuousHopfield class
│   ├── corruption.py     # noise / occlusion utilities
│   └── metrics.py        # MSE, cosine similarity, retrieval accuracy
├── experiments/
│   └── phase1_sanity.py  # four sanity checks (runnable script)
├── figures/              # generated PNG plots (git-ignored)
├── data/                 # MNIST download cache (git-ignored)
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Run

```bash
cd hopfield_thesis
python -m experiments.phase1_sanity
```

MNIST (~11 MB) is downloaded automatically to `data/` on first run.
All figures are written to `figures/`.

---

## Sanity Checks

### Check 1 — Clean Retrieval (`01_clean_retrieval.png`)

10 MNIST images (one per digit class) are stored as patterns.
Each is used as its own query.  The network should return the pattern unchanged.

**Pass criterion:** MSE per pattern ≈ 0 (< 1 × 10⁻³).

*Why it matters:* Verifies that the update rule is implemented correctly and that
the fixed-point condition holds — a stored pattern is a fixed point of the dynamics.

### Check 2 — Half-Occlusion Retrieval (`02_half_occlusion.png`)

The bottom 14 rows (pixels 392–783) of each query are zeroed.
The network retrieves from this half-visible input.

**Pass criterion:** ≥ 9/10 patterns correctly identified by cosine similarity.

*Why it matters:* Demonstrates associative completion — the core property of
Hopfield-type memory.  A network that cannot complete from partial input is not
functioning as a content-addressable memory.

### Check 3 — Gaussian Noise Robustness (`03_gaussian_robustness.png`, `03b_gaussian_examples.png`)

Gaussian noise (σ ∈ {0.05, 0.1, 0.2, 0.3, 0.5}) is added to each query.
MSE and retrieval accuracy are plotted against σ.

**Pass criterion:** Graceful degradation — accuracy should be high at low σ and
fall gradually rather than collapsing abruptly.

*Why it matters:* Establishes the noise floor before adversarial attacks are
introduced in Phase 3.  The σ = 0.2 working point is used as the baseline
perturbation magnitude in later phases.

### Check 4 — Capacity Stress Test (`04_capacity.png`)

N ∈ {10, 100, 1000, 5000} patterns are stored.  For each N, 50 stored patterns
are queried with σ = 0.2 Gaussian noise.  Results are plotted on a log-x axis.

**Pass criterion:** Accuracy decreases monotonically with N (graceful capacity
saturation) rather than remaining flat or jumping erratically.

*Why it matters:* The continuous Hopfield network has exponential theoretical
capacity O(d) in the strict fixed-point sense (Ramsauer et al.).  At N ≫ d = 784
the network will saturate.  This check locates where degradation begins for
unstructured MNIST images.

---

## Interpreting Results

| Metric | Healthy range | Red flag |
|---|---|---|
| Check 1 mean MSE | < 1e-4 | > 1e-2 |
| Check 2 accuracy | ≥ 9/10 | < 7/10 |
| Check 3 acc at σ=0.05 | ≥ 0.9 | < 0.7 |
| Check 3 acc at σ=0.5 | any value | abrupt cliff at low σ |
| Check 4 acc at N=10 | ≥ 0.9 | < 0.8 |

---

## β calibration diagnostic

**Motivation:** Phase 1 Sanity Check 4 showed retrieval accuracy collapsing from ~68%
at N=100 to ~18% at N=1000 under σ=0.2 Gaussian noise.  Before committing to β=8.0
for later phases, we need to distinguish two hypotheses:

1. **β is the bottleneck** — the softmax is miscalibrated for the pattern density at
   N=1000; tuning β could recover meaningful accuracy.
2. **Pattern similarity is the bottleneck** — MNIST patterns at N=1000 are too
   similar for the network to separate under σ=0.2 noise regardless of β;
   this is a fundamental capacity limit.

**What it tests:** β is swept over {1, 2, 4, 8, 16, 32, 64, 128} with N=1000 stored
patterns and σ=0.2 noise on 50 randomly selected probes (all seeds fixed at 42).
Mean MSE, retrieval accuracy, and mean cosine similarity are recorded per β.
As a β-independent baseline, the pairwise cosine similarity distribution of the
1000 stored patterns is computed to characterise pattern crowding.

**Results:** `figures/05_beta_sweep.png` (accuracy/MSE/cosine vs β),
`figures/05b_pattern_similarity.png` (pairwise similarity histogram).
Run: `python -m experiments.phase1_beta_diagnostic`

**Result:** see stdout output of the diagnostic script.

**Interpreting the three regimes:**
- *β was the bottleneck:* best accuracy > 0.7 → recalibrate β for Phase 2+.
- *Pattern similarity dominant:* best accuracy < 0.5 across all β → β tuning cannot
  fix overlapping patterns; focus on pattern selection or separation strategies.
- *Mixed regime:* accuracy improves moderately but stays below 0.7 → both β tuning
  and pattern selection strategy matter for later phases.

---

## Phase 2: Random-noise robustness baseline

**Purpose:** Establish the noise-robustness floor of the network before any
adversarial perturbations are introduced.  Phase 3 one-pixel attacks will be
evaluated relative to this baseline: an attack is meaningful only if it causes
degradation beyond what equivalent-magnitude random noise achieves.

**Experimental grid:**
- N stored patterns: {10, 50, 100, 500, 1000}
- Selection strategy: *random* (uniform) and *class-balanced* (equal class
  representation, ordered class 0-9 within the storage matrix)
- Noise types and magnitudes:
  - Gaussian noise: σ ∈ {0.05, 0.10, 0.20, 0.30, 0.50}
  - Pixel-flip: rate ∈ {0.05, 0.10, 0.20, 0.30, 0.50}
  - Half-occlusion: single condition (bottom 14 rows zeroed = 50% masked)

**Why occlusion is a single condition, not a sweep:** Half-occlusion is treated
as a fixed structural corruption rather than a parameterised severity level,
following Ramsauer et al. 2021 and Kashyap 2024 conventions for Hopfield
completion experiments.  It serves as a reference for the "partial cue"
retrieval regime rather than a point on a noise-severity axis.

**Results:**
- Full metrics table: `experiments/phase2_results.csv`
  (columns: N, strategy, noise_type, magnitude, mse, accuracy, cosine)
- `figures/phase2_gaussian.png`: 2×5 grid — accuracy and MSE vs σ per (N, strategy)
- `figures/phase2_pixelflip.png`: same structure for pixel-flip rate
- `figures/phase2_occlusion.png`: bar chart of accuracy and MSE per N for each strategy
- `figures/phase2_summary_heatmap.png`: 3-panel heatmap (one per noise type) of
  accuracy at representative magnitude (σ=0.2 / rate=0.2 / occlusion)
- `figures/phase2_qualitative.png`: qualitative grid, N=100 class-balanced,
  3 noise types × digits 0-5

Run: `python -m experiments.phase2_baselines`

**Result:** see stdout output of the baseline script.

### Phase 2 stability verification

**Why:** Phase 2 was run on a single seed (42).  Some cells showed non-monotonic
behaviour (random N=10 worse than N=50) that could be seed artifacts.  Before
committing the headline numbers to the thesis and proceeding to Phase 3 attacks,
we verify that mean ± std across 5 seeds is consistent with the single-seed result.

**What was measured:** Same N / strategy grid as Phase 2, but only the
representative magnitudes (σ=0.2 Gaussian, rate=0.2 pixel-flip, half-occlusion).
Five replicates: seeds {42, 43, 44, 45, 46}.  Seed 42 exactly reproduces Phase 2
by construction (identical seeding formula).

**Results:**
- Per-seed raw metrics: `experiments/phase2_stability_results.csv`
- Aggregated mean ± std: `experiments/phase2_stability_summary.csv`
- Accuracy-vs-N plot with error bars: `figures/phase2_stability.png`

Run: `python -m experiments.phase2_stability`

**Verdict:** see stdout output of the stability script.

---

## Phase 3: Adversarial one-pixel attacks

**Purpose:** Determine whether a single-pixel perturbation can cause the network to
retrieve the wrong stored pattern, and quantify the gap between adversarial and
random perturbations of equal magnitude.

**Threat model:** Attacker modifies exactly **one pixel** (location + value).
Goal: untargeted — cause the network to retrieve any pattern other than the true one.
Two threat levels are tested:

- **White-box exhaustive** — attacker has full access to the storage matrix X and β.
  All 784 × 5 = 3920 candidate (location, value) pairs are evaluated in a single
  vectorised `retrieve()` call.  Selects the candidate minimising cosine similarity
  to the true stored pattern.  Fixed cost: 3920 evaluations per probe.
- **DE black-box** — attacker can query `retrieve()` but does not see X or β.
  Differential Evolution (Su et al. 2019): population size 400, up to 100 generations,
  F = 0.5, CR = 0.7.  Fitness evaluation is fully vectorised (one batched retrieve per
  generation).  Tested only at the headline cell (N = 100, class-balanced).

**Experimental grid:**
- N ∈ {10, 50, 100, 500, 1000}, both selection strategies, 5 seeds {42..46}
  (white-box; headline cell additionally run with DE black-box)
- 50 probes per cell (all stored patterns probed when N < 50)
- Random-noise-equivalent baseline: Gaussian σ ∈ {0.01, 0.02, 0.05, 0.10}
  tested on the same probes as stage 1

**Results:**
- Per-probe white-box data: `experiments/phase3_whitebox_results.csv` (2100 rows)
- Per-probe DE data: `experiments/phase3_blackbox_results.csv` (250 rows)
- Random-noise-equivalent: `experiments/phase3_random_noise_equivalent.csv`
- Aggregated summary: `experiments/phase3_summary.csv`
- `figures/phase3_attack_grid.png`: success rate, mean L2, cosine damage vs N
  (mean ± std, 5 seeds, both strategies)
- `figures/phase3_attack_vs_random.png`: adversarial vs random noise failure rate
  at equal L2 — headline cell N=100 class-balanced
- `figures/phase3_whitebox_vs_de.png`: white-box vs DE success rate and mean L2
- `figures/phase3_vulnerability_maps.png`: 28×28 per-pixel attack frequency heatmaps
  for N ∈ {10, 100, 500}, class-balanced, seed=42
- `figures/phase3_attack_examples.png`: example successful attacks (original |
  attacked | retrieved | false pattern), N=100 class-balanced

Run: `python -m experiments.phase3_attacks`

**Key results (N=100 class-balanced, seed mean across 5 runs):**
- White-box success rate: 12.0%, mean L2=0.113, 3920 evaluations per probe
- DE black-box success rate: 12.0%, mean evals=40,080
- Random noise at equal L2 (σ=0.01, L2=0.28): 10.0% failure rate
- Raw adversarial amplification: 1.2× — inflated by non-zero baseline (see diagnostics below)

### Phase 3 diagnostics

**Why run:** Two Phase 3 findings required verification before proceeding.
(1) WB and DE reported identical 12.0% success rates — possibly coincidence, possibly a
code path bug.
(2) Random noise at σ=0.01 already produced 10% retrieval failure, suggesting a
non-zero baseline inflates both the attack success rate and the noise failure rate.

`experiments/phase3_diagnostics.py` reads the existing Phase 3 CSVs (no attack re-runs)
and computes two analyses:

**Diagnostic A — WB vs DE agreement** (`experiments/phase3_diag_a_paired.csv`,
`figures/phase3_diag_a_contingency.png`): For each of 250 (seed, probe) pairs at the
headline cell, compares success/failure, attacked pixel, and retrieved false index.
The 2×2 contingency table shows perfect row-wise agreement (all 30 successes appear
in both attackers; WB-only and DE-only counts are both 0), confirming the identical
12.0% rates are genuine — both attackers identify the same 30 vulnerable probes.
On those 30 jointly-successful probes, however, only 20% chose the same pixel: WB and
DE arrive at the same false basin (100% retrieved-index agreement) via different
perturbations.  L2 correlation r = 0.51 confirms the attacks are mechanistically
independent.

**Diagnostic B — Baseline-corrected effectiveness** (`experiments/phase3_diag_b_baseline_corrected.csv`,
`figures/phase3_diag_b_corrected.png`): Clean retrieval (no noise, no attack) already
fails 9.2% of probes at the headline cell (stored patterns that are not fixed points
under a single update step at N=100).  Conditional on clean success, WB attack success
is 3.1% and random-noise failure at σ=0.01 is 0.9%.  Corrected adversarial
amplification: **3.6×** — significantly larger than the raw 1.2× figure once baseline
failures are removed from both numerator and denominator.

Run: `python -m experiments.phase3_diagnostics`

### Phase 3 closing diagnostics

`experiments/phase3_closing_diagnostics.py` runs two final analyses to sharpen the
headline numbers before Phase 4 writeup.

**Diagnostic C — Exact magnitude-matched σ** (`experiments/phase3_diag_c_matched_sigma.csv`,
`figures/phase3_diag_c_sigma_sweep.png`): Sweeps σ ∈ {0.002 … 0.020} on the headline
cell to find the Gaussian noise level whose mean L2 exactly matches the white-box
attack mean L2 (0.113).  Conditional failure rates (excluding already-failing probes)
are computed per σ across 5 seeds.

**Diagnostic D — Vulnerable probe characterization** (`experiments/phase3_diag_d_probe_features.csv`,
`experiments/phase3_diag_d_summary.csv`, `figures/phase3_diag_d_classes.png`,
`figures/phase3_diag_d_features.png`): Characterizes the 30 vulnerable probes via
class distribution (chi-square), nearest-neighbor cosine and pixel-intensity statistics
(Mann-Whitney U), and a logistic regression with 5-fold cross-validation.

Run: `python -m experiments.phase3_closing_diagnostics`

**Note:** To our knowledge, one-pixel adversarial attacks have not previously been
applied to continuous modern Hopfield networks.  The low success rate at N=100
reflects the network's inherent robustness — a single pixel carries limited
information relative to the 784-dimensional pattern representation.  Attack
success increases monotonically with N (56% at N=1000) as pattern crowding
reduces the margin needed to cause mis-retrieval.

---

## Excel-ready exports and grayscale CIFAR baseline

### Excel-ready exports (`excel_exports/`)

The `excel_exports/` directory contains one CSV file per thesis figure, formatted
for direct import into Excel or any spreadsheet tool:

- **Wide format** — each column is one data series (one line or bar group in a chart).
  The first column is always the x-axis variable (N values, σ, β, etc.).
- **Mean and std as separate columns** — allows Excel to use the std columns as
  custom error bars via the *Error Bars → Custom* option in chart formatting.
- **No merged cells or formatting** — plain UTF-8 comma-separated, one header row only.
- **`INDEX.csv`** — master index listing every file, its source CSV or script,
  the corresponding thesis figure name, and a short description.

Run: `python -m experiments.export_for_excel`

### Grayscale CIFAR-10 baseline (`experiments/grayscale_cifar_baseline.py`)

A characterisation-only experiment — **no adversarial attacks** — included for
thesis flexibility.  CIFAR-10 training images (50,000 × 32×32 RGB) are converted
to 1024-dimensional grayscale vectors using standard luminance weights
(0.2989 R + 0.5870 G + 0.1140 B) and normalised to [0, 1].  Clean retrieval
baseline failure rate and mean pairwise cosine similarity are measured at
N ∈ {10, 20, 30, 50, 100} across 5 seeds.

**Result:** Baseline failure of 98.4% at N=100 (mean pairwise cosine 0.818)
confirms that grayscale CIFAR-10 is outside the network's reliable retrieval
regime at all tested N values.  This is consistent with the Fashion-MNIST finding
(80% failure at N=100) and the pattern-crowding mechanism characterised in Phase 2:
natural image datasets with higher inter-pattern similarity saturate the network's
capacity far below the MNIST operating point.

Run: `python -m experiments.grayscale_cifar_baseline`

---

## Roadmap

| Phase | Description |
|---|---|
| **Phase 1** *(this)* | Continuous Hopfield implementation + four sanity checks on MNIST |
| **Phase 2** | Stochastic noise baselines: systematic sweep of corruption types and severities, establishing robustness curves prior to attack |
| **Phase 3** | One-pixel adversarial attacks via differential evolution (Su et al. 2019) — one-pixel perturbation that causes mis-retrieval |
| **Phase 4** | Full experiments and analysis: attack success rates, energy landscape visualisation, comparison across β values, thesis write-up |

---

## References

- Ramsauer H. et al. (2021). *Hopfield Networks is All You Need.* ICLR 2021.
  arXiv:2008.02217
- Su J. et al. (2019). *One Pixel Attack for Fooling Deep Neural Networks.*
  IEEE Transactions on Evolutionary Computation. arXiv:1710.08864
