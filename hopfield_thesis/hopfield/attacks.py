"""
One-pixel adversarial attackers for the continuous Hopfield network.

Threat model: attacker modifies exactly ONE pixel (location + value).
Attack goal: untargeted — cause the retrieved nearest-neighbour to differ
from the true stored pattern index.

Two implementations share a common interface:
  WhiteBoxOnePixelAttacker  — vectorised exhaustive scan (3920 candidates,
                              single batched retrieve() call).
  DEBlackBoxOnePixelAttacker — Differential Evolution à la Su et al. 2019,
                               fully vectorised fitness evaluation.
"""

from __future__ import annotations

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Shared interface
# ─────────────────────────────────────────────────────────────────────────────

class OnePixelAttacker:
    """Base class — subclasses implement attack()."""

    def attack(
        self,
        query: torch.Tensor,
        true_index: int,
        network,
    ) -> dict:
        """
        Returns
        -------
        dict with keys:
          success          bool
          pixel_i          int   (row in 28×28)
          pixel_j          int   (col in 28×28)
          pixel_value      float (adversarial value written)
          original_value   float (pixel value before attack)
          perturbation_l2  float (|pixel_value − original_value|)
          cosine_to_true   float (cosine sim of retrieved to true pattern)
          retrieved_index  int   (nearest stored pattern after attack)
          evaluations      int   (number of retrieve() calls consumed)
        """
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# White-box exhaustive attacker
# ─────────────────────────────────────────────────────────────────────────────

class WhiteBoxOnePixelAttacker(OnePixelAttacker):
    """
    Exhaustive white-box one-pixel attack.

    Tests all 784 pixel locations × 5 candidate values = 3920 candidates
    in a SINGLE batched retrieve() call.  Selects the candidate that
    minimises cosine similarity between the retrieved and the true stored
    pattern.

    evaluations = 3920 (fixed).
    """

    _CANDIDATE_VALUES = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

    def attack(
        self,
        query: torch.Tensor,
        true_index: int,
        network,
    ) -> dict:
        device = query.device
        X = network.X                                  # (784, N)
        cands = self._CANDIDATE_VALUES.to(device)

        n_locs  = 784
        n_vals  = cands.shape[0]                       # 5
        n_cands = n_locs * n_vals                      # 3920

        # pixel index for each candidate  (784 * 5 entries)
        locs = torch.arange(n_locs, device=device).repeat_interleave(n_vals)  # (3920,)
        vals = cands.repeat(n_locs)                                            # (3920,)

        # (784, 3920): column k is query with pixel locs[k] set to vals[k]
        queries = query.unsqueeze(1).expand(-1, n_cands).clone()
        queries[locs, torch.arange(n_cands, device=device)] = vals

        # One batched retrieve
        retrieved = network.retrieve(queries, steps=1)   # (784, 3920)

        # Cosine similarity to the true stored pattern
        true_pat = X[:, true_index]                      # (784,)
        dots     = retrieved.T @ true_pat                # (3920,)
        r_norms  = retrieved.norm(dim=0)                 # (3920,)
        t_norm   = true_pat.norm()
        cos_sims = dots / (r_norms * t_norm).clamp(min=1e-8)   # (3920,)

        # Best attack = lowest cosine to true pattern
        worst_k   = int(cos_sims.argmin().item())
        worst_loc = int(locs[worst_k].item())
        worst_val = float(vals[worst_k].item())
        worst_cos = float(cos_sims[worst_k].item())

        # Nearest stored pattern for the worst candidate
        worst_ret = retrieved[:, worst_k]               # (784,)
        X_norms   = X.norm(dim=0)                       # (N,)
        cos_all   = (X.T @ worst_ret) / (X_norms * worst_ret.norm()).clamp(min=1e-8)
        retrieved_index = int(cos_all.argmax().item())

        pixel_i        = worst_loc // 28
        pixel_j        = worst_loc % 28
        original_value = float(query[worst_loc].item())

        return {
            "success":         retrieved_index != true_index,
            "pixel_i":         pixel_i,
            "pixel_j":         pixel_j,
            "pixel_value":     worst_val,
            "original_value":  original_value,
            "perturbation_l2": abs(worst_val - original_value),
            "cosine_to_true":  worst_cos,
            "retrieved_index": retrieved_index,
            "evaluations":     n_cands,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Black-box DE attacker
# ─────────────────────────────────────────────────────────────────────────────

class DEBlackBoxOnePixelAttacker(OnePixelAttacker):
    """
    Differential Evolution one-pixel attack (Su et al. 2019).

    Each individual encodes (i_float, j_float, v) ∈ [0,28) × [0,28) × [0,1].
    Coordinates are rounded to integers at evaluation time.

    Fitness: 1 − cosine_similarity(retrieved, true_pattern).
    Higher fitness = more damage to the true-pattern retrieval.

    Fitness evaluation is FULLY VECTORISED: one batched retrieve() per
    generation covers the entire population (pop_size queries at once).

    Early termination: if any individual achieves retrieval failure AND
    fitness > 0.5 (cosine_to_true < 0.5), stop immediately.
    """

    def __init__(
        self,
        pop_size: int = 400,
        max_gens: int = 100,
        F: float = 0.5,
        CR: float = 0.7,
        seed: int = 42,
    ) -> None:
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.F        = F
        self.CR       = CR
        self.seed     = seed

    # ── vectorised population evaluator ──────────────────────────────────────

    def _evaluate(
        self,
        pop_t: torch.Tensor,      # (pop_size, 3)
        query: torch.Tensor,      # (784,)
        true_index: int,
        network,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (fitness, retrieved_indices), both shape (pop_size,)."""
        device   = query.device
        pop_size = pop_t.shape[0]
        X        = network.X                            # (784, N)

        i_int   = pop_t[:, 0].long().clamp(0, 27)
        j_int   = pop_t[:, 1].long().clamp(0, 27)
        pix_idx = (i_int * 28 + j_int).clamp(0, 783)  # (pop_size,)
        v       = pop_t[:, 2].clamp(0.0, 1.0)

        queries = query.unsqueeze(1).expand(-1, pop_size).clone()  # (784, pop_size)
        queries[pix_idx, torch.arange(pop_size, device=device)] = v

        retrieved = network.retrieve(queries, steps=1)              # (784, pop_size)

        true_pat = X[:, true_index]                                 # (784,)
        dots     = retrieved.T @ true_pat                           # (pop_size,)
        r_norms  = retrieved.norm(dim=0)                            # (pop_size,)
        t_norm   = true_pat.norm()
        cos_sims = dots / (r_norms * t_norm).clamp(min=1e-8)
        fitness  = 1.0 - cos_sims                                   # (pop_size,)

        X_norms  = X.norm(dim=0).unsqueeze(1)                      # (N, 1)
        dots_all = X.T @ retrieved                                  # (N, pop_size)
        cos_all  = dots_all / (X_norms * r_norms.unsqueeze(0)).clamp(min=1e-8)
        ret_idx  = cos_all.argmax(dim=0)                            # (pop_size,)

        return fitness, ret_idx

    # ── main attack entry point ───────────────────────────────────────────────

    def attack(
        self,
        query: torch.Tensor,
        true_index: int,
        network,
    ) -> dict:
        rng    = np.random.RandomState(self.seed)
        device = query.device
        ps     = self.pop_size

        # Initialise population in continuous (i,j,v) space
        pop_np       = np.zeros((ps, 3), dtype=np.float32)
        pop_np[:, 0] = rng.uniform(0, 28, ps)
        pop_np[:, 1] = rng.uniform(0, 28, ps)
        pop_np[:, 2] = rng.uniform(0,  1, ps)
        pop_t        = torch.tensor(pop_np, dtype=torch.float32, device=device)

        fitness, ret_idx = self._evaluate(pop_t, query, true_index, network)
        total_evals = ps

        best_k   = int(fitness.argmax().item())
        best_ind = pop_t[best_k].clone()
        best_fit = float(fitness[best_k])
        best_ret = int(ret_idx[best_k].item())

        for _gen in range(self.max_gens):
            # Early stop: retrieval failure with high damage
            succ = ret_idx != true_index
            if succ.any() and float(fitness[succ].max()) > 0.5:
                break

            # Mutation: vectorised random index sampling (collisions rare at ps=400)
            a = torch.tensor(rng.randint(0, ps, ps), device=device)
            b = torch.tensor(rng.randint(0, ps, ps), device=device)
            c = torch.tensor(rng.randint(0, ps, ps), device=device)
            mutants = pop_t[a] + self.F * (pop_t[b] - pop_t[c])
            mutants[:, 0] = mutants[:, 0].clamp(0.0, 27.999)
            mutants[:, 1] = mutants[:, 1].clamp(0.0, 27.999)
            mutants[:, 2] = mutants[:, 2].clamp(0.0,  1.0)

            # Crossover
            cross = torch.tensor(
                rng.random((ps, 3)) < self.CR,
                dtype=torch.bool, device=device,
            )
            # Guarantee at least one dimension crosses per individual
            no_cross = ~cross.any(dim=1)
            if no_cross.any():
                forced = torch.tensor(
                    rng.randint(0, 3, int(no_cross.sum().item())),
                    device=device,
                )
                cross[no_cross.nonzero(as_tuple=True)[0], forced] = True

            trial_t = torch.where(cross, mutants, pop_t)
            trial_t[:, 0] = trial_t[:, 0].clamp(0.0, 27.999)
            trial_t[:, 1] = trial_t[:, 1].clamp(0.0, 27.999)
            trial_t[:, 2] = trial_t[:, 2].clamp(0.0,  1.0)

            trial_fit, trial_ret = self._evaluate(trial_t, query, true_index, network)
            total_evals += ps

            # Greedy selection
            improve  = trial_fit > fitness
            pop_t    = torch.where(improve.unsqueeze(1), trial_t,   pop_t)
            fitness  = torch.where(improve,              trial_fit,  fitness)
            ret_idx  = torch.where(improve,              trial_ret,  ret_idx)

            k = int(fitness.argmax().item())
            if float(fitness[k]) > best_fit:
                best_fit = float(fitness[k])
                best_ind = pop_t[k].clone()
                best_ret = int(ret_idx[k].item())

        # Decode best individual
        i_int          = int(best_ind[0].clamp(0, 27).long().item())
        j_int          = int(best_ind[1].clamp(0, 27).long().item())
        pixel_val      = float(best_ind[2].clamp(0, 1).item())
        pix_idx        = i_int * 28 + j_int
        original_value = float(query[pix_idx].item())

        return {
            "success":         best_ret != true_index,
            "pixel_i":         i_int,
            "pixel_j":         j_int,
            "pixel_value":     pixel_val,
            "original_value":  original_value,
            "perturbation_l2": abs(pixel_val - original_value),
            "cosine_to_true":  1.0 - best_fit,
            "retrieved_index": best_ret,
            "evaluations":     total_evals,
        }
