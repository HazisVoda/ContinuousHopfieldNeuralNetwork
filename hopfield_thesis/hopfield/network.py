"""
Continuous modern Hopfield network — Ramsauer et al. 2021.

Storage matrix X has shape (d, N):  d = pattern dimension, N = number of stored patterns.
Update rule:  xi_new = X @ softmax(beta * X.T @ xi)
Energy:       E(xi) = -lse(beta, X.T @ xi) + 0.5*||xi||^2 + beta^-1*log(N) + 0.5*M^2
              where lse(beta, z) = beta^-1 * log(sum(exp(beta*z)))
              and M = max_i ||x_i||.
"""

import torch
import torch.nn.functional as F


class ContinuousHopfield:
    """Continuous modern Hopfield network (Ramsauer et al. 2021)."""

    def __init__(self, X: torch.Tensor, beta: float = 8.0) -> None:
        """
        Args:
            X:    Pattern matrix of shape (d, N).  Stored by reference so callers
                  can swap it out without reconstructing the network.
            beta: Inverse temperature.  Higher values sharpen the softmax and
                  approach classical winner-take-all retrieval.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (d, N), got shape {X.shape}")
        self.X = X.float()
        self.beta = float(beta)

    # ------------------------------------------------------------------
    # Core update rule
    # ------------------------------------------------------------------

    def retrieve(self, query: torch.Tensor, steps: int = 1) -> torch.Tensor:
        """
        Run the continuous Hopfield update rule for `steps` iterations.

        Args:
            query: Shape (d,) for a single query or (d, B) for a batch of B queries.
            steps: Number of synchronous update steps.

        Returns:
            Updated state tensor with the same shape as `query`.
        """
        squeeze = query.ndim == 1
        xi = query.float()
        if squeeze:
            xi = xi.unsqueeze(1)  # (d, 1)

        X = self.X.to(xi.device)

        for _ in range(steps):
            # logits: (N, B)
            logits = self.beta * (X.T @ xi)
            weights = F.softmax(logits, dim=0)   # (N, B)
            xi = X @ weights                     # (d, B)

        if squeeze:
            xi = xi.squeeze(1)
        return xi

    # ------------------------------------------------------------------
    # Energy function
    # ------------------------------------------------------------------

    def energy(self, xi: torch.Tensor) -> torch.Tensor:
        """
        Ramsauer energy E(xi).

        Includes constant terms (beta^-1 * log(N) and 0.5 * M^2) so that
        absolute energy values are comparable across calls.

        Args:
            xi: State vector of shape (d,) or (d, B).

        Returns:
            Scalar energy (or shape (B,) for batched input).
        """
        squeeze = xi.ndim == 1
        xi = xi.float()
        if squeeze:
            xi = xi.unsqueeze(1)

        X = self.X.to(xi.device)
        N = X.shape[1]

        # log-sum-exp term: lse(beta, X.T @ xi)  =>  shape (B,)
        logits = self.beta * (X.T @ xi)          # (N, B)
        lse = torch.logsumexp(logits, dim=0) / self.beta   # (B,)

        # quadratic term
        quad = 0.5 * (xi * xi).sum(dim=0)        # (B,)

        # constants
        M = X.norm(dim=0).max()
        const = (1.0 / self.beta) * torch.log(torch.tensor(float(N), device=xi.device)) \
                + 0.5 * M ** 2

        E = -lse + quad + const                   # (B,)

        if squeeze:
            E = E.squeeze(0)
        return E
