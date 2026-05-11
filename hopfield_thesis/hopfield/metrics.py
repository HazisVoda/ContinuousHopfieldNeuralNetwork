"""
Retrieval quality metrics.
"""

import torch


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean squared error between two flat vectors."""
    return float(((a.float() - b.float()) ** 2).mean())


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity in [-1, 1]."""
    a = a.float().flatten()
    b = b.float().flatten()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 0.0
    return float((a @ b) / denom)


def retrieval_accuracy(
    retrieved: torch.Tensor,
    X: torch.Tensor,
    true_index: int,
) -> bool:
    """
    Returns True iff the column of X most similar (by cosine) to `retrieved`
    is at position `true_index`.

    Args:
        retrieved:   Shape (d,).
        X:           Pattern matrix shape (d, N).
        true_index:  Expected nearest-neighbor column index.
    """
    r = retrieved.float().flatten()
    sims = torch.mv(X.float().T, r)          # (N,)
    norms = X.float().norm(dim=0) * r.norm() # (N,)
    norms = norms.clamp(min=1e-8)
    cos = sims / norms
    return int(cos.argmax().item()) == int(true_index)
