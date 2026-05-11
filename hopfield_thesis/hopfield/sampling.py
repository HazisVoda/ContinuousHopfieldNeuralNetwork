"""
Pattern selection utilities for MNIST experiments.

Both functions accept the (images, labels) tuple returned by load_mnist_train()
and return a (784, n) float32 storage matrix plus the list of dataset indices used.
Patterns are in [0, 1] by construction (normalization is inherited from the dataset).
"""

import torch


def sample_random(
    dataset: tuple[torch.Tensor, torch.Tensor],
    n: int,
    seed: int,
) -> tuple[torch.Tensor, list[int]]:
    """
    Uniform random sample of n images from the dataset.

    Args:
        dataset: (images, labels) — images shape (N_total, 784) float32 in [0, 1].
        n:       Number of patterns to sample.
        seed:    RNG seed for reproducibility.

    Returns:
        X:       Storage matrix shape (784, n), float32.
        indices: Dataset row indices of the selected images.
    """
    images, _ = dataset
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(images.shape[0], generator=rng)[:n]
    X = images[perm].float().T.contiguous()   # (784, n)
    return X, perm.tolist()


def sample_class_balanced(
    dataset: tuple[torch.Tensor, torch.Tensor],
    n: int,
    seed: int,
) -> tuple[torch.Tensor, list[int]]:
    """
    Sample n images with as-equal-as-possible class representation (10 classes).

    Allocation: base = n // 10 per class; the first (n % 10) classes each get
    one extra pattern.  For n divisible by 10, allocation is exactly n // 10
    per class.

    Patterns are returned class-ordered: all class-0 patterns first, then
    class-1, ..., class-9.  Within each class, patterns are selected uniformly
    at random.

    Args:
        dataset: (images, labels) — images shape (N_total, 784) float32 in [0, 1].
        n:       Number of patterns to sample.
        seed:    RNG seed for reproducibility.

    Returns:
        X:       Storage matrix shape (784, n), float32, columns ordered by class.
        indices: Dataset row indices ordered by class.
    """
    images, labels = dataset
    n_classes = 10
    base = n // n_classes
    extra = n % n_classes

    rng = torch.Generator()
    rng.manual_seed(seed)

    all_patterns: list[torch.Tensor] = []
    all_indices:  list[int]          = []

    for cls in range(n_classes):
        n_cls = base + (1 if cls < extra else 0)
        if n_cls == 0:
            continue
        cls_row_indices = (labels == cls).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(cls_row_indices), generator=rng)[:n_cls]
        selected = cls_row_indices[perm]
        all_patterns.append(images[selected].float())
        all_indices.extend(selected.tolist())

    X = torch.cat(all_patterns, dim=0).T.contiguous()   # (784, n)
    return X, all_indices
