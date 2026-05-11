"""
Corruption utilities for building noisy / occluded query vectors.

All functions accept a flat (784,) or spatial (28, 28) tensor and return
the same shape, clipped to [0, 1].  Random operations accept an optional
integer seed for reproducibility.
"""

import torch


def _to_2d(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Return (image_2d, was_flat).  Raises if shape is unexpected."""
    if image.shape == torch.Size([784]):
        return image.view(28, 28).clone(), True
    if image.shape == torch.Size([28, 28]):
        return image.clone(), False
    raise ValueError(f"Expected shape (784,) or (28, 28), got {image.shape}")


def _from_2d(img2d: torch.Tensor, was_flat: bool) -> torch.Tensor:
    return img2d.view(784) if was_flat else img2d


def mask_bottom_half(image: torch.Tensor) -> torch.Tensor:
    """Zero the bottom 14 rows (rows 14-27) of a 28x28 image."""
    img2d, was_flat = _to_2d(image)
    img2d[14:, :] = 0.0
    return _from_2d(img2d, was_flat)


def mask_random_patch(
    image: torch.Tensor,
    patch_size: int = 8,
    seed: int | None = None,
) -> torch.Tensor:
    """Zero a randomly placed `patch_size x patch_size` square."""
    img2d, was_flat = _to_2d(image)
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    max_row = 28 - patch_size
    max_col = 28 - patch_size
    row = int(torch.randint(0, max_row + 1, (1,), generator=rng).item())
    col = int(torch.randint(0, max_col + 1, (1,), generator=rng).item())
    img2d[row : row + patch_size, col : col + patch_size] = 0.0
    return _from_2d(img2d, was_flat)


def add_gaussian_noise(
    image: torch.Tensor,
    sigma: float = 0.2,
    seed: int | None = None,
) -> torch.Tensor:
    """Add N(0, sigma^2) noise and clip to [0, 1]."""
    img2d, was_flat = _to_2d(image)
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    noise = torch.zeros_like(img2d).normal_(0.0, sigma, generator=rng)
    img2d = (img2d + noise).clamp(0.0, 1.0)
    return _from_2d(img2d, was_flat)


def flip_pixels(
    image: torch.Tensor,
    flip_rate: float = 0.1,
    seed: int | None = None,
) -> torch.Tensor:
    """Replace each pixel with uniform [0,1] with probability `flip_rate`."""
    img2d, was_flat = _to_2d(image)
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
    mask = torch.rand(img2d.shape, generator=rng) < flip_rate
    random_values = torch.rand(img2d.shape, generator=rng)
    img2d[mask] = random_values[mask]
    return _from_2d(img2d, was_flat)
