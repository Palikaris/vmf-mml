"""
Data generation and loading utilities for von Mises-Fisher distributions.

This module provides functions to:
- Generate synthetic vMF samples
- Normalize vectors to the unit sphere
- Load real word embeddings (for future use)
"""

import numpy as np
from typing import Optional


def generate_vmf_samples(
    mu: np.ndarray,
    kappa: float,
    n_samples: int,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Generate samples from von Mises-Fisher distribution.

    Uses a method based on sampling the marginal distribution of the first
    coordinate and uniform sampling on the orthogonal sphere.

    Args:
        mu: Mean direction (unit vector of shape (dim,))
        kappa: Concentration parameter κ ≥ 0
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility

    Returns:
        Array of shape (n_samples, dim) with unit vectors sampled from vMF(μ, κ)

    Algorithm:
        For each sample:
        1. Sample w (cosine of angle to μ) from marginal distribution
        2. Sample v uniformly on (d-1)-sphere orthogonal to μ
        3. Construct x = √(1-w²)·v + w·μ

    References:
        - Wood, A. T. A. (1994). Simulation of the von Mises Fisher distribution.
        - Ulrich, G. (1984). Computer generation of distributions on the m-sphere.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Ensure mu is unit vector
    mu = np.asarray(mu, dtype=float)
    mu = mu / np.linalg.norm(mu)
    dim = len(mu)

    # Handle κ = 0 case: uniform on sphere
    if kappa == 0:
        samples = np.random.randn(n_samples, dim)
        samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)
        return samples

    # Generate samples
    samples = np.zeros((n_samples, dim))

    for i in range(n_samples):
        # Sample w from marginal distribution f(w) ∝ (1-w²)^((d-3)/2) exp(κw)
        w = _sample_weight(kappa, dim)

        # Sample v uniformly on (d-1)-dimensional unit sphere
        v = np.random.randn(dim - 1)
        v = v / np.linalg.norm(v)

        # Construct sample in the coordinate system where μ = [0, 0, ..., 0, 1]
        # Then rotate to align with actual μ
        x_canonical = np.concatenate([np.sqrt(1 - w**2) * v, [w]])

        # Rotate to align with μ using Householder reflection
        samples[i] = _householder_rotate(x_canonical, mu)

    return samples


def _sample_weight(kappa: float, dim: int) -> float:
    """
    Sample w from f(w) ∝ (1-w²)^((d-3)/2) exp(κw) for w ∈ [-1, 1].

    For d=3, this simplifies to f(w) ∝ exp(κw), which is straightforward.
    For general d, we use rejection sampling.

    Args:
        kappa: Concentration parameter
        dim: Dimensionality

    Returns:
        Sample w ∈ [-1, 1]
    """
    if dim == 3:
        # Special case for d=3: f(w) ∝ exp(κw)
        # CDF: F(w) = (exp(κw) - exp(-κ)) / (exp(κ) - exp(-κ))
        # Inverse CDF: w = log(u·(exp(κ) - exp(-κ)) + exp(-κ)) / κ

        # To avoid overflow for large κ, rewrite:
        # w = log(u·2·sinh(κ) + exp(-κ)) / κ
        #   = log(exp(-κ)·(u·(exp(2κ) - 1) + 1)) / κ
        #   = (-κ + log(u·(exp(2κ) - 1) + 1)) / κ

        u = np.random.rand()

        if kappa < 1e-10:
            # For very small κ, distribution is nearly uniform
            return 2 * u - 1
        elif kappa > 50:
            # For large κ, use asymptotic approximation
            # Distribution concentrates near w=1
            # f(w) ≈ exp(κ(w-1)) for w near 1
            # So w ≈ 1 + log(u)/κ
            return 1 + np.log(u) / kappa
        else:
            # General case
            exp_2k = np.exp(2 * kappa)
            w = (-kappa + np.log(u * (exp_2k - 1) + 1)) / kappa
            return w

    else:
        # General case: use rejection sampling (Wood 1994)
        # This is more complex but works for any dimension
        # For simplicity, using a basic rejection sampler

        b = (dim - 1) / (2 * kappa + np.sqrt(4 * kappa**2 + (dim - 1)**2))
        x0 = (1 - b) / (1 + b)
        c = kappa * x0 + (dim - 1) * np.log(1 - x0**2)

        while True:
            z = np.random.beta((dim - 1) / 2, (dim - 1) / 2)
            w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
            u = np.random.rand()

            # Acceptance criterion
            if kappa * w + (dim - 1) * np.log(1 - x0 * w) - c >= np.log(u):
                return w


def _householder_rotate(x: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Rotate vector x from canonical orientation [0,...,0,1] to align with μ.

    Uses Householder reflection if needed. If μ already points in the
    canonical direction, no rotation is needed.

    Args:
        x: Vector to rotate (assumed to be in coordinate system where μ = e_d)
        mu: Target direction (unit vector)

    Returns:
        Rotated vector aligned with μ
    """
    dim = len(mu)

    # Canonical direction [0, 0, ..., 0, 1]
    e_d = np.zeros(dim)
    e_d[-1] = 1.0

    # If μ is already e_d, no rotation needed
    if np.allclose(mu, e_d):
        return x

    # If μ is -e_d, special case
    if np.allclose(mu, -e_d):
        x_rot = x.copy()
        x_rot[-1] = -x_rot[-1]
        return x_rot

    # General case: use Householder reflection
    # Householder vector: v = μ - e_d
    # Normalized: v̂ = v / ||v||
    # Reflection: H = I - 2v̂v̂ᵀ
    # Apply: x' = Hx

    v = mu - e_d
    v = v / np.linalg.norm(v)

    # Householder reflection: x' = x - 2(v·x)v
    x_rot = x - 2 * np.dot(v, x) * v

    return x_rot


def normalize_to_unit_sphere(data: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit length.

    Args:
        data: Array of shape (n_samples, dim)

    Returns:
        Normalized array where each row has unit norm

    Note:
        Zero vectors will result in NaN. Handle with care.
    """
    norms = np.linalg.norm(data, axis=1, keepdims=True)

    # Avoid division by zero
    norms = np.where(norms > 0, norms, 1.0)

    return data / norms


def load_word_embeddings(
    filepath: str,
    max_samples: Optional[int] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Load word embeddings from file.

    Supports GloVe format: word val1 val2 ... valn (space-separated)

    Args:
        filepath: Path to embedding file
        max_samples: Optional limit on number of vectors to load
        normalize: Whether to normalize to unit sphere

    Returns:
        Array of shape (n_samples, dim)

    Note:
        This is a placeholder for Phase 10. For now, returns an error message.
    """
    raise NotImplementedError(
        "Word embedding loading will be implemented in Phase 10. "
        "For now, use generate_vmf_samples() for synthetic data."
    )
