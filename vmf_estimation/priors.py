"""
Prior functions for Minimum Message Length (MML) estimation.

This module implements three prior density functions for the concentration
parameter κ of von Mises-Fisher distributions, as described in Wallace & Dowe (1993).

All functions return log h(κ) for numerical stability in MML optimization.

References:
    Wallace, C. S., & Dowe, D. L. (1993). MML estimation of the von Mises
    concentration parameter. Technical Report 93/193, Dept. of Computer Science,
    Monash University.
"""

import numpy as np
from scipy import special
from .core import fisher_information


def log_h1_prior(kappa: float, dim: int) -> float:
    """
    h₁ prior: h₁(κ) = 1/κ (Wallace & Dowe 1993, line 297).

    This is an improper prior (doesn't integrate to 1). It corresponds to
    a uniform prior on log(κ) and is motivated by the approximation
    M₂(0,κ) ≈ N(0, 1/κ) for large κ.

    Args:
        kappa: Concentration parameter κ > 0
        dim: Dimensionality (must be 2 for Wallace & Dowe formulation)

    Returns:
        log h₁(κ) = -log(κ)

    Notes:
        - Improper prior (integrates to infinity)
        - Puts large weight on small κ (diverges as κ→0)
        - From Wallace & Dowe (1993), used with modified Fisher info (equation 28)
        - Most aggressive in penalizing large κ
    """
    if kappa <= 0:
        return -np.inf

    if dim != 2:
        raise ValueError("h1 prior formula from Wallace & Dowe (1993) is only defined for d=2")

    # h₁(κ) = 1/κ
    # log h₁(κ) = -log(κ)
    return -np.log(kappa)


def log_h2_prior(kappa: float, dim: int) -> float:
    """
    Cauchy prior: h₂(κ) = 2/(π(1+κ²)) (Wallace & Dowe 1993, line 297).

    This is a proper prior (integrates to 1). It's the Cauchy distribution
    with scale parameter 1, which is bounded at the origin.

    Args:
        kappa: Concentration parameter κ > 0
        dim: Dimensionality (must be 2 for Wallace & Dowe formulation)

    Returns:
        log h₂(κ) = log(2) - log(π) - log(1 + κ²)

    Notes:
        - Proper prior (integrates to 1)
        - Cauchy distribution with scale 1
        - Used with modified Fisher info (equation 28) to prevent divergence
        - Moderate penalty on large κ
    """
    if kappa <= 0:
        return -np.inf

    if dim != 2:
        raise ValueError("h2 prior formula from Wallace & Dowe (1993) is only defined for d=2")

    # h₂(κ) = 2/(π(1+κ²))
    # log h₂(κ) = log(2) - log(π) - log(1+κ²)
    return np.log(2.0) - np.log(np.pi) - np.log(1.0 + kappa**2)


def log_h3_prior(kappa: float, dim: int) -> float:
    """
    Wallace-Dowe recommended prior: h₃(κ) = κ/(1+κ²)^(3/2).

    This prior is "flat at the origin in Cartesian coordinates" (line 301)
    and is the recommended choice in Wallace & Dowe (1993). It provides a
    good balance - proper, normalized, and doesn't diverge at the origin.

    Args:
        kappa: Concentration parameter κ > 0
        dim: Dimensionality (must be 2 for Wallace & Dowe formulation)

    Returns:
        log h₃(κ) = log(κ) - (3/2)·log(1 + κ²)

    Notes:
        - Proper prior (integrates to 1)
        - Wallace & Dowe's recommended choice
        - Bounded and non-divergent in Cartesian (X,Y) coordinates
        - Uses standard (unmodified) Fisher info (equation 27, line 321)
        - Best overall MML performance in the paper
    """
    if kappa <= 0:
        return -np.inf

    if dim != 2:
        raise ValueError("h3 prior formula from Wallace & Dowe (1993) is only defined for d=2")

    # h₃(κ) = κ/(1+κ²)^(3/2)
    # log h₃(κ) = log(κ) - (3/2)·log(1+κ²)
    return np.log(kappa) - 1.5 * np.log(1.0 + kappa**2)


# Dictionary mapping prior names to functions for convenience
PRIOR_FUNCTIONS = {
    'h1': log_h1_prior,
    'h2': log_h2_prior,
    'h3': log_h3_prior,
}


def get_prior_function(prior_name: str):
    """
    Get prior function by name.

    Args:
        prior_name: One of 'h1', 'h2', 'h3'

    Returns:
        Prior function

    Raises:
        ValueError: If prior_name is not recognized
    """
    if prior_name not in PRIOR_FUNCTIONS:
        raise ValueError(
            f"Unknown prior '{prior_name}'. "
            f"Must be one of: {list(PRIOR_FUNCTIONS.keys())}"
        )
    return PRIOR_FUNCTIONS[prior_name]
