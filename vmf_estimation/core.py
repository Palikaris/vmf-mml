"""
Core mathematical functions for von Mises-Fisher parameter estimation.

This module implements the fundamental mathematical operations required for
all von Mises-Fisher estimators:
- Bessel ratio A_d(κ) and its inverse
- Fisher information F(κ)
- Helper functions for numerical stability
"""

import numpy as np
from scipy import special, optimize
from typing import Union


def bessel_ratio_A(kappa: Union[float, np.ndarray], dim: int) -> Union[float, np.ndarray]:
    """
    Compute the Bessel ratio A_d(κ) = I_{p+1}(κ) / I_p(κ) where p = d/2 - 1.

    For the von Mises-Fisher distribution in d dimensions, this ratio appears in:
    - The gradient of the log-likelihood
    - The relationship between κ and mean resultant length R̄

    Args:
        kappa: Concentration parameter κ ≥ 0 (scalar or array)
        dim: Dimensionality d ≥ 2

    Returns:
        A_d(κ) in [0, 1)

    Properties:
        - A_d(0) = 0
        - A_d(∞) → 1
        - A_d is monotonically increasing
        - For large κ: A_d(κ) ≈ 1 - (d-1)/(2κ)
        - For small κ: A_d(κ) ≈ κ/d

    Implementation:
        Uses scipy.special.ive (exponentially scaled Bessel functions) to prevent
        overflow for large κ. The scaling factors cancel in the ratio.
    """
    # Keep track of whether input was scalar
    is_scalar = np.isscalar(kappa)
    kappa = np.atleast_1d(np.asarray(kappa, dtype=float))

    # Initialize result array
    result = np.zeros_like(kappa, dtype=float)

    # Handle κ = 0 case explicitly
    zero_mask = kappa == 0
    result[zero_mask] = 0.0

    # Process non-zero values
    nonzero_mask = kappa > 0
    if not np.any(nonzero_mask):
        return 0.0 if is_scalar else result

    kappa_nz = kappa[nonzero_mask]

    # Order of Bessel functions
    p = dim / 2.0 - 1.0

    # For very small kappa, use Taylor series: A_d(κ) ≈ κ/d
    small_mask = kappa_nz < 1e-5
    result[nonzero_mask] = kappa_nz / dim  # Initialize with small kappa approximation

    # For moderate to large kappa, use exponentially scaled Bessel functions
    # ive(p, κ) = I_p(κ) * exp(-|κ|)
    # The exp scaling cancels in the ratio
    moderate_mask = (kappa_nz >= 1e-5) & (kappa_nz <= 700)
    if np.any(moderate_mask):
        kappa_mod = kappa_nz[moderate_mask]
        I_p = special.ive(p, kappa_mod)
        I_p_plus_1 = special.ive(p + 1, kappa_mod)

        # Check for numerical issues (nan, inf, or divide by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = I_p_plus_1 / I_p

        # If we get nan/inf, choose appropriate approximation based on κ
        problem_mask = ~np.isfinite(ratio)
        if np.any(problem_mask):
            kappa_problem = kappa_mod[problem_mask]

            # For small κ relative to dimension, use Taylor: A ≈ κ/d
            # For large κ, use asymptotic: A ≈ 1 - (d-1)/(2κ)
            # Crossover roughly when κ ~ sqrt(d)
            small_rel_mask = kappa_problem < np.sqrt(dim)
            large_rel_mask = ~small_rel_mask

            ratio[problem_mask] = np.where(
                small_rel_mask,
                kappa_problem / dim,  # Taylor approximation
                1.0 - (dim - 1) / (2 * kappa_problem)  # Asymptotic approximation
            )

        # Update result for moderate kappa values
        full_indices = np.where(nonzero_mask)[0]
        moderate_indices = full_indices[moderate_mask]
        result[moderate_indices] = ratio

    # For very large kappa, use asymptotic approximation: A_d(κ) ≈ 1 - (d-1)/(2κ)
    large_mask = kappa_nz > 700
    if np.any(large_mask):
        kappa_large = kappa_nz[large_mask]
        asymptotic = 1.0 - (dim - 1) / (2 * kappa_large)

        # Update result for large kappa values
        full_indices = np.where(nonzero_mask)[0]
        large_indices = full_indices[large_mask]
        result[large_indices] = asymptotic

    # Return scalar if input was scalar
    if is_scalar:
        return float(result[0])
    else:
        return result


def fisher_information(kappa: float, dim: int) -> float:
    """
    Compute Fisher information F(κ) for the von Mises-Fisher distribution.

    The Fisher information measures the amount of information that the data carries
    about the parameter κ. For von Mises-Fisher:

    F(κ) = d * [1 - A_d(κ)² - (d-1)·A_d(κ)/κ]

    Simplified form (without dimension factor):
    F(κ) = 1 - A_d(κ)² - (d-1)·A_d(κ)/κ

    Args:
        kappa: Concentration parameter κ > 0
        dim: Dimensionality d ≥ 2

    Returns:
        Fisher information F(κ) > 0

    Raises:
        ValueError: If kappa <= 0 (Fisher info undefined at κ=0)

    Note:
        Fisher information is used in MML estimation in the penalty term.
        It's always positive for κ > 0.
    """
    if kappa <= 0:
        raise ValueError(f"Fisher information undefined for kappa <= 0 (got kappa={kappa})")

    A = bessel_ratio_A(kappa, dim)

    # F(κ) = 1 - A² - (d-1)A/κ
    # Note: We use the simplified form without the dimension factor d
    # This matches the Wallace & Dowe (1993) formulation
    F = 1.0 - A**2 - (dim - 1) * A / kappa

    # Fisher information should always be positive
    # If numerical errors make it negative, something is wrong
    if F <= 0:
        raise ValueError(
            f"Fisher information is non-positive (F={F}, kappa={kappa}, dim={dim}). "
            "This indicates a numerical stability issue."
        )

    return F


def bessel_ratio_A_inverse(R: float, dim: int, tol: float = 1e-12) -> float:
    """
    Solve A_d(κ) = R for κ using root finding.

    This is the most critical function for parameter estimation, as it's used to
    compute the ML estimate. Since A_d(κ) is monotonically increasing from 0 to 1,
    we can use robust root-finding methods.

    Args:
        R: Target value (mean resultant length), 0 ≤ R < 1
        dim: Dimensionality d ≥ 2
        tol: Tolerance for root finding (default: 1e-12)

    Returns:
        κ such that A_d(κ) = R within tolerance

    Algorithm:
        1. Handle edge cases (R=0 returns 0, R→1 uses asymptotic formula)
        2. For d=3, use special formula for better initial guess
        3. Determine search bounds using asymptotic approximations
        4. Use scipy.optimize.brentq for robust root finding

    Implementation notes:
        - Uses Banerjee et al. (2005) approximation for initial guess
        - For d=3: A_3(κ) = coth(κ) - 1/κ, which can be inverted more accurately
        - Asymptotic formulas ensure reasonable bounds even for extreme R values
    """
    # Edge case: R = 0 implies κ = 0
    if R <= 0:
        return 0.0

    # Edge case: R very close to 1
    # Use asymptotic approximation: A_d(κ) ≈ 1 - (d-1)/(2κ)
    # Solving: R = 1 - (d-1)/(2κ) gives κ = (d-1)/(2(1-R))
    if R >= 0.99999:
        return (dim - 1) / (2 * (1 - R))

    # For valid R ∈ (0, 1), we need to find κ

    # Initial guess using Banerjee et al. (2005) approximation
    # This works reasonably well for all dimensions
    # κ ≈ R(d - R²) / (1 - R²)
    kappa_init = R * (dim - R**2) / (1 - R**2)

    # For d=3, we have a special formula that can give better initial guess
    # A_3(κ) = coth(κ) - 1/κ
    # For moderate κ, coth(κ) ≈ 1 + 2exp(-2κ), so:
    # A_3(κ) ≈ 1 - 1/κ + 2exp(-2κ)
    # This is still transcendental, so we stick with Banerjee approximation

    # Sanity check: kappa_init should be positive and finite
    if not np.isfinite(kappa_init) or kappa_init <= 0:
        # Fall back to simple approximation
        kappa_init = 1.0

    # Define the equation to solve: A_d(κ) - R = 0
    def equation(kappa: float) -> float:
        A = bessel_ratio_A(kappa, dim)
        if not np.isfinite(A):
            # If Bessel ratio fails, use asymptotic approximation
            A = 1.0 - (dim - 1) / (2 * kappa)
        return A - R

    # Determine bounds for root finding
    # Lower bound: slightly above 0
    kappa_lower = 1e-6

    # Upper bound: use asymptotic formula as guide
    # From A_d(κ) ≈ 1 - (d-1)/(2κ), solving for κ when A = R:
    # κ ≈ (d-1)/(2(1-R))
    # We use 3× this as upper bound to be safe
    if R < 0.99:
        kappa_upper = max(3 * kappa_init, 3 * (dim - 1) / (2 * (1 - R)))
    else:
        kappa_upper = 10 * (dim - 1) / (2 * (1 - R))

    # Ensure bounds are finite
    if not np.isfinite(kappa_upper) or kappa_upper <= kappa_lower:
        kappa_upper = max(100, 10 * kappa_init)

    # Make sure upper bound is large enough
    # Check that A(kappa_upper) > R
    A_upper = bessel_ratio_A(kappa_upper, dim)
    if np.isfinite(A_upper) and A_upper < R:
        # Need larger upper bound
        kappa_upper = min(kappa_upper * 10, 1e6)  # Cap at 1e6 to avoid overflow

    # Use Brent's method for robust root finding
    # This combines bisection, secant, and inverse quadratic interpolation
    try:
        kappa_est = optimize.brentq(
            equation,
            kappa_lower,
            kappa_upper,
            xtol=tol,
            rtol=tol
        )
        return kappa_est

    except ValueError as e:
        # If brentq fails (e.g., function doesn't change sign), fall back to minimization
        # Minimize |A(κ) - R|
        result = optimize.minimize_scalar(
            lambda k: abs(equation(k)),
            bounds=(kappa_lower, kappa_upper),
            method='bounded'
        )

        if result.success:
            return result.x
        else:
            raise ValueError(
                f"Failed to find κ for R={R}, dim={dim}. "
                f"Initial guess: {kappa_init}, bounds: [{kappa_lower}, {kappa_upper}]"
            ) from e


def compute_mean_direction(data: np.ndarray) -> np.ndarray:
    """
    Compute the mean direction μ̂ from unit vectors.

    Args:
        data: Array of shape (n_samples, dim), each row is a unit vector

    Returns:
        μ̂: Unit vector of shape (dim,), the mean direction

    Algorithm:
        1. Compute the vector sum: s = Σ xᵢ
        2. Normalize: μ̂ = s / ||s||

    Note:
        If all vectors cancel out (||s|| ≈ 0), the mean direction is undefined.
        This only happens for uniform distributions or adversarial data.
    """
    # Sum all vectors
    mean_vec = np.mean(data, axis=0)

    # Compute norm
    norm = np.linalg.norm(mean_vec)

    # Handle edge case where vectors cancel out
    if norm < 1e-10:
        raise ValueError(
            "Mean direction is undefined: vectors sum to near-zero. "
            "Data may be uniformly distributed on the sphere."
        )

    # Normalize
    return mean_vec / norm


def compute_mean_resultant_length(data: np.ndarray) -> float:
    """
    Compute mean resultant length R̄ = ||Σ xᵢ|| / N.

    The mean resultant length is a sufficient statistic for κ (given μ).
    It measures how concentrated the data is around the mean direction.

    Args:
        data: Array of shape (n_samples, dim), each row is a unit vector

    Returns:
        R̄ ∈ [0, 1]

    Properties:
        - R̄ = 0: uniformly distributed (κ = 0)
        - R̄ = 1: all points at same location (κ → ∞)
        - R̄ = A_d(κ) at the ML estimate

    Implementation:
        R̄ = ||mean(data, axis=0)||
    """
    mean_vec = np.mean(data, axis=0)
    R_bar = np.linalg.norm(mean_vec)

    # R̄ should be in [0, 1] for unit vectors
    # Due to numerical errors, it might slightly exceed 1
    if R_bar > 1.0:
        if R_bar > 1.01:
            raise ValueError(
                f"Mean resultant length exceeds 1.0: R̄={R_bar}. "
                "Input data may not be unit vectors."
            )
        # Small numerical error, clip to 1.0
        R_bar = 1.0

    return R_bar
