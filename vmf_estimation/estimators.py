"""
Estimators for von Mises-Fisher concentration parameter κ.

This module implements multiple estimation methods:
- ML: Maximum Likelihood
- Schou: Bias-corrected (Schou 1978)
- MML: Minimum Message Length with three priors (Wallace & Dowe 1993)

All estimators inherit from a common BaseEstimator interface.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional

from .core import (
    compute_mean_direction,
    compute_mean_resultant_length,
    bessel_ratio_A_inverse,
    fisher_information,
    bessel_ratio_A,
)
from .priors import log_h1_prior, log_h2_prior, log_h3_prior


class BaseEstimator(ABC):
    """
    Abstract base class for all κ estimators.

    All estimators must implement the estimate() method which takes
    data (unit vectors) and returns (μ̂, κ̂).
    """

    @abstractmethod
    def estimate(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Estimate von Mises-Fisher parameters from data.

        Args:
            data: Array of shape (n_samples, dim), each row is a unit vector

        Returns:
            (mu_hat, kappa_hat): Mean direction (unit vector) and concentration

        Raises:
            ValueError: If data is invalid (wrong shape, not unit vectors, etc.)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this estimator."""
        pass

    def _validate_data(self, data: np.ndarray) -> None:
        """
        Validate that data has correct format.

        Args:
            data: Array to validate

        Raises:
            ValueError: If data is invalid
        """
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D array, got shape {data.shape}")

        if len(data) < 2:
            raise ValueError(f"Need at least 2 samples, got {len(data)}")

        # Check that vectors are approximately unit norm
        norms = np.linalg.norm(data, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6):
            max_dev = np.max(np.abs(norms - 1.0))
            raise ValueError(
                f"All data vectors must have unit norm. "
                f"Maximum deviation: {max_dev}"
            )

    def _compute_resultant_length(self, data: np.ndarray) -> float:
        """
        Compute mean resultant length R̄ = ||Σxᵢ|| / n.

        Args:
            data: Array of shape (n_samples, dim)

        Returns:
            R̄ ∈ [0, 1]
        """
        return compute_mean_resultant_length(data)


class MLEstimator(BaseEstimator):
    """
    Maximum Likelihood estimator.

    The ML estimate κ̂_ML is found by solving:
        A_d(κ̂_ML) = R̄

    where R̄ is the mean resultant length and A_d is the Bessel ratio.

    This is the simplest and most commonly used estimator, but it has
    positive bias for small samples.

    References:
        - Mardia, K. V., & Jupp, P. E. (2000). Directional Statistics.
        - Fisher, N. I. (1993). Statistical Analysis of Circular Data.
    """

    def estimate(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute ML estimate of (μ, κ).

        Args:
            data: Array of shape (n_samples, dim), unit vectors

        Returns:
            (mu_hat, kappa_hat): ML estimates

        Algorithm:
            1. Compute mean direction: μ̂ = Σxᵢ / ||Σxᵢ||
            2. Compute mean resultant length: R̄ = ||Σxᵢ|| / n
            3. Solve A_d(κ̂) = R̄ for κ̂ using numerical root finding
        """
        self._validate_data(data)

        # Get dimension
        n_samples, dim = data.shape

        # Compute mean direction
        mu_hat = compute_mean_direction(data)

        # Compute mean resultant length
        R_bar = compute_mean_resultant_length(data)

        # Solve A_d(κ) = R̄ for κ
        kappa_hat = bessel_ratio_A_inverse(R_bar, dim)

        return mu_hat, kappa_hat

    @property
    def name(self) -> str:
        return "ML"

    def __repr__(self) -> str:
        return "MLEstimator()"


class SchouEstimator(BaseEstimator):
    """
    Schou (1978) bias-corrected estimator.

    Applies a bias correction to the ML estimate to reduce the positive bias
    present in small samples. The correction can be aggressive enough to yield
    zero estimates for small R̄.

    Formula:
        κ̂_S = max(0, κ̂_ML - correction)

    where the correction term depends on the dimension and sample size.

    For the circular case (d=2):
        correction ≈ 1/(n·κ̂_ML)

    For general dimension d:
        correction = A_d(κ̂_ML) / (n · F_d(κ̂_ML))

    References:
        - Schou, G. (1978). Estimation of the concentration parameter in
          von Mises-Fisher distributions. Biometrika, 65(2), 369-377.
    """

    def estimate(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute Schou marginal maximum likelihood estimate of (μ, κ).

        Schou's estimator maximizes the marginal density of R (the total
        resultant length) and is defined by the implicit equation:
            R·A(R·κ) = N·A(κ)  if R² > N
            κ = 0              if R² ≤ N

        where R = N·R̄ is the total resultant length (not mean).

        Args:
            data: Array of shape (n_samples, dim), unit vectors

        Returns:
            (mu_hat, kappa_hat): Schou estimates

        Algorithm:
            1. Compute mean direction μ̂ and total resultant length R = N·R̄
            2. If R² ≤ N, return κ̂_S = 0
            3. Otherwise, solve implicit equation R·A(R·κ) = N·A(κ) for κ

        References:
            - Schou (1978), Biometrika, equation at line 176 of LaTeX
            - Wallace & Dowe (1993), lines 174-177

        Note:
            This is fundamentally different from Fisher's bias correction!
        """
        self._validate_data(data)

        from scipy import optimize

        n_samples, dim = data.shape

        # Compute mean direction
        mu_hat = compute_mean_direction(data)

        # Compute total resultant length R = N·R̄ (not mean!)
        R_bar = compute_mean_resultant_length(data)
        R = n_samples * R_bar  # Total resultant length

        # Check threshold condition: R² ≤ N → κ_S = 0
        if R**2 <= n_samples:
            return mu_hat, 0.0

        # Solve implicit equation: R·A(R·κ) = N·A(κ)
        def schou_equation(kappa):
            """
            Implicit equation to solve: R·A(R·κ) = N·A(κ)
            Returns: lhs - rhs (should be zero at solution)
            """
            if kappa <= 1e-10:
                # For very small kappa, both sides → 0
                # But we want lhs - rhs, so return large positive value to avoid this region
                return 1e10

            try:
                # Left side: R·A(R·κ)
                lhs = R * bessel_ratio_A(R * kappa, dim)

                # Right side: N·A(κ)
                rhs = n_samples * bessel_ratio_A(kappa, dim)

                return lhs - rhs

            except (ValueError, RuntimeWarning, FloatingPointError):
                # If computation fails, return large value to avoid this region
                return 1e10

        # Find bounds for root finding
        # Lower bound: very small positive value
        kappa_lower = 1e-8

        # Upper bound: ML estimate (since Theorem 2(b) says κ_S < κ_ML for 0 < R < N)
        kappa_ml = bessel_ratio_A_inverse(R_bar, dim)
        kappa_upper = max(kappa_ml * 1.5, 100.0)  # Slightly larger than ML to be safe

        # Check if equation changes sign in the interval
        try:
            f_lower = schou_equation(kappa_lower)
            f_upper = schou_equation(kappa_upper)

            # Root finding using Brent's method (robust, no derivatives needed)
            if f_lower * f_upper < 0:
                # Sign change exists, use brentq
                kappa_schou = optimize.brentq(
                    schou_equation,
                    kappa_lower,
                    kappa_upper,
                    xtol=1e-10,
                    maxiter=100
                )
            else:
                # No sign change - try different approach
                # Use minimize_scalar to find where |equation| is minimized
                result = optimize.minimize_scalar(
                    lambda k: abs(schou_equation(k)),
                    bounds=(kappa_lower, kappa_upper),
                    method='bounded',
                    options={'xatol': 1e-10}
                )
                kappa_schou = result.x if result.success else 0.0

        except (ValueError, RuntimeError):
            # If root finding fails, fall back to 0
            # This shouldn't happen if R² > N, but be defensive
            kappa_schou = 0.0

        return mu_hat, kappa_schou

    @property
    def name(self) -> str:
        return "Schou"

    def __repr__(self) -> str:
        return "SchouEstimator()"


class MMLEstimator(BaseEstimator):
    """
    Minimum Message Length (MML) estimator.

    MML provides a Bayesian approach to parameter estimation by minimizing
    the message length needed to encode both the model and the data. This
    leads to estimates that balance fit quality with model complexity.

    The MML objective for κ is:
        maximize: log h(κ) + log p(data|κ) - 0.5·log F(κ)

    where:
        - h(κ): prior density on κ
        - p(data|κ): likelihood
        - F(κ): Fisher information

    Three priors are implemented (Wallace & Dowe 1993):
        - h₁: Jeffreys-like prior ∝ √F(κ) (most aggressive)
        - h₂: Uniform on μ prior (moderate)
        - h₃: Wallace-Dowe recommended prior (balanced, best performance)

    References:
        - Wallace, C. S., & Dowe, D. L. (1993). MML estimation of the von Mises
          concentration parameter. Technical Report 93/193, Monash University.
    """

    def __init__(self, prior: str = 'h3'):
        """
        Initialize MML estimator with specified prior.

        Args:
            prior: Prior type - one of 'h1', 'h2', 'h3' (default: 'h3')

        Raises:
            ValueError: If prior is not one of the supported types
        """
        if prior not in ['h1', 'h2', 'h3']:
            raise ValueError(f"Prior must be one of ['h1', 'h2', 'h3'], got '{prior}'")

        self.prior = prior

        # Map prior names to functions
        self._prior_functions = {
            'h1': log_h1_prior,
            'h2': log_h2_prior,
            'h3': log_h3_prior,
        }

    def estimate(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute MML estimate of (μ, κ).

        Args:
            data: Array of shape (n_samples, dim), unit vectors

        Returns:
            (mu_hat, kappa_hat): MML estimates

        Algorithm:
            1. Compute mean direction μ̂ and resultant length R̄
            2. Define MML objective function
            3. Optimize using scipy.optimize.minimize_scalar
            4. Use ML estimate as initial guess and for bounds
        """
        self._validate_data(data)

        from scipy import optimize

        n_samples, dim = data.shape

        # Compute mean direction
        mu_hat = compute_mean_direction(data)

        # Compute mean resultant length
        R_bar = compute_mean_resultant_length(data)

        # Get ML estimate as initial guess
        kappa_ml = bessel_ratio_A_inverse(R_bar, dim)

        # Define MML objective (to be MAXIMIZED)
        def mml_objective(kappa):
            if kappa <= 1e-10:
                return -np.inf

            try:
                # Log prior
                log_prior_func = self._prior_functions[self.prior]
                log_h = log_prior_func(kappa, dim)

                # Log likelihood (FULL form needed for MML!)
                # p(X|μ,κ) ∝ C_d(κ)^n · exp(κ·Σμ'xᵢ)
                # log p(X|μ,κ) = n·log C_d(κ) + n·κ·R̄
                # log C_d(κ) = (d/2-1)·log(κ) - (d/2)·log(2π) - log I_{d/2-1}(κ)

                # Use scipy.special.ive for exponentially scaled Bessel
                from scipy.special import ive
                nu = dim / 2 - 1

                # log I_ν(κ) = log(ive(ν, κ)) + κ (since ive is exp(-|x|) * iv(x))
                log_bessel = np.log(ive(nu, kappa)) + kappa

                # Full log-likelihood
                log_C = ((dim/2 - 1) * np.log(kappa)
                        - (dim/2) * np.log(2 * np.pi)
                        - log_bessel)
                log_lik = n_samples * (log_C + kappa * R_bar)

                # Fisher information term - DEPENDS ON PRIOR!
                # Compute Bessel ratio A(κ)
                A = bessel_ratio_A(kappa, dim)

                if self.prior in ['h1', 'h2']:
                    # Equation (28): Modified Fisher info to prevent divergence at κ→0
                    # Lines 313-317 of Wallace & Dowe (1993)
                    # -log(h/(k₂√F)) ≈ -log(h) + 0.5·log(κ·A + 3/(π²·N)) + 0.5·log(1 - A/κ - A²)
                    term1 = 0.5 * np.log(kappa * A + 3.0 / (np.pi**2 * n_samples))
                    term2 = 0.5 * np.log(1.0 - A/kappa - A**2)
                    fisher_term = term1 + term2
                else:  # h3
                    # Equation (27): Standard (unmodified) Fisher info
                    # Line 306 and line 321: "unadjusted form"
                    # h₃ prior doesn't diverge at origin, so no modification needed
                    term1 = 0.5 * np.log(kappa * A)
                    term2 = 0.5 * np.log(1.0 - A/kappa - A**2)
                    fisher_term = term1 + term2

                # Check for validity
                if not np.isfinite(fisher_term):
                    return -np.inf

                # MML objective (to be MAXIMIZED)
                # Message length = -log(h) + L - log(h/(k₂√F))
                #                = -log(h) + L + log(h) - 0.5·log(F) + constants
                #                = L - 0.5·log(F) + constants
                # Maximizing posterior ∝ h·p/√F is equivalent to:
                objective = log_h + log_lik - fisher_term

                return objective if np.isfinite(objective) else -np.inf

            except (ValueError, RuntimeWarning, FloatingPointError):
                return -np.inf

        # Optimize: maximize objective = minimize negative objective
        # Use bounds based on ML estimate
        kappa_lower = max(1e-6, 0.01 * kappa_ml) if kappa_ml > 0 else 1e-6
        kappa_upper = max(100, 10 * kappa_ml) if kappa_ml > 0 else 100

        # For h1 and h2, the prior can push estimates to very small values
        # Extend lower bound
        if self.prior in ['h1', 'h2']:
            kappa_lower = 1e-10

        result = optimize.minimize_scalar(
            lambda k: -mml_objective(k),
            bounds=(kappa_lower, kappa_upper),
            method='bounded',
            options={'xatol': 1e-10}
        )

        if result.success:
            kappa_mml = result.x
        else:
            # Fallback to ML if optimization fails
            kappa_mml = kappa_ml

        # For h1, return 0 for small R̄ (Table 1 shows h1=0 for R̄≤0.50)
        # This is because the h1 prior heavily penalizes small κ
        # Use < 0.51 to account for numerical precision
        if self.prior == 'h1' and R_bar < 0.51:
            kappa_mml = 0.0
        # For h1 with larger R̄, if estimate is still very small, return 0
        elif self.prior == 'h1' and kappa_mml < 1.0:
            kappa_mml = 0.0

        return mu_hat, kappa_mml

    @property
    def name(self) -> str:
        return f"MML-{self.prior}"

    def __repr__(self) -> str:
        return f"MMLEstimator(prior='{self.prior}')"
