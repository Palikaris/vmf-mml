"""
von Mises-Fisher Parameter Estimation Library

This library implements multiple methods for estimating the concentration parameter κ
of von Mises-Fisher distributions on the unit hypersphere.

Estimators:
    - ML: Maximum Likelihood
    - Schou: Bias-corrected (Schou 1978)
    - MML: Minimum Message Length with three priors (Wallace & Dowe 1993)
      - h1: Jeffreys-like prior
      - h2: Uniform on μ prior
      - h3: Wallace-Dowe recommended prior

References:
    Wallace, C. S., & Dowe, D. L. (1993). MML estimation of the von Mises
    concentration parameter. Technical Report 93/193, Dept. of Computer Science,
    Monash University.
"""

__version__ = "0.1.0"

# Import main classes and functions when they're ready
# from .estimators import MLEstimator, SchouEstimator, MMLEstimator
# from .core import bessel_ratio_A, bessel_ratio_A_inverse, fisher_information
# from .data import generate_vmf_samples, normalize_to_unit_sphere

__all__ = [
    # Core functions
    # "bessel_ratio_A",
    # "bessel_ratio_A_inverse",
    # "fisher_information",
    # Estimators
    # "MLEstimator",
    # "SchouEstimator",
    # "MMLEstimator",
    # Data utilities
    # "generate_vmf_samples",
    # "normalize_to_unit_sphere",
    # Mixture models
    "MovMF",
    # Model selection
    "select_k",
    "mml_message_length",
    "aic",
    "bic",
    # KL divergence
    "kl_vmf",
    "kl_mixture_mc",
]
