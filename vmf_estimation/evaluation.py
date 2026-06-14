"""
Evaluation framework for comparing von Mises-Fisher estimators.

This module provides Monte Carlo simulation tools to evaluate and compare
the performance of different κ estimators in terms of bias, variance, and MSE.

References:
    Wallace, C. S., & Dowe, D. L. (1993). MML estimation of the von Mises
    concentration parameter. Technical Report 93/193, Monash University.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Simple fallback progress indicator
    class tqdm:
        def __init__(self, iterable=None, desc="", leave=True, total=None):
            self.iterable = iterable
            self.desc = desc
            self.total = total
            self.n = 0
        def __iter__(self):
            return iter(self.iterable) if self.iterable else iter([])
        def set_description(self, desc):
            if self.total and HAS_TQDM is False:
                print(f"\r{desc} ({self.n}/{self.total})", end="", flush=True)
        def update(self, n):
            self.n += n
        def close(self):
            if self.total and HAS_TQDM is False:
                print()  # Newline after progress

from .estimators import MLEstimator, SchouEstimator, MMLEstimator
from .data import generate_vmf_samples


@dataclass
class EstimatorResult:
    """Results from evaluating a single estimator."""
    name: str
    estimates: np.ndarray  # Shape: (n_trials,)
    bias: float
    variance: float
    mse: float
    rmse: float
    mean_estimate: float
    std_estimate: float


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo evaluation of multiple estimators."""
    true_kappa: float
    sample_size: int
    n_trials: int
    estimator_results: Dict[str, EstimatorResult]


def evaluate_estimator(
    estimator,
    true_mu: np.ndarray,
    true_kappa: float,
    sample_size: int,
    n_trials: int = 1000,
    random_seed: Optional[int] = None,
    show_progress: bool = True
) -> EstimatorResult:
    """
    Evaluate a single estimator using Monte Carlo simulation.

    Args:
        estimator: Estimator instance (ML, Schou, or MML)
        true_mu: True mean direction (unit vector)
        true_kappa: True concentration parameter
        sample_size: Number of samples per trial
        n_trials: Number of Monte Carlo trials
        random_seed: Random seed for reproducibility
        show_progress: Show progress bar

    Returns:
        EstimatorResult with bias, variance, and MSE statistics
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    estimates = np.zeros(n_trials)

    # Run Monte Carlo trials
    iterator = range(n_trials)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Evaluating {estimator.name}", leave=False)

    for trial in iterator:
        # Generate random sample
        data = generate_vmf_samples(true_mu, true_kappa, sample_size)

        # Estimate κ
        _, kappa_hat = estimator.estimate(data)
        estimates[trial] = kappa_hat

    # Compute statistics
    mean_estimate = np.mean(estimates)
    bias = mean_estimate - true_kappa
    variance = np.var(estimates, ddof=1)  # Sample variance
    mse = bias**2 + variance
    rmse = np.sqrt(mse)
    std_estimate = np.std(estimates, ddof=1)

    return EstimatorResult(
        name=estimator.name,
        estimates=estimates,
        bias=bias,
        variance=variance,
        mse=mse,
        rmse=rmse,
        mean_estimate=mean_estimate,
        std_estimate=std_estimate
    )


def compare_estimators(
    true_kappa: float,
    sample_size: int,
    dim: int = 2,
    n_trials: int = 1000,
    random_seed: Optional[int] = None,
    show_progress: bool = True
) -> MonteCarloResults:
    """
    Compare all estimators (ML, Schou, MML-h1, h2, h3) using Monte Carlo.

    Args:
        true_kappa: True concentration parameter
        sample_size: Number of samples per trial
        dim: Dimension (default: 2 for circular)
        n_trials: Number of Monte Carlo trials
        random_seed: Random seed for reproducibility
        show_progress: Show progress bar

    Returns:
        MonteCarloResults containing results for all estimators
    """
    # True mean direction (pointing in first coordinate direction)
    true_mu = np.zeros(dim)
    true_mu[0] = 1.0

    # Initialize estimators
    estimators = [
        MLEstimator(),
        SchouEstimator(),
        MMLEstimator(prior='h1'),
        MMLEstimator(prior='h2'),
        MMLEstimator(prior='h3'),
    ]

    # Evaluate each estimator
    results = {}
    for estimator in estimators:
        result = evaluate_estimator(
            estimator,
            true_mu,
            true_kappa,
            sample_size,
            n_trials,
            random_seed,
            show_progress
        )
        results[estimator.name] = result

    return MonteCarloResults(
        true_kappa=true_kappa,
        sample_size=sample_size,
        n_trials=n_trials,
        estimator_results=results
    )


def evaluate_over_grid(
    kappa_values: List[float],
    sample_sizes: List[int],
    dim: int = 2,
    n_trials: int = 1000,
    random_seed: Optional[int] = None,
    show_progress: bool = True
) -> Dict[Tuple[float, int], MonteCarloResults]:
    """
    Evaluate estimators over a grid of (κ, N) values.

    Args:
        kappa_values: List of true κ values to test
        sample_sizes: List of sample sizes to test
        dim: Dimension
        n_trials: Number of Monte Carlo trials per configuration
        random_seed: Random seed
        show_progress: Show progress bar

    Returns:
        Dictionary mapping (κ, N) to MonteCarloResults
    """
    results = {}

    total = len(kappa_values) * len(sample_sizes)
    if show_progress:
        pbar = tqdm(total=total, desc="Grid evaluation")

    for kappa in kappa_values:
        for n in sample_sizes:
            if show_progress:
                pbar.set_description(f"κ={kappa:.2f}, N={n}")

            result = compare_estimators(
                true_kappa=kappa,
                sample_size=n,
                dim=dim,
                n_trials=n_trials,
                random_seed=random_seed,
                show_progress=False  # Don't show inner progress
            )

            results[(kappa, n)] = result

            if show_progress:
                pbar.update(1)

    if show_progress:
        pbar.close()

    return results


def print_comparison_table(mc_results: MonteCarloResults) -> None:
    """
    Print a formatted comparison table of estimator performance.

    Args:
        mc_results: Monte Carlo results to display
    """
    print("=" * 90)
    print(f"Monte Carlo Comparison: κ={mc_results.true_kappa:.2f}, "
          f"N={mc_results.sample_size}, trials={mc_results.n_trials}")
    print("=" * 90)
    print(f"{'Estimator':<12} {'Mean':<10} {'Bias':<10} {'Std Dev':<10} "
          f"{'RMSE':<10} {'MSE':<10}")
    print("-" * 90)

    for name, result in mc_results.estimator_results.items():
        print(f"{name:<12} "
              f"{result.mean_estimate:<10.4f} "
              f"{result.bias:<10.4f} "
              f"{result.std_estimate:<10.4f} "
              f"{result.rmse:<10.4f} "
              f"{result.mse:<10.4f}")

    print("=" * 90)
    print(f"True κ: {mc_results.true_kappa:.4f}")
    print()


def compute_relative_efficiency(
    mc_results: MonteCarloResults,
    baseline: str = "ML"
) -> Dict[str, float]:
    """
    Compute relative efficiency of each estimator compared to baseline.

    Relative efficiency = MSE(baseline) / MSE(estimator)
    Values > 1 mean the estimator is more efficient than baseline.

    Args:
        mc_results: Monte Carlo results
        baseline: Name of baseline estimator (default: "ML")

    Returns:
        Dictionary mapping estimator name to relative efficiency
    """
    baseline_mse = mc_results.estimator_results[baseline].mse

    efficiencies = {}
    for name, result in mc_results.estimator_results.items():
        if result.mse > 0:
            efficiencies[name] = baseline_mse / result.mse
        else:
            efficiencies[name] = np.inf

    return efficiencies
