#!/usr/bin/env python3
"""
Run Monte Carlo evaluation of vMF estimators.

This script evaluates and compares all five estimators (ML, Schou, MML-h1, h2, h3)
using Monte Carlo simulation across different κ values and sample sizes.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vmf_estimation.evaluation import (
    compare_estimators,
    evaluate_over_grid,
    print_comparison_table,
    compute_relative_efficiency
)


def main():
    """Run Monte Carlo evaluations."""
    print("\n" + "=" * 90)
    print("MONTE CARLO EVALUATION OF VMF ESTIMATORS")
    print("=" * 90)
    print()

    # Single configuration test
    print("Test 1: Single Configuration (κ=2.0, N=16)")
    print("-" * 90)

    results = compare_estimators(
        true_kappa=2.0,
        sample_size=16,
        n_trials=1000,
        random_seed=42,
        show_progress=True
    )

    print_comparison_table(results)

    # Compute relative efficiencies
    efficiencies = compute_relative_efficiency(results, baseline="ML")
    print("Relative Efficiency (vs ML):")
    print("-" * 90)
    for name, eff in efficiencies.items():
        if eff < np.inf:
            print(f"{name:<12} {eff:.4f}  {'[Better than ML]' if eff > 1 else '[Worse than ML]'}")
        else:
            print(f"{name:<12} Perfect (MSE=0)")
    print()

    # Grid evaluation
    print("\n" + "=" * 90)
    print("Test 2: Grid Evaluation")
    print("=" * 90)
    print()

    kappa_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    sample_sizes = [8, 16, 32, 64]

    print(f"κ values: {kappa_values}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Total configurations: {len(kappa_values) * len(sample_sizes)}")
    print(f"Trials per configuration: 1000")
    print()

    grid_results = evaluate_over_grid(
        kappa_values=kappa_values,
        sample_sizes=sample_sizes,
        n_trials=1000,
        random_seed=42,
        show_progress=True
    )

    # Print summary for each configuration
    print("\n" + "=" * 90)
    print("SUMMARY: Best Estimator by Configuration")
    print("=" * 90)
    print(f"{'κ':<8} {'N':<8} {'Best (MSE)':<20} {'Best (Bias)':<20}")
    print("-" * 90)

    for (kappa, n), result in sorted(grid_results.items()):
        # Find best by MSE
        best_mse = min(result.estimator_results.items(),
                       key=lambda x: x[1].mse)

        # Find best by absolute bias
        best_bias = min(result.estimator_results.items(),
                        key=lambda x: abs(x[1].bias))

        print(f"{kappa:<8.2f} {n:<8} "
              f"{best_mse[0]:<20} {best_bias[0]:<20}")

    print("=" * 90)
    print()

    # Key findings
    print("=" * 90)
    print("KEY FINDINGS")
    print("=" * 90)
    print()

    # Analyze small vs large N
    small_n_results = [r for (k, n), r in grid_results.items() if n == 8]
    large_n_results = [r for (k, n), r in grid_results.items() if n == 64]

    print("Small N (N=8) - Average relative efficiency vs ML:")
    print("-" * 90)
    for est_name in ["Schou", "MML-h1", "MML-h2", "MML-h3"]:
        efficiencies = []
        for result in small_n_results:
            ml_mse = result.estimator_results["ML"].mse
            est_mse = result.estimator_results[est_name].mse
            if est_mse > 0:
                efficiencies.append(ml_mse / est_mse)

        avg_eff = np.mean(efficiencies)
        print(f"{est_name:<12} {avg_eff:.4f}")

    print()
    print("Large N (N=64) - Average relative efficiency vs ML:")
    print("-" * 90)
    for est_name in ["Schou", "MML-h1", "MML-h2", "MML-h3"]:
        efficiencies = []
        for result in large_n_results:
            ml_mse = result.estimator_results["ML"].mse
            est_mse = result.estimator_results[est_name].mse
            if est_mse > 0:
                efficiencies.append(ml_mse / est_mse)

        avg_eff = np.mean(efficiencies)
        print(f"{est_name:<12} {avg_eff:.4f}")

    print()
    print("=" * 90)
    print("Evaluation complete! Results saved for visualization.")
    print("=" * 90)


if __name__ == "__main__":
    main()
