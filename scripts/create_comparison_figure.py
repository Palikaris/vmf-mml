#!/usr/bin/env python3
"""
Create publication-ready comparison figure of vMF estimators.

Reproduces Wallace & Dowe (1993) Figure 1 showing estimates of κ
as a function of R̄ for N=16.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vmf_estimation.estimators import MLEstimator, SchouEstimator, MMLEstimator
from vmf_estimation.core import bessel_ratio_A_inverse


def create_data_with_exact_R(R_bar, n, dim):
    """Create data with exact R̄ for testing"""
    if dim == 2:
        theta = np.arccos(np.clip(R_bar, 0, 1))
        data = np.zeros((n, 2))
        n_half = n // 2
        data[:n_half] = np.array([np.cos(theta), np.sin(theta)])
        data[n_half:] = np.array([np.cos(theta), -np.sin(theta)])
        return data
    raise NotImplementedError()


def main():
    """Create comparison figure."""
    print("Creating publication-ready comparison figure...")
    print()

    # Sample size
    N = 16

    # R̄ values to evaluate
    R_bar_values = np.linspace(0.01, 0.99, 100)

    # Initialize estimators
    ml_est = MLEstimator()
    schou_est = SchouEstimator()
    mml_h1 = MMLEstimator(prior='h1')
    mml_h2 = MMLEstimator(prior='h2')
    mml_h3 = MMLEstimator(prior='h3')

    # Compute estimates for each R̄
    kappa_ml = np.zeros_like(R_bar_values)
    kappa_schou = np.zeros_like(R_bar_values)
    kappa_h1 = np.zeros_like(R_bar_values)
    kappa_h2 = np.zeros_like(R_bar_values)
    kappa_h3 = np.zeros_like(R_bar_values)

    print("Computing estimates...")
    for i, R_bar in enumerate(R_bar_values):
        if (i+1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(R_bar_values)}")

        # Generate deterministic data
        data = create_data_with_exact_R(R_bar, N, 2)

        # Compute estimates
        _, kappa_ml[i] = ml_est.estimate(data)
        _, kappa_schou[i] = schou_est.estimate(data)
        _, kappa_h1[i] = mml_h1.estimate(data)
        _, kappa_h2[i] = mml_h2.estimate(data)
        _, kappa_h3[i] = mml_h3.estimate(data)

    # Create figure
    print("Creating figure...")
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot curves
    ax.plot(R_bar_values, kappa_ml, 'k-', linewidth=2, label='ML', zorder=5)
    ax.plot(R_bar_values, kappa_schou, 'b--', linewidth=2, label='Schou', zorder=4)
    ax.plot(R_bar_values, kappa_h1, 'r:', linewidth=2, label='MML-h₁', zorder=3)
    ax.plot(R_bar_values, kappa_h2, 'g-.', linewidth=2, label='MML-h₂', zorder=2)
    ax.plot(R_bar_values, kappa_h3, 'm-', linewidth=2, label='MML-h₃', zorder=1, alpha=0.8)

    # Formatting
    ax.set_xlabel('Mean Resultant Length (R̄)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Estimated κ', fontsize=14, fontweight='bold')
    ax.set_title(f'Comparison of vMF Estimators (N={N})',
                 fontsize=16, fontweight='bold', pad=20)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Legend
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)

    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(kappa_ml) * 1.1)

    # Add annotations for key features
    # Annotation 1: h₁ = 0 for small R̄
    idx_zero = np.where(R_bar_values > 0.5)[0][0]
    ax.annotate('h₁ = 0 for R̄ < 0.5',
                xy=(R_bar_values[idx_zero-5], kappa_h1[idx_zero-5]),
                xytext=(0.3, max(kappa_ml)*0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=11, color='red')

    # Annotation 2: Schou = 0 for small R̄
    idx_schou = np.where(kappa_schou > 0)[0][0]
    ax.annotate(f'Schou = 0 for R̄ ≤ {R_bar_values[idx_schou-1]:.2f}',
                xy=(R_bar_values[idx_schou-1], 0),
                xytext=(0.15, max(kappa_ml)*0.2),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                fontsize=11, color='blue')

    # Annotation 3: h₃ recommended
    idx_mid = len(R_bar_values) // 2
    ax.annotate('h₃ recommended\n(Wallace & Dowe 1993)',
                xy=(R_bar_values[idx_mid], kappa_h3[idx_mid]),
                xytext=(0.7, max(kappa_ml)*0.15),
                arrowprops=dict(arrowstyle='->', color='magenta', lw=1.5),
                fontsize=11, color='magenta',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3))

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent.parent / "figures" / "estimator_comparison.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    # Also save as PDF for publication
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")

    # Show plot
    # plt.show()

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    # Find crossover points
    print("\nKey Features:")
    print("-" * 70)

    # Where h1 becomes non-zero
    idx_h1_nonzero = np.where(kappa_h1 > 0)[0][0]
    print(f"h₁ becomes non-zero at R̄ ≈ {R_bar_values[idx_h1_nonzero]:.3f}")

    # Where Schou becomes non-zero
    idx_schou_nonzero = np.where(kappa_schou > 0)[0][0]
    print(f"Schou becomes non-zero at R̄ ≈ {R_bar_values[idx_schou_nonzero]:.3f}")

    # Maximum values
    print(f"\nMaximum estimates at R̄=0.99:")
    print(f"  ML:     {kappa_ml[-1]:.2f}")
    print(f"  Schou:  {kappa_schou[-1]:.2f}")
    print(f"  MML-h₁: {kappa_h1[-1]:.2f}")
    print(f"  MML-h₂: {kappa_h2[-1]:.2f}")
    print(f"  MML-h₃: {kappa_h3[-1]:.2f}")

    # Ordering
    print(f"\nTypical ordering (moderate R̄=0.5):")
    idx_mid = np.argmin(np.abs(R_bar_values - 0.5))
    estimates_mid = {
        'ML': kappa_ml[idx_mid],
        'Schou': kappa_schou[idx_mid],
        'h₁': kappa_h1[idx_mid],
        'h₂': kappa_h2[idx_mid],
        'h₃': kappa_h3[idx_mid]
    }
    sorted_est = sorted(estimates_mid.items(), key=lambda x: x[1], reverse=True)
    for name, val in sorted_est:
        print(f"  {name:<8} {val:.4f}")

    print("=" * 70)
    print("\nFigure creation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
