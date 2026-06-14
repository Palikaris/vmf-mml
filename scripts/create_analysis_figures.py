#!/usr/bin/env python3
"""
Create comprehensive analysis figures for vMF estimators.

Creates multiple publication-ready figures:
1. Bias vs κ for different sample sizes
2. Variance vs κ for different sample sizes
3. MSE vs κ for different sample sizes
4. Relative efficiency vs sample size
5. Bias-variance decomposition
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vmf_estimation.evaluation import compare_estimators


def create_bias_variance_figures():
    """Create comprehensive bias and variance analysis figures."""
    print("Creating analysis figures...")
    print("=" * 70)

    # Parameter grid
    kappa_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    sample_sizes = [8, 16, 32, 64]
    n_trials = 1000

    # Collect results
    print("\nRunning Monte Carlo simulations...")
    print(f"κ values: {kappa_values}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Trials per configuration: {n_trials}")
    print()

    results = {}
    total = len(kappa_values) * len(sample_sizes)
    count = 0

    for kappa in kappa_values:
        for N in sample_sizes:
            count += 1
            print(f"[{count}/{total}] κ={kappa:.1f}, N={N}")

            result = compare_estimators(
                true_kappa=kappa,
                sample_size=N,
                n_trials=n_trials,
                random_seed=42,
                show_progress=False
            )
            results[(kappa, N)] = result

    print("\nSimulations complete! Creating figures...")
    print()

    # Create figures directory
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    # =========================================================================
    # Figure 1: Bias vs κ for different sample sizes
    # =========================================================================
    print("Creating Figure 1: Bias vs κ...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Bias Analysis: Estimator Performance vs κ',
                 fontsize=16, fontweight='bold', y=0.995)

    for idx, N in enumerate(sample_sizes):
        ax = axes[idx // 2, idx % 2]

        kappas = []
        bias_ml = []
        bias_schou = []
        bias_h1 = []
        bias_h2 = []
        bias_h3 = []

        for kappa in kappa_values:
            r = results[(kappa, N)]
            kappas.append(kappa)
            bias_ml.append(r.estimator_results["ML"].bias)
            bias_schou.append(r.estimator_results["Schou"].bias)
            bias_h1.append(r.estimator_results["MML-h1"].bias)
            bias_h2.append(r.estimator_results["MML-h2"].bias)
            bias_h3.append(r.estimator_results["MML-h3"].bias)

        ax.plot(kappas, bias_ml, 'k-', linewidth=2, marker='o', label='ML')
        ax.plot(kappas, bias_schou, 'b--', linewidth=2, marker='s', label='Schou')
        ax.plot(kappas, bias_h1, 'r:', linewidth=2, marker='^', label='MML-h₁')
        ax.plot(kappas, bias_h2, 'g-.', linewidth=2, marker='v', label='MML-h₂')
        ax.plot(kappas, bias_h3, 'm-', linewidth=2, marker='d', label='MML-h₃', alpha=0.8)

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('True κ', fontsize=11, fontweight='bold')
        ax.set_ylabel('Bias', fontsize=11, fontweight='bold')
        ax.set_title(f'N = {N}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.set_xscale('log')

    plt.tight_layout()
    bias_path = figures_dir / "bias_analysis.png"
    plt.savefig(bias_path, dpi=300, bbox_inches='tight')
    plt.savefig(bias_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"  Saved: {bias_path}")
    plt.close()

    # =========================================================================
    # Figure 2: MSE vs κ for different sample sizes
    # =========================================================================
    print("Creating Figure 2: MSE vs κ...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Mean Squared Error vs κ',
                 fontsize=16, fontweight='bold', y=0.995)

    for idx, N in enumerate(sample_sizes):
        ax = axes[idx // 2, idx % 2]

        kappas = []
        mse_ml = []
        mse_schou = []
        mse_h1 = []
        mse_h2 = []
        mse_h3 = []

        for kappa in kappa_values:
            r = results[(kappa, N)]
            kappas.append(kappa)
            mse_ml.append(r.estimator_results["ML"].mse)
            mse_schou.append(r.estimator_results["Schou"].mse)
            mse_h1.append(r.estimator_results["MML-h1"].mse)
            mse_h2.append(r.estimator_results["MML-h2"].mse)
            mse_h3.append(r.estimator_results["MML-h3"].mse)

        ax.plot(kappas, mse_ml, 'k-', linewidth=2, marker='o', label='ML')
        ax.plot(kappas, mse_schou, 'b--', linewidth=2, marker='s', label='Schou')
        ax.plot(kappas, mse_h1, 'r:', linewidth=2, marker='^', label='MML-h₁')
        ax.plot(kappas, mse_h2, 'g-.', linewidth=2, marker='v', label='MML-h₂')
        ax.plot(kappas, mse_h3, 'm-', linewidth=2, marker='d', label='MML-h₃', alpha=0.8)

        ax.set_xlabel('True κ', fontsize=11, fontweight='bold')
        ax.set_ylabel('MSE', fontsize=11, fontweight='bold')
        ax.set_title(f'N = {N}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.set_xscale('log')
        ax.set_yscale('log')

    plt.tight_layout()
    mse_path = figures_dir / "mse_analysis.png"
    plt.savefig(mse_path, dpi=300, bbox_inches='tight')
    plt.savefig(mse_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"  Saved: {mse_path}")
    plt.close()

    # =========================================================================
    # Figure 3: Relative efficiency vs N
    # =========================================================================
    print("Creating Figure 3: Relative efficiency vs N...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Relative Efficiency vs Sample Size (Baseline: ML)',
                 fontsize=16, fontweight='bold', y=0.995)

    axes = axes.flatten()

    for idx, kappa in enumerate(kappa_values):
        ax = axes[idx]

        ns = []
        eff_schou = []
        eff_h1 = []
        eff_h2 = []
        eff_h3 = []

        for N in sample_sizes:
            r = results[(kappa, N)]
            ns.append(N)

            ml_mse = r.estimator_results["ML"].mse
            eff_schou.append(ml_mse / r.estimator_results["Schou"].mse)
            eff_h1.append(ml_mse / r.estimator_results["MML-h1"].mse)
            eff_h2.append(ml_mse / r.estimator_results["MML-h2"].mse)
            eff_h3.append(ml_mse / r.estimator_results["MML-h3"].mse)

        ax.plot(ns, eff_schou, 'b--', linewidth=2, marker='s', label='Schou')
        ax.plot(ns, eff_h1, 'r:', linewidth=2, marker='^', label='MML-h₁')
        ax.plot(ns, eff_h2, 'g-.', linewidth=2, marker='v', label='MML-h₂')
        ax.plot(ns, eff_h3, 'm-', linewidth=2, marker='d', label='MML-h₃', alpha=0.8)

        ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1.5,
                   alpha=0.7, label='ML (baseline)')
        ax.set_xlabel('Sample Size (N)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Relative Efficiency', fontsize=11, fontweight='bold')
        ax.set_title(f'κ = {kappa}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.set_xscale('log', base=2)

        # Add horizontal line at 1.0 with label
        ax.fill_between(ns, 1.0, 2.0, alpha=0.1, color='green',
                        label='Better than ML' if idx == 0 else '')

    plt.tight_layout()
    eff_path = figures_dir / "efficiency_analysis.png"
    plt.savefig(eff_path, dpi=300, bbox_inches='tight')
    plt.savefig(eff_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"  Saved: {eff_path}")
    plt.close()

    # =========================================================================
    # Figure 4: Bias-Variance Decomposition (for N=16)
    # =========================================================================
    print("Creating Figure 4: Bias-variance decomposition...")

    N = 16
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Bias-Variance-MSE Decomposition (N={N})',
                 fontsize=16, fontweight='bold', y=1.02)

    estimators = ["ML", "Schou", "MML-h1", "MML-h2", "MML-h3"]
    colors = ['black', 'blue', 'red', 'green', 'magenta']

    for est, color in zip(estimators, colors):
        kappas = []
        biases = []
        variances = []
        mses = []

        for kappa in kappa_values:
            r = results[(kappa, N)]
            res = r.estimator_results[est]
            kappas.append(kappa)
            biases.append(res.bias**2)  # Squared bias
            variances.append(res.variance)
            mses.append(res.mse)

        # Plot bias²
        axes[0].plot(kappas, biases, color=color, linewidth=2,
                    marker='o', label=est)

        # Plot variance
        axes[1].plot(kappas, variances, color=color, linewidth=2,
                    marker='s', label=est)

        # Plot MSE
        axes[2].plot(kappas, mses, color=color, linewidth=2,
                    marker='d', label=est)

    axes[0].set_xlabel('True κ', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Bias²', fontsize=12, fontweight='bold')
    axes[0].set_title('Squared Bias', fontsize=13, fontweight='bold')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=9)

    axes[1].set_xlabel('True κ', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Variance', fontsize=12, fontweight='bold')
    axes[1].set_title('Variance', fontsize=13, fontweight='bold')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=9)

    axes[2].set_xlabel('True κ', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('MSE', fontsize=12, fontweight='bold')
    axes[2].set_title('MSE = Bias² + Variance', fontsize=13, fontweight='bold')
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='best', fontsize=9)

    plt.tight_layout()
    decomp_path = figures_dir / "bias_variance_decomposition.png"
    plt.savefig(decomp_path, dpi=300, bbox_inches='tight')
    plt.savefig(decomp_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f"  Saved: {decomp_path}")
    plt.close()

    # =========================================================================
    # Summary statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: Overall Best Estimators")
    print("=" * 70)

    # Average relative efficiency
    print("\nAverage Relative Efficiency (vs ML) across all configurations:")
    print("-" * 70)

    avg_eff = {}
    for est in ["Schou", "MML-h1", "MML-h2", "MML-h3"]:
        effs = []
        for (kappa, N), r in results.items():
            ml_mse = r.estimator_results["ML"].mse
            est_mse = r.estimator_results[est].mse
            if est_mse > 0:
                effs.append(ml_mse / est_mse)
        avg_eff[est] = np.mean(effs)
        print(f"{est:<12} {avg_eff[est]:.4f}  "
              f"{'[BETTER]' if avg_eff[est] > 1 else '[WORSE]'}")

    # Best by configuration
    print("\nBest estimator by (κ, N) configuration (by MSE):")
    print("-" * 70)
    print(f"{'κ':<8} {'N':<8} {'Best':<12} {'MSE':<10} {'vs ML':<10}")
    print("-" * 70)

    for (kappa, N), r in sorted(results.items()):
        best = min(r.estimator_results.items(), key=lambda x: x[1].mse)
        ml_mse = r.estimator_results["ML"].mse
        improvement = (ml_mse / best[1].mse - 1) * 100

        print(f"{kappa:<8.1f} {N:<8} {best[0]:<12} {best[1].mse:<10.4f} "
              f"{improvement:>6.1f}% better")

    print("=" * 70)
    print("\nAll figures saved to figures/ directory!")
    print("=" * 70)


if __name__ == "__main__":
    create_bias_variance_figures()
