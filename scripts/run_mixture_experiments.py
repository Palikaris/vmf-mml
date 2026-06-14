"""
Simulated data experiments for vMF mixture model selection.

Generates K-component vMF mixtures with known K and tests whether
AIC, BIC, and MML select the correct number of components.
Also computes KL divergence from the true to the inferred mixture
(requested by David Dowe).

Results are saved to figures/model_selection_simulated.{pdf,png} and
a summary table is printed to stdout.

Decision log:
  D5: M-step uses MML-h3
  D3: MML multinomial estimate
  D4: Unary code for K
"""

import os
import sys
import time
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vmf_estimation.data import generate_vmf_samples
from vmf_estimation.model_selection import select_k
from vmf_estimation.kl import kl_mixture_mc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCENARIOS = [
    # (true K, dim, kappa, N) — well-separated clusters
    (2, 2, 5.0, 100),
    (2, 2, 5.0, 300),
    (3, 2, 5.0, 150),
    (3, 2, 5.0, 500),
    (2, 3, 4.0, 100),
    (3, 3, 4.0, 200),
    (2, 10, 8.0, 200),
    (3, 10, 8.0, 300),
    # Harder: low kappa (more overlap)
    (2, 2, 2.0, 300),
    (3, 2, 2.0, 500),
]

N_TRIALS = 20
K_RANGE = list(range(1, 7))
MOVMF_KWARGS = {"n_init": 2, "max_iter": 50, "tol": 1e-4}
N_MC_KL = 5000   # Monte Carlo samples for KL estimation


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_mixture(true_K, dim, kappa, N, rng):
    """Return (X, true_weights, true_means, true_kappas) for a balanced vMF mixture."""
    angles = np.linspace(0, 2 * math.pi, true_K, endpoint=False)
    if dim >= 2:
        means = np.zeros((true_K, dim))
        means[:, 0] = np.cos(angles)
        means[:, 1] = np.sin(angles)
        # Normalise to unit vectors (already unit for dim=2; for dim>2 need renorm)
        means = means / np.linalg.norm(means, axis=1, keepdims=True)

    n_per = N // true_K
    parts = []
    for i in range(true_K):
        seed = int(rng.randint(0, 10 ** 6))
        parts.append(generate_vmf_samples(means[i], kappa, n_per, random_state=seed))

    true_weights = np.full(true_K, 1.0 / true_K)
    true_kappas = np.full(true_K, kappa)
    return np.vstack(parts), true_weights, means, true_kappas


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiments():
    records = []

    for (true_K, dim, kappa, N) in SCENARIOS:
        rng = np.random.RandomState(42)
        counts = {"aic": defaultdict(int), "bic": defaultdict(int), "mml": defaultdict(int)}
        kl_acc = {"aic": 0.0, "bic": 0.0, "mml": 0.0}
        t0 = time.time()

        for trial in range(N_TRIALS):
            X, true_w, true_means, true_kappas = make_mixture(true_K, dim, kappa, N, rng)

            # Run all three criteria with the SAME fitted models (shared computation)
            _, results = select_k(
                X, k_range=K_RANGE, method="mml",
                movmf_kwargs=dict(random_state=trial, **MOVMF_KWARGS),
            )

            for method in ["aic", "bic", "mml"]:
                best_k = min(results, key=lambda k: results[k][method])
                counts[method][best_k] += 1
                # KL divergence from true to inferred best-K model
                best_model = results[best_k]["model"]
                kl = kl_mixture_mc(
                    true_w, true_means, true_kappas, best_model,
                    n_mc=N_MC_KL, random_state=trial,
                )
                kl_acc[method] += kl

        elapsed = time.time() - t0

        n = N_TRIALS
        row = {
            "true_K": true_K, "dim": dim, "kappa": kappa, "N": N,
            "aic_correct": counts["aic"][true_K] / n,
            "bic_correct": counts["bic"][true_K] / n,
            "mml_correct": counts["mml"][true_K] / n,
            "aic_kl": kl_acc["aic"] / n,
            "bic_kl": kl_acc["bic"] / n,
            "mml_kl": kl_acc["mml"] / n,
            "elapsed": elapsed,
        }
        records.append(row)

        print(
            f"K={true_K}, d={dim:2d}, κ={kappa:.1f}, N={N:4d} | "
            f"Acc: AIC={row['aic_correct']:.2f} BIC={row['bic_correct']:.2f} MML={row['mml_correct']:.2f} | "
            f"KL: AIC={row['aic_kl']:.3f} BIC={row['bic_kl']:.3f} MML={row['mml_kl']:.3f} "
            f"({elapsed:.1f}s)"
        )

    return records


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _bold(val: str, is_best: bool) -> str:
    """Wrap in ANSI bold if best."""
    return f"\033[1m{val}\033[0m" if is_best else val


def print_summary_table(records):
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'K':>3} {'d':>3} {'κ':>5} {'N':>5} "
          f"| {'Acc-AIC':>8} {'Acc-BIC':>8} {'Acc-MML':>8} "
          f"| {'KL-AIC':>8} {'KL-BIC':>8} {'KL-MML':>8}")
    print("-" * 100)

    acc_avgs = {"aic": 0.0, "bic": 0.0, "mml": 0.0}
    kl_avgs  = {"aic": 0.0, "bic": 0.0, "mml": 0.0}

    for r in records:
        accs = {"aic": r["aic_correct"], "bic": r["bic_correct"], "mml": r["mml_correct"]}
        kls  = {"aic": r["aic_kl"],     "bic": r["bic_kl"],     "mml": r["mml_kl"]}
        best_acc = max(accs, key=accs.get)
        best_kl  = min(kls,  key=kls.get)   # lower KL = better

        def fmt_acc(m): return _bold(f"{accs[m]:.2f}", m == best_acc)
        def fmt_kl(m):  return _bold(f"{kls[m]:.3f}",  m == best_kl)

        print(
            f"{r['true_K']:>3} {r['dim']:>3} {r['kappa']:>5.1f} {r['N']:>5} "
            f"| {fmt_acc('aic'):>8} {fmt_acc('bic'):>8} {fmt_acc('mml'):>8} "
            f"| {fmt_kl('aic'):>8} {fmt_kl('bic'):>8} {fmt_kl('mml'):>8}"
        )
        for m in ["aic", "bic", "mml"]:
            acc_avgs[m] += accs[m]
            kl_avgs[m]  += kls[m]

    n = len(records)
    for m in ["aic", "bic", "mml"]:
        acc_avgs[m] /= n
        kl_avgs[m]  /= n

    best_avg_acc = max(acc_avgs, key=acc_avgs.get)
    best_avg_kl  = min(kl_avgs,  key=kl_avgs.get)

    def fmt_avg_acc(m): return _bold(f"{acc_avgs[m]:.2f}", m == best_avg_acc)
    def fmt_avg_kl(m):  return _bold(f"{kl_avgs[m]:.3f}",  m == best_avg_kl)

    print("-" * 100)
    print(
        f"{'Mean':>17}      "
        f"| {fmt_avg_acc('aic'):>8} {fmt_avg_acc('bic'):>8} {fmt_avg_acc('mml'):>8} "
        f"| {fmt_avg_kl('aic'):>8} {fmt_avg_kl('bic'):>8} {fmt_avg_kl('mml'):>8}"
    )
    print()
    return {"acc_avgs": acc_avgs, "kl_avgs": kl_avgs}


# ---------------------------------------------------------------------------
# Plotting — two panels: accuracy + KL
# ---------------------------------------------------------------------------

def plot_results(records):
    os.makedirs("figures", exist_ok=True)
    n = len(records)
    labels = [
        f"K={r['true_K']}, d={r['dim']},\nκ={r['kappa']:.1f}, N={r['N']}"
        for r in records
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # ---- Panel A: accuracy ----
    ax = axes[0]
    x = np.arange(n)
    w = 0.25
    ax.bar(x - w, [r["aic_correct"] for r in records], w, label="AIC", color="#4C72B0", alpha=0.85)
    ax.bar(x,     [r["bic_correct"] for r in records], w, label="BIC", color="#DD8452", alpha=0.85)
    ax.bar(x + w, [r["mml_correct"] for r in records], w, label="MML", color="#55A868", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Fraction of correct K selections")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model selection accuracy")
    ax.legend(); ax.axhline(0.8, color="gray", linestyle="--", lw=0.8, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    # ---- Panel B: KL divergence ----
    ax = axes[1]
    ax.bar(x - w, [r["aic_kl"] for r in records], w, label="AIC", color="#4C72B0", alpha=0.85)
    ax.bar(x,     [r["bic_kl"] for r in records], w, label="BIC", color="#DD8452", alpha=0.85)
    ax.bar(x + w, [r["mml_kl"] for r in records], w, label="MML", color="#55A868", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("KL(true ‖ inferred) [nits]")
    ax.set_title("KL divergence (true ‖ inferred mixture)\nlower is better")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"vMF mixture model selection on simulated data  "
        f"({N_TRIALS} trials / scenario, K candidates = {K_RANGE})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig("figures/model_selection_simulated.pdf", dpi=150)
    fig.savefig("figures/model_selection_simulated.png", dpi=150)
    print("Saved figures/model_selection_simulated.{pdf,png}")
    plt.close(fig)


if __name__ == "__main__":
    print(f"Running {N_TRIALS} trials × {len(SCENARIOS)} scenarios...")
    records = run_experiments()
    summary = print_summary_table(records)
    plot_results(records)
    print(
        f"Overall averages — "
        f"Acc: AIC={summary['acc_avgs']['aic']:.2f} "
        f"BIC={summary['acc_avgs']['bic']:.2f} "
        f"MML={summary['acc_avgs']['mml']:.2f}  |  "
        f"KL: AIC={summary['kl_avgs']['aic']:.3f} "
        f"BIC={summary['kl_avgs']['bic']:.3f} "
        f"MML={summary['kl_avgs']['mml']:.3f}"
    )
