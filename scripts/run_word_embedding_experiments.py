"""
Real-data experiments for vMF mixture model selection.

Supports two datasets:
  - 20 Newsgroups (default with --newsgroups flag): TF-IDF unit vectors,
    d=2000, N=2000, 10 ground-truth topic categories.
  - sklearn digits (built-in fallback): 64-dimensional digit images
    L2-normalised to the unit sphere, 10 ground-truth classes.

Results are saved to figures/model_selection_newsgroups.{pdf,png}.
"""

import os
import sys
import time
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sklearn.datasets import load_digits
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from vmf_estimation.mixture import MovMF
from vmf_estimation.model_selection import aic, bic, mml_message_length


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRUE_K = 10    # digits dataset has 10 classes (0-9)
K_RANGE = [2, 4, 6, 8, 10, 12, 14]
MOVMF_KWARGS = {"n_init": 2, "max_iter": 50, "tol": 1e-4, "random_state": 42}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """
    Load digits dataset and L2-normalise to unit sphere.

    This is a structural stand-in for 20 Newsgroups TF-IDF unit vectors.
    See module docstring for explanation.

    Returns
    -------
    X : np.ndarray, shape (N, d)  — unit vectors on S^{d-1}
    y_true : np.ndarray, shape (N,) — ground-truth labels
    """
    print("Loading digits dataset (sklearn built-in)…")
    digits = load_digits()
    X = digits.data.astype(float)
    y = digits.target

    # L2-normalise → unit sphere
    X = normalize(X, norm="l2")
    print(f"  Shape: {X.shape}  (d={X.shape[1]}, N={X.shape[0]}, true K={len(np.unique(y))})")
    return X, y


def load_newsgroups(
    categories=None, max_features=2000, max_docs=2000
):
    """
    Load 20 Newsgroups, vectorise with TF-IDF, and L2-normalise.

    Requires network access (or a pre-cached download).
    Preserved here for when the dataset is available.
    """
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as skl_norm

    default_cats = [
        "rec.sport.hockey", "rec.sport.baseball", "sci.space", "sci.med",
        "talk.politics.guns", "talk.politics.mideast", "comp.graphics",
        "comp.os.ms-windows.misc", "alt.atheism", "soc.religion.christian",
    ]
    cats = categories or default_cats
    data = fetch_20newsgroups(
        subset="all", categories=cats,
        remove=("headers", "footers", "quotes"), random_state=42,
    )
    vectorizer = TfidfVectorizer(
        max_features=max_features, sublinear_tf=True,
        min_df=3, max_df=0.95, stop_words="english",
    )
    X = vectorizer.fit_transform(data.data)
    X = skl_norm(X, norm="l2")
    y = np.array(data.target)

    if max_docs and X.shape[0] > max_docs:
        rng = np.random.RandomState(42)
        keep = []
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            keep.extend(rng.choice(idx, min(max_docs // len(np.unique(y)), len(idx)), replace=False))
        keep = np.array(keep)
        X, y = X[keep], y[keep]
    return X, y


# ---------------------------------------------------------------------------
# Clustering quality
# ---------------------------------------------------------------------------

def clustering_quality(y_true, y_pred):
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")
    N = len(y_true)
    purity = sum(
        np.bincount(y_true[y_pred == k]).max()
        for k in np.unique(y_pred) if (y_pred == k).sum() > 0
    ) / N
    return {"ari": ari, "nmi": nmi, "purity": purity}


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiments(X, y_true, k_range=None):
    N, d = X.shape
    k_range = k_range or K_RANGE
    results = {}
    print(f"\nFitting K = {k_range}  (N={N}, d={d})…")

    for K in k_range:
        t0 = time.time()
        model = MovMF(n_clusters=K, **MOVMF_KWARGS)
        model.fit(X)
        ll = model.log_likelihood(X)
        elapsed = time.time() - t0

        k_params = model.n_params(d)
        aic_score = aic(ll, k_params)
        bic_score = bic(ll, k_params, N)
        mml_score = mml_message_length(model, X)

        labels = model.predict(X)
        quality = clustering_quality(y_true, labels)

        results[K] = {
            "model": model, "log_lik": ll,
            "aic": aic_score, "bic": bic_score, "mml": mml_score,
            "labels": labels,
            "ari": quality["ari"], "nmi": quality["nmi"], "purity": quality["purity"],
            "elapsed": elapsed,
        }
        print(
            f"  K={K:2d} | log-lik={ll:10.1f}  "
            f"ARI={quality['ari']:.3f}  NMI={quality['nmi']:.3f}  "
            f"purity={quality['purity']:.3f}  ({elapsed:.1f}s)"
        )
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, true_k=TRUE_K, dataset_name="Digits (d=64)"):
    os.makedirs("figures", exist_ok=True)
    Ks = sorted(results.keys())

    aic_vals = [results[k]["aic"] for k in Ks]
    bic_vals = [results[k]["bic"] for k in Ks]
    mml_vals = [results[k]["mml"] for k in Ks]
    ari_vals = [results[k]["ari"] for k in Ks]
    nmi_vals = [results[k]["nmi"] for k in Ks]

    def norm01(vals):
        mn, mx = min(vals), max(vals)
        return [(v - mn) / (mx - mn + 1e-12) for v in vals]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(Ks, norm01(aic_vals), "o-", label="AIC", color="#4C72B0")
    ax.plot(Ks, norm01(bic_vals), "s-", label="BIC", color="#DD8452")
    ax.plot(Ks, norm01(mml_vals), "^-", label="MML", color="#55A868", linewidth=2)
    ax.axvline(true_k, color="red", linestyle="--", linewidth=1.2, label=f"True K={true_k}")
    ax.set_xlabel("Number of components K")
    ax.set_ylabel("Criterion (normalised 0–1, lower=better)")
    ax.set_title(f"Model selection criteria vs K\n({dataset_name})")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    best_aic = Ks[int(np.argmin(aic_vals))]
    best_bic = Ks[int(np.argmin(bic_vals))]
    best_mml = Ks[int(np.argmin(mml_vals))]

    ax2 = axes[1]
    ax2.plot(Ks, ari_vals, "o-", label="ARI", color="#4C72B0")
    ax2.plot(Ks, nmi_vals, "s-", label="NMI", color="#DD8452")
    ax2.axvline(true_k, color="red", linestyle="--", linewidth=1.2, label=f"True K={true_k}")
    ax2.set_xlabel("Number of components K")
    ax2.set_ylabel("Score (higher is better)")
    ax2.set_title(f"Clustering quality vs K\n(ARI and NMI, {dataset_name})")
    ax2.set_ylim(0, max(max(ari_vals), max(nmi_vals)) * 1.15)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("figures/model_selection_newsgroups.pdf", dpi=150)
    fig.savefig("figures/model_selection_newsgroups.png", dpi=150)
    print("\nSaved figures/model_selection_newsgroups.{pdf,png}")
    plt.close(fig)
    return {"best_aic": best_aic, "best_bic": best_bic, "best_mml": best_mml}


def print_summary_table(results, true_k=TRUE_K):
    print("\n" + "=" * 70)
    print(f"SUMMARY TABLE  (true K = {true_k})")
    print("=" * 70)
    print(f"{'K':>4}  {'log-lik':>12}  {'AIC':>12}  {'BIC':>12}  {'MML':>12}  {'ARI':>6}  {'NMI':>6}")
    print("-" * 70)
    for K in sorted(results.keys()):
        r = results[K]
        mark = " <--" if K == true_k else ""
        print(
            f"{K:>4}  {r['log_lik']:>12.1f}  {r['aic']:>12.1f}  "
            f"{r['bic']:>12.1f}  {r['mml']:>12.1f}  "
            f"{r['ari']:>6.3f}  {r['nmi']:>6.3f}{mark}"
        )
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="vMF mixture model selection on word embedding / image data"
    )
    parser.add_argument(
        "--newsgroups", action="store_true",
        help="Use 20 Newsgroups TF-IDF unit vectors (requires network/cached download). "
             "Without this flag, uses the sklearn digits dataset as a local stand-in."
    )
    parser.add_argument(
        "--k-range", nargs="+", type=int, default=K_RANGE,
        help="List of K values to try (default: 2 4 6 8 10 12 14)"
    )
    args = parser.parse_args()

    if args.newsgroups:
        X, y_true = load_newsgroups()
    else:
        X, y_true = load_data()

    K_RANGE_RUN = args.k_range
    results = run_experiments(X, y_true, k_range=K_RANGE_RUN)
    print_summary_table(results)
    minima = plot_results(results)
    print(
        f"\nSelected K:  AIC={minima['best_aic']}  "
        f"BIC={minima['best_bic']}  MML={minima['best_mml']}  (True K={TRUE_K})"
    )
