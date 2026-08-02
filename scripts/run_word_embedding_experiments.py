"""
Real-data experiments for vMF mixture model selection.

Supports two datasets:
  - 20 Newsgroups (DEFAULT): TF-IDF unit vectors, d=2000, N=2000, 10
    ground-truth topic categories.  This is the experiment reported in the
    paper; running this script with no arguments reproduces it (the corpus
    is downloaded on first use and cached by scikit-learn).
  - sklearn digits (--digits): 64-dimensional digit images L2-normalised to
    the unit sphere, 10 ground-truth classes.  A local, download-free
    stand-in used during development; NOT the experiment reported in the
    paper.

Results are saved to figures/model_selection_<dataset>.{pdf,png}, so a
digits run cannot overwrite the newsgroups figure.
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

TRUE_K = 10    # both datasets have 10 ground-truth classes
K_RANGE = [5, 10, 15, 20, 25]    # as reported in the paper
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

    Ten categories (two each of sport, science, politics, computing and
    religion), giving K=10 ground-truth clusters, as reported in the paper.
    Requires network access on first use; scikit-learn caches thereafter.
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


def reduce_lsa(X, n_components, random_state=42):
    """
    Latent semantic analysis: project onto n_components singular vectors and
    re-normalise to the unit sphere S^{n_components - 1}.

    The vMF model is unchanged by this step -- it applies to any set of unit
    vectors.  What changes is the regime: reducing d while holding N fixed
    raises the per-tangent-dimension Fisher information F_mu = n_i k_i A_d(k_i)
    of each component mean, which is what collapses to zero at d ~ N.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize as skl_norm

    print(f"Reducing {X.shape[1]}D -> {n_components}D via TruncatedSVD…")
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    Xr = svd.fit_transform(X)
    ev = float(svd.explained_variance_ratio_.sum())
    Xr = skl_norm(Xr, norm="l2")
    print(f"  Shape: {Xr.shape}   explained variance: {ev:.3f}")
    return Xr


def report_fisher_mu(model, N, d, label="", quiet=False):
    """
    Report the per-tangent-dimension Fisher information of each component mean,

        F_mu = n_i * kappa_i * A_d(kappa_i),

    and how many components fall below 2*pi.  The MML encoding cost for a
    component mean is

        (d-1)/2 * max(0, 1 + log(F_mu / (2*pi*e))),

    which is exactly zero when F_mu < 2*pi.  This is the quantity behind the
    degenerate-mean behaviour reported for the unreduced TF-IDF vectors, so it
    is worth measuring rather than only asserting.
    """
    from vmf_estimation.core import bessel_ratio_A

    kappas = np.asarray(model.kappas_, dtype=float)
    n_i = np.asarray(model.weights_, dtype=float) * N
    A = np.array([bessel_ratio_A(float(k), d) for k in kappas])
    F_mu = n_i * kappas * A
    below = int((F_mu < 2 * math.pi).sum())

    if not quiet:
        print(f"\n  F_mu diagnostic {label} (d={d}, N={N}, K={len(kappas)}):")
        print(f"    kappa_i : min={kappas.min():.3f}  median={np.median(kappas):.3f}"
              f"  max={kappas.max():.3f}")
        print(f"    n_i     : min={n_i.min():.1f}  median={np.median(n_i):.1f}"
              f"  max={n_i.max():.1f}")
        print(f"    F_mu    : min={F_mu.min():.3g}  median={np.median(F_mu):.3g}"
              f"  max={F_mu.max():.3g}   (2*pi = {2 * math.pi:.3f})")
        print(f"    components with F_mu < 2*pi: {below}/{len(kappas)} "
              f"-> mean encoding cost is zero for these")
    return {"kappas": kappas.tolist(), "n_i": n_i.tolist(),
            "F_mu": F_mu.tolist(), "n_below_2pi": below}


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


def run_baselines(X, y_true, k_range):
    """
    Non-vMF baselines requested in review: spherical k-means and, where the
    representation is dense and low-dimensional enough to admit it, a Gaussian
    mixture in the reduced space.

    Note that X is already L2-normalised, so Lloyd's algorithm on X with
    Euclidean distance is exactly spherical k-means: for unit vectors,
    ||x - c||^2 = 2 - 2 x.c is monotone decreasing in cosine similarity.
    """
    from sklearn.cluster import KMeans

    N, d = X.shape
    out = {}
    dense_ok = (not scipy_sparse_check(X)) and d <= 400

    print(f"\nBaselines (N={N}, d={d})…")
    for K in k_range:
        row = {}
        km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
        row["skmeans"] = clustering_quality(y_true, km.labels_)

        if dense_ok:
            from sklearn.mixture import GaussianMixture
            gm = GaussianMixture(
                n_components=K, covariance_type="diag",
                max_iter=200, n_init=2, random_state=42,
            ).fit(X)
            row["gmm"] = clustering_quality(y_true, gm.predict(X))
        else:
            row["gmm"] = None

        out[K] = row
        g = row["gmm"]
        print(f"  K={K:3d} | sph. k-means ARI={row['skmeans']['ari']:.3f} "
              f"NMI={row['skmeans']['nmi']:.3f}"
              + (f"   | GMM ARI={g['ari']:.3f} NMI={g['nmi']:.3f}"
                 if g else "   | GMM skipped (sparse or d>400)"))
    return out


def scipy_sparse_check(X):
    import scipy.sparse
    return scipy.sparse.issparse(X)


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
        fmu = report_fisher_mu(model, N, d, quiet=True)

        results[K] = {
            "model": model, "log_lik": ll,
            "aic": aic_score, "bic": bic_score, "mml": mml_score,
            "labels": labels,
            "ari": quality["ari"], "nmi": quality["nmi"], "purity": quality["purity"],
            "fisher_mu": fmu,
            "elapsed": elapsed,
        }
        print(
            f"  K={K:2d} | log-lik={ll:10.1f}  "
            f"ARI={quality['ari']:.3f}  NMI={quality['nmi']:.3f}  "
            f"purity={quality['purity']:.3f}  "
            f"F_mu(med)={np.median(fmu['F_mu']):.3g} "
            f"[{fmu['n_below_2pi']}/{K} < 2pi]  ({elapsed:.1f}s)"
        )
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, true_k=TRUE_K, dataset_name="20 Newsgroups (d=2000)"):
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
    slug = "digits" if dataset_name.lower().startswith("digits") else "newsgroups"
    if "lsa" in dataset_name.lower():
        slug += "_lsa"
    fig.savefig(f"figures/model_selection_{slug}.pdf", dpi=150)
    fig.savefig(f"figures/model_selection_{slug}.png", dpi=150)
    print(f"\nSaved figures/model_selection_{slug}.{{pdf,png}}")
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
        "--digits", action="store_true",
        help="Use the sklearn digits dataset (local, no download) instead of "
             "20 Newsgroups. This is a development stand-in and is NOT the "
             "experiment reported in the paper."
    )
    parser.add_argument(
        "--k-range", nargs="+", type=int, default=K_RANGE,
        help="List of K values to try (default: 5 10 15 20 25, as in the paper)"
    )
    parser.add_argument(
        "--baselines", action="store_true",
        help="Also run spherical k-means and (in reduced dense space) a "
             "Gaussian mixture, reporting ARI/NMI against the ground truth."
    )
    parser.add_argument(
        "--lsa", type=int, default=None, metavar="DIM",
        help="Apply latent semantic analysis (TruncatedSVD) to DIM dimensions "
             "and re-normalise to the unit sphere before fitting. Typical: 100."
    )
    args = parser.parse_args()

    if args.digits:
        X, y_true = load_data()
        dataset_name = f"Digits (d={X.shape[1]})"
    else:
        X, y_true = load_newsgroups()
        dataset_name = f"20 Newsgroups (d={X.shape[1]})"

    if args.lsa:
        X = reduce_lsa(X, args.lsa)
        dataset_name = f"{dataset_name.split(' (')[0]} + LSA (d={X.shape[1]})"

    K_RANGE_RUN = args.k_range
    results = run_experiments(X, y_true, k_range=K_RANGE_RUN)
    print_summary_table(results)

    if args.baselines:
        base = run_baselines(X, y_true, K_RANGE_RUN)
        print("\n" + "=" * 70)
        print("BASELINE COMPARISON  (ARI / NMI against ground truth)")
        print("=" * 70)
        print(f"{'K':>4}  {'vMF mixture':>18}  {'sph. k-means':>18}  {'GMM':>18}")
        for K in K_RANGE_RUN:
            v = results[K]
            s = base[K]["skmeans"]
            g = base[K]["gmm"]
            gtxt = f"{g['ari']:.3f} / {g['nmi']:.3f}" if g else "n/a"
            print(f"{K:>4}  {v['ari']:.3f} / {v['nmi']:.3f}".ljust(26)
                  + f"{s['ari']:.3f} / {s['nmi']:.3f}".rjust(18)
                  + f"{gtxt}".rjust(20))
        print()
    minima = plot_results(results, dataset_name=dataset_name)
    print(
        f"\nSelected K:  AIC={minima['best_aic']}  "
        f"BIC={minima['best_bic']}  MML={minima['best_mml']}  (True K={TRUE_K})"
    )
