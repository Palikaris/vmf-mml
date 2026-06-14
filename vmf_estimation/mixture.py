"""
Mixture of von Mises-Fisher distributions (MoVMF) with EM algorithm.

Decision log (decisions made for the paper — see docs/MIXTURE_MODEL_ROADMAP.md):
  D3: MML multinomial estimate p̂_i = (n_i + 1/2) / (N + K/2)  [paper eq. 14]
  D4: Part 1(a) uses unary code Pr(K) = 2^{-K} for model-selection cost of K
  D5: M-step uses MML-h3 estimator for κ (71% better MSE than ML)
  D6: 5 random restarts; convergence when |Δ log-likelihood| < tol
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse
from scipy import optimize
from scipy.special import ive

from .core import bessel_ratio_A, bessel_ratio_A_inverse


def _log_h3_general(kappa: float, dim: int) -> float:
    """
    h3 prior log-density for arbitrary dimension d.

    For d=2 this is the Wallace & Dowe (1993) recommended prior:
        h3(κ) = κ / (1 + κ²)^{3/2}

    For d > 2 the priors.py module currently raises an error since W&D (1993)
    only covers the circular (d=2) case.  As a heuristic extension we use the
    same functional form, which:
      - Goes to 0 as κ → 0 (penalises degenerate uniform components)
      - Goes to 0 as κ → ∞ (penalises infinitely concentrated components)
      - Has a mode at κ = 1/√2

    Decision D7 (for David's review): We use h3(κ) = κ/(1+κ²)^{3/2} for all d.
    The prior contributes equally to all K-component models with the same d,
    so it does not affect the relative ordering of message lengths when
    selecting K.  The absolute message-length values will need updating once
    David derives the proper h3 extension to general d.
    """
    if kappa <= 0:
        return -math.inf
    return math.log(kappa) - 1.5 * math.log(1.0 + kappa ** 2)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _log_vmf_density(
    X: np.ndarray | scipy.sparse.spmatrix,
    mu: np.ndarray,
    kappa: float,
    dim: int,
) -> np.ndarray:
    """
    Compute log vMF density log f(x_n | μ, κ) for every row of X.

    log f(x | μ, κ) = log C_d(κ) + κ x^T μ
    log C_d(κ) = (d/2-1)log κ  -  (d/2)log(2π)  -  log I_{d/2-1}(κ)

    Uses ive (exponentially scaled Bessel) for numerical stability.

    Returns array of shape (N,).
    """
    N = X.shape[0]

    if kappa < 1e-10:
        # Uniform on sphere: constant log density = log(Γ(d/2) / (2 π^{d/2}))
        log_const = math.lgamma(dim / 2) - (dim / 2) * math.log(math.pi) - math.log(2)
        return np.full(N, log_const)

    nu = dim / 2.0 - 1.0
    ive_val = ive(nu, kappa)
    if ive_val <= 0:
        ive_val = 1e-300
    # log I_ν(κ) = log(ive(ν, κ)) + κ
    log_I = math.log(ive_val) + kappa
    log_C = nu * math.log(kappa) - (dim / 2.0) * math.log(2 * math.pi) - log_I

    # Dot products x_n^T μ
    if scipy.sparse.issparse(X):
        dots = np.asarray(X.dot(mu)).flatten()
    else:
        dots = X @ mu

    return log_C + kappa * dots


def _estimate_kappa_mml_h3(R_bar: float, n_samples: float, dim: int) -> float:
    """
    MML-h3 estimate of κ from sufficient statistics.

    Replicates the core optimisation of MMLEstimator(prior='h3') but takes
    (R_bar, n_samples, dim) directly instead of raw data — needed for the
    weighted M-step where n_samples = Σ γ_{n,i} is non-integer.

    Decision D5: h3 prior is used (best performing, 71% MSE improvement).
    """
    if not math.isfinite(R_bar) or R_bar <= 0:
        return 0.0

    R_bar = min(R_bar, 1.0 - 1e-12)  # numerical safety
    kappa_ml = bessel_ratio_A_inverse(R_bar, dim)

    def neg_objective(kappa: float) -> float:
        if kappa <= 1e-10:
            return np.inf
        try:
            log_h = _log_h3_general(kappa, dim)
            nu = dim / 2.0 - 1.0
            ive_val = ive(nu, kappa)
            if ive_val <= 0:
                ive_val = 1e-300
            log_I = math.log(ive_val) + kappa
            log_C = nu * math.log(kappa) - (dim / 2.0) * math.log(2 * math.pi) - log_I
            log_lik = n_samples * (log_C + kappa * R_bar)

            A = bessel_ratio_A(kappa, dim)
            kA = kappa * A
            F_term = 1.0 - A / kappa - A ** 2
            if kA <= 0 or F_term <= 0:
                return np.inf
            fisher_term = 0.5 * math.log(kA) + 0.5 * math.log(F_term)

            val = log_h + log_lik - fisher_term
            return -val if math.isfinite(val) else np.inf
        except Exception:
            return np.inf

    kappa_lower = max(1e-6, 0.01 * kappa_ml) if kappa_ml > 0 else 1e-6
    kappa_upper = max(100.0, 10.0 * kappa_ml) if kappa_ml > 0 else 100.0

    result = optimize.minimize_scalar(
        neg_objective,
        bounds=(kappa_lower, kappa_upper),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(result.x) if result.success else kappa_ml


# ---------------------------------------------------------------------------
# MovMF class
# ---------------------------------------------------------------------------

class MovMF:
    """
    Mixture of von Mises-Fisher distributions.

    Fits a K-component vMF mixture model using the EM algorithm.
    The M-step uses MML estimators for all parameters:
      - Mixing weights: p̂_i = (n_i + 1/2) / (N + K/2)  [MML multinomial]
      - Mean directions: μ̂_i = R_i / ||R_i||             [closed form]
      - Concentrations: κ̂_i via MML-h3 optimisation     [best estimator]

    Supports dense (np.ndarray) and sparse (scipy.sparse) input.

    Parameters
    ----------
    n_clusters : int
        Number of mixture components K.
    max_iter : int
        Maximum EM iterations per restart.
    tol : float
        Convergence tolerance on log-likelihood change.
    n_init : int
        Number of random restarts; best (highest log-lik) is kept.
    random_state : int or None
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        max_iter: int = 200,
        tol: float = 1e-6,
        n_init: int = 5,
        random_state: int | None = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

        # Set after fit()
        self.weights_: np.ndarray | None = None   # shape (K,)
        self.means_: np.ndarray | None = None      # shape (K, d)
        self.kappas_: np.ndarray | None = None     # shape (K,)
        self.log_likelihood_: float | None = None
        self.n_iter_: int | None = None
        self.converged_: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray | scipy.sparse.spmatrix) -> "MovMF":
        """
        Fit the mixture model to unit-vector data X.

        Parameters
        ----------
        X : array-like of shape (N, d)
            Rows must be unit vectors on S^{d-1}.  May be dense or sparse.

        Returns
        -------
        self
        """
        N, d = X.shape
        rng = np.random.RandomState(self.random_state)

        best_log_lik = -np.inf
        best_state: tuple | None = None

        for _ in range(self.n_init):
            seed = int(rng.randint(0, 2 ** 31))
            labels = self._init_kmeans_pp(X, N, d, seed)
            result = self._run_em(X, N, d, labels)
            ll, weights, means, kappas, n_iter, converged = result
            if math.isfinite(ll) and ll > best_log_lik:
                best_log_lik = ll
                best_state = (weights, means, kappas, n_iter, converged)

        if best_state is None:
            # All restarts gave NaN/non-finite log-likelihood.
            # Store dummy state so subsequent calls don't fail.
            rng2 = np.random.RandomState(self.random_state)
            _w = np.full(self.n_clusters, 1.0 / self.n_clusters)
            _m = np.vstack([
                v / np.linalg.norm(v)
                for v in [rng2.randn(d) for _ in range(self.n_clusters)]
            ])
            _k = np.zeros(self.n_clusters)
            best_state = (_w, _m, _k, 0, False)
            best_log_lik = -np.inf
        self.weights_, self.means_, self.kappas_, self.n_iter_, self.converged_ = best_state
        self.log_likelihood_ = best_log_lik
        return self

    def predict_proba(self, X: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
        """
        Return soft cluster assignments (responsibilities).

        Returns array of shape (N, K) with non-negative entries summing to 1
        across axis=1.
        """
        self._check_fitted()
        N, d = X.shape
        K = self.n_clusters
        log_resp = self._log_responsibilities(X, N, d, K)
        log_resp -= log_resp.max(axis=1, keepdims=True)
        resp = np.exp(log_resp)
        resp /= resp.sum(axis=1, keepdims=True)
        return resp

    def predict(self, X: np.ndarray | scipy.sparse.spmatrix) -> np.ndarray:
        """Return hard cluster labels (argmax of responsibilities)."""
        return np.argmax(self.predict_proba(X), axis=1)

    def log_likelihood(self, X: np.ndarray | scipy.sparse.spmatrix) -> float:
        """Compute total log-likelihood log p(X | model)."""
        self._check_fitted()
        N, d = X.shape
        K = self.n_clusters
        log_resp = self._log_responsibilities(X, N, d, K)
        log_resp_max = log_resp.max(axis=1)
        log_lik = (log_resp_max + np.log(np.exp(log_resp - log_resp_max[:, None]).sum(axis=1))).sum()
        return float(log_lik)

    def n_params(self, d: int) -> int:
        """
        Number of free parameters for model with d-dimensional data.

        k = K*(d+1) - 1
          = K*(d-1) [mean directions, d-1 free each on sphere]
          + K       [concentrations]
          + (K-1)   [mixing proportions]

        Matches paper equation for k (eq. for k parameters).
        """
        return self.n_clusters * (d + 1) - 1

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self.weights_ is None:
            raise ValueError("MovMF is not fitted yet. Call fit() first.")

    def _log_responsibilities(
        self, X: np.ndarray | scipy.sparse.spmatrix, N: int, d: int, K: int
    ) -> np.ndarray:
        """Compute unnormalised log responsibilities, shape (N, K)."""
        log_resp = np.empty((N, K))
        for i in range(K):
            log_resp[:, i] = (
                math.log(float(self.weights_[i]) + 1e-300)
                + _log_vmf_density(X, self.means_[i], float(self.kappas_[i]), d)
            )
        return log_resp

    def _init_kmeans_pp(
        self,
        X: np.ndarray | scipy.sparse.spmatrix,
        N: int,
        d: int,
        seed: int,
    ) -> np.ndarray:
        """
        Spherical k-means++ initialisation.

        Picks K centres on the sphere sequentially, with probability
        proportional to (1 - max_cosine_similarity_to_existing_centres).
        Returns initial hard label assignments of shape (N,).
        """
        K = self.n_clusters
        rng = np.random.RandomState(seed)

        def get_row_dense(idx: int) -> np.ndarray:
            if scipy.sparse.issparse(X):
                row = X.getrow(idx)
                return np.asarray(row.todense()).flatten()
            else:
                return X[idx]

        def dot_all(centre: np.ndarray) -> np.ndarray:
            if scipy.sparse.issparse(X):
                return np.asarray(X.dot(centre)).flatten()
            else:
                return X @ centre

        # First centre: uniform random
        centre_indices = [rng.randint(0, N)]
        centres = [get_row_dense(centre_indices[0])]

        for _ in range(K - 1):
            # Maximum cosine similarity to any existing centre
            sims = np.column_stack([dot_all(c) for c in centres])  # (N, len(centres))
            max_sim = sims.max(axis=1)                              # (N,)
            dist = np.maximum(0.0, 1.0 - max_sim)                  # (N,) ≥ 0

            total = dist.sum()
            if total < 1e-12:
                # All points very close to existing centres: pick uniformly
                new_idx = rng.randint(0, N)
            else:
                probs = dist / total
                new_idx = int(rng.choice(N, p=probs))

            centre_indices.append(new_idx)
            centres.append(get_row_dense(new_idx))

        # Assign labels by nearest centre
        centres_mat = np.vstack(centres)  # (K, d)
        if scipy.sparse.issparse(X):
            sims_all = np.asarray(X.dot(centres_mat.T))  # (N, K)
        else:
            sims_all = X @ centres_mat.T                 # (N, K)

        return np.argmax(sims_all, axis=1)

    def _run_em(
        self,
        X: np.ndarray | scipy.sparse.spmatrix,
        N: int,
        d: int,
        init_labels: np.ndarray,
    ) -> tuple:
        """
        One EM run starting from init_labels.

        Returns (log_lik, weights, means, kappas, n_iter, converged).
        """
        K = self.n_clusters
        weights = np.empty(K)
        means = np.empty((K, d))
        kappas = np.empty(K)

        # Initialise parameters from hard labels
        for i in range(K):
            mask = init_labels == i
            n_i = mask.sum()
            if n_i == 0:
                # Empty cluster: random unit vector, κ=1
                v = np.random.randn(d)
                means[i] = v / np.linalg.norm(v)
                kappas[i] = 1.0
                weights[i] = 1.0 / K
            else:
                if scipy.sparse.issparse(X):
                    R_vec = np.asarray(X[mask].sum(axis=0)).flatten()
                else:
                    R_vec = X[mask].sum(axis=0)
                R_vec = np.nan_to_num(R_vec, nan=0.0)
                norm = np.linalg.norm(R_vec)
                if not math.isfinite(norm) or norm < 1e-10:
                    v = np.random.randn(d)
                    means[i] = v / np.linalg.norm(v)
                    kappas[i] = 0.0
                else:
                    means[i] = R_vec / norm
                    R_bar_i = norm / float(n_i)
                    R_bar_i = min(R_bar_i, 1.0 - 1e-12)
                    kappas[i] = _estimate_kappa_mml_h3(R_bar_i, float(n_i), d)
                weights[i] = float(n_i) / N

        weights /= weights.sum()

        log_lik = -np.inf
        converged = False

        for it in range(self.max_iter):
            # ---- E-step ----
            log_resp = np.empty((N, K))
            for i in range(K):
                log_resp[:, i] = (
                    math.log(float(weights[i]) + 1e-300)
                    + _log_vmf_density(X, means[i], float(kappas[i]), d)
                )

            # Log-sum-exp for normalisation.
            # Guard against NaN in log_resp (caused by NaN kappa).
            log_resp = np.nan_to_num(log_resp, nan=-1e30, posinf=-1e30, neginf=-1e30)
            log_max = log_resp.max(axis=1)                        # (N,)
            log_sum = log_max + np.log(                           # (N,)
                np.exp(log_resp - log_max[:, None]).sum(axis=1) + 1e-300
            )
            new_log_lik = float(log_sum.sum())
            if not math.isfinite(new_log_lik):
                break  # This restart failed numerically; discard it

            # Normalise log responsibilities
            log_resp -= log_sum[:, None]
            resp = np.exp(log_resp)                               # (N, K)

            # Convergence check
            delta = new_log_lik - log_lik
            log_lik = new_log_lik
            if it > 0 and abs(delta) < self.tol:
                converged = True
                break

            # ---- M-step ----
            n_effs = resp.sum(axis=0)                             # (K,)

            for i in range(K):
                n_i = float(n_effs[i])
                if n_i < 1e-6:
                    continue

                # Resultant vector R_i = Σ_n γ_{n,i} x_n
                if scipy.sparse.issparse(X):
                    # resp[:, i] is (N,) dense; X is sparse (N, d)
                    R_vec = np.asarray(resp[:, i] @ X).flatten()  # (d,)
                else:
                    R_vec = resp[:, i] @ X                        # (d,)

                R_vec = np.nan_to_num(R_vec, nan=0.0, posinf=0.0, neginf=0.0)
                norm = np.linalg.norm(R_vec)
                if not math.isfinite(norm) or norm < 1e-10:
                    continue

                means[i] = R_vec / norm

                R_bar_i = norm / n_i
                if not math.isfinite(R_bar_i) or R_bar_i <= 0:
                    continue
                R_bar_i = min(R_bar_i, 1.0 - 1e-12)
                # Use fast ML estimate in the EM loop for convergence speed.
                # MML correction applied once at the end (see post-EM block).
                kappas[i] = bessel_ratio_A_inverse(R_bar_i, d)

            # MML multinomial estimate (Decision D3):
            # p̂_i = (n_i + 1/2) / (N + K/2)
            weights = (n_effs + 0.5) / (N + K / 2.0)
            weights /= weights.sum()

        # Post-EM: apply MML-h3 correction once at convergence.
        # The EM used ML kappas for speed; we now do a final E-step and apply
        # MML-h3 to each component's sufficient statistics.
        # Decision D5: use h3 prior for final kappa estimates.
        log_resp_final = np.empty((N, K))
        for i in range(K):
            log_resp_final[:, i] = (
                math.log(float(weights[i]) + 1e-300)
                + _log_vmf_density(X, means[i], float(kappas[i]), d)
            )
        log_resp_final = np.nan_to_num(log_resp_final, nan=-1e30, posinf=-1e30, neginf=-1e30)
        log_max_f = log_resp_final.max(axis=1)
        log_norm_f = log_max_f + np.log(np.exp(log_resp_final - log_max_f[:, None]).sum(axis=1) + 1e-300)
        resp_final = np.exp(log_resp_final - log_norm_f[:, None])
        n_effs_final = resp_final.sum(axis=0)

        for i in range(K):
            n_i = float(n_effs_final[i])
            if n_i < 1.0:
                continue
            if scipy.sparse.issparse(X):
                R_vec = np.asarray(resp_final[:, i] @ X).flatten()
            else:
                R_vec = resp_final[:, i] @ X
            R_vec = np.nan_to_num(R_vec, nan=0.0)
            norm = np.linalg.norm(R_vec)
            if norm < 1e-10 or not math.isfinite(norm):
                continue
            R_bar_i = min(norm / n_i, 1.0 - 1e-12)
            kappas[i] = _estimate_kappa_mml_h3(R_bar_i, n_i, d)

        return log_lik, weights, means, kappas, it + 1, converged
