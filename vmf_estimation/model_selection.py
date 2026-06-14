"""
Model selection criteria for vMF mixture models.

Implements AIC, BIC, and MML message length for comparing models with
different numbers of components K.

Decision log:
  D4: Part 1(a) cost uses unary code Pr(K) = 2^{-K}, giving K*ln(2) nits.
      (Wallace tree code is more efficient but doesn't change qualitative results.)
"""

from __future__ import annotations

import math
import numpy as np
import scipy.sparse

from .core import bessel_ratio_A, bessel_ratio_A_inverse
from .mixture import MovMF, _log_vmf_density, _estimate_kappa_mml_h3, _log_h3_general


# ---------------------------------------------------------------------------
# Classic criteria (use ML log-likelihood from fitted MovMF)
# ---------------------------------------------------------------------------

def aic(log_lik: float, n_params: int) -> float:
    """
    Akaike Information Criterion.

    AIC = -2 * log_lik + 2 * k
    Lower is better.

    Parameters
    ----------
    log_lik : float
        Log-likelihood at the ML estimate.
    n_params : int
        Number of free parameters k.
    """
    return -2.0 * log_lik + 2.0 * n_params


def bic(log_lik: float, n_params: int, n_samples: int) -> float:
    """
    Bayesian Information Criterion (Schwarz 1978).

    BIC = -2 * log_lik + k * log(N)
    Lower is better.

    Parameters
    ----------
    log_lik : float
        Log-likelihood at the ML estimate.
    n_params : int
        Number of free parameters k.
    n_samples : int
        Number of data points N.
    """
    return -2.0 * log_lik + n_params * math.log(n_samples)


# ---------------------------------------------------------------------------
# MML message length
# ---------------------------------------------------------------------------

def mml_message_length(model: MovMF, X: np.ndarray | scipy.sparse.spmatrix) -> float:
    """
    Compute the two-part MML message length for a fitted MovMF model.

    Following the paper (sec. 4.4) and Wallace & Dowe (1993), the message
    has four parts:

    Part 1(a): Cost of stating K components.
        Unary code: K * ln(2) nits  (Decision D4)

    Part 1(b): Cost of stating mixing proportions p_1,...,p_K.
        Multinomial MML cost from the fitted p̂_i values.

    Part 1(c) + Part 2: For each component C_i, the two-part MML cost
        of stating (μ_i, κ_i) and the data assigned to that component.
        The decoupling (via Part 1(d)) allows per-component calculation.

    Returns the total message length in nits (natural log scale).
    Lower is better.

    Parameters
    ----------
    model : MovMF
        A fitted mixture model.
    X : array-like of shape (N, d)
        The same data used for fitting.
    """
    model._check_fitted()
    N, d = X.shape
    K = model.n_clusters

    weights = model.weights_
    means = model.means_
    kappas = model.kappas_

    # ---- Responsibilities (soft assignments) ----
    log_resp = np.empty((N, K))
    for i in range(K):
        log_resp[:, i] = (
            math.log(float(weights[i]) + 1e-300)
            + _log_vmf_density(X, means[i], float(kappas[i]), d)
        )
    log_max = log_resp.max(axis=1)
    log_sum = log_max + np.log(np.exp(log_resp - log_max[:, None]).sum(axis=1) + 1e-300)
    log_resp -= log_sum[:, None]
    resp = np.exp(log_resp)                   # (N, K)
    n_effs = resp.sum(axis=0)                 # effective counts

    total_msg = 0.0

    # ---- Part 1(a): cost of K (unary code, Decision D4) ----
    # Pr(K) = 2^{-K}, so cost = K * ln(2)
    part1a = K * math.log(2.0)
    total_msg += part1a

    # ---- Part 1(b): cost of multinomial p_1,...,p_K ----
    # Following paper eq. (14) and surrounding text:
    # Fisher(p) = N^{K-1} / prod(p_i)
    # Uniform prior h(p) = (K-1)!
    # MML multinomial cost (from Wallace 1987 / wallace2005 sec. 5.4.1):
    #   -log h(p) + 0.5*log Fisher(p) + D/2*(1 - log κ_D)
    # where D = K-1 for the multinomial
    # Using the MML formula from paper (sec. 4.4):
    #   0.5*log(N^{K-1} / prod p_i) - log((K-1)!) + (K-1)/2
    D_mult = K - 1
    if D_mult > 0:
        fisher_mult = (K - 1) * math.log(N) - sum(
            math.log(float(p) + 1e-300) for p in weights
        )
        log_prior_mult = math.lgamma(K)          # log (K-1)!
        # MML cost (dropping the κ_D constant which cancels in comparisons):
        part1b = -log_prior_mult + 0.5 * fisher_mult + D_mult / 2.0
    else:
        part1b = 0.0
    total_msg += part1b

    # ---- Part 1(c): per-component parameter encoding cost ----
    # The data cost is handled separately as the total mixture log-likelihood
    # (Part 2).  Using per-component weighted log-likelihoods (the EM Q-function
    # bound) would allow concentrated sub-clusters to have negative data costs,
    # biasing MML toward over-fitting.  The correct two-part MML code uses:
    #   Part 1 = model encoding (K, π, μ_i, κ_i for each i)
    #   Part 2 = data under the full mixture model = -log_likelihood(X)
    #
    # For κ_i: D=1 parameter with MML-h3 prior.
    #   Cost = -log h3(κ̂_i) + 0.5*log(n_i * F_unit(κ̂_i)) + D/2
    #   where F_unit(κ) = 1 - A_d(κ)^2 - (d-1)/κ * A_d(κ) is per-sample Fisher info.
    #   (n_i provides the correct sample-size scaling of Fisher information.)
    #
    # For μ_i: D = d-1 parameters with uniform prior on S^{d-1}.
    #   Fisher info for μ in the tangent space = n_i * κ_i * A_d(κ_i) per dimension.
    #   Cost = (d-1)/2 * (1 + log(n_i * κ_i * A_d(κ_i) / (2πe)))
    from .core import bessel_ratio_A as _A
    for i in range(K):
        n_i = float(n_effs[i])
        kappa_i = float(kappas[i])

        if kappa_i > 1e-8 and n_i > 0:
            A_i = _A(kappa_i, d)
            F_unit = 1.0 - A_i / kappa_i - A_i ** 2   # per-sample Fisher info for κ
            log_h3 = _log_h3_general(kappa_i, d)
            if F_unit > 0:
                log_F_kappa = math.log(n_i) + math.log(F_unit)
            else:
                log_F_kappa = math.log(max(n_i, 1.0))
            # MML parameter encoding for κ: -log h3(κ̂) + 0.5*log(n_i*F_unit) + D/2
            kappa_enc = -log_h3 + 0.5 * log_F_kappa + 0.5
        else:
            kappa_enc = 0.5

        # MML parameter encoding for μ (uniform prior on S^{d-1})
        if d > 1 and n_i > 0 and kappa_i > 1e-8:
            A_i = _A(kappa_i, d) if kappa_i > 1e-8 else 0.0
            F_mu = n_i * kappa_i * A_i   # Fisher info per tangent dimension
            if F_mu > 0:
                mu_enc = max(0.0, (d - 1) / 2.0 * (1.0 + math.log(F_mu / (2 * math.pi * math.e))))
            else:
                mu_enc = 0.0
        else:
            mu_enc = 0.0

        total_msg += kappa_enc + mu_enc

    # ---- Part 2: data cost under the full mixture model ----
    # Use the actual mixture log-likelihood so that over-fitted models are
    # penalised correctly (per-component weighted likelihoods can be negative
    # for concentrated clusters, which would otherwise favour large K).
    total_msg += -model.log_likelihood(X)

    return total_msg


# ---------------------------------------------------------------------------
# Model selection: scan over K
# ---------------------------------------------------------------------------

def select_k(
    X: np.ndarray | scipy.sparse.spmatrix,
    k_range: range | list[int],
    method: str = "mml",
    movmf_kwargs: dict | None = None,
) -> tuple[int, dict]:
    """
    Fit MovMF models for each K in k_range and select the best K.

    Parameters
    ----------
    X : array-like of shape (N, d)
        Unit-vector data.
    k_range : iterable of int
        Values of K to try.
    method : {'mml', 'aic', 'bic'}
        Selection criterion.  Lower score = better model.
    movmf_kwargs : dict or None
        Extra keyword arguments passed to MovMF().

    Returns
    -------
    best_k : int
        The selected number of components.
    results : dict
        Dictionary mapping K -> dict with keys 'model', 'score',
        'log_lik', 'aic', 'bic', 'mml'.
    """
    N, d = X.shape
    kwargs = movmf_kwargs or {}
    results = {}

    for K in k_range:
        model = MovMF(n_clusters=K, **kwargs)
        model.fit(X)

        ll = model.log_likelihood(X)
        k_params = model.n_params(d)

        aic_score = aic(ll, k_params)
        bic_score = bic(ll, k_params, N)
        mml_score = mml_message_length(model, X)

        results[K] = {
            "model": model,
            "log_lik": ll,
            "aic": aic_score,
            "bic": bic_score,
            "mml": mml_score,
        }

    score_key = method.lower()
    best_k = min(results, key=lambda k: results[k][score_key])
    return best_k, results
