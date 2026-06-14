"""
KL divergence between von Mises-Fisher distributions and their mixtures.

KL(f1 || f2) between two vMF distributions has a closed form.
KL between two mixtures is computed by Monte Carlo sampling.
"""

import math
import numpy as np
from scipy import special

from .core import bessel_ratio_A


def _log_c_d(kappa: float, d: int) -> float:
    """log normalisation constant log C_d(κ) = (d/2-1)log κ - (d/2)log(2π) - log I_{d/2-1}(κ)."""
    p = d / 2.0 - 1.0
    log_bessel = math.log(float(special.ive(p, kappa)) + 1e-300) + kappa
    return p * math.log(kappa + 1e-300) - (d / 2.0) * math.log(2.0 * math.pi) - log_bessel


def kl_vmf(mu1: np.ndarray, kappa1: float, mu2: np.ndarray, kappa2: float, d: int) -> float:
    """
    KL divergence from vMF(μ₁,κ₁) to vMF(μ₂,κ₂) in closed form.

    KL(f1 || f2) = log(C_d(κ1)/C_d(κ2)) + A_d(κ1) * (κ1 - κ2 * (μ1·μ2))

    Derivation: E_{f1}[log f1 - log f2] where E_{f1}[x] = A_d(κ1) * μ1.

    Args:
        mu1, mu2: Unit vectors on S^{d-1}
        kappa1, kappa2: Concentration parameters (>= 0)
        d: Ambient dimension

    Returns:
        KL divergence (>= 0, in nits)
    """
    cos_angle = float(np.dot(mu1, mu2))
    cos_angle = max(-1.0, min(1.0, cos_angle))

    A1 = bessel_ratio_A(kappa1, d)
    log_ratio = _log_c_d(kappa1, d) - _log_c_d(kappa2, d)
    return log_ratio + A1 * (kappa1 - kappa2 * cos_angle)


def _sample_vmf(mu: np.ndarray, kappa: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample n unit vectors from vMF(μ, κ) using Wood (1994) rejection sampling."""
    d = len(mu)
    if kappa < 1e-6:
        # Uniform on S^{d-1}
        samples = rng.standard_normal((n, d))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        return samples

    # Use the rejection sampling approach for vMF (Ulrich 1984 / Wood 1994)
    # Step 1: sample t from the W distribution (one-dimensional)
    b = (-2.0 * kappa + math.sqrt(4.0 * kappa**2 + (d - 1)**2)) / (d - 1)
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (d - 1) * math.log(1.0 - x0**2)

    samples = np.empty((n, d))
    # Rotate samples to be around [1, 0, ..., 0] then apply Householder to μ
    e1 = np.zeros(d)
    e1[0] = 1.0

    # Build Householder to map e1 → mu
    u = e1 - mu
    norm_u = np.linalg.norm(u)

    accepted = 0
    while accepted < n:
        batch = max(n - accepted, 32)
        # Sample W (concentration component)
        z = rng.beta((d - 1) / 2.0, (d - 1) / 2.0, size=batch)
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u_test = rng.uniform(size=batch)
        mask = kappa * w + (d - 1) * np.log(1.0 - x0 * w) - c >= np.log(u_test)

        w_good = w[mask]
        if len(w_good) == 0:
            continue
        take = min(len(w_good), n - accepted)
        w_good = w_good[:take]

        # Sample uniformly on S^{d-2}
        v = rng.standard_normal((take, d - 1))
        v /= np.linalg.norm(v, axis=1, keepdims=True)

        # Construct samples around e1
        sq = np.sqrt(np.maximum(0.0, 1.0 - w_good**2))[:, None]
        pts = np.concatenate([w_good[:, None], sq * v], axis=1)

        # Rotate to μ via Householder (if μ ≠ e1)
        if norm_u > 1e-12:
            u_hat = u / norm_u
            pts = pts - 2.0 * (pts @ u_hat)[:, None] * u_hat

        samples[accepted: accepted + take] = pts
        accepted += take

    return samples


def _log_mixture_density(X: np.ndarray, weights, means, kappas, d: int) -> np.ndarray:
    """Log density of a vMF mixture at each row of X. Returns (N,) array."""
    from .mixture import _log_vmf_density
    N = X.shape[0]
    K = len(weights)
    log_comp = np.empty((N, K))
    for i in range(K):
        log_comp[:, i] = math.log(float(weights[i]) + 1e-300) + _log_vmf_density(
            X, means[i], float(kappas[i]), d
        )
    log_max = log_comp.max(axis=1)
    log_sum = log_max + np.log(np.exp(log_comp - log_max[:, None]).sum(axis=1) + 1e-300)
    return log_sum


def kl_mixture_mc(
    true_weights,
    true_means: np.ndarray,
    true_kappas,
    fitted_model,
    n_mc: int = 5000,
    random_state: int = 0,
) -> float:
    """
    Monte Carlo estimate of KL(true mixture || fitted mixture).

    KL(p || q) ≈ (1/M) Σ_{x ~ p} [log p(x) - log q(x)]

    Samples x from the TRUE mixture, evaluates both densities.

    Args:
        true_weights: (K_true,) mixing weights of the true mixture
        true_means:   (K_true, d) unit-vector means of the true mixture
        true_kappas:  (K_true,) concentration parameters of the true mixture
        fitted_model: fitted MovMF object
        n_mc: number of Monte Carlo samples
        random_state: RNG seed

    Returns:
        Estimated KL divergence (>= 0, in nits).  Returns np.inf on failure.
    """
    rng = np.random.default_rng(random_state)
    K_true = len(true_weights)
    d = true_means.shape[1]

    # Sample from true mixture: draw component index then sample from that component
    comp_counts = rng.multinomial(n_mc, true_weights)
    samples_list = []
    for i in range(K_true):
        n_i = int(comp_counts[i])
        if n_i > 0:
            s = _sample_vmf(true_means[i], float(true_kappas[i]), n_i, rng)
            samples_list.append(s)
    if not samples_list:
        return math.inf
    X = np.vstack(samples_list)

    # log p(x) under the true mixture
    log_p = _log_mixture_density(X, true_weights, true_means, true_kappas, d)

    # log q(x) under the fitted mixture
    log_q = _log_mixture_density(
        X, fitted_model.weights_, fitted_model.means_, fitted_model.kappas_, d
    )

    kl = float(np.mean(log_p - log_q))
    return max(0.0, kl)
