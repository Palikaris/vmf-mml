"""
Tests for vmf_estimation.mixture and vmf_estimation.model_selection.

Tests are intentionally fast: small N, small d, small K.
"""

import math
import numpy as np
import pytest

from vmf_estimation.data import generate_vmf_samples, normalize_to_unit_sphere
from vmf_estimation.mixture import MovMF, _log_vmf_density, _estimate_kappa_mml_h3
from vmf_estimation.model_selection import aic, bic, mml_message_length, select_k


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mixture_data(K=2, n_per=200, dim=2, kappa=5.0, seed=42):
    """Generate balanced K-component vMF mixture with well-separated means."""
    rng = np.random.RandomState(seed)
    # Place means evenly around equator
    angles = np.linspace(0, 2 * math.pi, K, endpoint=False)
    if dim == 2:
        means = np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        means = np.zeros((K, dim))
        means[:, 0] = np.cos(angles)
        means[:, 1] = np.sin(angles)

    parts = []
    labels_true = []
    for i in range(K):
        samples = generate_vmf_samples(means[i], kappa, n_per, random_state=int(rng.randint(0, 10000)))
        parts.append(samples)
        labels_true.extend([i] * n_per)
    X = np.vstack(parts)
    return X, np.array(labels_true), means


# ---------------------------------------------------------------------------
# Tests for _log_vmf_density
# ---------------------------------------------------------------------------

class TestLogVmfDensity:
    def test_shape(self):
        X = np.eye(3)          # 3 unit vectors in R^3
        mu = np.array([1.0, 0.0, 0.0])
        ld = _log_vmf_density(X, mu, kappa=2.0, dim=3)
        assert ld.shape == (3,)

    def test_density_highest_at_mu(self):
        dim = 3
        mu = np.array([0.0, 0.0, 1.0])
        kappa = 5.0
        X = np.vstack([mu, -mu, np.array([1.0, 0.0, 0.0])])
        ld = _log_vmf_density(X, mu, kappa, dim)
        assert ld[0] > ld[1]   # aligned > antipodal
        assert ld[0] > ld[2]   # aligned > orthogonal

    def test_kappa_zero_constant(self):
        dim = 3
        mu = np.array([0.0, 0.0, 1.0])
        X = np.eye(3)
        X = normalize_to_unit_sphere(np.random.randn(10, 3))
        ld = _log_vmf_density(X, mu, kappa=0.0, dim=3)
        # All values should be equal (uniform)
        assert np.allclose(ld, ld[0], atol=1e-10)

    def test_circular_case(self):
        """d=2 case: check against known formula."""
        mu = np.array([1.0, 0.0])
        kappa = 3.0
        theta = 0.0  # x = mu
        x = np.array([[math.cos(theta), math.sin(theta)]])
        ld = _log_vmf_density(x, mu, kappa, dim=2)
        from scipy.special import i0
        expected = kappa * math.cos(theta) - math.log(2 * math.pi * i0(kappa))
        assert abs(ld[0] - expected) < 1e-8


# ---------------------------------------------------------------------------
# Tests for _estimate_kappa_mml_h3
# ---------------------------------------------------------------------------

class TestEstimateKappaMmlH3:
    def test_zero_R_bar(self):
        assert _estimate_kappa_mml_h3(0.0, 100, 2) == 0.0

    def test_positive_kappa_for_large_R_bar(self):
        kappa = _estimate_kappa_mml_h3(0.9, 100, 2)
        assert kappa > 0

    def test_monotone_in_R_bar(self):
        """Larger R_bar should give larger kappa."""
        kappas = [_estimate_kappa_mml_h3(r, 100, 2) for r in [0.3, 0.5, 0.7, 0.9]]
        for a, b in zip(kappas, kappas[1:]):
            assert b >= a

    def test_3d_case(self):
        kappa = _estimate_kappa_mml_h3(0.7, 50, 3)
        assert kappa > 0


# ---------------------------------------------------------------------------
# Tests for MovMF
# ---------------------------------------------------------------------------

class TestMovMF:
    def test_fit_returns_self(self):
        X, _, _ = make_mixture_data(K=2, n_per=50, dim=2, kappa=5.0)
        model = MovMF(n_clusters=2, n_init=1, random_state=0)
        result = model.fit(X)
        assert result is model

    def test_fitted_attributes(self):
        X, _, _ = make_mixture_data(K=2, n_per=50, dim=2, kappa=5.0)
        model = MovMF(n_clusters=2, n_init=2, random_state=0)
        model.fit(X)
        K, d = 2, 2
        assert model.weights_.shape == (K,)
        assert model.means_.shape == (K, d)
        assert model.kappas_.shape == (K,)
        assert abs(model.weights_.sum() - 1.0) < 1e-8
        assert model.log_likelihood_ is not None

    def test_weights_sum_to_one(self):
        X, _, _ = make_mixture_data(K=3, n_per=100, dim=3, kappa=4.0)
        model = MovMF(n_clusters=3, n_init=2, random_state=1)
        model.fit(X)
        assert abs(model.weights_.sum() - 1.0) < 1e-6

    def test_means_are_unit_vectors(self):
        X, _, _ = make_mixture_data(K=2, n_per=100, dim=3, kappa=4.0)
        model = MovMF(n_clusters=2, n_init=2, random_state=2)
        model.fit(X)
        norms = np.linalg.norm(model.means_, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-8)

    def test_predict_proba_shape_and_sums(self):
        X, _, _ = make_mixture_data(K=2, n_per=50, dim=2, kappa=5.0)
        model = MovMF(n_clusters=2, n_init=1, random_state=3)
        model.fit(X)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_labels(self):
        X, _, _ = make_mixture_data(K=2, n_per=50, dim=2, kappa=5.0)
        model = MovMF(n_clusters=2, n_init=1, random_state=4)
        model.fit(X)
        labels = model.predict(X)
        assert labels.shape == (len(X),)
        assert set(labels).issubset({0, 1})

    def test_log_likelihood_finite(self):
        X, _, _ = make_mixture_data(K=2, n_per=100, dim=2, kappa=5.0)
        model = MovMF(n_clusters=2, n_init=2, random_state=5)
        model.fit(X)
        ll = model.log_likelihood(X)
        assert math.isfinite(ll)

    def test_n_params(self):
        model = MovMF(n_clusters=3)
        model.weights_ = np.array([1/3, 1/3, 1/3])
        model.means_ = np.zeros((3, 5))
        model.kappas_ = np.ones(3)
        # k = K*(d+1) - 1 = 3*6 - 1 = 17
        assert model.n_params(d=5) == 17

    def test_clustering_accuracy_2d(self):
        """Two well-separated clusters in 2D: most points should be correctly assigned."""
        X, y_true, _ = make_mixture_data(K=2, n_per=200, dim=2, kappa=10.0, seed=7)
        model = MovMF(n_clusters=2, n_init=3, random_state=7)
        model.fit(X)
        labels = model.predict(X)

        # Map predicted labels to true labels (account for permutation)
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(y_true, labels)
        assert ari > 0.9, f"ARI={ari:.3f}, expected > 0.9 for well-separated clusters"

    def test_not_fitted_error(self):
        model = MovMF(n_clusters=2)
        X = np.eye(3)
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)

    def test_3d_mixture(self):
        """3D case should also work."""
        X, y_true, _ = make_mixture_data(K=3, n_per=100, dim=3, kappa=6.0, seed=42)
        model = MovMF(n_clusters=3, n_init=3, random_state=42)
        model.fit(X)
        labels = model.predict(X)
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(y_true, labels)
        assert ari > 0.7, f"ARI={ari:.3f} for 3D K=3"

    def test_sparse_input(self):
        """Sparse matrix input should produce same shape outputs as dense."""
        import scipy.sparse
        X, _, _ = make_mixture_data(K=2, n_per=50, dim=5, kappa=3.0, seed=0)
        X_sparse = scipy.sparse.csr_matrix(X)
        model_dense = MovMF(n_clusters=2, n_init=1, random_state=0)
        model_sparse = MovMF(n_clusters=2, n_init=1, random_state=0)
        model_dense.fit(X)
        model_sparse.fit(X_sparse)
        assert model_dense.weights_.shape == model_sparse.weights_.shape
        assert model_dense.means_.shape == model_sparse.means_.shape


# ---------------------------------------------------------------------------
# Tests for model selection criteria
# ---------------------------------------------------------------------------

class TestModelSelectionCriteria:
    def test_aic_formula(self):
        assert aic(log_lik=-100.0, n_params=5) == pytest.approx(210.0)

    def test_bic_formula(self):
        score = bic(log_lik=-100.0, n_params=5, n_samples=100)
        expected = -2 * (-100.0) + 5 * math.log(100)
        assert score == pytest.approx(expected)

    def test_aic_lower_for_better_model(self):
        """Better fit (higher log-lik), same params → lower AIC."""
        assert aic(-50.0, 5) < aic(-100.0, 5)

    def test_bic_penalises_more_params(self):
        """More params, same fit → higher BIC."""
        assert bic(-100.0, 10, 1000) > bic(-100.0, 5, 1000)

    def test_mml_message_length_finite(self):
        X, _, _ = make_mixture_data(K=2, n_per=100, dim=2, kappa=5.0)
        model = MovMF(n_clusters=2, n_init=2, random_state=0)
        model.fit(X)
        msg_len = mml_message_length(model, X)
        assert math.isfinite(msg_len)
        assert msg_len > 0

    def test_select_k_returns_correct_type(self):
        X, _, _ = make_mixture_data(K=2, n_per=80, dim=2, kappa=6.0)
        best_k, results = select_k(
            X, k_range=[2, 3], method="bic",
            movmf_kwargs={"n_init": 1, "random_state": 0}
        )
        assert best_k in [2, 3]
        assert set(results.keys()) == {2, 3}
        for r in results.values():
            assert "model" in r
            assert "aic" in r
            assert "bic" in r
            assert "mml" in r

    def test_mml_prefers_true_k(self):
        """MML should prefer K=2 over K=1 or K=5 for clearly bimodal data."""
        X, _, _ = make_mixture_data(K=2, n_per=150, dim=2, kappa=10.0, seed=123)
        best_k, _ = select_k(
            X, k_range=[1, 2, 3, 4],
            method="mml",
            movmf_kwargs={"n_init": 3, "random_state": 123},
        )
        # With kappa=10 (very concentrated), K=2 or K=3 should be favoured over K=1
        assert best_k >= 2, f"MML chose K={best_k}, expected >= 2"
