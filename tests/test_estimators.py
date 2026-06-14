"""
Tests for estimator classes.

These tests validate that each estimator works correctly on synthetic data.
"""

import numpy as np
import pytest
from vmf_estimation.estimators import MLEstimator
from vmf_estimation.data import generate_vmf_samples


class TestMLEstimator:
    """Test Maximum Likelihood estimator"""

    def test_name(self):
        """Estimator should have correct name"""
        est = MLEstimator()
        assert est.name == "ML"

    def test_estimate_shape(self):
        """Estimate should return correct shapes"""
        est = MLEstimator()

        # Generate test data
        mu_true = np.array([1, 0, 0])
        kappa_true = 2.0
        data = generate_vmf_samples(mu_true, kappa_true, 100, random_state=42)

        mu_hat, kappa_hat = est.estimate(data)

        assert mu_hat.shape == (3,)
        assert isinstance(kappa_hat, (float, np.floating))
        assert kappa_hat > 0

    def test_perfect_alignment(self):
        """ML should give κ→∞ for perfectly aligned data"""
        est = MLEstimator()

        # All vectors point in same direction
        data = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ])

        mu_hat, kappa_hat = est.estimate(data)

        # μ̂ should be [1, 0, 0]
        assert np.allclose(mu_hat, [1, 0, 0])

        # κ̂ should be very large (R̄ = 1)
        assert kappa_hat > 100  # Will be very large for R̄ ≈ 1

    def test_known_kappa_recovery(self):
        """ML should recover approximately correct κ for known distribution"""
        est = MLEstimator()

        true_kappa = 3.0
        mu_true = np.array([0, 0, 1])
        n_samples = 500

        # Generate data
        data = generate_vmf_samples(mu_true, true_kappa, n_samples, random_state=42)

        # Estimate
        mu_hat, kappa_hat = est.estimate(data)

        # μ̂ should point in same direction as true μ
        assert np.dot(mu_hat, mu_true) > 0.9

        # κ̂ should be close to true κ (within 20% for 500 samples)
        rel_error = abs(kappa_hat - true_kappa) / true_kappa
        assert rel_error < 0.2, f"ML: κ̂={kappa_hat}, true={true_kappa}, error={rel_error}"

    def test_different_dimensions(self):
        """ML should work for different dimensions"""
        est = MLEstimator()

        for dim in [2, 3, 5, 10]:
            mu_true = np.zeros(dim)
            mu_true[0] = 1.0
            kappa_true = 2.0

            data = generate_vmf_samples(mu_true, kappa_true, 200, random_state=42)
            mu_hat, kappa_hat = est.estimate(data)

            assert mu_hat.shape == (dim,)
            assert kappa_hat > 0
            assert np.isfinite(kappa_hat)

    def test_consistency(self):
        """ML estimates should improve with more data"""
        est = MLEstimator()

        true_kappa = 2.0
        mu_true = np.array([1, 0, 0])

        errors = []
        sample_sizes = [50, 100, 500, 1000]

        for n in sample_sizes:
            data = generate_vmf_samples(mu_true, true_kappa, n, random_state=42)
            _, kappa_hat = est.estimate(data)
            error = abs(kappa_hat - true_kappa)
            errors.append(error)

        # Error should decrease with more samples
        assert errors[-1] < errors[0]

    def test_invalid_data_raises(self):
        """ML should raise on invalid data"""
        est = MLEstimator()

        # Wrong shape
        with pytest.raises(ValueError, match="must be 2D"):
            est.estimate(np.array([1, 2, 3]))

        # Too few samples
        with pytest.raises(ValueError, match="at least 2 samples"):
            data = np.array([[1, 0, 0]])
            est.estimate(data)

        # Not unit vectors
        with pytest.raises(ValueError, match="unit norm"):
            data = np.array([
                [2, 0, 0],
                [0, 3, 0],
            ])
            est.estimate(data)

    def test_reproducibility(self):
        """Same data should give same estimates"""
        est = MLEstimator()

        data = generate_vmf_samples(np.array([0, 1, 0]), 1.5, 100, random_state=999)

        mu1, kappa1 = est.estimate(data)
        mu2, kappa2 = est.estimate(data)

        assert np.allclose(mu1, mu2)
        assert kappa1 == kappa2

    def test_small_kappa(self):
        """ML should handle small κ (dispersed data)"""
        est = MLEstimator()

        true_kappa = 0.1  # Very dispersed
        mu_true = np.array([1, 0, 0])

        data = generate_vmf_samples(mu_true, true_kappa, 200, random_state=42)
        mu_hat, kappa_hat = est.estimate(data)

        # Should give small estimate
        assert 0 < kappa_hat < 1.0

    def test_large_kappa(self):
        """ML should handle large κ (concentrated data)"""
        est = MLEstimator()

        true_kappa = 20.0  # Very concentrated
        mu_true = np.array([0, 0, 1])

        data = generate_vmf_samples(mu_true, true_kappa, 200, random_state=42)
        mu_hat, kappa_hat = est.estimate(data)

        # Should give large estimate
        assert kappa_hat > 10.0

    def test_2d_circular_case(self):
        """ML should work for 2D (circular) data"""
        est = MLEstimator()

        # Generate 2D data
        angles = np.random.vonmises(0, 5.0, 100)  # Use numpy's von Mises
        data = np.column_stack([np.cos(angles), np.sin(angles)])

        mu_hat, kappa_hat = est.estimate(data)

        assert mu_hat.shape == (2,)
        assert kappa_hat > 0
        assert np.isclose(np.linalg.norm(mu_hat), 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
