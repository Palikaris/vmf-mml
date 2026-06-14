"""
Tests for data generation functions.

These tests validate that vMF sampling produces correct distributions.
"""

import numpy as np
import pytest
from vmf_estimation.data import (
    generate_vmf_samples,
    normalize_to_unit_sphere,
    _sample_weight,
    _householder_rotate,
)
from vmf_estimation.core import compute_mean_resultant_length


class TestGenerateVMFSamples:
    """Test von Mises-Fisher sample generation"""

    def test_output_shape(self):
        """Generated samples should have correct shape"""
        mu = np.array([1, 0, 0])
        kappa = 1.0
        n_samples = 100

        samples = generate_vmf_samples(mu, kappa, n_samples, random_state=42)

        assert samples.shape == (n_samples, 3)

    def test_unit_norm(self):
        """All generated samples should have unit norm"""
        mu = np.array([0, 1, 0])
        kappa = 2.0
        n_samples = 50

        samples = generate_vmf_samples(mu, kappa, n_samples, random_state=42)

        norms = np.linalg.norm(samples, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)

    def test_zero_kappa_uniform(self):
        """κ=0 should give uniform distribution on sphere"""
        mu = np.array([1, 0, 0])
        kappa = 0.0
        n_samples = 1000

        samples = generate_vmf_samples(mu, kappa, n_samples, random_state=42)

        # For uniform distribution, mean resultant length should be near 0
        R_bar = compute_mean_resultant_length(samples)
        assert R_bar < 0.2  # Should be small for uniform distribution

    def test_large_kappa_concentrated(self):
        """Large κ should give concentrated distribution"""
        mu = np.array([0, 0, 1])
        kappa = 50.0
        n_samples = 100

        samples = generate_vmf_samples(mu, kappa, n_samples, random_state=42)

        # Samples should concentrate around μ
        # Check that most samples have positive z-component
        z_components = samples[:, 2]
        assert np.mean(z_components > 0.9) > 0.8  # Most should be close to μ

    def test_mean_direction(self):
        """Mean direction should point toward μ"""
        mu = np.array([1, 0, 0])
        mu = mu / np.linalg.norm(mu)
        kappa = 5.0
        n_samples = 500

        samples = generate_vmf_samples(mu, kappa, n_samples, random_state=42)

        # Compute mean direction
        mean_dir = np.mean(samples, axis=0)
        mean_dir = mean_dir / np.linalg.norm(mean_dir)

        # Should be close to μ
        assert np.dot(mean_dir, mu) > 0.8

    def test_different_dimensions(self):
        """Should work for different dimensions"""
        for dim in [2, 3, 5, 10]:
            mu = np.zeros(dim)
            mu[0] = 1.0
            kappa = 2.0
            n_samples = 50

            samples = generate_vmf_samples(mu, kappa, n_samples, random_state=42)

            assert samples.shape == (n_samples, dim)
            norms = np.linalg.norm(samples, axis=1)
            assert np.allclose(norms, 1.0, atol=1e-10)

    def test_reproducibility(self):
        """Same random seed should give same samples"""
        mu = np.array([0, 1, 0])
        kappa = 1.5
        n_samples = 20

        samples1 = generate_vmf_samples(mu, kappa, n_samples, random_state=123)
        samples2 = generate_vmf_samples(mu, kappa, n_samples, random_state=123)

        assert np.allclose(samples1, samples2)

    def test_non_canonical_mu(self):
        """Should work for μ not aligned with coordinate axes"""
        mu = np.array([1, 1, 1])
        mu = mu / np.linalg.norm(mu)
        kappa = 3.0
        n_samples = 100

        samples = generate_vmf_samples(mu, kappa, n_samples, random_state=42)

        # All samples should be unit vectors
        norms = np.linalg.norm(samples, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)

        # Mean direction should point toward μ
        mean_dir = np.mean(samples, axis=0)
        mean_dir = mean_dir / np.linalg.norm(mean_dir)
        assert np.dot(mean_dir, mu) > 0.7


class TestSampleWeight:
    """Test the weight sampling function"""

    def test_d3_range(self):
        """For d=3, w should be in [-1, 1]"""
        kappa = 2.0
        dim = 3

        for _ in range(100):
            w = _sample_weight(kappa, dim)
            assert -1 <= w <= 1

    def test_d3_concentration(self):
        """For d=3 with large κ, w should be near 1"""
        kappa = 20.0
        dim = 3
        n_samples = 100

        weights = [_sample_weight(kappa, dim) for _ in range(n_samples)]

        # Most weights should be positive and large
        assert np.mean(np.array(weights) > 0.5) > 0.9

    def test_general_d_range(self):
        """For general d, w should be in [-1, 1]"""
        for dim in [5, 10, 20]:
            kappa = 1.0
            for _ in range(50):
                w = _sample_weight(kappa, dim)
                assert -1 <= w <= 1


class TestHouseholderRotate:
    """Test Householder rotation"""

    def test_canonical_to_canonical(self):
        """Rotating canonical to canonical should be identity"""
        dim = 3
        x = np.array([0.5, 0.3, np.sqrt(1 - 0.5**2 - 0.3**2)])
        mu = np.array([0, 0, 1])

        x_rot = _householder_rotate(x, mu)

        assert np.allclose(x_rot, x)

    def test_preserves_norm(self):
        """Rotation should preserve norm"""
        x = np.array([0.1, 0.2, np.sqrt(1 - 0.1**2 - 0.2**2)])
        mu = np.array([1, 1, 1])
        mu = mu / np.linalg.norm(mu)

        x_rot = _householder_rotate(x, mu)

        assert np.isclose(np.linalg.norm(x_rot), np.linalg.norm(x))

    def test_alignment(self):
        """Rotated vector should align with mu correctly"""
        # Start with vector in canonical orientation (last component = w)
        w = 0.8
        x = np.array([0, 0, w])

        # Rotate to different mu
        mu = np.array([1, 0, 0])

        x_rot = _householder_rotate(x, mu)

        # Component in mu direction should be w
        assert np.isclose(np.dot(x_rot, mu), w, atol=1e-10)


class TestNormalizeToUnitSphere:
    """Test vector normalization"""

    def test_unit_vectors_unchanged(self):
        """Unit vectors should remain unchanged"""
        data = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])

        normalized = normalize_to_unit_sphere(data)

        assert np.allclose(normalized, data)

    def test_normalization(self):
        """Non-unit vectors should be normalized"""
        data = np.array([
            [2, 0, 0],
            [0, 3, 0],
            [1, 1, 1],
        ])

        normalized = normalize_to_unit_sphere(data)

        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1.0)

    def test_random_vectors(self):
        """Random vectors should normalize correctly"""
        np.random.seed(42)
        data = np.random.randn(100, 5)

        normalized = normalize_to_unit_sphere(data)

        norms = np.linalg.norm(normalized, axis=1)
        assert np.allclose(norms, 1.0)

    def test_2d(self):
        """Should work in 2D"""
        data = np.array([
            [3, 4],
            [1, 1],
        ])

        normalized = normalize_to_unit_sphere(data)

        expected = np.array([
            [0.6, 0.8],
            [1/np.sqrt(2), 1/np.sqrt(2)],
        ])

        assert np.allclose(normalized, expected)


class TestIntegration:
    """Integration tests combining data generation with estimation"""

    def test_ml_estimation_roundtrip(self):
        """Generate data with known κ, estimate, should be close"""
        from vmf_estimation.core import (
            compute_mean_resultant_length,
            bessel_ratio_A_inverse,
        )

        true_kappa = 3.0
        mu = np.array([0, 0, 1])
        n_samples = 500
        dim = 3

        # Generate data
        samples = generate_vmf_samples(mu, true_kappa, n_samples, random_state=42)

        # Estimate κ using ML
        R_bar = compute_mean_resultant_length(samples)
        kappa_ml = bessel_ratio_A_inverse(R_bar, dim)

        # Should be reasonably close (within 20% for 500 samples)
        rel_error = abs(kappa_ml - true_kappa) / true_kappa
        assert rel_error < 0.2, f"ML estimate {kappa_ml} too far from true {true_kappa}"

    def test_consistency_across_sample_sizes(self):
        """Larger samples should give better estimates"""
        from vmf_estimation.core import (
            compute_mean_resultant_length,
            bessel_ratio_A_inverse,
        )

        true_kappa = 2.0
        mu = np.array([1, 0, 0])
        dim = 3

        errors = []
        sample_sizes = [50, 100, 500, 1000]

        for n in sample_sizes:
            samples = generate_vmf_samples(mu, true_kappa, n, random_state=42)
            R_bar = compute_mean_resultant_length(samples)
            kappa_est = bessel_ratio_A_inverse(R_bar, dim)
            error = abs(kappa_est - true_kappa)
            errors.append(error)

        # Errors should generally decrease with more samples
        # (though not strictly monotonic due to randomness)
        assert errors[-1] < errors[0]  # 1000 samples better than 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
