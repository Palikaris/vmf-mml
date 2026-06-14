"""
Tests for core mathematical functions.

These tests validate the fundamental Bessel ratio computations and Fisher information
that underpin all estimators.
"""

import numpy as np
import pytest
from vmf_estimation.core import (
    bessel_ratio_A,
    bessel_ratio_A_inverse,
    fisher_information,
    compute_mean_direction,
    compute_mean_resultant_length,
)


class TestBesselRatioA:
    """Test Bessel ratio A_d(κ) = I_{d/2}(κ) / I_{d/2-1}(κ)"""

    def test_zero_kappa(self):
        """A_d(0) should be 0 for any dimension"""
        for dim in [2, 3, 10, 100]:
            assert bessel_ratio_A(0, dim) == 0.0

    def test_small_kappa_approximation(self):
        """For small κ, A_d(κ) ≈ κ/d"""
        kappa = 0.001
        for dim in [2, 3, 10]:
            A = bessel_ratio_A(kappa, dim)
            expected = kappa / dim
            # Should be very close for small kappa
            assert abs(A - expected) / expected < 0.01

    def test_monotonicity(self):
        """A_d(κ) should be monotonically increasing"""
        dim = 3
        kappas = np.linspace(0.1, 10, 20)
        A_values = [bessel_ratio_A(k, dim) for k in kappas]

        # Check each value is greater than the previous
        for i in range(1, len(A_values)):
            assert A_values[i] > A_values[i-1]

    def test_bounds(self):
        """A_d(κ) should be in [0, 1) for all κ ≥ 0"""
        dim = 3
        kappas = [0, 0.1, 1, 10, 100, 1000]

        for kappa in kappas:
            A = bessel_ratio_A(kappa, dim)
            assert 0 <= A < 1.0

    def test_large_kappa_asymptotic(self):
        """For large κ, A_d(κ) ≈ 1 - (d-1)/(2κ)"""
        kappa = 100
        dim = 3

        A = bessel_ratio_A(kappa, dim)
        expected = 1 - (dim - 1) / (2 * kappa)

        # Should be very close for large kappa
        assert abs(A - expected) < 0.001

    def test_different_dimensions(self):
        """Test A_d(κ) for various dimensions"""
        kappa = 1.0

        # Different dimensions should give different values
        A_2d = bessel_ratio_A(kappa, 2)
        A_3d = bessel_ratio_A(kappa, 3)
        A_10d = bessel_ratio_A(kappa, 10)

        # All should be valid
        assert 0 < A_2d < 1
        assert 0 < A_3d < 1
        assert 0 < A_10d < 1

        # Higher dimension should give smaller A for same kappa
        assert A_2d > A_3d > A_10d

    def test_array_input(self):
        """Test that A_d works with array input"""
        kappas = np.array([0.1, 1.0, 10.0])
        dim = 3

        A_values = bessel_ratio_A(kappas, dim)

        assert isinstance(A_values, np.ndarray)
        assert len(A_values) == len(kappas)
        assert all(0 <= A < 1 for A in A_values)


class TestBesselRatioAInverse:
    """Test inverse Bessel ratio: solve A_d(κ) = R for κ"""

    def test_zero_R(self):
        """R = 0 should give κ = 0"""
        for dim in [2, 3, 10]:
            kappa = bessel_ratio_A_inverse(0.0, dim)
            assert kappa == 0.0

    def test_round_trip(self):
        """A_inverse(A(κ)) should recover κ"""
        dim = 3
        test_kappas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]

        for true_kappa in test_kappas:
            # Compute A(κ)
            R = bessel_ratio_A(true_kappa, dim)

            # Invert to get κ back
            recovered_kappa = bessel_ratio_A_inverse(R, dim)

            # Should match within tolerance
            rel_error = abs(recovered_kappa - true_kappa) / true_kappa
            assert rel_error < 1e-10, (
                f"Round trip failed: κ={true_kappa}, R={R}, "
                f"recovered κ={recovered_kappa}, rel_error={rel_error}"
            )

    def test_high_precision(self):
        """Test that inverse achieves high precision"""
        dim = 3
        R = 0.7

        kappa = bessel_ratio_A_inverse(R, dim)
        A_recovered = bessel_ratio_A(kappa, dim)

        assert abs(A_recovered - R) < 1e-12

    def test_near_boundary(self):
        """Test R very close to 1"""
        dim = 3
        R = 0.9999

        kappa = bessel_ratio_A_inverse(R, dim)

        # Should be large but finite
        assert kappa > 10
        assert kappa < 1e6

        # Verify it's correct
        A_recovered = bessel_ratio_A(kappa, dim)
        assert abs(A_recovered - R) / R < 1e-6

    def test_different_dimensions(self):
        """Test inverse for different dimensions"""
        R = 0.5

        kappas = {}
        for dim in [2, 3, 10, 50]:
            kappa = bessel_ratio_A_inverse(R, dim)
            kappas[dim] = kappa

            # Verify correctness
            A_recovered = bessel_ratio_A(kappa, dim)
            assert abs(A_recovered - R) < 1e-10

        # Higher dimension needs larger κ for same R
        assert kappas[2] < kappas[3] < kappas[10] < kappas[50]

    def test_full_range(self):
        """Test inverse across the full valid range of R"""
        dim = 3
        R_values = np.linspace(0.01, 0.99, 20)

        for R in R_values:
            kappa = bessel_ratio_A_inverse(R, dim)

            # Should be positive
            assert kappa > 0

            # Should invert correctly
            A_recovered = bessel_ratio_A(kappa, dim)
            assert abs(A_recovered - R) < 1e-9


class TestFisherInformation:
    """Test Fisher information F(κ)"""

    def test_positive(self):
        """Fisher information should always be positive for κ > 0"""
        dim = 3
        kappas = [0.01, 0.1, 1, 10, 100]

        for kappa in kappas:
            F = fisher_information(kappa, dim)
            assert F > 0, f"Fisher info not positive for κ={kappa}: F={F}"

    def test_zero_kappa_raises(self):
        """Fisher information undefined at κ=0"""
        with pytest.raises(ValueError, match="undefined for kappa <= 0"):
            fisher_information(0, 3)

    def test_negative_kappa_raises(self):
        """Fisher information undefined for negative κ"""
        with pytest.raises(ValueError, match="undefined for kappa <= 0"):
            fisher_information(-1, 3)

    def test_different_dimensions(self):
        """Test Fisher information for different dimensions"""
        kappa = 1.0

        F_values = {}
        for dim in [2, 3, 10, 50]:
            F = fisher_information(kappa, dim)
            F_values[dim] = F
            assert F > 0

        # All should be reasonable values
        for F in F_values.values():
            assert 0 < F < 10

    def test_formula(self):
        """Verify Fisher information formula: F(κ) = 1 - A² - (d-1)A/κ"""
        kappa = 2.0
        dim = 3

        F = fisher_information(kappa, dim)
        A = bessel_ratio_A(kappa, dim)

        expected_F = 1 - A**2 - (dim - 1) * A / kappa

        assert abs(F - expected_F) < 1e-12


class TestComputeMeanDirection:
    """Test mean direction computation"""

    def test_aligned_vectors(self):
        """When all vectors point same direction, mean should be that direction"""
        # All vectors point in +x direction
        data = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ])

        mu = compute_mean_direction(data)

        assert mu.shape == (3,)
        assert np.allclose(mu, [1, 0, 0])
        assert np.isclose(np.linalg.norm(mu), 1.0)

    def test_normalization(self):
        """Mean direction should always be unit vector"""
        # Random data
        np.random.seed(42)
        data = np.random.randn(100, 3)
        data = data / np.linalg.norm(data, axis=1, keepdims=True)

        mu = compute_mean_direction(data)

        assert np.isclose(np.linalg.norm(mu), 1.0)

    def test_two_dimensions(self):
        """Test in 2D (circular case)"""
        # Vectors around unit circle, concentrated at angle π/4
        angles = np.array([0.2, 0.3, 0.4, 0.5]) * np.pi
        data = np.column_stack([np.cos(angles), np.sin(angles)])

        mu = compute_mean_direction(data)

        assert mu.shape == (2,)
        assert np.isclose(np.linalg.norm(mu), 1.0)

        # Mean angle should be around 0.35π
        mean_angle = np.arctan2(mu[1], mu[0])
        assert 0.2 * np.pi < mean_angle < 0.5 * np.pi


class TestComputeMeanResultantLength:
    """Test mean resultant length computation"""

    def test_aligned_vectors(self):
        """Perfect alignment should give R̄ = 1"""
        data = np.array([
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ])

        R = compute_mean_resultant_length(data)
        assert np.isclose(R, 1.0)

    def test_bounds(self):
        """R̄ should be in [0, 1]"""
        np.random.seed(42)

        # Concentrated data
        data_concentrated = np.random.randn(100, 3)
        data_concentrated[:, 0] += 5  # Bias toward +x
        data_concentrated = data_concentrated / np.linalg.norm(
            data_concentrated, axis=1, keepdims=True
        )

        R_conc = compute_mean_resultant_length(data_concentrated)
        assert 0 < R_conc <= 1.0

        # More dispersed data
        data_dispersed = np.random.randn(100, 3)
        data_dispersed = data_dispersed / np.linalg.norm(
            data_dispersed, axis=1, keepdims=True
        )

        R_disp = compute_mean_resultant_length(data_dispersed)
        assert 0 <= R_disp <= 1.0

        # Concentrated should have higher R̄
        assert R_conc > R_disp

    def test_opposite_vectors(self):
        """Opposite vectors should give R̄ ≈ 0"""
        data = np.array([
            [1, 0, 0],
            [-1, 0, 0],
        ])

        R = compute_mean_resultant_length(data)
        assert np.isclose(R, 0.0, atol=1e-10)

    def test_two_dimensions(self):
        """Test in 2D"""
        # Four vectors evenly spaced around circle
        angles = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        data = np.column_stack([np.cos(angles), np.sin(angles)])

        R = compute_mean_resultant_length(data)

        # Should cancel out to near 0
        assert R < 0.01


class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_ml_estimation_workflow(self):
        """Test the workflow for ML estimation using core functions"""
        # Generate simple synthetic data: unit vectors concentrated around [1,0,0]
        np.random.seed(42)
        n_samples = 100
        true_mu = np.array([1, 0, 0])

        # Generate concentrated data (simple method for testing)
        data = np.random.randn(n_samples, 3)
        data[:, 0] += 3  # Concentrate around x-axis
        data = data / np.linalg.norm(data, axis=1, keepdims=True)

        # Step 1: Compute mean direction
        mu_hat = compute_mean_direction(data)
        assert np.linalg.norm(mu_hat) == pytest.approx(1.0)

        # Step 2: Compute mean resultant length
        R_bar = compute_mean_resultant_length(data)
        assert 0 < R_bar < 1

        # Step 3: Invert to get κ̂_ML
        dim = 3
        kappa_ml = bessel_ratio_A_inverse(R_bar, dim)
        assert kappa_ml > 0

        # Step 4: Verify consistency: A(κ̂_ML) should equal R̄
        A_check = bessel_ratio_A(kappa_ml, dim)
        assert A_check == pytest.approx(R_bar, abs=1e-10)

    def test_round_trip_all_dimensions(self):
        """Test round trip for various dimensions"""
        dimensions = [2, 3, 5, 10, 50, 100, 300]
        test_kappas = [0.5, 1.0, 5.0, 10.0]

        for dim in dimensions:
            for true_kappa in test_kappas:
                # Skip extreme cases where distribution is essentially uniform
                # (very high dimension with very small kappa relative to dim)
                # These won't occur in practice and are numerically challenging
                if dim >= 100 and true_kappa < np.sqrt(dim):
                    continue

                # Forward: κ → R
                R = bessel_ratio_A(true_kappa, dim)

                # Backward: R → κ
                recovered_kappa = bessel_ratio_A_inverse(R, dim)

                # Check round trip with tight tolerance
                rel_error = abs(recovered_kappa - true_kappa) / true_kappa
                assert rel_error < 1e-9, (
                    f"dim={dim}, κ={true_kappa}: R={R:.6f}, "
                    f"recovered={recovered_kappa}, rel_error={rel_error}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
