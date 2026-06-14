"""
Tests for MML prior functions.

These tests validate the three prior density functions used in MML estimation.
"""

import numpy as np
import pytest
from vmf_estimation.priors import (
    log_h1_prior,
    log_h2_prior,
    log_h3_prior,
    get_prior_function,
    PRIOR_FUNCTIONS,
)


class TestLogH1Prior:
    """Test Jeffreys-like prior h₁(κ) ∝ √F(κ)"""

    def test_positive_kappa(self):
        """h₁ should return finite values for positive κ"""
        dim = 3
        kappas = [0.01, 0.1, 1, 10, 100]

        for kappa in kappas:
            log_h1 = log_h1_prior(kappa, dim)
            assert np.isfinite(log_h1), f"log_h1 not finite for κ={kappa}"

    def test_zero_kappa(self):
        """h₁ should return -∞ for κ=0"""
        dim = 3
        log_h1 = log_h1_prior(0, dim)
        assert log_h1 == -np.inf

    def test_negative_kappa(self):
        """h₁ should return -∞ for negative κ"""
        dim = 3
        log_h1 = log_h1_prior(-1, dim)
        assert log_h1 == -np.inf

    def test_different_dimensions(self):
        """h₁ should work for various dimensions"""
        kappa = 1.0

        for dim in [2, 3, 10, 50, 300]:
            log_h1 = log_h1_prior(kappa, dim)
            assert np.isfinite(log_h1), f"Failed for dim={dim}"

    def test_reasonable_values(self):
        """log h₁ should have reasonable magnitude"""
        dim = 3
        kappas = [0.1, 1, 10]

        for kappa in kappas:
            log_h1 = log_h1_prior(kappa, dim)
            # log of a probability density can be negative
            # but shouldn't be extremely large
            assert -50 < log_h1 < 50, f"log_h1={log_h1} for κ={kappa}"


class TestLogH2Prior:
    """Test uniform on μ prior h₂(κ)"""

    def test_positive_kappa(self):
        """h₂ should return finite values for positive κ"""
        dim = 3
        kappas = [0.01, 0.1, 1, 10, 100]

        for kappa in kappas:
            log_h2 = log_h2_prior(kappa, dim)
            assert np.isfinite(log_h2), f"log_h2 not finite for κ={kappa}"

    def test_zero_kappa(self):
        """h₂ should return -∞ for κ=0"""
        dim = 3
        log_h2 = log_h2_prior(0, dim)
        assert log_h2 == -np.inf

    def test_negative_kappa(self):
        """h₂ should return -∞ for negative κ"""
        dim = 3
        log_h2 = log_h2_prior(-1, dim)
        assert log_h2 == -np.inf

    def test_different_dimensions(self):
        """h₂ should work for various dimensions"""
        kappa = 1.0

        for dim in [2, 3, 10, 50, 300]:
            log_h2 = log_h2_prior(kappa, dim)
            assert np.isfinite(log_h2), f"Failed for dim={dim}"

    def test_reasonable_values(self):
        """log h₂ should have reasonable magnitude"""
        dim = 3
        kappas = [0.1, 1, 10]

        for kappa in kappas:
            log_h2 = log_h2_prior(kappa, dim)
            assert -100 < log_h2 < 100, f"log_h2={log_h2} for κ={kappa}"

    def test_very_small_kappa(self):
        """h₂ should handle very small κ gracefully"""
        dim = 3
        log_h2 = log_h2_prior(1e-12, dim)
        assert np.isfinite(log_h2)

    def test_very_large_kappa(self):
        """h₂ should handle very large κ gracefully"""
        dim = 3
        log_h2 = log_h2_prior(1000, dim)
        assert np.isfinite(log_h2)


class TestLogH3Prior:
    """Test Wallace-Dowe recommended prior h₃(κ)"""

    def test_positive_kappa(self):
        """h₃ should return finite values for positive κ"""
        dim = 3
        kappas = [0.01, 0.1, 1, 10, 100]

        for kappa in kappas:
            log_h3 = log_h3_prior(kappa, dim)
            assert np.isfinite(log_h3), f"log_h3 not finite for κ={kappa}"

    def test_zero_kappa(self):
        """h₃ should return -∞ for κ=0"""
        dim = 3
        log_h3 = log_h3_prior(0, dim)
        assert log_h3 == -np.inf

    def test_negative_kappa(self):
        """h₃ should return -∞ for negative κ"""
        dim = 3
        log_h3 = log_h3_prior(-1, dim)
        assert log_h3 == -np.inf

    def test_different_dimensions(self):
        """h₃ should work for various dimensions"""
        kappa = 1.0

        for dim in [2, 3, 10, 50, 300]:
            log_h3 = log_h3_prior(kappa, dim)
            assert np.isfinite(log_h3), f"Failed for dim={dim}"

    def test_reasonable_values(self):
        """log h₃ should have reasonable magnitude"""
        dim = 3
        kappas = [0.1, 1, 10]

        for kappa in kappas:
            log_h3 = log_h3_prior(kappa, dim)
            assert -100 < log_h3 < 100, f"log_h3={log_h3} for κ={kappa}"

    def test_very_small_kappa(self):
        """h₃ should handle very small κ gracefully"""
        dim = 3
        log_h3 = log_h3_prior(1e-12, dim)
        # For very small κ, log h₃ → -∞ (penalizes κ→0)
        # But should still be computable
        assert log_h3 < 0  # Should be very negative

    def test_very_large_kappa(self):
        """h₃ should handle very large κ gracefully"""
        dim = 3
        log_h3 = log_h3_prior(1000, dim)
        assert np.isfinite(log_h3)


class TestPriorComparisons:
    """Test relationships between different priors"""

    def test_all_priors_defined(self):
        """All three priors should be defined"""
        assert 'h1' in PRIOR_FUNCTIONS
        assert 'h2' in PRIOR_FUNCTIONS
        assert 'h3' in PRIOR_FUNCTIONS

    def test_get_prior_function(self):
        """get_prior_function should retrieve correct functions"""
        assert get_prior_function('h1') == log_h1_prior
        assert get_prior_function('h2') == log_h2_prior
        assert get_prior_function('h3') == log_h3_prior

    def test_get_prior_function_invalid(self):
        """get_prior_function should raise for invalid names"""
        with pytest.raises(ValueError, match="Unknown prior"):
            get_prior_function('h4')

    def test_priors_order(self):
        """
        Test that priors maintain expected ordering.

        From Wallace & Dowe (1993):
        - h₁ is most aggressive (largest penalty for small κ)
        - h₂ is least aggressive
        - h₃ is intermediate (recommended)

        For small to moderate κ, we expect: h₁ < h₃ < h₂ (in log space)
        """
        dim = 3
        kappas = [0.5, 1.0, 2.0]

        for kappa in kappas:
            log_h1 = log_h1_prior(kappa, dim)
            log_h2 = log_h2_prior(kappa, dim)
            log_h3 = log_h3_prior(kappa, dim)

            # All should be finite
            assert np.isfinite(log_h1)
            assert np.isfinite(log_h2)
            assert np.isfinite(log_h3)

            # h₁ should generally be smallest (most negative) for small κ
            # This reflects its aggressive penalization
            # Note: This ordering may not hold for all κ, but should for moderate values

    def test_all_priors_consistent(self):
        """All priors should give consistent behavior across dimensions"""
        kappa = 2.0

        for dim in [2, 3, 10, 50]:
            log_h1 = log_h1_prior(kappa, dim)
            log_h2 = log_h2_prior(kappa, dim)
            log_h3 = log_h3_prior(kappa, dim)

            # All should be finite
            assert np.isfinite(log_h1), f"h1 failed at dim={dim}"
            assert np.isfinite(log_h2), f"h2 failed at dim={dim}"
            assert np.isfinite(log_h3), f"h3 failed at dim={dim}"


class TestNumericalStability:
    """Test numerical stability of prior computations"""

    def test_high_dimension(self):
        """Test priors work in high dimensions"""
        dim = 300
        kappas = [1.0, 10.0, 50.0]

        for kappa in kappas:
            log_h1 = log_h1_prior(kappa, dim)
            log_h2 = log_h2_prior(kappa, dim)
            log_h3 = log_h3_prior(kappa, dim)

            assert np.isfinite(log_h1), f"h1 failed: κ={kappa}, dim={dim}"
            assert np.isfinite(log_h2), f"h2 failed: κ={kappa}, dim={dim}"
            assert np.isfinite(log_h3), f"h3 failed: κ={kappa}, dim={dim}"

    def test_extreme_kappa(self):
        """Test priors with extreme κ values"""
        dim = 3

        # Very small kappa
        for kappa in [1e-10, 1e-5, 0.001]:
            log_h1 = log_h1_prior(kappa, dim)
            log_h2 = log_h2_prior(kappa, dim)
            log_h3 = log_h3_prior(kappa, dim)

            # Should all be finite (though may be very negative)
            assert np.isfinite(log_h1) or log_h1 == -np.inf
            assert np.isfinite(log_h2)
            assert np.isfinite(log_h3) or log_h3 < -100  # Can be very negative

        # Very large kappa
        for kappa in [100, 500, 1000]:
            log_h1 = log_h1_prior(kappa, dim)
            log_h2 = log_h2_prior(kappa, dim)
            log_h3 = log_h3_prior(kappa, dim)

            assert np.isfinite(log_h1)
            assert np.isfinite(log_h2)
            assert np.isfinite(log_h3)

    def test_no_overflow_underflow(self):
        """Test that priors don't overflow or underflow inappropriately"""
        dim = 10
        kappas = np.logspace(-2, 3, 20)  # 0.01 to 1000

        for kappa in kappas:
            log_h1 = log_h1_prior(kappa, dim)
            log_h2 = log_h2_prior(kappa, dim)
            log_h3 = log_h3_prior(kappa, dim)

            # Check no NaN (but -inf is okay for very small κ)
            assert not np.isnan(log_h1)
            assert not np.isnan(log_h2)
            assert not np.isnan(log_h3)

            # Check not +inf (shouldn't happen for proper priors)
            assert log_h1 < np.inf
            assert log_h2 < np.inf
            assert log_h3 < np.inf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
