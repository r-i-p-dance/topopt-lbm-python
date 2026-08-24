import numpy as np
import pytest
from topopt.src.core.brinkman import alpha_from_density, d_alpha_d_rho_bar


def test_alpha_zero_at_fluid():
    """alpha(rho_bar=1) should be 0 (fully fluid)."""
    result = alpha_from_density(np.array([1.0]), alpha_max=100.0, q=0.1)
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


def test_alpha_max_at_solid():
    """alpha(rho_bar=0) should equal alpha_max (fully solid)."""
    result = alpha_from_density(np.array([0.0]), alpha_max=100.0, q=0.1)
    np.testing.assert_allclose(result, 100.0, atol=1e-12)


def test_alpha_monotonic():
    """alpha should decrease monotonically as rho_bar increases."""
    rho = np.linspace(0, 1, 20)
    alpha = alpha_from_density(rho, alpha_max=100.0, q=0.1)
    diffs = np.diff(alpha)
    assert np.all(diffs <= 0), "alpha should be monotonically decreasing"


def test_d_alpha_matches_finite_difference():
    """Analytical d_alpha/d_rho should match centered finite difference."""
    rho_test = np.linspace(0.05, 0.95, 10)          # avoid endpoints
    eps = 1e-6
    for rho in rho_test:
        analytical = d_alpha_d_rho_bar(rho, alpha_max=100.0, q=0.1)
        fd = (alpha_from_density(rho + eps, 100.0, 0.1)
              - alpha_from_density(rho - eps, 100.0, 0.1)) / (2 * eps)
        np.testing.assert_allclose(
            analytical, fd, rtol=1e-4,
            err_msg=f"Derivative mismatch at rho={rho}: analytical={analytical}, fd={fd}"
        )


@pytest.mark.parametrize("q_val", [0.001, 0.01, 0.1, 1.0, 10.0])
def test_alpha_endpoints_at_various_q(q_val):
    """Regardless of q, alpha should still be 0 at rho=1 and alpha_max at rho=0."""
    assert alpha_from_density(1.0, 100.0, q_val) == pytest.approx(0.0)
    assert alpha_from_density(0.0, 100.0, q_val) == pytest.approx(100.0)