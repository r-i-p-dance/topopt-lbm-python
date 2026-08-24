import numpy as np
import pytest
from topopt.src.core.projection import heaviside_projection, d_heaviside_d_rho_e


def test_heaviside_symmetric_at_eta():
    """H(eta) should equal 0.5 at rho=eta (the transition point)."""
    result = heaviside_projection(0.5, beta=5.0, eta=0.5)
    np.testing.assert_allclose(result, 0.5, atol=1e-12)


def test_heaviside_endpoints():
    """H(0) ≈ 0 and H(1) ≈ 1 for sharp projection (large beta)."""
    beta = 20.0
    assert heaviside_projection(0.0, beta=beta) < 0.01
    assert heaviside_projection(1.0, beta=beta) > 0.99


def test_heaviside_monotonic():
    """H should be monotonically increasing in rho_tilde."""
    rho = np.linspace(0, 1, 30)
    h = heaviside_projection(rho, beta=5.0)
    assert np.all(np.diff(h) >= 0)


def test_d_heaviside_matches_finite_difference():
    """Analytical derivative should match centered finite difference."""
    rho_test = np.linspace(0.05, 0.95, 10)
    eps = 1e-6
    beta = 5.0
    for rho in rho_test:
        analytical = d_heaviside_d_rho_e(rho, beta=beta)
        fd = (heaviside_projection(rho + eps, beta=beta)
              - heaviside_projection(rho - eps, beta=beta)) / (2 * eps)
        np.testing.assert_allclose(
            analytical, fd, rtol=1e-4,
            err_msg=f"Derivative mismatch at rho={rho}"
        )


@pytest.mark.parametrize("beta_val", [1.0, 5.0, 10.0, 50.0])
def test_derivative_positive(beta_val):
    """dH/drho should be positive everywhere (H is increasing)."""
    rho = np.linspace(0.05, 0.95, 20)
    for r in rho:
        assert d_heaviside_d_rho_e(r, beta=beta_val) >= 0