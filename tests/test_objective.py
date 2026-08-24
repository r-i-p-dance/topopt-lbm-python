import numpy as np
from topopt.src.core.objective import dissipation_objective, dJ_df


def test_dissipation_zero_when_no_flow():
    """J = 0 when u = 0 everywhere."""
    ux = np.zeros((8, 32))
    uy = np.zeros((8, 32))
    rho_e = 0.5 * np.ones((8, 32))
    J = dissipation_objective(rho_e, ux, uy, alpha_max=100.0, q=0.1)
    assert J == 0.0

def test_dissipation_zero_when_fully_fluid():
    """J = 0 when rho_e = 0 everywhere (alpha = 0)."""
    ux = 0.05 * np.ones((8, 32))
    uy = np.zeros((8, 32))
    rho_e = np.ones((8, 32))
    J = dissipation_objective(rho_e, ux, uy, alpha_max=100.0, q=0.1)
    assert J == 0.0

def test_dissipation_positive_when_solid_has_flow():
    """J > 0 when there's flow through nominally-solid cells (the design penalty)."""
    ux = 0.05 * np.ones((8, 32))
    rho_e = 0.9 * np.ones((8, 32))              # mostly solid
    J = dissipation_objective(rho_e, ux, np.zeros_like(ux), alpha_max=100.0, q=0.1)
    assert J > 0.0