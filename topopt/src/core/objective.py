import numpy as np
from topopt.src.core.brinkman import alpha_from_density


def dissipation_objective(rho_bar, ux, uy, alpha_max, q):
    """Dissipation objective: sum over cells of alpha * |u|^2.

    Returns:
        J: scalar objective value
    """
    alpha = alpha_from_density(rho_bar, alpha_max, q)
    u_sq = ux**2 + uy**2
    return np.sum(alpha * u_sq)

def dJ_df(rho_bar, ux, uy, rho, cx, cy, alpha_max, q):
    """Adjoint source: S_i = dJ/df_i, exactly.

    The discrete adjoint requires S = dJ/df with no projection and no
    lattice weights. For J = sum_e alpha(rho_bar_e) |u_e|^2:

        dJ/du_a   = 2 * alpha * u_a
        du_a/df_i = (c_ia - u_a) / rho        (quotient rule: u = sum(f c)/rho,
                                               rho = sum(f), so the 1/rho
                                               differentiates too)

        => dJ/df_i = (2*alpha/rho) * [ (c_i . u) - |u|^2 ]

    An earlier version carried a 3*w_i weighting borrowed from the
    continuous-adjoint literature. That belongs to a different formulation;
    in a discrete adjoint the source is the plain derivative.
    """
    alpha = alpha_from_density(rho_bar, alpha_max, q)
    u_sq = ux**2 + uy**2

    nx, ny = ux.shape
    source = np.zeros((9, nx, ny))
    for i in range(9):
        cu = cx[i] * ux + cy[i] * uy
        source[i] = 2.0 * alpha * (cu - u_sq) / rho
    return source