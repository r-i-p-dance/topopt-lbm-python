import numpy as np
from topopt.src.core.brinkman import d_alpha_d_rho_bar
from topopt.src.core.projection import d_heaviside_d_rho_e

def assemble_sensitivity(forward, adjoint, alpha_max, beta, q, eta=0.5):
    """G_e = dalpha/drho_bar * drho_bar/drho_e * [ |u|^2 + (omega^2/2) * sum_i(g_i * f_neq_i) ]

    Explicit term  : dJ/dalpha at fixed f, = |u|^2.
    Implicit term  : g^T dC/dalpha, where dC_i/dalpha = (omega^2/2) f_i^neq
                     (from dw/dalpha = -omega^2/2 and dC_i/domega = -f_i^neq),
                     and g is the adjoint state BEFORE the adjoint collision,
                     i.e. g = P^T S^T B^T f_hat. Contracting against the
                     end-of-step f_hat instead is wrong by one collision.
    """
    fwd = forward
    u_sq = fwd.ux**2 + fwd.uy**2
    f_neq = fwd.f - fwd.f_eq
    indirect = 0.5 * fwd.omega_eff**2 * np.sum(adjoint.g * f_neq, axis=0)

    d_alpha = d_alpha_d_rho_bar(fwd.rho_bar, alpha_max, q)
    if beta is not None:
        d_proj = d_heaviside_d_rho_e(fwd.rho_e, beta, eta)
    else:
        d_proj = 1.0

    G = d_alpha * d_proj * (u_sq + indirect)
    G[fwd.obstacle] = 0.0
    return G