import numpy as np
from tests.fixtures.forward import PressureBrinkman
from tests.fixtures.adjoint import PressureAdjoint
from topopt.src.core.objective import dissipation_objective
from topopt.src.core.sensitivity import assemble_sensitivity


def test_gradient_matches_finite_difference():
    """The centerpiece test: verify adjoint gradient against FD to 4+ digits.
    
    On a very small grid so FD is affordable. Perturb rho_e at a few random
    cells, compute (J(+eps) - J(-eps)) / (2*eps), compare with assembled G.
    """
    beta       = 5.0
    ny         = 16
    aspect     = 5
    nx         = int(aspect * ny)
    alpha_max  = 100.0
    q          = 0.1
    tau        = 0.933
    Re         = 1.0
    tol_solver = 1e-12

    rho_e = 0.5 * np.ones((nx, ny))
    kwargs = dict(ny=ny, aspect=aspect, tau_lbm=tau, Re=Re,
                      alpha_max=alpha_max, q=q, beta=beta, periodic_x = True)      # <-- no projection

    # Solve forward
    fwd = PressureBrinkman(rho_e=rho_e.copy(), **kwargs)
    fwd.converge(tol=tol_solver)

    # Solve adjoint
    adj = PressureAdjoint(fwd)
    adj.converge(tol=tol_solver)

    # Assemble adjoint-based sensitivity
    G = assemble_sensitivity(fwd, adj, alpha_max, q, beta=beta)

    # FD at 5 random cells
    rng = np.random.RandomState(69)
    cells = [(rng.randint(1, nx-1), rng.randint(1, ny-1)) for _ in range(5)]
    eps = 1e-4

    for (i, j) in cells:
        # +eps
        rho_plus = rho_e.copy()
        rho_plus[i, j] += eps
        fwd_p = PressureBrinkman(ny=ny, aspect=aspect, tau_lbm=tau, Re=Re,
                                alpha_max=alpha_max, q=q, rho_e=rho_plus, beta=beta, periodic_x=True)
        fwd_p.converge(tol=tol_solver)
        J_plus = dissipation_objective(fwd_p.rho_bar, fwd_p.ux, fwd_p.uy, alpha_max, q)

        # -eps
        rho_minus = rho_e.copy()
        rho_minus[i, j] -= eps
        fwd_m = PressureBrinkman(ny=ny, aspect=aspect, tau_lbm=tau, Re=Re,
                                alpha_max=alpha_max, q=q, rho_e=rho_minus, beta=beta, periodic_x=True)
        fwd_m.converge(tol=tol_solver)
        J_minus = dissipation_objective(fwd_m.rho_bar, fwd_m.ux, fwd_m.uy, alpha_max, q)

        G_fd = (J_plus - J_minus) / (2 * eps)
        G_adj = G[i, j]

        rel_err = abs(G_adj - G_fd) / (abs(G_fd) + 1e-12)
        assert rel_err < 1e-3, \
            f"At cell ({i},{j}): adjoint={G_adj}, FD={G_fd}, rel_err={rel_err}"



