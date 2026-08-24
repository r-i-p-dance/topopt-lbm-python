"""Verify the adjoint source equals dJ/df exactly.

The discrete adjoint requires S = dJ/df with no weighting. This checks the
production dJ_df() against a direct finite-difference derivative of the
objective with respect to individual populations.

Run from the project root:
    python tests\\diagnose_source.py
"""

import numpy as np
from fixtures.forward import PressureBrinkman
from topopt.src.core.objective import dissipation_objective, dJ_df


def main(ny=8, aspect=3, alpha_max=100.0, q=0.1):
    nx = aspect * ny
    rho_e = 0.5 * np.ones((nx, ny))
    fwd = PressureBrinkman(ny=ny, aspect=aspect, tau_lbm=0.933, Re=1.0,
                           alpha_max=alpha_max, q=q, beta=None,
                           rho_e=rho_e, periodic_x=True)
    fwd.converge(tol=1e-10)

    S = dJ_df(fwd.rho_bar, fwd.ux, fwd.uy, fwd.rho, fwd.cx, fwd.cy,
              alpha_max, q)

    base_f = fwd.f.copy()

    def J_of_f(state):
        """Objective as a function of f alone: recompute macros, then J."""
        saved = fwd.f
        fwd.f = state.copy()
        fwd.macro()
        J = dissipation_objective(fwd.rho_bar, fwd.ux, fwd.uy, alpha_max, q)
        fwd.f = saved
        fwd.macro()
        return J

    rng = np.random.RandomState(0)
    eps = 1e-7
    print(f"{'k':>2} {'i':>3} {'j':>3} {'code':>13} {'FD':>13} {'ratio':>8}")
    for _ in range(12):
        k = rng.randint(0, 9)
        i = rng.randint(1, nx - 1)
        j = rng.randint(1, ny - 1)

        fp = base_f.copy(); fp[k, i, j] += eps
        fm = base_f.copy(); fm[k, i, j] -= eps
        fd = (J_of_f(fp) - J_of_f(fm)) / (2 * eps)

        r = S[k, i, j] / fd if abs(fd) > 1e-16 else np.nan
        print(f"{k:2d} {i:3d} {j:3d} {S[k,i,j]:+13.5e} {fd:+13.5e} {r:8.4f}")

if __name__ == "__main__":
    main()