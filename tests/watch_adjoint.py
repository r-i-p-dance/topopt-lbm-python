"""Watch the adjoint converge now that all transposes are verified.

Run from the project root:
    python tests\\watch_adjoint.py
"""

import numpy as np
from fixtures.forward import PressureBrinkman
from fixtures.adjoint import PressureAdjoint

ny, aspect = 8, 3
nx = aspect * ny
rho_e = 0.5 * np.ones((nx, ny))

fwd = PressureBrinkman(ny=ny, aspect=aspect, tau_lbm=0.933, Re=1.0,
                       alpha_max=100.0, q=0.1, beta=None, rho_e=rho_e)
fwd.converge(tol=1e-10)
print(f"forward converged in {fwd.it} steps")

adj = PressureAdjoint(fwd)
print(f"peak|source| = {np.max(np.abs(adj.source)):.3e}\n")

old = adj.f.copy()
for step in range(1, 60001):
    adj.step()
    if step % 500 == 0:
        change = np.max(np.abs(adj.f - old))
        peak = np.max(np.abs(adj.f))
        total = np.sum(adj.f)                 # tracks the suspected neutral mode
        old = adj.f.copy()
        print(f"step {step:6d}  change={change:.3e}  peak={peak:.3e}  sum={total:+.6e}")
        if not np.isfinite(peak):
            print("DIVERGED"); break
        if change < 1e-12:
            print(f"CONVERGED at step {step}"); break