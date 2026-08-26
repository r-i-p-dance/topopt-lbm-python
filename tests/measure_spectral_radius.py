"""Measure the spectral radius of the forward iteration and its adjoint.

Both are estimated by power iteration on the LINEAR part of one solver step:
the forward's Jacobian about the converged state f* (extracted by central
differences, since collision is nonlinear), and the adjoint operator with
its source switched off.

Why this is worth running: eigenvalues are invariant under transpose, so
spec(M^T) = spec(M) exactly. If the forward converges its spectral radius is
below 1, and an exact adjoint must therefore converge too. A mismatch is
PROOF that the adjoint is not the transpose of the forward — no judgement
call required. This is what localised the streaming-permutation and
obstacle-mask bugs when the per-operator dot-product tests were all passing.

A radius of exactly 1.000000 indicates a neutral mode rather than a bug:
with periodic streaming, total mass is conserved, so a uniform shift in f is
an eigenvector with eigenvalue 1. Convergence is then governed by the second
largest eigenvalue.

Run from the project root:
    python tests\\measure_spectral_radius.py
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
f_star = fwd.f.copy()
ux_s, uy_s, rho_s, feq_s = (fwd.ux.copy(), fwd.uy.copy(),
                            fwd.rho.copy(), fwd.f_eq.copy())

def forward_jacobian(x, eps=1e-7):
    """J @ x for the full forward step, by central differences about f*."""
    def full_step(state):
        fwd.f = state.copy()
        fwd.ux, fwd.uy = ux_s.copy(), uy_s.copy()
        fwd.rho, fwd.f_eq = rho_s.copy(), feq_s.copy()
        fwd.step()
        return fwd.f.copy()
    out = (full_step(f_star + eps*x) - full_step(f_star - eps*x)) / (2*eps)
    fwd.f, fwd.ux, fwd.uy = f_star.copy(), ux_s.copy(), uy_s.copy()
    fwd.rho, fwd.f_eq = rho_s.copy(), feq_s.copy()
    return out

adj = PressureAdjoint(fwd)
zero_src = np.zeros_like(adj.source)

def adjoint_operator(x):
    """M^T @ x — the pure linear part, source switched off."""
    saved = adj.source
    adj.f, adj.source = x.copy(), zero_src
    adj.step()
    out = adj.f.copy()
    adj.source = saved
    return out

def power_iteration(op, shape, n=300, seed=0):
    rng = np.random.RandomState(seed)
    v = rng.randn(*shape); v /= np.linalg.norm(v)
    lam = 0.0
    for k in range(n):
        w = op(v)
        lam = np.linalg.norm(w)
        if lam < 1e-300 or not np.isfinite(lam):
            return lam
        v = w / lam
    return lam

rho_M  = power_iteration(forward_jacobian, fwd.f.shape)
rho_MT = power_iteration(adjoint_operator,  fwd.f.shape)
print(f"rho(M)   [forward Jacobian] = {rho_M:.6f}")
print(f"rho(M^T) [adjoint operator] = {rho_MT:.6f}")