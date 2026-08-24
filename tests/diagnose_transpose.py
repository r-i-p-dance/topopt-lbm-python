"""
Transpose verification for the discrete adjoint.

METHOD RULE (learned the hard way — two bugs hid behind violations of it):
  * Every operator is exercised through the OBJECT'S OWN METHOD, never a
    hand-built reconstruction. A reconstruction can pass while the real
    method fails, because it doesn't touch the same state.
  * Every attribute a method mutates is saved and restored around the call.

ORDERING RULE:
  Forward runs C, P, S, B  =>  M = B . S . P . C
  Transpose is M^T = C^T . P^T . S^T . B^T, which APPLIES as
  B^T first, then S^T, then P^T, then C^T (i.e. reverse execution order).
"""

import numpy as np
from fixtures.forward import PressureBrinkman
from fixtures.adjoint import PressureAdjoint
from topopt.src.cases.pipe_bend.adjoint import PipeBendAdjoint
from topopt.src.cases.pipe_bend.forward import PipeBendForward


# --- state handling -------------------------------------------------------

FWD_STATE = ("f", "f_new", "ux", "uy", "rho", "f_eq")
ADJ_STATE = ("f", "f_new")


def snapshot(obj, names):
    return {n: getattr(obj, n).copy() for n in names}


def restore(obj, snap):
    for n, v in snap.items():
        setattr(obj, n, v.copy())


# def setup(ny=8, aspect=3):
#     nx = aspect * ny
#     rho_e = 0.5 * np.ones((nx, ny))
#     fwd = PressureBrinkman(ny=ny, aspect=aspect, tau_lbm=0.933, Re=1.0,
#                            alpha_max=100.0, q=0.1, beta=None, rho_e=rho_e)
#     fwd.converge(tol=1e-10)
#     adj = PressureAdjoint(fwd)
#     adj.source = np.zeros_like(adj.source)      # linear part only
#     return fwd, adj

def setup(ny=32, aspect=1):
    nx = aspect * ny
    rho_e = 0.5 * np.ones((nx, ny))
    fwd = PipeBendForward(nx=nx, ny=ny, tau_lbm=0.6, Re=10.0,
                          rho_e=rho_e, alpha_max=100.0, q=0.1, beta=None)
    fwd.converge(tol=1e-10)
    adj = PipeBendAdjoint(fwd)
    adj.source = np.zeros_like(adj.source)
    return fwd, adj


# --- operator application -------------------------------------------------

def fwd_apply(fwd, base, ops, x, eps):
    """Jacobian of `ops` at f*, applied to x, by central difference.
    Full forward state is reset before each evaluation."""
    def run(state):
        restore(fwd, base)
        fwd.f = state.copy()
        for op in ops:
            op()
        return fwd.f.copy()
    plus = run(base["f"] + eps * x)
    minus = run(base["f"] - eps * x)
    restore(fwd, base)
    return (plus - minus) / (2.0 * eps)


def adj_apply(adj, ops, x):
    """Apply the adjoint ops to x via the object's own methods."""
    saved = snapshot(adj, ADJ_STATE)
    adj.f = x.copy()
    adj.f_new = np.zeros_like(x)
    for op in ops:
        op()
    out = adj.f.copy()
    restore(adj, saved)
    return out


def dot_test(fwd, adj, fwd_ops, adj_ops, eps, seed=0):
    base = snapshot(fwd, FWD_STATE)
    rng = np.random.RandomState(seed)
    u = rng.randn(*fwd.f.shape)
    v = rng.randn(*fwd.f.shape)
    Mu = fwd_apply(fwd, base, fwd_ops, u, eps)
    MTv = adj_apply(adj, adj_ops, v)
    return float(np.sum(Mu * v)), float(np.sum(u * MTv))


def report(fwd, adj, name, fwd_ops, adj_ops, tol=1e-4):
    """Report at two eps values. If rel is unchanged, the error is real;
    if it scales with eps, it is finite-difference noise."""
    line = f"  {name:12s}"
    worst = 0.0
    for eps in (1e-6, 1e-7):
        lhs, rhs = dot_test(fwd, adj, fwd_ops, adj_ops, eps)
        rel = abs(lhs - rhs) / max(abs(rhs), 1e-300)
        worst = max(worst, rel)
        line += f"   eps={eps:.0e}: rel={rel:.2e}"
    print(("OK   " if worst < tol else "FAIL ") + line)


def main():
    fwd, adj = setup()
    print(f"forward converged in {fwd.it} steps")
    print(f"obstacle   fwd={fwd.obstacle.sum():3d}  adj={adj.obstacle.sum():3d}"
          f"  shared={adj.obstacle is fwd.obstacle}")
    print(f"periodic_x fwd={fwd.periodic_x}  adj={adj.periodic_x}")
    print()

    C = [fwd.macro, fwd.equilibrium, fwd.collision]
    P = [fwd.bounce_back_obstacle]
    S = [fwd.stream]
    B = [fwd.apply_boundary_conditions]

    Ct = [adj.collision]
    Pt = [adj.bounce_back_obstacle]
    St = [adj.stream]
    Bt = [adj.apply_boundary_conditions]

    print("--- individual operators ---")
    report(fwd, adj, "collision",   C, Ct)
    report(fwd, adj, "bounce_back", P, Pt)
    report(fwd, adj, "stream",      S, St)
    report(fwd, adj, "zou_he_bc",   B, Bt)

    print()
    print("--- compositions (adjoint applies in REVERSE order) ---")
    report(fwd, adj, "C",       C,           Ct)
    report(fwd, adj, "P.C",     C + P,       Pt + Ct)
    report(fwd, adj, "S.P.C",   C + P + S,   St + Pt + Ct)
    report(fwd, adj, "B.S.P.C", C + P + S + B, Bt + St + Pt + Ct)


if __name__ == "__main__":
    main()