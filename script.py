import numpy as np
from tests.fixtures.forward import PressureBrinkman
from tests.fixtures.adjoint import PressureAdjoint
from tests.test_transpose import _dot_test, _setup
from topopt.src.core.adjoint_base import AdjointLattice
from topopt.src.core.objective import dissipation_objective
from topopt.src.core.sensitivity import assemble_sensitivity

# Re  = 1.0
# ny  = 16
# tol = 1e-8

# brinkman = record_development(PressureBrinkman, nx=5*ny, ny=ny, Re=Re, tol=tol,
#                     every_min=5, every_max=100, accelerate_over=3000, 
#                     path=f"results/tests/anim_Ny{ny}_BrinkmanPressure_Re{int(Re)}.gif",
#                     cmap='RdBu_r', interpolation='nearest', 
#                     print_progress=False)

# ref = record_development(PressurePoiseuille, nx=5*ny, ny=ny, Re=Re, tol=tol,
#                     every_min=5, every_max=100, accelerate_over=3000, 
#                     path=f"results/tests/anim_Ny{ny}_PressurePoiseuille_Re{int(Re)}.gif",
#                     cmap='RdBu_r', interpolation='nearest', 
#                     print_progress=False)

# u_ref = np.sqrt(ref.ux**2 + ref.uy**2)
# u_brinkman = np.sqrt(brinkman.ux**2 + brinkman.uy**2)
# plot_field_comparison(u_ref, u_brinkman,
#                       save_path=f"results/tests/diff_BrinkmanVSRef_{ny}_Re{int(Re)}_tol{tol}",
#                       title=f"BrinkmanForced vs PoiseuilleForced"
#                     )


def test_stream_transpose_via_methods():
    """Test the objects' own stream() methods, restoring BOTH buffers."""
    fwd, adj = _setup()

    def fwd_stream(x):
        sf, sfn = fwd.f, fwd.f_new
        fwd.f, fwd.f_new = x.copy(), np.zeros_like(x)
        fwd.stream()
        out = fwd.f.copy()
        fwd.f, fwd.f_new = sf, sfn
        return out

    def adj_stream(x):
        sf, sfn = adj.f, adj.f_new
        adj.f, adj.f_new = x.copy(), np.zeros_like(x)
        adj.stream()
        out = adj.f.copy()
        adj.f, adj.f_new = sf, sfn
        return out

    lhs, rhs = _dot_test(fwd_stream, adj_stream, fwd.f.shape)
    print(f"lhs={lhs:.12e} rhs={rhs:.12e} rel={abs(lhs-rhs)/abs(rhs):.2e}")
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)

test_stream_transpose_via_methods()