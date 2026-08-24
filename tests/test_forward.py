import numpy as np
from lbm.src.cases.pressure import PressurePoiseuille
from tests.fixtures.forward import PressureBrinkman


def test_all_fluid_matches_pressure_poiseuille():
    """With rho_e = 1 everywhere (fully fluid), BrinkmanForced should reproduce ForcedPoiseuille."""
    ny = 16
    kwargs = dict(ny=ny, Re=1.0, tau_lbm=0.933)

    ref = PressurePoiseuille(**kwargs)
    ref.converge(tol=1e-10)

    ltc = PressureBrinkman(**kwargs)                # rho_e defaults to all-ones
    ltc.converge(tol=1e-10)

    np.testing.assert_allclose(
        ltc.ux, ref.ux, atol=1e-8,
        err_msg="All-fluid BrinkmanForced doesn't match ForcedPoiseuille"
    )


def test_all_solid_kills_flow():
    """With rho_e = 0 everywhere (fully solid), velocity should decay to ~zero."""
    ny     = 16
    aspect = 5
    nx     = int(aspect * ny)
    ltc = PressureBrinkman(ny=ny, aspect=aspect, rho_e=np.zeros((nx, ny)), alpha_max=1000.0)
    ltc.run(3000)
    max_vel = np.max(np.sqrt(ltc.ux**2 + ltc.uy**2))
    assert max_vel < 1e-3, f"All-solid should suppress flow, got max_vel={max_vel}"


def test_partial_design_reduces_flow():
    """Solidifying half the domain should reduce peak velocity substantially."""
    ny = 16
    aspect = 5
    nx = int(aspect * ny)
    all_fluid = PressureBrinkman(ny=ny, aspect=aspect)
    all_fluid.converge(tol=1e-9)
    fluid_peak = np.max(np.abs(all_fluid.ux))


    partial = PressureBrinkman(ny=ny, aspect=aspect, rho_e=np.ones((nx, ny))*0.5)
    partial.converge(tol=1e-9)
    partial_peak = np.max(np.abs(partial.ux))

    assert partial_peak < 0.5 * fluid_peak, \
        f"Half-solidifying should ~halve peak velocity: fluid={fluid_peak}, partial={partial_peak}"