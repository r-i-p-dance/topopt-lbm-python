import numpy as np
from topopt.src.cases.base_case import BaseCase
from topopt.src.cases.pressure_fork.forward import PressureForkForward
from topopt.src.cases.pressure_fork.adjoint import PressureForkAdjoint


class PressureForkCase(BaseCase):
    """One inlet, two pressure outlets at different pressures.

    Run twice with rho_east and rho_south swapped: the pair of designs shows
    how much of the topology is driven by the pressure boundary condition
    versus by the geometry of the domain.
    """

    def __init__(self, outlet_asymmetry=1e-3, less_pressure="south",
                 inlet_lo=0.7, inlet_hi=0.9,
                 outlet_lo=0.7, outlet_hi=0.9, **kwargs):

        if less_pressure not in ("south", "east"):
            raise ValueError("less_pressure must be 'south' or 'east'")
        self.outlet_asymmetry = outlet_asymmetry
        self.rho_east = 1.0 - (outlet_asymmetry if less_pressure == "east" else 0.0)
        self.rho_south = 1.0 - (outlet_asymmetry if less_pressure == "south" else 0.0)
        self.inlet_lo, self.inlet_hi = inlet_lo, inlet_hi
        self.outlet_lo, self.outlet_hi = outlet_lo, outlet_hi
        super().__init__(**kwargs)

    def _setup_geometry(self):
        nx, ny = self.nx, self.ny
        j0, j1 = int(self.inlet_lo * ny), int(self.inlet_hi * ny)
        i0, i1 = int(self.outlet_lo * nx), int(self.outlet_hi * nx)

        for sl in (np.s_[:, -1], np.s_[0, :], np.s_[-1, :], np.s_[:, 0]):
            self.fixed_mask[sl] = True
            self.fixed_values[sl] = 0.0

        for sl in (np.s_[0, j0:j1], np.s_[-1, j0:j1], np.s_[i0:i1, 0]):
            self.fixed_mask[sl] = True
            self.fixed_values[sl] = 1.0

    def build_solvers(self, alpha_max, beta):
        self.forward = PressureForkForward(
            nx=self.nx, ny=self.ny, tau_lbm=self.tau_lbm, Re=self.Re,
            rho_e=self.rho_e.copy(), alpha_max=alpha_max, q=self.q, beta=beta,
            rho_east=self.rho_east, rho_south=self.rho_south,
            inlet_lo=self.inlet_lo, inlet_hi=self.inlet_hi,
            outlet_lo=self.outlet_lo, outlet_hi=self.outlet_hi)
        self.adjoint = PressureForkAdjoint(self.forward)
        return self.forward, self.adjoint