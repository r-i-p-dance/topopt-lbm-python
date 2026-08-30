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

    def __init__(self, nx, ny, volume_fraction, Re, tau_lbm, q=0.1,
                 xi=0.5, less_pressure="south",
                 inlet_lo=0.7, inlet_hi=0.9,
                 outlet_lo=0.7, outlet_hi=0.9):
        self.xi = xi
        self.less_pressure = less_pressure
        self.inlet_lo, self.inlet_hi = inlet_lo, inlet_hi
        self.outlet_lo, self.outlet_hi = outlet_lo, outlet_hi
        super().__init__(nx, ny, volume_fraction, Re, tau_lbm, q)

    def _setup_geometry(self):
        nx, ny = self.nx, self.ny
        j_from, j_to = int(self.inlet_lo * ny), int(self.inlet_hi * ny)
        i_from, i_to = int(self.outlet_lo * nx), int(self.outlet_hi * nx)

        for wall in (np.s_[:, -1], np.s_[0, :], np.s_[-1, :], np.s_[:, 0]):
            self.fixed_mask[wall] = True
            self.fixed_values[wall] = 0.0

        for opening in (np.s_[0, j_from:j_to], np.s_[-1, j_from:j_to],
                        np.s_[i_from:i_to, 0]):
            self.fixed_mask[opening] = True
            self.fixed_values[opening] = 1.0

    def build_solvers(self, alpha_max, beta):
        self.forward = PressureForkForward(
            nx=self.nx, ny=self.ny, tau_lbm=self.tau_lbm, Re=self.Re,
            xi=self.xi, less_pressure=self.less_pressure,
            rho_e=self.rho_e.copy(), alpha_max=alpha_max, q=self.q, beta=beta,
            inlet_lo=self.inlet_lo, inlet_hi=self.inlet_hi,
            outlet_lo=self.outlet_lo, outlet_hi=self.outlet_hi)
        self.adjoint = PressureForkAdjoint(self.forward)
        return self.forward, self.adjoint