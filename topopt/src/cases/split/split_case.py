import numpy as np
from topopt.src.cases.base_case import BaseCase
from topopt.src.cases.split.forward import SplitForward
from topopt.src.cases.split.adjoint import SplitAdjoint


class SplitCase(BaseCase):
    """One inlet, two outlets, with the flow split PRESCRIBED.

    Two dimensionless inputs, both resolution-independent by construction:

      Re     total flux through the inlet
      split  fraction of that flux sent to the south outlet; the east
             outlet takes 1 - split by mass conservation

    Unlike an outlet-pressure specification, neither depends on the domain's
    resistance, so the same pair of numbers describes the same problem at
    any grid size and at any point during the optimization.

    Run twice with split = 0.3 and split = 0.7 for the swap comparison.
    """

    def __init__(self, nx, ny, volume_fraction, Re, tau_lbm, q=0.1,
                 split=0.5,
                 inlet_lo=0.7, inlet_hi=0.9,
                 outlet_lo=0.7, outlet_hi=0.9):
        self.split = split
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

        for opening in (np.s_[0, j_from:j_to],        # inlet
                        np.s_[-1, j_from:j_to],       # east outlet
                        np.s_[i_from:i_to, 0]):       # south outlet
            self.fixed_mask[opening] = True
            self.fixed_values[opening] = 1.0

    def build_solvers(self, alpha_max, beta):
        self.forward = SplitForward(
            nx=self.nx, ny=self.ny, tau_lbm=self.tau_lbm, Re=self.Re,
            rho_e=self.rho_e.copy(), alpha_max=alpha_max, q=self.q, beta=beta,
            split=self.split,
            inlet_lo=self.inlet_lo, inlet_hi=self.inlet_hi,
            outlet_lo=self.outlet_lo, outlet_hi=self.outlet_hi)
        self.adjoint = SplitAdjoint(self.forward)
        return self.forward, self.adjoint