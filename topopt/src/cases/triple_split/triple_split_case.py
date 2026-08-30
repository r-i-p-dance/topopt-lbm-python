import numpy as np
from topopt.src.cases.base_case import BaseCase
from topopt.src.cases.triple_split.forward import TripleSplitForward
from topopt.src.cases.triple_split.adjoint import TripleSplitAdjoint


class TripleSplitCase(BaseCase):
    """One inlet, three outlets, with the flow distribution PRESCRIBED.

    Inputs, all dimensionless and resolution-independent:

      Re       total flux through the inlet
      split_1  fraction to outlet 1 (east wall, upper)
      split_2  fraction to outlet 2 (east wall, lower)

    Outlet 3 (south wall) takes 1 - split_1 - split_2 by mass conservation
    and carries the pressure anchor. A uniform distributor is
    split_1 = split_2 = 1/3.

    Openings span most of the domain deliberately: inlet top-left, two
    outlets down the right edge, one along the lower-left floor.
    """

    def __init__(self, nx, ny, volume_fraction, Re, tau_lbm, q=0.1,
                 split_1=1/3, split_2=1/3,
                 inlet_lo=0.7, inlet_hi=0.9,
                 outlet_1_lo=0.7, outlet_1_hi=0.9,
                 outlet_2_lo=0.2, outlet_2_hi=0.4,
                 outlet_3_lo=0.5, outlet_3_hi=0.7):
        self.split_1, self.split_2 = split_1, split_2
        self.inlet_lo, self.inlet_hi = inlet_lo, inlet_hi
        self.outlet_1_lo, self.outlet_1_hi = outlet_1_lo, outlet_1_hi
        self.outlet_2_lo, self.outlet_2_hi = outlet_2_lo, outlet_2_hi
        self.outlet_3_lo, self.outlet_3_hi = outlet_3_lo, outlet_3_hi
        super().__init__(nx, ny, volume_fraction, Re, tau_lbm, q)

    def _setup_geometry(self):
        nx, ny = self.nx, self.ny

        for wall in (np.s_[:, -1], np.s_[0, :], np.s_[-1, :], np.s_[:, 0]):
            self.fixed_mask[wall] = True
            self.fixed_values[wall] = 0.0

        openings = (
            np.s_[0, int(self.inlet_lo*ny):int(self.inlet_hi*ny)],
            np.s_[-1, int(self.outlet_1_lo*ny):int(self.outlet_1_hi*ny)],
            np.s_[-1, int(self.outlet_2_lo*ny):int(self.outlet_2_hi*ny)],
            np.s_[int(self.outlet_3_lo*nx):int(self.outlet_3_hi*nx), 0],
        )
        for opening in openings:
            self.fixed_mask[opening] = True
            self.fixed_values[opening] = 1.0

    def build_solvers(self, alpha_max, beta):
        self.forward = TripleSplitForward(
            nx=self.nx, ny=self.ny, tau_lbm=self.tau_lbm, Re=self.Re,
            rho_e=self.rho_e.copy(), alpha_max=alpha_max, q=self.q, beta=beta,
            split_1=self.split_1, split_2=self.split_2,
            inlet_lo=self.inlet_lo, inlet_hi=self.inlet_hi,
            outlet_1_lo=self.outlet_1_lo, outlet_1_hi=self.outlet_1_hi,
            outlet_2_lo=self.outlet_2_lo, outlet_2_hi=self.outlet_2_hi,
            outlet_3_lo=self.outlet_3_lo, outlet_3_hi=self.outlet_3_hi)
        self.adjoint = TripleSplitAdjoint(self.forward)
        return self.forward, self.adjoint