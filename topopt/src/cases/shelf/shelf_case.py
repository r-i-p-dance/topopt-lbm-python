import numpy as np
from topopt.src.cases.base_case import BaseCase
from topopt.src.cases.shelf.forward import ShelfForward
from topopt.src.cases.shelf.adjoint import ShelfAdjoint


class ShelfCase(BaseCase):
    """Pipe bend obstructed by a fixed cantilever below the inlet."""

    def __init__(self, inlet_lo=0.5, inlet_hi=0.7,
                 outlet_lo=0.4, outlet_hi=0.6, shelf_extent=0.5, **kwargs):
        self.inlet_lo, self.inlet_hi = inlet_lo, inlet_hi
        self.outlet_lo, self.outlet_hi = outlet_lo, outlet_hi
        self.shelf_extent = shelf_extent
        super().__init__(**kwargs)

    def _setup_geometry(self):
        nx, ny = self.nx, self.ny
        j_from, j_to = int(self.inlet_lo * ny), int(self.inlet_hi * ny)
        i_from, i_to = int(self.outlet_lo * nx), int(self.outlet_hi * nx)
        inlet_width = j_to - j_from

        for wall in (np.s_[:, -1], np.s_[0, :], np.s_[-1, :], np.s_[:, 0]):
            self.fixed_mask[wall] = True
            self.fixed_values[wall] = 0.0

        for opening in (np.s_[0, j_from:j_to], np.s_[i_from:i_to, 0]):
            self.fixed_mask[opening] = True
            self.fixed_values[opening] = 1.0

        # The shelf is fixed geometry, not a design variable: pin it solid
        # so the optimizer cannot dissolve it and so the volume constraint
        # counts it correctly.
        shelf = np.s_[0:int(self.shelf_extent * nx),
                      j_from - inlet_width//2:j_from]
        self.fixed_mask[shelf] = True
        self.fixed_values[shelf] = 0.0

    def build_solvers(self, alpha_max, beta):
        self.forward = ShelfForward(
            nx=self.nx, ny=self.ny, tau_lbm=self.tau_lbm, Re=self.Re,
            rho_e=self.rho_e.copy(), alpha_max=alpha_max, q=self.q, beta=beta,
            inlet_lo=self.inlet_lo, inlet_hi=self.inlet_hi,
            outlet_lo=self.outlet_lo, outlet_hi=self.outlet_hi,
            shelf_extent=self.shelf_extent)
        self.adjoint = ShelfAdjoint(self.forward)
        return self.forward, self.adjoint