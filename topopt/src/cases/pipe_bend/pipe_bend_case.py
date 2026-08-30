import numpy as np
from topopt.src.cases.base_case import BaseCase
from topopt.src.cases.pipe_bend.forward import PipeBendForward
from topopt.src.cases.pipe_bend.adjoint import PipeBendAdjoint


class PipeBendCase(BaseCase):
    """Fluid enters the west wall and must exit the south wall; the optimizer
    carves a bent channel connecting them."""

    def __init__(self, inlet_lo=0.7, inlet_hi=0.9,
                 outlet_lo=0.7, outlet_hi=0.9, **kwargs):
        self.inlet_lo = inlet_lo
        self.inlet_hi = inlet_hi
        self.outlet_lo = outlet_lo
        self.outlet_hi = outlet_hi
        super().__init__(**kwargs)

    def _setup_geometry(self):
        nx, ny = self.nx, self.ny
        j_from, j_to = int(self.inlet_lo * ny), int(self.inlet_hi * ny)
        i_from, i_to = int(self.outlet_lo * nx), int(self.outlet_hi * nx)

        # Outer walls: pinned solid (they are obstacles; the design there is
        # inert, but pinning keeps the volume measure honest).
        self.fixed_mask[:, -1] = True;  self.fixed_values[:, -1] = 0.0
        self.fixed_mask[-1, :] = True;  self.fixed_values[-1, :] = 0.0
        self.fixed_mask[0, :] = True;   self.fixed_values[0, :] = 0.0
        self.fixed_mask[:, 0] = True;   self.fixed_values[:, 0] = 0.0

        # Openings: pinned fluid.
        self.fixed_mask[0, j_from:j_to] = True
        self.fixed_values[0, j_from:j_to] = 1.0
        self.fixed_mask[i_from:i_to, 0] = True
        self.fixed_values[i_from:i_to, 0] = 1.0

    def build_solvers(self, alpha_max, beta):
        self.forward = PipeBendForward(
            nx=self.nx, ny=self.ny, tau_lbm=self.tau_lbm, Re=self.Re,
            rho_e=self.rho_e.copy(), alpha_max=alpha_max, q=self.q, beta=beta,
            inlet_lo=self.inlet_lo, inlet_hi=self.inlet_hi,
            outlet_lo=self.outlet_lo, outlet_hi=self.outlet_hi)
        self.adjoint = PipeBendAdjoint(self.forward)
        return self.forward, self.adjoint