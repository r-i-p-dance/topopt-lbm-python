import numpy as np
from topopt.src.core.forward import BrinkmanLattice
from lbm.src.core.kernels import (nb_zou_he_velocity_west,
                                  nb_zou_he_pressure_south)


class PipeBendForward(BrinkmanLattice):
    """Pipe bend: parabolic velocity inlet on the west wall, pressure outlet
    on the south wall, solid walls elsewhere.

    Openings are defined as fractions of the grid so the geometry scales
    across resolutions. Re is based on the inlet width.
    """

    def __init__(self, Re, inlet_lo, inlet_hi,
                 outlet_lo, outlet_hi, **kwargs):
        super().__init__(**kwargs)

        self.j_from = int(inlet_lo * self.ny)
        self.j_to = int(inlet_hi * self.ny)
        self.i_from = int(outlet_lo * self.nx)
        self.i_to = int(outlet_hi * self.nx)
        self.rho_out = 1.0
        self.periodic_x = True          # streaming must be a permutation

        # Re on the inlet width: set Re, derive u_max
        self.inlet_width = self.j_to - self.j_from
        self.Re = Re
        self.u_max = Re * self.nu / self.inlet_width
        ma = self.u_max * np.sqrt(3.0)
        if ma > 0.1:
            raise ValueError(
                f"Mach {ma:.3f} > 0.1. Lower Re, lower tau, or raise ny "
                f"(Re={Re}, tau={self.tau_lbm}, u_max={self.u_max:.4f}).")

        # parabolic inlet profile across the opening
        y = np.arange(self.inlet_width)
        w = self.inlet_width
        self.u_profile = (-4.0 * self.u_max / w**2) * y * (y - w)

        self._build_walls()

    def _build_walls(self):
        """Solid everywhere on the outer boundary except the two openings."""
        obs = np.zeros((self.nx, self.ny), dtype=bool)
        obs[:, -1] = True                       # top
        obs[-1, :] = True                       # right
        obs[0, :] = True                        # left ...
        obs[0, self.j_from:self.j_to] = False   # ... except inlet
        obs[:, 0] = True                        # bottom ...
        obs[self.i_from:self.i_to, 0] = False   # ... except outlet
        self.obstacle = obs

    def apply_boundary_conditions(self):
        nb_zou_he_velocity_west(self.f, self.ux, self.uy, self.rho,
                                self.u_profile, self.j_from, self.j_to)
        nb_zou_he_pressure_south(self.f, self.ux, self.uy, self.rho,
                                 self.rho_out, self.i_from, self.i_to)