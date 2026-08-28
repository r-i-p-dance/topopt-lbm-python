import numpy as np
from topopt.src.core.forward import BrinkmanLattice
from lbm.src.core.kernels import (nb_zou_he_velocity_west,
                                  nb_zou_he_pressure_east,
                                  nb_zou_he_pressure_south)


class PressureForkForward(BrinkmanLattice):
    """One velocity inlet (west), two pressure outlets (east, south).

    Geometry: the inlet sits high on the west wall; one outlet faces it
    directly across on the east wall; the other sits on the south wall at
    the mirrored fraction along x.

    Physics note: with a velocity inlet, the inlet density is COMPUTED, not
    prescribed - it floats to whatever value drives the prescribed flux
    through the current design's resistance. The controlled variable is
    therefore the difference between the two OUTLET densities. The domain's
    own resistance corresponds to a drop of ~1e-3 in rho at these
    parameters, so an outlet difference of that order makes the pressure
    boundary condition compete evenly with geometry for the flow split.
    """

    def __init__(self, Re, rho_east, rho_south,
                 inlet_lo, inlet_hi,
                 outlet_lo, outlet_hi, **kwargs):
        super().__init__(**kwargs)
        self.rho_east = rho_east
        self.rho_south = rho_south
        self.j_from = int(inlet_lo * self.ny)
        self.j_to = int(inlet_hi * self.ny)
        self.i_from = int(outlet_lo * self.nx)
        self.i_to = int(outlet_hi * self.nx)
        self.i_east = self.nx - 1

        self.periodic_x = True          # streaming must be a permutation

        self.inlet_width = self.j_to - self.j_from
        self.Re = Re
        self.u_max = Re * self.nu / self.inlet_width
        ma = self.u_max * np.sqrt(3.0)
        if ma > 0.1:
            raise ValueError(
                f"Mach {ma:.3f} > 0.1. Lower Re, lower tau, or raise ny.")

        y = np.arange(self.inlet_width)
        w = self.inlet_width
        self.u_profile = (-4.0 * self.u_max / w**2) * y * (y - w)

        self._build_walls()

    def _build_walls(self):
        obs = np.zeros((self.nx, self.ny), dtype=bool)
        obs[:, -1] = True                                    # north
        obs[0, :] = True                                     # west ...
        obs[0, self.j_from:self.j_to] = False                # ... inlet
        obs[-1, :] = True                                    # east ...
        obs[-1, self.j_from:self.j_to] = False               # ... outlet A
        obs[:, 0] = True                                     # south ...
        obs[self.i_from:self.i_to, 0] = False                # ... outlet B
        self.obstacle = obs

    def apply_boundary_conditions(self):
        nb_zou_he_velocity_west(self.f, self.ux, self.uy, self.rho,
                                self.u_profile, self.j_from, self.j_to)
        nb_zou_he_pressure_east(self.f, self.ux, self.uy, self.rho,
                                self.rho_east, self.j_from, self.j_to)
        nb_zou_he_pressure_south(self.f, self.ux, self.uy, self.rho,
                                 self.rho_south, self.i_from, self.i_to)