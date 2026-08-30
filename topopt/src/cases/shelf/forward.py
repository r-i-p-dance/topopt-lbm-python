import numpy as np
from topopt.src.core.forward import BrinkmanLattice
from lbm.src.core.kernels import (nb_zou_he_velocity_west,
                                  nb_zou_he_pressure_south)


class ShelfForward(BrinkmanLattice):
    """Pipe bend with a fixed cantilever obstruction below the inlet.

    Inlet high on the west wall; outlet on the south wall, mirrored to the
    LEFT of centre so the flow must pass under the shelf rather than around
    its free end on the near side.

    The shelf spans from the west wall to 3/4 of the domain length, sits
    immediately below the inlet, and is as thick as the inlet is tall. The
    question it poses: does the optimizer route straight along the shelf,
    turn sharply at its free end, and run straight to the outlet — or does
    it cut a smooth arc? At low Re dissipation is dominated by shear, which
    favours the shortest wetted path; as Re rises, momentum resists turning
    and an arc should become cheaper.
    """

    def __init__(self, Re,
                 inlet_lo, inlet_hi,
                 outlet_lo, outlet_hi,
                 shelf_extent, **kwargs):
        super().__init__(**kwargs)

        self.j_from = int(inlet_lo * self.ny)
        self.j_to = int(inlet_hi * self.ny)
        self.i_from = int(outlet_lo * self.nx)
        self.i_to = int(outlet_hi * self.nx)
        self.rho_out = 1.0
        self.periodic_x = True

        self.inlet_width = self.j_to - self.j_from
        self.Re = Re
        self.u_max = Re * self.nu / self.inlet_width

        # shelf: same thickness as the inlet, directly beneath it
        self.shelf_i_to = int(shelf_extent * self.nx)
        self.shelf_j_to = self.j_from
        self.shelf_j_from = self.j_from - self.inlet_width // 2

        self.u_profile = self.analytical_profile()
        self._check_stability_at_init()
        self._build_walls()

    def _build_walls(self):
        obstacle = np.zeros((self.nx, self.ny), dtype=bool)
        obstacle[:, -1] = True                              # north
        obstacle[-1, :] = True                              # east
        obstacle[0, :] = True                               # west ...
        obstacle[0, self.j_from:self.j_to] = False          # ... inlet
        obstacle[:, 0] = True                               # south ...
        obstacle[self.i_from:self.i_to, 0] = False          # ... outlet
        obstacle[0:self.shelf_i_to,
                 self.shelf_j_from:self.shelf_j_to] = True  # shelf
        self.obstacle = obstacle

    def apply_boundary_conditions(self):
        nb_zou_he_velocity_west(self.f, self.ux, self.uy, self.rho,
                                self.u_profile, self.j_from, self.j_to)
        nb_zou_he_pressure_south(self.f, self.ux, self.uy, self.rho,
                                 self.rho_out, self.i_from, self.i_to)

    def _check_stability_at_init(self):
        ma = self.u_max * np.sqrt(3.0)
        if ma > 0.1:
            raise ValueError(f"Mach {ma:.3f} > 0.1. Lower Re, lower tau, or raise ny.")

    def analytical_profile(self):
        y = np.arange(self.inlet_width)
        w = self.inlet_width
        return (-4.0 * self.u_max / w**2) * y * (y - w)