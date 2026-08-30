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

    def __init__(self, Re, xi, less_pressure, inlet_lo, inlet_hi,
                 outlet_lo, outlet_hi, **kwargs):
        super().__init__(**kwargs)

        if less_pressure not in ("south", "east"):
            raise ValueError("less_pressure must be 'south' or 'east'")
        if not 0.0 <= xi < 1.0:
            raise ValueError(
                f"xi = {xi} outside [0, 1). At xi = 1 the favoured outlet "
                f"takes all the flow and the other carries zero; above it "
                f"the other outlet reverses and becomes an inlet. Use "
                f"xi <= 0.7 for margin, since d_rho_nat only estimates the "
                f"branch resistance.")

        self.j_from = int(inlet_lo * self.ny)
        self.j_to = int(inlet_hi * self.ny)
        self.i_from = int(outlet_lo * self.nx)
        self.i_to = int(outlet_hi * self.nx)
        self.i_east = self.nx - 1
        self.inlet_width = self.j_to - self.j_from

        self.Re = Re
        self.u_max = Re * self.nu / self.inlet_width

        # Derive the outlet densities from the dimensionless asymmetry.
        # Both are computable before any solve: nu comes from tau, u_max
        # from Re, and the geometry from the grid.
        self.xi = xi
        self.less_pressure = less_pressure
        self.outlet_asymmetry = xi * self.natural_pressure_scale()
        self.rho_east = 1.0 - (self.outlet_asymmetry if less_pressure == "east" else 0.0)
        self.rho_south = 1.0 - (self.outlet_asymmetry if less_pressure == "south" else 0.0)

        self.periodic_x = True          # streaming must be a permutation

        self.u_profile = self.analytical_profile()
        self._check_stability_at_init()
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

    def inlet_density(self):
        """Mean rho across the inlet opening, as derived by the Zou-He BC.

        The velocity BC computes this from the populations arriving at the
        west wall from inside the domain:

            rho = (f0 + f2 + f4 + 2*(f3 + f6 + f7)) / (1 - ux)

        f3, f6 and f7 carry the flow state, so this is a READING of the
        current solution, not a constant of the problem. It has no useful
        value until the solver has relaxed: on the initial equilibrium
        (u = 0, rho = 1) the same expression collapses to 1/(1 - ux), which
        is the wrong answer by more than an order of magnitude.
        """
        return float(np.mean(self.rho[0, self.j_from:self.j_to]))

    def inlet_pressure_drop(self):
        """Mean rho at the inlet, minus the outlet reference.

        With a velocity inlet, rho at the inlet is DERIVED, not prescribed:
        it floats to whatever value drives the prescribed flux through the
        current design's resistance. This is therefore a measurement of how
        hard the design is to push flow through, and it is the quantity the
        outlet pressures should be judged against.

        Compare it to the natural scale for the domain:

            delta_rho = 24 * nu * L * u_max / H^2

        which for these parameters is about 1e-3. Outlet asymmetries much
        smaller than that are swamped by the domain's own resistance;
        asymmetries much larger dominate the flow split entirely.
        """
        return self.inlet_density() - self.rho_east

    def natural_pressure_scale(self):
        """The pressure drop the domain's own resistance demands, from the
        Poiseuille relation with L = nx and H = inlet width."""
        path_length = self.nx - 1
        return (24.0 * self.nu * path_length * self.u_max / self.inlet_width**2)

    def outlet_fluxes(self):
        """Volumetric flux through each outlet.

        The optimizer may simply wall off the higher-resistance outlet —
        a legitimate optimum for a pure dissipation objective, but it makes
        a two-outlet comparison meaningless. Watching both from iteration 0
        is how you find out.
        """
        flux_east = float(np.sum(self.ux[self.i_east, self.j_from:self.j_to]))
        flux_south = float(np.sum(self.uy[self.i_from:self.i_to, 0]))
        return flux_east, flux_south

    def _check_stability_at_init(self):
            ma = self.u_max * np.sqrt(3.0)
            if ma > 0.1:
                raise ValueError(
                    f"Mach {ma:.3f} > 0.1. Lower Re, lower tau, or raise ny.")

    def analytical_profile(self):
        y = np.arange(self.inlet_width)
        w = self.inlet_width
        return (-4.0 * self.u_max / w**2) * y * (y - w)
