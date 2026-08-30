import numpy as np
from topopt.src.core.forward import BrinkmanLattice
from lbm.src.core.kernels import (nb_zou_he_velocity_west,
                                  nb_zou_he_velocity_south,
                                  nb_zou_he_pressure_east)


class SplitForward(BrinkmanLattice):
    """One velocity inlet (west), two outlets with a PRESCRIBED flow split.

    Boundary specification
    ----------------------
      west  : velocity, peak u_max = Re * nu / H. Fixes the total flux Q_in.
      south : velocity, scaled so its flux is exactly split * Q_in.
      east  : pressure, rho = 1. The anchor.

    Mass conservation then gives, with no further machinery:

        Q_east = Q_in - Q_south = (1 - split) * Q_in

    so BOTH outlets carry outward flow for any 0 < split < 1. This is an
    algebraic consequence of prescribing two fluxes in a conserved system,
    not a condition that has to be checked. The inlet pressure is whatever
    it needs to be and never has to be inspected.

    Why not prescribe outlet PRESSURES instead (the earlier PressureFork
    approach): the flow direction then depends on whether the inlet pressure
    exceeds both outlets, and the inlet pressure is an OUTPUT that depends
    on the design's resistance. That resistance changes by orders of
    magnitude during a run as alpha_max ramps and the topology forms —
    measured directly: the inlet-to-outlet drop went +1.6e-5 -> +4.6e-5 ->
    -1.9e-5 within one run, changing sign. No fixed outlet pressure
    difference can be correct throughout.

    A second benefit: the split is enforced by the boundary condition, so
    the optimizer CANNOT wall off a branch. Doing so would force the
    prescribed flux through cells with alpha ~ alpha_max, and the objective
    J = sum(alpha |u|^2) would explode. With prescribed pressures, blocking
    a branch is cheap and needs an extra flux constraint (hence MMA) to
    prevent.

    Exactly one boundary must prescribe pressure — it sets the absolute
    density level, without which the whole field drifts.

    Extends to N outlets unchanged: prescribe velocity on N-1 of them with
    splits summing to less than 1, pressure on the last.
    """

    def __init__(self, Re, split, inlet_lo, inlet_hi,
                 outlet_lo, outlet_hi, **kwargs):
        super().__init__(**kwargs)

        if not 0.0 < split < 1.0:
            raise ValueError(
                f"split = {split} outside (0, 1). It is the fraction of the "
                f"inlet flux sent to the south outlet; the east outlet takes "
                f"the remainder, so both endpoints starve one branch.")

        self.j_from = int(inlet_lo * self.ny)
        self.j_to = int(inlet_hi * self.ny)
        self.i_from = int(outlet_lo * self.nx)
        self.i_to = int(outlet_hi * self.nx)
        self.i_east = self.nx - 1

        self.inlet_width = self.j_to - self.j_from
        self.outlet_width = self.i_to - self.i_from
        self.split = split
        self.rho_east = 1.0                     # the pressure anchor

        self.Re = Re
        self.u_max = Re * self.nu / self.inlet_width

        self.periodic_x = True          # streaming must be a permutation

        self.u_profile = self._inlet_profile()
        self.flux_in = float(np.sum(self.u_profile))
        self.u_profile_south = self._south_profile()
        self.flux_south_target = float(np.sum(-self.u_profile_south))

        self._check_stability_at_init()
        self._build_walls()

    # ------------------------------------------------------------------
    def _inlet_profile(self):
        """Parabolic profile across the inlet, peak u_max, flowing +x."""
        y_local = np.arange(self.inlet_width)
        width = self.inlet_width
        return (-4.0 * self.u_max / width**2) * y_local * (y_local - width)

    def _south_profile(self):
        """Parabolic profile across the south outlet, scaled so its flux is
        exactly split * Q_in, flowing -y (out of the domain).

        The shape is chosen first and then rescaled by the DISCRETE sum, not
        by an analytic integral: a parabola summed over a handful of cells
        differs from its integral by several percent, and that error would
        show up directly as a wrong split.
        """
        x_local = np.arange(self.outlet_width)
        width = self.outlet_width
        shape = (-4.0 / width**2) * x_local * (x_local - width)   # peak 1
        shape_sum = float(np.sum(shape))
        if shape_sum <= 0.0:
            raise ValueError(
                f"South outlet spans {self.outlet_width} cells — too few for "
                f"a profile. Raise nx or widen the opening.")
        # negative: outward through the south face
        return -shape * (self.split * self.flux_in / shape_sum)

    def _check_stability_at_init(self):
        """Mach check on BOTH prescribed profiles.

        A narrow south outlet carrying a large split can exceed the low-Mach
        limit even when the inlet is comfortable, because the same flux is
        squeezed through fewer cells.
        """
        for name, profile in (("inlet", self.u_profile),
                              ("south outlet", self.u_profile_south)):
            mach = float(np.max(np.abs(profile))) * np.sqrt(3.0)
            if mach > 0.1:
                raise ValueError(
                    f"Mach {mach:.3f} at the {name} exceeds 0.1 — LBM "
                    f"low-Mach assumption violated. Lower Re, lower split, "
                    f"widen the opening, or raise the resolution.")
        if self.tau_lbm <= 0.55:
            raise ValueError(f"tau={self.tau_lbm} too close to 0.5.")

    def _build_walls(self):
        obstacle = np.zeros((self.nx, self.ny), dtype=bool)
        obstacle[:, -1] = True                              # north
        obstacle[0, :] = True                               # west ...
        obstacle[0, self.j_from:self.j_to] = False          # ... inlet
        obstacle[-1, :] = True                              # east ...
        obstacle[-1, self.j_from:self.j_to] = False         # ... outlet, anchor
        obstacle[:, 0] = True                               # south ...
        obstacle[self.i_from:self.i_to, 0] = False          # ... outlet, split
        self.obstacle = obstacle

    def apply_boundary_conditions(self):
        nb_zou_he_velocity_west(self.f, self.ux, self.uy, self.rho,
                                self.u_profile, self.j_from, self.j_to)
        nb_zou_he_velocity_south(self.f, self.ux, self.uy, self.rho,
                                 self.u_profile_south,
                                 self.i_from, self.i_to)
        nb_zou_he_pressure_east(self.f, self.ux, self.uy, self.rho,
                                self.rho_east, self.j_from, self.j_to)

    # ------------------------------------------------------------------
    def mass_balance(self):
        """Q_in - sum(Q_out), relative to Q_in.

        Should sit at the LBM compressibility level, O(Ma^2) ~ 1e-4 here.
        A growing residual means two boundary index ranges overlap, so one
        opening is written twice per step — the one failure mode this
        formulation still admits.
        """
        flux_east, flux_south = self.outlet_fluxes()
        return (self.q_in - flux_east - flux_south) / self.q_in

    def hydraulic_power(self):
        """Pumping power the design demands: Q_in * delta_p, with p = rho/3.

        Physically interpretable and comparable across alpha, unlike
        J = sum(alpha |u|^2), which grows with alpha even as the design
        improves.
        """
        delta_p = (self.inlet_density() - self.rho_east) / 3.0
        return self.q_in * delta_p
    
    def outlet_fluxes(self):
        """Measured flux through each outlet, signed positive outward.

        South should reproduce split * Q_in to solver tolerance (it is
        prescribed). East should reproduce (1 - split) * Q_in by mass
        conservation. A discrepancy in either means the solve has not
        converged, or a boundary range overlaps another.
        """
        flux_east = float(np.sum(self.ux[self.i_east,
                                         self.j_from:self.j_to]))
        flux_south = float(np.sum(-self.uy[self.i_from:self.i_to, 0]))
        return flux_east, flux_south

    def inlet_density(self):
        """Mean rho at the inlet, derived by the velocity BC.

        Diagnostic only. Unlike the pressure-outlet formulation, nothing
        depends on its value — the flow direction is fixed by mass
        conservation, not by whether this exceeds the outlet densities.
        """
        return float(np.mean(self.rho[0, self.j_from:self.j_to]))

    def south_density(self):
        return float(np.mean(self.rho[self.i_from:self.i_to, 0]))