import numpy as np
from topopt.src.core.forward import BrinkmanLattice
from lbm.src.core.kernels import (nb_zou_he_velocity_west,
                                  nb_zou_he_velocity_east,
                                  nb_zou_he_pressure_south)


class TripleSplitForward(BrinkmanLattice):
    """One velocity inlet (west), three outlets with a PRESCRIBED split.

    Layout
    ------
      inlet     west wall,  inlet_lo  .. inlet_hi     velocity, sets Q_in
      outlet 1  east wall,  outlet_1_lo .. outlet_1_hi  velocity, split_1
      outlet 2  east wall,  outlet_2_lo .. outlet_2_hi  velocity, split_2
      outlet 3  south wall, outlet_3_lo .. outlet_3_hi  PRESSURE anchor

    Outlet 3 takes the remainder by mass conservation:

        Q_3 = (1 - split_1 - split_2) * Q_in

    so all three carry outward flow provided split_1 + split_2 < 1. As with
    the two-outlet case this is algebraic, not a condition to be checked.

    Exactly ONE boundary prescribes pressure. It sets the absolute density
    level; without it the whole field drifts, since velocity BCs derive
    density rather than fixing it. With N outlets you prescribe velocity on
    N-1 and pressure on the last, whatever N is.

    The two east outlets share one kernel with disjoint index ranges. The
    ranges MUST NOT overlap: an overlapping cell would be written twice per
    step, the adjoint would transpose an operator that was applied twice,
    and mass balance would drift. Checked at construction.
    """

    def __init__(self, Re, split_1, split_2,
                 inlet_lo, inlet_hi,
                 outlet_1_lo, outlet_1_hi,
                 outlet_2_lo, outlet_2_hi,
                 outlet_3_lo, outlet_3_hi, **kwargs):
        super().__init__(**kwargs)

        if split_1 <= 0.0 or split_2 <= 0.0:
            raise ValueError("split_1 and split_2 must both be positive.")
        if split_1 + split_2 >= 1.0:
            raise ValueError(
                f"split_1 + split_2 = {split_1 + split_2:.3f} must be < 1: "
                f"outlet 3 takes the remainder, and a non-positive remainder "
                f"would starve or reverse it.")

        self.j_in_from = int(inlet_lo * self.ny)
        self.j_in_to = int(inlet_hi * self.ny)
        self.j_o1_from = int(outlet_1_lo * self.ny)
        self.j_o1_to = int(outlet_1_hi * self.ny)
        self.j_o2_from = int(outlet_2_lo * self.ny)
        self.j_o2_to = int(outlet_2_hi * self.ny)
        self.i_o3_from = int(outlet_3_lo * self.nx)
        self.i_o3_to = int(outlet_3_hi * self.nx)
        self.i_east = self.nx - 1

        if not (self.j_o1_to <= self.j_o2_from or self.j_o2_to <= self.j_o1_from):
            raise ValueError(
                f"East outlets overlap: rows [{self.j_o1_from},{self.j_o1_to}) "
                f"and [{self.j_o2_from},{self.j_o2_to}). Ranges on the same "
                f"wall must be disjoint.")

        self.inlet_width = self.j_in_to - self.j_in_from
        self.split_1 = split_1
        self.split_2 = split_2
        self.split_3 = 1.0 - split_1 - split_2
        self.rho_south = 1.0                    # the pressure anchor

        self.Re = Re
        self.u_max = Re * self.nu / self.inlet_width
        self.periodic_x = True          # streaming must be a permutation

        self.u_profile = self._scaled_profile(self.inlet_width, 1.0,
                                              self.u_max, sign=+1.0)
        self.flux_in = float(np.sum(self.u_profile))

        # Outlet profiles: shape chosen first, then rescaled by the DISCRETE
        # sum so each flux is exactly split_k * Q_in. A parabola summed over
        # a handful of cells differs from its integral by several percent,
        # and that error would appear directly as a wrong split.
        self.u_profile_o1 = self._scaled_profile(
            self.j_o1_to - self.j_o1_from, split_1 * self.flux_in, sign=+1.0)
        self.u_profile_o2 = self._scaled_profile(
            self.j_o2_to - self.j_o2_from, split_2 * self.flux_in, sign=+1.0)

        self.q_o1_target = split_1 * self.flux_in
        self.q_o2_target = split_2 * self.flux_in
        self.q_o3_target = self.split_3 * self.flux_in

        self._check_stability_at_init()
        self._build_walls()

    # ------------------------------------------------------------------
    @staticmethod
    def _scaled_profile(width, target_flux, peak=None, sign=+1.0):
        """Parabolic profile over `width` cells.

        If `peak` is given the profile has that peak value (used for the
        inlet, where Re fixes u_max). Otherwise it is rescaled so its
        discrete sum equals `target_flux`.
        """
        if width < 3:
            raise ValueError(
                f"Opening spans {width} cells — too few for a profile. "
                f"Raise the resolution or widen the opening.")
        local = np.arange(width)
        shape = (-4.0 / width**2) * local * (local - width)      # peak 1
        if peak is not None:
            return sign * peak * shape
        return sign * shape * (target_flux / float(np.sum(shape)))

    def _check_stability_at_init(self):
        """Mach check on every prescribed profile.

        A narrow outlet carrying a large split can breach the low-Mach limit
        even when the inlet is comfortable, because the same flux is forced
        through fewer cells.
        """
        for name, profile in (("inlet", self.u_profile),
                              ("outlet 1", self.u_profile_o1),
                              ("outlet 2", self.u_profile_o2)):
            mach = float(np.max(np.abs(profile))) * np.sqrt(3.0)
            if mach > 0.1:
                raise ValueError(
                    f"Mach {mach:.3f} at the {name} exceeds 0.1. Lower Re, "
                    f"lower that split, widen the opening, or raise ny.")
        if self.tau_lbm <= 0.55:
            raise ValueError(f"tau={self.tau_lbm} too close to 0.5.")

    def _build_walls(self):
        obstacle = np.zeros((self.nx, self.ny), dtype=bool)
        obstacle[:, -1] = True                                    # north
        obstacle[0, :] = True                                     # west ...
        obstacle[0, self.j_in_from:self.j_in_to] = False          # ... inlet
        obstacle[-1, :] = True                                    # east ...
        obstacle[-1, self.j_o1_from:self.j_o1_to] = False         # ... outlet 1
        obstacle[-1, self.j_o2_from:self.j_o2_to] = False         # ... outlet 2
        obstacle[:, 0] = True                                     # south ...
        obstacle[self.i_o3_from:self.i_o3_to, 0] = False          # ... outlet 3
        self.obstacle = obstacle

    def apply_boundary_conditions(self):
        nb_zou_he_velocity_west(self.f, self.ux, self.uy, self.rho,
                                self.u_profile, self.j_in_from, self.j_in_to)
        nb_zou_he_velocity_east(self.f, self.ux, self.uy, self.rho,
                                self.u_profile_o1, self.j_o1_from, self.j_o1_to)
        nb_zou_he_velocity_east(self.f, self.ux, self.uy, self.rho,
                                self.u_profile_o2, self.j_o2_from, self.j_o2_to)
        nb_zou_he_pressure_south(self.f, self.ux, self.uy, self.rho,
                                 self.rho_south, self.i_o3_from, self.i_o3_to)

    # ------------------------------------------------------------------
    def outlet_fluxes(self):
        """Measured flux through each outlet, signed positive outward.

        Outlets 1 and 2 are prescribed and should reproduce their targets to
        solver tolerance. Outlet 3 is the anchor and should reproduce
        split_3 * Q_in by mass conservation.
        """
        q1 = float(np.sum(self.ux[self.i_east, self.j_o1_from:self.j_o1_to]))
        q2 = float(np.sum(self.ux[self.i_east, self.j_o2_from:self.j_o2_to]))
        q3 = float(np.sum(-self.uy[self.i_o3_from:self.i_o3_to, 0]))
        return q1, q2, q3

    def mass_balance(self):
        """(Q_in - sum Q_out) / Q_in. Should sit at O(Ma^2) ~ 1e-4.

        A growing residual means two boundary ranges overlap.
        """
        q1, q2, q3 = self.outlet_fluxes()
        return (self.flux_in - q1 - q2 - q3) / self.flux_in

    def inlet_density(self):
        """Mean rho at the inlet, derived by the velocity BC. Diagnostic
        only — flow direction is fixed by mass conservation, not by this."""
        return float(np.mean(self.rho[0, self.j_in_from:self.j_in_to]))

    def hydraulic_power(self):
        """Q_in * delta_p with p = rho/3. Comparable across alpha, unlike
        J = sum(alpha |u|^2), which grows with alpha regardless of design."""
        return self.flux_in * (self.inlet_density() - self.rho_south) / 3.0