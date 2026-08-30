from topopt.src.core.adjoint_base import AdjointLattice


class TripleSplitAdjoint(AdjointLattice):
    """Exact transpose of the four forward boundary conditions.

    Rules, as everywhere in this project:
      1. Directions the forward RECONSTRUCTS get adjoint zero.
      2. Directions the forward READS pick up coupling from the
         reconstructed adjoints, via the transposed Jacobian.
      3. Index ranges must match the forward exactly.

    Reconstructed sets:
      west velocity inlet    {1, 5, 8}   K = ux / (1 - ux)
      east velocity outlets  {3, 6, 7}   K = ux / (1 + ux)
      south pressure anchor  {2, 5, 6}   rho prescribed

    Sign structure differs by boundary and cannot be mirrored. A VELOCITY BC
    derives rho from the populations, so the derivative of rho*u with
    respect to each input is positive; the west inlet reconstructs
    f1 = f3 + (2/3)KS and gains POSITIVE coupling, while the east outlet
    reconstructs f3 = f1 - (2/3)KS and gains NEGATIVE coupling. A PRESSURE
    BC prescribes rho, giving negative coefficients of yet another form.
    Each was derived from its own kernel.

    The two east outlets apply the same transpose on disjoint row ranges.
    They must not overlap — the forward checks this at construction.
    """

    def _velocity_east_transpose(self, u_profile, j_from, j_to):
        """B^T for one east velocity outlet, over rows [j_from, j_to)."""
        f = self.f
        east = self.fwd.i_east
        rows = slice(j_from, j_to)
        k = u_profile / (1.0 + u_profile)

        a3 = f[3, east, rows].copy()
        a7 = f[7, east, rows].copy()
        a6 = f[6, east, rows].copy()

        f[0, east, rows] += -(2/3)*k*a3 - (1/6)*k*a7 - (1/6)*k*a6
        f[1, east, rows] += (1.0 - (4/3)*k)*a3 - (1/3)*k*a7 - (1/3)*k*a6
        f[2, east, rows] += (-(2/3)*k*a3 + (0.5 - (1/6)*k)*a7
                             + (-0.5 - (1/6)*k)*a6)
        f[4, east, rows] += (-(2/3)*k*a3 + (-0.5 - (1/6)*k)*a7
                             + (0.5 - (1/6)*k)*a6)
        f[5, east, rows] += -(4/3)*k*a3 + (1.0 - (1/3)*k)*a7 - (1/3)*k*a6
        f[8, east, rows] += -(4/3)*k*a3 - (1/3)*k*a7 + (1.0 - (1/3)*k)*a6

        f[3, east, rows] = 0.0
        f[7, east, rows] = 0.0
        f[6, east, rows] = 0.0

    def apply_boundary_conditions(self):
        f = self.f
        fwd = self.fwd

        # ---- west velocity inlet: reconstructs {1, 5, 8} ---------------
        rows_in = slice(fwd.j_in_from, fwd.j_in_to)
        u_in = fwd.u_profile
        k_west = u_in / (1.0 - u_in)

        a1 = f[1, 0, rows_in].copy()
        a5 = f[5, 0, rows_in].copy()
        a8 = f[8, 0, rows_in].copy()

        f[0, 0, rows_in] += (2/3)*k_west*a1 + (1/6)*k_west*a5 + (1/6)*k_west*a8
        f[2, 0, rows_in] += ((2/3)*k_west*a1 + ((1/6)*k_west - 0.5)*a5
                             + ((1/6)*k_west + 0.5)*a8)
        f[3, 0, rows_in] += ((1.0 + (4/3)*k_west)*a1 + (1/3)*k_west*a5
                             + (1/3)*k_west*a8)
        f[4, 0, rows_in] += ((2/3)*k_west*a1 + ((1/6)*k_west + 0.5)*a5
                             + ((1/6)*k_west - 0.5)*a8)
        f[6, 0, rows_in] += ((4/3)*k_west*a1 + (1/3)*k_west*a5
                             + (1.0 + (1/3)*k_west)*a8)
        f[7, 0, rows_in] += ((4/3)*k_west*a1 + (1.0 + (1/3)*k_west)*a5
                             + (1/3)*k_west*a8)

        f[1, 0, rows_in] = 0.0
        f[5, 0, rows_in] = 0.0
        f[8, 0, rows_in] = 0.0

        # ---- east velocity outlets: reconstruct {3, 7, 6} --------------
        self._velocity_east_transpose(fwd.u_profile_o1,
                                      fwd.j_o1_from, fwd.j_o1_to)
        self._velocity_east_transpose(fwd.u_profile_o2,
                                      fwd.j_o2_from, fwd.j_o2_to)

        # ---- south pressure anchor: reconstructs {2, 5, 6} -------------
        cols = slice(fwd.i_o3_from, fwd.i_o3_to)
        b2 = f[2, cols, 0].copy()
        b5 = f[5, cols, 0].copy()
        b6 = f[6, cols, 0].copy()

        f[0, cols, 0] += -(2/3)*b2 - (1/6)*b5 - (1/6)*b6
        f[1, cols, 0] += -(2/3)*b2 - (2/3)*b5 + (1/3)*b6
        f[3, cols, 0] += -(2/3)*b2 + (1/3)*b5 - (2/3)*b6
        f[4, cols, 0] += -(1/3)*b2 - (1/3)*b5 - (1/3)*b6
        f[7, cols, 0] += -(4/3)*b2 + (2/3)*b5 - (1/3)*b6
        f[8, cols, 0] += -(4/3)*b2 - (1/3)*b5 + (2/3)*b6

        f[2, cols, 0] = 0.0
        f[5, cols, 0] = 0.0
        f[6, cols, 0] = 0.0