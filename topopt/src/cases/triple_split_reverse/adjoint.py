from topopt.src.cases.triple_split.adjoint import TripleSplitAdjoint


class TripleSplitReverseAdjoint(TripleSplitAdjoint):
    """Exact transpose of the four forward boundary conditions.

    Inherits _velocity_east_transpose from TripleSplitAdjoint, which is the
    same operation on the same wall.

    Reconstructed sets, and what zeroes:
      west velocity inlet   {1, 5, 8}   K = ux / (1 - ux),  ux > 0
      east velocity outlet  {3, 6, 7}   K = ux / (1 + ux),  ux > 0
      south velocity outlet {2, 5, 6}   K = uy / (1 - uy),  uy < 0
      west pressure anchor  {1, 5, 8}   rho prescribed

    Note the west wall carries two DIFFERENT transposes on disjoint ranges:
    the velocity inlet's coefficients are positive (rho is derived from the
    populations, so d(rho*u)/df > 0), while the pressure anchor's are
    negative (rho is prescribed). Same wall, same reconstructed set,
    opposite signs — a direct illustration of why boundary transposes must
    be derived per kernel and never mirrored.

    Index ranges must match the forward exactly: applying B^T where the
    forward applied nothing transposes an operator that does not exist.
    """

    def apply_boundary_conditions(self):
        f = self.f
        fwd = self.fwd

        # ---- west velocity inlet: reconstructs {1, 5, 8} ---------------
        rows_in = slice(fwd.j_in_from, fwd.j_in_to)
        u_in = fwd.u_profile
        k_in = u_in / (1.0 - u_in)

        a1 = f[1, 0, rows_in].copy()
        a5 = f[5, 0, rows_in].copy()
        a8 = f[8, 0, rows_in].copy()

        f[0, 0, rows_in] += (2/3)*k_in*a1 + (1/6)*k_in*a5 + (1/6)*k_in*a8
        f[2, 0, rows_in] += ((2/3)*k_in*a1 + ((1/6)*k_in - 0.5)*a5
                             + ((1/6)*k_in + 0.5)*a8)
        f[3, 0, rows_in] += ((1.0 + (4/3)*k_in)*a1 + (1/3)*k_in*a5
                             + (1/3)*k_in*a8)
        f[4, 0, rows_in] += ((2/3)*k_in*a1 + ((1/6)*k_in + 0.5)*a5
                             + ((1/6)*k_in - 0.5)*a8)
        f[6, 0, rows_in] += ((4/3)*k_in*a1 + (1/3)*k_in*a5
                             + (1.0 + (1/3)*k_in)*a8)
        f[7, 0, rows_in] += ((4/3)*k_in*a1 + (1.0 + (1/3)*k_in)*a5
                             + (1/3)*k_in*a8)

        f[1, 0, rows_in] = 0.0
        f[5, 0, rows_in] = 0.0
        f[8, 0, rows_in] = 0.0

        # ---- east velocity outlet: reconstructs {3, 7, 6} --------------
        self._velocity_east_transpose(fwd.u_profile_o1,
                                      fwd.j_o1_from, fwd.j_o1_to)

        # ---- south velocity outlet: reconstructs {2, 5, 6} -------------
        cols = slice(fwd.i_o2_from, fwd.i_o2_to)
        u_south = fwd.u_profile_o2
        k_south = u_south / (1.0 - u_south)

        b2 = f[2, cols, 0].copy()
        b5 = f[5, cols, 0].copy()
        b6 = f[6, cols, 0].copy()

        f[0, cols, 0] += (2/3)*k_south*b2 + (1/6)*k_south*b5 + (1/6)*k_south*b6
        f[1, cols, 0] += ((2/3)*k_south*b2 + ((1/6)*k_south - 0.5)*b5
                          + ((1/6)*k_south + 0.5)*b6)
        f[3, cols, 0] += ((2/3)*k_south*b2 + ((1/6)*k_south + 0.5)*b5
                          + ((1/6)*k_south - 0.5)*b6)
        f[4, cols, 0] += ((1.0 + (4/3)*k_south)*b2 + (1/3)*k_south*b5
                          + (1/3)*k_south*b6)
        f[7, cols, 0] += ((4/3)*k_south*b2 + (1.0 + (1/3)*k_south)*b5
                          + (1/3)*k_south*b6)
        f[8, cols, 0] += ((4/3)*k_south*b2 + (1/3)*k_south*b5
                          + (1.0 + (1/3)*k_south)*b6)

        f[2, cols, 0] = 0.0
        f[5, cols, 0] = 0.0
        f[6, cols, 0] = 0.0

        # ---- west pressure anchor: reconstructs {1, 5, 8} --------------
        # rho is PRESCRIBED here, so d(rho*ux)/df is negative — hence the
        # minus signs, opposite in sign to the velocity inlet a few rows up
        # on the same wall.
        rows_o3 = slice(fwd.j_o3_from, fwd.j_o3_to)
        c1 = f[1, 0, rows_o3].copy()
        c5 = f[5, 0, rows_o3].copy()
        c8 = f[8, 0, rows_o3].copy()

        f[0, 0, rows_o3] += -(2/3)*c1 - (1/6)*c5 - (1/6)*c8
        f[2, 0, rows_o3] += -(2/3)*c1 - (2/3)*c5 + (1/3)*c8
        f[3, 0, rows_o3] += -(1/3)*c1 - (1/3)*c5 - (1/3)*c8
        f[4, 0, rows_o3] += -(2/3)*c1 + (1/3)*c5 - (2/3)*c8
        f[6, 0, rows_o3] += -(4/3)*c1 - (1/3)*c5 + (2/3)*c8
        f[7, 0, rows_o3] += -(4/3)*c1 + (2/3)*c5 - (1/3)*c8

        f[1, 0, rows_o3] = 0.0
        f[5, 0, rows_o3] = 0.0
        f[8, 0, rows_o3] = 0.0