from topopt.src.core.adjoint_base import AdjointLattice


class SplitAdjoint(AdjointLattice):
    """Exact transpose of the three forward boundary conditions.

    Two structural rules, as everywhere else in this project:

      1. Directions the forward RECONSTRUCTS get adjoint zero — their
         pre-BC value is overwritten, so nothing downstream depends on it.
      2. Directions the forward READS pick up coupling from the
         reconstructed adjoints, with coefficients from the transposed
         Jacobian.

    Index ranges must match the forward exactly: applying B^T where the
    forward applied nothing transposes an operator that does not exist.

    Reconstructed sets:
      west velocity inlet   {1, 5, 8}
      south velocity outlet {2, 5, 6}
      east pressure outlet  {3, 6, 7}

    Coefficient signs differ between the velocity and pressure boundaries.
    A velocity BC DERIVES rho from the populations, so the derivative of
    rho*u with respect to each input is positive and the coupling terms
    carry +K. A pressure BC PRESCRIBES rho, so the same derivative is
    negative. Never mirror one boundary's coefficients onto another —
    derive each from its own kernel.

    Verify with tests/diagnose_transpose.py after any change here; every
    operator and composition must report ~1e-11.
    """

    def apply_boundary_conditions(self):
        f = self.f
        fwd = self.fwd
        js = slice(fwd.j_from, fwd.j_to)
        iss = slice(fwd.i_from, fwd.i_to)
        east = fwd.i_east

        # ---- west velocity inlet: reconstructs {1, 5, 8} ----------------
        # K = ux / (1 - ux), from rho = (known) / (1 - ux)
        u_in = fwd.u_profile
        k_west = u_in / (1.0 - u_in)

        a1 = f[1, 0, js].copy()
        a5 = f[5, 0, js].copy()
        a8 = f[8, 0, js].copy()

        f[0, 0, js] += (2/3)*k_west*a1 + (1/6)*k_west*a5 + (1/6)*k_west*a8
        f[2, 0, js] += ((2/3)*k_west*a1 + ((1/6)*k_west - 0.5)*a5
                        + ((1/6)*k_west + 0.5)*a8)
        f[3, 0, js] += ((1.0 + (4/3)*k_west)*a1 + (1/3)*k_west*a5
                        + (1/3)*k_west*a8)
        f[4, 0, js] += ((2/3)*k_west*a1 + ((1/6)*k_west + 0.5)*a5
                        + ((1/6)*k_west - 0.5)*a8)
        f[6, 0, js] += ((4/3)*k_west*a1 + (1/3)*k_west*a5
                        + (1.0 + (1/3)*k_west)*a8)
        f[7, 0, js] += ((4/3)*k_west*a1 + (1.0 + (1/3)*k_west)*a5
                        + (1/3)*k_west*a8)

        f[1, 0, js] = 0.0
        f[5, 0, js] = 0.0
        f[8, 0, js] = 0.0

        # ---- south velocity outlet: reconstructs {2, 5, 6} --------------
        # K = uy / (1 - uy). uy is NEGATIVE for outflow, so K < 0 — the
        # coupling terms change sign relative to the inlet, which is why
        # this cannot be a mirrored copy of the west block.
        u_south = fwd.u_profile_south
        k_south = u_south / (1.0 - u_south)

        b2 = f[2, iss, 0].copy()
        b5 = f[5, iss, 0].copy()
        b6 = f[6, iss, 0].copy()

        f[0, iss, 0] += (2/3)*k_south*b2 + (1/6)*k_south*b5 + (1/6)*k_south*b6
        f[1, iss, 0] += ((2/3)*k_south*b2 + ((1/6)*k_south - 0.5)*b5
                         + ((1/6)*k_south + 0.5)*b6)
        f[3, iss, 0] += ((2/3)*k_south*b2 + ((1/6)*k_south + 0.5)*b5
                         + ((1/6)*k_south - 0.5)*b6)
        f[4, iss, 0] += ((1.0 + (4/3)*k_south)*b2 + (1/3)*k_south*b5
                         + (1/3)*k_south*b6)
        f[7, iss, 0] += ((4/3)*k_south*b2 + (1.0 + (1/3)*k_south)*b5
                         + (1/3)*k_south*b6)
        f[8, iss, 0] += ((4/3)*k_south*b2 + (1/3)*k_south*b5
                         + (1.0 + (1/3)*k_south)*b6)

        f[2, iss, 0] = 0.0
        f[5, iss, 0] = 0.0
        f[6, iss, 0] = 0.0

        # ---- east pressure outlet: reconstructs {3, 7, 6} ---------------
        # rho is PRESCRIBED here, so the derivative of rho*ux with respect
        # to each input is negative — hence the minus signs throughout.
        c3 = f[3, east, js].copy()
        c7 = f[7, east, js].copy()
        c6 = f[6, east, js].copy()

        f[0, east, js] += -(2/3)*c3 - (1/6)*c7 - (1/6)*c6
        f[1, east, js] += -(1/3)*c3 - (1/3)*c7 - (1/3)*c6
        f[2, east, js] += -(2/3)*c3 + (1/3)*c7 - (2/3)*c6
        f[4, east, js] += -(2/3)*c3 - (2/3)*c7 + (1/3)*c6
        f[5, east, js] += -(4/3)*c3 + (2/3)*c7 - (1/3)*c6
        f[8, east, js] += -(4/3)*c3 - (1/3)*c7 + (2/3)*c6

        f[3, east, js] = 0.0
        f[7, east, js] = 0.0
        f[6, east, js] = 0.0