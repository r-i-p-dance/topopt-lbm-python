from topopt.src.core.adjoint_base import AdjointLattice


class ShelfAdjoint(AdjointLattice):
    """Exact transpose of the shelf forward boundary conditions.

    Rules, both structural:
      1. Directions the forward RECONSTRUCTS get adjoint zero (their pre-BC
         value is overwritten, so nothing downstream depends on it).
         West velocity inlet reconstructs {1,5,8}; south pressure outlet
         reconstructs {2,5,6}.
      2. Directions the forward READS pick up coupling from the
         reconstructed adjoints.

    Index ranges MUST match the forward exactly — applying B^T where the
    forward applied nothing breaks the transpose.

    Coefficient signs differ between the two boundaries: the velocity inlet
    derives rho from the populations (positive K terms), while the pressure
    outlet prescribes rho (negative terms). Derived per-kernel, not mirrored.
    """

    def apply_boundary_conditions(self):
        f = self.f
        fwd = self.fwd
        js = slice(fwd.j_from, fwd.j_to)
        iss = slice(fwd.i_from, fwd.i_to)

        # ---- west velocity inlet: reconstructed {1, 5, 8} ----------------
        u = fwd.u_profile                       # shape (inlet_width,)
        K = u / (1.0 - u)

        a1 = f[1, 0, js].copy()
        a5 = f[5, 0, js].copy()
        a8 = f[8, 0, js].copy()

        f[0, 0, js] += (2/3)*K*a1 + (1/6)*K*a5 + (1/6)*K*a8
        f[2, 0, js] += (2/3)*K*a1 + ((1/6)*K - 0.5)*a5 + ((1/6)*K + 0.5)*a8
        f[3, 0, js] += (1.0 + (4/3)*K)*a1 + (1/3)*K*a5 + (1/3)*K*a8
        f[4, 0, js] += (2/3)*K*a1 + ((1/6)*K + 0.5)*a5 + ((1/6)*K - 0.5)*a8
        f[6, 0, js] += (4/3)*K*a1 + (1/3)*K*a5 + (1.0 + (1/3)*K)*a8
        f[7, 0, js] += (4/3)*K*a1 + (1.0 + (1/3)*K)*a5 + (1/3)*K*a8

        f[1, 0, js] = 0.0
        f[5, 0, js] = 0.0
        f[8, 0, js] = 0.0

        # ---- south pressure outlet: reconstructed {2, 5, 6} --------------
        b2 = f[2, iss, 0].copy()
        b5 = f[5, iss, 0].copy()
        b6 = f[6, iss, 0].copy()

        f[0, iss, 0] += -(2/3)*b2 - (1/6)*b5 - (1/6)*b6
        f[1, iss, 0] += -(2/3)*b2 - (2/3)*b5 + (1/3)*b6
        f[3, iss, 0] += -(2/3)*b2 + (1/3)*b5 - (2/3)*b6
        f[4, iss, 0] += -(1/3)*b2 - (1/3)*b5 - (1/3)*b6
        f[7, iss, 0] += -(4/3)*b2 + (2/3)*b5 - (1/3)*b6
        f[8, iss, 0] += -(4/3)*b2 - (1/3)*b5 + (2/3)*b6

        f[2, iss, 0] = 0.0
        f[5, iss, 0] = 0.0
        f[6, iss, 0] = 0.0