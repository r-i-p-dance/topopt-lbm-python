from topopt.src.core.adjoint_base import AdjointLattice


class PressureForkAdjoint(AdjointLattice):
    """Exact transpose of the three forward boundary conditions.

    Both rules from the pipe-bend adjoint apply unchanged: directions the
    forward RECONSTRUCTS get adjoint zero, and directions it READS pick up
    coupling from the reconstructed adjoints. Index ranges must match the
    forward exactly.

    West velocity inlet reconstructs {1,5,8}; east pressure outlet
    reconstructs {3,6,7}; south pressure outlet reconstructs {2,5,6}.
    """

    def apply_boundary_conditions(self):
        f, fwd = self.f, self.fwd
        js = slice(fwd.j_from, fwd.j_to)
        iss = slice(fwd.i_from, fwd.i_to)
        E = fwd.i_east

        # ---- west velocity inlet -------------------------------------
        u = fwd.u_profile
        K = u / (1.0 - u)
        a1 = f[1, 0, js].copy(); a5 = f[5, 0, js].copy(); a8 = f[8, 0, js].copy()

        f[0, 0, js] += (2/3)*K*a1 + (1/6)*K*a5 + (1/6)*K*a8
        f[2, 0, js] += (2/3)*K*a1 + ((1/6)*K - 0.5)*a5 + ((1/6)*K + 0.5)*a8
        f[3, 0, js] += (1.0 + (4/3)*K)*a1 + (1/3)*K*a5 + (1/3)*K*a8
        f[4, 0, js] += (2/3)*K*a1 + ((1/6)*K + 0.5)*a5 + ((1/6)*K - 0.5)*a8
        f[6, 0, js] += (4/3)*K*a1 + (1/3)*K*a5 + (1.0 + (1/3)*K)*a8
        f[7, 0, js] += (4/3)*K*a1 + (1.0 + (1/3)*K)*a5 + (1/3)*K*a8

        f[1, 0, js] = 0.0; f[5, 0, js] = 0.0; f[8, 0, js] = 0.0

        # ---- east pressure outlet ------------------------------------
        b3 = f[3, E, js].copy(); b7 = f[7, E, js].copy(); b6 = f[6, E, js].copy()

        f[0, E, js] += -(2/3)*b3 - (1/6)*b7 - (1/6)*b6
        f[1, E, js] += -(1/3)*b3 - (1/3)*b7 - (1/3)*b6
        f[2, E, js] += -(2/3)*b3 + (1/3)*b7 - (2/3)*b6
        f[4, E, js] += -(2/3)*b3 - (2/3)*b7 + (1/3)*b6
        f[5, E, js] += -(4/3)*b3 + (2/3)*b7 - (1/3)*b6
        f[8, E, js] += -(4/3)*b3 - (1/3)*b7 + (2/3)*b6

        f[3, E, js] = 0.0; f[7, E, js] = 0.0; f[6, E, js] = 0.0

        # ---- south pressure outlet -----------------------------------
        c2 = f[2, iss, 0].copy(); c5 = f[5, iss, 0].copy(); c6 = f[6, iss, 0].copy()

        f[0, iss, 0] += -(2/3)*c2 - (1/6)*c5 - (1/6)*c6
        f[1, iss, 0] += -(2/3)*c2 - (2/3)*c5 + (1/3)*c6
        f[3, iss, 0] += -(2/3)*c2 + (1/3)*c5 - (2/3)*c6
        f[4, iss, 0] += -(1/3)*c2 - (1/3)*c5 - (1/3)*c6
        f[7, iss, 0] += -(4/3)*c2 + (2/3)*c5 - (1/3)*c6
        f[8, iss, 0] += -(4/3)*c2 - (1/3)*c5 + (2/3)*c6

        f[2, iss, 0] = 0.0; f[5, iss, 0] = 0.0; f[6, iss, 0] = 0.0