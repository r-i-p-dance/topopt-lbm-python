from topopt.src.core.adjoint_base import AdjointLattice

class PressureAdjoint(AdjointLattice):
    """Discrete adjoint of the Zou-He pressure-driven channel.

    The adjoint BC is the exact transpose B^T of the forward Zou-He
    Jacobian. Two structural facts:

      1. Directions the forward RECONSTRUCTS get adjoint zero — their
         pre-BC value is overwritten, so nothing downstream depends on it.
         West reconstructs {1,5,8}; east reconstructs {3,6,7}.
      2. The remaining directions are NOT passed through: each receives
         coupling from the reconstructed adjoints, because the forward
         reconstruction READ their values.

    Coefficients derived by differentiating nb_zou_he_pressure_west/east
    w.r.t. each input population and transposing. Note the east sign
    convention differs from west (its ux expression adds the population
    sum rather than subtracting it), so east is NOT a naive mirror.

    "Zero the inward populations, leave the rest alone" is the CONTINUOUS
    adjoint BC, shown inconsistent for open flow systems by Luo, Chen,
    Yaji & Tao, Int. J. Heat Mass Transfer (2025).
    """

    def apply_boundary_conditions(self):
        f  = self.f
        W  = 0
        E  = self.fwd.nx - 1
        js = slice(1, self.fwd.ny - 1)      # match forward: skip corners

        # ---- West inlet: reconstructed {1, 5, 8} --------------------
        a1 = f[1, W, js].copy()
        a5 = f[5, W, js].copy()
        a8 = f[8, W, js].copy()

        f[0, W, js] += -2.0/3.0*a1 - 1.0/6.0*a5 - 1.0/6.0*a8
        f[2, W, js] += -2.0/3.0*a1 - 2.0/3.0*a5 + 1.0/3.0*a8
        f[3, W, js] += -1.0/3.0*a1 - 1.0/3.0*a5 - 1.0/3.0*a8
        f[4, W, js] += -2.0/3.0*a1 + 1.0/3.0*a5 - 2.0/3.0*a8
        f[6, W, js] += -4.0/3.0*a1 - 1.0/3.0*a5 + 2.0/3.0*a8
        f[7, W, js] += -4.0/3.0*a1 + 2.0/3.0*a5 - 1.0/3.0*a8

        f[1, W, js] = 0.0
        f[5, W, js] = 0.0
        f[8, W, js] = 0.0

        # ---- East outlet: reconstructed {3, 7, 6} -------------------
        b3 = f[3, E, js].copy()
        b7 = f[7, E, js].copy()
        b6 = f[6, E, js].copy()

        f[0, E, js] += -2.0/3.0*b3 - 1.0/6.0*b7 - 1.0/6.0*b6
        f[1, E, js] += -1.0/3.0*b3 - 1.0/3.0*b7 - 1.0/3.0*b6
        f[2, E, js] += -2.0/3.0*b3 + 1.0/3.0*b7 - 2.0/3.0*b6
        f[4, E, js] += -2.0/3.0*b3 - 2.0/3.0*b7 + 1.0/3.0*b6
        f[5, E, js] += -4.0/3.0*b3 + 2.0/3.0*b7 - 1.0/3.0*b6
        f[8, E, js] += -4.0/3.0*b3 - 1.0/3.0*b7 + 2.0/3.0*b6

        f[3, E, js] = 0.0
        f[7, E, js] = 0.0
        f[6, E, js] = 0.0
        