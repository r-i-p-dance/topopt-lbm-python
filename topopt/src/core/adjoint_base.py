import numpy as np
from lbm.src.core.lattice import BaseLattice
from topopt.src.core.objective import dJ_df
from lbm.src.core.kernels import adjoint_collide_kernel, stream_kernel
    

class AdjointLattice(BaseLattice):
    """Solves the discrete adjoint of BrinkmanForced (or any forward lattice).

    Holds a reference to the forward solver — reads its ux, uy, omega_eff,
    f_eq — and computes the adjoint distribution f_hat that satisfies
    the adjoint equation.
    """

    def __init__(self, forward, **kwargs):
        super().__init__(nx=forward.nx, ny=forward.ny,
                     tau_lbm=forward.tau_lbm, **kwargs)
        self.fwd = forward
        self.obstacle = forward.obstacle 
        self.periodic_x = forward.periodic_x
        self.opposite = forward.opposite
        self.f = np.zeros_like(forward.f)
        self.f_new = np.zeros_like(forward.f)
        self.g = np.zeros_like(forward.f)
        self.it = 0
        self.source = dJ_df(
            forward.rho_bar, forward.ux, forward.uy, forward.rho,
            forward.cx, forward.cy,
            forward.alpha_max, forward.q
        )
        self.neg_cx = -forward.cx
        self.neg_cy = -forward.cy

    def macro(self):
        """Adjoint first moments, WITHOUT dividing by sum(f).

        The forward divides momentum by rho = sum(f), which is a physical
        density near 1. The adjoint's sum(f_hat) is not a density: it is an
        arbitrary moment that can pass through zero, so dividing by it
        produces spurious spikes of 1e4-1e5 at cells where the positive and
        negative components happen to cancel — even though f_hat itself stays
        bounded within +/-0.15. Those spikes are a visualisation artifact,
        not a feature of the adjoint field.

        These moments are diagnostic only; the sensitivity contracts self.g
        against f_neq and never reads them.
        """
        self.rho[:] = np.sum(self.f, axis=0)
        self.ux[:] = np.einsum("kij,k->ij", self.f, self.cx.astype(np.float64))
        self.uy[:] = np.einsum("kij,k->ij", self.f, self.cy.astype(np.float64))
        self.ux[self.obstacle] = 0.0
        self.uy[self.obstacle] = 0.0

    def collision(self):
        self.g = self.f.copy()        # state before C^T, for the sensitivity
        adjoint_collide_kernel(self.f, self.fwd.ux, self.fwd.uy,
                               self.fwd.omega_eff, self.source,
                               self.w, self.cx, self.cy, self.obstacle)

    def stream(self):
        stream_kernel(self.f, self.f_new, self.neg_cx, self.neg_cy, self.fwd.periodic_x)
        self.f, self.f_new = self.f_new, self.f

    def apply_boundary_conditions(self):
        pass

    def step(self):
        self.apply_boundary_conditions()
        self.stream()                       
        self.bounce_back_obstacle()
        self.collision()           

    def update_source(self):
        """Recompute the adjoint source from the current forward solution.
        Call after the forward has been re-converged for a new design."""
        self.source = dJ_df(
            self.fwd.rho_bar, self.fwd.ux, self.fwd.uy, self.fwd.rho,
            self.fwd.cx, self.fwd.cy, self.fwd.alpha_max, self.fwd.q)        

    def check_stability_running(self):
        """No-op. The adjoint field is a sensitivity, not a fluid velocity;
        its moments have no Mach interpretation. Stability of the adjoint is
        governed by rho(M^T) = rho(M), which the forward's own check covers."""
        pass