import numpy as np
from lbm.src.core.kernels import brinkman_collide_kernel
from topopt.src.core.brinkman import alpha_from_density
from topopt.src.core.projection import heaviside_projection
from lbm.src.core.lattice import BaseLattice

class BrinkmanMixin:
    """Adds design-dependent Brinkman collision to any BaseLattice-descended case.
    
    Not usable on its own — must be combined with a case class that provides
    the physics (walls, forcing, boundary conditions).
    
    Requires the combined class to inherit from BaseLattice.
    """
    def __init__(self, rho_e=None, alpha_max=100.0, q=0.1, beta=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha_max = alpha_max
        self.q = q
        self.beta = beta
        if rho_e is None:
            rho_e = np.ones((self.nx, self.ny))
        self.update_design(rho_e)

    def update_design(self, rho_e, alpha_max=None, beta=None):
        """Set a new design (and optionally new continuation parameters).
        Recomputes rho_bar and the effective relaxation field."""
        if alpha_max is not None:
            self.alpha_max = alpha_max
        if beta is not None:
            self.beta = beta
        self.rho_e = rho_e
        self.rho_bar = (heaviside_projection(rho_e, self.beta)
                        if self.beta is not None else rho_e)

        self.omega_eff = self._compute_omega()

    def _compute_omega(self):
        """
        Effective relaxation frequency incorporating Brinkman drag implicitly.
    
        omega_eff = 2 / (2*tau + alpha(rho_bar))
    
        Stability: unconditional. Alpha can be 0, 100, or 1000 — omega_eff
        stays in (0, 1/tau], which is always within the LBM stability window.
        No explicit forcing term needed or added.
        """
        
        alpha = alpha_from_density(self.rho_bar, self.alpha_max, self.q)
        return 2.0 / (2.0 * self.tau_lbm + alpha)

    def collision(self):
        brinkman_collide_kernel(self.f, self.f_eq, self.omega_eff, self.obstacle)



class BrinkmanLattice(BrinkmanMixin, BaseLattice):
    """Brinkman collision on a bare lattice — no geometry, no boundary
    conditions. Case classes inherit from this and supply both."""
    pass
