import numpy as np
from topopt.src.core.brinkman import alpha_from_density
from topopt.src.core.projection import heaviside_projection

class BrinkmanMixin:
    """Adds design-dependent Brinkman collision to any BaseLattice-descended case.
    
    Not usable on its own — must be combined with a case class that provides
    the physics (walls, forcing, boundary conditions).
    
    Requires the combined class to inherit from BaseLattice.
    """
    def __init__(self, rho_e=None, alpha_max=100.0, q=0.1, beta=None, **kwargs):
        super().__init__(**kwargs)                    # calls the other parent's __init__
        self.alpha_max = alpha_max
        self.q = q
        self.beta = beta  
        self.rho_e = rho_e if rho_e is not None else np.ones((self.nx, self.ny))
        self.rho_bar = self.rho_e if beta is None else heaviside_projection(self.rho_e, beta)
        self.omega_eff = self._compute_omega()

    def update_design(self, rho_e):
        self.rho_e = rho_e                            # store the RAW design
        if self.beta is not None:
            self.rho_bar = heaviside_projection(rho_e, self.beta)
        else:
            self.rho_bar = rho_e                      # no projection: bar == raw
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
        fluid = ~self.obstacle
        # apply collision only at fluid cells
        self.f[:, fluid] = (self.f[:, fluid]
                        - self.omega_eff[fluid][None, :]
                        * (self.f[:, fluid] - self.f_eq[:, fluid]))


