import numpy as np

class BaseCase:
    """Defines the optimization problem: geometry, fixed cells, volume target.

    Owns nx, ny, volume_fraction — the single source of truth for those.
    Holds one forward and one adjoint instance, reused (warm-started) across
    optimization iterations.
    """

    def __init__(self, nx, ny, volume_fraction, Re, tau_lbm, q=0.1):
        self.nx = nx
        self.ny = ny
        self.volume_fraction = volume_fraction
        self.Re = Re
        self.tau_lbm = tau_lbm
        self.q = q
        self.nu = (tau_lbm - 0.5) / 3.0

        self.fixed_mask = np.zeros((nx, ny), dtype=bool)
        self.fixed_values = np.ones((nx, ny))
        self.rho_e = volume_fraction * np.ones((nx, ny))

        self._setup_geometry()
        self.rho_e[self.fixed_mask] = self.fixed_values[self.fixed_mask]

        self.forward = None
        self.adjoint = None

    def _setup_geometry(self):
        """Subclass sets fixed_mask / fixed_values and any case constants."""
        raise NotImplementedError

    def build_solvers(self, alpha_max, beta):
        """Create the forward and adjoint once; reused for all iterations."""
        raise NotImplementedError