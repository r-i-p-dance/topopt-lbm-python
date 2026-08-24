import numpy as np

def alpha_from_density(rho_bar, alpha_max, q):
    """Convex interpolation
    rho_e=1 → alpha=0 (fluid), rho_e=0 → alpha=alpha_max (solid)."""
    return alpha_max * q * (1.0 - rho_bar) / (q + rho_bar)

def d_alpha_d_rho_bar(rho_bar, alpha_max, q):
    """
    d(alpha)/d(rho_bar) = alpha_max * q * (q+1) / (q + 1 - rho_bar)^2
    Used in the sensitivity formula G.
    """
    return -alpha_max * q * (1.0 + q) / (q + rho_bar)**2
    
