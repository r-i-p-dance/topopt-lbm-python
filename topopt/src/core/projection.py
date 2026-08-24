import numpy as np

def heaviside_projection(rho_e, beta, eta=0.5):
    """
    Smooth Heaviside projection. Maps rho_e in [0,1] to physical
    density rho_bar in [0,1], concentrated near 0 and 1.
    beta=1: nearly linear. beta=32: near hard threshold at eta.
    """
    return (np.tanh(beta * eta) + np.tanh(beta * (rho_e - eta))) \
         / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))

def d_heaviside_d_rho_e(rho_e, beta, eta=0.5):
    """
    d(rho_bar)/d(rho_e) — needed to correct sensitivity via chain rule.
    """
    return beta * (1 - np.tanh(beta * (rho_e - eta))**2) \
         / (np.tanh(beta * eta) + np.tanh(beta * (1 - eta)))



