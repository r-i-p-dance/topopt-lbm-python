import numpy as np


class OptimalityCriteria:
    """Additive OC update with bisection on the volume-constraint multiplier.

    Structure follows the modern convention (rho=1 fluid, 0 solid).
    Volume is measured on the PROJECTED density rho_bar, not the raw design.
    """
    def __init__(self, target_volume, move=0.2, learning_rate=1.0,
                 projection_fn=None):
        self.target_vol = target_volume
        self.move = move
        self.lr = learning_rate
        self.projection_fn = projection_fn          # heaviside_projection(rho, beta) or None

    def update(self, design_var, G, beta=None, fixed_mask=None, fixed_values=None):
        """Return updated design satisfying the volume constraint.

        design_var: current raw design (nx, ny)
        G: filtered sensitivity (nx, ny)
        beta: current Heaviside sharpness (for volume measured on projected density)
        fixed_mask/fixed_values: cells pinned to fixed values (inlets, walls)
        """
        g_scale = float(np.max(np.abs(G))) + 1e-6
        l1, l2 = -g_scale * 10.0, g_scale * 10.0

        while (l2 - l1) > 1e-8 * g_scale:
            lmid = 0.5 * (l1 + l2)

            new_dv = np.maximum(
                np.maximum(0.0, design_var - self.move),
                np.minimum(np.minimum(1.0, design_var + self.move),
                           design_var - self.lr * (G + lmid))
            )

            if fixed_mask is not None:
                new_dv = np.where(fixed_mask, fixed_values, new_dv)

            # Volume on projected density (modern convention: rho_bar = fluid-ness)
            if self.projection_fn is not None and beta is not None:
                rho_bar = self.projection_fn(new_dv, beta)
            else:
                rho_bar = new_dv
            cur_volume = np.mean(rho_bar)

            # Modern convention: rho_bar is fluid fraction; target_vol is fluid target.
            # More fluid than target → need MORE penalty (raise lower bound).
            if cur_volume > self.target_vol:
                l1 = lmid
            else:
                l2 = lmid

        return new_dv