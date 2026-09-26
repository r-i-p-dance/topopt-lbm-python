import numpy as np
from scipy.ndimage import convolve


class SensitivityFilter:
    """Weighted-neighbor smoothing of the sensitivity field (Sigmund's method).

    Applied to the raw sensitivity G after it's computed from forward+adjoint,
    before it feeds the optimizer. Prevents checkerboarding and mesh dependence.
    """
    def __init__(self, radius):
        self.radius = radius
        self.kernel = self._build_kernel(radius)
        self.kernel_sum = None          # per-cell normalization, set on first apply

    # def _build_kernel(self, R):
    #     """Cone weights: w = max(0, R - dist)."""
    #     k = int(np.ceil(R))
    #     y, x = np.ogrid[-k:k+1, -k:k+1]
    #     dist = np.sqrt(x**2 + y**2)
    #     return np.maximum(0.0, R - dist)

    def _build_kernel(self, radius):
        """Gaussian weights, sigma = radius/2.

        The conventional cone kernel max(0, R - dist) quantizes coarsely on
        a Cartesian grid at small radius: at R = 2 the 5x5 kernel has only a
        few distinct weight levels, so diagonal features resolve into
        one-cell staircases. Gaussian weights vary smoothly with distance
        and do not produce that artifact, at the cost of a slightly less
        crisp length-scale cutoff.
        """
        half = int(np.ceil(radius))
        y, x = np.ogrid[-half:half+1, -half:half+1]
        sigma = radius / 2.0
        return np.exp(-(x**2 + y**2) / (2.0 * sigma**2))

    def apply(self, G):
        """Return the filtered sensitivity."""
        weighted = convolve(G, self.kernel, mode='reflect')
        # per-cell normalization (sum of weights, accounting for boundary reflection)
        norm = convolve(np.ones_like(G), self.kernel, mode='reflect')
        return weighted / np.maximum(norm, 1e-12)