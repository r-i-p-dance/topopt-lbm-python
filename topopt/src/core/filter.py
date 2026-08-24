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

    def _build_kernel(self, R):
        """Cone weights: w = max(0, R - dist)."""
        k = int(np.ceil(R))
        y, x = np.ogrid[-k:k+1, -k:k+1]
        dist = np.sqrt(x**2 + y**2)
        return np.maximum(0.0, R - dist)

    def apply(self, G):
        """Return the filtered sensitivity."""
        weighted = convolve(G, self.kernel, mode='reflect')
        # per-cell normalization (sum of weights, accounting for boundary reflection)
        norm = convolve(np.ones_like(G), self.kernel, mode='reflect')
        return weighted / np.maximum(norm, 1e-12)