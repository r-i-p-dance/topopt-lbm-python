import numpy as np
from topopt.src.core.filter import SensitivityFilter


def test_filter_preserves_constant():
    """Filtering a constant sensitivity returns the same constant."""
    filt = SensitivityFilter(radius=3.0)
    G = -0.5 * np.ones((32, 32))
    np.testing.assert_allclose(filt.apply(G), -0.5, atol=1e-12)


def test_filter_smooths_spike():
    """A single-cell spike spreads to neighbors (reduced peak, nonzero surround)."""
    filt = SensitivityFilter(radius=3.0)
    G = np.zeros((32, 32))
    G[16, 16] = -1.0
    Gf = filt.apply(G)
    assert Gf[16, 16] > -1.0                 # peak reduced (spread out)
    assert Gf[16, 15] < 0.0                  # neighbor now nonzero


def test_filter_preserves_sign():
    """Filtering a uniformly-negative field keeps it negative."""
    filt = SensitivityFilter(radius=2.0)
    G = -np.abs(np.random.RandomState(0).randn(16, 16))
    assert np.all(filt.apply(G) <= 0)