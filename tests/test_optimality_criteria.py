import numpy as np
import pytest
from topopt.src.opt.optimality_criteria import (MultiplicativeOC, AdditiveOC,
                                                MonotoneNormalisation)

# Sensitivities in this problem are O(1e-5). Testing with O(1) synthetic data
# hides scale bugs: an earlier bisection bracket scaled to max|G| passed with
# G ~ N(0,1) and failed completely at the real magnitude.
G_SCALE = 1e-5


def make_optimizers():
    return [
        MultiplicativeOC(move=0.2, eta=0.5, rho_min=1e-3, projection_fn=None),
        AdditiveOC(move=0.1, learning_rate=0.5,
                   normaliser=MonotoneNormalisation(), projection_fn=None),
    ]


@pytest.fixture(params=[0, 1])
def opt(request):
    o = make_optimizers()[request.param]
    o.volume_fraction = 0.3
    return o


def _G(seed, nx=32, ny=32, mostly_negative=True):
    rng = np.random.RandomState(seed)
    g = rng.randn(nx, ny)
    if mostly_negative:
        g -= 1.0                       # matches the real sign distribution
    return G_SCALE * g


def test_hits_target_volume(opt):
    rho = 0.5 * np.ones((32, 32))
    new = opt.update(rho, _G(0))
    np.testing.assert_allclose(new.mean(), 0.3, atol=1e-3)


def test_hits_target_volume_at_tiny_sensitivity(opt):
    """Near convergence |G| collapses by orders of magnitude. The volume
    constraint must still be met — a bracket scaled to max|G| fails here."""
    rho = 0.5 * np.ones((32, 32))
    new = opt.update(rho, 1e-4 * _G(1))
    np.testing.assert_allclose(new.mean(), 0.3, atol=1e-3)


def test_respects_move_limit(opt):
    rho = 0.5 * np.ones((32, 32))
    new = opt.update(rho, _G(2))
    assert np.max(np.abs(new - rho)) <= opt.move + 1e-8


def test_respects_bounds(opt):
    rho = 0.5 * np.ones((32, 32))
    new = opt.update(rho, _G(3))
    assert new.min() >= 0.0 - 1e-12
    assert new.max() <= 1.0 + 1e-12


def test_fixed_cells_unchanged(opt):
    rho = 0.5 * np.ones((32, 32))
    mask = np.zeros((32, 32), dtype=bool)
    vals = np.ones((32, 32))
    mask[0, :] = True                   # pinned fluid inlet
    mask[-1, :] = True; vals[-1, :] = 0.0   # pinned solid wall
    new = opt.update(rho, _G(4), fixed_mask=mask, fixed_values=vals)
    np.testing.assert_allclose(new[0, :], 1.0)
    np.testing.assert_allclose(new[-1, :], 0.0)


def test_moves_toward_negative_sensitivity(opt):
    """G < 0 means more fluid reduces J, so those cells should move UP
    relative to cells where G is less negative."""
    rho = 0.5 * np.ones((32, 32))
    G = np.full((32, 32), -1.0 * G_SCALE)
    G[10:14, 10:14] = -10.0 * G_SCALE       # strongly favours fluid here
    new = opt.update(rho, G)
    assert new[10:14, 10:14].mean() > new[0:4, 0:4].mean()


def test_multiplicative_is_scale_invariant():
    """Rescaling G rescales lambda identically, leaving the ratio unchanged.
    This is why MultiplicativeOC needs no normalisation strategy."""
    o1 = MultiplicativeOC(move=0.2, projection_fn=None); o1.volume_fraction = 0.3
    o2 = MultiplicativeOC(move=0.2, projection_fn=None); o2.volume_fraction = 0.3
    rho = 0.5 * np.ones((32, 32))
    G = _G(5)
    np.testing.assert_allclose(o1.update(rho, G), o2.update(rho, 1000.0 * G),
                               atol=1e-10)