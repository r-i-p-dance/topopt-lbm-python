# tests/test_optimality_criteria.py
import numpy as np
from topopt.src.opt.optimality_criteria import OptimalityCriteria


def test_oc_hits_target_volume():
    """After update, mean design should equal target volume."""
    oc = OptimalityCriteria(target_volume=0.4, move=0.5, learning_rate=1.0)
    rho = 0.5 * np.ones((16, 16))
    G = np.random.RandomState(0).randn(16, 16) - 1.0
    new = oc.update(rho, G)
    np.testing.assert_allclose(new.mean(), 0.4, atol=1e-3)


def test_oc_respects_move_limit():
    oc = OptimalityCriteria(target_volume=0.4, move=0.2)
    rho = 0.5 * np.ones((16, 16))
    G = -np.abs(np.random.RandomState(1).randn(16, 16))
    new = oc.update(rho, G)
    assert np.max(np.abs(new - rho)) <= 0.2 + 1e-8


def test_oc_respects_bounds():
    oc = OptimalityCriteria(target_volume=0.4)
    rho = 0.5 * np.ones((16, 16))
    G = -np.abs(np.random.RandomState(2).randn(16, 16))
    new = oc.update(rho, G)
    assert new.min() >= 0.0 and new.max() <= 1.0


def test_oc_fixed_cells_unchanged():
    """Cells in fixed_mask keep their fixed_values."""
    oc = OptimalityCriteria(target_volume=0.4, move=0.5)
    rho = 0.5 * np.ones((16, 16))
    G = np.random.RandomState(3).randn(16, 16)
    mask = np.zeros((16, 16), dtype=bool)
    mask[0, :] = True                          # pin the inlet column
    vals = np.ones((16, 16))
    new = oc.update(rho, G, fixed_mask=mask, fixed_values=vals)
    np.testing.assert_allclose(new[0, :], 1.0)