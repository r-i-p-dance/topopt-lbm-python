import numpy as np
import pytest
from topopt.src.opt.continuation import (GeometricContinuation,
                                         StaircaseContinuation)

SCHEDULES = [
    lambda: GeometricContinuation(alpha_start=5.0, alpha_end=100.0,
                                  beta_start=1.0, beta_end=8.0,
                                  beta_delay=20, complete_by=120),
    lambda: StaircaseContinuation(alpha_start=5.0, alpha_end=100.0,
                                  beta_start=1.0, beta_end=8.0,
                                  beta_delay=20, hold=20, n_steps=6),
]


@pytest.mark.parametrize("make", SCHEDULES)
def test_monotonic(make):
    """Neither parameter may ever decrease — the design would un-penalise."""
    s = make()
    a = [s.alpha_max(i) for i in range(300)]
    b = [s.beta(i) for i in range(300)]
    assert np.all(np.diff(a) >= -1e-12)
    assert np.all(np.diff(b) >= -1e-12)


@pytest.mark.parametrize("make", SCHEDULES)
def test_capped_at_end(make):
    s = make()
    assert s.alpha_max(10_000) == pytest.approx(s.alpha_end)
    assert s.beta(10_000) == pytest.approx(s.beta_end)


@pytest.mark.parametrize("make", SCHEDULES)
def test_starts_at_start(make):
    s = make()
    assert s.alpha_max(0) == pytest.approx(s.alpha_start)
    assert s.beta(0) == pytest.approx(s.beta_start)


@pytest.mark.parametrize("make", SCHEDULES)
def test_beta_held_during_delay(make):
    s = make()
    assert s.beta(0) == pytest.approx(s.beta_start)
    assert s.beta(s.beta_delay - 1) == pytest.approx(s.beta_start)
    assert s.beta(s.beta_delay + 40) > s.beta_start


@pytest.mark.parametrize("make", SCHEDULES)
def test_is_complete_matches_final_iteration(make):
    """The driver gates convergence on is_complete, so final_iteration must
    be the first iteration at which it becomes true."""
    s = make()
    n = s.final_iteration
    assert not s.is_complete(n - 1)
    assert s.is_complete(n)


def test_geometric_rate_derived_from_complete_by():
    """The rate is derived, not supplied — this is what makes complete_by
    a defensible parameter rather than a magic growth constant."""
    s = GeometricContinuation(alpha_start=5.0, alpha_end=100.0,
                              beta_start=1.0, beta_end=8.0,
                              beta_delay=20, complete_by=120)
    assert s.alpha_max(120) == pytest.approx(100.0)
    assert s.alpha_max(119) < 100.0