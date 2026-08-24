import numpy as np
from topopt.src.opt.continuation import ContinuationSchedule

def test_alpha_monotonic():
    sched = ContinuationSchedule(alpha_start=1.0, alpha_end=100.0, alpha_growth_rate=1.1)
    alphas = [sched.alpha(i) for i in range(100)]
    diffs = np.diff(alphas)
    assert np.all(diffs >= 0), "alpha should be non-decreasing"


def test_alpha_capped_at_end():
    sched = ContinuationSchedule(alpha_start=1.0, alpha_end=10.0, alpha_growth_rate=1.5)
    assert sched.alpha(1000) == 10.0


def test_beta_held_during_delay():
    sched = ContinuationSchedule(beta_start=1.0, beta_end=10.0, beta_growth_rate=1.1, beta_delay=50)
    assert sched.beta(0) == 1.0
    assert sched.beta(49) == 1.0
    assert sched.beta(50) == 1.0
    assert sched.beta(100) > 1.0