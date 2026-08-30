import numpy as np


class BaseContinuation:
    """Interface for continuation schedules.

    Continuation exists because the problem is non-convex. Starting at high
    alpha and beta, the optimizer freezes into whatever local minimum it
    lands in first. Starting soft gives a nearly convex problem with a smooth
    grey solution, which tightening then walks toward a binary one.
    """

    def __init__(self, alpha_start=5.0, alpha_end=10.0,
                 beta_start=1.0, beta_end=2.0, beta_delay=20):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_delay = beta_delay

    def alpha_max(self, iteration):
        raise NotImplementedError

    def beta(self, iteration):
        raise NotImplementedError

    def notify(self, iteration, change):
        """Hook for schedules that react to the design; no-op otherwise."""

    def is_complete(self, iteration):
        return (self.alpha_max(iteration) >= self.alpha_end - 1e-12 and
                self.beta(iteration) >= self.beta_end - 1e-12)

    def describe(self):
        return (f"{type(self).__name__}: alpha {self.alpha_start}->"
                f"{self.alpha_end}, beta {self.beta_start}->{self.beta_end}")


class GeometricContinuation(BaseContinuation):
    """Smooth geometric ramp, parameterised by WHEN it should finish.

    Growth rates are derived from complete_by rather than supplied, so the
    tunable is a defensible design decision ("continuation finishes by
    iteration 120") rather than an unjustifiable rate ("1.017 per iteration").

    Preferred over the staircase variant for two reasons: J decreases
    continuously rather than plateauing within a block, so a windowed
    convergence test cannot fire spuriously; and there is no arbitrary
    block length to justify.
    """

    def __init__(self, complete_alpha_by, complete_beta_by, **kw):
        super().__init__(**kw)
        self.complete_alpha_by = complete_alpha_by
        self.complete_beta_by = complete_beta_by
        self.alpha_growth = (self.alpha_end / self.alpha_start) ** (1.0 / complete_alpha_by)
        beta_span = max(complete_beta_by - self.beta_delay, 1)
        self.beta_growth = (self.beta_end / self.beta_start) ** (1.0 / beta_span)

    def alpha_max(self, iteration):
        return min(self.alpha_start * self.alpha_growth ** iteration,
                   self.alpha_end)

    def beta(self, iteration):
        if iteration < self.beta_delay:
            return self.beta_start
        k = iteration - self.beta_delay
        return min(self.beta_start * self.beta_growth ** k, self.beta_end)

    @property
    def final_iteration(self):
        return max(self.complete_alpha_by, self.complete_beta_by)

    def describe(self):
        return (super().describe() +
                f", geometric, complete by it {self.final_iteration} "
                f"(rates {self.alpha_growth:.4f}/{self.beta_growth:.4f})")


class StaircaseContinuation(BaseContinuation):
    """Hold parameters fixed for `hold` iterations, then step up.

    Kept for comparison. Each block lets the design re-converge before the
    target moves, but J plateaus within a block, which defeats windowed
    convergence detection, and the block length is arbitrary.
    """

    def __init__(self, hold=20, n_steps=6, **kw):
        super().__init__(**kw)
        self.hold = hold
        self.n_steps = n_steps
        self.alpha_factor = (self.alpha_end / self.alpha_start) ** (1.0 / n_steps)
        self.beta_factor = (self.beta_end / self.beta_start) ** (1.0 / n_steps)

    def alpha_max(self, iteration):
        k = min(iteration // self.hold, self.n_steps)
        return min(self.alpha_start * self.alpha_factor ** k, self.alpha_end)

    def beta(self, iteration):
        if iteration < self.beta_delay:
            return self.beta_start
        k = min((iteration - self.beta_delay) // self.hold, self.n_steps)
        return min(self.beta_start * self.beta_factor ** k, self.beta_end)

    @property
    def final_iteration(self):
        return self.beta_delay + self.n_steps * self.hold

    def describe(self):
        return (super().describe() +
                f", staircase {self.n_steps} x {self.hold} iters, "
                f"complete at it {self.final_iteration}")