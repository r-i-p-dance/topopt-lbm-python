import numpy as np
from topopt.src.core.projection import heaviside_projection


class MonotoneNormalisation:
    """Divide G by the largest max|G| seen so far; the reference never decays.

    Needed only by AdditiveOC. The additive update is not scale-invariant:
    the Lagrange multiplier absorbs the MEAN of G (enforcing volume) but
    nothing absorbs its SCALE, so the raw step is proportional to |G|
    (~1e-5 here, against a move limit of 0.1). Normalising by the
    INSTANTANEOUS max fixes the step size but removes convergence — the step
    stays full-size however small the gradient becomes. A monotone reference
    keeps learning_rate meaningful early while letting the step shrink with
    the gradient later.
    """

    def __init__(self):
        self.reference = None

    def __call__(self, sensitivity):
        peak = float(np.max(np.abs(sensitivity)))
        if peak < 1e-30:
            return sensitivity
        self.reference = peak if self.reference is None else max(peak, self.reference)
        return sensitivity / self.reference

    def describe(self):
        return "monotone max"


class BaseOptimizer:
    """Interface: update(rho_e, sensitivity, iteration, beta, fixed_mask,
    fixed_values) -> rho_new.

    volume_fraction and contract_from are wired by the driver from the case
    and the continuation schedule respectively, so they cannot be supplied
    inconsistently. self.lam records the multiplier for plotting.
    """

    def __init__(self, move=0.2, projection_fn=heaviside_projection,
                 move_decay=0.95, move_floor=0.0):
        self.volume_fraction = None     # set by the driver from the case
        self.contract_from = None       # set by the driver from continuation
        self.move = move
        self.move_decay = move_decay
        self.move_floor = move_floor
        self.projection_fn = projection_fn
        self.lam = 0.0

    def current_move(self, iteration):
        """Move limit for this iteration, contracted after continuation ends.

        A near-binary design has almost no cells strictly inside their
        bounds, so the OC fixed point B/lambda = 1 — which holds only for
        interior cells — does not exist, and interface cells flip
        indefinitely. The amplitude of that residual flipping is
        proportional to the move limit, so contracting the trust region
        once the problem stops changing damps it.
        """
        if self.contract_from is None or iteration < self.contract_from:
            return self.move
        steps_since = iteration - self.contract_from
        return max(self.move * self.move_decay ** steps_since, self.move_floor)

    def projected_volume(self, design, beta):
        if self.projection_fn is not None and beta is not None:
            return float(np.mean(self.projection_fn(design, beta)))
        return float(np.mean(design))

    def update(self, rho_e, sensitivity, iteration, beta=None,
               fixed_mask=None, fixed_values=None):
        raise NotImplementedError

    def describe(self):
        return type(self).__name__


class MultiplicativeOC(BaseOptimizer):
    """Classic Bendsoe-Sigmund optimality criteria update.

        rho_new = clip( rho * (-G / lambda)^eta )

    clipped to [rho - move, rho + move] and [rho_min, 1], with lambda found
    by bisection so the projected volume meets the target.

    Why this form rather than the additive one:

    * Scale-invariant. Rescaling G rescales lambda identically, leaving the
      ratio unchanged. No normalisation strategy, no learning rate.

    * Converges toward the KKT point for interior cells. The condition is
      -G_e/lambda = 1, so at the optimum the multiplier is 1 and the update
      is the identity. Note this holds only for cells strictly INSIDE their
      bounds; a near-binary design has almost none, which is why
      current_move() contracts the trust region late in the run.

    * Damped at the projection interface. At high beta the Heaviside
      derivative amplifies G by ~beta on interface cells; the eta = 0.5
      exponent turns that into sqrt(beta).

    Cells with G > 0 (more fluid would increase dissipation) give a negative
    ratio; those are clamped to zero, sending the cell toward solid at the
    move limit. rho_min must be strictly positive: a cell at exactly zero
    can never recover under a multiplicative update.

    Bisection runs in log-space because lambda spans many decades and linear
    bisection over such a bracket wastes most of its iterations.
    """

    def __init__(self, move=0.2, eta=0.5, rho_min=0.05, convergence_window=10,
                 projection_fn=heaviside_projection, n_bisect=40,
                 move_decay=1.0, move_floor=0.0, bisect_tol=1e-10):
        super().__init__(move, projection_fn, move_decay, move_floor)
        self.eta = eta
        self.rho_min = rho_min
        self.n_bisect = n_bisect
        self.bisect_tol = bisect_tol
        self.diagnostics = {}

        """ move this somewhere else. it doesn't seem like it belongs here """
        self.convergence_window = convergence_window

    def update(self, rho_e, sensitivity, iteration, beta=None,
               fixed_mask=None, fixed_values=None):
        move = self.current_move(iteration)
        benefit = -sensitivity                  # > 0 where more fluid helps
        scale = float(np.max(np.abs(benefit)))
        if scale < 1e-30:
            return (np.where(fixed_mask, fixed_values, rho_e)
                    if fixed_mask is not None else rho_e)

        # bracket lambda over 12 decades around the sensitivity scale
        log_low = np.log10(scale) - 9.0
        log_high = np.log10(scale) + 3.0
        design_new, volume = rho_e, 0.0

        for _ in range(self.n_bisect):
            log_mid = 0.5 * (log_low + log_high)
            lam = 10.0 ** log_mid

            ratio = np.maximum(benefit / lam, 0.0) ** self.eta
            design_new = np.clip(rho_e * ratio, rho_e - move, rho_e + move)
            design_new = np.clip(design_new, self.rho_min, 1.0)
            if fixed_mask is not None:
                design_new = np.where(fixed_mask, fixed_values, design_new)

            volume = self.projected_volume(design_new, beta)
            # volume decreases as lambda increases
            if volume > self.volume_fraction:
                log_low = log_mid
            else:
                log_high = log_mid

        self.lam = 10.0 ** (0.5 * (log_low + log_high))

        # Diagnostics: which mechanism is moving the design.
        free = ~fixed_mask if fixed_mask is not None else np.ones_like(rho_e, bool)
        delta = design_new - rho_e
        self.diagnostics = {
            # Cells with G > 0 get ratio = 0 exactly, so they drop by the full
            # move limit INDEPENDENTLY of lambda. If this fraction is large
            # and changes between iterations, a whole block of the design is
            # slammed down one iteration and over-compensated the next.
            "frac_positive_G": float(np.mean(benefit[free] <= 0.0)),
            "frac_at_floor": float(np.mean(design_new[free] <= self.rho_min * 1.001)),
            "frac_clip_down": float(np.mean(delta[free] <= -move * 0.999)),
            "frac_clip_up": float(np.mean(delta[free] >= move * 0.999)),
        }


        if abs(volume - self.volume_fraction) > 1e-3:
            print(f"  [OC] volume {volume:.4f} vs target "
                  f"{self.volume_fraction:.4f} — not reachable within the "
                  f"move limit this iteration")
        return design_new

    def describe(self):
        return (f"MultiplicativeOC(move={self.move}, eta={self.eta}, "
                f"rho_min={self.rho_min}, move_decay={self.move_decay}, "
                f"move_floor={self.move_floor})")


class AdditiveOC(BaseOptimizer):
    """Additive OC: rho_new = rho - lr * (G_norm + lambda).

    Kept for comparison against MultiplicativeOC. Requires a normalisation
    strategy and a beta-scaled move limit to behave; see MultiplicativeOC's
    docstring for why neither is needed there.

    The bisection bracket must let lambda saturate the move limit in both
    directions. Since rho is in [0,1], |lr*lambda| >= 1 suffices; scaling
    the bracket to max|G| alone fails when the sensitivity is small, which
    is the normal case.
    """

    def __init__(self, move=0.1, learning_rate=0.5, convergence_window=10, normaliser=None,
                 projection_fn=heaviside_projection, n_bisect=200,
                 move_decay=0.95, move_floor=0.01):
        super().__init__(move, projection_fn, move_decay, move_floor)
        self.lr = learning_rate
        self.normaliser = normaliser or MonotoneNormalisation()
        self.n_bisect = n_bisect
        self.convergence_window = convergence_window

    def current_move(self, iteration, beta=None):
        """Base contraction, then a further 1/sqrt(beta) factor.

        The Heaviside derivative is a spike of height ~beta and width
        ~1/beta; the geometric mean of those balances overshoot against
        progress. MultiplicativeOC gets this damping from its eta exponent
        instead and so does not need the extra factor.
        """
        move = super().current_move(iteration)
        if beta is not None and beta > 1.0:
            move = move / np.sqrt(beta)
        return move

    def update(self, rho_e, sensitivity, iteration, beta=None,
               fixed_mask=None, fixed_values=None):
        sensitivity = self.normaliser(sensitivity)
        move = self.current_move(iteration, beta)

        span = float(np.max(np.abs(sensitivity))) + 1.0 / self.lr
        lam_low, lam_high = -span, span
        design_new, volume = rho_e, 0.0

        for _ in range(self.n_bisect):
            lam_mid = 0.5 * (lam_low + lam_high)
            design_new = np.maximum(
                np.maximum(0.0, rho_e - move),
                np.minimum(np.minimum(1.0, rho_e + move),
                           rho_e - self.lr * (sensitivity + lam_mid)))
            if fixed_mask is not None:
                design_new = np.where(fixed_mask, fixed_values, design_new)

            volume = self.projected_volume(design_new, beta)
            if volume > self.volume_fraction:
                lam_low = lam_mid
            else:
                lam_high = lam_mid
            if (lam_high - lam_low) < 1e-12 * span:
                break

        self.lam = 0.5 * (lam_low + lam_high)
        return design_new

    def describe(self):
        return (f"AdditiveOC(move={self.move}, lr={self.lr}, "
                f"norm={self.normaliser.describe()})")