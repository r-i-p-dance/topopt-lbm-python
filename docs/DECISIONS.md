# Decision log

Why things are the way they are. Add an entry when a choice was
non-obvious or when you rejected an alternative.

## Sensitivity formula (two terms)
G = dalpha/drho_bar * drho_bar/drho_e * [|u|^2 + (omega^2/2) * sum(g * f_neq)]

Design enters through omega_eff = 2/(2*tau + alpha), so the implicit term
carries d(omega)/d(alpha) = -omega^2/2 and dR/d(omega) = -f_neq. Papers
using velocity-scaling in the equilibrium have a different single-term
formula — not applicable here.

`g` is the adjoint state BEFORE the adjoint collision (P^T S^T B^T f_hat),
not the end-of-step field. The design perturbation is born in collision and
propagates through P, S, B, so the transposed chain must reach the adjoint
before C^T.

## Multiplicative OC, not additive
The additive form is not scale-invariant: lambda absorbs the mean of G but
nothing absorbs its scale, so the step is proportional to |G| (~1e-5).
Every normalisation strategy either fails to fix the size or removes
convergence. Multiplicative is scale-invariant for free and its optimum is
a fixed point of the update.

## Geometric continuation, not staircase
Staircase makes J plateau within each block, which defeats windowed
convergence detection, and the block length is arbitrary. Geometric is
parameterised by `complete_by` so the tunable is a design decision, not a
magic growth rate. Ramp |G| at roughly 1%/iteration.

## alpha_end from greyness, not intuition
Both cases reach greyness ~0.008 at alpha_max = 40. Higher values buy
nothing and destabilise: lambda loses track of |G|, B/lambda exceeds ~1000,
every cell saturates the move limit and the design is scrambled in one
iteration. Watch B/lambda — a plateau is healthy, unbounded growth is a
warning.

## beta_end = 2, not 8
Greyness is 0.015 at beta=2 and 0.005 at beta=8 — 1% more binarity for a
gradient that becomes a delta function. At high beta, 98% of cells fall
outside the projection band and lose their sensitivity entirely, which
defeats both OC forms.

## Convergence metric on velocity for ObstacleChannel
Velocity inlet derives rho; zero-gradient outflow anchors nothing; with
periodic_x=False streaming does not conserve mass. Total mass drifts
secularly, so max|f - f_old| never falls below tolerance while the flow is
genuinely converged. Velocity is a momentum/density ratio and is blind to
the drift. Cases with a pressure anchor converge on `f` correctly.

## Adjoint has no Mach constraint
`check_stability_running` is a no-op on AdjointLattice. The adjoint field is
a sensitivity, not a velocity; its moments carry no Mach interpretation.
For the same reason `AdjointLattice.macro` does NOT divide by sum(f) —
that sum is not a density and can pass through zero, producing spurious
1e5 spikes from a field bounded within +/-0.15.