# URSS: Topology Optimization of Laminar Flows via Lattice Boltzmann

Two sibling repos, both installed editable into `topopt-lbm-python/.venv`:
- `lbm-2d-python` — verified D2Q9 LBM solver
- `topopt-lbm-python` — adjoint-based topology optimization on top of it

## Prior decisions

`docs/DECISIONS.md` records why non-obvious choices were made and which
alternatives were rejected. Read it before proposing changes to the
optimizer, the continuation schedule, the sensitivity formula, or the
convergence criteria — those have all been through several iterations and
the failure modes are documented.

## Non-negotiables

**The adjoint is verified to machine precision. Do not modify it casually.**
Every operator (collision, streaming, bounce-back, Zou-He) has been proven
to be the exact discrete transpose of its forward counterpart via
dot-product tests at 1e-11, and the full gradient matches finite
differences exactly. If you touch `core/adjoint_base.py`, any
`cases/*/adjoint.py`, `core/sensitivity.py`, or `core/objective.py`, run:

    python tests/diagnose_transpose.py
    python tests/measure_spectral_radius.py

All operators and compositions must report ~1e-11, and rho(M^T) must equal
rho(M). If they don't, the change is wrong.

**Streaming must be a permutation.** `periodic_x = True` on every case with
open boundaries. Non-periodic streaming is not invertible, so S^T is not
backward streaming and the adjoint breaks. Zou-He overwrites exactly the
wrapped-in populations, so the forward result is bit-identical.

**Adjoint boundary conditions:** zero the directions the forward
RECONSTRUCTS; the directions it READS pick up coupling terms from the
reconstructed adjoints. Index ranges must match the forward exactly.
Derive per-kernel — never mirror one boundary's coefficients to another,
the sign conventions differ.

## Design principles

**If removing code doesn't make it worse, remove it.** No unused
parameters, no defaults that are never reached, no abstraction layers that
exist to unify a duplication you just introduced. This codebase grows fast;
strip it.

**No single-letter variable names.** `continuation`, not `c`. `delta`, not
`d`. Keep the same name for the same thing across files.

**No `*args`.** Write signatures out explicitly so the reader and the
editor can both see what a class needs.

**Test at realistic magnitudes.** Sensitivities here are ~1e-5. A test
using `randn()` (order 1) passed a bisection bracket that failed completely
on real data.

**Test the object's own method, never a reconstruction of it.** Two bugs
hid behind that violation — a hand-built closure passed `fwd.obstacle`
explicitly while production read `self.obstacle`, which was all-False.

## Key conventions

- `rho_e = 1` is FLUID, `rho_e = 0` is SOLID (modern convention)
- Set `Re`, derive `u_max = Re * nu / L_char`. Never fix `u_max`.
- `obstacle` = where fluid cannot go (physics, on the lattice).
  `fixed_mask` = where the optimizer may not look (optimization, on the
  case). They overlap but differ: openings are fluid yet pinned.
- Physics runs on `rho_bar` (projected), never raw `rho_e`.
- Convergence is judged on the OBJECTIVE, not the design variable: a
  near-binary design has no interior cells, so the OC fixed point does not
  exist and interface cells flip forever.

## Visual identity

`lbm/src/plot/style.py` is the single source. Amber = physical fluid
quantities. Cyan = dual and error quantities. Magenta = the third thing
that is neither. Near-black ground; saturated accents; CVD-safe via the
cyan/amber axis plus dash patterns, never hue muting alone. Neue Haas
Grotesk. Figures go on a poster — they are graphic design, not plots.

Verify colormaps with `python tests/preview_palette.py`; sequential maps
must report monotonic luminance.

## Setup

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -e ..\lbm-2d-python
    pip install -e .
    pip install pytest

Verify: `python -c "import lbm, topopt; print(lbm.__file__, topopt.__file__)"`
Both must print source paths, not site-packages. `__file__ = None` means a
stale namespace package — clean site-packages and reinstall.