"""Save a finished run completely, so it never has to be re-run.

One compressed npz per run, holding the final fields, the per-iteration
series the recorder collected, and a JSON blob of every scalar. That is
enough to replot the figure (see plot/replot.py) and to compare runs across
resolutions (see study/mesh_independence.py) without touching the solver.

    field_*     (nx, ny)     final state
    series_*    (n,)         per-iteration history
    meta_json   0-d unicode  every scalar and string, as JSON

The scalars are one JSON document rather than ~35 flat keys because npz has
no place for a string and no nesting, and because a new case then needs no
change here to have its parameters saved.

Two design pairs are stored. The driver rebinds rho_e after the final
optimizer update, so the design that was SOLVED and the design that was
RETURNED are one iteration apart: rho_bar, ux, uy and G all belong to
rho_e, not to rho_e_final.
"""

import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The twelve series the recorder accumulates.
RECORDER_SERIES = ("loss", "lam", "alpha", "beta", "fwd_iters", "adj_iters",
                   "greyness", "change_max", "change_mean", "move_limit",
                   "b_over_lam", "change_ratio")

# Big arrays and lattice constants, kept out of the JSON.
_SKIP = {"f", "f_new", "f_eq", "g", "source", "rho", "ux", "uy", "obstacle",
         "rho_e", "rho_bar", "omega_eff", "fixed_mask", "fixed_values",
         "c", "cx", "cy", "w", "opposite", "neg_cx", "neg_cy",
         "kernel", "kernel_sum", "forward", "adjoint", "fwd"}


def _scalars(obj):
    """Every scalar attribute of an object, for the metadata blob."""
    return {name: value.item() if isinstance(value, np.generic) else value
            for name, value in vars(obj).items()
            if not name.startswith("_") and name not in _SKIP
            and isinstance(value, (bool, int, float, str, np.generic))}


def hydraulics(fwd):
    """Pumping power and the dimensionless pressure drop.

    p = rho/3, so delta_p is the inlet-to-outlet density difference over 3.
    The Euler number delta_p/u_max^2 is the quantity to compare across
    resolutions: raw power scales with the number of cells spanning the
    inlet, so it changes with ny even for the same physical design.
    """
    # The pressure anchor is whichever boundary is held at a fixed density,
    # and each case names it after its own geometry: rho_out, rho_east,
    # rho_south, rho_west. Take whichever float the forward defines rather
    # than matching a list of names, so a new case needs no change here.
    rho_ref = next(value for name, value in vars(fwd).items()
                   if name.startswith("rho_") and isinstance(value, float))
    delta_p = (fwd.inlet_density() - rho_ref) / 3.0 if hasattr(
        fwd, "inlet_density") else (
        float(np.mean(fwd.rho[0, fwd.j_from:fwd.j_to])) - rho_ref) / 3.0
    # From the profile, not fwd.flux_in: only the split cases store that
    # attribute, and it is the sum of exactly this array.
    flux_in = float(np.sum(fwd.u_profile))
    return {"power": flux_in * delta_p,
            "delta_p": delta_p,
            "euler_number": delta_p / fwd.u_max ** 2,
            "flux_in": flux_in}


def save_run_archive(path, case, fwd, adj, driver, recorder,
                     rho_e, rho_e_final, rho_bar_final, G, G_raw,
                     alpha_max, beta, design_state, run_state):
    """Write the complete final state of one run."""
    from topopt.src.core.brinkman import alpha_from_density

    arrays = {
        "field_rho_e": rho_e, "field_rho_bar": fwd.rho_bar,
        "field_rho_e_final": rho_e_final, "field_rho_bar_final": rho_bar_final,
        "field_obstacle": fwd.obstacle, "field_fixed_mask": case.fixed_mask,
        "field_ux": fwd.ux, "field_uy": fwd.uy, "field_rho": fwd.rho,
        "field_omega_eff": fwd.omega_eff,
        "field_alpha": alpha_from_density(fwd.rho_bar, alpha_max, case.q),
        "field_G": G, "field_G_raw": G_raw,
        "field_adj_ux": adj.ux, "field_adj_uy": adj.uy,
    }

    if recorder is not None:
        for name in RECORDER_SERIES:
            arrays["series_" + name] = np.asarray(getattr(recorder, name))

    meta = {
        "case": {"class": type(case).__name__, **_scalars(case)},
        "forward": {"class": type(fwd).__name__, **_scalars(fwd)},
        "design_state": {**design_state, "alpha_max": alpha_max, "beta": beta},
        "hydraulic": hydraulics(fwd),
        # radius is in CELLS, so this ratio is what a mesh study must hold
        # constant — the same radius at two resolutions is a different filter.
        "filter": {"radius": driver.filter.radius,
                   "radius_over_ny": driver.filter.radius / case.ny},
        "optimizer": driver.optimizer.describe(),
        "continuation": driver.continuation.describe(),
        "recorder": {"vmax_fwd": recorder.vmax_fwd} if recorder else {},
        "run": run_state,
    }
    arrays["meta_json"] = np.array(json.dumps(meta))

    path = Path(path)
    np.savez_compressed(path, **arrays)
    print(f"Archive saved to: {path}  ({path.stat().st_size // 1024} KB)")


@dataclass
class RunArchive:
    """A loaded run. Fields and series are reachable as attributes."""

    stem: str
    meta: dict
    fields: dict
    series: dict

    def __getattr__(self, name):
        if name in self.fields:
            return self.fields[name]
        if name in self.series:
            return self.series[name]
        raise AttributeError(name)

    @property
    def nx(self):
        return self.meta["case"]["nx"]

    @property
    def ny(self):
        return self.meta["case"]["ny"]

    @property
    def u_max(self):
        return self.meta["forward"]["u_max"]

    @property
    def u_norm(self):
        """Velocity magnitude over this run's own u_max, as the grid
        convergence study normalises it."""
        return np.hypot(self.ux, self.uy) / self.u_max

    def design(self, threshold=0.5):
        """Boolean fluid mask — rho_e = 1 is fluid."""
        return self.rho_bar >= threshold

    def describe(self):
        return f"{self.stem}: {self.meta['case']['class']} {self.nx}x{self.ny}"


def load_run(path):
    fields, series, meta = {}, {}, {}
    with np.load(path) as data:
        for key in data.files:
            if key == "meta_json":
                meta = json.loads(str(data[key]))
            elif key.startswith("field_"):
                fields[key[6:]] = data[key]
            elif key.startswith("series_"):
                series[key[7:]] = data[key]
    return RunArchive(Path(path).stem, meta, fields, series)


def load_runs(pattern="*", results_root="results"):
    """Every matching archive, coarsest first — a mesh sweep's input."""
    paths = sorted(glob.glob(str(Path(results_root) / "arrays" / f"{pattern}.npz")))
    return sorted((load_run(p) for p in paths), key=lambda a: a.ny)
