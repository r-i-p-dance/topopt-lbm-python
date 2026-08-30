"""Compare finished runs across resolutions, from their archives alone.

A mesh-independence study asks whether the design the optimizer found is a
property of the physics or of the grid. Both runs are loaded from disk, so
answering it costs nothing beyond the runs themselves.

All resolutions are powers of two, so the coarse field is block-upsampled to
the fine grid by an integer factor using the solver's own `upsample`.

Note the filter radius is measured in CELLS: two runs at different ny with
the same radius are solving different regularised problems, and the study
would be measuring the filter rather than the mesh. Each archive stores
radius_over_ny, which is the quantity to hold constant.
"""

from pathlib import Path

import numpy as np

from lbm.src.plot import style
from lbm.src.plot.convergence import plot_convergence
from lbm.src.plot.difference import plot_field_comparison
from lbm.src.study.grid_convergence import upsample

OUT_DIR = "results/plots/mesh"


def compare_runs(ref, coarse, threshold=0.5):
    """Measure `coarse` against `ref`. Both are loaded RunArchives.

    Returns the velocity L2 error, the fraction of designable cells where
    the two resolutions agree on solid vs fluid, and the hydraulic numbers.
    """
    factor = ref.ny // coarse.ny
    u_up = upsample(coarse.u_norm, factor)
    obstacle_up = upsample(coarse.obstacle, factor)
    rho_up = upsample(coarse.rho_bar, factor)

    fluid = (~ref.obstacle) & (~obstacle_up)
    L2 = np.sqrt(np.sum((u_up - ref.u_norm)[fluid] ** 2)
                 / np.sum(ref.u_norm[fluid] ** 2))

    # Denominator is the cells the optimizer was free to choose: the pinned
    # wall border always agrees and would inflate the number.
    designable = ~ref.fixed_mask
    agree = ref.design(threshold)[designable] == (rho_up >= threshold)[designable]

    hydraulic_ref = ref.meta["hydraulic"]
    hydraulic_coarse = coarse.meta["hydraulic"]
    return {
        "ny_ref": ref.ny, "ny_coarse": coarse.ny,
        "L2_velocity": float(L2),
        "agreement": float(np.mean(agree)),
        "power_ref": hydraulic_ref["power"],
        "power_coarse": hydraulic_coarse["power"],
        "euler_ref": hydraulic_ref["euler_number"],
        "euler_coarse": hydraulic_coarse["euler_number"],
    }


def plot_comparison(ref, coarse, result, out_dir=OUT_DIR):
    """Velocity and design difference panels for one resolution pair."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    factor = ref.ny // coarse.ny
    obstacle_up = upsample(coarse.obstacle, factor)
    stem = f"{coarse.ny}_vs_{ref.ny}"

    plot_field_comparison(
        ref.u_norm, upsample(coarse.u_norm, factor),
        save_path=str(out / f"velocity_{stem}"),
        title=f"|u|/u_max, Ny={ref.ny} vs {coarse.ny}   "
              f"L2={result['L2_velocity']:.3e}",
        obstacle_ref=ref.obstacle, obstacle_coarse=obstacle_up)

    plot_field_comparison(
        ref.rho_bar, upsample(coarse.rho_bar, factor),
        save_path=str(out / f"design_{stem}"),
        title=f"Design, Ny={ref.ny} vs {coarse.ny}   "
              f"agreement={100 * result['agreement']:.1f}%",
        # rho_bar already spans [0, 1], so no gamma compression.
        cmap_field=style.DESIGN, cmap_diff=style.SEQUENTIAL_COOL, gamma=1.0,
        obstacle_ref=ref.obstacle, obstacle_coarse=obstacle_up)


def mesh_independence_study(runs, out_dir=OUT_DIR, threshold=0.5):
    """Compare every run against the finest. Returns the results."""
    runs = sorted(runs, key=lambda a: a.ny)
    ref, coarser = runs[-1], runs[:-1]
    results = [compare_runs(ref, c, threshold) for c in coarser]

    print(f"Reference: {ref.describe()}")
    for result in results:
        print(f"  Ny {result['ny_coarse']:>4} vs {result['ny_ref']:<4}  "
              f"L2={result['L2_velocity']:.3e}  "
              f"agreement={100 * result['agreement']:.1f}%  "
              f"Eu={result['euler_coarse']:.4g}")

    for archive, result in zip(coarser, results):
        plot_comparison(ref, archive, result, out_dir)

    if len(results) >= 2:
        plot_convergence([r["ny_coarse"] for r in results],
                         [r["L2_velocity"] for r in results],
                         save_path=str(Path(out_dir) / "L2_convergence.png"),
                         title=f"Mesh convergence vs Ny={ref.ny}")
    return results
