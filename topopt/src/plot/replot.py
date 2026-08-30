"""Rebuild a run's figure from its archive — no solver, no ffmpeg.

Everything the recorder draws is in the archive, so this reloads it and
calls the recorder's own `draw_fields` / `draw_series`. Using the same
drawing code as the movie is the point: a second implementation would drift.

    python -m topopt.src.plot.replot results/arrays/triple_32Ny.npz
    python -m topopt.src.plot.replot results/arrays/triple_32Ny.npz \
        --format png pdf --dpi 600 --layout fields

Output goes to results/plots/replots/, one level below results/plots/,
because utils/cleanup.py treats files sitting directly in results/plots/ as
run artifacts and would delete a stray figure there.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from lbm.src.plot import style
from topopt.src.plot.optimization_recorder import OptimizationRecorder
from topopt.src.utils.archive import RECORDER_SERIES, load_run

OUT_DIR = Path("results") / "plots" / "replots"


def replot_archive(archive, save_path=None, dpi=300, formats=("png",),
                   layout="full", fig_w=10.0, final_design=False):
    """Redraw the recorder figure from an archive and save it.

    layout: "full" (all 12 panels), "fields" (the 2x2 block) or "metrics".
    final_design draws the design after the last optimizer update rather
    than the one the flow was solved on.
    """
    recorder = OptimizationRecorder(
        # vmax_fwd is frozen on the movie's first frame and never revisited,
        # so restoring it keeps the forward panel on the same colour scale.
        vmax_fwd=archive.meta["recorder"].get("vmax_fwd"))
    recorder.setup_static(archive.nx, archive.ny,
                          volume_fraction=archive.meta["case"]["volume_fraction"],
                          fig_w=fig_w)
    recorder.set_obstacle(archive.obstacle)

    for name in RECORDER_SERIES:
        setattr(recorder, name, list(archive.series[name]))

    recorder.draw_fields(
        np.hypot(archive.ux, archive.uy),
        np.hypot(archive.adj_ux, archive.adj_uy),
        archive.G,
        archive.rho_bar_final if final_design else archive.rho_bar)
    recorder.draw_series()

    if layout != "full":
        # Removing axes is enough: style.save crops with bbox_inches="tight",
        # so the canvas shrinks to whatever is left.
        drop = (recorder.metric_axes if layout == "fields"
                else recorder.field_axes)
        for ax in drop:
            ax.remove()

    save_path = Path(save_path) if save_path else OUT_DIR / archive.stem
    save_path.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        target = save_path.with_suffix(f".{extension}")
        style.save(recorder.fig, target, dpi=dpi)
        print(f"Replotted {archive.stem} -> {target}")
    plt.close(recorder.fig)


def main():
    parser = argparse.ArgumentParser(prog="python -m topopt.src.plot.replot")
    parser.add_argument("archive")
    parser.add_argument("--out", default=None)
    parser.add_argument("--format", nargs="+", default=["png"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--layout", default="full",
                        choices=("full", "fields", "metrics"))
    parser.add_argument("--fig-width", type=float, default=10.0)
    parser.add_argument("--final-design", action="store_true")
    args = parser.parse_args()

    replot_archive(load_run(args.archive), args.out, dpi=args.dpi,
                   formats=args.format, layout=args.layout,
                   fig_w=args.fig_width, final_design=args.final_design)


if __name__ == "__main__":
    main()
