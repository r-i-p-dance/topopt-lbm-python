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


def _pair_figure(recorder, mode):
    """Just the flow above the design, on the poster grid.

    Each field stays F modules square, two leadings apart with one leading of
    margin — the same rhythm the horizontal recorder gives its field block.

    Reuses the recorder's own images rather than re-normalising, so the two
    panels carry exactly the colour scales the movie used.
    """
    F = recorder.modules
    m = style.MARGINS[mode]
    unit = style.BASELINE_MM / style.MM_PER_IN
    total_w, total_h = F + 2 * m, 2 * F + 2 + 2 * m
    fig = plt.figure(figsize=(total_w * unit, total_h * unit))

    # From the bottom: margin, design F, gap 2, flow F, margin.
    for source, bottom in ((recorder.img1, m + F + 2), (recorder.img4, m)):
        ax = fig.add_axes([m / total_w, bottom / total_h,
                           F / total_w, F / total_h])
        ax.imshow(source.get_array(), cmap=source.cmap, norm=source.norm,
                  origin="lower", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(source.axes.get_title())

    style.apply_figure_style(fig, fig.axes, image_axes=fig.axes)
    return fig


def replot_archive(archive, save_path=None, dpi=300, formats=("pdf",),
                   layout="full", modules=11, orientation="vertical",
                   mode="poster", final_design=False):
    """Redraw the recorder figure from an archive and save it.

    layout: "full" (all 12 panels), "fields" (the 2x2 block), "metrics", or
    "pair" (the flow above the design, for a README).
    modules is the field panel's side on the poster grid, orientation is
    where the metric plots sit beside it, and mode is "poster" or "readme" —
    how much margin the page carries, see style.MARGINS.
    final_design draws the design after the last optimizer update rather
    than the one the flow was solved on.
    """
    recorder = OptimizationRecorder(
        # vmax_fwd is frozen on the movie's first frame and never revisited,
        # so restoring it keeps the forward panel on the same colour scale.
        vmax_fwd=archive.meta["recorder"].get("vmax_fwd"),
        modules=modules, orientation=orientation, mode=mode)
    recorder.setup_static(archive.nx, archive.ny)
    recorder.set_obstacle(archive.obstacle)

    for name in RECORDER_SERIES:
        setattr(recorder, name, list(archive.series[name]))

    recorder.draw_fields(
        np.hypot(archive.ux, archive.uy),
        np.hypot(archive.adj_ux, archive.adj_uy),
        archive.G,
        archive.rho_bar_final if final_design else archive.rho_bar)
    recorder.draw_series()

    if layout == "pair":
        fig = _pair_figure(recorder, mode)
        plt.close(recorder.fig)
    else:
        fig = recorder.fig
        if layout != "full":
            # Removing axes is enough: style.save crops with
            # bbox_inches="tight", so the canvas shrinks to what is left.
            for ax in (recorder.metric_axes if layout == "fields"
                       else recorder.field_axes):
                ax.remove()

    # "full" and "pair" are whole numbers of modules and must be saved at that
    # size. "fields" and "metrics" exist precisely to crop away what is left,
    # so they are off-grid by design and keep the tight save.
    write = style.save_exact if layout in ("full", "pair") else style.save

    save_path = Path(save_path) if save_path else OUT_DIR / archive.stem
    save_path.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        target = save_path.with_suffix(f".{extension}")
        write(fig, target, dpi=dpi)
        print(f"Replotted {archive.stem} -> {target}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(prog="python -m topopt.src.plot.replot")
    parser.add_argument("archive")
    parser.add_argument("--out", default=None)
    parser.add_argument("--format", nargs="+", default=["pdf"])
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--layout", default="full",
                        choices=("full", "fields", "metrics", "pair"))
    parser.add_argument("--modules", type=int, default=11)
    parser.add_argument("--orientation", default="vertical",
                        choices=("vertical", "horizontal"))
    parser.add_argument("--mode", default="poster",
                        choices=("poster", "readme"))
    parser.add_argument("--final-design", action="store_true")
    args = parser.parse_args()

    replot_archive(load_run(args.archive), args.out, dpi=args.dpi,
                   formats=args.format, layout=args.layout,
                   modules=args.modules, orientation=args.orientation,
                   mode=args.mode,
                   final_design=args.final_design)


if __name__ == "__main__":
    main()
