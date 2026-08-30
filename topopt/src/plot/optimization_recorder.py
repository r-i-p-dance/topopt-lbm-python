import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FFMpegWriter
from matplotlib.ticker import (ScalarFormatter, MaxNLocator, AutoMinorLocator,
                               FixedLocator, FuncFormatter, LogLocator,
                               NullFormatter)
from lbm.src.plot import style

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Neue Haas Grotesk Display Pro",
                        "Helvetica Neue", "Helvetica", "Arial"],
    "font.weight": "medium",
    "axes.titleweight": "medium",
})

# Metric panel typography. Small, but never so small that a number on the
# poster stops being a number.
TICK_SIZE = 7
TITLE_SIZE = 9.5
LABEL_SIZE = 7.5


# Major-tick mantissas, coarse to fine. Decades and thirds-of-a-decade
# ONLY. Both are near-equal ratios — 10, and 3.16 — so consecutive ticks
# land at near-equal distances on a log axis. Sets like (1, 2, 5) or
# (1, 2, 3, 5, 7) are round numbers but sit at visibly unequal spacings,
# and a linear set such as 20/30/40/50 is worse still: evenly-spaced
# numbers at unevenly-spaced positions read as a broken axis.
_LADDERS = ((1.0,), (1.0, 3.0))
_MAX_TICKS = 5


def _candidate_ticks(subs, lo, hi):
    """Every subs*10^k in a range comfortably wrapping [lo, hi]."""
    lo_e = int(np.floor(np.log10(lo))) - 1
    hi_e = int(np.ceil(np.log10(hi))) + 1
    return sorted(s * 10.0**e
                  for e in range(lo_e, hi_e + 1) for s in subs)


def _axis_formatter(ticks):
    """One notation for the whole axis, decided from all of its ticks.

    Deciding per tick is what produced axes reading 0.01, 0.03 and then
    3 x 10^-3: each label was individually defensible and the column as a
    whole was unreadable. The choice has to be made once, for the set.
    """
    plain = all(1e-2 <= t < 1e4 for t in ticks)

    def fmt(value, _pos=None):
        if value <= 0:
            return ""
        if plain:
            return f"{value:g}"
        exponent = int(np.floor(np.log10(value) + 1e-9))
        mantissa = value / 10.0**exponent
        if mantissa >= 9.995:                  # 1e-3 must not print as 10e-4
            mantissa, exponent = mantissa / 10.0, exponent + 1
        return rf"${mantissa:g}\times10^{{{exponent}}}$"

    return fmt


def _retick_log(ax):
    """Even, round, capped major ticks on a log axis at any span.

    The limits are snapped OUTWARD onto the tick set, rather than the ticks
    being fitted inside whatever the data happened to reach. That is what
    buys a tick at each end of the axis and keeps the spacing uniform;
    letting the data set the limits is what leaves a panel holding a single
    label.

    Runs every frame, because the panels autoscale every frame.
    """
    lo, hi = ax.get_ylim()
    if not (lo > 0 and hi > lo):
        return

    chosen = None
    for subs in _LADDERS:
        values = _candidate_ticks(subs, lo, hi)
        below = [v for v in values if v <= lo * (1.0 + 1e-9)]
        above = [v for v in values if v >= hi * (1.0 - 1e-9)]
        lo_s = below[-1] if below else lo
        hi_s = above[0] if above else hi
        ticks = [v for v in values
                 if lo_s * (1.0 - 1e-9) <= v <= hi_s * (1.0 + 1e-9)]
        # Ladders run coarse to fine, so once one overflows, so do the rest.
        if chosen is not None and len(ticks) > _MAX_TICKS:
            break
        chosen = (lo_s, hi_s, ticks)

    lo_s, hi_s, ticks = chosen
    if len(ticks) > _MAX_TICKS:
        # Even one label per decade is too many, so keep every k-th. The
        # stride is uniform, so the surviving ticks stay evenly spaced; the
        # top of the axis may lose its label, which costs less than a
        # crowded column of exponents.
        stride = int(np.ceil(len(ticks) / _MAX_TICKS))
        ticks = ticks[::stride]

    # auto=True is load-bearing: a plain set_ylim turns y-autoscaling OFF,
    # which would freeze these limits at frame one and let the series walk
    # straight off the panel. The limits are re-snapped every frame anyway,
    # right after autoscale_view has recomputed them from the data.
    ax.set_ylim(lo_s, hi_s, auto=True)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(_axis_formatter(ticks)))
    # Minor gridlines carry the decade structure. Without them a series
    # spanning well under a decade is hard to tell from a linear one —
    # because over that little range it very nearly is one.
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs="all", numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())


class OptimizationRecorder:
    """Records the optimization loop as an MP4.

    Layout:
      Rows 0-1: four field panels as a 2x2 block (each spans 2 columns)
        [0,0:2] Forward velocity     [0,2:4] Adjoint momentum
        [1,0:2] Sensitivity          [1,2:4] Pipe design
      Rows 2-3: eight metric plots, four per row
        dissipation | design change | alpha & beta  | solver iterations
        greyness    | OC multiplier | B / lambda    | volume error

      Column 0 is the pair being minimised, both amber because both are
      physical measures. The OC multiplier sits beside B / lambda, which is
      derived from it, which also collects the magenta panels into one run
      along the bottom row instead of scattering them across the grid.

    One optimization iteration = one frame. Grid size and output path are
    supplied by the driver via setup(), so the caller repeats neither.

    MP4 rather than GIF: a GIF holds 256 colours per frame, and with three
    colormaps in one figure the adaptive palette leaves too few true levels,
    so intermediate values get dithered into coloured speckle. H.264 is full
    24-bit colour, produces much smaller files, and embeds in slides.

    Requires ffmpeg on PATH.
    """

    def __init__(self, fps=15, dpi=150, gamma=0.6, vmax_fwd=None,
                 bitrate=4000, g_linthresh_pct=50, still_dpi=300):
        self.fps = fps
        self.dpi = dpi
        # The still is a poster asset and is not paying the movie's per-frame
        # encode cost, so it has no reason to inherit the movie's resolution.
        self.still_dpi = still_dpi
        self.gamma = gamma
        self.vmax_fwd = vmax_fwd
        self.bitrate = bitrate
        self.path = None
        self.nx = self.ny = None
        self.g_linthresh_pct = g_linthresh_pct
        # Wall mask. setup() runs before the driver has built the solvers,
        # so this arrives later via set_obstacle().
        self._solid = None

        self.loss, self.lam = [], []
        self.alpha, self.beta = [], []
        self.fwd_iters, self.adj_iters = [], []
        self.greyness = []
        self.change_max, self.change_mean, self.move_limit = [], [], []
        self.b_over_lam = []
        self.volume_error = []

    # ------------------------------------------------------------------
    def setup(self, nx, ny, path, volume_fraction=None):
        """Build the figure once the grid size and output path are known."""
        self.nx, self.ny = nx, ny
        self.path = path
        self.volume_fraction = volume_fraction
        print(f"Recording to: {self.path}")

        # Every panel is strictly square, so the whole layout follows from
        # one number: the width of a single grid column. A field panel spans
        # two columns PLUS the gutter between them, so its row has to be
        # exactly that much taller — the obvious 2:1 height ratio is wrong
        # by the width of one gutter, and matplotlib pays for the error by
        # shrinking the square and leaving a gap. Deriving the figure height
        # from the column width instead of guessing it is what makes the
        # panels tile with no slack anywhere.
        WSPACE, HSPACE = 0.42, 0.20
        FIELD_SPAN = 2 + WSPACE                 # field panel width, in columns
        LEFT, RIGHT, TOP, BOTTOM = 0.07, 0.985, 0.975, 0.03

        fig_w = 10.0
        column = (RIGHT - LEFT) * fig_w / (4 + 3 * WSPACE)
        row_heights = [FIELD_SPAN * column, FIELD_SPAN * column,
                       column, column]
        # hspace is a fraction of the MEAN row height, not of each row.
        gutters = 3 * HSPACE * (sum(row_heights) / 4)
        fig_h = (sum(row_heights) + gutters) / (TOP - BOTTOM)

        self.fig = plt.figure(figsize=(fig_w, fig_h))
        gs = GridSpec(4, 4, figure=self.fig,
                      height_ratios=[FIELD_SPAN, FIELD_SPAN, 1, 1],
                      hspace=HSPACE, wspace=WSPACE,
                      left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM)

        self.ax1 = self.fig.add_subplot(gs[0, 0:2])
        self.ax2 = self.fig.add_subplot(gs[0, 2:4])
        self.ax3 = self.fig.add_subplot(gs[1, 0:2])
        self.ax4 = self.fig.add_subplot(gs[1, 2:4])

        self.ax5 = self.fig.add_subplot(gs[2, 0])
        self.ax6 = self.fig.add_subplot(gs[2, 1])
        self.ax7 = self.fig.add_subplot(gs[2, 2])
        self.ax8 = self.fig.add_subplot(gs[2, 3])
        self.ax9 = self.fig.add_subplot(gs[3, 0])
        self.ax10 = self.fig.add_subplot(gs[3, 1])
        self.ax11 = self.fig.add_subplot(gs[3, 2])
        self.ax12 = self.fig.add_subplot(gs[3, 3])

        self.field_axes = (self.ax1, self.ax2, self.ax3, self.ax4)
        self.metric_axes = (self.ax5, self.ax6, self.ax7, self.ax8,
                            self.ax9, self.ax10, self.ax11, self.ax12)

        z = np.zeros((nx, ny))
        vf = self.vmax_fwd if self.vmax_fwd is not None else 1.0

        # Copies, not the module-level maps: set_bad mutates the colormap in
        # place, and these objects are shared across every figure in both
        # repos.
        #
        # The design panel takes the wall mask as well. DESIGN already
        # renders a zero design variable as near-ground, so without the mask
        # the pinned walls and the solid the optimizer chose come out the
        # same colour. Painting the walls SOLID separates what cannot change
        # from what was decided, and keeps the geometry reading identically
        # across all four panels.
        def _with_solid(cmap):
            cmap = cmap.copy()
            cmap.set_bad(style.SOLID)
            return cmap

        self.img1 = self.ax1.imshow(          # forward velocity: physical
            z.T, cmap=_with_solid(style.SEQUENTIAL_FLOW), origin="lower",
            aspect="auto",
            norm=mcolors.PowerNorm(gamma=self.gamma, vmin=0.0, vmax=vf))
        self.img2 = self.ax2.imshow(          # adjoint momentum: dual
            z.T, cmap=_with_solid(style.SEQUENTIAL_DUAL), origin="lower",
            aspect="auto",
            norm=mcolors.PowerNorm(gamma=self.gamma, vmin=0.0, vmax=1.0))
        self.img3 = self.ax3.imshow(
            z.T, cmap=_with_solid(style.DIVERGING), origin="lower",
            aspect="auto",
            norm=mcolors.TwoSlopeNorm(vmin=-1e-4, vcenter=0, vmax=1e-4))
        self.img4 = self.ax4.imshow(
            z.T, cmap=_with_solid(style.DESIGN), origin="lower",
            aspect="auto", vmin=0, vmax=1)

        # Every metric line is achromatic — see style.S_METRIC. A panel with
        # one series gets TEXT solid; a panel with two adds MUTED dashed;
        # a reference line gets MUTED dotted. Colour stays on the fields
        # above, which is what the eye should reach first.
        colour, linestyle = style.S_METRIC
        self.line_loss, = self.ax5.semilogy([], [], color=colour, ls=linestyle, lw=2)

        # Design change against the move limit. When the solid line meets
        # the dotted one, every moving cell is clipped and the optimizer is
        # saturated rather than converging. The gap between max and mean
        # shows how localised the motion is.
        colour, linestyle = style.S_METRIC
        self.line_dmax, = self.ax6.semilogy([], [], color=colour, ls=linestyle,
                                            lw=2, label="max")
        colour, linestyle = style.S_ALT
        self.line_dmean, = self.ax6.semilogy([], [], color=colour, ls=linestyle,
                                             lw=1.6, label="mean")
        colour, linestyle = style.S_REF
        self.line_move, = self.ax6.semilogy([], [], color=colour, ls=linestyle,
                                            lw=1.2, label="limit")

        colour, linestyle = style.S_METRIC
        self.line_alpha, = self.ax7.plot([], [], color=colour, ls=linestyle,
                                         lw=2, label="alpha")
        colour, linestyle = style.S_ALT
        self.line_beta, = self.ax7.plot([], [], color=colour, ls=linestyle,
                                        lw=2, label="beta")

        colour, linestyle = style.S_METRIC
        self.line_fwd, = self.ax8.plot([], [], color=colour, ls=linestyle,
                                       lw=2, label="fwd")
        colour, linestyle = style.S_ALT
        self.line_adj, = self.ax8.plot([], [], color=colour, ls=linestyle,
                                       lw=2, label="adj")

        # Greyness: 0 = fully binary. The manufacturability number.
        colour, linestyle = style.S_METRIC
        self.line_grey, = self.ax9.semilogy([], [], color=colour, ls=linestyle, lw=2)

        # OC multiplier. Placed next to B / lambda, which is built from it,
        # so the two read as one pair.
        colour, linestyle = style.S_METRIC
        self.line_lam, = self.ax10.plot([], [], color=colour, ls=linestyle, lw=2)

        # B / lambda: the leading indicator. A plateau is healthy; unbounded
        # growth means lambda has lost the gradient and every cell is about
        # to saturate.
        colour, linestyle = style.S_METRIC
        self.line_ratio, = self.ax11.semilogy([], [], color=colour, ls=linestyle, lw=2)

        # Volume error against target. Should be flat at zero; a departure
        # means the constraint was unreachable within the move limit.
        colour, linestyle = style.S_METRIC
        self.line_vol, = self.ax12.plot([], [], color=colour, ls=linestyle, lw=2)

        # Fixed corners, not "best": "best" is recomputed on every draw, so
        # in a movie the legend would hop between corners frame to frame.
        # Each corner is the one its series leaves empty: solver iteration
        # counts fall as the design settles, so those curves end up bottom
        # right and the legend sits opposite them.
        for ax, corner in ((self.ax6, "lower left"),
                           (self.ax7, "upper left"),
                           (self.ax8, "upper right")):
            ax.legend(loc=corner, fontsize=LABEL_SIZE, frameon=False,
                      labelcolor=style.TEXT, handlelength=1.6,
                      borderpad=0.2, labelspacing=0.25)

        self.ax1.set_title("Forward velocity", fontsize=11)
        self.ax2.set_title("Adjoint momentum", fontsize=11)
        self.ax3.set_title("Sensitivity", fontsize=11)
        self.title4 = self.ax4.set_title("Pipe design", fontsize=11)

        # The four log panels say so in the title. The minor gridlines carry
        # the decade structure, but a reader should not have to infer the
        # scale from the spacing of the rules.
        for ax, name in zip(self.metric_axes,
                            ("Energy dissipation (log)", "Design change (log)",
                             "Alpha & Beta", "Solver iterations",
                             "Greyness (log)", "OC multiplier",
                             "B / lambda (log)", "Volume error")):
            ax.set_title(name, fontsize=TITLE_SIZE)

        for ax in self.field_axes:
            ax.set_box_aspect(1)
            ax.set_xticks([])
            ax.set_yticks([])

        self.log_axes = (self.ax5, self.ax6, self.ax9, self.ax11)

        for ax in self.metric_axes:
            ax.set_box_aspect(1)
            ax.set_xlabel("iteration", fontsize=LABEL_SIZE, labelpad=2)
            ax.tick_params(axis="both", which="major",
                           labelsize=TICK_SIZE, pad=1.5, length=3)
            ax.tick_params(axis="both", which="minor", length=1.5)
            # Iteration count is an integer, and five labels is the point at
            # which a slope can be read off the axis rather than guessed.
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            # Keeps the newest point off the frame as the series grows.
            ax.margins(x=0.04, y=0.10)

        # Linear axes: pull the exponent into a compact offset label so
        # ticks read "-2.2" rather than "-0.00002245".
        for ax in (self.ax7, self.ax8, self.ax10, self.ax12):
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 3))
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
            ax.yaxis.get_offset_text().set_color(style.MUTED)

        for ax in self.log_axes:
            _retick_log(ax)

        style.apply_figure_style(
            self.fig, list(self.field_axes) + list(self.metric_axes),
            image_axes=self.field_axes)

        # Applied after apply_figure_style, which turns on a single major
        # grid. Major rules tie a point to its tick; the minor rules are
        # what make a log axis legible AS a log axis.
        for ax in self.metric_axes:
            ax.grid(True, which="major", color=style.RULE, alpha=0.55,
                    lw=0.7, ls="-")
            ax.grid(True, which="minor", color=style.RULE, alpha=0.20,
                    lw=0.5, ls="-")
            ax.set_axisbelow(True)

        # yuv420p for QuickTime/PowerPoint compatibility; the scale filter
        # forces even pixel dimensions, which H.264 requires and which an
        # odd figsize*dpi would otherwise violate.
        self.writer = FFMpegWriter(
            fps=self.fps, codec="libx264", bitrate=self.bitrate,
            extra_args=["-pix_fmt", "yuv420p",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"])
        self.writer.setup(self.fig, self.path, dpi=self.dpi)

    # ------------------------------------------------------------------
    def set_obstacle(self, obstacle):
        """Walls, painted style.SOLID on the three physics panels.

        Separate from setup() because the driver builds the recorder before
        it builds the solvers, so there is no obstacle to hand over yet.
        Optional: without it the walls simply render at the ramp's zero end,
        as they did before.
        """
        self._solid = np.asarray(obstacle, dtype=bool)

    def _mask(self, field):
        """NaN out the walls so the colormap's `bad` colour takes over."""
        if self._solid is None:
            return field
        return np.where(self._solid, np.nan, field)

    # ------------------------------------------------------------------
    def capture(self, loop, fwd_vel, adj_vel, G, rho_bar,
                J, lam, alpha, beta, fwd_iters, adj_iters,
                greyness=None, change_max=None, change_mean=None,
                move_limit=None, G_max=None, volume=None):
        """Record one optimization iteration as a frame.

        The second metric row is optional: pass the extra keywords and it
        is populated, omit them and those panels stay empty. This keeps the
        recorder usable from any driver.
        """
        # forward: fix the scale from the first frame so it doesn't flicker
        if self.vmax_fwd is None:
            self.vmax_fwd = max(float(np.percentile(fwd_vel, 99.5)), 1e-10)
            self.img1.set_norm(mcolors.PowerNorm(
                gamma=self.gamma, vmin=0.0, vmax=self.vmax_fwd))
        # Masked only at the point of drawing: the norms above are still
        # computed from the raw field, so adding the wall mask changes what
        # the panels look like and not how they are scaled.
        self.img1.set_data(self._mask(fwd_vel).T)

        # adjoint: rescale each frame, its magnitude drifts with continuation
        vmax_adj = max(float(np.percentile(adj_vel, 99)), 1e-10)
        self.img2.set_norm(mcolors.PowerNorm(
            gamma=self.gamma, vmin=0.0, vmax=vmax_adj))
        self.img2.set_data(self._mask(adj_vel).T)

        # sensitivity: symmetric log. At high beta the Heaviside derivative
        # confines the gradient to interface cells, giving a dynamic range
        # of 100x or more; a linear norm shows only the spikes.
        g_abs = np.abs(G)
        nonzero = g_abs[g_abs > 0]
        if nonzero.size:
            g_hi = max(float(np.percentile(g_abs, 99.5)), 1e-30)
            g_lo = max(float(np.percentile(nonzero, self.g_linthresh_pct)),
                       g_hi * 1e-3)
        else:
            g_hi, g_lo = 1e-12, 1e-15
        self.img3.set_norm(mcolors.SymLogNorm(linthresh=g_lo, vmin=-g_hi,
                                              vmax=g_hi, base=10))
        self.img3.set_data(self._mask(G).T)

        # Masked for drawing only — the volume readout below still counts
        # the wall cells, because they are design variables and the volume
        # constraint is applied over all of them.
        self.img4.set_data(self._mask(rho_bar).T)
        self.title4.set_text(f"Pipe design | vol={np.mean(rho_bar):.3f}")

        self.loss.append(J)
        self.lam.append(lam)
        self.alpha.append(alpha)
        self.beta.append(beta)
        self.fwd_iters.append(fwd_iters)
        self.adj_iters.append(adj_iters)

        steps = range(len(self.loss))
        self.line_loss.set_data(steps, self.loss)
        self.line_lam.set_data(steps, self.lam)
        self.line_alpha.set_data(steps, self.alpha)
        self.line_beta.set_data(steps, self.beta)
        self.line_fwd.set_data(steps, self.fwd_iters)
        self.line_adj.set_data(steps, self.adj_iters)

        if greyness is not None:
            self.greyness.append(max(greyness, 1e-12))
            self.line_grey.set_data(range(len(self.greyness)), self.greyness)

        if change_max is not None:
            self.change_max.append(max(change_max, 1e-12))
            self.change_mean.append(max(change_mean or 1e-12, 1e-12))
            self.move_limit.append(max(move_limit or 1e-12, 1e-12))
            span = range(len(self.change_max))
            self.line_dmax.set_data(span, self.change_max)
            self.line_dmean.set_data(span, self.change_mean)
            self.line_move.set_data(span, self.move_limit)

        if G_max is not None and lam:
            self.b_over_lam.append(G_max / max(abs(lam), 1e-300))
            self.line_ratio.set_data(range(len(self.b_over_lam)), self.b_over_lam)

        if volume is not None and self.volume_fraction is not None:
            self.volume_error.append(volume - self.volume_fraction)
            self.line_vol.set_data(range(len(self.volume_error)),
                                   self.volume_error)

        for ax in self.metric_axes:
            ax.relim()
            ax.autoscale_view()

        # The panels autoscale every frame, so the log subdivisions have to
        # be re-chosen every frame too: a series that spans a tenth of a
        # decade early on and four decades later needs different ticks at
        # each stage to stay readable.
        for ax in self.log_axes:
            _retick_log(ax)

        self.writer.grab_frame()

    # ------------------------------------------------------------------
    def save_last_frame(self, path):
        """Save the still through the same path as every other figure.

        PNG, not JPEG. JPEG subsamples chroma, and a 2pt saturated stroke on
        a near-black ground is almost entirely transition pixels, so the
        stroke's colour gets averaged toward the ground: measured here, cyan
        landed 23 levels off at the movie's 150 dpi. That is why the stills
        did not match the figures saved directly. PNG is exact, and at
        still_dpi the strokes are wider in pixels as well.
        """
        style.save(self.fig, path, dpi=self.still_dpi)
        print(f"Final frame saved to: {path}")

    def save_arrays(self, path, **arrays):
        np.savez_compressed(path, **arrays)
        print(f"Arrays saved to: {path}")

    def close(self, plot_path):
        self.save_last_frame(plot_path)
        self.writer.finish()
        plt.close(self.fig)