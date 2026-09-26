import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FFMpegWriter
from matplotlib.ticker import (ScalarFormatter, MaxNLocator, AutoMinorLocator,
                               FixedLocator, FuncFormatter, LogLocator,
                               NullFormatter)
from lbm.src.plot import style

# Type comes from style.py's rcParams — absolute points, identical to every
# other poster figure. Nothing here passes a fontsize.

# The eight metric panels, in grid order.
METRIC_TITLES = ("energy dissipation (log)", "design change (log)",
                 "change mean / max", "alpha & beta",
                 "greyness (log)", "OC multiplier",
                 "B / lambda (log)", "solver iterations")

# Major-tick mantissas, coarse to fine. Decades and thirds-of-a-decade
# ONLY. Both are near-equal ratios — 10, and 3.16 — so consecutive ticks
# land at near-equal distances on a log axis. Sets like (1, 2, 5) or
# (1, 2, 3, 5, 7) are round numbers but sit at visibly unequal spacings,
# and a linear set such as 20/30/40/50 is worse still: evenly-spaced
# numbers at unevenly-spaced positions read as a broken axis.
_LADDERS = ((1.0,), (1.0, 3.0))
# Four, not five: the labels are absolute-sized and a metric panel is small.
# Rotated, "1x10^-3" is 37.4 pt tall; five ticks on a 5-module panel leaves
# 33.0 pt between them and they collide, four leaves 44.0 pt and they do not.
_MAX_TICKS = 4


def _candidate_ticks(subs, lo, hi):
    """Every subs*10^k in a range comfortably wrapping [lo, hi]."""
    lo_e = int(np.floor(np.log10(lo))) - 1
    hi_e = int(np.ceil(np.log10(hi))) + 1
    return sorted(s * 10.0**e
                  for e in range(lo_e, hi_e + 1) for s in subs)


def _axis_formatter(ticks):
    """One notation for the whole axis, plus the factor it was divided by.

    Returns (formatter, factor_label).

    The factor comes from style.factor_exponent, which is also what the
    convergence plot uses, so a log axis reads the same way in both repos.

    Deciding per tick is what produced axes reading 0.01, 0.03 and then
    3 x 10^-3: each label was individually defensible and the column as a
    whole was unreadable. The choice has to be made once, for the set.
    """
    exponent = style.factor_exponent(ticks)
    scale = 10.0**exponent
    if all(style.PLAIN_LO <= t / scale < style.PLAIN_HI for t in ticks):

        def fmt(value, _pos=None):
            return f"{value / scale:g}" if value > 0 else ""

        return fmt, style.factor_text(exponent)

    # Span too wide to factor into short numbers — label each tick in full.
    def fmt(value, _pos=None):
        if value <= 0:
            return ""
        e = int(np.floor(np.log10(value) + 1e-9))
        mantissa = value / 10.0**e
        if mantissa >= 9.995:                  # 1e-3 must not print as 10e-4
            mantissa, e = mantissa / 10.0, e + 1
        return rf"${mantissa:g}\times10^{{{e}}}$"

    return fmt, ""


def _retick_log(ax):
    """Even, round, capped major ticks on a log axis at any span.

    The limits are snapped OUTWARD onto the tick set, rather than the ticks
    being fitted inside whatever the data happened to reach. That is what
    buys a tick at each end of the axis and keeps the spacing uniform;
    letting the data set the limits is what leaves a panel holding a single
    label.

    Returns the factor the labels were divided by, as a label for the
    panel's corner, or "" when they were left as they are.

    Runs every frame, because the panels autoscale every frame.
    """
    lo, hi = ax.get_ylim()
    if not (lo > 0 and hi > lo):
        return ""

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
    formatter, factor = _axis_formatter(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(formatter))
    # Minor gridlines carry the decade structure. Without them a series
    # spanning well under a decade is hard to tell from a linear one —
    # because over that little range it very nearly is one.
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs="all", numticks=100))
    ax.yaxis.set_minor_formatter(NullFormatter())
    return factor


class OptimizationRecorder:
    """Records the optimization loop as an MP4.

    Layout:
      Rows 0-1: four field panels as a 2x2 block
        Forward velocity     Adjoint momentum
        Sensitivity          Pipe design
      Rows 2-3: eight metric plots, four per row
        dissipation | design change | change mean/max | alpha & beta
        greyness    | OC multiplier | B / lambda      | solver iterations

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
                 bitrate=4000, g_linthresh_pct=50, still_dpi=300,
                 modules=11):
        self.fps = fps
        self.dpi = dpi
        # Field panel side, in poster modules. The whole layout follows from
        # it — see _build_figure. Odd, and at least 11.
        self.modules = modules
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
        self.change_ratio = []

    # ------------------------------------------------------------------
    def setup(self, nx, ny, path):
        """Build the figure and open the movie writer."""
        self.nx, self.ny = nx, ny
        self.path = path
        print(f"Recording to: {self.path}")
        self._build_figure(nx, ny)
        # yuv420p for QuickTime/PowerPoint compatibility; the scale filter
        # forces even pixel dimensions, which H.264 requires and which an
        # odd figsize*dpi would otherwise violate.
        self.writer = FFMpegWriter(
            fps=self.fps, codec="libx264", bitrate=self.bitrate,
            extra_args=["-pix_fmt", "yuv420p",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"])
        self.writer.setup(self.fig, self.path, dpi=self.dpi)

    def setup_static(self, nx, ny):
        """The same figure with no movie behind it — needs no ffmpeg.

        This is what lets a saved run be replotted: the replot draws through
        exactly the same code as the movie did, so the two cannot drift.
        """
        self.nx, self.ny = nx, ny
        self.path = None
        self.writer = None
        self._build_figure(nx, ny)

    def _build_figure(self, nx, ny):
        """Everything except the writer: axes, artists, ticks, styling.

        The whole layout follows from one number: F, the field panel's side in
        poster modules. Both panels are square, and a metric panel is sized so
        that two of them plus the gap between span one field exactly:

            m = (F - 1) / 2       so    m + 1 + m = F

        Gaps are one module, except below the field block and between the two
        metric rows, where they are three: that gap carries the upper row's x
        tick labels AND the lower row's title, which at 13 pt is 31.6 pt of
        content and does not fit in one module's 26.4 pt.

        Spacer rows and columns rather than hspace/wspace, because the gaps
        are not uniform and those two parameters are single figure-wide
        fractions. With them at zero every panel edge lands on a whole module.

        F must be ODD, or m falls on a half module, and at least 11: the
        longest title is 4.53 modules wide, so a shorter panel clips it.
        """
        F = self.modules
        m = (F - 1) / 2
        cols = [m, 1, m, 1, m, 1, m]            # sums to 2F + 1
        rows = [F, 1, F, 3, m, 3, m]            # sums to 3F + 6
        total_w = sum(cols) + 2                 # + one module of margin a side
        total_h = sum(rows) + 2

        unit = style.BASELINE_MM / style.MM_PER_IN
        self.fig = plt.figure(figsize=(total_w * unit, total_h * unit))
        gs = GridSpec(len(rows), len(cols), figure=self.fig,
                      width_ratios=cols, height_ratios=rows,
                      hspace=0, wspace=0,
                      left=1 / total_w, right=1 - 1 / total_w,
                      top=1 - 1 / total_h, bottom=1 / total_h)

        self.ax1 = self.fig.add_subplot(gs[0, 0:3])
        self.ax2 = self.fig.add_subplot(gs[0, 4:7])
        self.ax3 = self.fig.add_subplot(gs[2, 0:3])
        self.ax4 = self.fig.add_subplot(gs[2, 4:7])

        self.ax5 = self.fig.add_subplot(gs[4, 0])
        self.ax6 = self.fig.add_subplot(gs[4, 2])
        self.ax7 = self.fig.add_subplot(gs[4, 4])
        self.ax8 = self.fig.add_subplot(gs[4, 6])
        self.ax9 = self.fig.add_subplot(gs[6, 0])
        self.ax10 = self.fig.add_subplot(gs[6, 2])
        self.ax11 = self.fig.add_subplot(gs[6, 4])
        self.ax12 = self.fig.add_subplot(gs[6, 6])

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
        self.line_loss, = self.ax5.semilogy([], [], color=colour, ls=linestyle)

        # Design change against the move limit. When the solid line meets
        # the dotted one, every moving cell is clipped and the optimizer is
        # saturated rather than converging. The gap between max and mean
        # shows how localised the motion is.
        colour, linestyle = style.S_METRIC
        self.line_dmax, = self.ax6.semilogy([], [], color=colour, ls=linestyle,
                                            label="max")
        colour, linestyle = style.S_ALT
        self.line_dmean, = self.ax6.semilogy([], [], color=colour, ls=linestyle,
                                             label="mean")
        colour, linestyle = style.S_REF
        self.line_move, = self.ax6.semilogy([], [], color=colour, ls=linestyle,
                                            label="limit")

        # How localised the design motion is: mean change over max change,
        # so 1 means every moving cell moves by the same amount (the whole
        # design is clipped at the move limit) and a small value means only
        # a handful of interface cells are still moving. Bounded by 1, so it
        # is read linearly rather than on a log axis.
        colour, linestyle = style.S_METRIC
        self.line_change_ratio, = self.ax7.plot([], [], color=colour,
                                                ls=linestyle)

        colour, linestyle = style.S_METRIC
        self.line_alpha, = self.ax8.plot([], [], color=colour, ls=linestyle,
                                         label="alpha")
        colour, linestyle = style.S_ALT
        self.line_beta, = self.ax8.plot([], [], color=colour, ls=linestyle,
                                        label="beta")

        # Greyness: 0 = fully binary. The manufacturability number.
        colour, linestyle = style.S_METRIC
        self.line_grey, = self.ax9.semilogy([], [], color=colour, ls=linestyle)

        # OC multiplier. Placed next to B / lambda, which is built from it,
        # so the two read as one pair.
        colour, linestyle = style.S_METRIC
        self.line_lam, = self.ax10.plot([], [], color=colour, ls=linestyle)

        # B / lambda: the leading indicator. A plateau is healthy; unbounded
        # growth means lambda has lost the gradient and every cell is about
        # to saturate.
        colour, linestyle = style.S_METRIC
        self.line_ratio, = self.ax11.semilogy([], [], color=colour, ls=linestyle)

        colour, linestyle = style.S_METRIC
        self.line_fwd, = self.ax12.plot([], [], color=colour, ls=linestyle,
                                        label="fwd")
        colour, linestyle = style.S_ALT
        self.line_adj, = self.ax12.plot([], [], color=colour, ls=linestyle,
                                        label="adj")

        # Fixed corners, not "best": "best" is recomputed on every draw, so
        # in a movie the legend would hop between corners frame to frame.
        # Each corner is the one its series leaves empty: solver iteration
        # counts fall as the design settles, so those curves end up bottom
        # right and the legend sits opposite them.
        for ax, corner in ((self.ax6, "lower left"),
                           # matplotlib spells it "center"
                           (self.ax8, "center right"),
                           (self.ax12, "upper right")):
            ax.legend(loc=corner, frameon=False, labelcolor=style.TEXT,
                      handlelength=1.6, borderpad=0.2, labelspacing=0.25)

        self.ax1.set_title("forward velocity")
        self.ax2.set_title("adjoint momentum")
        self.ax3.set_title("sensitivity")
        self.title4 = self.ax4.set_title("pipe design")

        # The four log panels say so in the title. The minor gridlines carry
        # the decade structure, but a reader should not have to infer the
        # scale from the spacing of the rules.
        #
        # Each panel also carries a label holding the power of ten its y axis
        # was divided by — see _axis_formatter. Created empty and re-texted
        # every frame, because the factor follows the data.
        self.chips = {}
        for ax, name in zip(self.metric_axes, METRIC_TITLES):
            ax.set_title(name)
            self.chips[ax] = style.factor_label(ax, "")

        # No box_aspect: the grid slots are square by construction, and
        # box_aspect would renegotiate the panel inside its slot and pull it
        # off the module boundaries.
        for ax in self.field_axes:
            ax.set_xticks([])
            ax.set_yticks([])

        self.log_axes = (self.ax5, self.ax6, self.ax9, self.ax11)

        for ax in self.metric_axes:
            # No x axis label. What the axis is belongs in the title; the
            # gutter is one module and the tick labels need all of it.
            ax.tick_params(axis="both", which="minor", length=1.5)
            # Three intervals, not five: a label is absolute-sized and a
            # metric panel is 5 modules, so five of them would touch.
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            # Keeps the newest point off the frame as the series grows.
            ax.margins(x=0.04, y=0.10)

        # Linear axes: pull the exponent into a compact offset label so
        # ticks read "-2.2" rather than "-0.00002245".
        #
        # Matplotlib parks that offset ON the axes corner, crossing both
        # spines. It goes into the panel's chip instead. The native artist
        # cannot simply be moved: YAxis._update_offset_text_position rewrites
        # its position in DISPLAY coordinates on every draw, so any transform
        # set here is silently undone — and while it is merely hidden it
        # still holds a string, which the title auto-placement reads. Hence
        # style.py pinning axes.titley.
        self.linear_axes = (self.ax7, self.ax8, self.ax10, self.ax12)
        for ax in self.linear_axes:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((-2, 3))
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.get_offset_text().set_visible(False)

        for ax in self.log_axes:
            self.chips[ax].set_text(_retick_log(ax))

        style.apply_figure_style(
            self.fig, list(self.field_axes) + list(self.metric_axes),
            image_axes=self.field_axes)

        # Applied after apply_figure_style, which turns on a single major
        # grid. Major rules tie a point to its tick; the minor rules are
        # what make a log axis legible AS a log axis.
        for ax in self.metric_axes:
            ax.grid(True, which="major", color=style.RULE,
                    alpha=style.GRID_ALPHA, lw=style.GRID_PT, ls="-")
            ax.grid(True, which="minor", color=style.RULE,
                    alpha=style.GRID_MINOR_ALPHA, lw=style.GRID_PT / 2, ls="-")
            ax.set_axisbelow(True)

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
    def draw_fields(self, fwd_vel, adj_vel, G, rho_bar):
        """Repaint the four field panels. Also the replot's entry point."""
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
        self.title4.set_text(f"pipe design | vol={np.mean(rho_bar):.3f}")

    def draw_series(self):
        """Redraw the eight metric panels from the series lists as they are.

        Each line takes its own range(len(...)) rather than a shared one:
        b_over_lam is appended only on iterations where lam is non-zero, so
        the lists are genuinely different lengths and a shared x-axis would
        silently shift that curve.
        """
        self.line_loss.set_data(range(len(self.loss)), self.loss)
        self.line_lam.set_data(range(len(self.lam)), self.lam)
        self.line_alpha.set_data(range(len(self.alpha)), self.alpha)
        self.line_beta.set_data(range(len(self.beta)), self.beta)
        self.line_fwd.set_data(range(len(self.fwd_iters)), self.fwd_iters)
        self.line_adj.set_data(range(len(self.adj_iters)), self.adj_iters)
        self.line_change_ratio.set_data(range(len(self.change_ratio)),
                                        self.change_ratio)
        self.line_grey.set_data(range(len(self.greyness)), self.greyness)
        self.line_dmax.set_data(range(len(self.change_max)), self.change_max)
        self.line_dmean.set_data(range(len(self.change_mean)), self.change_mean)
        self.line_move.set_data(range(len(self.move_limit)), self.move_limit)
        self.line_ratio.set_data(range(len(self.b_over_lam)), self.b_over_lam)

        for ax in self.metric_axes:
            ax.relim()
            ax.autoscale_view()

        # The panels autoscale every frame, so the log subdivisions have to
        # be re-chosen every frame too: a series that spans a tenth of a
        # decade early on and four decades later needs different ticks at
        # each stage to stay readable.
        for ax in self.log_axes:
            self.chips[ax].set_text(_retick_log(ax))

        # Here rather than in _build_figure because a locator builds fresh
        # Tick objects whenever the count changes, and these panels re-tick
        # every frame.
        for ax in self.metric_axes:
            style.rotate_y_labels(ax)

        # get_yticklabels above forced the ticks to regenerate, so the
        # formatters have just recomputed their offsets.
        for ax in self.linear_axes:
            self.chips[ax].set_text(ax.yaxis.get_major_formatter().get_offset())

    # ------------------------------------------------------------------
    def capture(self, loop, fwd_vel, adj_vel, G, rho_bar,
                J, lam, alpha, beta, fwd_iters, adj_iters,
                greyness=None, change_max=None, change_mean=None,
                move_limit=None, G_max=None):
        """Record one optimization iteration as a frame.

        The second metric row is optional: pass the extra keywords and it
        is populated, omit them and those panels stay empty. This keeps the
        recorder usable from any driver.
        """
        self.draw_fields(fwd_vel, adj_vel, G, rho_bar)

        self.loss.append(J)
        self.lam.append(lam)
        self.alpha.append(alpha)
        self.beta.append(beta)
        self.fwd_iters.append(fwd_iters)
        self.adj_iters.append(adj_iters)

        if greyness is not None:
            self.greyness.append(max(greyness, 1e-12))

        if change_max is not None:
            self.change_max.append(max(change_max, 1e-12))
            self.change_mean.append(max(change_mean or 1e-12, 1e-12))
            self.move_limit.append(max(move_limit or 1e-12, 1e-12))
            self.change_ratio.append(change_mean / change_max)

        if G_max is not None and lam:
            self.b_over_lam.append(G_max / max(abs(lam), 1e-300))

        self.draw_series()

        if self.writer is not None:
            self.writer.grab_frame()

    # ------------------------------------------------------------------
    def save_last_frame(self, path):
        """Save the still through the same path as every other figure.

        PNG, not JPEG. JPEG subsamples chroma, and a 2pt saturated stroke on
        a near-black ground is almost entirely transition pixels, so the
        stroke's colour gets averaged toward the ground: measured here, cyan
        landed 23 levels off at the movie's 150 dpi. That is why the stills
        did not match the figures saved directly. PNG is exact, and at
        still_dpi the strokes are wider in pixels as well. As a PDF the type
        is vector and only the four field panels are raster.

        save_exact, not save: the tight crop would make the page size depend
        on how long the tick labels happened to be, and the figure would stop
        being a whole number of modules.
        """
        style.save_exact(self.fig, path, dpi=self.still_dpi)
        print(f"Final frame saved to: {path}")

    def close(self, plot_path):
        self.save_last_frame(plot_path)
        if self.writer is not None:
            self.writer.finish()
        plt.close(self.fig)