import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FFMpegWriter
from matplotlib.ticker import (ScalarFormatter, MaxNLocator,
                               LogLocator, NullFormatter)
from lbm.src.plot import style

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Neue Haas Grotesk Display Pro",
                        "Helvetica Neue", "Helvetica", "Arial"],
    "font.weight": "medium",
    "axes.titleweight": "medium",
})


class OptimizationRecorder:
    """Records the optimization loop as an MP4.

    Layout:
      Rows 0-1: four field panels as a 2x2 block (each spans 2 columns)
        [0,0:2] Forward velocity     [0,2:4] Adjoint momentum
        [1,0:2] Sensitivity          [1,2:4] Pipe design
      Row 2: four metric plots (dissipation, OC multiplier, alpha & beta,
             solver iterations)

    One optimization iteration = one frame. Grid size and output path are
    supplied by the driver via setup(), so the caller repeats neither.

    MP4 rather than GIF: a GIF holds 256 colours per frame, and with inferno,
    coolwarm and gray in one figure the adaptive palette leaves too few true
    greys, so intermediate values on the design panel get dithered into
    coloured speckle. H.264 is full 24-bit colour, produces much smaller
    files, and embeds directly in slides.

    Requires ffmpeg on PATH.
    """

    def __init__(self, fps=15, dpi=150, gamma=0.6, vmax_fwd=None,
                 bitrate=4000, g_linthresh_pct=50):
        self.fps = fps
        self.dpi = dpi
        self.gamma = gamma
        self.vmax_fwd = vmax_fwd
        self.bitrate = bitrate
        self.path = None
        self.nx = self.ny = None
        self.loss, self.lam = [], []
        self.alpha, self.beta = [], []
        self.fwd_iters, self.adj_iters = [], []
        self.g_linthresh_pct = g_linthresh_pct

    # ------------------------------------------------------------------
    def setup(self, nx, ny, path):
        """Build the figure once the grid size and output path are known."""
        self.nx, self.ny = nx, ny
        self.path = path
        print(f"Recording to: {self.path}")

        self.fig = plt.figure(figsize=(10, 13.4))
        gs = GridSpec(3, 4, figure=self.fig,
                      height_ratios=[1, 1, 0.5], hspace=0.13, wspace=0.35)

        self.ax1 = self.fig.add_subplot(gs[0, 0:2])
        self.ax2 = self.fig.add_subplot(gs[0, 2:4])
        self.ax3 = self.fig.add_subplot(gs[1, 0:2])
        self.ax4 = self.fig.add_subplot(gs[1, 2:4])

        self.ax5 = self.fig.add_subplot(gs[2, 0])
        self.ax6 = self.fig.add_subplot(gs[2, 1])
        self.ax7 = self.fig.add_subplot(gs[2, 2])
        self.ax8 = self.fig.add_subplot(gs[2, 3])

        z = np.zeros((nx, ny))
        vf = self.vmax_fwd if self.vmax_fwd is not None else 1.0

        self.img1 = self.ax1.imshow(          # forward velocity: physical
            z.T, cmap=style.SEQUENTIAL_FLOW, origin="lower", aspect="auto",
            norm=mcolors.PowerNorm(gamma=self.gamma, vmin=0.0, vmax=vf))
        self.img2 = self.ax2.imshow(          # adjoint momentum: dual
            z.T, cmap=style.SEQUENTIAL_DUAL, origin="lower", aspect="auto",
            norm=mcolors.PowerNorm(gamma=self.gamma, vmin=0.0, vmax=1.0))
        self.img3 = self.ax3.imshow(
            z.T, cmap=style.DIVERGING, origin="lower", aspect="auto",
            norm=mcolors.TwoSlopeNorm(vmin=-1e-4, vcenter=0, vmax=1e-4))
        self.img4 = self.ax4.imshow(
            z.T, cmap=style.DESIGN, origin="lower", aspect="auto",
            vmin=0, vmax=1)

        c, ls = style.S_LOSS
        self.img5, = self.ax5.semilogy([], [], color=c, ls=ls, lw=2)
        c, ls = style.S_LAM
        self.img6, = self.ax6.plot([], [], color=c, ls=ls, lw=2)
        c, ls = style.S_ALPHA
        self.img7a, = self.ax7.plot([], [], color=c, ls=ls, lw=2, label="alpha")
        c, ls = style.S_BETA
        self.img7b, = self.ax7.plot([], [], color=c, ls=ls, lw=2, label="beta")
        self.ax7.legend(loc="upper left", fontsize=6, frameon=False,
                        labelcolor=style.MUTED)
        c, ls = style.S_FWD
        self.img8a, = self.ax8.plot([], [], color=c, ls=ls, lw=2, label="fwd")
        c, ls = style.S_ADJ
        self.img8b, = self.ax8.plot([], [], color=c, ls=ls, lw=2, label="adj")
        self.ax8.legend(loc="upper left", fontsize=6, frameon=False,
                        labelcolor=style.MUTED)

        self.ax1.set_title("Forward velocity", fontsize=11)
        self.ax2.set_title("Adjoint momentum", fontsize=11)
        self.ax3.set_title("Sensitivity", fontsize=11)
        self.title4 = self.ax4.set_title("Pipe design", fontsize=11)

        self.ax5.set_title("Energy dissipation", fontsize=9)
        self.ax6.set_title("OC multiplier", fontsize=9)
        self.ax7.set_title("Alpha & Beta", fontsize=9)
        self.ax8.set_title("Solver iterations", fontsize=9)

        for ax in (self.ax1, self.ax2, self.ax3, self.ax4):
            ax.set_box_aspect(1)
            ax.set_xticks([])
            ax.set_yticks([])

        for ax in (self.ax5, self.ax6, self.ax7, self.ax8):
            ax.set_box_aspect(1)
            ax.tick_params(axis="both", labelsize=6, pad=1)
            ax.grid(True, which="major", alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))

        for ax in (self.ax6, self.ax7, self.ax8):
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((-2, 3))
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.get_offset_text().set_fontsize(6)

        self.ax5.yaxis.set_major_locator(LogLocator(numticks=4))
        self.ax5.yaxis.set_minor_locator(LogLocator(subs="all", numticks=12))
        self.ax5.yaxis.set_minor_formatter(NullFormatter())

        style.apply_figure_style(
            self.fig,
            [self.ax1, self.ax2, self.ax3, self.ax4,
             self.ax5, self.ax6, self.ax7, self.ax8],
            image_axes=(self.ax1, self.ax2, self.ax3, self.ax4))
        # yuv420p for QuickTime/PowerPoint compatibility; the scale filter
        # forces even pixel dimensions, which H.264 requires and which an
        # odd figsize*dpi would otherwise violate.
        self.writer = FFMpegWriter(
            fps=self.fps, codec="libx264", bitrate=self.bitrate,
            extra_args=["-pix_fmt", "yuv420p",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"])
        self.writer.setup(self.fig, self.path, dpi=self.dpi)

    # ------------------------------------------------------------------
    def capture(self, loop, fwd_vel, adj_vel, G, rho_bar,
                J, lam, alpha, beta, fwd_iters, adj_iters):
        """Record one optimization iteration as a frame."""
        # forward: fix the scale from the first frame so it doesn't flicker
        if self.vmax_fwd is None:
            self.vmax_fwd = max(float(np.percentile(fwd_vel, 99.5)), 1e-10)
            self.img1.set_norm(mcolors.PowerNorm(
                gamma=self.gamma, vmin=0.0, vmax=self.vmax_fwd))
        self.img1.set_data(fwd_vel.T)

        # adjoint: rescale each frame, its magnitude drifts with continuation
        vmax_adj = max(float(np.percentile(adj_vel, 99)), 1e-10)
        self.img2.set_norm(mcolors.PowerNorm(
            gamma=self.gamma, vmin=0.0, vmax=vmax_adj))
        self.img2.set_data(adj_vel.T)

        g_abs = np.abs(G)
        nz = g_abs[g_abs > 0]
        if nz.size:
            g_hi = max(float(np.percentile(g_abs, 99.5)), 1e-30)
            g_lo = max(float(np.percentile(nz, self.g_linthresh_pct)),
                       g_hi * 1e-3)
        else:
            g_hi, g_lo = 1e-12, 1e-15
        self.img3.set_norm(mcolors.SymLogNorm(linthresh=g_lo, vmin=-g_hi,
                                              vmax=g_hi, base=10))

        self.img3.set_data(G.T)

        self.img4.set_data(rho_bar.T)
        self.title4.set_text(f"Pipe design | vol={np.mean(rho_bar):.3f}")

        self.loss.append(J)
        self.lam.append(lam)
        self.alpha.append(alpha)
        self.beta.append(beta)
        self.fwd_iters.append(fwd_iters)
        self.adj_iters.append(adj_iters)

        n = range(len(self.loss))
        self.img5.set_data(n, self.loss)
        self.img6.set_data(n, self.lam)
        self.img7a.set_data(n, self.alpha)
        self.img7b.set_data(n, self.beta)
        self.img8a.set_data(n, self.fwd_iters)
        self.img8b.set_data(n, self.adj_iters)

        for ax in (self.ax5, self.ax6, self.ax7, self.ax8):
            ax.relim()
            ax.autoscale_view()

        self.writer.grab_frame()

    # ------------------------------------------------------------------
    def save_last_frame(self, path):
        self.fig.savefig(path, dpi=self.dpi, format="jpg",
                         bbox_inches="tight", facecolor=style.GROUND)
        print(f"Final frame saved to: {path}")

    def save_arrays(self, path, **arrays):
        np.savez_compressed(path, **arrays)
        print(f"Arrays saved to: {path}")

    def close(self, plot_path):
        self.save_last_frame(plot_path)
        self.writer.finish()
        plt.close(self.fig)