"""Periodic dump of forward/adjoint/sensitivity fields to text, for
diagnosing spatial patterns that are hard to read off an animation."""

import numpy as np
from pathlib import Path

from lbm.src.utils.paths import get_safe_filepath


class FieldDump:
    """Writes coarse field snapshots to a text file every `every` iterations.

    Fields are downsampled to at most `max_cols` columns so a 64x64 grid
    stays readable; statistics are computed on the FULL field, not the
    downsampled view.
    """

    def __init__(self, every=25, max_cols=32):
        self.every = every
        self.max_cols = max_cols
        self.path = None
        self.fh = None

    def setup(self, path):
        """Called by the driver with the resolved run path."""
        self.path = Path(path)
        self.fh = open(self.path, "w", encoding="utf-8")
        print(f"Field dump: {self.path}")

    def _w(self, line=""):
        self.fh.write(line + "\n")

    def _grid(self, name, F, obstacle=None, fmt="{:+.2e}"):
        nx, ny = F.shape
        step = max(1, nx // self.max_cols)
        self._w(f"--- {name} ---  [rows j (bottom->top), cols x, "
                f"every {step} cell(s)]")
        for j in range(ny - 1, -1, -step):
            cells = []
            for i in range(0, nx, step):
                if obstacle is not None and obstacle[i, j]:
                    cells.append("  solid ")
                else:
                    cells.append(fmt.format(F[i, j]))
            self._w(f"j={j:3d} | " + " ".join(cells))
        self._w("")

    def _stats(self, name, F, mask=None):
        v = F[mask] if mask is not None else F.ravel()
        v = v[np.isfinite(v)]
        if v.size == 0:
            self._w(f"{name:16s}: (empty)")
            return
        a = np.abs(v)
        self._w(f"{name:16s}: min={v.min():+.3e} max={v.max():+.3e} "
                f"mean={v.mean():+.3e} std={v.std():.3e} | "
                f"|.| p50={np.percentile(a,50):.3e} p99={np.percentile(a,99):.3e} "
                f"p100={a.max():.3e}  dyn_range={a.max()/max(np.percentile(a,50),1e-300):.1e}")

    def maybe_capture(self, it, fwd, adj, G, rho_bar, alpha_max, beta):
        if self.fh is None or it % self.every != 0:
            return

        fluid = ~fwd.obstacle
        solid = fwd.obstacle
        fwd_vel = np.sqrt(fwd.ux**2 + fwd.uy**2)
        adj_vel = np.sqrt(adj.ux**2 + adj.uy**2)

        self._w("=" * 110)
        self._w(f"ITERATION {it}   alpha_max={alpha_max:.1f}  beta={beta:.2f}")
        self._w("=" * 110)

        # statistics split by region — this is what identifies whether the
        # bright cells sit in solid or fluid
        self._w("STATISTICS")
        self._stats("fwd_vel all", fwd_vel)
        self._stats("fwd_vel fluid", fwd_vel, fluid)
        self._stats("fwd_vel solid", fwd_vel, solid)
        self._stats("adj_mom all", adj_vel)
        self._stats("adj_mom fluid", adj_vel, fluid)
        self._stats("adj_mom solid", adj_vel, solid)
        self._stats("adj f all", adj.f)
        self._stats("sensitivity G", G)
        self._stats("source", adj.source)
        self._stats("omega_eff", fwd.omega_eff)
        self._stats("rho_bar", rho_bar)
        self._w(f"{'adj iters':16s}: {adj.it}   fwd iters: {fwd.it}")

        # locate the extreme adjoint cells: are they in solid or fluid?
        k = 8
        flat = np.argsort(adj_vel.ravel())[-k:][::-1]
        self._w("")
        self._w(f"top {k} adjoint cells:")
        for idx in flat:
            i, j = np.unravel_index(idx, adj_vel.shape)
            self._w(f"   ({i:3d},{j:3d})  adj_vel={adj_vel[i,j]:.4e}  "
                    f"fwd_vel={fwd_vel[i,j]:.4e}  rho_bar={rho_bar[i,j]:.3f}  "
                    f"{'SOLID' if fwd.obstacle[i,j] else 'fluid'}")
        self._w("")

        self._grid("rho_bar", rho_bar, fmt="{:5.2f}  ")
        self._grid("fwd_vel", fwd_vel, fwd.obstacle)
        self._grid("adj_mom", adj_vel, fwd.obstacle)
        self._grid("G", G, fwd.obstacle)
        self.fh.flush()

    def close(self):
        if self.fh is not None:
            self.fh.close()