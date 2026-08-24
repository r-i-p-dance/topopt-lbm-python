import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from fixtures.forward import PressureBrinkman
from fixtures.adjoint import PressureAdjoint
from lbm.src.plot.animation import Recorder
from topopt.src.core.objective import dissipation_objective
from topopt.src.core.sensitivity import assemble_sensitivity

def gradient_error_field(ny=16, aspect=3, alpha_max=100.0, q=0.1,
                         beta=None, tau=0.933, Re=1.0,
                         tol_solver=1e-10, eps=1e-4, rho0=0.5,
                         out_path="C:\\CONSANARCHY\\Warwick\\URSS\\topopt-lbm-python\\results\\tests\\gradient_error_report.txt",
                         make_plot=True,
                         cmap_field='coolwarm', cmap_diff='RdBu_r'):
    """Full-field adjoint-vs-FD gradient diagnostic.

    Writes a text report with both fields, the ratio, and a structured summary.
    Set make_plot=True to also save the figure.
    """
    nx = aspect * ny
    rho_e = rho0 * np.ones((nx, ny))
    kwargs = dict(ny=ny, aspect=aspect, tau_lbm=tau, Re=Re,
                  alpha_max=alpha_max, q=q, beta=beta, periodic_x=True)

    # --- adjoint sensitivity: one forward + one adjoint solve ---
    fwd = PressureBrinkman(rho_e=rho_e.copy(), **kwargs)
    fwd.converge(tol=tol_solver)
    adj = PressureAdjoint(fwd)
    adj.converge(tol=tol_solver)
    G_adj = assemble_sensitivity(fwd, adj, alpha_max, q, beta=beta)

    # --- full finite-difference field (skip solid wall rows j=0, j=ny-1) ---
    G_fd = np.full((nx, ny), np.nan)
    for i in range(nx):
        for j in range(1, ny - 1):
            rp = rho_e.copy(); rp[i, j] += eps
            fp = PressureBrinkman(rho_e=rp, **kwargs); fp.converge(tol=tol_solver)
            Jp = dissipation_objective(fp.rho_bar, fp.ux, fp.uy, alpha_max, q)
            rm = rho_e.copy(); rm[i, j] -= eps
            fm = PressureBrinkman(rho_e=rm, **kwargs); fm.converge(tol=tol_solver)
            Jm = dissipation_objective(fm.rho_bar, fm.ux, fm.uy, alpha_max, q)
            G_fd[i, j] = (Jp - Jm) / (2 * eps)
        print(f"\rcolumn {i+1}/{nx} done", end="", flush=True)
    print("\nFD field complete")

    ratio = G_adj / G_fd

    # --- write the report ---
    with open(out_path, "w") as fh:
        def w(line=""): fh.write(line + "\n")

        w("=" * 70)
        w("GRADIENT VERIFICATION REPORT — adjoint sensitivity vs finite difference")
        w("=" * 70)
        w(f"grid: nx={nx}, ny={ny} (aspect={aspect})")
        w(f"params: alpha_max={alpha_max}, q={q}, beta={beta}, tau={tau}, Re={Re}")
        w(f"solver tol={tol_solver}, FD eps={eps}, uniform design rho_e={rho0}")
        w(f"forward converged in {fwd.it} steps; adjoint in {adj.it} steps")
        w("wall rows j=0 and j=ny-1 are solid (FD skipped, shown as nan)")
        w("")

        # helper to print a (nx, ny) field as an aligned grid: rows = j (wall at
        # top/bottom), cols = x. Transposed so it reads like the physical channel.
        def dump_field(name, F, fmt="{:+.3e}"):
            w(f"--- {name} ---  [rows: j=0..{ny-1} bottom->top, cols: x=0..{nx-1}]")
            for j in range(ny):
                row = "  ".join(
                    ("  nan   " if np.isnan(F[i, j]) else fmt.format(F[i, j]))
                    for i in range(nx)
                )
                w(f"j={j:2d} | {row}")
            w("")

        dump_field("ADJOINT G_adj", G_adj)
        dump_field("FINITE DIFF G_fd", G_fd)
        dump_field("RATIO G_adj/G_fd (target 1.0)", ratio, fmt="{:+.3f}")

        # --- structured summary: where is the error? ---
        w("=" * 70)
        w("SUMMARY")
        w("=" * 70)

        valid = ~np.isnan(G_fd)
        relerr = np.abs(G_adj - G_fd) / (np.abs(G_fd) + 1e-14)

        # interior x (away from both inlet/outlet), interior y (away from walls)
        x_int = slice(4, nx - 4)
        y_int = slice(3, ny - 3)
        interior = np.zeros((nx, ny), bool); interior[x_int, y_int] = True
        interior &= valid

        near_wall = valid.copy()
        near_wall[:, 3:ny-3] = False           # keep only j in {1,2, ny-3..ny-2}
        x_bound = valid.copy()
        x_bound[4:nx-4, :] = False             # keep only x in {0..3, nx-4..nx-1}

        def stat(name, mask):
            if mask.sum() == 0:
                w(f"{name:22s}: (no cells)"); return
            r = ratio[mask]; e = relerr[mask]
            w(f"{name:22s}: cells={mask.sum():4d}  "
              f"ratio[min/med/max]={np.nanmin(r):+.3f}/{np.nanmedian(r):+.3f}/{np.nanmax(r):+.3f}  "
              f"relerr[med/max]={np.nanmedian(e):.3f}/{np.nanmax(e):.3f}")

        stat("ALL fluid cells", valid)
        stat("INTERIOR (x&y away)", interior)
        stat("NEAR WALL (j<=2 or >=ny-3)", near_wall)
        stat("NEAR X-BOUNDARY", x_bound)
        w("")

        # per-row median ratio (shows y-dependence at a glance)
        w("per-row median ratio (y-dependence):")
        for j in range(ny):
            m = valid[:, j]
            if m.sum():
                w(f"  j={j:2d}: median ratio = {np.nanmedian(ratio[m, j]):+.3f}  "
                  f"(over {m.sum()} cells)")
        w("")

        # per-column median ratio, interior y only (shows x-dependence)
        w("per-column median ratio, interior y=3..ny-4 (x-dependence):")
        for i in range(nx):
            col = ratio[i, 3:ny-3]
            col = col[~np.isnan(col)]
            if col.size:
                w(f"  x={i:2d}: median ratio = {np.median(col):+.3f}")

    print(f"\nReport written to {out_path}")

    if make_plot:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        fig, ax = plt.subplots(4, 1, figsize=(12, 12))
        vmax = np.nanpercentile(np.abs(G_fd), 98)
        for a, (F, t, vm, vc, vx) in zip(ax, [
            (G_adj, "G_adj", -vmax, 0, vmax),
            (G_fd, "G_fd", -vmax, 0, vmax),
            (ratio, "ratio", -1, 1, 3),
            (G_adj - G_fd, "diff", -vmax, 0, vmax)]):
            im = a.imshow(F.T, origin="lower", cmap=cmap_diff if t == "diff" or t == "ratio" else cmap_field,
                          norm=mcolors.TwoSlopeNorm(vmin=vm, vcenter=vc, vmax=vx),
                          aspect="auto")
            a.set_title(t); plt.colorbar(im, ax=a)
        plt.tight_layout()
        plt.savefig("C:\\CONSANARCHY\\Warwick\\URSS\\topopt-lbm-python\\results\\tests\\gradient_error_field.png", 
                    dpi=130)

    return G_adj, G_fd


if __name__ == "__main__":
    gradient_error_field()