import numpy as np
from topopt.src.core import objective
from topopt.src.core.objective import dissipation_objective
from topopt.src.core.projection import heaviside_projection
from topopt.src.core.sensitivity import assemble_sensitivity
from timeit import default_timer as timer
from topopt.src.core.projection import heaviside_projection
from pathlib import Path
from lbm.src.utils.paths import RunPaths
from collections import deque



class TopOptDriver:
    """Forward → adjoint → sensitivity → design update, until the design
    stops changing.

    The case is the single source of truth for nx, ny and volume_fraction;
    the driver wires those into the optimizer and recorder so they cannot
    be given inconsistently.

    Solvers are built once and warm-started: each iteration updates the
    design in place and re-converges from the previous solution. The fixed
    point does not depend on the starting guess, so this is exact — and
    much faster, since consecutive designs differ by at most `move`.
    """

    def __init__(self, case, sensitivity_filter, optimizer, continuation,
                 recorder=None, field_dump=None,
                 run_name="run", results_root="results"):
        self.case = case
        self.filter = sensitivity_filter
        self.optimizer = optimizer
        self.continuation = continuation
        self.recorder = recorder
        self.field_dump = field_dump

        self.paths = RunPaths(run_name, results_root)

        optimizer.volume_fraction = case.volume_fraction
        optimizer.contract_from = getattr(continuation, "final_iteration", None)
        self.convergence_window = getattr(optimizer, "convergence_window")

        if recorder is not None:
            recorder.setup(case.nx, case.ny, self.paths.animation,
                           volume_fraction=case.volume_fraction)
        if field_dump is not None:
            field_dump.setup(self.paths.fields)

    def run(self, max_iter=200, tol_J=5e-3, tol_grey=2e-2,
            solver_tol=1e-8, ewa_beta=0.85):
        """
        tol_change : threshold on MEAN |delta rho|. The mean, not the max:
                     max is an extreme-value statistic over N cells, so a
                     design that is optimal apart from a few interface cells
                     flipping still reports the full move limit. No run has
                     ever converged on a max-based criterion for this reason.
        tol_J      : threshold on the relative spread of J over `window`
                     iterations. Catches limit cycles, where the design
                     change is constant but J alternates between states.
        ewa_beta   : smoothing for the exponentially weighted average iteration
                     time. The first iterations are slower (cold start, Numba compilation),
                     so a plain mean would overestimate the remaining time.
        log_dir    : where the run log is written. The filename is taken from the
                     recorder's resolved GIF stem, so the log, the GIF and the .npz all
                     carry the same run number.
        """
        case = self.case

        fwd, adj = case.build_solvers(self.continuation.alpha_max(0), self.continuation.beta(0))

        # The recorder is built in __init__, before the solvers exist, so
        # the wall geometry can only be handed over here.
        if self.recorder is not None:
            self.recorder.set_obstacle(fwd.obstacle)

        # name the log after the GIF so artifacts stay matched
        log = _Tee(self.paths.log)

        log("=" * 100)
        log("TOPOLOGY OPTIMIZATION RUN")
        log("=" * 100)
        log(f"run              : {self.paths.stem}")
        log(f"case             : {type(case).__name__}")
        log(f"grid             : nx={case.nx}, ny={case.ny}")
        log(f"volume_fraction  : {case.volume_fraction}")

        if hasattr(fwd, "xi"):
            log(f"outlet asymmetry : xi={fwd.xi:.3f}  "
                f"outlet_asymmetry={fwd.outlet_asymmetry:.3e}  "
                f"(natural scale {fwd.natural_pressure_scale():.3e})")
            log(f"outlet densities : east={fwd.rho_east:.6f}  "
                f"south={fwd.rho_south:.6f}  less_pressure={fwd.less_pressure}")
            
        if hasattr(fwd, "split"):
            log(f"prescribed split : {fwd.split:.3f} south / "
                f"{1 - fwd.split:.3f} east")
            log(f"inlet flux Q_in  : {fwd.flux_in:.6e}")
            log(f"target flux Q_south   : {fwd.flux_south_target:.6e}")

        if hasattr(fwd, "split_3"):
            log(f"prescribed splits: {fwd.split_1:.3f} / "
                f"{fwd.split_2:.3f} / {fwd.split_3:.3f} "
                f"(outlet 3 = anchor, takes the remainder)")
            log(f"inlet flux Q_in  : {fwd.q_in:.6e}")

        log(f"Re               : {case.Re}")
        log(f"tau_lbm          : {case.tau_lbm}    nu={case.nu:.6f}")
        log(f"q                : {case.q}")
        log(f"filter radius    : {self.filter.radius}")
        log(f"optimizer        : {self.optimizer.describe()}")
        log(f"continuation     : {self.continuation.describe()}")
        log(f"driver           : max_iter={max_iter}, tol_J={tol_J}, "
            f"solver_tol={solver_tol}")
        log(f"animation        : {self.paths.animation if self.recorder else '(none)'}")
        log(f"field dump       : {self.paths.fields if self.field_dump else '(none)'}")
        log("=" * 100)
        log("")

        rho_e = case.rho_e.copy()
        
        t_start = timer()
        ewa = None
        it = 0
        converged = False
        J_window = deque(maxlen=self.convergence_window)
        greyness_window = deque(maxlen=self.convergence_window)

        for it in range(max_iter):
            t_iter = timer()

            alpha_max = self.continuation.alpha_max(it)
            beta = self.continuation.beta(it)

            fwd.update_design(rho_e, alpha_max=alpha_max, beta=beta)
            fwd.converge(tol=solver_tol)

            if hasattr(fwd, "outlet_fluxes"):
                fluxes = fwd.outlet_fluxes()
                total = sum(fluxes) + 1e-30
                shares = " ".join(f"{q/total:.3f}" for q in fluxes)
                flux_text = (f"split=[{shares}] "
                             f"mass balance={fwd.mass_balance():+.1e} "
                             f"hydraulic power={fwd.hydraulic_power():.3e}  ")
            else:
                flux_text = ""

            adj.update_source()
            adj.converge(tol=solver_tol)

            G_raw = assemble_sensitivity(fwd, adj, alpha_max, beta, case.q,)
            G = self.filter.apply(G_raw)
            G_max = float(np.max(np.abs(G)))

            rho_bar = fwd.rho_bar
            J = dissipation_objective(rho_bar, fwd.ux, fwd.uy, alpha_max, case.q)

            adj.macro()
                
            if self.field_dump is not None:
                self.field_dump.maybe_capture(it, fwd, adj, G, rho_bar, alpha_max, beta)

            rho_new = self.optimizer.update(
                rho_e, G, beta=beta, iteration=it,
                fixed_mask=case.fixed_mask, fixed_values=case.fixed_values)

            rho_bar_new = (heaviside_projection(rho_new, beta)
                           if beta is not None else rho_new)
            volume = float(np.mean(rho_bar_new))

            # greyness: 0 = fully binary, 1 = all cells at 0.5
            grey = float(np.mean(4.0 * rho_bar_new * (1.0 - rho_bar_new)))

            change = np.abs(rho_new - rho_e)
            change_max = float(change.max())
            change_mean = float(change.mean())
            rho_e = rho_new

            if self.recorder is not None:
                self.recorder.capture(
                    loop=it,
                    fwd_vel=np.sqrt(fwd.ux**2 + fwd.uy**2),
                    adj_vel=np.sqrt(adj.ux**2 + adj.uy**2),
                    G=G, rho_bar=rho_bar, J=J,
                    lam=self.optimizer.lam, alpha=alpha_max, beta=beta,
                    fwd_iters=fwd.it, adj_iters=adj.it,
                    greyness=grey, change_max=change_max,
                    change_mean=change_mean,
                    move_limit=self.optimizer.current_move(it),
                    G_max=G_max, volume=volume)
            

            self.continuation.notify(it, change_mean)
            J_window.append(J)
            greyness_window.append(grey)

            dt = timer() - t_iter
            ewa = dt if ewa is None else ewa_beta * ewa + (1.0 - ewa_beta) * dt
            elapsed = timer() - t_start
            remaining = (max_iter - it - 1) * ewa

            

            log(f"it {it:4d}  J={J:.6e}  vol={volume:.4f}  "
                f"{flux_text}"
                f"grey={grey:.3f}  dmax={change_max:.3e} "
                f"dmean={change_mean:.3e}  |G|max={float(np.max(np.abs(G))):.2e}  "
                f"lam={self.optimizer.lam:+.3e}  "
                f"move={self.optimizer.current_move(it):.3f}  "
                f"alpha={alpha_max:6.1f} beta={beta:5.2f}  "
                f"fwd={fwd.it:6d} adj={adj.it:6d}  "
                f"| elapsed {_hms(elapsed)}  ewa {ewa:5.2f}s  "
                f"left ~{_hms(remaining)}")

            # Converged when the design has settled (mean change), the
            # objective has stopped varying over a window, and continuation
            # has finished so the problem itself is no longer changing.
            if len(J_window) == self.convergence_window and self.continuation.is_complete(it):
                j_spread = ((max(J_window) - min(J_window))
                            / max(abs(np.mean(J_window)), 1e-30))
                grey_spread = ((max(greyness_window) - min(greyness_window))
                               / max(np.mean(greyness_window), 1e-30))
                if j_spread < tol_J and grey_spread < tol_grey:
                    converged = True
                    log(f"\nConverged: J spread {j_spread:.2e}, greyness "
                        f"spread {grey_spread:.2e} over the last {self.convergence_window} "
                        f"iterations")
                    break

        total = timer() - t_start
        log("")
        log("=" * 100)
        log(f"{'CONVERGED' if converged else 'STOPPED (max_iter reached)'} "
            f"at iteration {it}")
        log(f"total time       : {_hms(total)}")
        log(f"final objective  : {J:.6e}")
        log(f"final volume     : {volume:.4f}  (target {case.volume_fraction})")
        log(f"final greyness   : {grey:.3f}  (0 = binary)")
        log(f"final alpha/beta : {alpha_max:.1f} / {beta:.1f}")
        log("=" * 100)
        log.close()

        if self.recorder is not None:
            self.recorder.save_arrays(
                self.paths.arrays,
                rho_e=rho_e, rho_bar=fwd.rho_bar, obstacle=fwd.obstacle,
                ux=fwd.ux, uy=fwd.uy, G=G,
                objective=np.array(self.recorder.loss))
            self.recorder.close(self.paths.plot)
        if self.field_dump is not None:
            self.field_dump.close()

        return rho_e, fwd.rho_bar




class _Tee:
    """Write to stdout and a log file at once, so a long run is both
    watchable live and recoverable afterwards."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "w", encoding="utf-8")

    def __call__(self, line=""):
        print(line)
        self.fh.write(line + "\n")
        self.fh.flush()          # survive a crash or a Ctrl-C mid-run

    def close(self):
        self.fh.close()




def _hms(seconds):
        """Seconds -> 'HHh MMm SSs'."""
        s = int(max(seconds, 0))
        return f"{s // 3600:02d}h {(s % 3600) // 60:02d}m {s % 60:02d}s"