from topopt.src.cases.pipe_bend.pipe_bend_case import PipeBendCase
from topopt.src.cases.pressure_fork.pressure_fork_case import PressureForkCase
from topopt.src.core.filter import SensitivityFilter
from topopt.src.opt.continuation import *
from topopt.src.opt.driver import TopOptDriver
from topopt.src.plot.field_dump import FieldDump
from topopt.src.plot.optimization_recorder import OptimizationRecorder
from topopt.src.opt.optimality_criteria import MultiplicativeOC
from topopt.src.opt.continuation import GeometricContinuation

resolution = 32

driver = TopOptDriver(
    run_name="pipe_bend_geoCont_multOC",

    case=PipeBendCase(nx=resolution, ny=resolution, volume_fraction=0.3,
                        Re=1.0, tau_lbm=0.6, q=0.1),

    sensitivity_filter=SensitivityFilter(radius=2.0),

    optimizer=MultiplicativeOC(move=0.2, eta=0.5, rho_min=1e-3),

    continuation=GeometricContinuation(complete_by=100,
                        alpha_start=0.1, alpha_end=40.0,
                        beta_start=1.0, beta_end=2.0, beta_delay=80),

    recorder=OptimizationRecorder(g_linthresh_pct=75),

    field_dump=FieldDump(every=20),
)

driver.run(max_iter=400)