from topopt.src.cases.pipe_bend.pipe_bend_case import PipeBendCase
from topopt.src.cases.triple_split.triple_split_case import TripleSplitCase
from topopt.src.cases.split.split_case import SplitCase
from topopt.src.cases.shelf.shelf_case import ShelfCase
from topopt.src.cases.triple_split_reverse.triple_split_reverse_case import TripleSplitReverseCase
from topopt.src.core.filter import SensitivityFilter
from topopt.src.opt.continuation import *
from topopt.src.opt.driver import TopOptDriver
from topopt.src.plot.field_dump import FieldDump
from topopt.src.plot.optimization_recorder import OptimizationRecorder
from topopt.src.opt.optimality_criteria import MultiplicativeOC
from topopt.src.opt.continuation import GeometricContinuation
from topopt.src.study.mesh_independence import mesh_independence_study
from topopt.src.utils.archive import load_runs, load_run


resolution = 32

driver = TopOptDriver(
    run_name=f"triple_readme",

    case=TripleSplitCase(nx=resolution, ny=resolution, volume_fraction=0.6,
                        Re=1.0, tau_lbm=0.6,
                        split_1=0.2, split_2=0.1),

    sensitivity_filter=SensitivityFilter(radius=resolution//16),

    optimizer=MultiplicativeOC(move=0.2, convergence_window=25),

    continuation=GeometricContinuation(
                        complete_alpha_by=100, beta_delay=100, complete_beta_by=120,
                        alpha_start=0.1, alpha_end=20.0,
                        beta_start=1.0, beta_end=2.0),

    recorder=OptimizationRecorder(orientation='horizontal', fmt='gif', g_linthresh_pct=60, modules=12),

    field_dump=FieldDump(every=20),
)

driver.run(max_iter=300)


driver = TopOptDriver(
    run_name=f"triple_readme",

    case=TripleSplitCase(nx=resolution, ny=resolution, volume_fraction=0.6,
                        Re=1.0, tau_lbm=0.6,
                        split_1=0.2, split_2=0.1),

    sensitivity_filter=SensitivityFilter(radius=resolution//16),

    optimizer=MultiplicativeOC(move=0.2, convergence_window=25),

    continuation=GeometricContinuation(
                        complete_alpha_by=100, beta_delay=100, complete_beta_by=120,
                        alpha_start=0.1, alpha_end=20.0,
                        beta_start=1.0, beta_end=2.0),

    recorder=OptimizationRecorder(orientation='horizontal', fmt='gif', g_linthresh_pct=60, modules=12),

    field_dump=FieldDump(every=20),
)

driver.run(max_iter=300)


# resolutions = [32]

# for r in resolutions:
#     driver = TopOptDriver(
#         run_name=f"triple_readme",

#         case=PipeBendCase(nx=r, ny=r, volume_fraction=0.3,
#                             Re=1.0, tau_lbm=0.6, q=0.1),

#         sensitivity_filter=SensitivityFilter(radius=r//16),

#         optimizer=MultiplicativeOC(move=0.3, convergence_window=10),

#         continuation=GeometricContinuation(
#                             complete_alpha_by=30, beta_delay=30, complete_beta_by=40,
#                             alpha_start=10, alpha_end=25.0,
#                             beta_start=1.0, beta_end=4.0),

#         recorder=OptimizationRecorder(g_linthresh_pct=60, modules=11),

#         field_dump=FieldDump(every=20),
#     )

#     driver.run(max_iter=100)


# ---------------------------------------------------------------- mesh study
# Every resolution archived under this name, finest one taken as the
# reference. Set `resolution` to 16, 32, 64 ... and run this file once each;
# the archives accumulate in results/arrays/ and the study picks them all up,
# so no run is ever repeated just to compare it.
#
# The filter radius is resolution//16, i.e. a fixed fraction of ny. That is
# what makes the comparison meaningful: the radius is measured in CELLS, so
# holding it at a constant number would change the regularised problem with
# every refinement and the study would be measuring the filter.


# run_64 = load_run("C:\\CONSANARCHY\\Warwick\\URSS\\topopt-lbm-python\\results\\arrays\\triple_study_64Ny.npz")
# run_128 = load_run("C:\\CONSANARCHY\\Warwick\\URSS\\topopt-lbm-python\\results\\arrays\\triple_study_128Ny.npz")
# runs = [run_64, run_128]
# runs = load_runs("triple_final_*Ny")
# mesh_independence_study(runs, modules=(17,11))




