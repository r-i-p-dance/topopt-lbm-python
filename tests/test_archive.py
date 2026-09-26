"""Archive round-trip, replot and mesh comparison, on tiny runs."""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from topopt.src.cases.triple_split.triple_split_case import TripleSplitCase
from topopt.src.core.filter import SensitivityFilter
from topopt.src.core.projection import heaviside_projection
from topopt.src.opt.continuation import GeometricContinuation
from topopt.src.opt.driver import TopOptDriver
from topopt.src.opt.optimality_criteria import MultiplicativeOC
from topopt.src.plot.optimization_recorder import OptimizationRecorder
from topopt.src.plot.replot import replot_archive
from topopt.src.study.mesh_independence import compare_runs
from topopt.src.utils.archive import load_run


def run(root, ny=16, radius=1.0, name="t"):
    driver = TopOptDriver(
        run_name=name, results_root=str(root),
        case=TripleSplitCase(nx=ny, ny=ny, volume_fraction=0.6, Re=1.0,
                             tau_lbm=0.6, q=0.1),
        sensitivity_filter=SensitivityFilter(radius=radius),
        optimizer=MultiplicativeOC(move=0.2, eta=0.5, rho_min=1e-3),
        continuation=GeometricContinuation(
            complete_alpha_by=5, complete_beta_by=6, alpha_start=0.1,
            alpha_end=20.0, beta_start=1.0, beta_end=2.0),
        recorder=OptimizationRecorder(dpi=60))
    driver.run(max_iter=4, solver_tol=1e-6)
    return load_run(driver.paths.arrays)


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    return run(tmp_path_factory.mktemp("run") / "results")


def test_design_pairs_are_consistent(archive):
    """The reason four design arrays are stored rather than two: rho_bar
    belongs to rho_e, and rho_bar_final to rho_e_final, one update later."""
    beta = archive.meta["design_state"]["beta"]
    assert np.allclose(archive.rho_bar,
                       heaviside_projection(archive.rho_e, beta))
    assert np.allclose(archive.rho_bar_final,
                       heaviside_projection(archive.rho_e_final, beta))
    assert not np.allclose(archive.rho_e, archive.rho_e_final)


def test_metadata_and_series(archive):
    assert archive.meta["case"]["class"] == "TripleSplitCase"
    assert archive.meta["case"]["split_1"] == pytest.approx(1 / 3)
    assert archive.meta["filter"]["radius_over_ny"] == pytest.approx(1 / 16)
    assert len(archive.loss) == 4
    # mean/max is a fraction: 1 means the whole design is clipped at the
    # move limit, small means only a few interface cells still move.
    assert np.all((archive.change_ratio > 0) & (archive.change_ratio <= 1))


def test_replot_needs_no_ffmpeg(archive, tmp_path, monkeypatch):
    import topopt.src.plot.optimization_recorder as recorder_module
    monkeypatch.setattr(recorder_module, "FFMpegWriter", None)
    replot_archive(archive, tmp_path / "fig", dpi=80, formats=("png",))
    assert (tmp_path / "fig.png").stat().st_size > 5_000


def test_compare_runs(tmp_path):
    root = tmp_path / "results"
    coarse = run(root, ny=16, radius=1.0, name="c")   # radius/ny held fixed
    fine = run(root, ny=32, radius=2.0, name="f")
    result = compare_runs(fine, coarse)
    assert result["L2_velocity"] > 0
    assert 0.0 <= result["agreement"] <= 1.0

    # Against itself the comparison must be exact, which catches transposes
    # and mask errors in one line.
    same = compare_runs(fine, fine)
    assert same["L2_velocity"] == 0.0
    assert same["agreement"] == 1.0
