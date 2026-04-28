import json
import os
from pathlib import Path

import importlib.util
import numpy as np
import pytest


PLANT = "indy7"
KNOTS = 8
URDF_PATH = "examples/indy7_description/indy7.urdf"
MODEL_DIR = "examples/indy7_description"


def _require_cuda_tracking_enabled():
    if os.environ.get("GATO_RUN_TRACKING_TESTS") != "1":
        pytest.skip("set GATO_RUN_TRACKING_TESTS=1 to run CUDA tracking tests")

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available to torch in this environment")

    module_name = f"bsqp.bsqpN{KNOTS}_{PLANT}"
    if importlib.util.find_spec(module_name) is None:
        pytest.skip(f"built solver extension is not available: {module_name}")


def _run_short_indy7_tracking():
    _require_cuda_tracking_enabled()

    import pinocchio as pin

    from bsqp.common import figure8
    from bsqp.config import INDY7_START_CONFIGS
    from bsqp.mpc_controller import MPC_GATO

    model, _, _ = pin.buildModelsFromUrdf(URDF_PATH, MODEL_DIR)
    x_start = np.hstack((INDY7_START_CONFIGS["ready"], np.zeros(model.nv)))
    reference = figure8(
        dt=0.01,
        A_x=0.08,
        A_z=0.08,
        offset=[0.0, 0.5, 0.6],
        period=1.0,
        cycles=1,
    )

    controller = MPC_GATO(
        model=model,
        model_path=URDF_PATH,
        N=KNOTS,
        dt=0.01,
        batch_size=1,
        plant_type=PLANT,
        track_full_stats=True,
        solver_params={
            "max_sqp_iters": 1,
            "max_pcg_iters": 80,
            "pcg_tol": 1e-4,
        },
    )

    _, stats = controller.run_mpc_fig8(
        x_start,
        reference,
        sim_dt=0.001,
        sim_time=float(os.environ.get("GATO_TRACKING_SMOKE_SIM_TIME", "0.03")),
    )
    return stats


def _tracking_summary(stats):
    timestamps = np.asarray(stats["timestamps"], dtype=np.float64)
    solve_times = np.asarray(stats["solve_times"], dtype=np.float64)
    goal_distances = np.asarray(stats["goal_distances"], dtype=np.float64)

    return {
        "iterations": int(timestamps.size),
        "avg_solve_time_ms": float(np.mean(solve_times)),
        "max_solve_time_ms": float(np.max(solve_times)),
        "avg_goal_distance_m": float(np.mean(goal_distances)),
        "max_goal_distance_m": float(np.max(goal_distances)),
        "final_goal_distance_m": float(goal_distances[-1]),
    }


def _assert_successful_tracking(stats):
    timestamps = np.asarray(stats["timestamps"], dtype=np.float64)
    solve_times = np.asarray(stats["solve_times"], dtype=np.float64)
    goal_distances = np.asarray(stats["goal_distances"], dtype=np.float64)
    ee_actual = np.asarray(stats["ee_actual"], dtype=np.float64)

    assert timestamps.size >= 1
    assert solve_times.shape == timestamps.shape
    assert goal_distances.shape == timestamps.shape
    assert ee_actual.shape == (timestamps.size, 3)
    assert np.isfinite(solve_times).all()
    assert np.isfinite(goal_distances).all()
    assert np.isfinite(ee_actual).all()
    assert np.all(solve_times >= 0.0)

    max_avg_error = float(os.environ.get("GATO_TRACKING_MAX_AVG_ERROR_M", "1.0"))
    assert float(np.mean(goal_distances)) < max_avg_error


def _write_tracking_artifacts(stats, artifact_dir):
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary = _tracking_summary(stats)
    (artifact_dir / "tracking_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timestamps = np.asarray(stats["timestamps"], dtype=np.float64)
    solve_times = np.asarray(stats["solve_times"], dtype=np.float64)
    goal_distances = np.asarray(stats["goal_distances"], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.0), sharex=True)
    axes[0].plot(timestamps, goal_distances, marker="o")
    axes[0].set_ylabel("Tracking error [m]")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(timestamps, solve_times, marker="o")
    axes[1].set_xlabel("Simulation time [s]")
    axes[1].set_ylabel("GPU solve [ms]")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(artifact_dir / "tracking_smoke.png", dpi=150)
    plt.close(fig)

    return summary


@pytest.mark.cuda
@pytest.mark.tracking
def test_short_figure8_tracking_succeeds():
    stats = _run_short_indy7_tracking()

    _assert_successful_tracking(stats)


@pytest.mark.cuda
@pytest.mark.tracking
@pytest.mark.performance
def test_short_figure8_tracking_writes_performance_artifacts(tmp_path):
    if os.environ.get("GATO_RUN_PERFORMANCE_TESTS") != "1":
        pytest.skip("set GATO_RUN_PERFORMANCE_TESTS=1 to write performance artifacts")

    stats = _run_short_indy7_tracking()
    _assert_successful_tracking(stats)

    artifact_dir = Path(os.environ.get("GATO_TEST_ARTIFACT_DIR", tmp_path))
    summary = _write_tracking_artifacts(stats, artifact_dir)

    max_avg_solve_ms = float(os.environ.get("GATO_PERF_MAX_AVG_SOLVE_MS", "100.0"))
    assert summary["avg_solve_time_ms"] < max_avg_solve_ms
