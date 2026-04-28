import numpy as np

from bsqp.common import figure8, initialize_warm_start


def test_warm_start_repeats_current_state_at_every_knot_with_zero_controls():
    nx = 4
    nu = 2
    knots = 3
    x_start = np.array([0.25, -0.5, 1.5, 2.0])

    warm = initialize_warm_start(x_start, N=knots, nx=nx, nu=nu)

    assert warm.shape == (knots * (nx + nu) - nu,)
    for knot in range(knots):
        knot_start = knot * (nx + nu)
        np.testing.assert_allclose(warm[knot_start:knot_start + nx], x_start)

        control_start = knot_start + nx
        control_stop = min(control_start + nu, warm.size)
        np.testing.assert_allclose(warm[control_start:control_stop], 0.0)


def test_figure8_returns_finite_pose_references_for_requested_cycles():
    dt = 0.1
    period = 2.0
    cycles = 3

    traj = figure8(dt=dt, period=period, cycles=cycles)

    expected_points_per_cycle = int(period / dt)
    expected_shape = (expected_points_per_cycle * cycles * 6,)
    assert traj.shape == expected_shape

    reshaped = traj.reshape(-1, 6)
    assert np.isfinite(reshaped).all()
    np.testing.assert_allclose(reshaped[:, 3:], 0.0)


def test_figure8_repeats_the_same_closed_reference_each_cycle():
    dt = 0.05
    period = 1.0
    cycles = 2

    traj = figure8(dt=dt, period=period, cycles=cycles)

    points_per_cycle = int(period / dt)
    cycle_a, cycle_b = traj.reshape(cycles, points_per_cycle, 6)
    np.testing.assert_allclose(cycle_a, cycle_b)
