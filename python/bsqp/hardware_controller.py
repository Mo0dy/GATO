import threading
import time
from typing import Callable, Optional
import numpy as np

from .mpc_controller_m3 import MPC_GATO, MPCState

class MPCHardwareController:
    """
    Dual-thread controller that runs an MPC solver in a background thread and streams
    torques at a fixed control rate. Robot I/O is provided via callbacks.

    Required callbacks:
      - get_state_fn() -> np.ndarray: returns current [q, dq]
      - send_torque_fn(u: np.ndarray) -> None: sends torque command
    """
    def __init__(
        self,
        mpc: MPC_GATO,
        goals: np.ndarray,
        control_dt: float,
        solve_period: float,
        get_state_fn: Callable[[], np.ndarray],
        send_torque_fn: Callable[[np.ndarray], None],
        sensor_lag_s: float = 0.0,
    ) -> None:
        self.mpc = mpc
        self.goals = goals
        self.control_dt = control_dt
        self.solve_period = solve_period
        self.get_state_fn = get_state_fn
        self.send_torque_fn = send_torque_fn
        self.sensor_lag_s = sensor_lag_s

        # Shared plan
        self._plan_lock = threading.Lock()
        self._plan_XU: Optional[np.ndarray] = None
        self._plan_start_time: float = 0.0

        # Threads
        self._stop = threading.Event()
        self._solve_thread: Optional[threading.Thread] = None
        self._control_thread: Optional[threading.Thread] = None

        # MPC runtime state
        self._state: Optional[MPCState] = None

    def start(self, x_start: np.ndarray) -> None:
        # Initialize MPC and warm start
        _, _, x_curr, ee_g = self.mpc.init_mpc(x_start, self.goals, problem_type='figure8')
        XU_best, XU_batch, ee_g_batch = self.mpc.warm_start_mpc(x_curr, ee_g)

        self._state = MPCState(
            x_curr=x_curr,
            x_last=x_curr.copy(),
            u_last=XU_best[self.mpc.nx:self.mpc.nx + self.mpc.nu].copy(),
            XU_best=XU_best,
            XU_batch=XU_batch,
            ee_g=ee_g,
            ee_g_batch=ee_g_batch,
            solve_time=self.mpc.dt,
            accumulated_time=0.0,
            total_sim_time=0.0,
            current_goal_idx=0,
            goal_start_time=time.monotonic(),
        )

        # Publish initial plan
        with self._plan_lock:
            self._plan_XU = XU_best.copy()
            self._plan_start_time = time.monotonic()

        # Launch threads
        self._stop.clear()
        self._solve_thread = threading.Thread(target=self._solve_loop, name='mpc_solve', daemon=True)
        self._control_thread = threading.Thread(target=self._control_loop, name='mpc_control', daemon=True)
        self._solve_thread.start()
        self._control_thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._solve_thread:
            self._solve_thread.join(timeout=timeout)
        if self._control_thread:
            self._control_thread.join(timeout=timeout)

    # -----------------------
    # Internal loops
    # -----------------------
    def _solve_loop(self) -> None:
        assert self._state is not None
        last_solve = 0.0
        base = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now - base >= last_solve + self.solve_period:
                last_solve += self.solve_period

                # Get latest state and optionally predict forward for sensor lag
                x_meas = self.get_state_fn()
                x_pred = self.mpc.predict_state_forward(
                    x_meas,
                    self._state.u_last,
                    self.sensor_lag_s,
                )
                self._state.x_curr = x_pred

                # Store for batch selection later
                self._state.x_last = x_meas

                # Run one MPC step to update plan (figure-8 goals assumed)
                ok = self.mpc.srun_mpc_figure8(
                    state=self._state,
                    goals=self.goals,
                    sim_dt=self.control_dt,
                    sim_time=1e9,  # not used for stopping here
                    stats={'timestamps': [], 'solve_times': [], 'goal_distances': [], 'ee_actual': [], 'joint_positions': [], 'joint_velocities': []}
                )
                # Update published plan
                with self._plan_lock:
                    self._plan_XU = self._state.XU_best.copy()
                    self._plan_start_time = now

            # Sleep a little to avoid busy wait
            time.sleep(min(0.001, 0.25 * self.solve_period))

    def _control_loop(self) -> None:
        next_tick = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_tick:
                # Sample plan
                with self._plan_lock:
                    plan = self._plan_XU.copy() if self._plan_XU is not None else None
                    plan_start = self._plan_start_time

                if plan is None:
                    # Fallback safety control using current measured state
                    x = self.get_state_fn()
                    q = x[:self.mpc.nq]
                    dq = x[self.mpc.nq:]
                    u = self.mpc.safety_controller(q, dq)
                else:
                    # Compute horizon index based on time since plan start
                    t_in_plan = max(0.0, now - plan_start)
                    k = int(t_in_plan // self.mpc.dt)
                    k = min(k, self.mpc.N - 1)
                    # Extract control
                    u_idx = self.mpc.nx + (self.mpc.nx + self.mpc.nu) * k
                    u = plan[u_idx:u_idx + self.mpc.nu]

                # Send command
                try:
                    self.send_torque_fn(u)
                except Exception:
                    # Do not crash control thread on I/O issues
                    pass

                # Schedule next tick
                next_tick += self.control_dt
            else:
                time.sleep(min(0.0005, max(0.0, next_tick - now)))
