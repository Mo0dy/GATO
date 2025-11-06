import sys
import time
import numpy as np
import pinocchio as pin
from .interface import BSQP
from .common import rk4
from .config import DEFAULT_SOLVER_PARAMS
from dataclasses import dataclass

# Import force estimator if available
sys.path.append('./examples')
try:
    from force_estimator import ForceEstimator
except ImportError:
    ForceEstimator = None

@dataclass
class MPCState:
    x_curr: np.ndarray
    x_last: np.ndarray
    u_last: np.ndarray
    XU_best: np.ndarray
    XU_batch: np.ndarray
    ee_g: np.ndarray
    ee_g_batch: np.ndarray
    solve_time: float
    accumulated_time: float
    total_sim_time: float
    current_goal_idx: int
    goal_start_time: float

class MPC_GATO:

    def __init__(
        self,
        model,
        model_path,
        N=32,
        dt=0.03125,
        batch_size=1,
        constant_f_ext=None,
        track_full_stats=False,
        plant_type='indy7',
        pendulum_config=None,
        solver_params=None,
    ):
        """
        Initialize MPC controller.
        
        Args:
            model: Pinocchio model
            model_path: Path to URDF file
            N: Prediction horizon (knot points)
            dt: Time step
            batch_size: Number of parallel trajectories
            constant_f_ext: Constant external force/torque (optional)
            track_full_stats: If True, track all stats; if False, only essential ones
            plant_type: Plant identifier used for selecting dynamics (e.g., 'indy7', 'iiwa14')
            pendulum_config: Optional dict with keys: mass, length, damping, initial_angle
        """
        # Store original model for solver (without pendulum)
        self.solver_model = model
        
        # Add pendulum to simulation model if configured
        if pendulum_config is not None:
            self.model = self._add_pendulum_to_model(model.copy(), pendulum_config)
            self.pendulum_config = pendulum_config
            self.has_pendulum = True
            # Store dimensions
            self.nq_robot = self.solver_model.nq
            self.nv_robot = self.solver_model.nv
        else:
            self.model = model
            self.pendulum_config = None
            self.has_pendulum = False
            self.nq_robot = model.nq
            self.nv_robot = model.nv
            
        self.model.gravity.linear = np.array([0, 0, -9.81])
        self.data = self.model.createData()
        
        # Initialize solver with configurable parameters
        solver_cfg = DEFAULT_SOLVER_PARAMS.copy()
        if solver_params is not None:
            solver_cfg.update(solver_params)
            # print("Using custom solver parameters:")

        self.solver = BSQP(
            model_path=model_path,
            batch_size=batch_size,
            N=N,
            dt=dt,
            plant_type=plant_type,
            max_sqp_iters=solver_cfg['max_sqp_iters'],
            kkt_tol=solver_cfg['kkt_tol'],
            max_pcg_iters=solver_cfg['max_pcg_iters'],
            pcg_tol=solver_cfg['pcg_tol'],
            solve_ratio=solver_cfg['solve_ratio'],
            mu=solver_cfg['mu'],
            q_cost=solver_cfg['q_cost'],
            qd_cost=solver_cfg['qd_cost'],
            u_cost=solver_cfg['u_cost'],
            N_cost=solver_cfg['N_cost'],
            q_lim_cost=solver_cfg['q_lim_cost'],
            vel_lim_cost=solver_cfg['vel_lim_cost'],
            ctrl_lim_cost=solver_cfg['ctrl_lim_cost'],
            rho=solver_cfg['rho'],
        )

        self.solver_params = solver_cfg
        
        self.nq = self.model.nq
        self.nv = self.model.nv
        self.nx = self.nq_robot + self.nv_robot  # Solver state dimension (robot only)
        self.nu = self.solver_model.nv  # Control dimension (robot only)
        self.N = N
        self.dt = dt
        self.batch_size = batch_size
        self.track_full_stats = track_full_stats
        
        # Setup external forces if provided
        self.setup_external_forces(constant_f_ext)
        
        # Setup force estimator for batch > 1
        self.setup_force_estimator()
        
        # Goal tracking parameters
        self.goal_timeout = 5.0
        self.goal_threshold = 0.05
        self.velocity_threshold = 1.0
        self.sim_dt = 0.001
        
        # Buffer for current state batch (robot only)
        self._x_curr_batch_buffer = np.zeros((self.batch_size, self.nx))

    def setup_external_forces(self, constant_f_ext):
        """Setup external forces for simulation."""
        self.constant_f_ext_world = constant_f_ext if constant_f_ext is not None else np.zeros(6)
        
        # Create force vector for Pinocchio simulation
        self.actual_f_ext = pin.StdVec_Force()
        for _ in range(self.model.njoints):
            self.actual_f_ext.append(pin.Force.Zero())
            
        if constant_f_ext is not None:
            self.actual_f_ext[-1] = pin.Force(constant_f_ext[:3], constant_f_ext[3:])
    
    def setup_force_estimator(self):
        """Initialize force estimator for batch processing."""
        if self.batch_size > 1 and ForceEstimator is not None:
            self.force_estimator = ForceEstimator(
                batch_size=self.batch_size,
                initial_radius=5.0,
                min_radius=2.0,
                max_radius=20.0,
                smoothing_factor=0.5
            )
        else:
            self.force_estimator = None

    def update_force_batch(self, q):
        """Update force hypotheses for batch solving."""
        if self.batch_size == 1 or self.force_estimator is None:
            return
            
        # Generate force batch
        force_batch = self.force_estimator.generate_batch()
        
        # Transform to GATO frame
        transformed_batch = np.zeros_like(force_batch)
        for i in range(self.batch_size):
            transformed_batch[i, :] = self.transform_force_to_gato_frame(q, force_batch[i, :])
            
        self.solver.set_f_ext_B(transformed_batch)
        
    def evaluate_best_trajectory(self, x_last, u_last, x_curr, dt):
        """Evaluate which trajectory best matches reality."""
        if self.batch_size == 1 or self.force_estimator is None:
            return 0
            
        # Simulate all hypotheses
        x_next_batch = self.solver.sim_forward(x_last, u_last, dt)
        
        # Calculate errors
        errors = np.linalg.norm(x_next_batch - x_curr[None, :], axis=1)
        best_id = np.argmin(errors)
        
        # Update estimator
        self.force_estimator.update(best_id, errors, alpha=0.6, beta=0.5)
        
        return best_id
        
    def transform_force_to_gato_frame(self, q, f_world):
        """Transform force from world frame to GATO frame."""
        data = self.solver_model.createData()
        q_robot = q[:self.nq_robot]
        
        pin.forwardKinematics(self.solver_model, data, q_robot)
        pin.updateFramePlacements(self.solver_model, data)
        
        # Joint indices - get end-effector parent joint
        jid_ee_fin = self.solver_model.getFrameId("EE")
        jid_ee_pin = self.solver_model.frames[jid_ee_fin].parentJoint
        jid_eep_pin = jid_ee_pin - 1  # End-effector parent joint
        
        # Get transformations
        transform_world_to_ee = data.oMi[jid_ee_pin]
        transform_world_to_jeep = data.oMi[jid_eep_pin]
        transform_jeep_to_ee = transform_world_to_jeep.inverse() * transform_world_to_ee
        
        # Transform force
        force_ee_world = pin.Force(f_world[:3], f_world[3:])
        force_ee_local = transform_world_to_ee.actInv(force_ee_world)
        wrench_jeep_local = transform_jeep_to_ee.actInv(force_ee_local)
        
        result = np.zeros(6)
        result[:3] = wrench_jeep_local.linear
        result[3:] = wrench_jeep_local.angular
        
        return result
    
    def init_mpc(
        self, 
        x_start, 
        targets,
        problem_type='goals'
    ):
        """
        Initialize MPC controller tracking discrete goal positions.
        
        """
        # Initialize essential statistics
        stats = {
            'timestamps': [],
            'solve_times': [],        # GPU solve time in ms
            'goal_distances': [],     # Tracking error in meters
            'ee_actual': [],          # Actual end-effector positions
            'joint_positions': [],    # Joint positions over time
            'joint_velocities': [],   # Joint velocities over time
        }
        
        # Add SQP iterations only if tracking full stats
        if self.track_full_stats:
            stats['sqp_iters'] = []
            stats['pcg_iters'] = []

        if problem_type == 'goals':    
            stats['goal_outcomes'] = ['not_reached'] * len(targets)
            stats['goal_reached_times'] = [None] * len(targets)
            stats['time_to_all_reached'] = None

        q = x_start[:self.nq]
        dq = x_start[self.nq:self.nx]

        ee_pos = self.solver.ee_pos(q)

        # Print starting state
        print(f"Starting joint positions: [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}, {q[4]:.4f}, {q[5]:.4f}, {q[6]:.4f}]rad")
        print(f"Starting EE position: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]m")
        
        # Solver uses robot-only state
        x_curr = x_start
        
        if problem_type == 'goals':
            # Initialize first goal
            current_goal_idx = 0
            current_goal = targets[current_goal_idx]
            ee_g = np.tile(np.concatenate([current_goal, np.zeros(3)]), self.N)
        elif problem_type == 'figure8':
            ee_g = targets[:6*self.N]
        
        return None, stats, x_curr, ee_g

    def warm_start_mpc(
        self,
        x_curr,
        ee_g
    ):
        """
        Warm start MPC controller.
        
        """
        x_curr_batch = np.tile(x_curr, (self.batch_size, 1))
        ee_g_batch = np.tile(ee_g, (self.batch_size, 1))

        # Initialize warm start
        XU = np.zeros(self.N*(self.nx+self.nu)-self.nu)
        for i in range(self.N):
            start_idx = i * (self.nx + self.nu)
            XU[start_idx:start_idx+self.nx] = x_curr
        XU_batch = np.tile(XU, (self.batch_size, 1))

        # Reset solver
        self.solver.reset_dual()
        
        # Warm up solve
        q = x_curr[:self.nq]
        self.update_force_batch(q)
        XU_batch, _ = self.solver.solve(x_curr_batch, ee_g_batch, XU_batch)
        XU_best = XU_batch[0, :]

        return XU_best, XU_batch, ee_g_batch
    
    def srun_mpc_figure8(
        self,
        state: MPCState,
        goals: np.ndarray,
        sim_dt: float,
        sim_time: float,
        stats: dict,
    ) -> bool:
        """
        Run MPC controller tracking figure-8 trajectory.
        
        """
        # Simulate forward with current control
        timestep = state.solve_time
        nsteps = int(timestep/sim_dt)

        q = state.x_curr[:self.nq]
        dq = state.x_curr[self.nq:]
        
        for i in range(nsteps):
            offset = int(i/(self.dt/sim_dt))
            u_idx = self.nx + (self.nx+self.nu)*min(offset, self.N-1)
            u = state.XU_best[u_idx:u_idx+self.nu]
            q, dq = rk4(self.model, self.data, q, dq, u, sim_dt, self.actual_f_ext)
            state.total_sim_time += sim_dt

        # Handle residual time
        if timestep % sim_dt > 1e-5:
            state.accumulated_time += timestep % sim_dt
            if state.accumulated_time >= sim_dt:
                state.accumulated_time = 0.0
                offset = int(nsteps/(self.dt/sim_dt))
                u_idx = self.nx + (self.nx+self.nu)*min(offset, self.N-1)
                u = state.XU_best[u_idx:u_idx+self.nu]
                q, dq = rk4(self.model, self.data, q, dq, u, sim_dt, self.actual_f_ext)
                state.total_sim_time += sim_dt

        # Update the current state
        state.x_curr = np.concatenate([q, dq])

        # Check if trajectory is complete
        eepos_offset = int(state.total_sim_time / self.dt)
        if eepos_offset >= len(goals)/6 - 6*self.N:
            return False
         
        # Prepare next optimization
        x_curr_batch = np.tile(state.x_curr, (self.batch_size, 1))
        ee_g = goals[6*eepos_offset:6*(eepos_offset+self.N)]
        state.ee_g_batch[:, :] = ee_g
        state.XU_batch[:, :self.nx] = state.x_curr
        
        # Update forces and solve
        self.update_force_batch(q)
        self.solver.reset_rho()

        start = time.time()
        XU_batch_new, gpu_solve_time = self.solver.solve(
            x_curr_batch, 
            state.ee_g_batch, 
            state.XU_batch)
        state.solve_time = time.time() - start

        # Select best trajectory
        best_id = self.evaluate_best_trajectory(
            state.x_last, 
            state.u_last, 
            state.x_curr, 
            max(sim_dt, round(self.dt / sim_dt) * sim_dt)
        )
        state.XU_best = XU_batch_new[best_id, :]
        state.XU_batch[:, :] = state.XU_best

        # Collect essential statistics
        ee_pos = self.solver.ee_pos(q)
        goal_dist = np.linalg.norm(ee_pos[:3] - ee_g[6:9])

        stats['timestamps'].append(state.total_sim_time)
        stats['solve_times'].append(gpu_solve_time/1000.0)  # Convert to ms
        stats['goal_distances'].append(goal_dist)
        stats['ee_actual'].append(ee_pos.copy())
        stats['joint_positions'].append(q.copy())
        stats['joint_velocities'].append(dq.copy())

        if self.track_full_stats:
                solver_stats = self.solver.get_stats()
                # Get first element from batch for sqp_iters
                sqp_iters = solver_stats['sqp_iters']
                if isinstance(sqp_iters, np.ndarray):
                    stats['sqp_iters'].append(int(sqp_iters[0]))
                else:
                    stats['sqp_iters'].append(int(sqp_iters))

        return True  


