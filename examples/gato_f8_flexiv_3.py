import FlexivPy.robot.sim.sim_robot as sim_robot
import FlexivPy.robot.robot_client as robot_client
import numpy as np
import time
import argparse
import easy_controllers
from FlexivPy.robot.dds.flexiv_messages import (
    FlexivCmd,
)
import argparse
import os

import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

import sys
# Ensure repo-local python/ is on path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PYTHON_DIR = os.path.normpath(os.path.join(_THIS_DIR, '..', 'python'))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)
from bsqp.interface import BSQP  # or from bsqp.bsqp_solver import BSQP
from bsqp.common import rk4
from bsqp.config import DEFAULT_SOLVER_PARAMS

# sys.path.append('/home/gabriel/workspace_iiwa/src/GATO/examples')
# from force_estimator import ForceEstimator
# from .gato_force_estimator import ImprovedForceEstimator
# from .force_estimator_cem import CEMForceEstimator

import time

from bsqp.mpc_controller_m3 import MPCState
from bsqp.mpc_controller_m3 import MPC_GATO
from bsqp.hardware_controller import MPCHardwareController
from bsqp.common import figure8

from bsqp.config import (
    PICKPLACE_DEFAULT_GOALS, 
    FIG8_DEFAULT_PARAMS, 
    FLEXIV_RIZON_4S_START_CONFIGS, 
    BATCH_COLORS
)

np.random.seed(42)

argp = argparse.ArgumentParser(description="FlexivPy")
argp.add_argument("--mode", type=str, default="sim", help="mode: real, sim, sim_async")
argp.add_argument("--hw", action="store_true", help="use threaded hardware-style controller")

args = argp.parse_args()

if args.mode not in ["sim", "real", "sim_async"]:
    raise ValueError("mode not recognized")

ref_kp = np.array([3000.0, 3000.0, 800.0, 800.0, 200.0, 200.0, 200.0])
ref_kv = np.array([80.0, 80.0, 40.0, 40.0, 8.0, 8.0, 8.0])

kp_scale = 1.0
kv_scale = 1.0

if args.mode in ["sim", "sim_async"]:
    # in mujoco lower gains work better!
    kp_scale = 0.2
    kv_scale = 0.2

ASSETS_PATH = "FlexivPy/assets/"
urdf = os.path.join(ASSETS_PATH, "flexiv_rizon4s_kinematics_vz.urdf")
meshes_package_dir = ASSETS_PATH  # contains the "meshes" directory

model, visual_model, collision_model = pin.buildModelsFromUrdf(urdf, meshes_package_dir)
pin_model = RobotWrapper.BuildFromURDF(urdf, meshes_package_dir)

# MPC Configuration
config = {
    'batch_sizes': [1, 32, 128],
    'N': 32,
    'dt': 0.01,
    'sim_time': 16.0,              # Total sim time
    'sim_dt': 0.001,               # Simulation timestep
    'start_config': 'home',        # Starting configuration ('zero', 'home', or 'ready')
    'f_ext': np.array([0.0, 0.0, -60.0, 0.0, 0.0, 0.0])  # External force [fx, fy, fz, mx, my, mz]
}

print("Configuration:")
print(f"  Batch sizes: {config['batch_sizes']}")
print(f"  Horizon: N={config['N']}, dt={config['dt']}s")
print(f"  Simulation: {config['sim_time']}s at {1/config['sim_dt']:.0f}Hz")
print(f"  External force: {config['f_ext'][:3]} N")

print("="*60+"\n")

# =============================================================================================================
print("Running Batch Size = 1")
print("="*60)

# Reference EE trajectory
fig8_traj = figure8(config['dt'], **FIG8_DEFAULT_PARAMS)

# Starting configuration
x_start = np.hstack((FLEXIV_RIZON_4S_START_CONFIGS['home'], np.zeros(7)))

frame_id = pin_model.model.getFrameId("flange")
p = pin_model.framePlacement(x_start[:pin_model.model.nq], frame_id, update_kinematics=True).translation
print("Initial ee_pose:", p)

mpc_1 = MPC_GATO(
    model=model,
    model_path=urdf,
    N=config['N'],
    dt=config['dt'],
    batch_size=1,
    plant_type='flexiv_rizon4s',
    constant_f_ext=config['f_ext'],
    track_full_stats=True
)

print("="*60+"\n")
# =============================================================================================================

if args.mode == "sim":
    robot = sim_robot.FlexivSim(
        render=True,
        q0=x_start[:pin_model.model.nq],
        pin_model=pin_model,
    )
elif args.mode == "sim_async":
    base_path = ""
    config = "FlexivPy/config/robot.yaml"
    xml_path = "FlexivPy/assets/mjmodel.xml"
    joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

    cmd = [
        "python",
        base_path + "FlexivPy/robot/sim/sim_robot_async.py",
        "--render",
        "--config",
        config,
        "--xml_path",
        xml_path,
        "--urdf",
        urdf,
        "--meshes_dir",
        meshes_package_dir,
        "--joints",
    ] + joints

    robot = robot_client.Flexiv_client(cmd)
elif args.mode == "real":
    # I can also start the server here if i provide a cmd,
    # similar to sim async.
    robot = robot_client.Flexiv_client()


try:
    print("current state of the robot")
    print(robot.get_robot_state())
    print("we can also get images and object poses if available!")
    env_image = robot.get_env_image()
    obj_poses = robot.get_env_state()
    if env_image is None:
        print("images are not available!")
    if obj_poses is None:
        print("object poses are not available!")
    # TODO: mass do not match!! -- fix this!!
    status = easy_controllers.run_controller(
        robot,
        easy_controllers.GoJointConfiguration(
            qdes=np.array([0.0, -0.698, 0.000, 1.571, -0.000, 0.698, -0.000]),
            max_v=0.2,
            max_extra_time_rel=0.2,
            kp_scale=kp_scale,
            kv_scale=kv_scale,
        ),
        dt=0.005,
        max_time=120,
        sync_sim=args.mode == "sim",
        dt_sim=robot.dt if args.mode == "sim" else None,
    )
    print("status is:", status)

    # =============================================================================================================
    x_start = np.hstack((np.array(robot.get_robot_state().q), np.array(robot.get_robot_state().dq)))
    # print("Starting MPC from state: \n", x_start)

    if args.hw:
        # Use threaded hardware-style controller
        def get_state_fn():
            s = robot.get_robot_state()
            return np.hstack((np.array(s.q), np.array(s.dq)))

        def send_torque_fn(u: np.ndarray):
            robot.set_cmd(FlexivCmd(tau_ff=u))

        ctrl = MPCHardwareController(
            mpc=mpc_1,
            goals=fig8_traj,
            control_dt=robot.dt,
            solve_period=config['dt'],
            get_state_fn=get_state_fn,
            send_torque_fn=send_torque_fn,
        )

        ctrl.start(x_start)
        max_time = 16.0
        t0 = time.time()
        try:
            if args.mode == "sim":
                # Step sim while control thread runs
                while time.time() - t0 < max_time:
                    robot.step()
                    time.sleep(robot.dt)
            else:
                while time.time() - t0 < max_time:
                    time.sleep(0.1)
        finally:
            ctrl.stop()
    else:
        # Original single-threaded loop
        _, mpc_stats_1, x_curr, ee_g = mpc_1.init_mpc(
            x_start,
            fig8_traj,
            problem_type="figure8"
        )

        ws_XU_best, ws_XU_batch, ee_g_batch = mpc_1.warm_start_mpc(
            x_curr,
            ee_g
        )

        # Start timing for the current goal
        tic_start = time.time()
        dt = 0.01
        counter = 0
        max_time = 16

        mpc_flag = True

        mpc_state = MPCState(x_curr=x_curr,
                             x_last=np.zeros_like(x_curr),
                             u_last=np.zeros_like(ws_XU_best[mpc_1.nx:mpc_1.nx + mpc_1.nu]),
                             XU_batch=ws_XU_batch,
                             XU_best=ws_XU_best,
                             ee_g=ee_g,
                             ee_g_batch=ee_g_batch,
                             solve_time=mpc_1.dt,
                             accumulated_time=0.0,
                             total_sim_time=0.0,
                             current_goal_idx=0,
                             goal_start_time=0.0
                             )

        while mpc_flag == True and mpc_state.total_sim_time < max_time:
            tic = time.time()
            elapsed_time = tic - tic_start if args.mode != "sim" else counter * dt

            # Store states for force estimation
            mpc_state.x_last = mpc_state.x_curr
            mpc_state.u_last = mpc_state.XU_best[mpc_1.nx:mpc_1.nx + mpc_1.nu]

            # Update the current state
            s = robot.get_robot_state()
            mpc_state.x_curr = np.hstack((np.array(s.q), np.array(s.dq)))

            # Run MPC for the current goal
            mpc_flag = mpc_1.srun_mpc_figure8(
                state=mpc_state,
                goals=fig8_traj,
                sim_dt=robot.dt,
                sim_time=max_time,
                stats=mpc_stats_1
            )

            if mpc_flag == False:
                break

            tau_ff = mpc_state.XU_best[mpc_1.nx:mpc_1.nx+mpc_1.nu]
            cmd = FlexivCmd(tau_ff=tau_ff)
            robot.set_cmd(cmd)

            if args.mode == "sim":
                dt = mpc_state.solve_time
                num_steps = int(dt // robot.dt)
                for i in range(num_steps):
                    robot.step()

            time.sleep(max(0, dt - (time.time() - tic)))

            if args.mode != "sim" and tic - tic_start > max_time:
                break
            if args.mode == "sim" and counter * dt > max_time:
                break

            counter += 1
    # =============================================================================================================

    # lets go back home
    print("trying to go back home")
    status = easy_controllers.run_controller(
        robot,
        easy_controllers.GoJointConfiguration(
            qdes=np.array([0.0, -0.698, 0.000, 1.571, -0.000, 0.698, -0.000]),
            max_v=0.2,
            max_extra_time_rel=0.2,
            kv_scale=kv_scale,
            kp_scale=kp_scale,
        ),
        sync_sim=args.mode == "sim",
        dt_sim=robot.dt if args.mode == "sim" else None,
        dt=0.005,
        max_time=120,
    )
    print("status is", status)


finally:
    print("inside finally!")
    print("note: closing the mujoco simulator sometimes throws a seg. fault")
    robot.close()
    print("done")
