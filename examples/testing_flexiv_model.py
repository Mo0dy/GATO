import sys
import time
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import meshcat.geometry as g
# import meshcat.transformations as tf

from pinocchio.robot_wrapper import RobotWrapper

# Add paths
sys.path.append('./python/bsqp')
sys.path.append('./python')

# from bsqp.mpc_controller import MPC_GATO
# from bsqp.config import (
#     PICKPLACE_DEFAULT_GOALS, 
#     PENDULUM_DEFAULT_PARAMS, 
#     FLEXIV_RIZON_4S_START_CONFIGS, 
#     PICKPLACE_SOLVER_PARAMS
# )

# def ee_pos(model, q):
#     data = model.createData()
#     pin.forwardKinematics(model, data, q)
    # return data.oMi[model.njoints - 1].translation

# def ee_pos(robot, q):
#     """Get end-effector position using RobotWrapper."""
#     pin.forwardKinematics(robot.model, robot.data, q)
#     jid_ee_pin = robot.model.getFrameId("flange")  # End-effector Reference Frame
#     jid_eep_pin = robot.model.frames[jid_ee_pin].parentJoint
#     return robot.data.oMi[jid_eep_pin].translation

def ee_pos(robot, q):
        frame_id = robot.model.getFrameId("flange")
        p = robot.framePlacement(q, frame_id, update_kinematics=True).translation
        return p

def rk4(robot, q, dq, u, dt, fext=None):
    """
    RK4 integration for forward dynamics using RobotWrapper.
    
    Args:
        robot: RobotWrapper object
        q: Joint positions
        dq: Joint velocities
        u: Control torques
        dt: Time step
        fext: External forces (optional)
    
    Returns:
        q_next: Joint positions at next timestep
        dq_next: Joint velocities at next timestep
    """
    
    model = robot.model
    data = robot.data

    if fext is None:
        fext = pin.StdVec_Force()
        for _ in range(model.njoints):
            fext.append(pin.Force.Zero())
    
    # RK4 integration steps
    k1q = dq
    k1v = pin.aba(model, data, q, dq, u, fext)
    print("k1v:", k1v)
    
    q2 = pin.integrate(model, q, k1q * dt / 2)
    k2q = dq + k1v * dt/2
    k2v = pin.aba(model, data, q2, k2q, u, fext)
    print("k2v:", k2v)
    
    q3 = pin.integrate(model, q, k2q * dt / 2)
    k3q = dq + k2v * dt/2
    k3v = pin.aba(model, data, q3, k3q, u, fext)
    print("k3v:", k3v)
    
    q4 = pin.integrate(model, q, k3q * dt)
    k4q = dq + k3v * dt
    k4v = pin.aba(model, data, q4, k4q, u, fext)
    print("k4v:", k4v)
    
    dq_next = dq + (dt/6) * (k1v + 2*k2v + 2*k3v + k4v)
    avg_dq = (k1q + 2*k2q + 2*k3q + k4q) / 6
    q_next = pin.integrate(model, q, avg_dq * dt)
    
    return q_next, dq_next

np.set_printoptions(linewidth=200)
np.random.seed(42)

#  Robot model
urdf_path = "flexiv_description/flexiv_rizon4s_kinematics.urdf"
model_dir = "flexiv_description/"
robot = RobotWrapper.BuildFromURDF(urdf_path, model_dir)

# MPC parameters
N = 16
dt = 0.1
sim_dt = 0.001

nq = robot.nq
nv = robot.nv
# 
q0 = [0.0, -0.698, 0.000, 1.571, -0.000, 0.698, -0.000]
x_start = np.hstack((q0, np.zeros(nv)))
q = x_start[:nq]
dq = x_start[nq:]
# 
# Print initial state
print(f"Initial joint positions: [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}, {q[4]:.4f}, {q[5]:.4f}, {q[6]:.4f}]rad")
print(f"Initial joint velocities: [{dq[0]:.4f}, {dq[1]:.4f}, {dq[2]:.4f}, {dq[3]:.4f}, {dq[4]:.4f}, {dq[5]:.4f}, {dq[6]:.4f}]rad/s")

# Compute initial end-effector position
ee_initial_pos = ee_pos(robot, q)
print(f"Initial end-effector position: [{ee_initial_pos[0]:.4f}, {ee_initial_pos[1]:.4f}, {ee_initial_pos[2]:.4f}]m")

# Initial controls (single control vector, not trajectory)
u_start = np.zeros(nv)

# State transition (don't pass None, omit fext or it will be created internally)
q, dq = rk4(robot, q, dq, u_start, sim_dt)
print(f"Post-RK4 joint positions: [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}, {q[4]:.4f}, {q[5]:.4f}, {q[6]:.4f}]rad")
print(f"Post-RK4 joint velocities: [{dq[0]:.4f}, {dq[1]:.4f}, {dq[2]:.4f}, {dq[3]:.4f}, {dq[4]:.4f}, {dq[5]:.4f}, {dq[6]:.4f}]rad/s")



