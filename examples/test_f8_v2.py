import os
import sys
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append('./python/bsqp')
sys.path.append('./python')

from bsqp.mpc_controller_m import MPC_GATO
from bsqp.common import figure8, rk4
from bsqp.config import (
    EXPERIMENT_BATCH_SIZES,
    FIG8_DEFAULT_PARAMS, 
    FLEXIV_RIZON_4S_START_CONFIGS,
    BATCH_COLORS
)

np.random.seed(42)

print("Imports complete")

# ====================================================================================
config = {
    'batch_sizes': [1],
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

# ====================================================================================
# Load robot model
urdf_path = "flexiv_description/flexiv_rizon4s_kinematics_vz.urdf"
model_dir = "flexiv_description/"

model, visual_model, collision_model = pin.buildModelsFromUrdf(urdf_path, model_dir)

print(f"  Loaded model")
print(f"  Joints: {model.njoints-1}")
print(f"  DOF: {model.nq}")
print(f"  Generalized velocity: {model.nv}")

# ====================================================================================
# Reference EE trajectory
fig8_traj = figure8(config['dt'], **FIG8_DEFAULT_PARAMS)

# Visualize
ref_points = fig8_traj.reshape(-1, 6)[:, :3]  # position components
# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(ref_points[:, 0], ref_points[:, 1], ref_points[:, 2], 'r-', alpha=0.6)
# ax.set_xlabel('X [m]')
# ax.set_ylabel('Y [m]')
# ax.set_zlabel('Z [m]')
# ax.view_init(elev=20, azim=135)
# plt.show()

print(f"  Total points: {len(ref_points)}")
print(f"  Duration: {len(ref_points) * config['dt']:.1f}s")

# ====================================================================================
results = {}

# starting configuration
x_start = np.hstack((FLEXIV_RIZON_4S_START_CONFIGS[config['start_config']], np.zeros(model.nv)))

print("Running experiments...\n")
print("=" * 60)

for batch_size in config['batch_sizes']:
    print(f"\nBatch size {batch_size}:")
    print("-" * 40)
    
    # Create controller
    mpc = MPC_GATO(
        model=model,
        model_path=urdf_path,
        N=config['N'],
        dt=config['dt'],
        batch_size=batch_size,
        plant_type='flexiv_rizon4s',
        constant_f_ext=config['f_ext'],
        track_full_stats=True
    )
    # Initialize MPC
    stats, x_curr, ee_g = mpc.init_mpc_fig8(x_start, fig8_traj)
    
    # Warm start
    XU_best, XU_batch, ee_g_batch = mpc.warm_start_mpc_fig8(x_curr, ee_g)
    
    # Initialize simulation state
    total_sim_time = 0.0
    accumulated_time = 0.0
    q = x_start[:mpc.nq]
    dq = x_start[mpc.nq:mpc.nx]
    
    # Main control loop
    mpc_flag = True
    solve_time = mpc.dt
    
    while mpc_flag and total_sim_time < config['sim_time']:
        # Store state for force estimation
        x_last = x_curr
        u_last = XU_best[mpc.nx:mpc.nx + mpc.nu]
        
        # Simulate forward with current control
        timestep = solve_time
        nsteps = int(timestep / config['sim_dt'])
        
        for i in range(nsteps):
            offset = int(i / (mpc.dt / config['sim_dt']))
            u_idx = mpc.nx + (mpc.nx + mpc.nu) * min(offset, mpc.N - 1)
            u = XU_best[u_idx:u_idx + mpc.nu]
            
            # Integrate dynamics
            q, dq = rk4(mpc.model, mpc.data, q, dq, u, config['sim_dt'], mpc.actual_f_ext)
            total_sim_time += config['sim_dt']
        
        # Handle residual time
        if timestep % config['sim_dt'] > 1e-5:
            accumulated_time += timestep % config['sim_dt']
            if accumulated_time >= config['sim_dt']:
                accumulated_time = 0.0
                offset = int(nsteps / (mpc.dt / config['sim_dt']))
                u_idx = mpc.nx + (mpc.nx + mpc.nu) * min(offset, mpc.N - 1)
                u = XU_best[u_idx:u_idx + mpc.nu]
                q, dq = rk4(mpc.model, mpc.data, q, dq, u, config['sim_dt'], mpc.actual_f_ext)
                total_sim_time += config['sim_dt']
        
        x_curr = np.concatenate([q, dq])
        
        # Run single MPC step
        XU_best, mpc_flag = mpc.srun_mpc_fig8(
            x_last,
            u_last,
            x_curr,
            XU_batch,
            ee_g_batch,
            total_sim_time,
            fig8_traj,
            stats
        )
        
        if not mpc_flag:
            break
        
        solve_time = mpc.dt  # Could track actual solve time if needed
    
    # Convert to numpy arrays
    for key in stats:
        if stats[key]:
            try:
                stats[key] = np.array(stats[key])
            except (ValueError, TypeError):
                # Keep as list if conversion fails
                pass
    
    # Print summary
    print(f"Avg error: {np.mean(stats['goal_distances']):.4f}m")
    print(f"Avg solve time: {np.mean(stats['solve_times']):.3f}ms")
    
    results[batch_size] = stats
    print(f"Completed {len(stats['timestamps'])} iterations")

print("\n" + "=" * 60)
print("All experiments complete")

# ====================================================================================
fig, axes = plt.subplots(1, 3, figsize=(10, 4))
batch_sizes = sorted(results.keys())

for idx, batch_size in enumerate(batch_sizes):
    ax = axes[idx]
    
    # Plot reference traj (dotted line)
    ax.plot(ref_points[:, 0], ref_points[:, 2], ':', 
            linewidth=1.0, alpha=0.5, label='Reference')
    
    # Plot actual trajectory
    ee_actual = results[batch_size]['ee_actual']
    color = BATCH_COLORS.get(batch_size, '#000000')
    ax.plot(ee_actual[:, 0], ee_actual[:, 2],
            color=color, linewidth=1.5, label=f'Batch Size = {batch_size}', alpha=0.8)
    
    if idx == 0:
        ax.set_ylabel('Z [m]', fontsize=14)
    else:
        ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    
    if idx == 1:
        ax.set_xlabel('X [m]', fontsize=14)
    
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim(-0.7, 0.0)
    ax.set_ylim(0.5, 1.1)

# Add single legend for all subplots
handles, labels = [], []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    handles.extend(h)
    labels.extend(l)

# Remove duplicates
unique_labels = []
unique_handles = []
for handle, label in zip(handles, labels):
    if label not in unique_labels:
        unique_labels.append(label)
        unique_handles.append(handle)

fig.legend(unique_handles, unique_labels, loc='upper center', 
           bbox_to_anchor=(0.5, 1.08), ncol=4)

plt.tight_layout()
plt.subplots_adjust(top=0.97)
plt.show()