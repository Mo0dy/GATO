#pragma once

#include <math.h>
#include <stdio.h>

#include "tiago_right_grid.cuh"
#include "tiago_right_limits.cuh"

using namespace sqp;

namespace grid {
    constexpr int NQ = NUM_JOINTS;
    constexpr int NX = 2 * NUM_JOINTS;
    constexpr int NU = NUM_JOINTS;
    constexpr int NEE = 6;
    constexpr int EE_POS_SIZE = 6;
}  // namespace grid

#include "utils/linalg.cuh"

namespace gato {
namespace plant {

        constexpr int TIAGO_SIGN_FLIP_JOINT = 5;  // arm_right_6_joint
        constexpr int TIAGO_NUM_JOINTS = 7;
        constexpr int TIAGO_STATE_SIZE = 2 * TIAGO_NUM_JOINTS;
        static_assert(TIAGO_LIMIT_JOINTS == TIAGO_NUM_JOINTS, "Tiago limit data must match the plant joint count");

        template<class T>
        __host__ __device__ constexpr T PI()
        {
                return static_cast<T>(3.14159);
        }

        template<class T>
        __host__ __device__ constexpr T GRAVITY()
        {
                return static_cast<T>(9.81);
        }

        template<class T>
        __host__ __device__ constexpr T tiagoJointSign(uint32_t joint_idx)
        {
                return joint_idx == TIAGO_SIGN_FLIP_JOINT ? static_cast<T>(-1) : static_cast<T>(1);
        }

        template<class T>
        __device__ void buildSharedMappedDynamicsVectors(T* s_mapped, const T* s_q, const T* s_qd, const T* s_u)
        {
                for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < grid::NUM_JOINTS; i += blockDim.x * blockDim.y) {
                        // GRiD only accepts positive principal axes. The generated Tiago
                        // URDF therefore flips arm_right_6_joint from axis -Z to +Z. Everywhere
                        // outside this wrapper remains in native Tiago coordinates, so the GRiD
                        // coordinate is the native coordinate multiplied by tiagoJointSign(i).
                        const T sign = tiagoJointSign<T>(i);
                        s_mapped[i] = s_q[i] * sign;
                        s_mapped[grid::NUM_JOINTS + i] = s_qd[i] * sign;
                        s_mapped[2 * grid::NUM_JOINTS + i] = s_u[i] * sign;
                        s_mapped[3 * grid::NUM_JOINTS + i] = static_cast<T>(0);
                }
                __syncthreads();
        }

        template<class T>
        __device__ void writeSharedMappedQddToTiago(T* s_qdd, const T* s_mapped)
        {
                for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < grid::NUM_JOINTS; i += blockDim.x * blockDim.y) {
                        // qdd is a vector in joint-coordinate space, so it crosses the same
                        // convention boundary as q, qd, and u.
                        s_qdd[i] = s_mapped[3 * grid::NUM_JOINTS + i] * tiagoJointSign<T>(i);
                }
                __syncthreads();
        }

        template<class T>
        __device__ void assertNoExternalWrench(const T* d_f_ext)
        {
                if (d_f_ext == nullptr) {
                        __syncthreads();
                        return;
                }

                for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < 6; i += blockDim.x * blockDim.y) {
                        assert(d_f_ext[i] == static_cast<T>(0) && "Tiago external wrench dynamics are not implemented");
                }
                __syncthreads();
        }

        template<class T>
        __device__ void remapForwardDynamicsJacobian(T* s_df_du, bool include_du)
        {
                for (int ind = threadIdx.x + threadIdx.y * blockDim.x; ind < 2 * grid::NUM_JOINTS * grid::NUM_JOINTS; ind += blockDim.x * blockDim.y) {
                        const int row = ind % grid::NUM_JOINTS;
                        const int col = ind / grid::NUM_JOINTS;
                        T sign = tiagoJointSign<T>(row);
                        if (col == TIAGO_SIGN_FLIP_JOINT || col == grid::NUM_JOINTS + TIAGO_SIGN_FLIP_JOINT) {
                                sign *= static_cast<T>(-1);
                        }
                        s_df_du[ind] *= sign;
                }

                if (!include_du) {
                        __syncthreads();
                        return;
                }

                for (int ind = threadIdx.x + threadIdx.y * blockDim.x; ind < grid::NUM_JOINTS * grid::NUM_JOINTS; ind += blockDim.x * blockDim.y) {
                        const int row = ind % grid::NUM_JOINTS;
                        const int col = ind / grid::NUM_JOINTS;
                        T sign = tiagoJointSign<T>(row);
                        if (col == TIAGO_SIGN_FLIP_JOINT) { sign *= static_cast<T>(-1); }
                        s_df_du[ind + 2 * grid::NUM_JOINTS * grid::NUM_JOINTS] *= sign;
                }
                __syncthreads();
        }

        template<class T>
        __device__ void computeTiagoToolPose(T* s_eePos, const T* s_q_grid, const grid::robotModel<T>* d_robotModel, T* s_workspace)
        {
                T* s_XmatsHom = s_workspace;
                T* s_temp = s_XmatsHom + 128;
                grid::load_update_XmatsHom_helpers<T>(s_XmatsHom, s_q_grid, d_robotModel, s_temp);
                grid::end_effector_pose_inner_arm_right_tool_joint<T>(s_eePos, s_q_grid, s_XmatsHom, s_temp);
        }

        template<class T>
        __device__ void computeTiagoToolPoseAndGradient(T* s_eePos, T* s_eePos_grad, const T* s_q_grid, const grid::robotModel<T>* d_robotModel, T* s_workspace)
        {
                T* s_XmatsHom = s_workspace;
                T* s_dXmatsHom = s_XmatsHom + 128;
                T* s_temp = s_dXmatsHom + 112;
                grid::load_update_XmatsHom_helpers<T>(s_XmatsHom, s_dXmatsHom, s_q_grid, d_robotModel, s_temp);
                grid::end_effector_pose_inner_arm_right_tool_joint<T>(s_eePos, s_q_grid, s_XmatsHom, s_temp);
                grid::end_effector_pose_gradient_inner_arm_right_tool_joint<T>(s_eePos_grad, s_q_grid, s_XmatsHom, s_dXmatsHom, s_temp);
        }

        template<typename T>
        void* initializeDynamicsConstMem()
        {
                grid::robotModel<T>* d_robotModel = grid::init_robotModel<T>();
                return (void*)d_robotModel;
        }

        template<typename T>
        void freeDynamicsConstMem(void* d_dynMem_const)
        {
                grid::robotModel<T> h_robotModel;
                gpuErrchk(cudaMemcpy(&h_robotModel, d_dynMem_const, sizeof(grid::robotModel<T>), cudaMemcpyDeviceToHost));
                gpuErrchk(cudaFree(h_robotModel.d_XImats));
                gpuErrchk(cudaFree(h_robotModel.d_topology_helpers));
                gpuErrchk(cudaFree(d_dynMem_const));
        }

        template<class T>
        __device__ T jointBarrier(T q, T q_min, T q_max)
        {
                T dist_min = q - q_min;
                T dist_max = q_max - q;
                dist_min = (dist_min <= 1e-10) ? 1e-10 : dist_min;
                dist_max = (dist_max <= 1e-10) ? 1e-10 : dist_max;
                return -log(dist_min) - log(dist_max);
        }

        template<class T>
        __device__ T jointBarrierGradient(T q, T q_min, T q_max)
        {
                T dist_min = q - q_min;
                T dist_max = q_max - q;
                const T eps = static_cast<T>(1e-6);

                if (dist_min >= static_cast<T>(0)) {
                        if (dist_min < eps) dist_min = eps;
                } else {
                        if (dist_min > -eps) dist_min = -eps;
                }

                if (dist_max >= static_cast<T>(0)) {
                        if (dist_max < eps) dist_max = eps;
                } else {
                        if (dist_max > -eps) dist_max = -eps;
                }

                return (-static_cast<T>(1) / dist_min) + (static_cast<T>(1) / dist_max);
        }

        template<class T>
        __device__ T jointBarrierHessian(T q, T q_min, T q_max)
        {
                T dist_min = q - q_min;
                T dist_max = q_max - q;
                const T eps = static_cast<T>(1e-6);

                T abs_min = dist_min >= static_cast<T>(0) ? dist_min : -dist_min;
                T abs_max = dist_max >= static_cast<T>(0) ? dist_max : -dist_max;
                if (abs_min < eps) abs_min = eps;
                if (abs_max < eps) abs_max = eps;

                return static_cast<T>(1.0) / (abs_min * abs_min) + static_cast<T>(1.0) / (abs_max * abs_max);
        }

        template<typename T>
        __device__ void forwardDynamics(T* s_qdd, T* s_q, T* s_qd, T* s_u, T* s_XITemp, void* d_dynMem_const)
        {
                __shared__ T s_mapped[4 * grid::NUM_JOINTS];
                T* q_grid = s_mapped;
                T* qd_grid = &s_mapped[grid::NUM_JOINTS];
                T* u_grid = &s_mapped[2 * grid::NUM_JOINTS];
                T* qdd_grid = &s_mapped[3 * grid::NUM_JOINTS];
                buildSharedMappedDynamicsVectors<T>(s_mapped, s_q, s_qd, s_u);

                T* s_XImats = s_XITemp;
                T* s_temp = &s_XITemp[72 * grid::NUM_JOINTS];
                grid::load_update_XImats_helpers<T>(s_XImats, q_grid, (grid::robotModel<T>*)d_dynMem_const, s_temp);
                __syncthreads();

                grid::forward_dynamics_inner<T>(qdd_grid, q_grid, qd_grid, u_grid, s_XImats, s_temp, gato::plant::GRAVITY<T>());
                writeSharedMappedQddToTiago<T>(s_qdd, s_mapped);
        }

        template<typename T>
        __device__ void forwardDynamics(T* s_qdd, T* s_q, T* s_qd, T* s_u, T* s_XITemp, void* d_dynMem_const, T* d_f_ext)
        {
                // The existing plant interface includes an external-wrench overload.
                // This GRiD Tiago header was generated without external-force
                // kernels, so fail if callers try to use a nonzero wrench.
                assertNoExternalWrench<T>(d_f_ext);
                forwardDynamics<T>(s_qdd, s_q, s_qd, s_u, s_XITemp, d_dynMem_const);
        }

        __host__ __device__ constexpr unsigned forwardDynamics_TempMemSize_Shared()
        {
                return grid::FD_DYNAMIC_SHARED_MEM_COUNT;
        }

        template<typename T, bool INCLUDE_DU = true>
        __device__ void forwardDynamicsAndGradient(T* s_df_du, T* s_qdd, const T* s_q, const T* s_qd, const T* s_u, T* s_temp_in, void* d_dynMem_const)
        {
                __shared__ T s_mapped[4 * grid::NUM_JOINTS];
                T* q_grid = s_mapped;
                T* qd_grid = &s_mapped[grid::NUM_JOINTS];
                T* u_grid = &s_mapped[2 * grid::NUM_JOINTS];
                T* qdd_grid = &s_mapped[3 * grid::NUM_JOINTS];
                buildSharedMappedDynamicsVectors<T>(s_mapped, s_q, s_qd, s_u);

                T* s_XITemp = s_temp_in;
                grid::robotModel<T>* d_robotModel = (grid::robotModel<T>*)d_dynMem_const;

                T* s_XImats = s_XITemp;
                T* s_vaf = &s_XITemp[72 * grid::NUM_JOINTS];
                T* s_dc_du = &s_vaf[18 * grid::NUM_JOINTS];
                T* s_Minv = &s_dc_du[2 * grid::NUM_JOINTS * grid::NUM_JOINTS];
                T* s_temp = &s_Minv[grid::NUM_JOINTS * grid::NUM_JOINTS];

                grid::load_update_XImats_helpers<T>(s_XImats, q_grid, d_robotModel, s_temp);
                grid::direct_minv_inner<T>(s_Minv, q_grid, s_XImats, s_temp);
                T* s_c = s_temp;
                grid::inverse_dynamics_inner<T>(s_c, s_vaf, q_grid, qd_grid, s_XImats, &s_temp[6], GRAVITY<T>());
                grid::forward_dynamics_finish<T>(qdd_grid, u_grid, s_c, s_Minv);
                grid::inverse_dynamics_inner_vaf<T>(s_vaf, q_grid, qd_grid, qdd_grid, s_XImats, s_temp, GRAVITY<T>());
                grid::inverse_dynamics_gradient_inner<T>(s_dc_du, q_grid, qd_grid, s_vaf, s_XImats, s_temp, GRAVITY<T>());

                for (int ind = threadIdx.x + threadIdx.y * blockDim.x; ind < 2 * grid::NUM_JOINTS * grid::NUM_JOINTS; ind += blockDim.x * blockDim.y) {
                        const int row = ind % grid::NUM_JOINTS;
                        const int dc_col_offset = ind - row;
                        T val = static_cast<T>(0);
                        for (int col = 0; col < grid::NUM_JOINTS; col++) {
                                const int index = (row <= col) * (col * grid::NUM_JOINTS + row) + (row > col) * (row * grid::NUM_JOINTS + col);
                                val += s_Minv[index] * s_dc_du[dc_col_offset + col];
                        }
                        s_df_du[ind] = -val;

                        if (INCLUDE_DU && ind < grid::NUM_JOINTS * grid::NUM_JOINTS) {
                                const int col = ind / grid::NUM_JOINTS;
                                const int index = (row <= col) * (col * grid::NUM_JOINTS + row) + (row > col) * (row * grid::NUM_JOINTS + col);
                                s_df_du[ind + 2 * grid::NUM_JOINTS * grid::NUM_JOINTS] = s_Minv[index];
                        }
                }
                __syncthreads();

                remapForwardDynamicsJacobian<T>(s_df_du, INCLUDE_DU);
                writeSharedMappedQddToTiago<T>(s_qdd, s_mapped);
        }

        template<typename T, bool INCLUDE_DU = true>
        __device__ void forwardDynamicsAndGradient(T* s_df_du, T* s_qdd, const T* s_q, const T* s_qd, const T* s_u, T* s_temp_in, void* d_dynMem_const, T* d_f_ext)
        {
                // See the forwardDynamics external-wrench overload above. Keep this
                // overload so generic solver code can compile for Tiago, but compute
                // the no-external-force dynamics that GRiD generated.
                assertNoExternalWrench<T>(d_f_ext);
                forwardDynamicsAndGradient<T, INCLUDE_DU>(s_df_du, s_qdd, s_q, s_qd, s_u, s_temp_in, d_dynMem_const);
        }

        __host__ __device__ constexpr unsigned forwardDynamicsAndGradient_TempMemSize_Shared()
        {
                return grid::FD_DU_MAX_SHARED_MEM_COUNT;
        }

        template<typename T>
        __device__ T trackingcost(uint32_t state_size,
                                  uint32_t control_size,
                                  uint32_t knot_points,
                                  T* s_xu,
                                  T* s_eePos_traj,
                                  T* s_temp,
                                  const grid::robotModel<T>* d_robotModel,
                                  T q_cost,
                                  T qd_cost,
                                  T u_cost,
                                  T N_cost,
                                  T q_lim_cost,
                                  T vel_lim_cost,
                                  T ctrl_lim_cost)
        {
                __shared__ T s_q_grid[grid::NUM_JOINTS];
                for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < grid::NUM_JOINTS; i += blockDim.x * blockDim.y) {
                        s_q_grid[i] = s_xu[i] * tiagoJointSign<T>(i);
                }
                __syncthreads();

                T err;
                const uint32_t threadsNeeded = state_size / 2 + control_size * (blockIdx.x < knot_points - 1);

                T* s_cost_vec = s_temp;
                T* s_eePos_cost = s_cost_vec + threadsNeeded + 3;
                T* s_ee_workspace = s_eePos_cost + grid::EE_POS_SIZE;
                computeTiagoToolPose<T>(s_eePos_cost, s_q_grid, d_robotModel, s_ee_workspace);

                for (int i = threadIdx.x; i < threadsNeeded; i += blockDim.x) {
                        if (i < state_size / 2) {
                                err = s_xu[i + state_size / 2];
                                s_cost_vec[i] = static_cast<T>(0.5) * qd_cost * err * err;
                                s_cost_vec[i] += q_lim_cost * jointBarrier(s_xu[i], JOINT_LIMITS<T>()[i][0], JOINT_LIMITS<T>()[i][1]);
                                s_cost_vec[i] += vel_lim_cost * jointBarrier(s_xu[i + state_size / 2], VEL_LIMITS<T>()[i][0], VEL_LIMITS<T>()[i][1]);
                        } else {
                                err = s_xu[i + state_size / 2];
                                s_cost_vec[i] = static_cast<T>(0.5) * u_cost * err * err;
                                s_cost_vec[i] += ctrl_lim_cost * jointBarrier(s_xu[i + state_size / 2], CTRL_LIMITS<T>()[i - state_size / 2][0], CTRL_LIMITS<T>()[i - state_size / 2][1]);
                        }
                }
                for (int i = threadIdx.x; i < 3; i += blockDim.x) {
                        err = s_eePos_cost[i] - s_eePos_traj[i];
                        s_cost_vec[threadsNeeded + i] = static_cast<T>(0.5) * (blockIdx.x == KNOT_POINTS - 1 ? N_cost : q_cost) * err * err;
                }
                __syncthreads();

                block::reduce<T>(threadsNeeded + 3, s_cost_vec);
                __syncthreads();

                return s_cost_vec[0];
        }

        __host__ unsigned trackingcost_TempMemCt_Shared(uint32_t state_size, uint32_t control_size, uint32_t knot_points)
        {
                (void)knot_points;
                return state_size / 2 + control_size + 3 + grid::EE_POS_SIZE + grid::EE_POS_DYNAMIC_SHARED_MEM_COUNT;
        }

        template<typename T, bool computeR = true>
        __device__ void trackingCostGradientAndHessian(uint32_t state_size,
                                                       uint32_t control_size,
                                                       T* s_xu,
                                                       T* s_eePos_traj,
                                                       T* s_Qk,
                                                       T* s_qk,
                                                       T* s_Rk,
                                                       T* s_rk,
                                                       T* s_temp,
                                                       void* d_dynMem_const,
                                                       T q_cost,
                                                       T qd_cost,
                                                       T u_cost,
                                                       T N_cost,
                                                       T q_lim_cost,
                                                       T vel_lim_cost,
                                                       T ctrl_lim_cost)
        {
                __shared__ T s_q_grid[grid::NUM_JOINTS];
                for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < grid::NUM_JOINTS; i += blockDim.x * blockDim.y) {
                        s_q_grid[i] = s_xu[i] * tiagoJointSign<T>(i);
                }
                __syncthreads();

                const grid::robotModel<T>* d_robotModel = static_cast<const grid::robotModel<T>*>(d_dynMem_const);
                T* s_eePos = s_temp;
                T* s_eePos_grad = s_eePos + grid::EE_POS_SIZE;
                T* s_ee_workspace = s_eePos_grad + 6 * grid::NUM_JOINTS;
                const uint32_t threads_needed = state_size + control_size * computeR;

                computeTiagoToolPoseAndGradient<T>(s_eePos, s_eePos_grad, s_q_grid, d_robotModel, s_ee_workspace);

                for (int i = threadIdx.x; i < threads_needed; i += blockDim.x) {
                        if (i < state_size) {
                                if (i < grid::NUM_JOINTS) {
                                        const T grad_x = s_eePos_grad[6 * i + 0] * tiagoJointSign<T>(i);
                                        const T grad_y = s_eePos_grad[6 * i + 1] * tiagoJointSign<T>(i);
                                        const T grad_z = s_eePos_grad[6 * i + 2] * tiagoJointSign<T>(i);
                                        s_qk[i] = (grad_x * (s_eePos[0] - s_eePos_traj[0]) + grad_y * (s_eePos[1] - s_eePos_traj[1]) + grad_z * (s_eePos[2] - s_eePos_traj[2]))
                                                  * (blockIdx.x == KNOT_POINTS - 1 ? N_cost : q_cost);
                                        s_qk[i] += q_lim_cost * jointBarrierGradient(s_xu[i], JOINT_LIMITS<T>()[i][0], JOINT_LIMITS<T>()[i][1]);
                                } else {
                                        s_qk[i] = qd_cost * s_xu[i];
                                        s_qk[i] += vel_lim_cost * jointBarrierGradient(s_xu[i], VEL_LIMITS<T>()[i - grid::NUM_JOINTS][0], VEL_LIMITS<T>()[i - grid::NUM_JOINTS][1]);
                                }
                        } else {
                                s_rk[i - state_size] = u_cost * s_xu[i];
                                s_rk[i - state_size] += ctrl_lim_cost * jointBarrierGradient(s_xu[i], CTRL_LIMITS<T>()[i - state_size][0], CTRL_LIMITS<T>()[i - state_size][1]);
                        }
                }
                __syncthreads();

                for (int i = threadIdx.x; i < threads_needed; i += blockDim.x) {
                        if (i < state_size) {
                                for (int j = 0; j < state_size; j++) {
                                        if (j < grid::NUM_JOINTS && i < grid::NUM_JOINTS) {
                                                const T grad_ix = s_eePos_grad[6 * i + 0] * tiagoJointSign<T>(i);
                                                const T grad_iy = s_eePos_grad[6 * i + 1] * tiagoJointSign<T>(i);
                                                const T grad_iz = s_eePos_grad[6 * i + 2] * tiagoJointSign<T>(i);
                                                const T grad_jx = s_eePos_grad[6 * j + 0] * tiagoJointSign<T>(j);
                                                const T grad_jy = s_eePos_grad[6 * j + 1] * tiagoJointSign<T>(j);
                                                const T grad_jz = s_eePos_grad[6 * j + 2] * tiagoJointSign<T>(j);
                                                s_Qk[i * state_size + j] = (grad_ix * grad_jx + grad_iy * grad_jy + grad_iz * grad_jz) * (blockIdx.x == KNOT_POINTS - 1 ? N_cost : q_cost);
                                                if (i == j) {
                                                        s_Qk[i * state_size + j] += q_lim_cost * jointBarrierHessian<T>(s_xu[i], JOINT_LIMITS<T>()[i][0], JOINT_LIMITS<T>()[i][1]);
                                                }
                                        } else {
                                                s_Qk[i * state_size + j] = (i == j) ? qd_cost : static_cast<T>(0);
                                                if (i == j) {
                                                        s_Qk[i * state_size + j] += vel_lim_cost * jointBarrierHessian<T>(s_xu[i], VEL_LIMITS<T>()[i - grid::NUM_JOINTS][0], VEL_LIMITS<T>()[i - grid::NUM_JOINTS][1]);
                                                }
                                        }
                                }
                        } else {
                                const uint32_t offset = i - state_size;
                                for (int j = 0; j < control_size; j++) {
                                        s_Rk[offset * control_size + j] = (offset == static_cast<uint32_t>(j)) ? u_cost : static_cast<T>(0);
                                        if (offset == static_cast<uint32_t>(j)) {
                                                s_Rk[offset * control_size + j] += ctrl_lim_cost * jointBarrierHessian<T>(s_xu[i], CTRL_LIMITS<T>()[offset][0], CTRL_LIMITS<T>()[offset][1]);
                                        }
                                }
                        }
                }
                __syncthreads();
        }

        template<typename T>
        __device__ void trackingCostGradientAndHessian_lastblock(uint32_t state_size,
                                                                 uint32_t control_size,
                                                                 T* s_xux,
                                                                 T* s_eePos_traj,
                                                                 T* s_Qk,
                                                                 T* s_qk,
                                                                 T* s_Rk,
                                                                 T* s_rk,
                                                                 T* s_Qkp1,
                                                                 T* s_qkp1,
                                                                 T* s_temp,
                                                                 void* d_dynMem_const,
                                                                 T q_cost,
                                                                 T qd_cost,
                                                                 T u_cost,
                                                                 T N_cost,
                                                                 T q_lim_cost,
                                                                 T vel_lim_cost,
                                                                 T ctrl_lim_cost)
        {
                trackingCostGradientAndHessian<T>(state_size, control_size, s_xux, s_eePos_traj, s_Qk, s_qk, s_Rk, s_rk, s_temp, d_dynMem_const, q_cost, qd_cost, u_cost, N_cost, q_lim_cost, vel_lim_cost, ctrl_lim_cost);
                trackingCostGradientAndHessian<T, false>(state_size, control_size, s_xux, &s_eePos_traj[grid::EE_POS_SIZE], s_Qkp1, s_qkp1, nullptr, nullptr, s_temp, d_dynMem_const, q_cost, qd_cost, u_cost, N_cost, q_lim_cost, vel_lim_cost, ctrl_lim_cost);
        }

        __host__ __device__ constexpr unsigned trackingCostGradientAndHessian_TempMemSize_Shared()
        {
                return grid::EE_POS_SIZE + 6 * grid::NUM_JOINTS + grid::DEE_POS_DYNAMIC_SHARED_MEM_COUNT;
        }

}  // namespace plant
}  // namespace gato
