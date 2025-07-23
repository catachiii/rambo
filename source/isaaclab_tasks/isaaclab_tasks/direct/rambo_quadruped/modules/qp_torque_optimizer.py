import torch
import numpy as np
import omni.kit.app
import weakref
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG

from qpth.qp import QPFunction, QPSolvers

from ..utils.helper import to_torch


class QPTorqueOptimizer:
    """Centroidal QP controller to optimize for joint torques."""

    def __init__(self, env):
        """Initializes the controller with desired weights and gains."""
        self._env = env

        self._device = self._env.device
        self._num_envs = self._env.num_envs

        self._base_position_kp = torch.stack([torch.tensor(self._env.cfg.qp_torque_optimizer_config["base_position_kp"],
                                                           device=self._device, dtype=torch.float32)] * self._num_envs,
                                             dim=0)
        self._base_position_kd = torch.stack([torch.tensor(self._env.cfg.qp_torque_optimizer_config["base_position_kd"],
                                                           device=self._device, dtype=torch.float32)] * self._num_envs,
                                             dim=0)
        self._base_orientation_kp = torch.stack(
            [torch.tensor(self._env.cfg.qp_torque_optimizer_config["base_orientation_kp"],
                          device=self._device,
                          dtype=torch.float32)] * self._num_envs, dim=0)
        self._base_orientation_kd = torch.stack(
            [torch.tensor(self._env.cfg.qp_torque_optimizer_config["base_orientation_kd"],
                          device=self._device,
                          dtype=torch.float32)] * self._num_envs, dim=0)
        self._desired_base_position = torch.zeros((self._num_envs, 3), device=self._device)
        self._desired_base_orientation_quat = torch.zeros((self._num_envs, 4), device=self._device)
        self._desired_linear_velocity = torch.zeros((self._num_envs, 3), device=self._device)
        self._desired_angular_velocity = torch.zeros((self._num_envs, 3), device=self._device)
        self._desired_linear_acceleration = torch.zeros((self._num_envs, 3), device=self._device)
        self._desired_angular_acceleration = torch.zeros((self._num_envs, 3), device=self._device)

        self._Wq = torch.tensor(self._env.cfg.qp_torque_optimizer_config["qp_weight_ddq"], device=self._device,
                                dtype=torch.float32)
        self._Wf = torch.tensor(self._env.cfg.qp_torque_optimizer_config["qp_weight_grf"], device=self._device,
                                dtype=torch.float32)
        self._Wfe = torch.tensor(self._env.cfg.qp_torque_optimizer_config["qp_weight_ee_force"], device=self._device,
                                 dtype=torch.float32)
        self._foot_friction_coef = self._env.cfg.qp_torque_optimizer_config["qp_foot_friction_coef"]

        self._inv_mass = torch.eye(3, device=self._device).repeat(self._num_envs, 1, 1) / 16.0870

        self._inv_inertia = torch.linalg.inv(
            torch.diag(to_torch(np.array([0.14, 0.35, 0.35]) * 1.5, device=self._device))).unsqueeze(0).repeat(
            self._num_envs, 1, 1)  # assume single rigid body

        # for debug visualization purpose
        self.grf = torch.zeros((self._num_envs, self._env.num_feet * 3), device=self._device)
        self.desired_acc = torch.zeros((self._num_envs, 6), device=self._device)
        self.debug_vis = self._env.cfg.qp_torque_optimizer_config["qp_debug_vis"]
        self.grf_vis_handle = None
        self._set_grf_vis(self.debug_vis)

    def get_grf(self):
        # get current quantities from the environment
        # all quantities are expressed in the projected com frame
        curr_base_pos = torch.zeros((self._num_envs, 3), device=self._device)
        curr_base_pos[:, 2] = self._env.base_height
        curr_base_quat = self._env.base_quat_rp
        curr_base_lin_vel = math_utils.quat_apply(curr_base_quat, self._env.base_lin_vel_b)
        curr_base_ang_vel = math_utils.quat_apply(curr_base_quat, self._env.base_ang_vel_b)

        self.desired_acc = compute_desired_acc(
            curr_base_pos,
            curr_base_quat,
            curr_base_lin_vel,
            curr_base_ang_vel,
            self._desired_base_position,
            self._desired_base_orientation_quat,
            self._desired_linear_velocity,
            self._desired_angular_velocity,
            self._desired_linear_acceleration,
            self._desired_angular_acceleration,
            self._base_position_kp,
            self._base_position_kd,
            self._base_orientation_kp,
            self._base_orientation_kd)
        self.desired_acc = torch.clip(self.desired_acc,
                                      to_torch([-30, -30, -30, -20, -20, -20], device=self._device),
                                      to_torch([30, 30, 30, 20, 20, 20], device=self._device))

        desired_acc_b = self.desired_acc.clone()
        desired_acc_b[:, :3] = torch.matmul(self._env.base_rot_mat_rp_t, desired_acc_b[:, :3, None])[:, :, 0]
        desired_acc_b[:, 3:] = torch.matmul(self._env.base_rot_mat_rp_t, desired_acc_b[:, 3:, None])[:, :, 0]

        desired_ee_force_fl_com = self._env._ee_force_commands
        desired_ee_force_fl_b = torch.matmul(self._env.base_rot_mat_rp_t, desired_ee_force_fl_com[:, :, None])[:, :, 0]

        # construct mass matrix
        # (linear 3 + angular 3) x (FL_hip, FL_thigh, FL_calf, FR, ..., RL, ..., RR, ...)
        mass_mat = construct_mass_mat(
            self._env.ee_pos_b,
            self._inv_mass,
            self._inv_inertia,
            device=self._device)

        # Solve QP
        if self._env.cfg.use_actual_contact:
            contact = self._env.foot_contacts
        else:
            contact = self._env.contact_generator.desired_contact_state

        self.grf, solved_acc, qp_cost = solve_grf_qpth(
            mass_mat,
            desired_acc_b,
            desired_ee_force_fl_b,
            self._Wq,
            self._Wf,
            self._Wfe,
            self._env.base_rot_mat_rp,  # used to calculate gravity_in_base only
            self._foot_friction_coef,
            contact,
            device=self._device)

        return self.grf, qp_cost, desired_acc_b, solved_acc

    @property
    def desired_base_position(self) -> torch.Tensor:
        return self._desired_base_position

    @desired_base_position.setter
    def desired_base_position(self, base_position: torch.Tensor):
        self._desired_base_position = to_torch(base_position, device=self._device)

    @property
    def desired_base_orientation_quat(self) -> torch.Tensor:
        return self._desired_base_orientation_quat

    @desired_base_orientation_quat.setter
    def desired_base_orientation_quat(self, orientation_quat: torch.Tensor):
        self._desired_base_orientation_quat = to_torch(orientation_quat, device=self._device)

    @property
    def desired_linear_velocity(self) -> torch.Tensor:
        return self._desired_linear_velocity

    @desired_linear_velocity.setter
    def desired_linear_velocity(self, desired_linear_velocity: torch.Tensor):
        self._desired_linear_velocity = to_torch(desired_linear_velocity, device=self._device)

    @property
    def desired_angular_velocity(self) -> torch.Tensor:
        return self._desired_angular_velocity

    @desired_angular_velocity.setter
    def desired_angular_velocity(self, desired_angular_velocity: torch.Tensor):
        self._desired_angular_velocity = to_torch(desired_angular_velocity, device=self._device)

    @property
    def desired_linear_acceleration(self):
        return self._desired_linear_acceleration

    @desired_linear_acceleration.setter
    def desired_linear_acceleration(self, desired_linear_acceleration: torch.Tensor):
        self._desired_linear_acceleration = to_torch(desired_linear_acceleration, device=self._device)

    @property
    def desired_angular_acceleration(self):
        return self._desired_angular_acceleration

    @desired_angular_acceleration.setter
    def desired_angular_acceleration(self, desired_angular_acceleration: torch.Tensor):
        self._desired_angular_acceleration = to_torch(desired_angular_acceleration, device=self._device)

    def _set_grf_vis(self, debug_vis):
        self._set_grf_vis_impl(debug_vis)
        if debug_vis:
            if self.grf_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self.grf_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._vis_callback(event)
                )
        else:
            if self.grf_vis_handle is not None:
                self.grf_vis_handle.unsubscribe()
                self.grf_vis_handle = None
        return True

    def _set_grf_vis_impl(self, debug_vis):
        if debug_vis:
            if not hasattr(self, "grf_visualizer"):
                self.grf_visualizer = []
                for i in range(self._env.num_feet):
                    marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                    marker_cfg.prim_path = "/Visuals/grf_" + str(i)
                    marker_cfg.markers["arrow"].scale = (0.2, 0.2, 0.2)
                    self.grf_visualizer.append(VisualizationMarkers(marker_cfg))

                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/desired_lin_acc"
                marker_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
                self.desired_lin_acc_visualizer = VisualizationMarkers(marker_cfg)

                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/desired_ang_acc"
                marker_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
                self.desired_ang_acc_visualizer = VisualizationMarkers(marker_cfg)

            for i in range(self._env.num_feet):
                self.grf_visualizer[i].set_visibility(True)
            self.desired_lin_acc_visualizer.set_visibility(True)
            self.desired_ang_acc_visualizer.set_visibility(True)
        else:
            if hasattr(self, "grf_visualizer"):
                for i in range(self._env.num_feet):
                    self.grf_visualizer[i].set_visibility(False)
                self.desired_lin_acc_visualizer.set_visibility(False)
                self.desired_ang_acc_visualizer.set_visibility(False)

    def _vis_callback(self, event):
        grf_w = self.grf.reshape((-1, 4, 3)).clone()
        grf_w = torch.matmul(math_utils.matrix_from_quat(self._env._robot.data.root_quat_w),
                             grf_w.transpose(1, 2)).transpose(1, 2)
        ee_pos_w = self._env._robot.data.body_state_w[:, self._env.feet_ids, 0:3].clone()
        for i in range(4):
            pos_w = ee_pos_w[:, i].clone()
            scale, quat = self._resolve_scale_and_quat_from_vector(
                self.grf_visualizer[i].cfg.markers["arrow"].scale,
                grf_w[:, i])
            # contact_state = self._env.foot_contacts[:, i]
            # scale *= contact_state.unsqueeze(-1)
            self.grf_visualizer[i].visualize(pos_w, quat, scale)

        pos_acc = self._env._robot.data.root_pos_w.clone()
        # pos_acc[:, 2] += 0.5
        lin_acc_w = self.desired_acc[:, :3]
        lin_acc_w = torch.matmul(math_utils.matrix_from_quat(math_utils.yaw_quat(self._env._robot.data.root_quat_w)),
                                 lin_acc_w.unsqueeze(-1)).squeeze(-1)
        scale, quat = self._resolve_scale_and_quat_from_vector(
            self.desired_lin_acc_visualizer.cfg.markers["arrow"].scale,
            lin_acc_w)
        self.desired_lin_acc_visualizer.visualize(pos_acc, quat, scale)
        ang_acc_w = self.desired_acc[:, 3:]
        ang_acc_w = torch.matmul(math_utils.matrix_from_quat(math_utils.yaw_quat(self._env._robot.data.root_quat_w)),
                                 ang_acc_w.unsqueeze(-1)).squeeze(-1)
        scale, quat = self._resolve_scale_and_quat_from_vector(
            self.desired_ang_acc_visualizer.cfg.markers["arrow"].scale,
            ang_acc_w)
        self.desired_ang_acc_visualizer.visualize(pos_acc, quat, scale)

    def _resolve_scale_and_quat_from_vector(self, default_scale, vector):
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=vector.device).repeat(vector.shape[0], 1)
        arrow_scale[:, 0] *= torch.norm(vector, dim=-1) * 0.4

        directions = torch.nn.functional.normalize(vector)
        dir_x = directions[:, 0]
        dir_y = directions[:, 1]
        dir_z = directions[:, 2]

        yaw = torch.atan2(dir_y, dir_x)
        pitch = torch.atan2(-dir_z, torch.sqrt(dir_x ** 2 + dir_y ** 2))
        roll = torch.zeros_like(yaw)
        quaternions = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

        return arrow_scale, quaternions


@torch.jit.script
def compute_desired_acc(
        base_position: torch.Tensor,
        base_orientation_quat: torch.Tensor,
        base_velocity: torch.Tensor,
        base_angular_velocity: torch.Tensor,
        desired_base_position: torch.Tensor,
        desired_base_orientation_quat: torch.Tensor,
        desired_linear_velocity: torch.Tensor,
        desired_angular_velocity: torch.Tensor,
        desired_linear_acceleration: torch.Tensor,
        desired_angular_acceleration: torch.Tensor,
        base_position_kp: torch.Tensor,
        base_position_kd: torch.Tensor,
        base_orientation_kp: torch.Tensor,
        base_orientation_kd: torch.Tensor,
):
    # all error terms are expressed in the projected com frame
    lin_pos_error = desired_base_position - base_position
    lin_vel_error = desired_linear_velocity - base_velocity
    desired_lin_acc = (base_position_kp * lin_pos_error +
                       base_position_kd * lin_vel_error +
                       desired_linear_acceleration)

    ang_pos_error = math_utils.quat_error(desired_base_orientation_quat, base_orientation_quat)
    ang_vel_error = desired_angular_velocity - base_angular_velocity
    desired_ang_acc = (base_orientation_kp * ang_pos_error +
                       base_orientation_kd * ang_vel_error +
                       desired_angular_acceleration)
    return torch.concatenate((desired_lin_acc, desired_ang_acc), dim=1)


@torch.jit.script
def convert_to_skew_symmetric_batch(foot_positions):
    """
    Converts foot positions (nx4x3) into skew-symmetric ones (nx3x12)
    """
    n = foot_positions.shape[0]
    x = foot_positions[:, :, 0]
    y = foot_positions[:, :, 1]
    z = foot_positions[:, :, 2]
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=1).reshape(
        (n, 3, 3, 4))
    # FL, FR, RL, RR
    return torch.concatenate([skew[:, :, :, 0], skew[:, :, :, 1], skew[:, :, :, 2], skew[:, :, :, 3]], dim=2)


@torch.jit.script
def construct_mass_mat(foot_positions,
                       inv_mass,
                       inv_inertia,
                       device: str = 'cuda'):
    num_envs = foot_positions.shape[0]
    # (linear 3 + angular 3) x (FL_hip, FL_thigh, FL_calf, FR, ..., RL, ..., RR, ...)
    mass_mat = torch.zeros((num_envs, 6, 12), device=device)
    inv_mass_concat = torch.concatenate([inv_mass] * 4, dim=2)
    mass_mat[:, :3] = inv_mass_concat
    px = convert_to_skew_symmetric_batch(foot_positions)
    mass_mat[:, 3:6] = torch.matmul(inv_inertia, px)
    return mass_mat


def solve_grf_qpth(mass_mat,
                   desired_acc,
                   desired_ee_force_fl,
                   Wq,
                   Wf,
                   Wfe,
                   base_rot_mat_rp,
                   foot_friction_coef: float,
                   foot_contact_state,
                   device: str = 'cuda'):
    # QP is solved in the body frame
    desired_ee_reaction_force = -desired_ee_force_fl

    base_rot_mat_rp_t = torch.transpose(base_rot_mat_rp, 1, 2)
    num_envs = mass_mat.shape[0]
    g = torch.zeros((num_envs, 6), device=device)
    g[:, 2] = -9.8

    g[:, :3] = torch.matmul(base_rot_mat_rp_t, g[:, :3, None])[:, :, 0]
    Q = torch.zeros((num_envs, 6, 6), device=device) + Wq[None, :]

    S_fl = torch.zeros((num_envs, 3, 12), device=device)
    S_fl[:, :, 0:3] = torch.eye(3, device=device)

    S_r = torch.zeros((num_envs, 9, 12), device=device)  # S_rest
    S_r[:, :, 3:] = torch.eye(9, device=device)

    R_fl = torch.zeros((num_envs, 3, 3), device=device) + (torch.eye(3, device=device) * Wfe)[None, :]
    R_r = torch.zeros((num_envs, 9, 9), device=device) + (torch.eye(9, device=device) * Wf)[None, :]

    quad_term = torch.bmm(torch.bmm(torch.transpose(mass_mat, 1, 2), Q), mass_mat) + \
                torch.bmm(torch.bmm(torch.transpose(S_fl, 1, 2), R_fl), S_fl) + \
                torch.bmm(torch.bmm(torch.transpose(S_r, 1, 2), R_r), S_r)

    linear_term = torch.bmm(torch.bmm(torch.transpose(mass_mat, 1, 2), Q), (g - desired_acc)[:, :, None])[:, :, 0] - \
                  torch.bmm(torch.bmm(torch.transpose(S_fl, 1, 2), R_fl), desired_ee_reaction_force[:, :, None])[:, :, 0]

    G = torch.zeros((mass_mat.shape[0], 18, 12), device=device)
    h = torch.zeros((mass_mat.shape[0], 18), device=device) + 1e-3
    for leg_id in range(3):
        G[:, leg_id * 2, leg_id * 3 + 2 + 3] = 1
        G[:, leg_id * 2 + 1, leg_id * 3 + 2 + 3] = -1

        row_id, col_id = 6 + leg_id * 4, leg_id * 3 + 3
        G[:, row_id, col_id] = 1
        G[:, row_id, col_id + 2] = -foot_friction_coef

        G[:, row_id + 1, col_id] = -1
        G[:, row_id + 1, col_id + 2] = -foot_friction_coef

        G[:, row_id + 2, col_id + 1] = 1
        G[:, row_id + 2, col_id + 2] = -foot_friction_coef

        G[:, row_id + 3, col_id + 1] = -1
        G[:, row_id + 3, col_id + 2] = -foot_friction_coef
        G[:, row_id:row_id + 4, col_id:col_id + 3] = torch.bmm(
            G[:, row_id:row_id + 4, col_id:col_id + 3], base_rot_mat_rp)

    contact_ids = foot_contact_state[:, 1:].nonzero()

    h[contact_ids[:, 0], contact_ids[:, 1] * 2] = 130
    h[contact_ids[:, 0], contact_ids[:, 1] * 2 + 1] = -10
    e = torch.autograd.Variable(torch.Tensor())

    qf = QPFunction(verbose=-1,
                    check_Q_spd=False,
                    solver=QPSolvers.PDIPM_BATCHED)
    # since we don't check spd for Q matrix, we need to add a small value to the diagonal
    quad_term_psd = quad_term + 1e-6 * torch.eye(quad_term.shape[-1], device=device).repeat(quad_term.shape[0], 1,
                                                                                            1) * torch.rand(1,
                                                                                                            device=device)
    grf = qf(quad_term_psd.double(), linear_term.double(), G.double(), h.double(), e, e).float()
    solved_acc = torch.bmm(mass_mat, grf[:, :, None])[:, :, 0] - g
    qp_cost = torch.bmm(
        torch.bmm((solved_acc - desired_acc)[:, :, None].transpose(1, 2), Q),
        (solved_acc - desired_acc)[:, :, None])[:, 0, 0]
    # print(qp_cost)
    return grf, solved_acc, qp_cost
