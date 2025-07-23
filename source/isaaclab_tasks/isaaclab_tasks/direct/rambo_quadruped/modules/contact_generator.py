import torch
import numpy as np
import omni.kit.app
import weakref
import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


class ContactGenerator:
    def __init__(self, env):
        self._env = env
        self._contact_sequence = self._env.cfg.contact_generator_config["contact_sequence"]
        self._contact_feet = []
        self._contact_max_index = []
        self._contact_mode = {}
        self._contact_timing = {}
        self._contact_extra = {}
        for key, value in self._contact_sequence.items():
            self._contact_feet.append(key)
            self._contact_mode[key] = []
            self._contact_timing[key] = []
            self._contact_extra[key] = []
            self._contact_max_index.append(len(value))
            for i in range(len(value)):
                if value[i][0] in ["stance", "phase", "swing"]:
                    if value[i][0] == "stance":
                        self._contact_mode[key].append(1.0)
                    elif value[i][0] == "phase":
                        self._contact_mode[key].append(0.0)
                    elif value[i][0] == "swing":
                        self._contact_mode[key].append(-1.0)
                else:
                    raise ValueError("Invalid contact mode")
                self._contact_timing[key].append(value[i][1])
                self._contact_extra[key].append(value[i][2:])

        self._contact_mode = torch.tensor(np.array([*self._contact_mode.values()]), dtype=torch.float32,
                                          device=self._env.device)
        self._contact_timing = torch.tensor(np.array([*self._contact_timing.values()]), dtype=torch.float32,
                                            device=self._env.device)
        self._contact_timing_total = torch.sum(self._contact_timing, dim=1)
        self._contact_timing_percentage = self._contact_timing / self._contact_timing_total[:, None]
        self._contact_extra = torch.tensor(np.array([*self._contact_extra.values()]), dtype=torch.float32,
                                           device=self._env.device)
        self._contact_timing_cumsum = torch.cumsum(self._contact_timing, dim=1)
        self._contact_timing_cumsum_percentage = torch.cumsum(self._contact_timing_percentage, dim=1)
        self._contact_max_index = torch.tensor(self._contact_max_index, device=self._env.device)

        # buffers
        self._current_contact_mode = torch.ones(self._env.num_envs, self._env.num_feet, dtype=torch.float32,
                                                device=self._env.device)
        self._next_contact_mode = torch.ones(self._env.num_envs, self._env.num_feet, dtype=torch.float32,
                                             device=self._env.device)
        self._after_current_contact_mode = torch.zeros(self._env.num_envs, self._env.num_feet, dtype=torch.float32,
                                                       device=self._env.device)
        self._until_next_contact_mode = torch.zeros(self._env.num_envs, self._env.num_feet, dtype=torch.float32,
                                                    device=self._env.device)

        self._current_contact_timing = torch.zeros(self._env.num_envs, self._env.num_feet, dtype=torch.float32,
                                                   device=self._env.device)
        self._current_contact_extra = torch.zeros(self._env.num_envs, self._env.num_feet, 3, dtype=torch.float32,
                                                  device=self._env.device)

        self._current_contact_phase = torch.zeros(self._env.num_envs, self._env.num_feet, dtype=torch.float32,
                                                  device=self._env.device)
        self._current_contact_state = torch.ones(self._env.num_envs, self._env.num_feet, dtype=torch.bool,
                                                 device=self._env.device)

        self._time_since_reset = torch.zeros(self._env.num_envs, device=self._env.device, dtype=torch.float32)

        self.debug_vis = self._env.cfg.contact_generator_config["contact_generator_debug_vis"]
        self.contact_generator_vis_handle = None
        self._set_contact_generator_vis(self.debug_vis)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self._time_since_reset[env_ids] = self._env.time_since_reset[env_ids]

        current_mode_index = torch.sum(
            self._time_since_reset[env_ids, None, None] >= (self._contact_timing_cumsum[None, :] + 1e-4),
            dim=-1)

        previous_mode_index = current_mode_index - 1
        next_mode_index = torch.clip(current_mode_index + 1, max=self._contact_max_index[None, :] - 1)

        for i in range(self._env.num_feet):
            self._current_contact_mode[env_ids, i] = torch.index_select(self._contact_mode[i], 0,
                                                                        current_mode_index[:, i])
            self._current_contact_timing[env_ids, i] = torch.index_select(self._contact_timing[i], 0,
                                                                          current_mode_index[:, i])
            self._current_contact_extra[env_ids, i] = torch.index_select(self._contact_extra[i], 0,
                                                                         current_mode_index[:, i])
            self._next_contact_mode[env_ids, i] = torch.index_select(self._contact_mode[i], 0,
                                                                     next_mode_index[:, i])
            self._after_current_contact_mode[env_ids, i] = (torch.where(previous_mode_index[:, i] < 0,
                                                                        self._time_since_reset[env_ids],
                                                                        self._time_since_reset[env_ids] -
                                                                        torch.index_select(
                                                                            self._contact_timing_cumsum[i], 0,
                                                                            torch.clip(previous_mode_index[:, i],
                                                                                       min=0))))

            self._until_next_contact_mode[env_ids, i] = torch.index_select(self._contact_timing_cumsum[i], 0,
                                                                           current_mode_index[:, i]) - \
                                                        self._time_since_reset[env_ids]

            # for stance (1.0)
            idx = self._current_contact_mode[env_ids, i] >= 0.5
            self._current_contact_phase[env_ids, i] = torch.where(idx,
                                                                  torch.ones_like(self._current_contact_phase[env_ids, i]),
                                                                  self._current_contact_phase[env_ids, i])
            self._current_contact_state[env_ids, i] = torch.where(idx, True, self._current_contact_state[env_ids, i])

            # for swing (-1.0)
            idx = self._current_contact_mode[env_ids, i] <= -0.5
            self._current_contact_phase[env_ids, i] = torch.where(idx,
                                                                  torch.zeros_like(self._current_contact_phase[env_ids, i]),
                                                                  self._current_contact_phase[env_ids, i])
            self._current_contact_state[env_ids, i] = torch.where(idx, False, self._current_contact_state[env_ids, i])

            # for phase (0.0)
            idx = torch.logical_and(self._current_contact_mode[env_ids, i] < 0.5,
                                    self._current_contact_mode[env_ids, i] > -0.5)
            self._current_contact_phase[env_ids, i] = torch.where(idx,
                                                                  torch.remainder(
                                                                      self._after_current_contact_mode[env_ids, i] +
                                                                      self._current_contact_extra[env_ids, i, 0] *
                                                                      self._current_contact_extra[env_ids, i, 1],
                                                                      self._current_contact_extra[env_ids, i, 1]) /
                                                                  self._current_contact_extra[env_ids, i, 1],
                                                                  self._current_contact_phase[env_ids, i])
            self._current_contact_state[env_ids, i] = torch.where(
                torch.logical_and(idx,
                                  self._current_contact_phase[env_ids, i] < self._current_contact_extra[env_ids, i, 2]),
                True, self._current_contact_state[env_ids, i])
            # 0 -> swing_ratio, contact
            self._current_contact_state[env_ids, i] = torch.where(
                torch.logical_and(idx,
                                  self._current_contact_phase[env_ids, i] >= self._current_contact_extra[
                                      env_ids, i, 2]),
                False, self._current_contact_state[env_ids, i])
            # swing_ratio -> 1, contact

    def update(self):
        self._time_since_reset = self._env.time_since_reset

        current_mode_index = torch.sum(
            self._time_since_reset[:, None, None] >= (self._contact_timing_cumsum[None, :] + 1e-4),
            dim=-1)

        previous_mode_index = current_mode_index - 1
        next_mode_index = torch.clip(current_mode_index + 1, max=self._contact_max_index[None, :] - 1)

        for i in range(self._env.num_feet):
            self._current_contact_mode[:, i] = torch.index_select(self._contact_mode[i], 0,
                                                                  current_mode_index[:, i])
            self._current_contact_timing[:, i] = torch.index_select(self._contact_timing[i], 0,
                                                                    current_mode_index[:, i])
            self._current_contact_extra[:, i] = torch.index_select(self._contact_extra[i], 0,
                                                                   current_mode_index[:, i])
            self._next_contact_mode[:, i] = torch.index_select(self._contact_mode[i], 0,
                                                               next_mode_index[:, i])
            self._after_current_contact_mode[:, i] = (torch.where(previous_mode_index[:, i] < 0,
                                                                  self._time_since_reset,
                                                                  self._time_since_reset -
                                                                  torch.index_select(
                                                                      self._contact_timing_cumsum[i], 0,
                                                                      torch.clip(previous_mode_index[:, i],
                                                                                 min=0))))

            self._until_next_contact_mode[:, i] = torch.index_select(self._contact_timing_cumsum[i], 0,
                                                                     current_mode_index[:, i]) - \
                                                  self._time_since_reset

            # for stance (1.0)
            idx = self._current_contact_mode[:, i] > 0.5
            self._current_contact_phase[:, i] = torch.where(idx,
                                                            torch.ones_like(self._current_contact_phase[:, i]),
                                                            self._current_contact_phase[:, i])
            self._current_contact_state[:, i] = torch.where(idx, True, self._current_contact_state[:, i])

            # for swing (-1.0)
            idx = self._current_contact_mode[:, i] < -0.5
            self._current_contact_phase[:, i] = torch.where(idx,
                                                            torch.zeros_like(self._current_contact_phase[:, i]),
                                                            self._current_contact_phase[:, i])
            self._current_contact_state[:, i] = torch.where(idx, False, self._current_contact_state[:, i])

            # for phase (0.0)
            idx = torch.logical_and(self._current_contact_mode[:, i] < 0.5,
                                    self._current_contact_mode[:, i] > -0.5)
            self._current_contact_phase[:, i] = torch.where(idx,
                                                            torch.remainder(
                                                                self._after_current_contact_mode[:, i] +
                                                                self._current_contact_extra[:, i, 0] *
                                                                self._current_contact_extra[:, i, 1],
                                                                self._current_contact_extra[:, i,
                                                                1]) / self._current_contact_extra[:, i, 1],
                                                            self._current_contact_phase[:, i])
            self._current_contact_state[:, i] = torch.where(
                torch.logical_and(idx,
                                  self._current_contact_phase[:, i] < self._current_contact_extra[:, i, 2]),
                True, self._current_contact_state[:, i])
            # 0 -> swing_ratio, contact
            self._current_contact_state[:, i] = torch.where(
                torch.logical_and(idx,
                                  self._current_contact_phase[:, i] >= self._current_contact_extra[
                                                                       :, i, 2]),
                False, self._current_contact_state[:, i])
            # swing_ratio -> 1, contact

    @property
    def desired_contact_state(self):
        return self._current_contact_state

    @property
    def desired_contact_phase(self):
        return self._current_contact_phase

    @property
    def desired_contact_mode(self):
        return self._current_contact_mode

    @property
    def desired_contact_mode_next(self):
        return self._next_contact_mode

    @property
    def until_next_contact_mode(self):
        return self._until_next_contact_mode

    @property
    def normalized_phase(self):
        # Normalize the phase to [0, 1) for stance and swing
        # (0 -> 1) stance, (0 -> 1) swing
        contact_phase = self._current_contact_phase.clone()
        # stance and swing are already in (0 -> 1) range
        # phase (0.0)
        for i in range(4):
            idx = torch.logical_and(self._current_contact_mode[:, i] < 0.5,
                                    self._current_contact_mode[:, i] > -0.5)

            contact_phase[:, i] = torch.where(
                torch.logical_and(idx, self._current_contact_phase[:, i] < self._current_contact_extra[:, i, 2]),
                self._current_contact_phase[:, i] / self._current_contact_extra[:, i, 2],
                contact_phase[:, i])

            contact_phase[:, i] = torch.where(
                torch.logical_and(idx, self._current_contact_phase[:, i] >= self._current_contact_extra[:, i, 2]),
                (self._current_contact_phase[:, i] - self._current_contact_extra[:, i, 2]) /
                (1.0 - self._current_contact_extra[:, i, 2]),
                contact_phase[:, i])

        return contact_phase

    @property
    def stance_duration(self):
        stance_duration = self._current_contact_timing.clone()
        # stance (1.0) is the contact timing
        # swing (-1.0) is 0.0
        idx = self._current_contact_mode <= -0.5
        stance_duration = torch.where(idx, torch.zeros_like(stance_duration), stance_duration)

        # phase (0.0)
        idx = torch.logical_and(self._current_contact_mode < 0.5,
                                self._current_contact_mode > -0.5)
        stance_duration = torch.where(idx, self._current_contact_extra[:, :, 1] * self._current_contact_extra[:, :, 2],
                                      stance_duration)
        return stance_duration

    def _set_contact_generator_vis(self, debug_vis):
        self._set_contact_generator_vis_impl(debug_vis)
        if debug_vis:
            if self.contact_generator_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self.contact_generator_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._contact_generator_vis_callback(event)
                )
        else:
            if self.contact_generator_vis_handle is not None:
                self.contact_generator_vis_handle.unsubscribe()
                self.contact_generator_vis_handle = None
        return True

    def _set_contact_generator_vis_impl(self, debug_vis):
        if debug_vis:
            if not hasattr(self, "contact_generator_visualizer"):
                self.contact_generator_visualizer = []
                self.current_time_visualizer = []
                self.phase_visualizer = []

                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/orientation"
                marker_cfg.markers["arrow"].scale = (0.1, 0.2, 0.2)
                self.orientation_visualizer = VisualizationMarkers(marker_cfg)

                for i in range(self._env.num_feet):
                    marker_cfg = CYLINDER_TIME_MARKER_CFG.copy()
                    marker_cfg.prim_path = "/Visuals/timing_foot_" + str(i)
                    marker_cfg.markers["current_time"].scale = (0.05, 0.05, 1.0)
                    self.current_time_visualizer.append(VisualizationMarkers(marker_cfg))

                    marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                    marker_cfg.prim_path = "/Visuals/phase_" + str(i)
                    marker_cfg.markers["arrow"].scale = (0.1, 0.1, 0.3)
                    self.phase_visualizer.append(VisualizationMarkers(marker_cfg))

                    contact_generator_visualizer_per_foot = []
                    for j in range(self._contact_max_index[i]):
                        if self._contact_mode[i, j] > 0.5:
                            marker_cfg = CYLINDER_STANCE_MARKER_CFG.copy()
                        elif self._contact_mode[i, j] < -0.5:
                            marker_cfg = CYLINDER_SWING_MARKER_CFG.copy()
                        else:
                            marker_cfg = CYLINDER_PHASE_MARKER_CFG.copy()
                        marker_cfg.prim_path = "/Visuals/contact_plan_foot_" + str(i) + "_mode" + str(j)
                        marker_cfg.markers["contact_plan"].scale = (
                            0.04, 0.04, self._contact_timing_percentage[i, j] * 0.2)
                        contact_generator_visualizer_per_foot.append(VisualizationMarkers(marker_cfg))

                    self.contact_generator_visualizer.append(contact_generator_visualizer_per_foot)

            for i in range(len(self.current_time_visualizer)):
                self.current_time_visualizer[i].set_visibility(True)

            for i in range(len(self.phase_visualizer)):
                self.phase_visualizer[i].set_visibility(True)

            for i in range(len(self.contact_generator_visualizer)):
                for j in range(len(self.contact_generator_visualizer[i])):
                    self.contact_generator_visualizer[i][j].set_visibility(True)

            self.orientation_visualizer.set_visibility(True)

        else:
            if hasattr(self, "contact_generator_visualizer"):
                for i in range(len(self.current_time_visualizer)):
                    self.current_time_visualizer[i].set_visibility(False)

                for i in range(len(self.phase_visualizer)):
                    self.phase_visualizer[i].set_visibility(False)

                for i in range(len(self.contact_generator_visualizer)):
                    for j in range(len(self.contact_generator_visualizer[i])):
                        self.contact_generator_visualizer[i][j].set_visibility(False)

                self.orientation_visualizer.set_visibility(False)

    def _contact_generator_vis_callback(self, event):
        arrow_pos_w = self._env._robot.data.root_pos_w.clone()
        arrow_pos_w[:, 2] += 1.0
        offset_mode = np.array([[0.2, 0.2, 0.0],  # FL
                                [0.2, -0.2, 0.0],  # FR
                                [-0.2, 0.2, 0.0],  # RL
                                [-0.2, -0.2, 0.0]])  # RR
        offset_phase = np.array([[0.1, 0.1, 0.0],  # FL
                                 [0.1, -0.1, 0.0],  # FR
                                 [-0.1, 0.1, 0.0],  # RL
                                 [-0.1, -0.1, 0.0]])

        self.orientation_visualizer.visualize(arrow_pos_w)

        for i in range(4):
            pos_w = arrow_pos_w.clone()
            pos_w[:, 0] += offset_mode[i, 0]
            pos_w[:, 1] += offset_mode[i, 1]

            pos_w_time = pos_w.clone()
            scale_time = torch.clip(self._time_since_reset / self._contact_timing_total[i], max=1.0) / 5.0
            pos_w_time[:, 2] += scale_time / 2.0
            scale_time = torch.cat(
                (
                    torch.ones_like(scale_time).unsqueeze(1),
                    torch.ones_like(scale_time).unsqueeze(1),
                    scale_time.unsqueeze(1)
                ), dim=1
            )
            self.current_time_visualizer[i].visualize(pos_w_time, scales=scale_time)

            pow_w_phase = pos_w.clone()
            pow_w_phase[:, 0] += offset_phase[i, 0]
            pow_w_phase[:, 1] += offset_phase[i, 1]

            pitch = self._current_contact_phase[:, i] * 2.0 * np.pi
            quat = math_utils.quat_from_euler_xyz(torch.zeros_like(pitch), pitch, torch.zeros_like(pitch))

            in_phase = torch.logical_and(self._current_contact_mode[:, i] < 0.5,
                                         self._current_contact_mode[:, i] > -0.5)
            scale_phase = torch.cat(
                (
                    torch.ones_like(in_phase).unsqueeze(1),
                    torch.ones_like(in_phase).unsqueeze(1),
                    in_phase.unsqueeze(1)
                ), dim=1
            )
            self.phase_visualizer[i].visualize(pow_w_phase, quat, scales=scale_phase)

            for j in range(self._contact_max_index[i]):
                pos_w_contact = pos_w.clone()
                if j == 0:
                    pos_w_contact[:, 2] += self._contact_timing_percentage[i][j] * 0.1
                elif j > 0:
                    pos_w_contact[:, 2] += self._contact_timing_cumsum_percentage[i][j - 1] * 0.2 + 0.1 * \
                                           self._contact_timing_percentage[i][j]
                self.contact_generator_visualizer[i][j].visualize(pos_w_contact)


CYLINDER_SWING_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "contact_plan": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/cylinder.usd",
            scale=(1.0, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        )
    }
)

CYLINDER_STANCE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "contact_plan": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/cylinder.usd",
            scale=(1.0, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.0)),
        )
    }
)

CYLINDER_PHASE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "contact_plan": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/cylinder.usd",
            scale=(1.0, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.0)),
        )
    }
)

CYLINDER_TIME_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "current_time": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/cylinder.usd",
            scale=(1.0, 1.0, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.5)),
        )
    }
)
