from __future__ import annotations

import torch
import numpy as np
import omni.kit.app
import weakref

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, \
    GREEN_SPHERE_MARKER_CFG, BLUE_SPHERE_MARKER_CFG, YELLOW_SPHERE_MARKER_CFG, CYAN_SPHERE_MARKER_CFG
import isaaclab.envs.mdp as mdp
from isaaclab.actuators import DelayedDCMotorCfg

from isaaclab_assets import UNITREE_GO2_CFG
from .modules import ContactGenerator, JointPositionController, QPTorqueOptimizer
from .utils.helper import to_torch

NOMINAL_BASE_HEIGHT = 0.45
FOOT_HEIGHT = 0.15
FOOT_CLEARANCE = 0.0
GAIT = "walking"

#################
FF_TORQUE = True
GRAVITY_COMPENSATION = True  # only for swing legs

BASE_ACTION = True
BASE_ACTION_SCALE = 5.0  # 2.0, 5.0(l), 10.0(lr), 20.0(lrr)

JOINT_ACTION = True
JOINT_ACTION_SCALE = 0.15  # 0.1, 0.2(l), 0.5(xlr)

SAMPLE_VEL_COMMANDS = True
SAMPLE_POS_COMMANDS = True
SAMPLE_FORCE_COMMANDS = True

POS_LIMIT_MARGIN = 0.1
TORQUE_LIMIT_SCALE = 0.9

#################
ALPHA = 0.5
POS_ALPHA = ALPHA
VEL_ALPHA = ALPHA
TOR_ALPHA = ALPHA

ADD_LINK_DR = True

ACTUATOR_DELAY = True
ACTUATOR_DELAY_STEPS = 10
################

VEL_X = [-0.5, 0.5]
VEL_Y = [0.0, 0.0]
AVEL_Z = [-0.5, 0.5]

FL_POS_X = [0.15, 0.30]
FL_POS_Y = [0, 0.2]
FL_POS_Z = [0.3, 0.9]

FR_POS_X = [0.15, 0.30]
FR_POS_Y = [-0.2, 0]
FR_POS_Z = [0.3, 0.9]

FL_FORCE_X = [-20.0, 20.0]
FL_FORCE_Y = [-20.0, 20.0]
FL_FORCE_Z = [-20.0, 20.0]

FR_FORCE_X = [-20.0, 20.0]
FR_FORCE_Y = [-20.0, 20.0]
FR_FORCE_Z = [-20.0, 20.0]

# FL_FORCE_X = [-10.0, 10.0]
# FL_FORCE_Y = [-10.0, 10.0]
# FL_FORCE_Z = [-10.0, 10.0]
#
# FR_FORCE_X = [-10.0, 10.0]
# FR_FORCE_Y = [-10.0, 10.0]
# FR_FORCE_Z = [-10.0, 10.0]

# FL_FORCE_X = [-0.0, 0.0]
# FL_FORCE_Y = [-40.0, 0.0]
# FL_FORCE_Z = [-0.0, 0.0]
#
# FR_FORCE_X = [-0.0, 0.0]
# FR_FORCE_Y = [-0.0, 40.0]
# FR_FORCE_Z = [-0.0, 0.0]

SYMMETRIC = True


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.5),
            "dynamic_friction_range": (0.5, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )

    add_base_com_pos = EventTerm(
        func=mdp.randomize_rigid_body_com_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_pos_distribution_params": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    if ADD_LINK_DR:
        add_fl_hip_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="FL_hip"),
                "mass_distribution_params": (-0.1, 0.1),
                "operation": "add",
            },
        )

        add_fl_hip_com_pos = EventTerm(
            func=mdp.randomize_rigid_body_com_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="FL_hip"),
                "com_pos_distribution_params": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
            },
        )

        add_fl_thigh_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="FL_thigh"),
                "mass_distribution_params": (-0.2, 0.2),
                "operation": "add",
            },
        )

        add_fl_thigh_com_pos = EventTerm(
            func=mdp.randomize_rigid_body_com_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="FL_thigh"),
                "com_pos_distribution_params": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
            },
        )

        add_fl_calf_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="FL_calf"),
                "mass_distribution_params": (-0.1, 0.1),
                "operation": "add",
            },
        )

        add_fl_calf_com_pos = EventTerm(
            func=mdp.randomize_rigid_body_com_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="FL_calf"),
                "com_pos_distribution_params": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
            },
        )

        add_fr_hip_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="FR_hip"),
                "mass_distribution_params": (-0.1, 0.1),
                "operation": "add",
            },
        )

        add_fr_hip_com_pos = EventTerm(
            func=mdp.randomize_rigid_body_com_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="FR_hip"),
                "com_pos_distribution_params": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
            },
        )

        add_fr_thigh_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="FR_thigh"),
                "mass_distribution_params": (-0.2, 0.2),
                "operation": "add",
            },
        )

        add_fr_thigh_com_pos = EventTerm(
            func=mdp.randomize_rigid_body_com_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="FR_thigh"),
                "com_pos_distribution_params": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
            },
        )

        add_fr_calf_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="FR_calf"),
                "mass_distribution_params": (-0.1, 0.1),
                "operation": "add",
            },
        )

        add_fr_calf_com_pos = EventTerm(
            func=mdp.randomize_rigid_body_com_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="FR_calf"),
                "com_pos_distribution_params": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
            },
        )

        add_rl_hip_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="RL_hip"),
                "mass_distribution_params": (-0.1, 0.1),
                "operation": "add",
            },
        )

        add_rl_hip_com_pos = EventTerm(
            func=mdp.randomize_rigid_body_com_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="RL_hip"),
                "com_pos_distribution_params": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
            },
        )

        add_rl_thigh_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="RL_thigh"),
                "mass_distribution_params": (-0.2, 0.2),
                "operation": "add",
            },
        )

        add_rl_thigh_com_pos = EventTerm(
            func=mdp.randomize_rigid_body_com_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="RL_thigh"),
                "com_pos_distribution_params": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
            },
        )

        add_rl_calf_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="RL_calf"),
                "mass_distribution_params": (-0.1, 0.1),
                "operation": "add",
            },
        )

        add_rl_calf_com_pos = EventTerm(
            func=mdp.randomize_rigid_body_com_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="RL_calf"),
                "com_pos_distribution_params": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
            },
        )

        add_rr_hip_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="RR_hip"),
                "mass_distribution_params": (-0.1, 0.1),
                "operation": "add",
            },
        )

        add_rr_hip_com_pos = EventTerm(
            func=mdp.randomize_rigid_body_com_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="RR_hip"),
                "com_pos_distribution_params": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
            },
        )

        add_rr_thigh_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="RR_thigh"),
                "mass_distribution_params": (-0.2, 0.2),
                "operation": "add",
            },
        )

        add_rr_thigh_com_pos = EventTerm(
            func=mdp.randomize_rigid_body_com_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="RR_thigh"),
                "com_pos_distribution_params": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
            },
        )

        add_rr_calf_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="RR_calf"),
                "mass_distribution_params": (-0.1, 0.1),
                "operation": "add",
            },
        )

        add_rr_calf_com_pos = EventTerm(
            func=mdp.randomize_rigid_body_com_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="RR_calf"),
                "com_pos_distribution_params": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
            },
        )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 6.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5)}},
    )


@configclass
class QPEnvCfg(DirectRLEnvCfg):
    viewer: ViewerCfg = ViewerCfg(
        eye=(5.0, 5.0, 5.0),
    )
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.002,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.0, replicate_physics=True)

    # robot
    nominal_base_height = NOMINAL_BASE_HEIGHT
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    robot.init_state.pos = (0.0, 0.0, nominal_base_height)
    robot.init_state.rot = (np.sqrt(2) / 2, 0.0, -np.sqrt(2) / 2, 0.0)

    if ACTUATOR_DELAY:
        robot.actuators = {
            "calf": DelayedDCMotorCfg(
                joint_names_expr=[".*_calf_joint"],
                effort_limit=40.887,  # 45.43, 40.887, 36.344
                saturation_effort=40.887,
                velocity_limit=15.70,
                stiffness=40.0,
                damping=1.0,
                friction=0.0,
                min_num_time_lags=0,
                max_num_time_lags=ACTUATOR_DELAY_STEPS,
                num_time_lags=None,
            ),
            "hip_thigh": DelayedDCMotorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
                effort_limit=21.33,  # 23.7, 21.33, 18.96
                saturation_effort=21.33,
                velocity_limit=30.1,
                stiffness=40.0,
                damping=1.0,
                friction=0.0,
                min_num_time_lags=0,
                max_num_time_lags=ACTUATOR_DELAY_STEPS,
                num_time_lags=None,
            ),
        }
    foot_height = FOOT_HEIGHT
    foot_clearance = 0.0

    robot.init_state.joint_pos = {
        "FL_hip_joint": 0.0,
        "RL_hip_joint": 0.0,
        "FR_hip_joint": 0.0,
        "RR_hip_joint": 0.0,
        "FL_thigh_joint": np.pi / 2,
        "FR_thigh_joint": np.pi / 2,
        "RL_thigh_joint": 1.0 + np.pi / 2,
        "RR_thigh_joint": 1.0 + np.pi / 2,
        "FL_calf_joint": -1.5,
        "FR_calf_joint": -1.5,
        "RL_calf_joint": -1.5,
        "RR_calf_joint": -1.5,
    }

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    add_feedforward_torque = FF_TORQUE

    including_base_action = BASE_ACTION
    including_joint_action = JOINT_ACTION

    base_action_scale = [BASE_ACTION_SCALE] * 6
    joint_action_scale = [JOINT_ACTION_SCALE] * 12

    # env
    decimation = 5

    num_actions = 0
    if including_base_action:
        num_actions += 6
    if including_joint_action:
        num_actions += 12
    if num_actions == 0:
        num_actions = 1  # to avoid zero action dimension

    history_length = 5  # include the current state
    num_obs_per_step = 1 + 3 + 3 + 3 + 12 + 12 + 4 + 4 + 12 + 3 + num_actions + 3 + 3 + 3 + 3
    num_observations = num_obs_per_step * history_length

    observation_space = num_observations
    action_space = num_actions

    action_scale = []
    if including_base_action:
        action_scale += base_action_scale
    if including_joint_action:
        action_scale += joint_action_scale
    if len(action_scale) == 0:
        action_scale = [0.1]

    feet_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    # the values taken from urdf files
    com_offset = np.array([0.0, 0.0, 0.0])
    hip_offset = np.array([[0.1934, 0.0465, 0.],  # FL
                           [0.1934, -0.0465, 0.],  # FR
                           [-0.1934, 0.0465, 0.],  # RL
                           [-0.1934, -0.0465, 0.]]) + com_offset  # RR
    link_lengths = np.array([0.0955, 0.213, 0.213])  # thigh, calf, foot

    episode_length_s = 10.0

    gait = GAIT  # walking
    contact_sequence = {}
    if gait == "walking":
        contact_sequence = {
            "FL": [
                ["swing", 0.5, 0.0, 0.0, 0.0],
                ["swing", 9.5, 0.0, 0.0, 0.0],
            ],
            "FR": [
                ["swing", 0.5, 0.0, 0.0, 0.0],
                ["swing", 9.5, 0.0, 0.0, 0.0],
            ],
            "RL": [
                ["stance", 0.5, 0.0, 0.0, 0.0],
                ["phase", 9.5, 0.5, 0.4, 0.7],
            ],
            "RR": [
                ["stance", 0.5, 0.0, 0.0, 0.0],
                ["phase", 9.5, 0.0, 0.4, 0.7],  # initial phase, duration, (0 -contact-> swing_ratio -swing-> 1)
            ],
        }
    else:
        raise ValueError("Unknown gait")
    contact_generator_config = {
        "contact_generator_debug_vis": False,
        "contact_sequence": contact_sequence,
    }

    joint_position_controller_config = {
        "joint_position_controller_debug_vis": True,
        "desired_joint_pos_stance": np.array([
            0.0, 0.0, 0.0, 0.0,  # hip
            np.pi / 2, np.pi / 2, 1.0 + np.pi / 2, 1.0 + np.pi / 2,  # thigh
            -1.5, -1.5, -1.5, -1.5,  # calf
        ]),
        "desired_joint_pos_swing": np.array([
            0.0, 0.0, 0.0, 0.0,  # hip
            np.pi / 2, np.pi / 2, 1.0 + np.pi / 2, 1.0 + np.pi / 2,  # thigh
            -2.2, -2.2, -2.2, -2.2,  # calf
        ]),
        "desired_foot_clearance": foot_clearance,
        "desired_foot_height": foot_height,
        "hip_positions_in_body_frame": np.array([[0.1934, 0.142, 0.0],  # FL
                                                 [0.1934, -0.142, 0.0],  # FR
                                                 [-0.1934, 0.142, 0.0],  # RL
                                                 [-0.1934, -0.142, 0.0]]),  # RR
    }

    qp_torque_optimizer_config = {
        "qp_debug_vis": True,
        "base_position_kp": np.array([50., 50., 50.]),
        "base_position_kd": np.array([10., 10., 10.]),
        "base_orientation_kp": np.array([50., 50., 50.]),
        "base_orientation_kd": np.array([10., 10., 10.]),
        "qp_weight_ddq": np.diag([1., 1., 1., 1., 1., 1.]),
        "qp_weight_grf": 1e-4,
        "qp_weight_ee_force": 1.,
        "qp_foot_friction_coef": 0.6,
    }

    # gravity compensation
    gravity_compensation_torque_for_swing_legs = GRAVITY_COMPENSATION  # added only if add_feedforward_torque is True

    # using the actual contact state of the robot
    use_actual_contact = False

    # termination
    terminate_on_undesired_foot_contact = False
    terminate_on_low_base_height = True
    terminate_on_large_orientation_error = True
    terminate_on_body_contact = True
    terminate_on_limb_contact = True

    # sample commands for certain tasks
    enable_sampled_velocity_commands = SAMPLE_VEL_COMMANDS
    velocity_debug_vis = enable_sampled_velocity_commands
    vel_range_x = VEL_X
    vel_range_y = VEL_Y
    avel_range_z = AVEL_Z

    enable_sampled_pos_commands = SAMPLE_POS_COMMANDS
    pos_debug_vis = enable_sampled_pos_commands
    fl_pos_x = FL_POS_X
    fl_pos_y = FL_POS_Y
    fl_pos_z = FL_POS_Z
    fr_pos_x = FR_POS_X
    fr_pos_y = FR_POS_Y
    fr_pos_z = FR_POS_Z

    enable_sampled_force_commands = SAMPLE_FORCE_COMMANDS
    force_debug_vis = enable_sampled_force_commands
    fl_force_x = FL_FORCE_X
    fl_force_y = FL_FORCE_Y
    fl_force_z = FL_FORCE_Z
    fr_force_x = FR_FORCE_X
    fr_force_y = FR_FORCE_Y
    fr_force_z = FR_FORCE_Z

    joint_pos_limit_margin = POS_LIMIT_MARGIN
    joint_torque_limit_scale = TORQUE_LIMIT_SCALE
    joint_torque_limit = [23.7, 23.7, 23.7, 23.7,
                          23.7, 23.7, 23.7, 23.7,
                          45.43, 45.43, 45.43, 45.43]

    # Sim2real
    randomize_initial_state = True
    events: EventCfg = EventCfg()
    obs_noise = True

    pos_alpha = POS_ALPHA
    vel_alpha = VEL_ALPHA
    tor_alpha = TOR_ALPHA


class QPEnv(DirectRLEnv):

    def __init__(self, cfg: QPEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.feet_ids, _ = self._robot.find_bodies(self.cfg.feet_names)
        self.num_feet = len(self.feet_ids)

        # Get specific body indices
        self._contact_base_id, _ = self._contact_sensor.find_bodies("base")
        self._contact_head_id, _ = self._contact_sensor.find_bodies("Head_.*")
        self._contact_feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._contact_thigh_ids, _ = self._contact_sensor.find_bodies(".*thigh")
        self._contact_calf_ids, _ = self._contact_sensor.find_bodies(".*calf")

        # some buffers
        self._time_since_reset = torch.zeros(self.num_envs, device=self.device)
        self._episode_length = self.cfg.episode_length_s / (self.cfg.sim.dt * self.cfg.decimation)
        self._obs_history = torch.zeros(
            self.num_envs, self.cfg.history_length, self.cfg.num_obs_per_step, device=self.device)
        self._last_action = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
        # X/Y linear velocity and yaw angular velocity commands
        self._velocity_commands = torch.zeros(self.num_envs, 3, device=self.device)

        self._ee_pos_fl_commands = torch.zeros(self.num_envs, 3, device=self.device)  # in the projected frame
        self._ee_pos_fl_commands[:, 0] = 0.2175
        self._ee_pos_fl_commands[:, 1] = 0.1225
        self._ee_pos_fl_commands[:, 2] = 0.4880

        self._ee_pos_fr_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._ee_pos_fr_commands[:, 0] = 0.2175
        self._ee_pos_fr_commands[:, 1] = -0.1225
        self._ee_pos_fr_commands[:, 2] = 0.4880

        self._ee_force_fl_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._ee_force_fr_commands = torch.zeros(self.num_envs, 3, device=self.device)

        self.desired_joint_pos = torch.zeros(self.num_envs, 12, device=self.device)

        # the ones that are used for the simulation
        self.desired_pos = torch.zeros(self.num_envs, 12, device=self.device)
        self.desired_pos = self._robot.data.default_joint_pos
        self.desired_vel = torch.zeros(self.num_envs, 12, device=self.device)
        self.desired_tor = torch.zeros(self.num_envs, 12, device=self.device)

        # some constants
        self.hip_offset = to_torch(self.cfg.hip_offset, device=self.device)
        self.link_lengths = to_torch(self.cfg.link_lengths, device=self.device)
        self.action_scale = to_torch(self.cfg.action_scale, device=self.device)
        self.joint_torque_limit = to_torch(self.cfg.joint_torque_limit, device=self.device).repeat(self.num_envs,
                                                                                                   1) * self.cfg.joint_torque_limit_scale

        # torque optimizer
        self.contact_generator = ContactGenerator(self)
        self.joint_position_controller = JointPositionController(self)
        self.torque_optimizer = QPTorqueOptimizer(self)

        self._prepare_rewards()
        self.set_debug_vis(self.cfg.velocity_debug_vis, self.cfg.pos_debug_vis, self.cfg.force_debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def step(self, action: torch.Tensor):
        # check NaNs
        if torch.any(torch.isnan(action)):
            if torch.any(torch.isnan(action)):
                # replace the nan with zeros
                action = torch.where(torch.isnan(action), torch.zeros_like(action), action)
                print("Warning: NaN detected in action")
                # raise ValueError("NaN detected in action")

        scaled_action = action * self.action_scale.unsqueeze(0)
        scaled_action = torch.clip(scaled_action, -self.action_scale.unsqueeze(0) * 2,
                                   self.action_scale.unsqueeze(0) * 2)

        self.torque_optimizer.desired_base_position = self._robot.data.default_root_state[:, 0:3]
        self.torque_optimizer.desired_base_orientation_quat = self._robot.data.default_root_state[:, 3:7]
        self.torque_optimizer.desired_linear_velocity = torch.stack((
            self._velocity_commands[:, 0],
            self._velocity_commands[:, 1],
            torch.zeros_like(self._velocity_commands[:, 1]),
        ), dim=1)
        self.torque_optimizer.desired_angular_velocity = torch.stack((
            torch.zeros_like(self._velocity_commands[:, 2]),
            torch.zeros_like(self._velocity_commands[:, 2]),
            self._velocity_commands[:, 2],
        ), dim=1)

        # all specified in the com projection frame
        if self.cfg.including_base_action:
            self.torque_optimizer.desired_linear_acceleration = scaled_action[:, 0:3]
            self.torque_optimizer.desired_angular_acceleration = scaled_action[:, 3:6]
            if self.cfg.including_joint_action:
                desired_motor_position = self.desired_joint_pos + scaled_action[:, 6:]
            else:
                desired_motor_position = self.desired_joint_pos
        else:
            self.torque_optimizer.desired_linear_acceleration = torch.zeros_like(
                self.torque_optimizer.desired_linear_acceleration)
            self.torque_optimizer.desired_angular_acceleration = torch.zeros_like(
                self.torque_optimizer.desired_angular_acceleration)

            if self.cfg.including_joint_action:
                desired_motor_position = self.desired_joint_pos + scaled_action
            else:
                desired_motor_position = self.desired_joint_pos

        desired_motor_position = torch.clamp(desired_motor_position,
                                             self._robot.data.default_joint_pos_limits[:, :,
                                             0] + self.cfg.joint_pos_limit_margin,
                                             self._robot.data.default_joint_pos_limits[:, :,
                                             1] - self.cfg.joint_pos_limit_margin)

        if self.cfg.add_feedforward_torque:
            # valid only for contact legs
            grf, qp_cost, _, _ = self.torque_optimizer.get_grf()
            stance_motor_torques = -torch.bmm(grf[:, None, :], self.all_foot_jacobian)[:, 0]  # include FL and FR

            if self.cfg.gravity_compensation_torque_for_swing_legs:
                swing_motor_torques = self.get_gravity_compensation_torques()
            else:
                swing_motor_torques = torch.zeros_like(stance_motor_torques)

            # get actions
            if self.cfg.use_actual_contact:
                contact_state_expanded = torch.tile(self.foot_contacts, (1, 3))
            else:
                contact_state_expanded = torch.tile(self.contact_generator.desired_contact_state, (1, 3))

            # override for FL and FR
            contact_state_expanded[:, 0] = 1.0
            contact_state_expanded[:, 1] = 1.0
            contact_state_expanded[:, 4] = 1.0
            contact_state_expanded[:, 5] = 1.0
            contact_state_expanded[:, 8] = 1.0
            contact_state_expanded[:, 9] = 1.0
            desired_joint_torque = swing_motor_torques + torch.where(contact_state_expanded,
                                                                     stance_motor_torques,
                                                                     torch.zeros_like(stance_motor_torques))
        else:
            qp_cost = torch.zeros(self.num_envs, device=self.device)
            desired_joint_torque = torch.zeros_like(desired_motor_position)

        desired_joint_torque = torch.clamp(desired_joint_torque, -self.joint_torque_limit, self.joint_torque_limit)

        self.desired_pos = self.cfg.pos_alpha * desired_motor_position + (1 - self.cfg.pos_alpha) * self.desired_pos
        self.desired_vel = torch.zeros_like(self.joint_vel)
        self.desired_tor = self.cfg.tor_alpha * desired_joint_torque + (1 - self.cfg.tor_alpha) * self.desired_tor

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.desired_pos[reset_env_ids] = desired_motor_position[reset_env_ids]
            self.desired_vel[reset_env_ids] = torch.zeros_like(self.joint_vel[reset_env_ids])
            self.desired_tor[reset_env_ids] = desired_joint_torque[reset_env_ids]

        external_force_com_fl = -self._ee_force_fl_commands
        external_force_com_fr = -self._ee_force_fr_commands

        gravity_vec_w = torch.tensor((0.0, 0.0, -1.0), device=self.device).repeat(self.num_envs, 1)
        projected_gravity_b = math_utils.quat_rotate_inverse(self._robot.data.root_quat_w, gravity_vec_w)
        base_rot_mat_rp_t = math_utils.rp_rotation_from_gravity_b(projected_gravity_b)
        base_quat_rp = math_utils.quat_from_matrix(base_rot_mat_rp_t.transpose(1, 2))

        external_force_w_fl = math_utils.quat_rotate(
            math_utils.quat_mul(self._robot.data.root_quat_w, math_utils.quat_inv(base_quat_rp)), external_force_com_fl)
        external_force_w_fr = math_utils.quat_rotate(
            math_utils.quat_mul(self._robot.data.root_quat_w, math_utils.quat_inv(base_quat_rp)), external_force_com_fr)

        body_id_fl = self.feet_ids[0]
        body_id_fr = self.feet_ids[1]
        body_quat_fl = self._robot.data.body_quat_w[:, body_id_fl]
        body_quat_fr = self._robot.data.body_quat_w[:, body_id_fr]
        external_force_b_fl = math_utils.quat_rotate_inverse(body_quat_fl, external_force_w_fl).unsqueeze(1)
        external_force_b_fr = math_utils.quat_rotate_inverse(body_quat_fr, external_force_w_fr).unsqueeze(1)

        external_torque_b_fl = torch.zeros_like(external_force_b_fl)
        external_torque_b_fr = torch.zeros_like(external_force_b_fr)

        self._robot.set_external_force_and_torque(external_force_b_fl, external_torque_b_fl,
                                                  env_ids=torch.arange(self.num_envs, dtype=torch.int64,
                                                                       device=self.device), body_ids=body_id_fl)
        self._robot.set_external_force_and_torque(external_force_b_fr, external_torque_b_fr,
                                                  env_ids=torch.arange(self.num_envs, dtype=torch.int64,
                                                                       device=self.device), body_ids=body_id_fr)

        for _ in range(self.cfg.decimation):
            # set actions into buffers
            self._robot.set_joint_position_target(self.desired_pos)
            self._robot.set_joint_velocity_target(self.desired_vel)
            self._robot.set_joint_effort_target(self.desired_tor)

            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

            self._time_since_reset += self.physics_dt

            # perform rendering if gui is enabled
        if self.sim.has_gui() or self.sim.has_rtx_sensors():
            self.sim.render()

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        self._time_since_reset = self.episode_length_buf * self.physics_dt * self.cfg.decimation

        # -- update contact generator and joint position controller
        self.contact_generator.update()
        self.desired_joint_pos = self.joint_position_controller.update()  # this comes from the contact scheduler

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs

        self.reward_buf = self._get_rewards(action)  # rewards may depend on dones

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self._last_action = action.clone()
        self.obs_buf = self._get_observations()

        # add observation noise
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

        # return observations, rewards, resets and extras
        # check NANs
        if torch.any(torch.isnan(self.obs_buf["policy"])):
            raise ValueError("NaN detected in observation buffer")

        in_contact = torch.any(self.contact_generator.desired_contact_state, dim=-1)
        self.extras['log']["Step Log/QP Cost"] = torch.mean(qp_cost * in_contact.float())
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            extras["Step Reward/" + key] = episodic_sum_avg / (
                    torch.mean(self.episode_length_buf[env_ids].float()) + 0.01)
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        self._time_since_reset[env_ids] = 0.0

        self.episode_length_buf[env_ids] = torch.randint_like(self.episode_length_buf[env_ids],
                                                              high=int(self.max_episode_length))
        self._time_since_reset[env_ids] = self.episode_length_buf[env_ids] * self.physics_dt * self.cfg.decimation

        self._obs_history[env_ids] = 0.0
        self._last_action[env_ids] = 0.0

        # Sample new commands
        self._velocity_commands[env_ids] = torch.zeros_like(self._velocity_commands[env_ids]).uniform_(0.0, 0.0)

        if self.cfg.enable_sampled_velocity_commands:
            non_stance = self.contact_generator.desired_contact_mode[env_ids].sum(
                dim=-1) < 3.9  # make sure it is not sampled when all foot in stance
            self._velocity_commands[env_ids, 0] = torch.zeros_like(self._velocity_commands[env_ids, 0]).uniform_(
                self.cfg.vel_range_x[0], self.cfg.vel_range_x[1]) * non_stance
            self._velocity_commands[env_ids, 1] = torch.zeros_like(self._velocity_commands[env_ids, 1]).uniform_(
                self.cfg.vel_range_y[0], self.cfg.vel_range_y[1]) * non_stance
            self._velocity_commands[env_ids, 2] = torch.zeros_like(self._velocity_commands[env_ids, 2]).uniform_(
                self.cfg.avel_range_z[0], self.cfg.avel_range_z[1]) * non_stance

        if self.cfg.enable_sampled_pos_commands:
            # sampled in the projected com frame
            non_stance = self.contact_generator.desired_contact_mode[env_ids].sum(
                dim=-1) < 3.9  # make sure it is not sampled when all foot in stance
            if SYMMETRIC:
                pos_x = torch.zeros_like(
                    self._ee_pos_fl_commands[env_ids, 0]).uniform_(self.cfg.fl_pos_x[0],
                                                                   self.cfg.fl_pos_x[1]) * non_stance
                pos_y = torch.zeros_like(
                    self._ee_pos_fl_commands[env_ids, 1]).uniform_(self.cfg.fl_pos_y[0],
                                                                   self.cfg.fl_pos_y[1]) * non_stance
                pos_z = torch.zeros_like(
                    self._ee_pos_fl_commands[env_ids, 2]).uniform_(self.cfg.fl_pos_z[0],
                                                                   self.cfg.fl_pos_z[1]) * non_stance
                self._ee_pos_fl_commands[env_ids, 0] = pos_x
                self._ee_pos_fl_commands[env_ids, 1] = pos_y
                self._ee_pos_fl_commands[env_ids, 2] = pos_z
                self._ee_pos_fr_commands[env_ids, 0] = pos_x
                self._ee_pos_fr_commands[env_ids, 1] = -pos_y
                self._ee_pos_fr_commands[env_ids, 2] = pos_z
            else:
                self._ee_pos_fl_commands[env_ids, 0] = torch.zeros_like(
                    self._ee_pos_fl_commands[env_ids, 0]).uniform_(self.cfg.fl_pos_x[0],
                                                                   self.cfg.fl_pos_x[1]) * non_stance
                self._ee_pos_fl_commands[env_ids, 1] = torch.zeros_like(
                    self._ee_pos_fl_commands[env_ids, 1]).uniform_(self.cfg.fl_pos_y[0],
                                                                   self.cfg.fl_pos_y[1]) * non_stance
                self._ee_pos_fl_commands[env_ids, 2] = torch.zeros_like(
                    self._ee_pos_fl_commands[env_ids, 2]).uniform_(self.cfg.fl_pos_z[0],
                                                                   self.cfg.fl_pos_z[1]) * non_stance
                self._ee_pos_fr_commands[env_ids, 0] = torch.zeros_like(
                    self._ee_pos_fr_commands[env_ids, 0]).uniform_(self.cfg.fr_pos_x[0],
                                                                   self.cfg.fr_pos_x[1]) * non_stance
                self._ee_pos_fr_commands[env_ids, 1] = torch.zeros_like(
                    self._ee_pos_fr_commands[env_ids, 1]).uniform_(self.cfg.fr_pos_y[0],
                                                                   self.cfg.fr_pos_y[1]) * non_stance
                self._ee_pos_fr_commands[env_ids, 2] = torch.zeros_like(
                    self._ee_pos_fr_commands[env_ids, 2]).uniform_(self.cfg.fr_pos_z[0],
                                                                   self.cfg.fr_pos_z[1]) * non_stance

        if self.cfg.enable_sampled_force_commands:
            # sampled in the projected com frame
            non_stance = self.contact_generator.desired_contact_mode[env_ids].sum(
                dim=-1) < 3.9  # make sure it is not sampled when all foot in stance
            if SYMMETRIC:
                force_x = torch.zeros_like(
                    self._ee_force_fl_commands[env_ids, 0]).uniform_(self.cfg.fl_force_x[0],
                                                                     self.cfg.fl_force_x[1]) * non_stance
                force_y = torch.zeros_like(
                    self._ee_force_fl_commands[env_ids, 1]).uniform_(self.cfg.fl_force_y[0],
                                                                     self.cfg.fl_force_y[1]) * non_stance
                force_z = torch.zeros_like(
                    self._ee_force_fl_commands[env_ids, 2]).uniform_(self.cfg.fl_force_z[0],
                                                                     self.cfg.fl_force_z[1]) * non_stance

                self._ee_force_fl_commands[env_ids, 0] = force_x
                self._ee_force_fl_commands[env_ids, 1] = force_y
                self._ee_force_fl_commands[env_ids, 2] = force_z
                self._ee_force_fr_commands[env_ids, 0] = force_x
                self._ee_force_fr_commands[env_ids, 1] = -force_y
                self._ee_force_fr_commands[env_ids, 2] = force_z
            else:
                self._ee_force_fl_commands[env_ids, 0] = torch.zeros_like(
                    self._ee_force_fl_commands[env_ids, 0]).uniform_(self.cfg.fl_force_x[0],
                                                                     self.cfg.fl_force_x[1]) * non_stance
                self._ee_force_fl_commands[env_ids, 1] = torch.zeros_like(
                    self._ee_force_fl_commands[env_ids, 1]).uniform_(self.cfg.fl_force_y[0],
                                                                     self.cfg.fl_force_y[1]) * non_stance
                self._ee_force_fl_commands[env_ids, 2] = torch.zeros_like(
                    self._ee_force_fl_commands[env_ids, 2]).uniform_(self.cfg.fl_force_z[0],
                                                                     self.cfg.fl_force_z[1]) * non_stance
                self._ee_force_fr_commands[env_ids, 0] = torch.zeros_like(
                    self._ee_force_fr_commands[env_ids, 0]).uniform_(self.cfg.fr_force_x[0],
                                                                     self.cfg.fr_force_x[1]) * non_stance
                self._ee_force_fr_commands[env_ids, 1] = torch.zeros_like(
                    self._ee_force_fr_commands[env_ids, 1]).uniform_(self.cfg.fr_force_y[0],
                                                                     self.cfg.fr_force_y[1]) * non_stance
                self._ee_force_fr_commands[env_ids, 2] = torch.zeros_like(
                    self._ee_force_fr_commands[env_ids, 2]).uniform_(self.cfg.fr_force_z[0],
                                                                     self.cfg.fr_force_z[1]) * non_stance

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 2] += 0.1  # initial height

        if self.cfg.randomize_initial_state:
            joint_pos += torch.rand_like(self._robot.data.default_joint_pos[env_ids]) * 0.2 - 0.1
            joint_vel += torch.rand_like(self._robot.data.default_joint_vel[env_ids]) * 0.1 - 0.05
            default_root_state[:, :3] += torch.rand_like(default_root_state[:, :3]) * 0.1 - 0.05
            default_root_state[:, 3:7] += torch.rand_like(default_root_state[:, 3:7]) * 0.1 - 0.05
            default_root_state[:, 3:7] = math_utils.normalize(default_root_state[:, 3:7])
            default_root_state[:, 7:] += torch.rand_like(default_root_state[:, 7:]) * 0.1 - 0.05

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.contact_generator.reset_idx(env_ids)
        self.desired_joint_pos[env_ids] = self.joint_position_controller.reset_idx(env_ids)

        self.contact_generator.reset_idx(env_ids)
        self.desired_joint_pos[env_ids] = self.joint_position_controller.reset_idx(env_ids)

    def _get_observations(self) -> dict:
        # states from GC and may be noisy
        latest_obs = torch.cat((
            self.base_height.unsqueeze(-1),
            self.projected_gravity_b,
            self.base_lin_vel_b,
            self.base_ang_vel_b,
            self.joint_pos - self._robot.data.default_joint_pos,
            self.joint_vel,
            self.contact_generator.desired_contact_phase,
            self.contact_generator.desired_contact_mode,
            self.desired_joint_pos - self._robot.data.default_joint_pos,
            self._velocity_commands,
            self._ee_pos_fl_commands,
            self._ee_pos_fr_commands,
            self._ee_force_fl_commands,
            self._ee_force_fr_commands,
            self._last_action,
        ), dim=-1)

        # update history
        if self.cfg.history_length > 1:
            self._obs_history = torch.cat((self._obs_history[:, 1:], latest_obs.unsqueeze(1)), dim=1)
        else:
            self._obs_history = latest_obs.unsqueeze(1)

        return {"policy": self._obs_history.view(self.num_envs, -1)}

    def _prepare_rewards(self):
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_orientation_reward",
                "track_height_reward",
                "track_lin_vel_reward",
                "track_ang_vel_reward",
                "track_ee_pos_fl_reward",
                "track_ee_pos_fr_reward",

                "penalize_contact_mismatch",
                "penalize_action_scale",
                "penalize_action_rate",
                "penalize_dof_acc",
                "penalize_dof_torque",
            ]
        }

    def _get_rewards(self, action) -> torch.Tensor:
        # task:
        gravity_vec_b = self._robot.data.projected_gravity_b.clone()
        gravity_vec_b_target = torch.zeros_like(gravity_vec_b)
        gravity_vec_b_target[:, 0] = -1.0
        track_orientation_error = torch.norm(gravity_vec_b - gravity_vec_b_target, dim=-1)
        track_orientation_reward = torch.exp(-torch.square(track_orientation_error) / 0.6 ** 2)

        base_height_error = torch.abs(self._robot.data.root_pos_w[:, 2] - self.cfg.nominal_base_height)
        track_height_reward = torch.exp(-torch.square(base_height_error) / 0.2 ** 2)

        transformed_lin_vel = torch.cat(
            [
                -self._robot.data.root_lin_vel_b.clone()[:, 2:3],
                self._robot.data.root_lin_vel_b.clone()[:, 1:2],
            ], dim=-1,
        )
        transformed_ang_vel = self._robot.data.root_ang_vel_b.clone()[:, 0]

        track_lin_vel_error = torch.norm(transformed_lin_vel - self._velocity_commands[:, :2], dim=-1)
        track_ang_vel_error = torch.abs(transformed_ang_vel - self._velocity_commands[:, 2])
        track_lin_vel_reward = torch.exp(-torch.square(track_lin_vel_error) / 0.3 ** 2)
        track_ang_vel_reward = torch.exp(-torch.square(track_ang_vel_error) / 0.4 ** 2)

        # tracking FL pos
        quat_yaw = math_utils.yaw_quat(self._robot.data.root_quat_w)

        ee_pos = self._robot.data.body_state_w[:, self.feet_ids, 0:3].clone()
        ee_pos_fl = ee_pos[:, 0]
        ee_pos_fl[:, :2] -= self._robot.data.root_pos_w[:, :2]
        ee_pos_fl_com = math_utils.quat_rotate_inverse(quat_yaw, ee_pos_fl)
        track_ee_pos_fl_error = torch.norm(ee_pos_fl_com - self._ee_pos_fl_commands, dim=-1)
        track_ee_pos_fl_reward = torch.exp(-torch.square(track_ee_pos_fl_error) / 0.1 ** 2)

        ee_pos_fr = ee_pos[:, 1]
        ee_pos_fr[:, :2] -= self._robot.data.root_pos_w[:, :2]
        ee_pos_fr_com = math_utils.quat_rotate_inverse(quat_yaw, ee_pos_fr)
        track_ee_pos_fr_error = torch.norm(ee_pos_fr_com - self._ee_pos_fr_commands, dim=-1)
        track_ee_pos_fr_reward = torch.exp(-torch.square(track_ee_pos_fr_error) / 0.1 ** 2)

        action_rate = torch.norm(self._last_action - action, dim=-1)
        penalize_action_rate = torch.exp(-torch.square(action_rate) / 10.0 ** 2)

        action_scale = torch.norm(action, dim=-1)
        penalize_action_scale = torch.exp(-torch.square(action_scale) / 8.0 ** 2)

        contact_unmatch = torch.sum(
            self.foot_contacts == torch.logical_not(self.contact_generator.desired_contact_state), dim=-1)
        penalize_contact_mismatch = torch.pow(0.5, contact_unmatch)

        dof_acc = torch.norm(self._robot.data.joint_acc, dim=-1)
        penalize_dof_acc = torch.exp(-torch.square(dof_acc) / 500.0 ** 2)

        dof_torque = torch.norm(self._robot.data.applied_torque, dim=-1)
        penalize_dof_torque = torch.exp(-torch.square(dof_torque) / 100.0 ** 2)

        tracking_rewards = {
            "track_orientation_reward": track_orientation_reward,
            "track_height_reward": track_height_reward,
            "track_lin_vel_reward": track_lin_vel_reward,
            "track_ang_vel_reward": track_ang_vel_reward,
            "track_ee_pos_fl_reward": track_ee_pos_fl_reward,
            "track_ee_pos_fr_reward": track_ee_pos_fr_reward,
        }
        penalty_rewards = {
            "penalize_contact_mismatch": penalize_contact_mismatch,
            "penalize_action_scale": penalize_action_scale,
            "penalize_action_rate": penalize_action_rate,
            "penalize_dof_acc": penalize_dof_acc,
            "penalize_dof_torque": penalize_dof_torque,
        }

        total_reward = torch.prod(torch.stack(list(tracking_rewards.values())), dim=0) + \
                       torch.prod(torch.stack(list(penalty_rewards.values())), dim=0)

        logging_rewards = {
            "track_orientation_reward": track_orientation_reward,
            "track_height_reward": track_height_reward,
            "track_lin_vel_reward": track_lin_vel_reward,
            "track_ang_vel_reward": track_ang_vel_reward,
            "track_ee_pos_fl_reward": track_ee_pos_fl_reward,
            "track_ee_pos_fr_reward": track_ee_pos_fr_reward,

            "penalize_contact_mismatch": penalize_contact_mismatch,
            "penalize_action_scale": penalize_action_scale,
            "penalize_action_rate": penalize_action_rate,
            "penalize_dof_acc": penalize_dof_acc,
            "penalize_dof_torque": penalize_dof_torque,
        }

        for key, value in logging_rewards.items():
            self._episode_sums[key] += value

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        is_unsafe = torch.zeros_like(time_out, dtype=torch.bool)

        if self.cfg.terminate_on_low_base_height:
            low_base_height = self._robot.data.root_pos_w[:, 2] < 0.3
            is_unsafe = torch.logical_or(is_unsafe, low_base_height)

        if self.cfg.terminate_on_large_orientation_error:
            gravity_vec_b = self._robot.data.projected_gravity_b.clone()
            gravity_vec_b_target = torch.zeros_like(gravity_vec_b)
            gravity_vec_b_target[:, 0] = -1.0
            orientation_error = torch.norm(gravity_vec_b - gravity_vec_b_target, dim=-1)
            is_unsafe = torch.logical_or(is_unsafe, orientation_error > 0.8)

        if self.cfg.terminate_on_undesired_foot_contact:
            undesired_foot_contacts = torch.sum(torch.logical_and(self.foot_contacts,
                                                                  torch.logical_not(
                                                                      self.contact_generator.desired_contact_state)),
                                                dim=-1) > 0
            is_unsafe = torch.logical_or(is_unsafe, undesired_foot_contacts)

        if self.cfg.terminate_on_body_contact:
            body_contact = torch.logical_or(self.has_body_contact, self.has_head_contact)
            is_unsafe = torch.logical_or(is_unsafe, body_contact)

        if self.cfg.terminate_on_limb_contact:
            limb_contact = torch.logical_or(self.calf_contacts, self.thigh_contacts)
            limb_contact = torch.sum(limb_contact, dim=1)
            is_unsafe = torch.logical_or(is_unsafe, limb_contact > 0)

        return is_unsafe, time_out

    def _pre_physics_step(self, action):
        pass

    def _apply_action(self, action):
        # not used
        pass

    def get_motor_angles_from_foot_positions(self, foot_local_positions):
        return motor_angles_from_foot_positions(foot_local_positions, self.hip_offset, self.link_lengths,
                                                device=self.device)

    def get_motor_angles_from_foot_positions_fl(self, foot_local_positions):
        return motor_angles_from_foot_positions_fl(foot_local_positions, self.hip_offset, self.link_lengths,
                                                   device=self.device)

    def get_motor_angles_from_foot_positions_fr(self, foot_local_positions):
        return motor_angles_from_foot_positions_fr(foot_local_positions, self.hip_offset, self.link_lengths,
                                                   device=self.device)

    @property
    def time_since_reset(self):
        return self._time_since_reset.clone()

    @property
    def generalized_coordinates(self):
        base_pos = self._robot.data.root_pos_w.clone()  # 0:3
        base_quat = self._robot.data.root_quat_w.clone()  # 3:7
        joint_pos = self._robot.data.joint_pos.clone()  # 7:19
        base_lin_vel = self._robot.data.root_lin_vel_w.clone()  # 19:22
        base_ang_vel = self._robot.data.root_ang_vel_w.clone()  # 22:25
        joint_vel = self._robot.data.joint_vel.clone()  # 25:37

        q = torch.cat([base_pos, base_quat, joint_pos, base_lin_vel, base_ang_vel, joint_vel], dim=1)

        if self.cfg.obs_noise:
            noise_xy = 0.05
            noise_z = 0.05
            noise_quat = 0.02
            noise_qj = 0.01
            noise_vl = 0.1
            noise_va = 0.15
            noise_vj = 1.5
            q[:, 0:2] += torch.rand_like(q[:, 0:2]) * 2 * noise_xy - noise_xy  # x, y
            q[:, 2] += torch.rand_like(q[:, 2]) * 2 * noise_z - noise_z  # z
            q[:, 3:7] += torch.rand_like(q[:, 3:7]) * 2 * noise_quat - noise_quat  # quat
            q[:, 3:7] = math_utils.normalize(q[:, 3:7])
            q[:, 7:19] += torch.rand_like(q[:, 7:19]) * 2 * noise_qj - noise_qj  # joint pos
            q[:, 19:22] += torch.rand_like(q[:, 19:22]) * 2 * noise_vl - noise_vl  # lin vel
            q[:, 22:25] += torch.rand_like(q[:, 22:25]) * 2 * noise_va - noise_va  # ang vel
            q[:, 25:37] += torch.rand_like(q[:, 25:37]) * 2 * noise_vj - noise_vj  # joint vel
        return q

    @property
    def base_transform_zrp(self):
        transform = torch.zeros(self.num_envs, 4, 4, device=self.device)
        transform[:, 3, 3] = 1.0
        transform[:, :3, :3] = self.base_rot_mat_rp
        transform[:, 2, 3] = self.base_height
        return transform

    @property
    def base_pos_w(self):
        return self.generalized_coordinates[:, :3].clone()

    @property
    def base_height(self):
        return self.base_pos_w[:, 2].clone()

    @property
    def base_quat(self):
        return self.generalized_coordinates[:, 3:7].clone()

    @property
    def base_rot_mat_rp(self):
        # R (root-base)
        # p_root = R * p_base
        return self.base_rot_mat_rp_t.transpose(1, 2)

    @property
    def base_quat_rp(self):
        # R (root-base)
        # p_root = R * p_base
        return math_utils.quat_from_matrix(self.base_rot_mat_rp)

    @property
    def base_rot_mat_rp_t(self):
        # R (base-root)
        # p_base = R * p_root
        return math_utils.rp_rotation_from_gravity_b(self.projected_gravity_b)

    @property
    def projected_gravity_b(self):
        # from GC
        gravity_vec_w = torch.tensor((0.0, 0.0, -1.0), device=self.device).repeat(self.num_envs, 1)
        return math_utils.quat_rotate_inverse(self.base_quat, gravity_vec_w)

    @property
    def base_lin_vel_w(self):
        # from GC
        return self.generalized_coordinates[:, 19:22].clone()

    @property
    def base_ang_vel_w(self):
        # from GC
        return self.generalized_coordinates[:, 22:25].clone()

    @property
    def base_lin_vel_b(self):
        # from GC
        return math_utils.quat_rotate_inverse(self.base_quat, self.base_lin_vel_w)

    @property
    def base_ang_vel_b(self):
        # from GC
        return math_utils.quat_rotate_inverse(self.base_quat, self.base_ang_vel_w)

    @property
    def joint_pos(self):
        # from GC
        return self.generalized_coordinates[:, 7:19].clone()

    @property
    def joint_vel(self):
        # from GC
        return self.generalized_coordinates[:, 25:37].clone()

    @property
    def ee_pos_b(self):
        # #envs x #feet x 3
        # depends on joint pos only, from GC
        # analytical solution
        theta_hip = self.joint_pos[:, 0:4]
        theta_thigh = self.joint_pos[:, 4:8]
        theta_knee = self.joint_pos[:, 8:12]
        l_hip = torch.tensor([1, -1, 1, -1], device=self.device) * self.link_lengths[0]
        l_up = self.link_lengths[1]
        l_low = self.link_lengths[2]

        leg_distance = torch.sqrt(l_up ** 2 + l_low ** 2 +
                                  2 * l_up * l_low * torch.cos(theta_knee))
        eff_swing = theta_thigh + theta_knee / 2

        off_x_hip = -(leg_distance * torch.sin(eff_swing)).view(self.num_envs, 4)
        off_z_hip = -(leg_distance * torch.cos(eff_swing)).view(self.num_envs, 4)
        off_y_hip = l_hip.unsqueeze(0).repeat(self.num_envs, 1)

        off_x = off_x_hip
        off_y = torch.cos(theta_hip.view(self.num_envs, 4)) * off_y_hip - \
                torch.sin(theta_hip.view(self.num_envs, 4)) * off_z_hip
        off_z = torch.sin(theta_hip.view(self.num_envs, 4)) * off_y_hip + \
                torch.cos(theta_hip.view(self.num_envs, 4)) * off_z_hip

        ee_pos_hip = torch.stack([off_x, off_y, off_z], dim=2)
        cal_ee_pos_b = ee_pos_hip + self.hip_offset.unsqueeze(0)

        return cal_ee_pos_b

    @property
    def all_foot_jacobian(self):
        # depends on joint pos only, from GC
        # analytical solution
        theta_hip = self.joint_pos[:, 0:4]
        theta_thigh = self.joint_pos[:, 4:8]
        theta_knee = self.joint_pos[:, 8:12]

        l_hip = torch.tensor([1, -1, 1, -1], device=self.device) * self.link_lengths[0]
        l_up = self.link_lengths[1]
        l_low = self.link_lengths[2]
        l_eff = torch.sqrt(l_up ** 2 + l_low ** 2 +
                           2 * l_up * l_low * torch.cos(theta_knee))
        t_eff = theta_thigh + theta_knee / 2

        J = torch.zeros((self.num_envs, 4, 3, 3), device=self.device)
        J[:, :, 0, 0] = 0
        J[:, :, 0, 1] = -l_eff * torch.cos(t_eff)
        J[:, :, 0, 2] = l_low * l_up * torch.sin(theta_knee) * torch.sin(
            t_eff) / l_eff - l_eff * torch.cos(t_eff) / 2
        J[:, :, 1, 0] = -l_hip * torch.sin(theta_hip) + l_eff * torch.cos(theta_hip) * torch.cos(t_eff)
        J[:, :, 1, 1] = -l_eff * torch.sin(theta_hip) * torch.sin(t_eff)
        J[:, :, 1, 2] = -l_low * l_up * torch.sin(theta_hip) * torch.sin(theta_knee) * torch.cos(
            t_eff) / l_eff - l_eff * torch.sin(theta_hip) * torch.sin(t_eff) / 2
        J[:, :, 2, 0] = l_hip * torch.cos(theta_hip) + l_eff * torch.sin(theta_hip) * torch.cos(t_eff)
        J[:, :, 2, 1] = l_eff * torch.sin(t_eff) * torch.cos(theta_hip)
        J[:, :, 2, 2] = l_low * l_up * torch.sin(theta_knee) * torch.cos(theta_hip) * torch.cos(
            t_eff) / l_eff + l_eff * torch.sin(t_eff) * torch.cos(theta_hip) / 2

        flattened_jacobian = torch.zeros((self.num_envs, 12, 12), device=self.device)
        flattened_jacobian[:, 0:3, [0, 4, 8]] = J[:, 0]
        flattened_jacobian[:, 3:6, [1, 5, 9]] = J[:, 1]
        flattened_jacobian[:, 6:9, [2, 6, 10]] = J[:, 2]
        flattened_jacobian[:, 9:12, [3, 7, 11]] = J[:, 3]
        return flattened_jacobian

    def get_gravity_compensation_torques(self):
        jacobian = self.jacobian
        masses = self.masses
        num_envs = jacobian.shape[0]
        num_joints = jacobian.shape[3] - 6

        limb_body_ids = [1, 2, 4, 5,  # hip
                         6, 7, 9, 10,  # thigh
                         11, 12, 13, 14,  # calf
                         15, 16, 17, 18]  # foot

        gravity = torch.tensor([0.0, 0.0, 9.81], device=self.device)

        gravity_torques = torch.zeros((num_envs, num_joints), device=self.device)
        for i in range(len(limb_body_ids)):
            idx = limb_body_ids[i]
            body_mass = masses[:, idx]
            body_jacobian = jacobian[:, i, :3, 6:]
            body_gravity_torques = torch.bmm(body_jacobian.permute(0, 2, 1),
                                             (body_mass.unsqueeze(-1) * gravity.unsqueeze(0)).unsqueeze(
                                                 -1)).squeeze(-1)
            gravity_torques += body_gravity_torques

        return gravity_torques

    ######################################################################################
    @property
    def masses(self):
        # fixed values
        return self._robot.root_physx_view.get_masses().clone().to(self.device)

    @property
    def inertias(self):
        # local frame, fixed values
        return self._robot.root_physx_view.get_inertias().clone().to(self.device)

    @property
    def total_mass(self):
        # sum of all bodies
        return self.masses.sum(dim=1)

    @property
    def contact_forces(self):
        return self._contact_sensor.data.net_forces_w_history.clone()

    @property
    def has_body_contact(self):
        return torch.any(
            torch.max(torch.norm(self.contact_forces[:, :, self._contact_base_id], dim=-1), dim=1)[0] > 0.0, dim=1)

    @property
    def has_head_contact(self):
        return torch.any(
            torch.max(torch.norm(self.contact_forces[:, :, self._contact_head_id], dim=-1), dim=1)[0] > 0.0, dim=1)

    @property
    def foot_contacts(self):
        # on hardware, we have contact sensors
        return torch.max(torch.norm(self.contact_forces[:, :, self._contact_feet_ids], dim=-1), dim=1)[0] > 0.0

    @property
    def foot_contact_forces(self):
        return torch.max(self.contact_forces[:, :, self._contact_feet_ids], dim=1)[0]

    @property
    def calf_contacts(self):
        return torch.max(torch.norm(self.contact_forces[:, :, self._contact_calf_ids], dim=-1), dim=1)[0] > 0.0

    @property
    def calf_contact_forces(self):
        return torch.max(self.contact_forces[:, :, self._contact_calf_ids], dim=1)[0]

    @property
    def thigh_contacts(self):
        return torch.max(torch.norm(self.contact_forces[:, :, self._contact_thigh_ids], dim=-1), dim=1)[0] > 0.0

    @property
    def thigh_contact_forces(self):
        return torch.max(self.contact_forces[:, :, self._contact_thigh_ids], dim=1)[0]

    @property
    def coms_pos_w(self):
        # COM is not exactly at the origin of the local frame!
        body_pos_w = self._robot.data.body_state_w.clone()[:, :, :3].clone()
        com_offset_b = self._robot.root_physx_view.get_coms().clone().to(self.device)[:, :, :3]
        com_offset_w = math_utils.quat_rotate(self.coms_quat.view(-1, 4), com_offset_b.view(-1, 3)).view(self.num_envs,
                                                                                                         -1, 3)
        return body_pos_w + com_offset_w

    @property
    def coms_quat(self):
        # local com frame is not exactly the same as the local frame, but they are almost the same!
        body_quat = self._robot.data.body_state_w[:, :, 3:7].clone()  # of the body frame
        return body_quat

    @property
    def com_pos_w(self):
        masses = self.masses
        coms = self.coms_pos_w
        return torch.sum(masses.unsqueeze(-1) * coms, dim=1) / masses.sum(dim=1)[:, None]

    @property
    def com_quat(self):
        # body 0 is the base
        return self._robot.data.body_state_w[:, 0, 3:7].clone()  # of the body frame

    @property
    def jacobian(self):
        # from simulation, in the inertia frame
        # #envs x #bodies x 6 x (6 + nj) / #envs x #bodies x (3lin+3ang) x (3blin+3bang+nj)
        # body:
        # base,
        # FL_hip, FR_hip, RL_hip, RR_hip,
        # FL_thigh, FR_thigh, RL_thigh, RR_thigh,
        # FL_calf, FR_calf, RL_calf, RR_calf
        # FL_foot, FR_foot, RL_foot, RR_foot
        # joints:
        # FL_hip, FR_hip, RL_hip, RR_hip,
        # FL_thigh, FR_thigh, RL_thigh, RR_thigh,
        # FL_calf, FR_calf, RL_calf, RR_calf
        jacobian_sim = self._robot.root_physx_view.get_jacobians().clone().to(self.device)
        body_ids = [0,  # base
                    1, 2, 4, 5,  # hip
                    6, 7, 9, 10,  # thigh
                    11, 12, 13, 14,  # calf
                    15, 16, 17, 18]  # foot
        jacobian_sim_clean = jacobian_sim[:, body_ids, :, :]
        return jacobian_sim_clean

    ###############################################################
    def set_debug_vis(self, velocity_debug_vis: bool, pos_debug_vis: bool, force_debug_vis: bool) -> bool:
        if not self.has_debug_vis_implementation:
            return False
        # toggle debug visualization objects
        self._set_debug_vis_impl(velocity_debug_vis, pos_debug_vis, force_debug_vis)
        # toggle debug visualization handles
        if velocity_debug_vis or pos_debug_vis or force_debug_vis:
            # create a subscriber for the post update event if it doesn't exist
            if self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            # remove the subscriber if it exists
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
        # return success
        return True

    def _set_debug_vis_impl(self, velocity_debug_vis: bool, pos_debug_vis: bool, force_debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if velocity_debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "base_vel_goal_visualizer"):
                # -- goal
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                # -- current
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)

        if pos_debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "ee_pos_fl_goal_visualizer"):
                # -- goal
                marker_cfg = GREEN_SPHERE_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/position_goal_fl"
                marker_cfg.markers["sphere"].radius = 0.04
                self.ee_pos_fl_goal_visualizer = VisualizationMarkers(marker_cfg)
                # -- current
                marker_cfg = BLUE_SPHERE_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/position_current_fl"
                marker_cfg.markers["sphere"].radius = 0.04
                self.ee_pos_fl_visualizer = VisualizationMarkers(marker_cfg)

                marker_cfg = YELLOW_SPHERE_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/position_goal_fr"
                marker_cfg.markers["sphere"].radius = 0.04
                self.ee_pos_fr_goal_visualizer = VisualizationMarkers(marker_cfg)
                # -- current
                marker_cfg = CYAN_SPHERE_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/position_current_fr"
                marker_cfg.markers["sphere"].radius = 0.04
                self.ee_pos_fr_visualizer = VisualizationMarkers(marker_cfg)

            # set their visibility to true
            self.ee_pos_fl_goal_visualizer.set_visibility(True)
            self.ee_pos_fr_goal_visualizer.set_visibility(True)
            self.ee_pos_fl_visualizer.set_visibility(True)
            self.ee_pos_fr_visualizer.set_visibility(True)
        else:
            if hasattr(self, "ee_pos_fl_goal_visualizer"):
                self.ee_pos_fl_goal_visualizer.set_visibility(False)
                self.ee_pos_fr_goal_visualizer.set_visibility(False)
                self.ee_pos_fl_visualizer.set_visibility(False)
                self.ee_pos_fr_visualizer.set_visibility(False)

        if force_debug_vis:
            if not hasattr(self, "desired_force_fl_visualizer"):
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/desired_force_fl"
                marker_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
                self.desired_force_fl_visualizer = VisualizationMarkers(marker_cfg)
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/desired_force_fr"
                marker_cfg.markers["arrow"].scale = (0.3, 0.3, 0.3)
                self.desired_force_fr_visualizer = VisualizationMarkers(marker_cfg)
            self.desired_force_fl_visualizer.set_visibility(True)
            self.desired_force_fr_visualizer.set_visibility(True)
        else:
            if hasattr(self, "desired_force_fl_visualizer"):
                self.desired_force_fl_visualizer.set_visibility(False)
                self.desired_force_fr_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if self.cfg.velocity_debug_vis:
            # get marker location
            # -- base state
            base_pos_w = self._robot.data.root_pos_w.clone()
            base_pos_w[:, 2] += 1.0
            # -- resolve the scales and quaternions

            vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self._velocity_commands[:, :2])
            vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.base_lin_vel_b[:, :2])
            # display markers
            self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
            self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

        if self.cfg.pos_debug_vis:
            # FL
            ee_pos = self._robot.data.body_state_w[:, self.feet_ids, 0:3].clone()
            ee_pos_FL = ee_pos[:, 0]
            ee_pos_FR = ee_pos[:, 1]

            # in projected com frame
            base_pos_w = self._robot.data.root_pos_w.clone()

            gravity_vec_w = torch.tensor((0.0, 0.0, -1.0), device=self.device).repeat(self.num_envs, 1)
            projected_gravity_b = math_utils.quat_rotate_inverse(self._robot.data.root_quat_w, gravity_vec_w)
            base_rot_mat_rp_t = math_utils.rp_rotation_from_gravity_b(projected_gravity_b)
            base_quat_rp = math_utils.quat_from_matrix(base_rot_mat_rp_t.transpose(1, 2))

            ee_pos_target_fl_com = self._ee_pos_fl_commands
            ee_pos_target_fl = math_utils.quat_rotate(
                math_utils.quat_mul(self._robot.data.root_quat_w, math_utils.quat_inv(base_quat_rp)),
                ee_pos_target_fl_com)
            ee_pos_target_fl[:, :2] += base_pos_w[:, :2]

            ee_pos_target_fr_com = self._ee_pos_fr_commands
            ee_pos_target_fr = math_utils.quat_rotate(
                math_utils.quat_mul(self._robot.data.root_quat_w, math_utils.quat_inv(base_quat_rp)),
                ee_pos_target_fr_com)
            ee_pos_target_fr[:, :2] += base_pos_w[:, :2]

            # display markers
            self.ee_pos_fl_goal_visualizer.visualize(translations=ee_pos_target_fl)
            self.ee_pos_fr_goal_visualizer.visualize(translations=ee_pos_target_fr)
            self.ee_pos_fl_visualizer.visualize(translations=ee_pos_FL)
            self.ee_pos_fr_visualizer.visualize(translations=ee_pos_FR)

        if self.cfg.force_debug_vis:
            desired_force_com_fl = self._ee_force_fl_commands  # projected com frame
            desired_force_com_fr = self._ee_force_fr_commands  # projected com frame

            gravity_vec_w = torch.tensor((0.0, 0.0, -1.0), device=self.device).repeat(self.num_envs, 1)
            projected_gravity_b = math_utils.quat_rotate_inverse(self._robot.data.root_quat_w, gravity_vec_w)
            base_rot_mat_rp_t = math_utils.rp_rotation_from_gravity_b(projected_gravity_b)
            base_quat_rp = math_utils.quat_from_matrix(base_rot_mat_rp_t.transpose(1, 2))

            desired_force_w_fl = math_utils.quat_rotate(
                math_utils.quat_mul(self._robot.data.root_quat_w, math_utils.quat_inv(base_quat_rp)),
                desired_force_com_fl)
            desired_force_w_fr = math_utils.quat_rotate(
                math_utils.quat_mul(self._robot.data.root_quat_w, math_utils.quat_inv(base_quat_rp)),
                desired_force_com_fr)

            scale_fl, quat_fl = self._resolve_scale_and_quat_from_vector(
                self.desired_force_fl_visualizer.cfg.markers["arrow"].scale,
                desired_force_w_fl)
            scale_fr, quat_fr = self._resolve_scale_and_quat_from_vector(
                self.desired_force_fr_visualizer.cfg.markers["arrow"].scale,
                desired_force_w_fr)

            base_pos_w = self._robot.data.root_pos_w.clone()
            ee_pos_target_com_fl = self._ee_pos_fl_commands
            ee_pos_target_com_fr = self._ee_pos_fr_commands

            gravity_vec_w = torch.tensor((0.0, 0.0, -1.0), device=self.device).repeat(self.num_envs, 1)
            projected_gravity_b = math_utils.quat_rotate_inverse(self._robot.data.root_quat_w, gravity_vec_w)
            base_rot_mat_rp_t = math_utils.rp_rotation_from_gravity_b(projected_gravity_b)
            base_quat_rp = math_utils.quat_from_matrix(base_rot_mat_rp_t.transpose(1, 2))

            ee_pos_target_w_fl = math_utils.quat_rotate(
                math_utils.quat_mul(self._robot.data.root_quat_w, math_utils.quat_inv(base_quat_rp)),
                ee_pos_target_com_fl)
            ee_pos_target_w_fl[:, :2] += base_pos_w[:, :2]
            pos_fl = ee_pos_target_w_fl

            ee_pos_target_w_fr = math_utils.quat_rotate(
                math_utils.quat_mul(self._robot.data.root_quat_w, math_utils.quat_inv(base_quat_rp)),
                ee_pos_target_com_fr)
            ee_pos_target_w_fr[:, :2] += base_pos_w[:, :2]
            pos_fr = ee_pos_target_w_fr

            self.desired_force_fl_visualizer.visualize(pos_fl, quat_fl, scale_fl)
            self.desired_force_fr_visualizer.visualize(pos_fr, quat_fr, scale_fr)

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        arrow_quat = math_utils.quat_mul(math_utils.yaw_quat(self._robot.data.root_quat_w), arrow_quat)

        return arrow_scale, arrow_quat

    def _resolve_scale_and_quat_from_vector(self, default_scale, vector):
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=vector.device).repeat(vector.shape[0], 1)
        arrow_scale[:, 0] *= torch.norm(vector, dim=-1) * 0.15

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
def motor_angles_from_foot_positions_fl(foot_local_positions,
                                        hip_offset,
                                        link_lengths,
                                        device: str = "cuda"):
    foot_positions_in_hip_frame = foot_local_positions - hip_offset[0]
    l_hip = link_lengths[0]
    l_up = link_lengths[1]
    l_low = link_lengths[2]

    x = foot_positions_in_hip_frame[:, 0]
    y = foot_positions_in_hip_frame[:, 1]
    z = foot_positions_in_hip_frame[:, 2]

    theta_calf = -torch.arccos(
        torch.clip((x ** 2 + y ** 2 + z ** 2 - l_hip ** 2 - l_low ** 2 - l_up ** 2) /
                   (2 * l_low * l_up), -1, 1))
    l = torch.sqrt(
        torch.clip(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * torch.cos(theta_calf),
                   1e-7, 1))
    theta_thigh = torch.arcsin(torch.clip(-x / l, -1, 1)) - theta_calf / 2
    c1 = l_hip * y - l * torch.cos(theta_thigh + theta_calf / 2) * z
    s1 = l * torch.cos(theta_thigh + theta_calf / 2) * y + l_hip * z
    theta_hip = torch.arctan2(s1, c1)

    # thetas: num_envs x 4
    joint_angles = torch.stack([
        theta_hip[:, None], theta_thigh[:, None], theta_calf[:, None]], dim=-1)
    return math_utils.wrap_to_pi(joint_angles.reshape((-1, 3)))


@torch.jit.script
def motor_angles_from_foot_positions_fr(foot_local_positions,
                                        hip_offset,
                                        link_lengths,
                                        device: str = "cuda"):
    foot_positions_in_hip_frame = foot_local_positions - hip_offset[1]
    l_hip = -link_lengths[0]
    l_up = link_lengths[1]
    l_low = link_lengths[2]

    x = foot_positions_in_hip_frame[:, 0]
    y = foot_positions_in_hip_frame[:, 1]
    z = foot_positions_in_hip_frame[:, 2]

    theta_calf = -torch.arccos(
        torch.clip((x ** 2 + y ** 2 + z ** 2 - l_hip ** 2 - l_low ** 2 - l_up ** 2) /
                   (2 * l_low * l_up), -1, 1))
    l = torch.sqrt(
        torch.clip(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * torch.cos(theta_calf),
                   1e-7, 1))
    theta_thigh = torch.arcsin(torch.clip(-x / l, -1, 1)) - theta_calf / 2
    c1 = l_hip * y - l * torch.cos(theta_thigh + theta_calf / 2) * z
    s1 = l * torch.cos(theta_thigh + theta_calf / 2) * y + l_hip * z
    theta_hip = torch.arctan2(s1, c1)

    # thetas: num_envs x 4
    joint_angles = torch.stack([
        theta_hip[:, None], theta_thigh[:, None], theta_calf[:, None]], dim=-1)
    return math_utils.wrap_to_pi(joint_angles.reshape((-1, 3)))


@torch.jit.script
def motor_angles_from_foot_positions(foot_local_positions,
                                     hip_offset,
                                     link_lengths,
                                     device: str = "cuda"):
    foot_positions_in_hip_frame = foot_local_positions - hip_offset
    l_hip = link_lengths[0] * torch.tensor([1, -1, 1, -1], device=device)
    l_up = link_lengths[1]
    l_low = link_lengths[2]

    x = foot_positions_in_hip_frame[:, :, 0]
    y = foot_positions_in_hip_frame[:, :, 1]
    z = foot_positions_in_hip_frame[:, :, 2]

    theta_calf = -torch.arccos(
        torch.clip((x ** 2 + y ** 2 + z ** 2 - l_hip ** 2 - l_low ** 2 - l_up ** 2) /
                   (2 * l_low * l_up), -1, 1))
    l = torch.sqrt(
        torch.clip(l_up ** 2 + l_low ** 2 + 2 * l_up * l_low * torch.cos(theta_calf),
                   1e-7, 1))
    theta_thigh = torch.arcsin(torch.clip(-x / l, -1, 1)) - theta_calf / 2
    c1 = l_hip * y - l * torch.cos(theta_thigh + theta_calf / 2) * z
    s1 = l * torch.cos(theta_thigh + theta_calf / 2) * y + l_hip * z
    theta_hip = torch.arctan2(s1, c1)

    # thetas: num_envs x 4
    joint_angles = torch.stack([
        theta_hip[:, 0, None], theta_hip[:, 1, None], theta_hip[:, 2, None], theta_hip[:, 3, None],
        theta_thigh[:, 0, None], theta_thigh[:, 1, None], theta_thigh[:, 2, None], theta_thigh[:, 3, None],
        theta_calf[:, 0, None], theta_calf[:, 1, None], theta_calf[:, 2, None], theta_calf[:, 3, None], ], dim=-1)
    return math_utils.wrap_to_pi(joint_angles.reshape((-1, 12)))
