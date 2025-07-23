# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents
from .qp_env import QPEnv, QPEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-RAMBO-Quadruped-Go2-v0",
    entry_point="isaaclab_tasks.direct.rambo_quadruped:QPEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QPEnvCfg,
        "crl2_cfg_entry_point": f"{agents.__name__}:crl2_flat_ppo_cfg.yaml",
    },
)
