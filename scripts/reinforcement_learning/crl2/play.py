# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from CRL2."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with CRL2.")

# environment arguments
parser.add_argument("--disable_fabric", action="store_true", default=False,
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

# -- experiment arguments
parser.add_argument("--experiment_name", type=str, default=None,
                    help="Name of the experiment folder where logs will be stored. By default, it is the task name.")
# -- load arguments
parser.add_argument("--load_run", type=str, default=None, help="Name of the run folder to load from.")
parser.add_argument("--load_checkpoint", type=str, default=None, help="Checkpoint file to load from.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch

import isaacsim.core.utils.torch as torch_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.crl2 import Crl2VecEnvWrapper

from crl2.algorithms import *
from crl2.utils.helpers import get_load_path, export_policy_as_jit, export_policy_as_onnx

EXPORT = False


def main():
    """Play with CRL2 agent."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric)

    env_cfg.scene.env_spacing = 10.0
    env_cfg.viewer.eye = [2.5, 2.5, 2.0]
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"

    agent_cfg = load_cfg_from_registry(args_cli.task, "crl2_cfg_entry_point")
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
    if args_cli.num_envs is not None:
        agent_cfg["general"]["num_envs"] = args_cli.num_envs
    if args_cli.experiment_name is not None:
        agent_cfg["general"]["experiment_name"] = args_cli.experiment_name
    if args_cli.load_run is not None:
        agent_cfg["general"]["load_run"] = args_cli.load_run
    else:
        agent_cfg["general"]["load_run"] = -1
    if args_cli.load_checkpoint is not None:
        agent_cfg["general"]["load_checkpoint"] = args_cli.load_checkpoint
    else:
        agent_cfg["general"]["load_checkpoint"] = -1

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "crl2", agent_cfg["general"]["experiment_name"])
    log_root_path = os.path.abspath(log_root_path)
    load_path = get_load_path(log_root_path, agent_cfg["general"]["load_run"],
                              agent_cfg["general"]["load_checkpoint"])

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = Crl2VecEnvWrapper(env)

    # set seed of the environment
    torch_utils.set_seed(agent_cfg["seed"])
    env.seed(agent_cfg["seed"])

    # create algorithm runner
    algorithm = eval(agent_cfg["algorithm"]["name"])
    runner = algorithm(task=args_cli.task,
                       env=env,
                       agent_cfg=agent_cfg,
                       train=False,
                       device=env.device)

    runner.load(load_path)
    print(f"[INFO] Loading model from: {load_path}.")

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs, _ = env.get_observations()

    if EXPORT:
        path = os.path.join(
            os.path.dirname(load_path),
            "exported",
        )
        name = "model"
        if agent_cfg["algorithm"]["empirical_normalization"]:
            export_policy_as_jit(runner.policy.policy_latent_net, runner.policy.action_mean_net,
                                 path, runner.obs_normalizer, filename=f"{name}.pt")
            export_policy_as_onnx(runner.policy.policy_latent_net, runner.policy.action_mean_net,
                                  path, runner.obs_normalizer, filename=f"{name}.onnx")
        else:
            export_policy_as_jit(runner.policy.policy_latent_net, runner.policy.action_mean_net,
                                 path, normalizer=None, filename=f"{name}.pt")
            export_policy_as_onnx(runner.policy.policy_latent_net, runner.policy.action_mean_net,
                                  path, normalizer=None, filename=f"{name}.onnx")
        print("--------------------------")
        print("Exported policy to: ", path)
        policy_jit_path = os.path.join(
            os.path.dirname(load_path),
            "exported",
            "model.pt"
        )
        policy_jit = torch.jit.load(policy_jit_path).to(env.unwrapped.device)
        test_input = torch.zeros_like(obs)
        print("loaded policy test output: ")
        print(policy(test_input))
        print("loaded jit policy test output: ")
        print(policy_jit(test_input))
        print("--------------------------")

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
