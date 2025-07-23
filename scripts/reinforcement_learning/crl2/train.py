# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with CRL2."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with CRL2.")

# environment arguments
parser.add_argument("--disable_fabric", action="store_true", default=False,
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# -- experiment arguments
parser.add_argument("--experiment_name", type=str, default=None,
                    help="Name of the experiment folder where logs will be stored. By default, it is the task name.")
parser.add_argument("--run_name", type=str, default=None,
                    help="Run name suffix to the log directory. By default, only timestamp is used.")
# -- load arguments
parser.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
parser.add_argument("--load_run", type=str, default=None, help="Name of the run folder to load from.")
parser.add_argument("--load_checkpoint", type=str, default=None, help="Checkpoint file to load from.")
# -- logger arguments
parser.add_argument("--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"},
                    help="Logger module to use.")
# -- video arguments
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=None, help="Length of the recorded video (in iterations).")
parser.add_argument("--video_interval", type=int, default=None,
                    help="Interval between video recordings (in iterations).")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

import isaacsim.core.utils.torch as torch_utils

from isaaclab.utils.io import dump_pickle, dump_yaml
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.crl2 import Crl2VecEnvWrapper

from crl2.algorithms import *
from crl2.utils.helpers import get_load_path

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    """Train with CRL2 agent."""
    # parse configuration
    agent_cfg = load_cfg_from_registry(args_cli.task, "crl2_cfg_entry_point")

    if args_cli.num_envs is not None:
        env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs,
                                use_fabric=not args_cli.disable_fabric)
    else:
        env_cfg = parse_env_cfg(args_cli.task, num_envs=agent_cfg["general"]["num_envs"],
                                use_fabric=not args_cli.disable_fabric)

    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
    if args_cli.num_envs is not None:
        agent_cfg["general"]["num_envs"] = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg["general"]["max_iterations"] = args_cli.max_iterations
    if args_cli.experiment_name is not None:
        agent_cfg["general"]["experiment_name"] = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg["general"]["run_name"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args_cli.run_name}"
    else:
        agent_cfg["general"]["run_name"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + agent_cfg["general"][
            "run_name"]
    if args_cli.resume is not None:
        agent_cfg["general"]["resume"] = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg["general"]["load_run"] = args_cli.load_run
    else:
        agent_cfg["general"]["load_run"] = -1
    if args_cli.load_checkpoint is not None:
        agent_cfg["general"]["load_checkpoint"] = args_cli.load_checkpoint
    else:
        agent_cfg["general"]["load_checkpoint"] = -1
    if args_cli.logger is not None:
        agent_cfg["general"]["logger"] = args_cli.logger
    if args_cli.video:
        agent_cfg["general"]["video"] = True

    if args_cli.video_length is not None:
        agent_cfg["general"]["video_length"] = args_cli.video_length
    if args_cli.video_interval is not None:
        agent_cfg["general"]["video_interval"] = args_cli.video_interval

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "crl2", agent_cfg["general"]["experiment_name"])
    log_root_path = os.path.abspath(log_root_path)
    log_dir = agent_cfg["general"]["run_name"]
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    resume_dir = None
    if agent_cfg["general"]["resume"]:
        # get path to previous checkpoint
        resume_dir = get_load_path(log_root_path,
                                   agent_cfg["general"]["load_run"],
                                   agent_cfg["general"]["load_checkpoint"])
        # override log directory
        log_dir = os.path.dirname(resume_dir)
        print(f"[INFO]: Loading model checkpoint from: {resume_dir}")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    print(f"[INFO] Dumped configuration.")

    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg.max_iterations = args_cli.max_iterations

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env = Crl2VecEnvWrapper(env)

    # set seed of the environment
    torch_utils.set_seed(agent_cfg["seed"])
    env.seed(agent_cfg["seed"])

    # create algorithm runner
    algorithm = eval(agent_cfg["algorithm"]["name"])
    runner = algorithm(task=args_cli.task,
                       env=env,
                       agent_cfg=agent_cfg,
                       log_dir=log_dir,
                       train=True,
                       device=env.device)

    if agent_cfg["general"]["resume"]:
        runner.load(resume_dir)

    # train
    runner.learn(agent_cfg["general"]["max_iterations"])

    # close env
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
