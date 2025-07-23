import math
import os
import statistics
import time
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from crl2.modules.normalizer import EmpiricalNormalization
from crl2.modules.policy import Policy
from crl2.modules.value import Value
from crl2.storage.rollout_storage import RolloutStorage

from torch.utils.tensorboard import SummaryWriter
from crl2.utils.wandb_utils import WandbSummaryWriter
from crl2.utils.video_record import VideoRecorder
from crl2.utils.helpers import class_to_dict


class PPO:
    def __init__(self,
                 task,
                 env,
                 agent_cfg,
                 log_dir=None,
                 train=True,
                 device='cpu'):
        self.task = task
        self.env = env

        self.gen_cfg = agent_cfg['general']
        self.alg_cfg = agent_cfg['algorithm']
        self.net_cfg = agent_cfg['network']

        self.log_dir = log_dir
        self.train = train
        self.device = device

        # from the environments
        self.num_envs = self.env.num_envs
        obs, extras = self.env.get_observations()
        self.num_obs = obs.shape[1]
        self.num_actions = self.env.num_actions

        self.policy = Policy(
            num_obs=self.num_obs,
            num_actions=self.num_actions,
            hidden_dims=self.net_cfg['policy_hidden'],
            activation=self.net_cfg['policy_activation'],
            log_std_init=self.net_cfg['log_std_init'],
            device=self.device
        ).to(self.device)

        self.num_values = 1
        self.values = [
            Value(
                num_obs=self.num_obs,
                hidden_dims=self.net_cfg['value_hidden'],
                activation=self.net_cfg['value_activation'],
                device=self.device
            ).to(self.device) for _ in range(self.num_values)]

        if self.alg_cfg['empirical_normalization']:
            self.obs_normalizer = EmpiricalNormalization(shape=self.num_obs, until=int(1.0e8)).to(self.device)

        if self.train:
            self.storage = RolloutStorage(num_envs=self.num_envs,
                                          num_transitions_per_env=self.alg_cfg['steps_per_env_iter'],
                                          num_obs=self.num_obs,
                                          num_values=self.num_values,
                                          num_actions=self.num_actions,
                                          device=self.device)

            self.transition = RolloutStorage.Transition()

            params = list(self.policy.parameters())
            for i in range(self.num_values):
                params += list(self.values[i].parameters())
            self.optimizer = optim.Adam(params, lr=self.alg_cfg['learning_rate'])
            self.learning_rate = self.alg_cfg['learning_rate']

            self.init_writer()
            if self.gen_cfg['video']:
                self.video_recorder = VideoRecorder(env=self.env,
                                                    video_folder=os.path.join(self.log_dir, 'videos'),
                                                    video_interval=self.gen_cfg['video_interval'],
                                                    video_length=self.gen_cfg['video_length'],
                                                    video_prefix=self.task,
                                                    video_fps=math.ceil(1 / self.env.unwrapped.step_dt))

        self.tot_steps = 0
        self.tot_time = 0
        self.current_iteration = 0
        self.save_interval = self.gen_cfg['save_interval']

    def learn(self, num_iterations=None):
        # set torch to train mode
        self.train_mode()

        new_obs, _ = self.env.reset()
        if self.alg_cfg['empirical_normalization']:
            new_obs = self.obs_normalizer(new_obs)

        if self.alg_cfg['random_episode_init']:
            self.env.episode_length_buf = torch.randint_like(input=self.env.episode_length_buf,
                                                             high=self.env.max_episode_length)

        self.prepare_log_episode_statistics()

        self.start_iter = self.current_iteration
        self.end_iter = self.start_iter + num_iterations
        for it in range(self.start_iter, self.end_iter):
            start = time.time()
            with torch.inference_mode():
                for i in range(self.alg_cfg['steps_per_env_iter']):
                    obs = new_obs  # normalized already
                    actions, log_prob = self.policy.act_and_log_prob(obs)
                    new_obs, rewards, dones, extras = self.env.step(actions)

                    rewards = [rewards]
                    self.process_env_step(obs, actions, log_prob, rewards, dones, extras)
                    if self.alg_cfg['empirical_normalization']:
                        new_obs = self.obs_normalizer(new_obs)

                    self.log_episode_statistics(rewards, dones, extras)
                    if self.gen_cfg['video']:
                        self.video_recorder.step(it)

                last_obs = obs.detach()
                last_values = [self.values[i](last_obs).detach() for i in range(self.num_values)]

                self.storage.compute_returns(last_values, self.alg_cfg['gamma'], self.alg_cfg['lam'])

            stop = time.time()
            collection_time = stop - start

            start = stop
            mean_value_loss, mean_surrogate_loss, kl_too_small, kl_too_large = self.update(it)
            stop = time.time()
            learn_time = stop - start

            self.current_iteration = it
            self.log(locals())

            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))

            self.episode_info.clear()

        self.current_iteration = self.end_iter
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.end_iter)))
        if self.gen_cfg['video']:
            self.video_recorder.close()
        if self.gen_cfg['logger'] == "wandb":
            self.writer.stop()

    def process_env_step(self,
                         obs,
                         actions,
                         log_prob,
                         rewards,
                         dones,
                         infos):
        self.transition.observations = obs.detach()

        self.transition.actions = actions.detach()
        self.transition.actions_log_prob = log_prob.detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()

        self.transition.rewards = [rewards[i].clone() for i in range(self.num_values)]
        self.transition.dones = dones
        self.transition.values = [self.values[i](obs).detach() for i in range(self.num_values)]
        if "time_outs" in infos:
            for i in range(self.num_values):
                self.transition.rewards[i] += self.alg_cfg['gamma'] * torch.squeeze(
                    self.transition.values[i] * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def update(self, it):
        mean_value_loss = [0 for _ in range(self.num_values)]
        mean_surrogate_loss = 0
        generator = self.storage.mini_batch_generator(self.alg_cfg['num_mini_batches'],
                                                      self.alg_cfg['num_learning_epochs'])
        num_updates = self.alg_cfg['num_learning_epochs'] * self.alg_cfg['num_mini_batches']
        kl_too_small = 0
        kl_too_large = 0
        for sample in generator:
            obs, actions, \
                target_values, advantages, returns, \
                old_actions_log_prob, old_mu, old_sigma = sample

            # update current evaluations
            self.policy.update_distribution(obs)
            actions_log_prob = self.policy.distribution.log_prob(actions)
            values = [self.values[i](obs) for i in range(self.num_values)]

            mu = self.policy.action_mean
            sigma = self.policy.action_std
            entropy = self.policy.entropy

            # adaptively change the lr for PPO using KL divergence
            if self.alg_cfg['desired_kl'] is not None and self.alg_cfg['schedule'] == 'adaptive':
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma / old_sigma + 1.e-5) + (
                                torch.square(old_sigma) + torch.square(old_mu - mu)) / (
                                2.0 * torch.square(sigma)) - 0.5, dim=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.alg_cfg['desired_kl'] * self.alg_cfg['schedule_compare']:
                        kl_too_large += 1
                        self.learning_rate = max(self.alg_cfg['min_learning_rate'],
                                                 self.learning_rate / self.alg_cfg['schedule_multiplier'])
                    elif self.alg_cfg['desired_kl'] / self.alg_cfg['schedule_compare'] > kl_mean > 0.0:
                        kl_too_small += 1
                        self.learning_rate = min(self.alg_cfg['max_learning_rate'],
                                                 self.learning_rate * self.alg_cfg['schedule_multiplier'])

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.learning_rate

            # Surrogate loss
            total_advantages = advantages[0]  # have one value function only
            ratio = torch.exp(actions_log_prob - torch.squeeze(old_actions_log_prob))
            surrogate = -torch.squeeze(total_advantages) * ratio
            surrogate_clipped = -torch.squeeze(total_advantages) * torch.clamp(ratio, 1.0 - self.alg_cfg['clip_ratio'],
                                                                               1.0 + self.alg_cfg['clip_ratio'])
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            value_loss = [0 for _ in range(self.num_values)]
            if self.alg_cfg['clip_value_target']:
                for i in range(self.num_values):
                    value_clipped = target_values[i] + (values[i] - target_values[i]).clamp(
                        -self.alg_cfg['clip_value'],
                        self.alg_cfg['clip_value'])
                    value_loss[i] = torch.max((values[i] - returns[i]).pow(2),
                                              (value_clipped - returns[i]).pow(2)).mean()
            else:
                for i in range(self.num_values):
                    value_loss[i] = (returns[i] - values[i]).pow(2).mean()

            # Compute total loss.
            loss = 0
            for i in range(self.num_values):
                loss += self.alg_cfg['value_loss_coef'] * value_loss[i]
            loss += surrogate_loss - self.alg_cfg['entropy_coef'] * entropy.mean()

            # Gradient step
            self.optimizer.zero_grad()
            loss.backward()

            params_list = list(self.policy.parameters())
            for i in range(self.num_values):
                params_list += list(self.values[i].parameters())
            nn.utils.clip_grad_norm_(params_list, self.alg_cfg['clip_grad_norm'])

            self.optimizer.step()

            for i in range(self.num_values):
                mean_value_loss[i] += value_loss[i].item()
            mean_surrogate_loss += surrogate_loss.item()

        mean_value_loss = [i / num_updates for i in mean_value_loss]
        mean_surrogate_loss /= num_updates

        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, kl_too_small, kl_too_large

    def save(self, path, infos=None):
        save_dict = {
            'policy_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iteration': self.current_iteration,
            'infos': infos,
        }
        for i in range(self.num_values):
            save_dict['value_dict_{}'.format(i)] = self.values[i].state_dict()
        if self.alg_cfg['empirical_normalization']:
            save_dict["obs_normalizer"] = self.obs_normalizer.state_dict()
        torch.save(save_dict, path)

    def load(self, path, load_values=False, load_optimizer=False):
        loaded_dict = torch.load(path)
        self.policy.load_state_dict(loaded_dict['policy_dict'])
        if load_values:
            for i in range(self.num_values):
                self.values[i].load_state_dict(loaded_dict['value_dict_{}'.format(i)])
        if load_optimizer:
            self.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        if self.alg_cfg['empirical_normalization']:
            self.obs_normalizer.load_state_dict(loaded_dict['obs_normalizer'])
        self.current_iteration = loaded_dict['iteration']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.policy.to(device)
            if self.alg_cfg['empirical_normalization']:
                self.obs_normalizer.to(device)

        def inference_policy(x):
            if self.alg_cfg['empirical_normalization']:
                normed_obs = self.obs_normalizer(x)
                return self.policy.act_inference(normed_obs)
            else:
                return self.policy.act_inference(x)

        return inference_policy

    def train_mode(self):
        self.policy.train()
        for i in range(self.num_values):
            self.values[i].train()
        if self.alg_cfg['empirical_normalization']:
            self.obs_normalizer.train()

    def eval_mode(self):
        self.policy.eval()
        for i in range(self.num_values):
            self.values[i].eval()
        if self.alg_cfg['empirical_normalization']:
            self.obs_normalizer.eval()

    def init_writer(self):
        # initialize writer
        if self.gen_cfg['logger'] == "wandb":
            config = {}
            config.update({"env_cfg": class_to_dict(self.env.cfg)})
            config.update({"general_cfg": self.gen_cfg})
            config.update({"algorithm_cfg": self.alg_cfg})
            config.update({"network_cfg": self.net_cfg})
            self.writer = WandbSummaryWriter(log_dir=self.log_dir,
                                             cfg=config,
                                             flush_secs=10,
                                             project=self.task,
                                             group=self.gen_cfg['experiment_name'],
                                             offline_mode=self.gen_cfg['offline_mode'])
        else:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

    def prepare_log_episode_statistics(self):
        self.episode_rew_sum = [torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for _ in
                                range(self.num_values)]
        self.episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.rew_buffer = [deque(maxlen=100) for _ in range(self.num_values)]
        self.len_buffer = deque(maxlen=100)
        self.episode_info = []

    def log_episode_statistics(self, rewards, dones, infos):
        if 'log' in infos:
            self.episode_info.append(infos['log'])
        new_envs = (dones > 0).nonzero(as_tuple=False)
        if len(new_envs) == 0:
            return
        else:
            for i in range(self.num_values):
                self.episode_rew_sum[i] += rewards[i]
                self.rew_buffer[i].extend(
                    self.episode_rew_sum[i][new_envs][:, 0].cpu().numpy().tolist())
                self.episode_rew_sum[i][new_envs] = 0
            self.episode_length += 1
            self.len_buffer.extend(self.episode_length[new_envs][:, 0].cpu().numpy().tolist())
            self.episode_length[new_envs] = 0

    def log(self, locs, width=100, pad=50):
        self.tot_steps += self.alg_cfg['steps_per_env_iter'] * self.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if len(self.episode_info) > 0:
            for key in self.episode_info[0]:
                info = 0
                for ep_info in self.episode_info:
                    info += ep_info[key]
                value = info / self.alg_cfg['steps_per_env_iter']
                self.writer.add_scalar(key, value, locs['it'])
                ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.policy.action_std.mean()
        fps = int(self.alg_cfg['steps_per_env_iter'] * self.num_envs / (locs['collection_time'] + locs['learn_time']))

        mean_value_losses = locs['mean_value_loss']
        for i in range(self.num_values):
            self.writer.add_scalar('Learn/value_function_loss_{}'.format(i), mean_value_losses[i], locs['it'])
        self.writer.add_scalar('Learn/surrogate_loss', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Learn/learning_rate', self.learning_rate, locs['it'])
        self.writer.add_scalar('Learn/kl_too_small', locs['kl_too_small'], locs['it'])
        self.writer.add_scalar('Learn/kl_too_large', locs['kl_too_large'], locs['it'])
        self.writer.add_scalar('Learn/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(self.len_buffer) > 0:
            for i in range(self.num_values):
                self.writer.add_scalar('Train/mean_reward_{}'.format(i), statistics.mean(self.rew_buffer[i]),
                                       locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(self.len_buffer), locs['it'])

        if self.gen_cfg['logger'] == "wandb":
            self.writer.flush_logger(locs['it'])

        str = f" \033[1m Learning iteration {locs['it'] + 1}/{self.end_iter} \033[0m "

        log_string = (f"""{'#' * width}\n"""
                      f"""{str.center(width, ' ')}\n\n"""
                      f"""{'Computation:':>{pad}} {fps:.0f} steps/s\n"""
                      f"""{'collection:' :>{pad}} {locs['collection_time']:.3f}s\n"""
                      f"""{'learning:':>{pad}} {locs['learn_time']:.3f}s\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_steps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1 - self.start_iter) * (
                               self.end_iter - locs['it']):.1f}s\n""")
        print(log_string)
