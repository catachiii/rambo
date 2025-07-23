import torch


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(self,
                 num_envs,
                 num_transitions_per_env,
                 num_obs,
                 num_values,
                 num_actions,
                 device='cpu'):

        self.device = device

        self.num_obs = num_obs
        self.num_values = num_values
        self.num_actions = num_actions

        self.observations = torch.zeros(num_transitions_per_env, num_envs, num_obs, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, num_actions, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        self.rewards = [torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device) for _ in
                        range(num_values)]
        self.values = [torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device) for _ in
                       range(num_values)]
        self.returns = [torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device) for _ in
                        range(num_values)]
        self.advantages = [torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device) for _ in
                           range(num_values)]

        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, num_actions, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, num_actions, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))

        for i in range(self.num_values):
            self.rewards[i][self.step].copy_(transition.rewards[i].view(-1, 1))
            self.values[i][self.step].copy_(transition.values[i].view(-1, 1))

        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = [0 for _ in range(self.num_values)]
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = [last_values[i] for i in range(self.num_values)]
            else:
                next_values = [self.values[i][step + 1] for i in range(self.num_values)]
            next_is_not_terminal = 1.0 - self.dones[step].float()

            for i in range(self.num_values):
                advantage[i] = self.rewards[i][step] + next_is_not_terminal * gamma * (
                        next_values[i] + lam * advantage[i]) - self.values[i][step]
                self.returns[i][step] = advantage[i] + self.values[i][step]

        # Compute and normalize the advantages
        for i in range(self.num_values):
            self.advantages[i] = self.returns[i] - self.values[i]
            self.advantages[i] = (self.advantages[i] - self.advantages[i].mean()) / (
                    self.advantages[i].std() + 1e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)

        values = [self.values[i].flatten(0, 1) for i in range(self.num_values)]
        returns = [self.returns[i].flatten(0, 1) for i in range(self.num_values)]
        advantages = [self.advantages[i].flatten(0, 1) for i in range(self.num_values)]

        actions = self.actions.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                actions_batch = actions[batch_idx]

                target_values_batch = [values[i][batch_idx] for i in range(self.num_values)]
                returns_batch = [returns[i][batch_idx] for i in range(self.num_values)]
                advantages_batch = [advantages[i][batch_idx] for i in range(self.num_values)]

                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                yield obs_batch, actions_batch, \
                    target_values_batch, advantages_batch, returns_batch, \
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch
