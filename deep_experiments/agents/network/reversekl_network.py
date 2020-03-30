from agents.network.base_network import BaseNetwork
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import MultivariateNormal
import quadpy
import itertools
from scipy.special import binom
import environments


class ReverseKLNetwork(BaseNetwork):
    def __init__(self, config):
        super(ReverseKLNetwork, self).__init__(config, [config.pi_lr, config.qf_vf_lr])

        self.config = config

        self.optim_type = config.optim_type

        self.use_true_q = False
        if config.use_true_q == "True":
            self.use_true_q = True

        self.rng = np.random.RandomState(config.random_seed)
        self.entropy_scale = config.entropy_scale

        if config.use_replay:
            self.batch_size = config.batch_size
        else:
            self.batch_size = 1

        # create network
        if config.env_name == 'ContinuousBandits':
            self.pi_net = LinearPolicyNetwork(self.state_dim, self.action_dim, self.action_max[0])
        else:
            self.pi_net = PolicyNetwork(self.state_dim, self.action_dim, config.actor_l1_dim, config.actor_l2_dim, self.action_max[0])

        self.q_net = SoftQNetwork(self.state_dim, self.action_dim, config.critic_l1_dim, config.critic_l2_dim)

        self.v_net = ValueNetwork(self.state_dim, config.critic_l1_dim, config.critic_l2_dim)
        self.target_v_net = ValueNetwork(self.state_dim, config.critic_l1_dim, config.critic_l2_dim)

        # copy to target_v_net
        for target_param, param in zip(self.target_v_net.parameters(), self.v_net.parameters()):
            target_param.data.copy_(param.data)

        self.device = torch.device("cpu")

        # optimizer
        self.pi_optimizer = optim.Adam(self.pi_net.parameters(), lr=self.learning_rate[0])
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate[1])
        self.v_optimizer = optim.Adam(self.v_net.parameters(), lr=self.learning_rate[1])

        dtype = torch.float32

        if self.action_dim == 1:
            self.N = config.N_param  # 1024

            scheme = quadpy.line_segment.clenshaw_curtis(self.N)
            # cut off endpoints since they should be zero but numerically might give nans
            self.intgrl_actions = (torch.tensor(scheme.points[1:-1], dtype=dtype).unsqueeze(-1) * torch.Tensor(self.action_max)).to(
                torch.float32)
            self.intgrl_weights = torch.tensor(scheme.weights[1:-1], dtype=dtype)

            self.intgrl_actions_len = np.shape(self.intgrl_actions)[0]

        else:
            self.l = config.l_param  # 5

            n_points = [1]
            for i in range(1, self.l):
                n_points.append(2 ** i + 1)

            schemes = [quadpy.line_segment.clenshaw_curtis(n_points[i]) for i in range(1, self.l)]
            points = [np.array([0.])] + [scheme.points[1:-1] for scheme in schemes]
            weights = [np.array([2.])] + [scheme.weights[1:-1] for scheme in schemes]

            # precalculate actions and weights
            self.intgrl_actions = []
            self.intgrl_weights = []

            for k in itertools.product(range(self.l), repeat=self.action_dim):
                if (np.sum(k) + self.action_dim < self.l) or (
                        np.sum(k) + self.action_dim > self.l + self.action_dim - 1):
                    continue
                coeff = (-1) ** (self.l + self.action_dim - np.sum(k) - self.action_dim + 1) * binom(
                    self.action_dim - 1, np.sum(k) + self.action_dim - self.l)

                for j in itertools.product(*[range(len(points[ki])) for ki in k]):
                    self.intgrl_actions.append(
                        torch.tensor([points[k[i]][j[i]] for i in range(self.action_dim)], dtype=dtype))
                    self.intgrl_weights.append(
                        coeff * np.prod([weights[k[i]][j[i]].squeeze() for i in range(self.action_dim)]))

            self.intgrl_weights = torch.tensor(self.intgrl_weights, dtype=dtype)
            self.intgrl_actions = torch.stack(self.intgrl_actions) * self.action_max
            self.intgrl_actions_len = np.shape(self.intgrl_actions)[0]

        self.tiled_intgrl_actions = self.intgrl_actions.unsqueeze(0).repeat(self.batch_size, 1, 1)
        self.stacked_intgrl_actions = self.tiled_intgrl_actions.reshape(-1, self.action_dim)
        self.tiled_intgrl_weights = self.intgrl_weights.unsqueeze(0).repeat(self.batch_size, 1)


    def sample_action(self, state_batch):
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action, log_prob, z, mean, log_std = self.pi_net.evaluate(state_batch)

        return action.detach().numpy()

    def predict_action(self, state_batch):

        state_batch = torch.FloatTensor(state_batch).to(self.device)

        # mean, log_std = self.pi_net(state_batch)
        _, _, _, mean, log_std = self.pi_net.evaluate(state_batch)

        return mean.detach().numpy()

    def update_network(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        gamma_batch = torch.FloatTensor(gamma_batch).to(self.device)

        reward_batch.unsqueeze_(-1)
        gamma_batch.unsqueeze_(-1)

        if not self.use_true_q:
            q_val = self.q_net(state_batch, action_batch)
            v_val = self.v_net(state_batch)
            new_action, log_prob, z, mean, log_std = self.pi_net.evaluate(state_batch)

            # q_loss, v_loss
            target_next_v_val = self.target_v_net(next_state_batch)
            target_q_val = reward_batch + gamma_batch * target_next_v_val
            q_value_loss = nn.MSELoss()(q_val, target_q_val.detach())

            new_q_val = self.q_net(state_batch, new_action)

            if self.config.q_update_type == 'sac':
                target_v_val = new_q_val - self.entropy_scale * log_prob

            elif self.config.q_update_type == 'non_sac':
                target_v_val = (reward_batch - self.entropy_scale * log_prob) + gamma_batch * target_next_v_val
            else:
                raise ValueError("invalid config.q_update_type")

            value_loss = nn.MSELoss()(v_val, target_v_val.detach())

        # pi_loss
        if self.optim_type == 'll':
            log_prob_target = new_q_val - v_val

            # loglikelihood update
            policy_loss = (-log_prob * (log_prob_target - self.entropy_scale * log_prob).detach()).mean()

        elif self.optim_type == 'hard_ll':
            log_prob_target = new_q_val - v_val
            policy_loss = (-log_prob * (log_prob_target).detach()).mean()

        elif self.optim_type == 'intg':
            stacked_state_batch = state_batch.unsqueeze(1).repeat(1, self.intgrl_actions_len, 1).reshape(-1, self.state_dim)

            if self.use_true_q:
                intgrl_q_val = torch.from_numpy(self.predict_true_q(stacked_state_batch, self.stacked_intgrl_actions)).to(torch.float32)

                intgrl_logprob = self.pi_net.get_logprob(state_batch, self.tiled_intgrl_actions)

                integrands = - torch.exp(intgrl_logprob.squeeze()) * (
                            (intgrl_q_val.squeeze()).detach() - self.entropy_scale * intgrl_logprob.squeeze())
            else:
                intgrl_q_val = self.q_net(stacked_state_batch, self.stacked_intgrl_actions)
                intgrl_v_val = v_val.unsqueeze(1).repeat(1, self.intgrl_actions_len, 1).reshape(-1, 1)

                intgrl_logprob = self.pi_net.get_logprob(state_batch, self.tiled_intgrl_actions)

                integrands = - torch.exp(intgrl_logprob.squeeze()) * ((intgrl_q_val.squeeze() - intgrl_v_val.squeeze()).detach() - self.entropy_scale * intgrl_logprob.squeeze())

            policy_loss = (integrands * self.intgrl_weights.repeat(self.batch_size)).reshape(self.batch_size, -1).sum(-1).mean(-1)

        if not self.use_true_q:
            self.q_optimizer.zero_grad()
            q_value_loss.backward()
            self.q_optimizer.step()

            self.v_optimizer.zero_grad()
            value_loss.backward()
            self.v_optimizer.step()

        self.pi_optimizer.zero_grad()
        policy_loss.backward()
        self.pi_optimizer.step()

    def update_target_network(self):
        for target_param, param in zip(self.target_v_net.parameters(), self.v_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def getQFunction(self, state):
        return lambda action: (self.q_net(torch.FloatTensor(state).to(self.device).unsqueeze(-1),
                                         torch.FloatTensor([action]).to(self.device).unsqueeze(-1))).detach().numpy()

    def getTrueQFunction(self, state):
        return lambda action: self.predict_true_q(np.expand_dims(state, 0), np.expand_dims([action], 0))

    # bandit setting
    def predict_true_q(self, inputs, action):
        q_val_batch = [getattr(environments.environments, self.config.env_name).reward_func(a[0]) for a in action]
        return np.expand_dims(q_val_batch, -1)

    def getPolicyFunction(self, state):

        _, _, _, mean, log_std = self.pi_net.evaluate(torch.FloatTensor(state).to(self.device).unsqueeze(-1))
        mean = mean.detach().numpy()
        std = (log_std.exp()).detach().numpy()
        return lambda action: 1/(std * np.sqrt(2 * np.pi)) * np.exp(- (action - mean)**2 / (2 * std**2))

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, l1_dim, l2_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, l1_dim)
        self.linear2 = nn.Linear(l1_dim, l2_dim)
        self.linear3 = nn.Linear(l2_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.device = torch.device("cpu")

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, l1_dim, l2_dim, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, l1_dim)
        self.linear2 = nn.Linear(l1_dim, l2_dim)
        self.linear3 = nn.Linear(l2_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.device = torch.device("cpu")

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, l1_dim, l2_dim, action_scale, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, l1_dim)
        self.linear2 = nn.Linear(l1_dim, l2_dim)

        self.mean_linear = nn.Linear(l2_dim, action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(l2_dim, action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_dim = action_dim
        self.action_scale = action_scale
        self.device = torch.device("cpu")

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = self.get_distribution(mean, std)

        z = normal.sample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z)

        if len(log_prob.shape) == 1:
            log_prob.unsqueeze_(-1)
        log_prob -= torch.log(1 - action.pow(2) + epsilon).sum(-1, keepdim=True)

        # scale to correct range
        action *= self.action_scale

        mean = torch.tanh(mean)
        mean *= self.action_scale

        return action, log_prob, z, mean, log_std


    def get_logprob(self, states, tiled_actions, epsilon=1e-6):

        normalized_actions = tiled_actions.permute(1, 0, 2) / self.action_scale
        atanh_actions = self.atanh(normalized_actions)

        mean, log_std = self.forward(states)
        std = log_std.exp()

        normal = self.get_distribution(mean, std)

        log_prob = normal.log_prob(atanh_actions)

        if len(log_prob.shape) == 2:
            log_prob.unsqueeze_(-1)

        log_prob -= torch.log(1 - normalized_actions.pow(2) + epsilon).sum(dim=-1,keepdim=True)
        stacked_log_prob = log_prob.permute(1, 0, 2).reshape(-1, 1)

        return stacked_log_prob

    def get_distribution(self, mean, std):
        if self.action_dim == 1:
            normal = Normal(mean, std)
        else:
            normal = MultivariateNormal(mean, torch.diag_embed(std))

        return normal



    def atanh(self, x):
        return (torch.log(1 + x) - torch.log(1 - x)) / 2

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        action = action.detach().cpu().numpy()
        return action[0]


class LinearPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_scale, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(LinearPolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.mean_linear = nn.Linear(state_dim, action_dim)
        self.mean_linear.bias.data.uniform_(-1.2, -1.2)

        self.log_std_linear = nn.Linear(state_dim, action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-1., -1.)

        self.action_dim = action_dim
        self.action_scale = action_scale
        self.device = torch.device("cpu")

    def forward(self, state):

        mean = self.mean_linear(state)
        log_std = self.log_std_linear(state)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = self.get_distribution(mean, std)

        z = normal.sample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z)

        if len(log_prob.shape) == 1:
            log_prob.unsqueeze_(-1)
        log_prob -= torch.log(1 - action.pow(2) + epsilon).sum(-1, keepdim=True)

        # scale to correct range
        action *= self.action_scale

        mean = torch.tanh(mean)
        mean *= self.action_scale

        return action, log_prob, z, mean, log_std


    def get_logprob(self, states, tiled_actions, epsilon=1e-6):

        normalized_actions = tiled_actions.permute(1, 0, 2) / self.action_scale
        atanh_actions = self.atanh(normalized_actions)

        mean, log_std = self.forward(states)
        std = log_std.exp()

        normal = self.get_distribution(mean, std)

        log_prob = normal.log_prob(atanh_actions)

        if len(log_prob.shape) == 2:
            log_prob.unsqueeze_(-1)

        log_prob -= torch.log(1 - normalized_actions.pow(2) + epsilon).sum(dim=-1,keepdim=True)
        stacked_log_prob = log_prob.permute(1, 0, 2).reshape(-1, 1)

        return stacked_log_prob

    def get_distribution(self, mean, std):
        if self.action_dim == 1:
            normal = Normal(mean, std)
        else:
            normal = MultivariateNormal(mean, torch.diag_embed(std))

        return normal

    def atanh(self, x):
        return (torch.log(1 + x) - torch.log(1 - x)) / 2

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        action = action.detach().cpu().numpy()
        return action[0]