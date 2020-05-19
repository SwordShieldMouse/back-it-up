from agents.network.base_network import BaseNetwork
import numpy as np
import environments

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import MultivariateNormal
import quadpy
import itertools
from scipy.special import binom


class HydraForwardKLNetwork(BaseNetwork):
    def __init__(self, config):
        super(HydraForwardKLNetwork, self).__init__(config, [config.pi_lr, config.qf_vf_lr])

        self.config = config
        self.optim_type = config.optim_type

        self.use_true_q = False
        if config.use_true_q == "True":
            self.use_true_q = True

        self.rng = np.random.RandomState(config.random_seed)
        self.entropy_scale = config.entropy_scale

        self.use_target = config.use_target

        assert not self.use_target

        if config.use_replay:
            self.batch_size = config.batch_size
        else:
            self.batch_size = 1

        self.use_hard_policy = config.use_hard_policy

        # create network
        if 'ContinuousBandits' in config.env_name:
            raise NotImplementedError

        self.hydra_net = HydraNetwork(self.state_dim, self.action_dim, config.actor_critic_dim, self.action_max[0])

        if self.use_target:
            raise NotImplementedError

        self.device = torch.device("cpu")

        # optimizer
        self.optimizer = optim.RMSprop(self.hydra_net.parameters(), lr=self.learning_rate[0])

        # for p in self.hydra_net.parameters():
        #     print(p)
        # exit()
        # self.vq_optimizer = optim.RMSprop(self.hydra_net.parameters(), lr=self.learning_rate[1])

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
            self.l = config.l_param

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
        self.stacked_intgrl_actions = self.tiled_intgrl_actions.reshape(-1, self.action_dim)  # (32 x 254, 1)
        self.tiled_intgrl_weights = self.intgrl_weights.unsqueeze(0).repeat(self.batch_size, 1)

        print("Num. Integration points: {}".format(self.intgrl_actions_len))

    def sample_action(self, state_batch):
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action, _, _, _, _ = self.hydra_net.pi_evaluate(state_batch)

        return action.detach().numpy()

    def predict_action(self, state_batch):

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        _, _, _, mean, _ = self.hydra_net.pi_evaluate(state_batch)

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
            q_val = self.hydra_net('q', state_batch, action_batch)
            v_val = self.hydra_net('v', state_batch)
            new_action, log_prob, _, _, _ = self.hydra_net.pi_evaluate(state_batch)

            # q_loss, v_loss
            if self.use_target:
                raise NotImplementedError
            else:
                target_next_v_val = self.hydra_net('v', next_state_batch)
            target_q_val = reward_batch + gamma_batch * target_next_v_val
            q_value_loss = nn.MSELoss()(q_val, target_q_val.detach())

            new_q_val = self.hydra_net('q', state_batch, new_action)

            if self.config.q_update_type == 'sac':
                target_v_val = new_q_val - self.entropy_scale * log_prob

            elif self.config.q_update_type == 'non_sac':
                target_v_val = (reward_batch - self.entropy_scale * log_prob) + gamma_batch * target_next_v_val
            else:
                raise ValueError("invalid config.q_update_type")
            value_loss = nn.MSELoss()(v_val, target_v_val.detach())

        # pi_loss
        if not self.use_hard_policy:
            if self.optim_type == 'll':
                raise NotImplementedError

            elif self.optim_type == 'intg':
                tiled_state_batch = state_batch.unsqueeze(1).repeat(1, self.intgrl_actions_len, 1)
                stacked_state_batch = tiled_state_batch.reshape(-1, self.state_dim)

                if self.use_true_q:
                    # predict_true_q
                    intgrl_q_val = torch.from_numpy(self.predict_true_q(stacked_state_batch, self.stacked_intgrl_actions)).to(torch.float32)
                else:
                    intgrl_q_val = self.hydra_net('q', stacked_state_batch, self.stacked_intgrl_actions)

                tiled_intgrl_q_val = intgrl_q_val.reshape(-1, self.intgrl_actions_len) / self.entropy_scale

                # compute Z
                constant_shift, _ = torch.max(tiled_intgrl_q_val, -1, keepdim=True)
                tiled_constant_shift = constant_shift.repeat(1, self.intgrl_actions_len)

                intgrl_exp_q_val = torch.exp(tiled_intgrl_q_val - tiled_constant_shift).detach()

                z = (intgrl_exp_q_val * self.tiled_intgrl_weights).sum(-1).detach()
                tiled_z = z.unsqueeze(-1).repeat(1, self.intgrl_actions_len).detach()

                boltzmann_prob = intgrl_exp_q_val / tiled_z

                intgrl_logprob = self.hydra_net.pi_get_logprob(state_batch, self.tiled_intgrl_actions)
                tiled_intgrl_logprob = intgrl_logprob.reshape(self.batch_size, self.intgrl_actions_len)

                integrands = boltzmann_prob * tiled_intgrl_logprob
                policy_loss = (-(integrands * self.tiled_intgrl_weights).sum(-1)).mean(-1)

            else:
                raise ValueError("Invalid optim_type")
        else:
            if self.use_true_q:

                # 1x1x1
                dummy_state_batch = torch.FloatTensor([0]).to(self.device).unsqueeze(-1).unsqueeze(-1)
                dummy_action_batch = torch.FloatTensor([getattr(environments.environments, self.config.env_name).get_max()]).to(self.device).unsqueeze(-1).unsqueeze(-1)

                policy_loss = (-self.hydra_net.pi_get_logprob(dummy_state_batch, dummy_action_batch)).mean()

            else:
                raise ValueError("Need to find explicit maximum, and need trueQ")

        loss = policy_loss
        if not self.use_true_q:

            loss += 10* (q_value_loss + value_loss)


        # print('loss:', q_value_loss, value_loss)
        # exit()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        raise NotImplementedError

    def getQFunction(self, state):
        return lambda action: (self.hydra_net('q', torch.FloatTensor(state).to(self.device).unsqueeze(-1),
                                         torch.FloatTensor([action]).to(self.device).unsqueeze(-1))).detach().numpy()

    def getTrueQFunction(self, state):
        return lambda action: self.predict_true_q(np.expand_dims(state, 0), np.expand_dims([action], 0))

    # bandit setting
    def predict_true_q(self, inputs, action):
        q_val_batch = [getattr(environments.environments, self.config.env_name).reward_func(a[0]) for a in action]
        return np.expand_dims(q_val_batch, -1)

    def getPolicyFunction(self, state):

        _, _, _, mean, std = self.hydra_net.pi_evaluate(torch.FloatTensor(state).to(self.device).unsqueeze(-1))
        mean = mean.detach().numpy()
        std = std.detach().numpy()
        return lambda action: 1/(std * np.sqrt(2 * np.pi)) * np.exp(- (action - mean)**2 / (2 * std**2))


class HydraNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, layer_dim, action_scale, init_w=3e-3):
        super(HydraNetwork, self).__init__()

        self.action_dim = action_dim
        self.action_scale = action_scale
        self.device = torch.device("cpu")

        # shared layer
        self.shared = nn.Linear(state_dim, layer_dim)

        # policy layer
        self.pi_net = nn.Linear(layer_dim, layer_dim)
        self.pi_net_mean = nn.Linear(layer_dim, action_dim)
        self.pi_net_mean.weight.data.uniform_(-init_w, init_w)
        self.pi_net_mean.bias.data.uniform_(-init_w, init_w)

        self.pi_net_std = nn.Linear(layer_dim, action_dim)
        self.pi_net_std.weight.data.uniform_(-init_w, init_w)
        self.pi_net_std.bias.data.uniform_(-init_w, init_w)

        # Q layer
        self.q_net = nn.Linear(layer_dim + action_dim, layer_dim)
        self.q_net2 = nn.Linear(layer_dim, 1)
        self.q_net2.weight.data.uniform_(-init_w, init_w)
        self.q_net2.bias.data.uniform_(-init_w, init_w)

        # V layer
        self.v_net = nn.Linear(layer_dim, layer_dim)
        self.v_net2 = nn.Linear(layer_dim, 1)
        self.v_net2.weight.data.uniform_(-init_w, init_w)
        self.v_net2.bias.data.uniform_(-init_w, init_w)

    def forward(self, type, state, action=None):

        x = F.relu(self.shared(state))
        if type == 'pi':
            assert action is None

            x = F.relu(self.pi_net(x))
            mean = self.pi_net_mean(x)
            std = self.pi_net_std(x)
            std = torch.log(1+torch.exp(std))

            return mean, std

        elif type == 'q':
            assert action is not None

            x = F.relu(self.q_net(torch.cat([x, action], 1)))
            x = self.q_net2(x)

            return x

        elif type == 'v':
            assert action is None
            x = F.relu(self.v_net(x))
            x = self.v_net2(x)

            return x

    def pi_evaluate(self, state, epsilon=1e-6):
        mean, std = self.forward('pi', state)
        normal = self.pi_get_distribution(mean, std)

        z = normal.sample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z)

        # TODO: Check what this does
        if len(log_prob.shape) == 1:
            log_prob.unsqueeze_(-1)
        log_prob -= torch.log(1 - action.pow(2) + epsilon).sum(-1, keepdim=True)

        action *= self.action_scale

        mean = torch.tanh(mean)
        mean *= self.action_scale
        return action, log_prob, z, mean, std

    def pi_get_distribution(self, mean, std):
        if self.action_dim == 1:
            normal = Normal(mean, std)
        else:
            normal = MultivariateNormal(mean, torch.diag_embed(std))
        return normal

    def pi_get_logprob(self, states, tiled_actions, epsilon=1e-6):
        normalized_actions = tiled_actions.permute(1, 0, 2) / self.action_scale
        atanh_actions = self.atanh(normalized_actions)

        mean, std = self.forward('pi', states)
        normal = self.pi_get_distribution(mean, std)

        log_prob = normal.log_prob(atanh_actions)

        # TODO: Check what this does
        if len(log_prob.shape) == 2:
            log_prob.unsqueeze_(-1)

        log_prob -= torch.log(1 - normalized_actions.pow(2) + epsilon).sum(dim=-1, keepdim=True)
        stacked_log_prob = log_prob.permute(1, 0, 2).reshape(-1, 1)
        return stacked_log_prob

    def atanh(self, x):
        return (torch.log(1 + x) - torch.log(1 - x)) / 2

