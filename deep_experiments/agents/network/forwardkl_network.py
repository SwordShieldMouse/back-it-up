from agents.network.base_network import BaseNetwork
import numpy as np
import environments

import torch
import torch.nn as nn
import torch.optim as optim

import quadpy
import itertools
from scipy.special import binom
from .representations.separate_network import *
from utils.main_utils import write_summary

class ForwardKLNetwork(BaseNetwork):
    def __init__(self, config):
        super(ForwardKLNetwork, self).__init__(config, [config.pi_lr, config.qf_vf_lr])

        torch.set_default_dtype(torch.float64)

        self.config = config
        self.optim_type = config.optim_type

        self.writer = config.writer
        self.writer_step = 0

        self.use_true_q = False
        if config.use_true_q == "True":
            self.use_true_q = True

        self.rng = np.random.RandomState(config.random_seed)
        self.entropy_scale = config.entropy_scale

        self.use_target = config.use_target

        if config.use_replay:
            self.batch_size = config.batch_size
        else:
            self.batch_size = 1

        self.use_hard_policy = config.use_hard_policy

        # create network
        if 'ContinuousBandits' in config.env_name:
            self.pi_net = LinearPolicyNetwork(self.state_dim, self.action_dim, self.action_max[0])
        else:
            self.pi_net = PolicyNetwork(self.state_dim, self.action_dim, config.actor_critic_dim, self.action_max[0])

        self.q_net = SoftQNetwork(self.state_dim, self.action_dim, config.actor_critic_dim)

        self.v_net = ValueNetwork(self.state_dim, config.actor_critic_dim)

        if self.use_target:
            self.target_v_net = ValueNetwork(self.state_dim, config.actor_critic_dim)

            # copy to target_v_net
            for target_param, param in zip(self.target_v_net.parameters(), self.v_net.parameters()):
                target_param.data.copy_(param.data)

        self.device = torch.device("cpu")

        # optimizer
        self.pi_optimizer = optim.RMSprop(self.pi_net.parameters(), lr=self.learning_rate[0])
        self.q_optimizer = optim.RMSprop(self.q_net.parameters(), lr=self.learning_rate[1])
        self.v_optimizer = optim.RMSprop(self.v_net.parameters(), lr=self.learning_rate[1])

        dtype = torch.float64

        if self.action_dim == 1:
            self.N = config.N_param  # 1024

            scheme = quadpy.line_segment.clenshaw_curtis(self.N)
            # cut off endpoints since they should be zero but numerically might give nans
            self.intgrl_actions = (torch.tensor(scheme.points[1:-1], dtype=dtype).unsqueeze(-1) * torch.Tensor(self.action_max)).to(
                torch.float64)
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
        state_batch = torch.DoubleTensor(state_batch).to(self.device)
        action, log_prob, z, pre_mean, mean, std = self.pi_net.evaluate(state_batch)

        #for dim in range(np.shape(action)[1]):
            # for tf 1.8
            #write_summary(self.writer, self.writer_step, pre_mean[0][dim], tag='pre_mean/[{}]'.format(dim))
            #write_summary(self.writer, self.writer_step, mean[0][dim], tag='mean/[{}]'.format(dim))
            #write_summary(self.writer, self.writer_step, std[0][dim], tag='std/[{}]'.format(dim))

            # for tf 1.14 and above
            # self.writer.add_scalar('mean/[{}]'.format(dim), mean[0][dim], self.writer_step)
            # self.writer.add_scalar('std/[{}]'.format(dim), std[0][dim], self.writer_step)

        #self.writer_step += 1
        return action.detach().numpy()

    def predict_action(self, state_batch):

        state_batch = torch.DoubleTensor(state_batch).to(self.device)
        _, _, _, _, mean, std = self.pi_net.evaluate(state_batch)

        return mean.detach().numpy()

    def update_network(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        state_batch = torch.DoubleTensor(state_batch).to(self.device)
        action_batch = torch.DoubleTensor(action_batch).to(self.device)
        next_state_batch = torch.DoubleTensor(next_state_batch).to(self.device)
        reward_batch = torch.DoubleTensor(reward_batch).to(self.device)
        gamma_batch = torch.DoubleTensor(gamma_batch).to(self.device)

        reward_batch.unsqueeze_(-1)
        gamma_batch.unsqueeze_(-1)

        if not self.use_true_q:
            q_val = self.q_net(state_batch, action_batch)
            v_val = self.v_net(state_batch)

            # q_loss, v_loss
            target_next_v_val = self.target_v_net(next_state_batch) if self.use_target else self.v_net(next_state_batch)
            target_q_val = reward_batch + gamma_batch * target_next_v_val
            q_value_loss = nn.MSELoss()(q_val, target_q_val.detach())

            # SAC paper samples actions again
            if self.config.q_update_type == 'sac':
                new_action, new_log_prob, z, pre_mean, mean, std = self.pi_net.evaluate(state_batch)
                new_q_val = self.q_net(state_batch, new_action)
                target_v_val = new_q_val - self.entropy_scale * new_log_prob

            elif self.config.q_update_type == 'non_sac':
                log_prob_batch = self.pi_net.get_logprob(state_batch, action_batch.unsqueeze_(1)).squeeze(-1)
                target_v_val = (reward_batch - self.entropy_scale * log_prob_batch) + gamma_batch * target_next_v_val
            else:
                raise ValueError("invalid config.q_update_type")
            value_loss = nn.MSELoss()(v_val, target_v_val.detach())

        # pi_loss
        if not self.use_hard_policy:
            if self.optim_type == 'intg':
                tiled_state_batch = state_batch.unsqueeze(1).repeat(1, self.intgrl_actions_len, 1)
                stacked_state_batch = tiled_state_batch.reshape(-1, self.state_dim)

                if self.use_true_q:
                    # predict_true_q
                    intgrl_q_val = torch.from_numpy(self.predict_true_q(stacked_state_batch, self.stacked_intgrl_actions)).to(torch.float64)
                else:
                    intgrl_q_val = self.q_net(stacked_state_batch, self.stacked_intgrl_actions)


                tiled_intgrl_q_val = intgrl_q_val.reshape(-1, self.intgrl_actions_len) / self.entropy_scale

                # compute Z
                constant_shift, _ = torch.max(tiled_intgrl_q_val, -1, keepdim=True)
                tiled_constant_shift = constant_shift.repeat(1, self.intgrl_actions_len)

                intgrl_exp_q_val = torch.exp(tiled_intgrl_q_val - tiled_constant_shift).detach()

                z = (intgrl_exp_q_val * self.tiled_intgrl_weights).sum(-1).detach()
                tiled_z = z.unsqueeze(-1).repeat(1, self.intgrl_actions_len).detach()

                boltzmann_prob = intgrl_exp_q_val / tiled_z

                tiled_intgrl_logprob = self.pi_net.get_logprob(state_batch, self.tiled_intgrl_actions).squeeze(-1)

                integrands = boltzmann_prob * tiled_intgrl_logprob
                policy_loss = (-(integrands * self.tiled_intgrl_weights).sum(-1)).mean(-1)

            else:
                raise ValueError("Invalid optim_type")
        else:
            if self.use_true_q:

                # 1x1x1
                dummy_state_batch = torch.DoubleTensor([0]).to(self.device).unsqueeze(-1).unsqueeze(-1)
                dummy_action_batch = torch.DoubleTensor([getattr(environments.environments, self.config.env_name).get_max()]).to(self.device).unsqueeze(-1).unsqueeze(-1)

                print()
                policy_loss = (-(self.pi_net.get_logprob(dummy_state_batch, dummy_action_batch)).reshape(-1, 1)).mean()

            else:
                raise ValueError("Need to find explicit maximum, and need trueQ")

        #write_summary(self.writer, self.writer_step, policy_loss, tag='loss/pi')
        #write_summary(self.writer, self.writer_step, q_value_loss, tag='loss/q')
        #write_summary(self.writer, self.writer_step, value_loss, tag='loss/v')

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
        return lambda action: (self.q_net(torch.DoubleTensor(state).to(self.device).unsqueeze(-1),
                                         torch.DoubleTensor([action]).to(self.device).unsqueeze(-1))).detach().numpy()

    def getTrueQFunction(self, state):
        return lambda action: self.predict_true_q(np.expand_dims(state, 0), np.expand_dims([action], 0))

    # bandit setting
    def predict_true_q(self, inputs, action):
        q_val_batch = [getattr(environments.environments, self.config.env_name).reward_func(a[0]) for a in action]
        return np.expand_dims(q_val_batch, -1)

    def getPolicyFunction(self, state):

        _, _, _, _, mean, std = self.pi_net.evaluate(torch.DoubleTensor(state).to(self.device).unsqueeze(-1))
        mean = mean.detach().numpy()
        std = std.detach().numpy()
        return lambda action: 1/(std * np.sqrt(2 * np.pi)) * np.exp(- (action - mean)**2 / (2 * std**2))
