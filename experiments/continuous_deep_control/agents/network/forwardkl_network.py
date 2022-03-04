from agents.network.base_network import BaseNetwork
import environments
import numpy as np
import torch.optim as optim
import quadpy
import itertools
from scipy.special import binom
from .representations.separate_network import *
from utils.main_utils import write_summary


class ForwardKLNetwork(BaseNetwork):
    def __init__(self, config):
        super(ForwardKLNetwork, self).__init__(config, [config.pi_lr, config.qf_vf_lr])

        torch.set_default_dtype(torch.float32)

        self.config = config
        self.optim_type = config.optim_type

        self.rng = np.random.RandomState(config.random_seed)
        self.entropy_scale = config.entropy_scale

        if config.use_replay:
            self.batch_size = config.batch_size
        else:
            self.batch_size = 1

        # use_true_q is only applicable for ContinuousBandits Environment where true action-value function is available
        self.use_true_q = config.use_true_q

        self.use_target = config.use_target
        self.use_hard_policy = config.use_hard_policy
        self.use_hard_value = config.use_hard_value

        # create network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._initialize_policy_network(config)
        self._initialize_value_networks(config)

        if self.use_target:
            self.target_v_net = ValueNetwork(self.state_dim, config.actor_critic_dim)

            # copy to target_v_net
            for target_param, param in zip(self.target_v_net.parameters(), self.v_net.parameters()):
                target_param.data.copy_(param.data)

        dtype = torch.float32

        # for WIS
        self.n_action_points = config.n_action_points
        self.action_scale = self.action_max[0]

        # Numerical Integration (Clenshaw-Curtis)
        if self.action_dim == 1:
            self.N = config.N_param

            scheme = quadpy.line_segment.clenshaw_curtis(self.N)

            # cut off endpoints since they should be zero but numerically might give nans
            self.intgrl_actions = (torch.tensor(scheme.points[1:-1], dtype=dtype).unsqueeze(-1) * torch.Tensor(self.action_max)).to(
                torch.float32)
            self.intgrl_weights = torch.tensor(scheme.weights[1:-1], dtype=dtype)

            self.intgrl_actions_len = np.shape(self.intgrl_actions)[0]
        # Numerical Integration for multiple-dimension
        else:
            self.l = config.l_param

            n_points = [1]
            for i in range(1, self.l):
                n_points.append(2 ** i + 1)

            schemes = [quadpy.line_segment.clenshaw_curtis(n_points[i]) for i in range(1, self.l)]
            points = [np.array([0.])] + [scheme.points[1:-1] for scheme in schemes]
            weights = [np.array([2.])] + [scheme.weights[1:-1] for scheme in schemes]

            # pre-calculate actions and weights
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

        if self.optim_type == 'intg':
            print("Num. Integration points: {}".format(self.intgrl_actions_len))
        elif self.optim_type == 'wis':
            print("WIS points: {}".format(self.n_action_points))

    def _initialize_policy_network(self, config):
        self.pi_net = PolicyNetwork(self.state_dim, self.action_dim, config.actor_critic_dim, self.action_max[0], config.n_hidden)
        self.pi_net = self.pi_net.to(self.pi_net.device)
        # optimizer
        self.pi_optimizer = optim.RMSprop(self.pi_net.parameters(), lr=self.learning_rate[0])

    def _initialize_value_networks(self, config):
        self.q_net = SoftQNetwork(self.state_dim, self.action_dim, config.actor_critic_dim, config.n_hidden)
        self.v_net = ValueNetwork(self.state_dim, config.actor_critic_dim, config.n_hidden)
        self.q_net = self.q_net.to(self.q_net.device)
        self.v_net = self.v_net.to(self.v_net.device)
        self.q_optimizer = optim.RMSprop(self.q_net.parameters(), lr=self.learning_rate[1])
        self.v_optimizer = optim.RMSprop(self.v_net.parameters(), lr=self.learning_rate[1])

    def sample_action(self, state_batch):
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action, log_prob, z, pre_mean, mean, std = self.pi_net.evaluate(state_batch)

        return action.detach().cpu().numpy()

    def predict_action(self, state_batch):

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        _, _, _, _, mean, std = self.pi_net.evaluate(state_batch)

        return mean.detach().cpu().numpy()

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

            # q_loss, v_loss
            target_next_v_val = self.target_v_net(next_state_batch) if self.use_target else self.v_net(next_state_batch)
            target_q_val = reward_batch + gamma_batch * target_next_v_val
            q_value_loss = nn.MSELoss()(q_val, target_q_val.detach())

            # SAC paper samples actions again
            if self.config.q_update_type == 'sac':
                new_action, new_log_prob, _, _, _, _ = self.pi_net.evaluate(state_batch)
                new_q_val = self.q_net(state_batch, new_action)
                target_v_val = (new_q_val - self.entropy_scale * new_log_prob) if not self.use_hard_value else new_q_val

            elif self.config.q_update_type == 'non_sac':
                log_prob_batch = self.pi_net.get_logprob(state_batch, action_batch.unsqueeze_(1)).squeeze(-1)
                target_v_val = ((reward_batch - self.entropy_scale * log_prob_batch) + gamma_batch * target_next_v_val) if not self.use_hard_value else (reward_batch + gamma_batch * target_next_v_val)
            else:
                raise ValueError("invalid config.q_update_type")
            value_loss = nn.MSELoss()(v_val, target_v_val.detach())

        # pi_loss
        if not self.use_hard_policy:
            if self.optim_type == 'intg':
                tiled_state_batch = state_batch.unsqueeze(1).repeat(1, self.intgrl_actions_len, 1)
                stacked_state_batch = tiled_state_batch.reshape(-1, self.state_dim)

                # predict_true_q
                if self.use_true_q:
                    intgrl_q_val = torch.from_numpy(self.predict_true_q(stacked_state_batch, self.stacked_intgrl_actions)).to(torch.float32)
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

            elif self.optim_type == 'wis':

                actions, log_pdfs, _, _, _, _ = self.pi_net.evaluate_multiple(state_batch, self.n_action_points, no_grad=True)

                tiled_state_batch = state_batch.unsqueeze(1).repeat(1, self.n_action_points, 1)
                stacked_state_batch = tiled_state_batch.reshape(-1, self.state_dim)
                stacked_action_batch = actions.reshape(-1, self.action_dim)
                stacked_q_val = self.q_net(stacked_state_batch, stacked_action_batch)  # (32 * 500, 1)
                tiled_q_val = stacked_q_val.reshape(-1, self.n_action_points)  # (32, 500)

                with torch.no_grad():
                    max_arg = torch.max(tiled_q_val / self.entropy_scale - log_pdfs, axis=1, keepdim=True)[0]
                    rho = torch.exp(tiled_q_val / self.entropy_scale - log_pdfs - max_arg)

                ratio = rho / rho.sum(-1, keepdim=True)
                policy_loss = - torch.sum(ratio * log_pdfs, axis=-1).mean()

            else:
                raise ValueError("Invalid optim_type")
        else:
            if self.use_true_q:

                # 1x1x1
                dummy_state_batch = torch.FloatTensor([0]).to(self.device).unsqueeze(-1).unsqueeze(-1)
                dummy_action_batch = torch.FloatTensor([getattr(environments.environments, self.config.env_name).get_max()]).to(self.device).unsqueeze(-1).unsqueeze(-1)
                policy_loss = (-(self.pi_net.get_logprob(dummy_state_batch, dummy_action_batch)).reshape(-1, 1)).mean()

            else:
                raise ValueError("Need to find explicit maximum, and need trueQ")

        if not self.use_true_q:
            self.q_optimizer.zero_grad()
            self.v_optimizer.zero_grad()
            self.pi_optimizer.zero_grad()

            q_value_loss.backward(retain_graph=True)
            value_loss.backward(retain_graph=True)
            policy_loss.backward()

            self.q_optimizer.step()
            self.v_optimizer.step()
            self.pi_optimizer.step()
        else:
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
                                         torch.FloatTensor([action]).to(self.device).unsqueeze(-1))).detach().cpu().numpy()

    def getTrueQFunction(self, state):
        return lambda action: self.predict_true_q(np.expand_dims(state, 0), np.expand_dims([action], 0))

    # bandit setting
    def predict_true_q(self, inputs, action):
        q_val_batch = [getattr(environments.environments, self.config.env_name).reward_func(a[0]) for a in action]
        return np.expand_dims(q_val_batch, -1)

    def getPolicyFunction(self, state):

        _, _, _, _, mean, std = self.pi_net.evaluate(torch.FloatTensor(state).to(self.device).unsqueeze(-1))
        mean = mean.detach().cpu().numpy()
        std = std.detach().cpu().numpy()
        return lambda action: 1/(std * np.sqrt(2 * np.pi)) * np.exp(- (action - mean)**2 / (2 * std**2))


class ForwardKL_GMM_Network(ForwardKLNetwork):
    def _initialize_policy_network(self, config):
        self.pi_net = PolicyNetworkGMM(self.state_dim, self.action_dim, config.actor_critic_dim, self.action_max[0], config.n_hidden,n_gmm_components=config.n_gmm_components)
        self.pi_net = self.pi_net.to(self.pi_net.device)
        self.pi_net.gmm_components = self.pi_net.gmm_components.to(self.pi_net.device).detach().requires_grad_(True)
        # optimizer
        self.pi_optimizer = optim.RMSprop(self.pi_net.parameters(), lr=self.learning_rate[0])
        self.pi_gmm_optimizer = optim.RMSprop([self.pi_net.gmm_components], lr=self.config.gmm_lr)

    def update_network(self, *args, **kwargs):
        assert self.optim_type == "wis"
        self.pi_gmm_optimizer.zero_grad()
        super(ForwardKL_GMM_Network, self).update_network(*args, **kwargs)
        self.pi_gmm_optimizer.step()
