import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import MultivariateNormal
import numpy as np

class HydraNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, layer_dim, action_scale, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(HydraNetwork, self).__init__()

        self.action_dim = action_dim
        self.action_scale = action_scale
        self.device = torch.device("cpu")

        # std ~= 1
        std_init_w = init_w + np.log(np.exp(1)-1)

        # shared layer
        self.shared = nn.Linear(state_dim, layer_dim)

        # policy layer
        self.pi_net = nn.Linear(layer_dim, layer_dim)
        self.pi_net_mean = nn.Linear(layer_dim, action_dim)
        self.pi_net_mean.weight.data.uniform_(-init_w, init_w)
        self.pi_net_mean.bias.data.uniform_(-init_w, init_w)

        self.pi_net_std = nn.Linear(layer_dim, action_dim)
        self.pi_net_std.weight.data.uniform_(-init_w, init_w)
        self.pi_net_std.bias.data.uniform_(-std_init_w, std_init_w)

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

        self.std_min = np.exp(-20)
        self.std_max = np.exp(2)

    def forward(self, type, state, action=None):

        x = F.relu(self.shared(state))
        if type == 'pi':
            assert action is None

            x = F.relu(self.pi_net(x))
            mean = self.pi_net_mean(x)
            std = self.pi_net_std(x)
            std = torch.log(1+torch.exp(std))
            std = torch.clamp(std, self.std_min, self.std_max)
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
            try:
                normal = MultivariateNormal(mean, torch.diag_embed(std))
            except:
                raise ValueError("Error occurred when constructing Multivariate Normal with: {}, {}".format(mean, std))
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

