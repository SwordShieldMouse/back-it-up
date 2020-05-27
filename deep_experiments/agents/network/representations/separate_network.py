import torch
torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import MultivariateNormal

import numpy as np



class ValueNetwork(nn.Module):
    def __init__(self, state_dim, layer_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, layer_dim)
        self.linear2 = nn.Linear(layer_dim, layer_dim)
        self.linear3 = nn.Linear(layer_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.device = torch.device("cpu")

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, layer_dim, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, layer_dim)
        self.linear2 = nn.Linear(layer_dim, layer_dim)
        self.linear3 = nn.Linear(layer_dim, 1)

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
    def __init__(self, state_dim, action_dim, layer_dim, action_scale, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, layer_dim)
        self.linear2 = nn.Linear(layer_dim, layer_dim)

        self.mean_linear = nn.Linear(layer_dim, action_dim)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(layer_dim, action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_dim = action_dim
        self.action_scale = action_scale
        self.device = torch.device("cpu")

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        std = F.softplus(self.log_std_linear(x), threshold=10)
        # log_std = torch.clamp(self.log_std_linear(x), self.log_std_min, self.log_std_max)
        # std = torch.exp(log_std)

        return mean, std

    def evaluate(self, state, epsilon=1e-6):
        pre_mean, std = self.forward(state)

        normal = self.get_distribution(pre_mean, std)

        z = normal.sample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z)

        if len(log_prob.shape) == 1:
            log_prob.unsqueeze_(-1)
        log_prob -= torch.log(1 - action.pow(2) + epsilon).sum(-1, keepdim=True)

        # scale to correct range
        action *= self.action_scale

        mean = torch.tanh(pre_mean)
        mean *= self.action_scale
        return action, log_prob, z, pre_mean, mean, std,

    def get_logprob(self, states, tiled_actions, epsilon=1e-6):

        normalized_actions = tiled_actions.permute(1, 0, 2) / self.action_scale
        atanh_actions = self.atanh(normalized_actions)

        mean, std = self.forward(states)
        normal = self.get_distribution(mean, std)

        log_prob = normal.log_prob(atanh_actions)

        if len(log_prob.shape) == 2:
            log_prob.unsqueeze_(-1)

        log_prob -= torch.log(1 - normalized_actions.pow(2)).sum(dim=-1, keepdim=True)
        stacked_log_prob = log_prob.permute(1, 0, 2)
        return stacked_log_prob

    def get_distribution(self, mean, std, epsilon=1e-6):

        try:
            # std += epsilon
            if self.action_dim == 1:
                normal = Normal(mean, std)
            else:
                normal = MultivariateNormal(mean, torch.diag_embed(std))

        except:
            print("Error occured with mean {}, std {}".format(mean, std))
            exit()
        return normal

    def atanh(self, x):
        return (torch.log(1 + x) - torch.log(1 - x)) / 2

    def get_action(self, state):
        state = torch.DoubleTensor(state).unsqueeze(0).to(self.device)
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        action = action.detach().cpu().numpy()
        return action[0]


class LinearPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_scale, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(LinearPolicyNetwork, self).__init__()

        torch.set_default_dtype(torch.float64)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.mean_linear = nn.Linear(state_dim, action_dim)
        self.mean_linear.bias.data.uniform_(-0.8, -0.8)

        p = np.log(np.exp(0.1)-1)
        self.log_std_linear = nn.Linear(state_dim, action_dim)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(p, p)

        self.action_dim = action_dim
        self.action_scale = action_scale
        self.device = torch.device("cpu")

    def forward(self, state):

        mean = self.mean_linear(state)

        # log_std = torch.clamp(self.log_std_linear(state), self.log_std_min, self.log_std_max)
        # std = torch.exp(log_std)

        std = F.softplus(self.log_std_linear(state), threshold=10)

        return mean, std

    def evaluate(self, state, epsilon=1e-6):
        mean, std = self.forward(state)

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
        return action, log_prob, z, mean, std

    def get_logprob(self, states, tiled_actions, epsilon=1e-6):

        normalized_actions = tiled_actions.permute(1, 0, 2) / self.action_scale
        atanh_actions = self.atanh(normalized_actions)

        mean, std = self.forward(states)
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
        state = torch.DoubleTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.forward(state)
        std = torch.log(1+log_std.exp())

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        action = action.detach().cpu().numpy()
        return action[0]
