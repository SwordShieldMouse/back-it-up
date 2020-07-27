import torch
torch.set_default_dtype(torch.float32)
import torch.nn as nn

import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import MultivariateNormal


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
        # self.linear1 = nn.Linear(state_dim, layer_dim)
        # self.linear2 = nn.Linear(layer_dim + action_dim, layer_dim)
        self.linear3 = nn.Linear(layer_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.device = torch.device("cpu")

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        # x = F.relu(self.linear1(state))
        # x = torch.cat([x, action], 1)
        # x = F.relu(self.linear2(x))
        # x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, layer_dim, action_scale, init_w=3e-3, log_std_min=-10, log_std_max=2):
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
        std = F.softplus(torch.clamp(self.log_std_linear(x), self.log_std_min, self.log_std_max), threshold=10)
        # log_std = F.tanh(self.log_std_linear(x))
        # log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        # std = torch.exp(log_std)

        return mean, std

    def evaluate(self, state, epsilon=1e-6):
        pre_mean, std = self.forward(state)

        normal = self.get_distribution(pre_mean, std)

        raw_action = normal.rsample()
        action = torch.tanh(raw_action)
        log_prob = normal.log_prob(raw_action)

        if len(log_prob.shape) == 1:
            log_prob.unsqueeze_(-1)
        log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + epsilon).sum(dim=-1, keepdim=True)

        # scale to correct range
        action = action * self.action_scale

        mean = torch.tanh(pre_mean) * self.action_scale
        return action, log_prob, raw_action, pre_mean, mean, std,

    # TODO: merge with evaluate
    def evaluate_multiple(self, state, num_actions, epsilon=1e-6):
        # state: (batch_size, state_dim)
        pre_mean, std = self.forward(state)

        normal = self.get_distribution(pre_mean, std)

        raw_action = normal.sample_n(num_actions)  # (num_actions, batch_size, action_dim)
        action = torch.tanh(raw_action)
        assert raw_action.shape == (num_actions, pre_mean.shape[0], pre_mean.shape[1])

        log_prob = normal.log_prob(raw_action)

        if len(log_prob.shape) == 2:
            log_prob.unsqueeze_(-1)

        log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + epsilon).sum(dim=-1, keepdim=True)

        log_prob = log_prob.permute(1, 0, 2).squeeze(-1)  # (batch_size, num_actions)
        action = action.permute(1, 0, 2)  # (batch_size, num_actions, 1)

        mean = torch.tanh(pre_mean) * self.action_scale

        # dimension of raw_action might be off
        return action, log_prob, raw_action, pre_mean, mean, std


    def get_logprob(self, states, tiled_actions, epsilon = 1e-6):

        normalized_actions = tiled_actions.permute(1, 0, 2) / self.action_scale
        atanh_actions = self.atanh(normalized_actions)

        mean, std = self.forward(states)
        normal = self.get_distribution(mean, std)

        log_prob = normal.log_prob(atanh_actions)

        if len(log_prob.shape) == 2:
            log_prob.unsqueeze_(-1)

        log_prob -= torch.log(self.action_scale * (1 - normalized_actions.pow(2)) + epsilon).sum(dim=-1, keepdim=True)
        stacked_log_prob = log_prob.permute(1, 0, 2)
        return stacked_log_prob

    def get_distribution(self, mean, std):

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
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)

        action = action.detach().cpu().numpy()
        return action[0]
