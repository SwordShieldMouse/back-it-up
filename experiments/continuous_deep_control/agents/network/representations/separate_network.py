import torch
torch.set_default_dtype(torch.float32)
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, layer_dim, n_hidden=1, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim, layer_dim)

        linear2 = [None for _ in range(n_hidden)]
        for i in range(n_hidden):
            linear2[i] = nn.Sequential(
                nn.Linear(layer_dim, layer_dim),
                nn.ReLU())


        self.linear2 = nn.Sequential(*linear2)

        self.linear3 = nn.Linear(layer_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, layer_dim, n_hidden=1, init_w=3e-3):
        super(SoftQNetwork, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, layer_dim)

        linear2 = [None for _ in range(n_hidden)]
        for i in range(n_hidden):
            linear2[i] = nn.Sequential(
                nn.Linear(layer_dim, layer_dim),
                nn.ReLU())

        self.linear2 = nn.Sequential(*linear2)

        self.linear3 = nn.Linear(layer_dim, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, layer_dim, action_scale, n_hidden=1, init_w=3e-3, log_std_min=-10, log_std_max=2):
        super(PolicyNetwork, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_dim, layer_dim)

        linear2 = [None for _ in range(n_hidden)]
        for i in range(n_hidden):
            linear2[i] = nn.Sequential(
                nn.Linear(layer_dim, layer_dim),
                nn.ReLU())

        self.linear2 = nn.Sequential(*linear2)

        self.layer_dim, self.action_dim, self.init_w = layer_dim, action_dim, init_w

        self._create_output_layer()
        self.action_scale = action_scale

    def _create_output_layer(self):
        self.mean_linear = nn.Linear(self.layer_dim, self.action_dim)
        self.mean_linear.weight.data.uniform_(-self.init_w, self.init_w)
        self.mean_linear.bias.data.uniform_(-self.init_w, self.init_w)

        self.log_std_linear = nn.Linear(self.layer_dim, self.action_dim)
        self.log_std_linear.weight.data.uniform_(-self.init_w, self.init_w)
        self.log_std_linear.bias.data.uniform_(-self.init_w, self.init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = self.linear2(x)

        mean = self.mean_linear(x)
        std = F.softplus(torch.clamp(self.log_std_linear(x), self.log_std_min, self.log_std_max), threshold=10)
        # log_std = F.tanh(self.log_std_linear(x))
        # log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        # std = torch.exp(log_std)

        return mean, std

    def evaluate(self, state, epsilon=1e-6, no_grad=False):
        pre_mean, std = self.forward(state)

        normal = self.get_distribution(pre_mean, std)

        if no_grad:
            raw_action = normal.sample()
        else:
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
    def evaluate_multiple(self, state, num_actions, epsilon=1e-6, no_grad=False):
        # state: (batch_size, state_dim)
        pre_mean, std = self.forward(state)

        normal = self.get_distribution(pre_mean, std)

        if no_grad:
            raw_action = normal.sample((num_actions,))
        else:
            raw_action = normal.rsample(sample_shape=(num_actions,))  # (num_actions, batch_size, action_dim)
        action = torch.tanh(raw_action)
        assert raw_action.shape == (num_actions, pre_mean.shape[0], pre_mean.shape[1])

        log_prob = normal.log_prob(raw_action)

        if len(log_prob.shape) == 2:
            log_prob.unsqueeze_(-1)

        log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + epsilon).sum(dim=-1, keepdim=True)

        log_prob = log_prob.permute(1, 0, 2).squeeze(-1)  # (batch_size, num_actions)
        action = action.permute(1, 0, 2)  # (batch_size, num_actions, 1)

        # scale to correct range
        action = action * self.action_scale
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

class PolicyNetworkGMM(PolicyNetwork):
    def __init__(self, *args, n_gmm_components=5, **kwargs):
        self.n_gmm_components = n_gmm_components
        # self.gmm_components is not returned in self.parameters()
        self.gmm_components = torch.nn.init.uniform_(torch.empty((n_gmm_components,), requires_grad=True), -1.0, 1.0)
        super(PolicyNetworkGMM, self).__init__(*args, **kwargs)

    def _create_output_layer(self):
        self.mean_linear = nn.Linear(self.layer_dim, self.action_dim * self.n_gmm_components)
        self.mean_linear.weight.data.uniform_(-self.init_w, self.init_w)
        self.mean_linear.bias.data.uniform_(-self.init_w, self.init_w)

        self.log_std_linear = nn.Linear(self.layer_dim, self.action_dim * self.n_gmm_components)
        self.log_std_linear.weight.data.uniform_(-self.init_w, self.init_w)
        self.log_std_linear.bias.data.uniform_(-self.init_w, self.init_w)

    def forward(self, state):
        mean, std = super(PolicyNetworkGMM, self).forward(state)
        mean = torch.reshape(mean, list(mean.shape[:-1]) + [self.n_gmm_components, self.action_dim] )
        std = torch.reshape(std, list(std.shape[:-1]) + [self.n_gmm_components, self.action_dim])
        return mean, std


    def evaluate(self, state, epsilon=1e-6, no_grad=None):
        all_pre_mean, all_std = self.forward(state)
        batch_size = int(all_pre_mean.shape[0])

        latent_dist = Categorical(logits=self.gmm_components)
        sampled_latent = latent_dist.sample(list(all_pre_mean.shape[:-2]))

        all_normals = self.get_distribution(all_pre_mean, all_std)

        this_pre_mean = all_pre_mean[range(batch_size), sampled_latent]
        this_std = all_std[range(batch_size), sampled_latent]
        normal = self.get_distribution(this_pre_mean, this_std)

        raw_action = normal.sample()
        action = torch.tanh(raw_action)

        all_log_probs = all_normals.log_prob(raw_action.unsqueeze(1))
        if self.action_dim == 1:
            all_log_probs.squeeze_(-1)

        log_prob = torch.logsumexp(all_log_probs + self.gmm_components, dim=-1)
        log_prob -= torch.logsumexp(self.gmm_components, dim=-1)
        if len(log_prob.shape) == 1:
            log_prob.unsqueeze_(-1)

        log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + epsilon).sum(dim=-1, keepdim=True)

        # scale to correct range
        action = action * self.action_scale

        mean = torch.tanh(this_pre_mean) * self.action_scale
        return action, log_prob, raw_action, this_pre_mean, mean, this_std

    def evaluate_multiple(self, state, num_actions, epsilon=1e-6, no_grad=None):
        # state: (batch_size, state_dim)
        all_pre_mean, all_std = self.forward(state)
        batch_size = int(all_pre_mean.shape[0])

        latent_dist = Categorical(logits=self.gmm_components)
        sampled_latent = latent_dist.sample(list(all_pre_mean.shape[:-2]) + [num_actions])
        all_normals = self.get_distribution(all_pre_mean, all_std)

        unr_sampled_latent = sampled_latent.reshape(-1)
        repeated_idxs = np.concatenate([[i] * num_actions for i in range(batch_size)])
        unr_this_pre_mean = all_pre_mean[repeated_idxs, unr_sampled_latent, :]
        unr_this_std = all_std[repeated_idxs, unr_sampled_latent, :]
        this_pre_mean = unr_this_pre_mean.reshape([batch_size, num_actions, -1])
        this_std = unr_this_std.reshape([batch_size, num_actions, -1])
        normal = self.get_distribution(this_pre_mean, this_std)

        raw_action = normal.sample()
        action = torch.tanh(raw_action)

        all_log_probs = all_normals.log_prob(raw_action.permute(1,0,2).unsqueeze(2))
        if self.action_dim == 1:
            all_log_probs.squeeze_(-1)
        all_log_probs = all_log_probs.permute(1,0,2)

        log_prob = torch.logsumexp(all_log_probs + self.gmm_components, dim=-1)
        log_prob -= torch.logsumexp(self.gmm_components, dim=-1)

        if len(log_prob.shape) == 2:
            log_prob.unsqueeze_(-1)

        log_prob -= torch.log(self.action_scale * (1 - action.pow(2)) + epsilon).sum(dim=-1, keepdim=True)

        log_prob = log_prob.squeeze(-1)  # (batch_size, num_actions)

        # scale to correct range
        action = action * self.action_scale

        return action, log_prob, None, None, None, None

    def get_logprob(self, states, tiled_actions, epsilon = 1e-6):
        raise NotImplementedError

    def get_action(self, state):
        raise NotImplementedError

if __name__ == "__main__":
    test_p = [(32, 3, 7, 128),(1, 3, 7, 128),(32, 1, 7, 128),(32, 3, 1, 128),(32, 3, 7, 1),(1, 1, 1, 1)]

    for batch_size, state_size, action_size, num_actions in test_p:
        state_batch = torch.ones([batch_size, state_size])
        # Regular
        reg_nn = PolicyNetwork(state_size, action_size, num_actions, 1)
        actions_single, log_pdfs_single, _, _, _, _ = reg_nn.evaluate(state_batch)
        actions_mult, log_pdfs_mult, _, _, _, _ = reg_nn.evaluate_multiple(state_batch, num_actions)
        # GMM
        gmm_nn = PolicyNetworkGMM(state_size, action_size, num_actions, 1)
        gmm_actions_single, gmm_log_pdfs_single, _, _, _, _ = gmm_nn.evaluate(state_batch)
        gmm_actions_mult, gmm_log_pdfs_mult, _, _, _, _ = gmm_nn.evaluate_multiple(state_batch, num_actions)
        # print("Single:\n\tAction: {} X {}\n\tLogprob: {} x {}".format(actions_single.shape, gmm_actions_single.shape, log_pdfs_single.shape, gmm_log_pdfs_single.shape))
        # print("Mult:\n\tAction: {} X {}\n\tLogprob: {} x {}".format(actions_mult.shape, gmm_actions_mult.shape,log_pdfs_mult.shape,gmm_log_pdfs_mult.shape))
        assert(actions_single.shape == gmm_actions_single.shape)
        assert(log_pdfs_single.shape == gmm_log_pdfs_single.shape)
        assert(actions_mult.shape == gmm_actions_mult.shape)
        assert(log_pdfs_mult.shape == gmm_log_pdfs_mult.shape)

