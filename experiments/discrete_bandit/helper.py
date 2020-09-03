import os
import sys
sys.path.append(os.getcwd())

import torch 
import torch.nn as nn
import numpy as np

n_actions = 3

# for learning action value function if desired
class batch_Q_approx(nn.Module):
    def __init__(self, hidden, n_nets):
        super().__init__()
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_actions)
            )
            for _ in range(n_nets)
        ])

    def BQ_forward(self):
        # for evaluating all nets and use in kl objective
        # result is of dim (n_nets, n_actions)
        return torch.stack([net(torch.tensor([1.])) for net in self.nets])

    def forward(self, a):
        # assume the rows of a are separate slices to apply to each net
        # return torch.stack([net(a[i]).squeeze() for i, net in enumerate(self.nets)])
        return torch.stack([net(torch.tensor([1.])) for net in self.nets])



# math tools
def sample_boltzmann(probs):
    # assume of shape (n_actions, n_inits)
    assert probs.shape == (n_actions, n_inits)
    # need to permute
    # torch.distributions.Categorical maps from (m, n) --> (m)
    m = torch.distributions.Categorical(probs = probs.permute(1, 0))
    return m.sample()

def approx_log(x):
    # for numerical stability
    return torch.log(x + 1e-5)

def init_params(param_range):
    return nn.Parameter(torch.tensor(param_range))

def get_random_inits(logitspace, n_samples):
    logits = np.random.uniform(low = logitspace[0], high = logitspace[1], size = n_samples)
    return logits
