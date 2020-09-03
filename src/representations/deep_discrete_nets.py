import torch 
import torch.nn as nn

class VQPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden, combo = ["v", "q", "pi"], separate_bodies = False):
        super().__init__()
        self.separate_bodies = separate_bodies
        # print(combo)
        if self.separate_bodies is False:
            self.body = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU()
            )

            self.v = nn.Linear(hidden, 1)
            self.q = nn.Linear(hidden, action_dim)
            self.policy_net = nn.Sequential(
                nn.Linear(hidden, action_dim),
                nn.Softmax(dim = -1)
            )
        else:
            self.v = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )
            self.q = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_dim)
            )
            self.policy_net = nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, action_dim),
                nn.Softmax(dim = -1)
            )
        self.combo = combo

    def forward(self, x):
        if self.separate_bodies is False:
            x = self.body(x)
        res = []
        if "v" in self.combo:
            res.append(self.v(x))
        if "q" in self.combo:
            res.append(self.q(x))
        if "pi" in self.combo:
            res.append(self.policy_net(x))
        if len(res) == 1:
            return res[0]
        else:
            return res
            