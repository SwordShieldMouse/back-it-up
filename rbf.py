import numpy as np
import torch

class Gaussian():
    def __init__(self, obs_dim, n_features):
        # choose random centres
        self.centres = np.random.randn(n_features, obs_dim)

    def __call__(self, s):
        try:
            return np.power(self.centres - s.view(1, -1), 2).sum(axis = -1).exp()
        except TypeError:
            return torch.exp(-torch.pow(torch.tensor(self.centres, dtype = torch.float) - s.view(1, -1), 2).sum(dim = -1))

class Fourier():
    def __init__(self, obs_dim, n_features, dtype = 'torch'):
        self.dtype = dtype
        if self.dtype == "torch":
            self.W = torch.randn(n_features, obs_dim)
            self.b = torch.rand(n_features) * 2 * np.pi
        elif self.dtype == "numpy":
            self.W = np.random.randn(n_features, obs_dim)
            self.b = np.random.rand(n_features) * 2 * np.pi
        self.n_features = n_features
        self.K = np.sqrt(2 / self.n_features)

    def __call__(self, s):
        if self.dtype == "torch":
            if len(s.shape) == 1:
                s.unsqueeze_(-1)
            cos_term = torch.cos(torch.matmul(self.W, s) + self.b.unsqueeze(-1))
            return self.K * cos_term.squeeze()
        else:
            cos_term = np.cos(np.matmul(self.W, s).squeeze() + self.b)
            return self.K * cos_term.squeeze()