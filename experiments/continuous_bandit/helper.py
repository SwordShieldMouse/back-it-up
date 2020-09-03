import os 
import sys
sys.path.append(os.getcwd())

import torch 
import torch.nn as nn
import numpy as np
import src.utils.math_utils as utils

class batch_Q_approx(nn.Module):
    def __init__(self, hidden, n_nets):
        super().__init__()
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )
            for _ in range(n_nets)
        ])

    def BQ_forward(self, a):
        # for evaluating all nets at a
        return torch.stack([net(a).squeeze() for net in self.nets])

    def forward(self, a):
        # assume the rows of a are separate slices to apply to each net
        return torch.stack([net(a[i]).squeeze() for i, net in enumerate(self.nets)])


def init_params(param_range):
    # return [nn.Parameter(torch.tensor(param)) for param in param_range]
    return nn.Parameter(torch.tensor(param_range))

def get_random_inits(muspace, logstdspace, n_samples, n_modes):
    mus = []
    logstds = []
    if n_modes == 1:
        mus = np.random.uniform(low = muspace[0], high = muspace[1], size = n_samples)
        logstds = np.random.uniform(low = logstdspace[0], high = logstdspace[1], size = n_samples)
    elif n_modes > 1:
        mus = np.random.uniform(low = muspace[0], high = muspace[1], size = (n_samples, n_modes))
        logstds = np.random.uniform(low = logstdspace[0], high = logstdspace[1], size = (n_samples, n_modes))
    return mus, logstds


def approx_log(x):
    # for numerical stability
    return torch.log(x + 1e-5)

# bandit stuff
def Q(a):
    a.clamp_(min = -1, max = 1)
    return torch.exp(-1/2 * torch.pow((2 * a + 1) / 0.2, 2)) + 1.5 * torch.exp(-1/2 * torch.pow((2 * a - 1) / 0.2, 2))


def Z(tau, weights, points):
    # partition func of exp(tau^{-1} Q) from -1 to 1
    assert tau > 0, "temp = {}".format(tau)
    return integrate(lambda a: torch.exp(Q(a) / tau), weights, points)

def gauss(a, mu, sigma):
    # print(sigma.shape)
    assert torch.min(sigma) > 0, "sigma = {}".format(sigma)
    norm = sigma * np.sqrt(2 * np.pi)
    pdf = torch.exp(-1/2 * torch.pow((a - mu) / sigma, 2)) / norm
    assert torch.min(pdf) >= 0, (pdf[pdf < 0], pdf[pdf != pdf])
    # assert pdf.shape == (a.numel(), mu.numel()), print(a.shape, mu.shape)
    # assert pdf.numel() == a.numel() * mu.numel(), a.numel()
    # assert pdf[pdf > 0].numel() == pdf.numel(), print(pdf, mu, sigma)
    return pdf

def gauss_mix(a, mus, sigmas, coeffs):
    assert torch.min(coeffs) >= 0, coeffs
    assert torch.max(coeffs) <= 1, coeffs
    assert torch.min(sigmas) > 0, "non-positive sigma"
    # print(mus, sigmas)
    pdf_list = torch.stack([gauss(a, mus[:, :, i], sigmas[:, :, i]) * coeffs[:, i] for i in range(coeffs.shape[-1])])
    # assert pdf_list.shape == (coeffs.shape[-1], a.shape[0], mus.shape[1])
    pdf = torch.sum(pdf_list, dim = 0)
    assert torch.min(pdf) >= 0, pdf
    # assert pdf.shape[0] == a.shape[0]
    return pdf

def tanh_gauss_mix(a, mus, sigmas, coeffs):
    norm = 1 - torch.tanh(utils.atanh(a)).pow(2)
    before = gauss_mix(utils.atanh(a), mus, sigmas, coeffs)
    assert torch.isnan(norm).sum() == 0
    assert torch.isnan(before).sum() == 0
    pdf = before / norm 
    assert torch.min(pdf) >= 0
    # assert pdf.shape == (a.numel(), mus.shape[1])
    assert pdf.numel() == a.numel() * mus.shape[1]
    # assert pdf[pdf > 0].numel() == pdf.numel()
    return pdf

def tanh_gauss(a, mu, sigma): 
    norm = 1 - torch.tanh(utils.atanh(a)).pow(2)
    before = gauss(utils.atanh(a), mu, sigma) 
    assert torch.isnan(norm).sum() == 0
    assert torch.isnan(before).sum() == 0
    pdf = before / norm
    assert torch.min(pdf) >= 0, pdf
    # assert pdf.shape == (a.numel(), mu.numel())
    # assert pdf.numel() == a.numel() * mu.numel()
    # assert pdf[pdf > 0].numel() == pdf.numel()
    return pdf

def sample_tanh_gauss(mu, sigma):
    assert sigma[sigma > 0].numel() == sigma.numel()
    m = torch.distributions.Normal(loc = mu, scale = sigma)
    return torch.tanh(m.sample())

def sample_tanh_gauss_mix(mus, sigmas, coeffs):
    # mu.shape = (n_inits, n_modes)
    # coeffs.shape = (n_inits, n_modes)
    n_modes = mus.shape[1]
    assert sigmas[sigmas > 0].numel() == sigmas.numel()
    assert torch.min(coeffs) > 0, torch.min(coeffs)
    m_coeff = torch.distributions.Categorical(probs = coeffs)
    ixs = m_coeff.sample()
    mu = torch.gather(mus, -1, ixs.unsqueeze(-1))
    sigma = torch.gather(sigmas, -1, ixs.unsqueeze(-1))
    m = torch.distributions.Normal(loc = mu, scale = sigma)
    return torch.tanh(m.sample())


# math tools
def integrate(f, weights, points):
    # print(f(points).shape, weights.shape)
    return torch.sum(weights * f(points))

def entropy(pi, weights, points):
    # print(pi(points))
    # print(approx_log(pi(points)))
    H = -integrate(lambda a: pi(a) * approx_log(pi(a)), weights, points)
    return H

def transform_std_param(std_param):
    # return the true std given std param
    return torch.log(1 + torch.exp(std_param))