import torch
import re
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy as sp
from collections import defaultdict
import time

torch.set_default_dtype(torch.float)

def atanh(x):
    return (torch.log(1 + x) - torch.log(1 - x)) / 2

def normal_cdf(x, mean, std):
    return (1 + sp.special.erf((x - mean) / (std * np.sqrt(2)))) / 2

def Beta(alpha, beta):
    return torch.exp(torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta))

def soft_update(target, src, tau):
    for target_param, param in zip(target.parameters(), src.parameters()):
        target_param.detach_()
        target_param.copy_(target_param * (1.0 - tau) +
                           param * tau)

def approx_log(x):
    return torch.log(x + 1e-5)

def rand_argmax(array):
    # get argmax of numpy array with random tie-breaking
    return np.argmax(np.random.random(array.shape) * (array == array.max()))
    # return np.random.choice(np.where(array == array.max())[0])

def gen_net(in_dim, out_dim, layers, hidden, act_final):
    # generate neural nets of given size
    l = []
    if layers == 0:
        # assuming linear network
        l.append(nn.Linear(in_dim, out_dim))
    else:
        l.append(nn.Linear(in_dim, hidden))
        for _ in range(layers - 1):
            l.append(nn.ReLU())
            l.append(nn.Linear(hidden, hidden))
        l.append(nn.ReLU())
        l.append(nn.Linear(hidden, out_dim))
    if act_final == "relu":
        l.append(nn.ReLU())
    elif act_final == "softmax":
        l.append(nn.Softmax(dim = -1))
    # print(l)
    return nn.Sequential(*l)
