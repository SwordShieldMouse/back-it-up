import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import time
import scipy.special
import quadpy

from helper import *
from env import *

from src.agents.tabular_agents import *

torch.set_default_dtype(torch.float)

n_states = 2
n_actions = 2
curr_exp = int(sys.argv[1])
std_ix = int(sys.argv[2])


# hyperparameters
learnQ = False
gamma = 0.9
STEPS = 500
OPTIMIZERS = ["adam", "rmsprop", "sgd"]
OPTIMS = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop, "sgd": torch.optim.SGD}
LRS = [0.005, 0.001]
KL = ["reverse", "forward"]
# KL = ["reverse"]
TAUS = [0, 0.01, 0.1, 0.4, 1]
# TAUS = [0]
n_samples = 1000 # number of iterates
MUSPACE = (-0.95, 0.95)
std_sets = [(0.1, 1)]
# std_sets = [(0.1, 0.3), (0.3, 1.)]
std_left = std_sets[std_ix][0]
std_right = std_sets[std_ix][1]
LOGSTDSPACE = (np.log(np.exp(std_left) - 1), np.log(np.exp(std_right) - 1))

n_total_experiments = np.prod([len(a) for a in [KL, LRS, OPTIMIZERS, TAUS]])
n_done = 0

accum = 1
params = []
for ix, l in enumerate([KL, LRS, OPTIMIZERS, TAUS]):
    params.append(l[(curr_exp // accum) % len(l)])
    accum *= len(l)
direction, lr, optim_name, tau = params
print(params)

exp_name = "{}_lr={}_optim={}_temp={}".format(direction, lr, optim_name, tau)
write_dir = "data/cont-switch-stay/polytope/std-{},{}/".format(std_left, std_right)
if learnQ is True:
    write_dir += "learnQ/"
else:
    write_dir += "notlearnQ/"
os.makedirs(write_dir, exist_ok=True)
np.save(write_dir + exp_name + "_polytope.npy", generate_boundary(gamma))

# initialize iterates
mus, logstds = get_random_inits(MUSPACE, LOGSTDSPACE, n_samples * n_states)
n_inits = n_samples
mus = init_params(mus.reshape((n_states, n_samples)))
logstds = init_params(logstds.reshape((n_states, n_samples)))
optim = OPTIMS[optim_name](params = [mus, logstds], lr = lr)

V_values = []

# for learning value functions if desired
Q_approx = torch.zeros((n_states, n_actions, n_inits))
V_approx = torch.zeros((n_states, n_inits))

s = torch.zeros(n_inits, dtype = torch.long) # keep track of current state of each iterate
sp_map = torch.zeros((n_states, n_actions), dtype = torch.long) # maps from s, a to sp
sp_map[0, 1] = 1
sp_map[1, 0] = 1

print("running {} | {} / {}".format(exp_name, curr_exp + 1, n_total_experiments))
print("learnQ = {} | initial std in ({}, {})".format(learnQ, std_left, std_right))
loss_t = 0
value_t = 0
pi_t = 0
t_start = time.time()
for step in range(STEPS):
    if (step + 1) % 100 == 0:
        loss_frac = loss_t / (pi_t + loss_t + value_t)
        value_frac = value_t / (pi_t + loss_t + value_t)
        pi_frac = pi_t / (pi_t + loss_t + value_t)
        print("{} / {} steps | fps = {} | value time = {} | loss time = {} | pi time = {}".format(step + 1, STEPS, step / (time.time() - t_start), value_frac, loss_frac, pi_frac))
    # get a batch pdf with all the actions
    t0 = time.time()
    pi = torch.cat([mus.unsqueeze(-1), transform_std_dev(logstds).unsqueeze(-1)], dim = -1)
    pdf = torch.stack([tanh_gauss(points.unsqueeze(-1), mus[s, :].unsqueeze(0), transform_std_dev(logstds[s, :]).unsqueeze(0)) for s in range(n_states)])
    assert pi.shape == (n_states, n_inits, 2), pi.shape
    assert pdf.shape == (n_states, n_points, n_inits), print(pdf.shape)
    pi_t += time.time() - t0
    # state transition if value functions are being learned
    if learnQ is True:
        with torch.no_grad():
            curr_pi = torch.cat([torch.gather(pi[:, 0, :], 0, s.view((1, n_inits))), torch.gather(pi[:, 1, :], 0, s.view((1, n_inits)))], dim = 0)
            assert curr_pi.shape == (n_actions, n_inits), curr_pi.shape
            m = torch.distributions.Categorical(probs = curr_pi.permute(1, 0))
            a = m.sample()
            assert a.numel() == n_inits, a.shape
            sp = sp_map[s, a]
            assert sp.shape == s.shape, sp.shape

            # update Q and V
            R = torch.tensor(r)[s, a].squeeze()
            H = -torch.sum(curr_pi * torch.log(curr_pi), dim = 0).squeeze()
            assert H.numel() == n_inits, H.shape
            assert R.numel() == n_inits, R.shape
            td_target = R + gamma * V_approx[s, np.arange(s.shape[0])]
            assert td_target.numel() == n_inits, td_target.shape
            V_approx[s, np.arange(s.shape[0])] += 1e-1 * (td_target + tau * H - V_approx[s, np.arange(s.shape[0])])
            Q_approx[s, a, np.arange(s.shape[0])] += 1e-1 * (td_target - Q_approx[s, a, np.arange(s.shape[0])])
            s = sp
    t0 = time.time()
    # calculate the true value functions
    V = get_batch_V(pi.detach().numpy(), gamma)
    # assert np.array_equal(V,np.array([get_V(pi[:, i, :].detach().numpy(), gamma) for i in range(n_inits)]))
    # V = [get_V(pi[:, i, :].detach().numpy(), gamma) for i in range(n_inits)]
    if learnQ is False:
        if tau == 0:
            Q = torch.tensor(get_batch_Q(pi.detach().numpy(), gamma)).permute(1, 2, 0)
            # assert torch.all(torch.eq(Q, torch.tensor([get_Q(pi[:, i, :].detach().numpy(), gamma) for i in range(n_inits)]).permute(1, 2, 0)))
        else:
            Q = torch.tensor(get_batch_soft_Q(pi.detach().numpy(), gamma, tau)).permute(1, 2, 0)
            
    else:
        Q = Q_approx
    value_t += time.time() - t0

    with torch.no_grad():
        assert Q.shape == (n_states, n_points, n_inits), Q.shape
        max_Q_tau = torch.max(Q / tau, dim = 1, keepdim=True)[0]
        normed_Q_arg = Q / tau - max_Q_tau
        BQ = torch.exp(normed_Q_arg) / torch.sum(torch.exp(normed_Q_arg), dim = 1, keepdim=True)
        if direction == "forward" and tau == 0:
            max_action = torch.zeros((n_states, n_inits), dtype = torch.int64)
            for s in range(n_states):
                for n in range(n_inits):
                    ma = (np.random.random(n_points) * (Q[s, :, n] == Q[s, :, n].max()).numpy()).argmax()
                    # print(np.random.random() * (Q[s, :, n] == Q[s, :, n].max()).numpy())
                    max_action[s, n] = int(ma)
            # max_action = torch.max(Q, dim = 1)[1]
            assert max_action.shape == (n_states, n_inits), max_action.shape
        # print(Q.shape, max_action.unsqueeze(1).shape)
        # print(max_action)
        assert BQ.shape == (n_states, n_points, n_inits), BQ.shape
    V_values.append(V) # should be of shape (STEPS, n_inits, n_states)
    
    t0 = time.time()
    # calculate the losses
    if direction == "reverse":
        if tau == 0:
            losses = -torch.sum(weights.view((1, -1, 1)) * Q.detach() * pdf, dim = 1).mean(0)
        else:
            losses = torch.sum(weights.view((1, -1, 1)) * pdf * approx_log(pdf / (BQ + 1e-5)), dim = 1).mean(0)
    elif direction == "forward":
        if tau == 0:
            pdf = torch.gather(pdf, 1, max_action.unsqueeze(1)).squeeze()
            # Get the pi corresponding to the max action
            assert pdf.shape == (n_states, n_inits), pdf.shape
            losses = -approx_log(pdf).mean(0)
        else:
            losses = torch.sum(weights.view((1, -1, 1)) * BQ * approx_log(BQ / (pdf + 1e-5)), dim = 1).mean(0)

    assert losses.numel() == n_inits, losses.shape
    optim.zero_grad()
    loss = losses.sum()
    assert torch.isnan(loss) == 0, losses[losses != losses].numel()
    loss.backward()
    optim.step()
    loss_t += time.time() - t0

    

t_end = time.time()
print("experiment took {}s".format(t_end - t_start))

# save data
V_values = np.array(V_values)
# print(V_values.shape)
os.makedirs(write_dir, exist_ok=True)
np.save(write_dir + exp_name + "_V.npy", V_values)
