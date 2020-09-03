import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import time

from helper import *

torch.set_default_dtype(torch.float)

n_states = 2
n_actions = 2


# hyperparameters
gamma = 0.9
STEPS = 500
OPTIMIZERS = ["adam", "sgd", "rmsprop"]
OPTIMS = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop, "sgd": torch.optim.SGD}
LRS = [1e-2, 5e-3]
KL = ["reverse", "forward"]
TAUS = [0, 0.01, 0.1, 1]

# get the lines of the true polytope
# generate semideterministic policies
n_points = 200
semi_det_pis = [np.array([[1, 0], [x , 1 - x]]) for x in np.linspace(0, 1, n_points)] + [np.array([[0, 1], [x , 1 - x]]) for x in np.linspace(0, 1, n_points)] + [np.flip(np.array([[1, 0], [x , 1 - x]]), 0) for x in np.linspace(0, 1, n_points)] + [np.flip(np.array([[0, 1], [x , 1 - x]]), 0) for x in np.linspace(0, 1, n_points)]
polytope_boundary = np.array([get_V(pi, gamma) for pi in semi_det_pis])
# print(polytope_boundary.shape)


n_samples = 1000
LOGITSPACE = (-1, 1)

n_total_experiments = np.prod([len(a) for a in [KL, LRS, OPTIMIZERS, TAUS]])
n_done = 0

def init_params(param_range):
    return nn.Parameter(torch.tensor(param_range))

def get_random_inits(logitspace, n_samples):
    logits = np.random.uniform(low = logitspace[0], high = logitspace[1], size = n_samples)
    return logits

curr_exp = int(sys.argv[1])

accum = 1
params = []
for ix, l in enumerate([KL, LRS, OPTIMIZERS, TAUS]):
    params.append(l[(curr_exp // accum) % len(l)])
    accum *= len(l)
direction, lr, optim_name, tau = params
print(params)

exp_name = "{}_lr={}_optim={}_temp={}".format(direction, lr, optim_name, tau)

# reset the init
logits = get_random_inits(LOGITSPACE, n_samples * n_actions * n_states)
n_inits = n_samples
logits = init_params(logits.reshape((n_states, n_actions, n_samples)))
# print(logits)
optim = OPTIMS[optim_name](params = [logits], lr = lr)

V_values = []

learnQ = False
Q_approx = torch.zeros((n_states, n_actions, n_inits))
V_approx = torch.zeros((n_states, n_inits))

s = torch.zeros(n_inits, dtype = torch.long) # keep track of current state of each iterate
sp_map = torch.zeros((n_states, n_actions), dtype = torch.long) # maps from s, a to sp
sp_map[0, 1] = 1
sp_map[1, 0] = 1

print("running {} | {} / {}".format(exp_name, curr_exp + 1, n_total_experiments))
print("learnQ = {}".format(learnQ))

t_start = time.time()
for _ in range(STEPS):
    # get a batch pdf with all the actions
    max_logits = torch.max(logits, dim = 1, keepdim=True)[0]
    pi = torch.exp(logits - max_logits) / torch.sum(torch.exp(logits - max_logits), dim = 1, keepdim=True)
    assert pi.shape == (n_states, n_actions, n_inits), print(pi.shape)

    # state transition
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

    # V = [get_V(pi[:, :, i].detach().numpy(), gamma) for i in range(n_inits)]
    V = get_batch_V(pi.detach().numpy(), gamma)
    assert np.array_equal(V, np.array([get_V(pi[:, :, i].detach().numpy(), gamma) for i in range(n_inits)]))
    if learnQ is False:
        if tau == 0:
            Q = torch.tensor(get_batch_Q(pi.detach().numpy(), gamma)).permute(1, 2, 0)
            # assert torch.all(torch.eq(Q, torch.tensor([get_Q(pi[:, :, i].detach().numpy(), gamma) for i in range(n_inits)]).permute(1, 2, 0)))
        else:
            Q = torch.tensor(get_batch_soft_Q(pi.detach().numpy(), gamma, tau)).permute(1, 2, 0)
            # assert torch.all(torch.eq(Q, torch.tensor([get_soft_Q(pi[:, :, i].detach().numpy(), gamma, tau) for i in range(n_inits)]).permute(1, 2, 0))), (Q - torch.tensor([get_soft_Q(pi[:, :, i].detach().numpy(), gamma, tau) for i in range(n_inits)]).permute(1, 2, 0))
    else:
        Q = Q_approx

    with torch.no_grad():
        max_Q = torch.max(Q, dim = 1, keepdim=True)[0]
        BQ = torch.exp((Q - max_Q) / tau) / torch.sum(torch.exp((Q - max_Q) / tau), dim = 1, keepdim=True)
        max_action = torch.max(Q, dim = 1)[1]
        assert Q.shape == (n_states, n_actions, n_inits), Q.shape
        assert BQ.shape == (n_states, n_actions, n_inits), BQ.shape
        assert max_action.shape == (n_states, n_inits), max_action.shape
    V_values.append(V) # should be of shape (STEPS, n_inits, n_states)
    if direction == "reverse":
        if tau == 0:
            losses = -torch.sum(Q * pi, dim = 1).mean(0)
        else:
            losses = torch.sum(pi * approx_log(pi / (BQ + 1e-5)), dim = 1).mean(0)
    elif direction == "forward":
        if tau == 0:
            # print(max_action.unsqueeze(0).shape, pi.shape)
            max_pis = torch.gather(pi, 1, max_action.unsqueeze(1))
            assert max_pis.numel() == n_states * n_inits, max_pis.shape
            losses = -approx_log(max_pis).mean(0)
        else:
            losses = torch.sum(BQ * approx_log(BQ / (pi + 1e-5)), dim = 1).mean(0)
            
    # assert losses.shape[0] == n_inits, losses.shape
    assert losses.numel() == n_inits, losses.shape
    optim.zero_grad()
    loss = losses.sum()
    assert torch.isnan(loss) == 0
    loss.backward()
    optim.step()

    

t_end = time.time()
print("experiment took {}s".format(t_end - t_start))

# save data
V_values = np.array(V_values)
# print(V_values.shape)
write_dir = "data/switch-stay/polytope/"
if learnQ is True:
    write_dir += "learnQ/"
else:
    write_dir += "notlearnQ/"
os.makedirs(write_dir, exist_ok=True)
np.save(write_dir + exp_name + "_V.npy", V_values)
np.save(write_dir + exp_name + "_polytope.npy", polytope_boundary)