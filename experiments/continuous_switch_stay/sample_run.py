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
import env

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
OPTIMIZERS = ["adam", "rmsprop"]
OPTIMS = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop, "sgd": torch.optim.SGD}
LRS = [0.005]
# KL = ["reverse", "forward"]
KL = ["reverse"]
# TAUS = [0, 0.01, 0.1, 0.4, 1]
TAUS = [0.01, 0.1, 0.4, 1]
n_action_points = 500
n_samples = 1000 # number of iterates
MUSPACE = (-0.95, 0.95)
# std_sets = [(0.1, 0.3), (0.3, 0.5), (0.5, 1)]
# std_sets = [(0.1, 0.3), (0.3, 1.)]
std_sets = [(0.1, 1)]
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
write_dir = "data/cont-switch-stay/polytope/{}samples/std-{},{}/".format(n_action_points, std_left, std_right)
if learnQ is True:
    write_dir += "learnQ/"
else:
    write_dir += "notlearnQ/"
os.makedirs(write_dir, exist_ok=True)
np.save(write_dir + exp_name + "_polytope.npy", env.generate_boundary(gamma))

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
    assert pi.shape == (n_states, n_inits, 2), pi.shape
    pi_t += time.time() - t0
    
    t0 = time.time()
    # calculate the true value functions
    V = get_batch_V(pi.detach().numpy(), gamma)
    
    # sample actions for everything except Hard FKL
    # print(logstds.min())
    if direction != "forward" or tau != 0:
        m = torch.distributions.Normal(loc = mus, scale = transform_std_dev(logstds))
        actions = m.sample_n(n_action_points)
        orig = m.log_prob(actions)
        correction = torch.log(1 - torch.tanh(actions).pow(2) + 1e-5)
        assert torch.isnan(orig).sum() == 0
        assert torch.isnan(correction).sum() == 0
        log_pdfs = orig - correction
        assert log_pdfs.shape == (n_action_points, mus.shape[0], mus.shape[1]), (log_pdfs.shape, mus.shape)

        # get action values
        with torch.no_grad():
            discrete_actions = (torch.tanh(actions) > 0).long().permute(1, 0, 2)
        # print(discrete_actions.shape)
        assert discrete_actions.shape == (n_states, n_action_points, n_inits)
        assert V.shape == (n_inits, n_states)
        Ps = torch.zeros((n_action_points, n_inits, n_states, n_states))
        rs = torch.zeros((n_action_points, n_inits, n_states))
        # Ps2 = torch.zeros((n_action_points, n_inits, n_states, n_states))
        # rs2 = torch.zeros((n_action_points, n_inits, n_states))
        for state in range(2):
            Ps[:, :, state, :] = torch.index_select(torch.tensor(env.P)[state, :, :], 1, discrete_actions[state, :, :].flatten()).reshape(n_states, n_action_points, n_inits).permute(1, 2, 0)
            rs[:, :, state] = torch.tensor(env.r)[state, :][discrete_actions[state, :, :]]
            # for init in range(n_inits):
            #     for action in range(n_action_points):
        #             # Ps2[:, :, state, :] = torch.index_select(torch.tensor(env.P)[state, :, :], 1, discrete_actions[state, :, :].flatten()).reshape(n_states, n_action_points, n_inits)
        #             Ps2[action, init, state, :] = torch.tensor(env.P)[state, discrete_actions[state, action, init], :]
                    # rs2[action, init, state] = torch.tensor(env.r)[state, discrete_actions[state, action, init]]
        # assert torch.all(torch.eq(Ps, Ps2))
        # assert torch.all(torch.eq(rs, rs2)), (rs, rs2)
        # print(torch.tensor(V).unsqueeze(1).unsqueeze(0).shape)
        if tau == 0:
            QA = rs  + gamma * torch.sum(Ps * torch.tensor(V).unsqueeze(1).unsqueeze(0), axis = -1)
        else:
            V_soft = get_batch_soft_V(pi.detach().numpy(), gamma, tau)
            QA = rs + gamma * torch.sum(Ps * torch.tensor(V_soft).unsqueeze(1).unsqueeze(0), axis = 3)
        QA = QA.permute(0, 2, 1)
        assert QA.shape == log_pdfs.shape, (QA.shape, log_pdfs.shape)
    else:
        Q = torch.tensor(get_batch_Q(pi.detach().numpy(), gamma)).permute(1, 2, 0)
        pdf = torch.stack([tanh_gauss(env.points.unsqueeze(-1), mus[s, :].unsqueeze(0), transform_std_dev(logstds[s, :]).unsqueeze(0)) for s in range(n_states)])
        assert pdf.shape == (n_states, env.n_points, n_inits), print(pdf.shape)
        with torch.no_grad():
            assert Q.shape == (n_states, env.n_points, n_inits), Q.shape
            max_Q_tau = torch.max(Q / tau, dim = 1, keepdim=True)[0]
            normed_Q_arg = Q / tau - max_Q_tau
            BQ = torch.exp(normed_Q_arg) / torch.sum(torch.exp(normed_Q_arg), dim = 1, keepdim=True)
            max_action = torch.zeros((n_states, n_inits), dtype = torch.int64)
            for s in range(n_states):
                for n in range(n_inits):
                    ma = (np.random.random(env.n_points) * (Q[s, :, n] == Q[s, :, n].max()).numpy()).argmax()
                    # print(np.random.random() * (Q[s, :, n] == Q[s, :, n].max()).numpy())
                    max_action[s, n] = int(ma)
            # max_action = torch.max(Q, dim = 1)[1]
            assert max_action.shape == (n_states, n_inits), max_action.shape
            # print(Q.shape, max_action.unsqueeze(1).shape)
            # print(max_action)
            assert BQ.shape == (n_states, env.n_points, n_inits), BQ.shape
    value_t += time.time() - t0
    V_values.append(V) # should be of shape (STEPS, n_inits, n_states)
    
    t0 = time.time()
    if direction == "reverse":
        if tau == 0:
            losses = -torch.mean(QA * log_pdfs, axis = 0).mean(axis = 0)
        else:
            losses = torch.mean(log_pdfs + log_pdfs.detach() * log_pdfs - QA * log_pdfs / tau, axis = 0).mean(axis = 0)
            # print((pi /).shape)
    elif direction == "forward":
        if tau == 0:
            pdf = torch.gather(pdf, 1, max_action.unsqueeze(1)).squeeze()
            # Get the pi corresponding to the max action
            assert pdf.shape == (n_states, n_inits), pdf.shape
            losses = -approx_log(pdf).mean(0)
        else:
            # do weighted importance sampling
            with torch.no_grad():
                max_arg = (QA / tau - log_pdfs).max(0, keepdim=True)[0]
                rho = torch.exp(QA / tau - log_pdfs - max_arg)
            # print(rho.shape, QA.shape, log_pdfs.shape)
            WIR = rho / rho.sum(0, keepdim=True)
            # print(WIR, rho)
            losses = - torch.sum(WIR * log_pdfs, axis = 0).mean(axis = 0)
            # print(losses.max())
            
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
np.save(write_dir + exp_name + "_V.npy", V_values)
np.save(write_dir + exp_name + "_polytope.npy", env.generate_boundary(gamma))
