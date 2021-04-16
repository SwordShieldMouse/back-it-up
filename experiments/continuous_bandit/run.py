import os
import sys
sys.path.append(os.getcwd())

import copy
import time
from itertools import product

import numpy as np
import quadpy
import torch
import torch.nn as nn

import src.utils.math_utils as utils
from helper import *

torch.set_default_dtype(torch.float)

# generate the numerical integration points
n_points = 1024
scheme = quadpy.line_segment.clenshaw_curtis(n_points)
ixs = np.argwhere((np.abs(scheme.points) < 1)) # for numerical stability
points = torch.tensor(scheme.points[ixs]).squeeze()
weights = torch.tensor(scheme.weights[ixs]).squeeze()
n_points = points.shape[0]

# set the seed for reproducibility
np.random.seed(609)
torch.manual_seed(609)

# Hard forward KL needs this
max_action = torch.tensor([0.5])

# hyperparameters
anneal_lr = False
learnQ = False
clip = False # whether we are clipping the action or passing it to a tanh
STEPS = 1000
OPTIMIZERS = ["adam", "rmsprop", "sgd"]
OPTIMS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}
LRS = [0.01]
KL = ["reverse", "forward"]
# TAUS = [0, 0.01, 0.1, 0.5, 1]
TAUS = [0.4]
n_samples = 1000 # for the option of sampling points
MUSPACE = (-0.95, 0.95)
LOGSTDSPACE = (np.log(np.exp(0.1) - 1), np.log(np.exp(1.0) - 1))

# choose which pdf function to use based on modes
if clip is False:
    pdf_func = tanh_gauss
else:
    pdf_func = gauss

n_total_experiments = np.prod([len(a) for a in [KL, LRS, OPTIMIZERS, TAUS]])
n_done = 0

n_modes = 1
curr_exp = int(sys.argv[1])

accum = 1
params = []
for ix, l in enumerate([KL, LRS, OPTIMIZERS, TAUS]):
    params.append(l[(curr_exp // accum) % len(l)])
    accum *= len(l)
direction, lr, optim_name, tau = params
print(params)


# initiate the params
mus, logstds = get_random_inits(MUSPACE, LOGSTDSPACE, n_samples, n_modes)
n_inits = n_samples
mus = init_params(mus)
logstds = init_params(logstds)

# for learning the action value function if we want
hidden = 100
Q_approx = batch_Q_approx(hidden, n_samples)
Q_optim = OPTIMS[optim_name](params = Q_approx.parameters(), lr = 1e-4)

# setup
exp_name = "{}_lr={}_optim={}_temp={}_nmodes={}".format(direction, lr, optim_name, tau, n_modes)
print("running {} | {} / {}".format(exp_name, curr_exp + 1, n_total_experiments))
print("clip = {} | learnQ = {} | modes = {} | anneal lr = {}".format(clip, learnQ, n_modes, anneal_lr))

# for recording values
mu_values = []
logstd_values = []
loss_values = []
coeff_values = []
t_start = time.time()

print("total param sets = {}".format(n_inits))
# print(mus, logstds)
QA = Q(points).unsqueeze(-1)
if tau != 0:
    partition = Z(tau, weights, points)
    BQ = torch.exp(QA / tau) / partition
    assert BQ[BQ > 0].numel() == BQ.numel(), BQ
else:
    BQ = torch.zeros_like(points)
    BQ.unsqueeze(-1)
# print(BQ.shape)
# assert BQ.numel() == points.numel()
BQ = BQ.expand((-1, n_inits))
assert BQ.shape == (points.shape[0], n_inits), BQ.shape
optim = OPTIMS[optim_name](params = [mus, logstds], lr = lr)
for step in range(STEPS):
    # decrease step size if desired
    if anneal_lr is True and step > 0:
        lr *= np.sqrt(step / (step + 1))
        # reinitialize LR every time
        optim = OPTIMS[optim_name](params = [mus, logstds], lr = lr)
    # estimate the action value function if desired
    if learnQ is True:
        # if using batch_Q_approx, then shape is (n_inits, points.shape[0]) before permutation
        QA_approx = Q_approx.BQ_forward(points.unsqueeze(-1).float()).squeeze().detach().permute(1, 0)
        # evaluate max action at all the points
        max_action = points[torch.max(QA_approx, dim = 0)[1]]
        if tau != 0:
            # batch approx
            max_Q = QA_approx.max(0)[0]
            partition = torch.sum(weights.unsqueeze(-1) * torch.exp((QA_approx - max_Q) / tau), dim = 0)
            BQ = torch.exp((QA_approx - max_Q) / tau) / partition
            assert BQ.shape == (points.shape[0], n_inits), BQ.shape

            assert torch.isnan(BQ).sum() == 0
        else:
            BQ = torch.zeros((points.shape[0], n_inits)) # have a zero BQ just for plotting purposes
    else:
        QA_approx = QA.expand((-1, n_inits))
        assert QA_approx.shape == (points.shape[0], n_inits), QA_approx.shape
    if (step + 1) % 100 == 0:
        print("step = {}".format(step + 1))
    loss = 0
    
    # calculate the PDFs
    mu_values.append(mus.detach().numpy().copy())
    logstd_values.append(logstds.detach().numpy().copy())
    pi = pdf_func(points.unsqueeze(-1), mus.unsqueeze(0), transform_std_param(logstds).unsqueeze(0))
    assert pi.shape == (points.shape[0], n_inits), print(pi.shape)
    
    # calculate the KL losses
    if direction == "reverse":
        if tau == 0:
            losses = -torch.sum(weights.unsqueeze(-1) * QA_approx.detach() * pi, dim = 0)
        else:
            losses = torch.sum(weights.unsqueeze(-1) * pi * approx_log(pi / (BQ + 1e-5)), dim = 0)
            # print((pi /).shape)
    elif direction == "forward":
        if tau == 0:
            pi = pdf_func(max_action, mus, transform_std_param(logstds))
            # Get the pi corresponding to the max action
            assert pi.numel() == n_inits, pi.shape
            losses = -approx_log(pi)
        else:
            losses = torch.sum(weights.unsqueeze(-1) * BQ * approx_log(BQ / (pi + 1e-5)), dim = 0)
            
    
    assert losses.numel() == n_inits, losses.shape 
    loss_values.append(losses.detach().numpy().copy())
    optim.zero_grad()
    loss = losses.sum()
    assert torch.isnan(loss) == 0
    
    loss.backward()
    optim.step()

    if learnQ is True:
        # draw an action from the current policy
        with torch.no_grad():
            actions = sample_tanh_gauss(mus, transform_std_param(logstds))
        assert actions.numel() == n_inits
        # batch 
        QA_approx = Q_approx(actions.unsqueeze(-1).float()).squeeze()
        assert QA_approx.numel() == actions.numel()
        # assert QA_approx.shape == (actions.shape[0], n_inits), QA_approx.shape
        Q_losses = (Q(actions) - QA_approx).pow(2)
        
        Q_loss = Q_losses.mean()
        assert torch.isnan(Q_loss) == 0
        Q_optim.zero_grad()
        Q_loss.backward()
        Q_optim.step()
t_end = time.time()
print("took {}s".format(t_end - t_start))
n_done += 1

# record the final pdf values
final_pdfs = pdf_func(points.unsqueeze(-1), mus.unsqueeze(0), transform_std_param(logstds).unsqueeze(0))
assert final_pdfs.shape == (points.shape[0], n_inits)


# save values
mu_values = np.array(mu_values)
logstd_values = np.array(logstd_values)
loss_values = np.array(loss_values)
coeff_values = np.array(coeff_values)

foldername = ""
if learnQ is True:
    # should record final learnQ in this case too
    foldername += "learnQ"
else:
    foldername += "notlearnQ"
if clip is True:
    foldername += "_clip"
if anneal_lr is True:
    foldername += "_anneal"
write_dir = "data/continuous_bandit/{}/nmodes={}/".format(foldername, n_modes)
os.makedirs(write_dir, exist_ok=True)
np.save(write_dir + exp_name + "_BQ.npy", BQ.detach().numpy())
np.save(write_dir + exp_name + "_points.npy", points.numpy())
np.save(write_dir + exp_name + "_pdf.npy", final_pdfs.detach().numpy())
np.save(write_dir + exp_name + "_mu.npy", mu_values)
np.save(write_dir + exp_name + "_logstd.npy", logstd_values)
np.save(write_dir + exp_name + "_loss.npy", loss_values)
