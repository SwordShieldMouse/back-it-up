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

# generate the numerical integration points for pdf
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
n_action_points = 500 # to be able to do WIS with FKL
OPTIMIZERS = ["adam", "rmsprop", "sgd"]
OPTIMS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop}
LRS = [0.005]
# KL = ["reverse", "forward"]
KL = ["reverse"]
# TAUS = [0, 0.01, 0.1, 0.4, 1]
TAUS = [0.01, 0.1, 0.4, 1]
n_samples = 1000 # for the option of sampling points
MUSPACE = (-0.95, 0.95)
LOGSTDSPACE = (np.log(np.exp(0.1) - 1), np.log(np.exp(1.0) - 1))

# choose which pdf function to use based on modes
n_modes = 1 # number of modes in the policy
if clip is False:
    pdf_func = tanh_gauss
else:
    pdf_func = gauss

n_total_experiments = np.prod([len(a) for a in [KL, LRS, OPTIMIZERS, TAUS]])
n_done = 0

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

# calculate BQ to compare the final PDFs
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

print("total param sets = {}".format(n_inits))
# print(mus, logstds)
optim = OPTIMS[optim_name](params = [mus, logstds], lr = lr)
t0 = time.time()
for step in range(STEPS):
    if (step + 1) % 100 == 0:
        print("step = {} | fps = {}".format(step + 1, step / (time.time() - t0)))
    loss = 0
    
    # sample actions by first sampling gaussian
    m = torch.distributions.Normal(loc = mus, scale = transform_std_param(logstds))
    actions = m.sample_n(n_action_points)
    assert actions.shape == (n_action_points, mus.shape[0])
    orig = m.log_prob(actions)
    correction = torch.log(1 - torch.tanh(actions).pow(2) + 1e-5)
    assert torch.isnan(orig).sum() == 0
    assert torch.isnan(correction).sum() == 0
    log_pdfs = orig - correction
    QA = Q(torch.tanh(actions))
    assert QA.shape == actions.shape
    # assert actions.shape == mus.shape
    mu_values.append(mus.detach().numpy().copy())
    logstd_values.append(logstds.detach().numpy().copy())

    # calculate the KL losses
    if direction == "reverse":
        if tau == 0:
            losses = -torch.mean(QA * log_pdfs, axis = 0)
        else:
            losses = torch.mean(log_pdfs + log_pdfs.detach() * log_pdfs - QA * log_pdfs / tau, axis = 0)
            # print((pi /).shape)
    elif direction == "forward":
        if tau == 0:
            pi = pdf_func(max_action, mus, transform_std_param(logstds))
            # Get the pi corresponding to the max action
            assert pi.numel() == n_inits, pi.shape
            losses = -approx_log(pi)
        else:
            # do weighted importance sampling
            with torch.no_grad():
                max_arg = torch.max(QA / tau - log_pdfs, axis = 0, keepdim = True)[0]
                rho = torch.exp(QA / tau - log_pdfs - max_arg)
            # print(rho.shape, QA.shape, log_pdfs.shape)
            WIR = rho / rho.sum(0, keepdim=True)
            losses = - torch.sum(WIR * log_pdfs, axis = 0)
            # print(losses.max())
            
    
    assert losses.numel() == n_inits, losses.shape 
    loss_values.append(losses.detach().numpy().copy())
    optim.zero_grad()
    loss = losses.sum()
    assert torch.isnan(loss) == 0
    
    loss.backward()
    optim.step()

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
write_dir = "data/continuous_bandit/sample{}_{}/nmodes={}/".format(n_action_points, foldername, n_modes)
os.makedirs(write_dir, exist_ok=True)
np.save(write_dir + exp_name + "_BQ.npy", BQ.detach().numpy())
np.save(write_dir + exp_name + "_points.npy", points.numpy())
np.save(write_dir + exp_name + "_pdf.npy", final_pdfs.detach().numpy())
np.save(write_dir + exp_name + "_mu.npy", mu_values)
np.save(write_dir + exp_name + "_logstd.npy", logstd_values)
np.save(write_dir + exp_name + "_loss.npy", loss_values)