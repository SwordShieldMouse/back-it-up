import os
import sys
sys.path.append(os.getcwd())

import torch 
import torch.nn as nn
import numpy as np
from itertools import product
import time
import copy
from helper import *


torch.set_default_dtype(torch.float)

np.random.seed(609)
torch.manual_seed(609)


# bandit stuff
Q = torch.tensor([1.5, 2., 2.])
n_actions = Q.numel()

# hyperparameters
learnQ = False
STEPS = 1000
OPTIMIZERS = ["adam", "sgd", "rmsprop"]
OPTIMS = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop, "sgd": torch.optim.SGD}
LRS = [1e-2, 5e-3]
KL = ["reverse", "forward"]
TAUS = [0, 0.01, 0.05, 0.1, 1]


n_samples = 1000 # for the option of sampling points
LOGITSPACE = (-1, 1)

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


# reset the init
logits = get_random_inits(LOGITSPACE, n_samples * n_actions)
n_inits = n_samples
logits = init_params(logits.reshape((n_actions, n_samples)))
# print(logits)

hidden = 100
Q_approx = batch_Q_approx(hidden, n_samples)
Q_optim = OPTIMS[optim_name](params = Q_approx.parameters(), lr = 1e-4)

# setup
exp_name = "{}_lr={}_optim={}_temp={}".format(direction, lr, optim_name, tau)
print("running {} | {} / {}".format(exp_name, curr_exp + 1, n_total_experiments))
print("learnQ = {}".format(learnQ))
optim = OPTIMS[optim_name](params = [logits], lr = lr)

# for recording values
prob_values = []
loss_values = []
t_start = time.time()

print("total param sets = {}".format(n_inits))

if tau != 0:
    normed_arg = Q / tau - torch.max(Q / tau)
    BQ = torch.exp(normed_arg) / torch.sum(torch.exp(normed_arg))
else:
    BQ = torch.tensor([0, 1/2, 1/2])
BQ.unsqueeze(-1)
Q.unsqueeze(-1)
QA = Q
# max_action = 3
max_action = torch.tensor([n_actions - 1] * n_inits)
for step in range(STEPS):
    if learnQ is True:
        # if using batch_Q_approx, then shape is (n_inits, n_actions) before permutation
        QA = Q_approx.BQ_forward().squeeze().detach().permute(1, 0)
        # evaluate max action at all the points
        max_action = torch.max(QA, dim = 0)[1]
        if tau != 0:
            # batch approx
            max_Q = QA.max(0, keepdim=True)[0]
            partition = torch.sum(torch.exp((QA - max_Q) / tau), dim = 0)
            BQ = torch.exp((QA - max_Q) / tau) / partition
            assert BQ.shape == (n_actions, n_inits), BQ.shape

            assert torch.isnan(BQ).sum() == 0
    if (step + 1) % 50 == 0:
        print("step = {}".format(step + 1))
    loss = 0

    # get a batch pdf with all the actions
    max_logits = torch.max(logits, dim = 0)[0]
    pi = torch.exp(logits - max_logits) / torch.sum(torch.exp(logits - max_logits), dim = 0, keepdim=True)
    prob_values.append(pi.detach().numpy().copy())
    assert pi.shape == (n_actions, n_inits), print(pi.shape)

    # add N(0, 1) noise to QA for gaussianness
    QA += torch.normal(mean = torch.zeros_like(QA), std = torch.ones_like(QA))

    # compute losses
    if direction == "reverse":
        if tau == 0:
            losses = -torch.sum(QA * pi, dim = 0)
        else:
            losses = torch.sum(pi * approx_log(pi / (BQ + 1e-5)), dim = 0)
    elif direction == "forward":
        if tau == 0:
            # print(max_action.unsqueeze(0).shape, pi.shape)
            max_pis = torch.gather(pi, 0, max_action.unsqueeze(0))
            assert max_pis.numel() == n_inits, max_pis.shape
            losses = -approx_log(max_pis)
        else:
            losses = torch.sum(BQ * approx_log(BQ / (pi + 1e-5)), dim = 0)
            
    # assert losses.shape[0] == n_inits, losses.shape
    assert losses.numel() == n_inits, losses.shape 
    loss_values.append(losses.squeeze().detach().numpy().copy())
    optim.zero_grad()
    loss = losses.sum()
    assert torch.isnan(loss) == 0, losses[losses != losses].numel()
    # print(losses.mean())
    # if (losses.mean() == 0):
    # print(losses.mean())
    loss.backward()
    optim.step()

    if learnQ is True:
        # draw an action from the current policy
        # is it reasonable to use just one critic network? maybe batch over all the initializations at first?
        with torch.no_grad():
            actions = sample_boltzmann(pi)
        assert actions.numel() == n_inits

        # batch 
        QA = Q_approx.BQ_forward().squeeze().permute(1, 0)
        # QA now of shape (n_actions, n_inits)
        QA = torch.gather(QA, 0, actions.unsqueeze(0))
        assert QA.numel() == actions.numel()
        # assert QA_approx.shape == (actions.shape[0], n_inits), QA_approx.shape
        Q_losses = (Q[actions, :].squeeze() - QA.squeeze()).pow(2)
        assert Q_losses.numel() == n_inits, (QA.shape, Q[actions].shape)
        Q_loss = Q_losses.mean()
        assert torch.isnan(Q_loss) == 0
        Q_optim.zero_grad()
        Q_loss.backward()
        Q_optim.step()
t_end = time.time()
print("took {}s".format(t_end - t_start))
n_done += 1
max_logits = torch.max(logits, dim = 0)[0]
final_pdfs = torch.exp(logits - max_logits) / torch.sum(torch.exp(logits - max_logits), dim = 0, keepdim=True)
assert final_pdfs.shape == (n_actions, n_inits)


# save values
logit_values = np.array(prob_values)
loss_values = np.array(loss_values)
# print(mu_values[0, :], mu_values[-1, :])
foldername = ""
if learnQ is True:
    # should record final learnQ in this case too
    foldername += "learnQ"
else:
    foldername += "notlearnQ"
write_dir = "data/discrete_bandit/{}/".format(foldername)
os.makedirs(write_dir, exist_ok=True)
np.save(write_dir + exp_name + "_BQ.npy", BQ.squeeze().detach().numpy())
np.save(write_dir + exp_name + "_pdf.npy", final_pdfs.detach().numpy())
np.save(write_dir + exp_name + "_prob.npy", prob_values)
np.save(write_dir + exp_name + "_loss.npy", loss_values)