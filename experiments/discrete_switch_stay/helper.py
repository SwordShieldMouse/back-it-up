import os 
import sys
sys.path.append(os.getcwd())

import numpy as np 
import torch

n_states = 2
n_actions = 2

P = np.zeros((n_states, n_actions, n_states))
r = np.zeros((n_states, n_actions))

# action 0 is stay, action 1 is switch
r[0, 0] = 1
r[0, 1] = -1
r[1, 0] = 2
r[1, 1] = 0

P[0, 0, 0] = 1
P[0, 1, 1] = 1
P[1, 0, 1] = 1
P[1, 1, 0] = 1

def approx_log(x):
    return torch.log(x + 1e-5)

def get_V(pi, gamma):
    # assert np.sum(pi.sum(1) - 1.) == 0, np.sum(pi.sum(1) - 1.)
    P_pi = np.sum(P * np.expand_dims(pi, axis = -1), axis = 1)
    r_pi = np.sum(r * pi, axis = -1)
    return np.linalg.inv(np.eye(2) - gamma * P_pi) @ r_pi 

def get_batch_V(pis, gamma):
    P_pis = np.stack([np.sum(P * np.expand_dims(pis[:, :, i], axis = -1), axis = 1) for i in range(pis.shape[2])])
    r_pis = np.stack([np.sum(r * pis[:, :, i], axis = -1) for i in range(pis.shape[2])])
    assert P_pis.shape == (pis.shape[2], n_states, n_states)
    assert r_pis.shape == (pis.shape[2], n_states)
    inv_arg = np.eye(n_states)[np.newaxis, :, :] - gamma * P_pis
    inv = np.linalg.inv(inv_arg)
    return np.sum(inv * r_pis[:, np.newaxis, :], axis = 2)

def get_batch_soft_V(pis, gamma, tau):
    P_pis = np.stack([np.sum(P * np.expand_dims(pis[:, :, i], axis = -1), axis = 1) for i in range(pis.shape[2])])
    r_pis = np.stack([np.sum(r * pis[:, :, i], axis = -1) for i in range(pis.shape[2])]) - tau * np.sum(pis * np.log(pis + 1e-5), axis = 1).T
    assert P_pis.shape == (pis.shape[2], n_states, n_states)
    assert r_pis.shape == (pis.shape[2], n_states)
    inv_arg = np.eye(n_states)[np.newaxis, :, :] - gamma * P_pis
    inv = np.linalg.inv(inv_arg)
    return np.sum(inv * r_pis[:, np.newaxis, :], axis = 2)


def get_batch_Q(pis, gamma):
    Vp = get_batch_V(pis, gamma)[:, np.newaxis, np.newaxis, :]
    assert Vp.shape == (pis.shape[2], 1, 1, n_states)
    EVp = np.sum(P[np.newaxis, :, :, :] * Vp, axis = -1)
    return r[np.newaxis, :, :] + gamma * EVp

def get_batch_soft_Q(pis, gamma, tau):
    Vp = get_batch_soft_V(pis, gamma, tau)[:, np.newaxis, np.newaxis, :]
    assert Vp.shape == (pis.shape[2], 1, 1, n_states)
    EVp = np.sum(P[np.newaxis, :, :, :] * Vp, axis = -1)
    return r[np.newaxis, :, :] + gamma * EVp


def get_soft_V(pi, gamma, tau):
    P_pi = np.sum(P * np.expand_dims(pi, axis = -1), axis = 1)
    r_pi = np.sum(r * pi, axis = -1) - tau * np.sum(pi * np.log(pi + 1e-5), axis = 1)
    return np.linalg.inv(np.eye(2) - gamma * P_pi) @ r_pi 


def get_Q(pi, gamma):
    V = get_V(pi, gamma)
    Vp =  np.sum(P * V.reshape((1, 1, n_states)), axis = -1)
    return r + gamma * Vp

def get_soft_Q(pi, gamma, tau):
    V = get_soft_V(pi, gamma, tau)
    Vp =  np.sum(P * V.reshape((1, 1, n_states)), axis = -1)
    return r + gamma * Vp
    

# checks on the transition matrix
for s in range(n_states):
    for a in range(n_actions):
        assert np.sum(P[s, a, :]) == 1

# checks on the value function
stay_pi = np.array([[1, 0], [1, 0]])
stay_V =  get_V(stay_pi, 0.99)
assert stay_V[0] == r[0, 0] / (1 - 0.99), stay_V
assert stay_V[1] == r[1, 0] / (1 - 0.99), stay_V

print("checks passed")
