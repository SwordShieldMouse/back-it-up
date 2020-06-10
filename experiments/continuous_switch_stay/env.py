import os 
import sys 
sys.path.append(os.getcwd())

import numpy as np 
import quadpy 
import torch

import src.utils.tab_rl as tab_rl

# numerical integration points
n_points = 1024
scheme = quadpy.line_segment.clenshaw_curtis(n_points)
ixs = np.argwhere((np.abs(scheme.points) < 1)) # for numerical stability
points = torch.tensor(scheme.points[ixs]).squeeze()
weights = torch.tensor(scheme.weights[ixs]).squeeze()
n_points = points.shape[0]


n_states = 2
n_actions = 2

P = np.zeros((n_states, n_actions, n_states))
r = np.zeros((n_states, n_actions))
cont_r = np.zeros((n_states, n_points)) # "continuous version" of r
cont_P = np.zeros((n_states, n_points, n_states)) # "continuous version" of P
"""
Env spec
- if action is greater than 0, treat as switch; otherwise, treat as stay
"""
# action 0 is stay, action 1 is switch
cont_r[0, points.numpy() <= 0] = 1
cont_r[0, points.numpy() > 0] = -1
cont_r[1, points.numpy() <= 0] = 2

cont_P[0, points.numpy() <= 0, 0] = 1
cont_P[0, points.numpy() > 0, 1] = 1
cont_P[1, points.numpy() <= 0, 1] = 1
cont_P[1, points.numpy() > 0, 0] = 1

r[0, 0] = 1
r[0, 1] = -1
r[1, 0] = 2
r[1, 1] = 0

P[0, 0, 0] = 1
P[0, 1, 1] = 1
P[1, 0, 1] = 1
P[1, 1, 0] = 1

def generate_boundary(gamma):
    n_points = 200
    semi_det_pis = [np.array([[1, 0], [x , 1 - x]]) for x in np.linspace(0, 1, n_points)] + [np.array([[0, 1], [x , 1 - x]]) for x in np.linspace(0, 1, n_points)] + [np.flip(np.array([[1, 0], [x , 1 - x]]), 0) for x in np.linspace(0, 1, n_points)] + [np.flip(np.array([[0, 1], [x , 1 - x]]), 0) for x in np.linspace(0, 1, n_points)]
    polytope_boundary = np.array([tab_rl.get_V(P, r, gamma, pi) for pi in semi_det_pis])
    return polytope_boundary