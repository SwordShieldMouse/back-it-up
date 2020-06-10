import os 
import sys 
sys.path.append(os.getcwd())

import numpy as np
import scipy.special
from matplotlib import pyplot as plt

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

def gauss_cdf(x, mu, sigma):
    return 1/2 * (1 + scipy.special.erf((x - mu) / sigma / np.sqrt(2)))

def gauss_int(a, b, mu, sigma):
    return gauss_cdf(b, mu, sigma) - gauss_cdf(a, mu, sigma)

"""
Env spec
- if action is greater than 0, treat as switch; otherwise, treat as stay
"""

def gauss_to_discrete(pi):
    # convert gaussian policy to equivalent discrete policy
    p_stay = gauss_cdf(0, pi[:, 0], pi[:, 1]).reshape((-1, 1))
    pi = np.concatenate((p_stay, 1 - p_stay), axis = 1)
    assert pi.shape == (n_states, n_actions)
    return pi

def get_r_pi(pi):
    # given mu, sigma in states, return r_pi
    p_stay = gauss_cdf(0, pi[:, 0], pi[:, 1])
    r_pi = r[:, 0] * p_stay + r[:, 1] * (1 - p_stay) 
    return r_pi

def get_P_pi(pi):
    P_pi = np.sum(P * np.expand_dims(gauss_to_discrete(pi), axis = -1), axis = 1)
    return P_pi

def get_V(pi, gamma):
    return np.linalg.inv(np.eye(n_states) - gamma * get_P_pi(pi)) @ get_r_pi(pi)


n_samples = 10000
gamma = 0.9
V_values = []
right_range = 10
left_range = -10
for _ in range(n_samples):
    test_pi = np.random.random((n_states, 2)) * (right_range - left_range) + left_range
    test_pi[:, 1] = np.exp(test_pi[:, 1])
    V_values.append(get_V(test_pi, gamma))

V_values = np.array(V_values)
plt.figure(figsize = (13, 8))
plt.scatter(x = V_values[:, 0], y = V_values[:, 1])
write_dir = "figs/switch-stay/continuous/"
os.makedirs(write_dir, exist_ok=True)
plt.savefig(write_dir + "polytope.png")
