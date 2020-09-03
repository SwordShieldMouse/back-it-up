import os 
import sys
sys.path.append(os.getcwd())

from src.agents.kl_agents import *
import src.envs.deep_envs as envs
import src.utils.sweeps as sweeps

import numpy as np
import torch
import time
from matplotlib import pyplot as plt

index = int(sys.argv[1])
max_frames = int(sys.argv[2])
max_frames_per_ep = int(sys.argv[3])

# env = envs.CartPole()
use_ep_length_r = False
# use_ep_length_r = True
# env = envs.LunarLander()
# env = envs.Acrobot()
env = envs.MountainCar()
# env = envs.MinAtar("space_invaders")
print(env.obs_dim, env.action_dim)
SEEDS = [609, 8166, 1286, 3403, 398, 404, 2757, 5536, 3535, 5768, 6034, 5703, 1885, 6052, 6434, 3026, 4009, 4212, 2829, 7483, 2267, 2861, 1444, 4950, 1845, 4048, 2521, 9204, 5936, 4626]
MORE_SEEDS = [53245, 14190, 20633, 47757, 75838, 97760, 93797, 24672, 40598, 98790, 58926, 65551, 36519, 50037, 58932]

## testing
LRS = [1e-5, 1e-4, 1e-3]
ALRS = [1e-5, 1e-4, 1e-3]

EPS_INIT = [1.0]

# regular deep settings
# HIDDEN = [256, 64]
FA = ["deep"]
HIDDEN = [64]

# MINATAR settings
# HIDDEN = [128]
# FA = ["minatar"]

OPTIMS = ["rmsprop"]


SOFTQTEMPS = [0.01, 0.1, 1.]
SOFTMAXTEMPS = [0.1, 0.5, 1]
SAC_UPDATE = [0]
DIRECTION = ["forward", "reverse"]


GAMMAS = [0.99]

HYPERPARAMS = {"gamma": GAMMAS, "hidden": HIDDEN, "seed": SEEDS, "alr": ALRS, "lr": LRS, "softmaxtemp": SOFTMAXTEMPS, "softQtemp": SOFTQTEMPS, "fa": FA, "sacupdate": SAC_UPDATE, "direction": DIRECTION, "optim": OPTIMS}
ALGS = [(HardForwardKL, ["hidden", "lr", "fa", "gamma", "seed", "sacupdate", "optim"]), (HardReverseKL, ["seed", "hidden", "fa", "gamma", "lr", "sacupdate", "optim"]), (ReverseKL, ["hidden", "fa", "gamma", "lr", "sacupdate", "softQtemp", "seed", "optim"]), (ForwardKL, ["lr", "softQtemp", "fa", "sacupdate", "gamma", "hidden", "seed", "optim"])]

# get total number of experiments
n_total_experiments = sweeps.get_n_total_experiments(ALGS, HYPERPARAMS)
alg, agent_params = sweeps.get_instance(ALGS, HYPERPARAMS, index)
print(agent_params)
seed = agent_params["seed"]
del agent_params["seed"]

np.random.seed(seed)
torch.manual_seed(seed)
try:
    env.env.seed(seed)
except:
    print("couldn't set env seed")
env_params = {"max_frames": max_frames, "max_frames_per_ep": max_frames_per_ep}

t_start = time.time()
agent = alg(env = env, env_params = env_params, agent_params = agent_params, use_ep_length_r=use_ep_length_r)
print("running experiment {}/{} | alg = {}".format(index + 1, n_total_experiments, agent.name))

agent.run()
data = np.array(agent.returns)

write_dir = "data/deep_control/{}/".format(env.name)
os.makedirs(write_dir, exist_ok=True)
np.save(write_dir + "{}_seed={}.npy".format(agent.name, seed), data)
t_end = time.time()
print("runtime = {}s".format(t_end - t_start))
