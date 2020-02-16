from linear_agents import *
import envs
import os 
import time
import numpy as np
from functools import partial
from itertools import product
import sys


SEEDS = [609, 8166, 1286, 3403, 398, 404, 2757, 5536, 3535, 5768, 6034, 5703, 1885, 6052, 6434, 3026, 4009, 4212, 2829, 7483, 2267, 2861, 1444, 4950, 1845, 4048, 2521, 9204, 5936, 4626]

LRS = np.power(10, [-1, -1.25, -1.5, -1.75, -2])
ALRS = LRS
EPS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
TEMPS = [0, 0.01, 0.1, 1]
GAMMAS = [0.99]
INTEGRATIONS = [0, 1]
SAC_UPDATE = [0]

HYPERPARAMS = {"sacupdate": SAC_UPDATE, "integration": INTEGRATIONS, "gamma": GAMMAS, "seed": SEEDS, "lr": LRS, "eps": EPS, "softmaxtemp": TEMPS[1:], "softQtemp": TEMPS, "alr": ALRS}

ALGS = [(HardForwardKL, ["lr", "softQtemp", "gamma", "alr", "sacupdate"]), (HardReverseKL, ["sacupdate", "integration", "gamma", "lr", "softQtemp", "alr"]), (ReverseKL, ["sacupdate", "alr", "gamma", "lr", "softQtemp", "softmaxtemp", "integration"]), (ForwardKL, ["integration", "lr", "softQtemp", "softmaxtemp", "gamma", "alr", "sacupdate"])]


index = int(sys.argv[1])
max_frames = int(sys.argv[2])
max_frames_per_ep = int(sys.argv[3])

env_params = {"max_frames": max_frames, "max_frames_per_ep": max_frames_per_ep}

# get total number of experiments
n_total_experiments = 0
# total_experiments = []
for alg, params_to_sweep in ALGS:
    n_combinations = np.prod([len(HYPERPARAMS[param]) for param in params_to_sweep])
    # total_experiments.append(n_combinations)
    n_total_experiments += n_combinations


accum = 1
curr_params = {}
curr_alg_ix = index % len(ALGS)
alg = ALGS[curr_alg_ix][0]
accum *= len(ALGS)
for ix, param in enumerate(ALGS[curr_alg_ix][1]):
    l = HYPERPARAMS[param]
    curr_params[param] = l[(index // accum) % len(l)]
    accum *= len(l)
# base params
agent_params = {}
# append alg-specific params
for param_name, param_value in curr_params.items():
    agent_params[param_name] = param_value
# seed = agent_params["seed"]
# del agent_params["seed"]

for seed in SEEDS:
    env = envs.CartPole(True)
    np.random.seed(seed)
    env.env.seed(seed)
    print("seed = {}".format(seed))

    t_start = time.time()
    agent = alg(env = env, env_params = env_params, agent_params = agent_params)
    print("running experiment {}/{} | alg = {}".format(index + 1, n_total_experiments, agent.name))
    print(alg, curr_params)

    agent.run()
    # data = np.array(agent.plotting_data)
    data = np.array(agent.avg_rewards)

    # create directory if necessary
    if not os.path.exists("data/ql/{}".format(env.name)):
        os.makedirs("data/ql/{}".format(env.name))
    np.save("data/ql/{}/{}_seed={}.npy".format(env.name, agent.name, seed), data)
    t_end = time.time()
    print("total runtime = {}s".format(t_end - t_start))
