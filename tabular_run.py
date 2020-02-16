from tabular_agents import *
import envs
import os
import time
from functools import partial
from itertools import product
import sys


SEEDS = [609, 8166, 1286, 3403, 398, 404, 2757, 5536, 3535, 5768, 6034, 5703, 1885, 6052, 6434, 3026, 4009, 4212, 2829, 7483, 2267, 2861, 1444, 4950, 1845, 4048, 2521, 9204, 5936, 4626]

LRS = (np.arange(10) + 1) / 10
ALRS = LRS
EPS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
TEMPS = [0, 0.01, 0.1, 1]
EPS_FINAL = [0.01]
EPS_ZERO_BY = [125, 250, 500, 1000]
GAMMAS = [0.99]
INTEGRATION = [0, 1]
SAC_UPDATE = [0]

HYPERPARAMS = {"sacupdate": SAC_UPDATE, "integration": INTEGRATION, "gamma": GAMMAS, "epszeroby": EPS_ZERO_BY, "epsfinal": EPS_FINAL, "epsinit": EPS, "alr": ALRS, "lr": LRS, "eps": EPS, "softmaxtemp": TEMPS[1:], "softQtemp": TEMPS}
ALGS = [(HardForwardKL, ["alr","lr", "softQtemp", "gamma", "sacupdate"]), (HardReverseKL, ["sacupdate", "integration", "gamma", "alr", "lr", "softQtemp"]), (ReverseKL, ["sacupdate", "integration", "gamma", "alr", "lr", "softQtemp", "softmaxtemp"]), (ForwardKL, ["sacupdate", "integration", "alr", "lr", "softQtemp", "softmaxtemp", "gamma"])]

index = int(sys.argv[1])
seed = SEEDS[index]
max_frames = int(sys.argv[2])
max_frames_per_ep = int(sys.argv[3])
env_params = {"max_frames": max_frames, "max_frames_per_ep": max_frames_per_ep}

# Get total number of experiments for this seed
n_total_experiments = 0
for alg, params_to_sweep in ALGS:
    # do all the params for one alg
    n_total_experiments += np.prod([len(HYPERPARAMS[param]) for param in params_to_sweep])
print("{} total experiments for seed = {}".format(n_total_experiments, seed))

total_runtime = 0
# for seed in SEEDS:
for alg, params_to_sweep in ALGS:
    # do all the params for one alg
    n_combinations = np.prod([len(HYPERPARAMS[param]) for param in params_to_sweep])
    print("{} combinations to run for {}".format(n_combinations, alg))
    for index in range(n_combinations):
        accum = 1
        curr_params = {}
        for ix, param in enumerate(params_to_sweep):
            l = HYPERPARAMS[param]
            curr_params[param] = l[(index // accum) % len(l)]
            accum *= len(l)
        # base params
        agent_params = {"noisyQ": 0, "learnQ": 1, "allstates": 0}
        # append alg-specific params
        for param_name, param_value in curr_params.items():
            agent_params[param_name] = param_value
        np.random.seed(seed)

        env = envs.FiveWorld(init_policy=False)

        agent = alg(env = env, agent_params = agent_params, env_params = env_params)
        print("running {}".format(agent.name))
        t_start = time.time()
        agent.run()

        data = np.array(agent.avg_rewards)

        # create directory if necessary
        if not os.path.exists("data/ql/{}".format(env.name)):
            os.makedirs("data/ql/{}".format(env.name), exist_ok=True)
        if not os.path.exists("data/ql/{}/probs".format(env.name)):
            os.makedirs("data/ql/{}/probs".format(env.name), exist_ok=True)
        if not os.path.exists("data/ql/{}/entropies".format(env.name)):
            os.makedirs("data/ql/{}/entropies".format(env.name), exist_ok=True)
        np.save("data/ql/{}/{}_seed={}.npy".format(env.name, agent.name, seed), data)
        # save policy probabilities
        # np.save("data/ql/{}/probs/{}_seed={}.npy".format(env.name, agent.name, seed), np.array(agent.all_probs))
        # save entropies
        # np.save("data/ql/{}/entropies/{}_seed={}.npy".format(env.name, agent.name, seed), np.array(agent.entropies))
        t_end = time.time()
        print("runtime = {}s".format(t_end - t_start))
        total_runtime += t_end - t_start

print("total runtime = {}s".format(total_runtime))