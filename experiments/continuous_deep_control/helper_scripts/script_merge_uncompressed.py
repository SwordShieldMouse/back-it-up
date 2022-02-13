from multiprocessing import Pool
import os
import itertools

results_dir = "results/_uncompressed_results"

def f(env, agent):
    os.system("python3 plot_scripts/merge_results_uncompressed.py {} {} --results_dir {} --num_runs 30".format(env, agent, results_dir))

envs = ["HalfCheetah-v2", "Pendulum-v0", "Swimmer-v2", "Reacher-v2"]
agents = ["ForwardKL", "ReverseKL"]

with Pool(4) as pool:
    pool.starmap(f, itertools.product(envs, agents))