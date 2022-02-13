from multiprocessing import Pool
import os
import itertools

store_dir = "results/_uncompressed_results"
output_plot_dir = "results/sensitivity"

def f(env,agent):
    os.system("python3 plot_scripts/plot_sensitivity_uncompressed.py {} {} --store_dir {} --output_plot_dir {} --num_runs 30".format(env,agent,store_dir,output_plot_dir))

envs = ["HalfCheetah-v2", "Pendulum-v0", "Swimmer-v2", "Reacher-v2"]
agents = ["ForwardKL","ReverseKL"]

with Pool(8) as pool:
    pool.starmap(f, itertools.product(envs, agents))