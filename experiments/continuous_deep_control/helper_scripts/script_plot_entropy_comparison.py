from multiprocessing import Pool
import os
import itertools

stored_dir = "results/_uncompressed_results"
output_plot_dir = "results/_plots/entropy_comparison"

def f(env,agent):
    os.system("python3 plot_scripts/plot_entropy_comparison_uncompressed.py {} --agent {} --stored_dir {} --output_plot_dir {}".format(env,agent, stored_dir, output_plot_dir))

envs = ["HalfCheetah-v2", "Pendulum-v0", "Swimmer-v2", "Reacher-v2"]
agents = ["ForwardKL","ReverseKL","both"]

with Pool(8) as pool:
    pool.starmap(f, itertools.product(envs, agents))