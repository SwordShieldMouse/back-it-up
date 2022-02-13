from multiprocessing import Pool
import os
import itertools

stored_dir = "results/_uncompressed_results"
output_plot_dir = "results/_plots/all_high_all_low"

def f(env):
    os.system("python3 plot_scripts/plot_all_high_all_low_uncompressed.py {} --stored_dir {} --output_plot_dir {}".format(env, stored_dir, output_plot_dir))

envs = ["HalfCheetah-v2", "Pendulum-v0", "Swimmer-v2", "Reacher-v2"]

with Pool(8) as pool:
    pool.map(f, envs)