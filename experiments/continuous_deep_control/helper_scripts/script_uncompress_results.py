from multiprocessing import Pool
import os

results_dir = "results/_results"

def f(env):
    os.system("python3 plot_scripts/uncompress_results.py {} --input_dir {}".format(env, results_dir))

envs = ["HalfCheetah-v2", "Pendulum-v0", "Swimmer-v2", "Reacher-v2"]

with Pool(4) as pool:
    pool.map(f, envs)