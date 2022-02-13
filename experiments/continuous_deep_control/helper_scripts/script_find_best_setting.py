from multiprocessing import Pool
import os
import itertools

merged_result_loc = "results/_uncompressed_results"
output_plot_dir = "results/_plots/individual_performance"

def f(env, agent, best_setting_type):
    os.system("python3 plot_scripts/find_agent_best_setting.py {} {} --best_setting_type {} --merged_result_loc {} --output_plot_dir {} --num_runs 30".format(env, agent, best_setting_type, merged_result_loc, output_plot_dir))

envs = ["HalfCheetah-v2", "Pendulum-v0", "Swimmer-v2", "Reacher-v2"]
agents = ["ForwardKL", "ReverseKL"]
best_setting_types = ["best", "top20"]

with Pool(12) as pool:
    pool.starmap(f, itertools.product(envs, agents, best_setting_types))