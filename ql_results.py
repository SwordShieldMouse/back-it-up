import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
import re
import os
import numpy as np
import sys
import utils
from itertools import product
import time

auc_fraction = float(sys.argv[1])
env = sys.argv[2]

write_dir = "figs/ql/{}/auc/".format(env)
read_dir = "data/ql/{}/".format(env)
os.makedirs(write_dir, exist_ok=True)

softmaxtemps = [0.001, 0.01, 0.1, 1]
softQtemps = [0, 0.001, 0.01, 0.1, 1]

algs_to_include = ["HardReverseKL", "HardForwardKL", "ReverseKL", "ForwardKL"]
params_to_fix = {"sacupdate": 0, "learnQ": 1, "noisyQ": 0, "allstates": 0}
print("algs: ", algs_to_include)
print("fixing:", params_to_fix)
colors = ["r", "g", "b", "c", "m", "k", "y", "#800080", "#00FF00", "#FF4500", "#BDB76B"]
styles = {"HardReverseKL": ("g", "--", (5, 20)), "HardForwardKL": ("b", "--", (5, 20)), "ReverseKL": ("g", "-", (5, 0)), "ForwardKL": ("b", "-", (5, 0))}
names = [os.path.basename(name) for name in glob.glob(read_dir + "*.npy")]

t_start = time.time()
print("top algs...")
utils.plot_top_algs(colors=colors, styles = styles, names=names, params_to_fix=params_to_fix, auc_fraction=auc_fraction, env = env, read_dir=read_dir, write_dir=write_dir, algs_to_include=algs_to_include)
t_end = time.time()
print("top algs done in {}s".format(t_end - t_start))

t_start = time.time()
print("sensitivity...")
utils.plot_sensitivity(x_param="lr", colors = colors, styles = styles, names = names, params_to_fix=params_to_fix, auc_fraction=auc_fraction, env=env, read_dir=read_dir, write_dir=write_dir, algs_to_include=algs_to_include)
t_end = time.time()
print("sensitivity done in {}s".format(t_end - t_start))