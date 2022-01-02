import os 
import sys
sys.path.append(os.getcwd())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import glob
import re
import os
import numpy as np
import copy
import sys
import src.utils.plotting as utils
from itertools import product
import time
from collections import defaultdict

import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%g'))
matplotlib.rcParams.update({'font.size': 65})

experiment = "deep_control"

base_read_dir = "./data/{}/".format(experiment)
ALL_ENV_NAMES = [x for x in os.listdir(base_read_dir)]

# parallel over env_names if needed
ENV_NAMES = ["Acrobot", "CartPole", "LunarLander", "asterix", "breakout", "freeway", "seaquest", "space_invaders"]
env = ENV_NAMES[int(sys.argv[2])]
auc_fraction = float(sys.argv[1])

frames = {"MountainCar": 50000, "Acrobot": 2e5, "CartPole": 2e5, "LunarLander": 5e5, "asterix": 2e6, "breakout": 2e6, "freeway": 2e6, "seaquest": 2e6, "space_invaders": 2e6}
markersize = 35
markersizes = {"forward": 20, "reverse": 35}
markers = {"forward": "o", "reverse": "x"}
mews = {"forward": 3, "reverse": 7}

write_dir = "figs/{}/{}/auc/".format(experiment, env)
read_dir = "data/{}/{}/".format(experiment, env)
os.makedirs(write_dir, exist_ok=True)

print("plotting for env = {} | {} / {}".format(env, int(sys.argv[2]) + 1, len(ENV_NAMES)))
x_params = ["lr", "alr", "softmaxtemp", "softQtemp", "integration", "sacupdate", "nfeatures", "hidden"]

temps = [0, 0.01, 0.1, 1]
top_algs = []
all_data = {}

all_fnames = []

for temp in temps:
    algs_to_include = ["HardReverseKL", "HardForwardKL", "ReverseKL", "ForwardKL"]
    styles = {"HardReverseKL": ("g", "--", (5, 40)), "HardForwardKL": ("b", "--", (5, 40)), "ReverseKL": ("g", "-", (5, 0)), "ForwardKL": ("b", "-", (5, 0))}
    params_to_ignore = {}
    if temp is None:
        params_to_fix = {}
    elif temp == 0:
        params_to_fix = {}
        algs_to_include = ["HardReverseKL", "HardForwardKL"]
        styles = {"HardReverseKL": ("g", "-", (5, 0)), "HardForwardKL": ("b", "-", (5, 0))}
    else:
        params_to_fix = {"softQtemp": temp}
    print("algs: ", algs_to_include)
    print("fixing:", params_to_fix)
    colors = ["r", "g", "b", "c", "m", "k", "y", "#800080", "#00FF00", "#FF4500", "#BDB76B"]
    fnames = [os.path.basename(name) for name in glob.glob(read_dir + "*.npy")]
    # read the data once
    # keys are base name + hyperparams and each value is a list of lists, one for each seed
    data, algs, fnames = utils.get_data(fnames, params_to_fix, algs_to_include, read_dir, params_to_ignore = params_to_ignore, window = 20, n_frames = frames[env])
    all_fnames += fnames
    all_data.update(data)
    t_start = time.time()
    print("top algs...")
    top_algs.append(utils.plot_top_algs(colors=colors, styles = styles, params_to_fix=params_to_fix, auc_fraction=auc_fraction, env = env, write_dir=write_dir, data = data))
    t_end = time.time()
    print("top algs done in {}s".format(t_end - t_start))

    # t_start = time.time()
    # print("sensitivity...")
    # utils.plot_sensitivity(x_params = x_params, colors = colors, styles = styles, fnames = fnames, params_to_fix=params_to_fix, auc_fraction=auc_fraction, env=env, write_dir=write_dir, data = data)
    # t_end = time.time()
    # print("sensitivity done in {}s".format(t_end - t_start))

# make plot of all top FKL and RKL algs, colour-coding by temp
print("combined plot...")
colours = [cm.jet(0.65 + (.99 - 0.65) * ix / len(temps)) for ix in range(len(temps))]
figsize = (18, 12)
fig = plt.figure(figsize = figsize)
n_frames = all_data[top_algs[-1][-1]].shape[-1]
markevery = n_frames // 30
if env == "Acrobot":
    ybotlim = -110
    ytoplim = -80
else:
    ybotlim = None 
    ytoplim = None
for ix in range(len(temps)):
    for alg in top_algs[ix]:
        if "Forward" in alg:
            marker = markers["forward"]
            name = "ForwardKL"
            mew = mews["forward"]
            markersize = markersizes["forward"]
        elif "Reverse" in alg:
            marker = markers["reverse"]
            name = "ReverseKL"
            mew = mews["reverse"]
            markersize = markersizes["reverse"]
        d = all_data[alg]
        # print(d, alg)
        std_error = np.std(d, axis = 0) / np.sqrt(d.shape[0])
        avg_rewards = np.mean(d, axis = 0)
        # print(d.shape)
        plt.plot(avg_rewards, color = colours[ix], label = name, marker = marker, ms = markersize, markevery=markevery, linewidth=0.5, mew = mew)
        # print(name, base_name)
        plt.fill_between(np.arange(d.shape[1]), avg_rewards - std_error, avg_rewards + std_error, alpha = 0.3, color = colours[ix])
plt.xlim(left = 0, right = n_frames)
xticks = np.linspace(0, n_frames // int(1e5), 6)
plt.xticks(ticks = xticks * int(1e5), labels = ["0"] + [np.around(tick, decimals = 1) for tick in xticks[1:]])
plt.xlabel("Hundreds of Thousands of Frames")
plt.ylabel("Average Return")
if ybotlim is not None:
    plt.ylim(bottom = ybotlim)
if ytoplim is not None:
    plt.ylim(top = ytoplim)
plt.subplots_adjust(bottom=0.17, left=0.2)
plt.savefig(write_dir + "UNLABELED_{}_all_kl.png".format(env),dpi=30)

# plt.legend()

legend_elements = [Line2D([0], [0], marker=markers["forward"], color='black', label='Forward KL',
                          markerfacecolor='black', markersize=markersizes["forward"], mew = mews["forward"]), Line2D([0], [0], marker=markers["reverse"], color='black', label='Reverse KL',
                          markerfacecolor='black', markersize=markersizes["reverse"], mew = mews["reverse"])]
plt.legend(handles = legend_elements, frameon=False)
plt.savefig(write_dir + "{}_all_kl.png".format(env),dpi=30)


# combined sensitivity
print("combined sensitivity...")
for x_param in x_params:
    if x_param == "softQtemp": 
        continue
    sensitivities = defaultdict(list)
    t_start = time.time()
    for temp in temps:
        # print(sensitivities)
        if temp == 0:
            base_names = ["HardReverseKL", "HardForwardKL"]
        else:
            base_names = ["ReverseKL", "ForwardKL"]
        for base_name in base_names:
            # get list of all possible x_params for this method, if they exist
            if temp != 0:
                curr_filenames = [name for name in all_fnames if name.startswith(base_name) and utils.find_param_value("softQtemp", name) == temp]
            else:
                curr_filenames = [name for name in all_fnames if name.startswith(base_name)]
            assert len(curr_filenames) != 0, (temp, base_name)
            params = list([utils.find_param_value(x_param, alg) for alg in curr_filenames]) # params actually swept over for this base_name
            # print(curr_filenames, set(params))
            if len(params) <= 1 or None in params:
                print("skipping plot : {}, {}".format(x_param, base_name))
                continue
            params = sorted(list(set(params)))
            # print(params)
            for param in params:
                # get all algs with this param
                # max over params
                d, curr_auc = utils.find_max_config(x_param, param, auc_fraction, curr_filenames, all_data)
                # d = np.stack([np.load(read_dir + alg, allow_pickle=True) for alg in max_config_alg], axis = 0) # max over other hyperparams
                assert curr_auc == np.mean(d), (curr_auc, np.mean(d))
                curr_std = np.std(np.mean(d, axis = -1)) / np.sqrt(d.shape[0])
                sensitivities[base_name.replace("Hard", "")].append([temp, param, curr_auc, curr_std]) # getting standard error over runs
                print("{} runs for {}".format(d.shape[0], base_name))
    if len(sensitivities.keys()) == 0:
        # means no alg had this hyperparam or only one sensitivity point
        print("no sensitivity points for {}; skipping".format(x_param))
        continue
    t_end = time.time()
    print("getting aucs took {}s".format(t_end - t_start))
    
    t_start = time.time()
    plt.figure(figsize = figsize)
    # print(sensitivities, x_param)
    params = []
    for base_name in sorted(sensitivities.keys()):
        if "Forward" in alg:
            marker = markers["forward"]
            name = "ForwardKL"
            mew = mews["forward"]
            markersize = markersizes["forward"]
        elif "Reverse" in alg:
            marker = markers["reverse"]
            name = "ReverseKL"
            mew = mews["reverse"]
            markersize = markersizes["reverse"]
        for ix in range(len(temps)):
            s = copy.deepcopy(np.array(sensitivities[base_name]))
            # print(base_name, s)
            temp_rows = np.where(s[:, 0] == temps[ix])
            # print(temp_rows)
            s = s[temp_rows[0], :]
            # print(s.shape, s)
            
            # assert len(s.shape) == 3, (s.shape, s)
            x = s[:, 1]
            if "lr" in x_param:
                x = np.log10(x)
            y = s[:, 2]
            params += list(x)
            std_error = s[:, 3]
            plt.plot(x, y, label = base_name, color = colours[ix], marker = marker, ms = markersize, mew = mew)
            plt.errorbar(x, y, yerr = std_error, color = colours[ix])
    params = sorted(list(set(params)))
    plt.subplots_adjust(bottom=0.17, left=0.2)
    if "lr" in x_param:
        plt.xticks(params)
        plt.xlabel(r"$\log_{10}(lr)$")
    else:
        plt.xticks(params)
        plt.xlabel(x_param)
    # if len(params) > 1:
    #     plt.xlim(left = np.min(params), right = np.max(params))
    plt.ylabel("Average {}-AUC".format(auc_fraction))
    plt_name = "kl_auc-{}_{}_{}_sensitivity.png".format(auc_fraction, env, x_param)
    fig_dir = write_dir
    os.makedirs(fig_dir, exist_ok = True)
    plt.savefig(fig_dir + "UNLABELED_" + plt_name)

    # plt.title("{} Sensitivity on {}".format(x_param, env))
    
    legend_elements = [Line2D([0], [0], marker=markers["forward"], color='black', label='Forward KL',
                            markerfacecolor='black', markersize=markersizes["forward"], mew = mews["forward"]), Line2D([0], [0], marker=markers["reverse"], color='black', label='Reverse KL',
                            markerfacecolor='black', markersize=markersizes["reverse"], mew = mews["reverse"])]
    plt.legend(handles = legend_elements, frameon=False)

    plt.savefig(fig_dir + plt_name)
    plt.close()
    t_end = time.time()
    print("plotting {} took {}s".format(x_param, t_end - t_start))