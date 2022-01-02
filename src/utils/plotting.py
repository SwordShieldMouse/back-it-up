import torch
import re
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy as sp
from collections import defaultdict
import time

def find_param_value(param, s):
    # find the param value corresponding to param in s
    res = re.search("_({}=([^_]*))".format(param), s)
    if res == None:
        return None
    assert param in res[1] # just to check that we have the right param
    try: # float param
        return float(res[2])
    except ValueError: # non-float param
        # print(res[2])
        return res[2]

def strip_seed(s):
    return re.search('^(.*)_(seed=(\d)*)', s)[1]

def filter_alg_names(algs, params_to_fix, algs_to_include, params_to_ignore):
    # can take name with or without seed
    algs2 = []
    for alg in algs:
        if not any([alg.startswith(base_name + "_") for base_name in algs_to_include]):
            continue
        append_or_not = True
        for param in params_to_fix:
            if param not in alg:
                append_or_not = False 
                break 
            if params_to_fix[param] != find_param_value(param, alg):
                append_or_not = False
                break
        for param in params_to_ignore:
            if params_to_ignore[param] == find_param_value(param, alg):
                append_or_not = False
                break
        if append_or_not is True:
            algs2.append(alg)
    return algs2

def get_base_name(alg):
    # given basename_hyperparams, returns basename
    return re.search('^(\w)*_', alg)[0][:-1]

def get_moving_avg(a, window):
    # given array of returns, get moving window average
    return np.convolve(a, np.ones((window, )) / window, mode = "full")

def uncompress_returns(a, window, n_frames):
    # assume data in format [(G_1, length_1), (G_2, length_2)....]
    # want to return moving average
    a = np.array(a)
    # n_frames = int(np.sum(a, axis = 0)[1])
    res = np.zeros(n_frames)
    ep = 0
    running_frame = 0
    for i in range(n_frames):
        if running_frame > a[ep, 1]:
            ep += 1
            running_frame = 0
        res[i] = np.mean(a[ep : ep + window, 0])
        running_frame += 1
    return res

def get_data(fnames, params_to_fix, algs_to_include, read_dir, params_to_ignore, window, n_frames):
    t_start = time.time()
    filtered_fnames = filter_alg_names(fnames, params_to_fix, algs_to_include, params_to_ignore)
    algs = set([re.search('^(.*)_(seed=(\d)*)', name)[1] for name in filtered_fnames])
    algs = filter_alg_names(algs, params_to_fix, algs_to_include, params_to_ignore)
    assert algs != [], "empty algs"
    assert filtered_fnames != [], "empty names"
    t_end = time.time()
    print("filtering took {}s".format(t_end - t_start))
    # read the data once and for all
    n_frames = np.load(read_dir + filtered_fnames[0], allow_pickle=True).shape[0]
    print("plotting data of length {}".format(n_frames))
    data = defaultdict(list)
    t_start = time.time()
    n_entries = 0
    for name in filtered_fnames:
        alg = re.search('^(.*)_(seed=(\d)*)', name)[1] # extract alg as key
        try:
            # data[alg].append(np.load(read_dir + name))#, allow_pickle=True))
            data[alg].append(uncompress_returns(np.load(read_dir + name), window, int(n_frames)))
            if len(data[alg][-1]) != n_frames:
                print("inconsistent input length of {}; removing...".format(len(data[alg][-1])))
                data[alg].pop()
            n_entries += 1
        except ValueError:
            print("problems in loading {}".format(name))
    t_end = time.time()
    
    # stack np arrays 
    for key in data:
        data[key] = np.stack(data[key], axis = 0)

    print("reading arrays took {}s".format(t_end - t_start))
    print("{} runs in total".format(n_entries))
    return data, algs, filtered_fnames

def plot_top_algs(colors, styles, data, params_to_fix, auc_fraction, env, write_dir, y_axis_label = "Average Return", x_axis_label = "Hundreds of Thousands of Frames", show_legend=True, suppress_title = False, yticks = None, ytoplim = None, ybotlim = None, figsize = (15, 10)):
    # fixed_params = "_".join(["{}={}".format(param, params_to_fix[param]) for param in params_to_fix])
    # os.makedirs(write_dir + fixed_params + "/", exist_ok=True)

    t_start = time.time()
    aucs = {key : np.mean(data[key][:, int((1 - auc_fraction) * data[key].shape[1]) : -1]) for key in data}
    sorted_aucs = sorted(aucs.items(), key = lambda item: item[1], reverse = True)
    t_end = time.time()
    print("getting sorted aucs took {}s".format(t_end - t_start))

    # plot the top n of each method
    # method is the name of an alg w/o hyperparameters
    t_start = time.time()
    base_names = sorted(set([get_base_name(key) for key in aucs.keys()]))
    n = 1
    plot_names = []
    for base_name in base_names:
        base_name_dict = [(alg, base_name) for alg, _ in sorted_aucs if alg.startswith(base_name + "_")]
        plot_names.extend(base_name_dict[:n])
    fig = plt.figure(figsize = figsize)
    for alg, base_name in plot_names:
        # print(name)
        d = data[alg]
        std_error = np.std(d, axis = 0) / np.sqrt(d.shape[0])
        print("{} runs for {}".format(d.shape[0], base_name))
        avg_rewards = np.mean(d, axis = 0)
        plt.plot(avg_rewards, color = styles[base_name][0], linestyle = styles[base_name][1], label = base_name, dashes = styles[base_name][2])
        plt.fill_between(np.arange(d.shape[1]), avg_rewards - std_error, avg_rewards + std_error, alpha = 0.3, color = styles[base_name][0])
    n_frames = d.shape[1]
    plt.subplots_adjust(bottom=0.17, left=0.2)
    xticks = np.linspace(0, n_frames // int(1e5), 6)
    plt.xticks(ticks = xticks * int(1e5), labels = ["0"] + [np.around(tick, decimals = 1) for tick in xticks[1:]])
    plt.xlim(left = 0, right = d.shape[1])
    if ytoplim is not None:
        plt.ylim(top = ytoplim)
    if ybotlim is not None:
        plt.ylim(bottom = ybotlim)
    # plot labeled and unlabeled
    combined_base_names = '_'.join(base_names)
    if params_to_fix:
        combined_param_names = "_".join(["{}={}".format(param, params_to_fix[param]) for param in params_to_fix])
    else:
        combined_param_names = ""
    plot_name = "auc{}-{}_compare_{}_UNLABELED.png".format(auc_fraction, env, combined_param_names)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    fig_dir = write_dir + combined_base_names + "/"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(fig_dir + plot_name, dpi=30)

    if show_legend is True:
        plt.legend(frameon=False)
    if yticks is not None:
        plt.yticks(yticks)
    if suppress_title is False:
        plt.title("Top Algs by {}-AUC on {} {}".format(auc_fraction, env, combined_param_names))
    # print(plot_names)
    plot_name = "auc{}-{}_compare_{}.png".format(auc_fraction, env, combined_param_names)
    plt.savefig(fig_dir + plot_name, dpi=30)
    plt.close()
    t_end = time.time()
    print("plotting took {}s".format(t_end - t_start))

    # return names of top algs
    return [key for key, _ in plot_names]


def find_max_config(x_param, param, auc_fraction, names, data):
    # names is alg + seed; there should only be one basename
    # with an x_param fixed, find max config of all other params
    # want to return alg names of config with highest auc
    # t_start = time.time()
    algs = [strip_seed(alg) for alg in names if find_param_value(x_param, alg) == param]
    aucs = defaultdict(int)
    # print(algs)
    for alg in set(algs):
        d = data[alg]
        aucs[alg] = np.mean(d[:, int((1 - auc_fraction) * d.shape[1]) : -1])
    max_alg = max(aucs, key = lambda k: aucs[k])
    # t_end = time.time()
    d = data[max_alg]
    return d[:, int((1 - auc_fraction) * d.shape[1]) : -1], aucs[max_alg]

def find_avg_config(curr_filenames, x_param, param, data, auc_fraction):
    d = []
    n_curr_runs = 0
    # curr_std = 0
    for alg in curr_filenames:
        if find_param_value(x_param, alg) == param:
            alg_d = data[strip_seed(alg)]
            n_curr_runs += alg_d.shape[0]
            # print(alg_d.shape, alg, strip_seed(alg))
            # print(alg_d.shape[1], alg_d)
            d.append(alg_d[:, int((1 - auc_fraction) * alg_d.shape[1]) : -1])
    curr_auc = np.mean(d)
    curr_std = np.std(np.mean(d, axis = -1)) / np.sqrt(n_curr_runs)
    return curr_auc, curr_std

def plot_sensitivity(x_params, colors, styles, fnames, data, params_to_fix, auc_fraction, env, write_dir, use_max = True, show_legend = True, suppress_title = False, xticks = None, yticks = None, figsize = (15, 10)):        
    # names array contains algs + the seed
    # an alg is with hyperparameters and w/o the seed
    fixed_params = "_".join(["{}={}".format(param, params_to_fix[param]) for param in params_to_fix])
    base_names = sorted(set([get_base_name(name) for name in data.keys()]))
    combined_base_names = '_'.join(base_names)
    # sensitivity curves for full auc
    # might also be desirable to have fractional auc
    # for each base name, have select param on x-axis and average auc / n_frames on the y-axis
    for x_param in x_params:
        # print("plotting {}".format(x_param))
        sensitivities = defaultdict(list)
        t_start = time.time()
        for base_name in base_names:
            # get list of all possible x_params for this method, if they exist
            curr_filenames = [name for name in fnames if name.startswith(base_name)]
            params = list([find_param_value(x_param, alg) for alg in curr_filenames]) # params actually swept over for this base_name
            # print(params)
            if len(params) <= 1 or None in params:
                print("skipping plot : {}, {}".format(x_param, base_name))
                continue
            params = sorted(list(set(params)))
            # print(params)
            for param in params:
                # get all algs with this param
                if use_max is True:
                    # max over params
                    d, curr_auc = find_max_config(x_param, param, auc_fraction, curr_filenames, data)
                    assert curr_auc == np.mean(d), (curr_auc, np.mean(d))
                    curr_std = np.std(np.mean(d, axis = -1)) / np.sqrt(d.shape[0])
                else:
                    # average over params
                    curr_auc, curr_std = find_avg_config(curr_filenames, x_param, param, data, auc_fraction)
                sensitivities[base_name].append([param, curr_auc, curr_std]) # getting standard error over runs
                print("{} runs for {}".format(d.shape[0], base_name))
        if len(sensitivities.keys()) == 0:
            # means no alg had this hyperparam or only one sensitivity point
            print("no sensitivity points for {}; skipping".format(x_param))
            continue
        t_end = time.time()
        print("getting aucs took {}s".format(t_end - t_start))
        
        t_start = time.time()
        plt.figure(figsize = figsize)
        # print(sensitivities)
        params = []
        for base_name in sorted(sensitivities.keys()):
            s = sensitivities[base_name]
            s = np.array(s)
            # print(s.shape)
            x = s[:, 0]
            if "lr" in x_param:
                x = np.log10(x)
            params += list(x)
            y = s[:, 1]
            std_error = s[:, 2]
            plt.plot(x, y, label = base_name, color = styles[base_name][0], linestyle = styles[base_name][1], marker = "o")
            plt.errorbar(x, y, yerr = std_error, color = styles[base_name][0], linestyle = styles[base_name][1])
            params = sorted(list(set(params)))
        plt.subplots_adjust(bottom=0.17, left=0.2)
        if "lr" in x_param:
            plt.xticks(params)
            plt.xlabel(r"$\log_{10}$(lr)")
        else:
            plt.xticks(params)
            plt.xlabel(x_param)
        # if len(params) > 1:
        #     plt.xlim(left = np.min(params), right = np.max(params))
        if params_to_fix:
            plt_name = "auc{}_usemax={}_{}_{}_sensitivity_{}.png".format(auc_fraction, use_max, env, x_param, fixed_params)
        else:
            plt_name = "auc{}_usemax={}_{}_{}_sensitivity.png".format(auc_fraction, use_max, x_param, env)
        fig_dir = write_dir + combined_base_names + "/"
        plt.ylabel("Average {}-AUC".format(auc_fraction))
        os.makedirs(fig_dir, exist_ok = True)
        plt.savefig(fig_dir + "UNLABELED_" + plt_name, dpi=30)

        if xticks is not None:
            plt.xticks(xticks)
        if yticks is not None:
            plt.yticks(yticks)
        if suppress_title is False:
            plt.title("{} Sensitivity on {} {}".format(x_param, env, fixed_params))
        if show_legend is True:
            plt.legend(frameon=False)
        plt.savefig(fig_dir + plt_name, dpi=30)
        plt.close()
        t_end = time.time()
        print("plotting {} took {}s".format(x_param, t_end - t_start))
