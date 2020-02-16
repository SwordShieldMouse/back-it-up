import torch
import re
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy as sp
from collections import defaultdict
import time

torch.set_default_dtype(torch.float)

def atanh(x):
    return (torch.log(1 + x) - torch.log(1 - x)) / 2

def normal_cdf(x, mean, std):
    return (1 + sp.special.erf((x - mean) / (std * np.sqrt(2)))) / 2

def Beta(alpha, beta):
    return torch.exp(torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta))

def soft_update(target, src, tau):
    for target_param, param in zip(target.parameters(), src.parameters()):
        target_param.detach_()
        target_param.copy_(target_param * (1.0 - tau) +
                           param * tau)

def find_param_value(param, s):
    # find the param value corresponding to param in s
    # return re.search("({}=)(\d*(\.|e-)?\d*)".format(param), s)[2]
    res = re.search("({}=((\de\-(\d*))|(\d\.?\d*)))".format(param), s)
    if res == None:
        return None
    return float(res[2])

def strip_seed(s):
    return re.search('^(.*)_(seed=(\d)*)', s)[1]

def reconstruct_array(a):
    # given an array of format [[moving window return, ep length], ...] reconstruct the array as the agent experienced it
    r = []
    for value, length in a:
        # print(value, length)
        r += [value] * int(length)
    return r

def filter_alg_names(algs, params_to_fix, algs_to_include):
    # can take name with or without seed
    algs2 = []
    for alg in algs:
        if not any([alg.startswith(base_name + "_") for base_name in algs_to_include]):
            continue
        append_or_not = True
        for param in params_to_fix:
            if param not in alg:
                continue 
            if params_to_fix[param] != find_param_value(param, alg):
                append_or_not = False
                break
        if append_or_not is True:
            algs2.append(alg)
    return algs2

def rand_argmax(array):
    # get argmax of numpy array with random tie-breaking
    return np.argmax(np.random.random(array.shape) * (array == array.max()))

def plot_top_algs(colors, styles, names, params_to_fix, auc_fraction, algs_to_include, env, read_dir, write_dir, y_axis_label = "Average Return", x_axis_label = "Frame", show_legend=True, suppress_title = False, xticks = None, yticks = None):
    fixed_params = "_".join(["{}={}".format(param, params_to_fix[param]) for param in params_to_fix])
    os.makedirs(write_dir + fixed_params + "/", exist_ok=True)

    t_start = time.time()   
    filtered_names = filter_alg_names(names, params_to_fix, algs_to_include)
    assert filtered_names != [], "filtered out all the data"
    t_end = time.time()
    print("filtering names took {}s".format(t_end - t_start))

    # get length of x-axis
    # n_x = int(np.load(read_dir + names[0]).sum(axis = 0)[1])
    n_x = np.load(read_dir + names[0]).shape[0]
    # print(np.load(read_dir + names[0]).shape)
    print("plotting data of length {}".format(n_x))

    all_data = defaultdict(list)

    t_start = time.time()
    for name in filtered_names:
        alg = re.search('^(.*)_(seed=(\d)*)', name)[1]
        try:
            all_data[alg].append(np.load(read_dir + name))
            if len(all_data[alg][-1]) != n_x:
                print("inconsistent input length of {}; removing...".format(len(all_data[alg][-1])))
                all_data[alg].pop()
            # all_data[alg].append(reconstruct_array(np.load(read_dir + name)))
        except ValueError:
            print("problems in loading {}".format(name))
        # print(all_data[alg].shape)
    t_end = time.time()
    print("reading arrays took {}s".format(t_end - t_start))

    # stack np arrays 
    data = all_data
    for key in data:
        # print(data[key][-1].shape)
        data[key] = np.stack(data[key], axis = 0)
        # print(data[key].shape)

    t_start = time.time()
    aucs = {key : np.nanmean(np.nansum(data[key][:, int((1 - auc_fraction) * data[key].shape[1]) : -1], axis = 1)) for key in data}
    sorted_aucs = sorted(aucs.items(), key = lambda item: item[1], reverse = True)
    t_end = time.time()
    print("getting sorted aucs took {}s".format(t_end - t_start))

    # plot the top n of each method
    # method is the name of an alg w/o hyperparameters
    t_start = time.time()
    algs = set([strip_seed(name) for name in filtered_names])
    base_names = sorted(set([re.search('^(\w)*_', name)[0][:-1] for name in algs]))
    combined_base_names = '_'.join(base_names)
    n = 1
    plot_names = []
    for base_name in base_names:
        counter = 0
        for name in sorted_aucs:
            if name[0].startswith(base_name + "_"):
                plot_names.append((name[0], base_name))
                counter += 1
            if counter >= n:
                break
    plt.figure(figsize = (13, 8))
    max_auc = 0
    for ix, (name, base_name) in enumerate(plot_names):
        # base_name = 
        d = data[name]
        std_error = np.std(d, axis = 0) / np.sqrt(d.shape[0])
        avg_rewards = np.nanmean(d, axis = 0)
        if max_auc < np.amax(avg_rewards):
            max_auc = np.amax(avg_rewards)
        # print(d.shape)
        plt.plot(avg_rewards, color = styles[base_name][0], linestyle = styles[base_name][1], label = name, dashes = styles[base_name][2])
        # print(name, base_name)
        plt.fill_between(np.arange(d.shape[1]), avg_rewards - std_error, avg_rewards + std_error, alpha = 0.3, color = styles[base_name][0])

    # plot labeled and unlabeled
    if params_to_fix:
        plt.savefig(write_dir + combined_base_names + "_auc{}-{}_compare_{}_UNLABELED.png".format(auc_fraction, env, "_".join(["{}={}".format(param, params_to_fix[param]) for param in params_to_fix])))
    else:   
        plt.savefig(write_dir + combined_base_names + "_auc{}{}_compare_UNLABELED.png".format(auc_fraction, env))

    if show_legend is True:
        plt.legend()
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    if suppress_title is False:
        plt.title("Top Algs by {}-AUC on {}, {}".format(auc_fraction, env, fixed_params))
    # print(plot_names)
    if params_to_fix:
        plt.savefig(write_dir + combined_base_names + "_auc{}-{}_compare_{}.png".format(auc_fraction, env, "_".join(["{}={}".format(param, params_to_fix[param]) for param in params_to_fix])))
    else:   
        plt.savefig(write_dir + combined_base_names + "_auc{}{}_compare.png".format(auc_fraction, env))
    plt.close()
    t_end = time.time()
    print("plotting took {}s".format(t_end - t_start))

    # also make separate plots of the top 5 for each method
    for base_name in base_names:
        plt.figure(figsize = (13, 8))
        counter = 0
        plot_names = []
        for name in sorted_aucs:
            if name[0].startswith(base_name + "_"): # add underscore to ensure we get the full name and not just match a substring of another name
                plot_names.append(name[0])
                counter += 1
            if counter == 5:
                break
        # print(plot_names)
        for ix, name in enumerate(plot_names):
            # data = np.load("data/{}".format(name))
            d = data[name]
            # print(d.shape)
            std_error = np.nanstd(d, axis = 0) / np.sqrt(d.shape[0])
            avg_rewards = np.nanmean(d, axis = 0)
            # print(d)
            plt.plot(avg_rewards, colors[ix], label = name)
            plt.fill_between(np.arange(d.shape[1]), avg_rewards - std_error, avg_rewards + std_error, alpha = 0.3, color = colors[ix])
        # plt.ylim(top = max_auc)
        plt.title("Top {} on {}, {}".format(base_name, env, fixed_params))
        if show_legend is True:
            plt.legend()
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        if params_to_fix:
            plt.savefig(write_dir + fixed_params + "/" + "top_{}_{}_{}.png".format(env, base_name, fixed_params))
        else:
            plt.savefig(write_dir + "top_{}_{}.png".format(env, base_name))
        plt.close()

def find_max_config(x_param, param, auc_fraction, names, read_dir):
    # names is alg + seed; there should only be one basename
    # with an x_param fixed, find max config of all other params
    # want to return alg names of config with highest auc
    t_start = time.time()
    algs_with_seed = [alg for alg in names if find_param_value(x_param, alg) == param]
    # print(algs_with_seed)
    aucs = defaultdict(int)
    for alg_with_seed in algs_with_seed:
        base_name = strip_seed(alg_with_seed)
        try:
            d = np.load(read_dir + alg_with_seed)
        except:
            d = np.load(read_dir + alg_with_seed, allow_pickle=True)
        aucs[base_name] += np.sum(d[int((1 - auc_fraction) * d.shape[0]) : -1])
    max_alg = max(aucs, key = lambda k: aucs[k])
    t_end = time.time()
    # print("finding max config for {} took {}s".format(x_param, t_end - t_start))
    return [alg_with_seed for alg_with_seed in algs_with_seed if strip_seed(alg_with_seed) in max_alg]

def plot_sensitivity(x_param, colors, styles, names, params_to_fix, algs_to_include, auc_fraction, env, read_dir, write_dir, use_max = True, show_legend = True, suppress_title = False, xticks = None, yticks = None):        
    # names array contains algs + the seed
    # an alg is with hyperparameters and w/o the seed
    t_start = time.time()
    algs = set([re.search('^(.*)_(seed=(\d)*)', name)[1] for name in names])
    algs = filter_alg_names(algs, params_to_fix, algs_to_include)
    # print(algs)
    names = filter_alg_names(names, params_to_fix, algs_to_include)
    fixed_params = "_".join(["{}={}".format(param, params_to_fix[param]) for param in params_to_fix])
    assert algs != [], "empty algs"
    assert names != [], "empty names"
    t_end = time.time()
    print("filtering took {}s".format(t_end - t_start))
    base_names = sorted(set([re.search('^(\w)*_', name)[0][:-1] for name in algs]))
    combined_base_names = '_'.join(base_names)
    # sensitivity curves for full auc
    # might also be desirable to have fractional auc
    # for each base name, have select param on x-axis and average auc / n_frames on the y-axis
    sensitivities = {base_name: [] for base_name in base_names}
    max_auc = 0
    # print(base_names)
    t_start = time.time()
    for base_name in base_names:
        # get list of all possible x_params for this method
        curr_filenames = [name for name in names if name.startswith(base_name)]
        params = list([float(re.search('({}=((\de\-(\d*))|(\d\.?\d*)))'.format(x_param), alg)[2]) for alg in curr_filenames])
        params = sorted(list(set(params)))
        # print(params, algs)
        for param in params:
            # get all algs with this param
            if use_max is True:
                # max over params

                max_config_alg = find_max_config(x_param, param, auc_fraction, curr_filenames, read_dir)
                # d = np.stack([reconstruct_array(np.load(read_dir + alg)) for alg in max_config_alg], axis = 0) 
                d = np.stack([np.load(read_dir + alg) for alg in max_config_alg], axis = 0) # max over other hyperparams
                # d = [np.prod(np.load(read_dir + alg), axis = 1).sum() for alg in max_config_alg]
                
                # print(d.shape)
            else:
                # average over params
                # d = [np.load(read_dir + alg).prod(axis = 1).sum() for alg in curr_filenames if find_param_value(x_param, alg) == param ]

                d = np.stack([np.load(read_dir + alg) for alg in curr_filenames if find_param_value(x_param, alg) == param ], axis = 0) #"{}={}".format(x_param, param) in alg], axis = 0).squeeze() 

                # d = np.stack([reconstruct_array(np.load(read_dir + alg)) for alg in curr_filenames if find_param_value(x_param, alg) == param ], axis = 0) #"{}={}".format(x_param, param) in alg], axis = 0).squeeze() # average over other hyperparams
            d = d[:, int((1 - auc_fraction) * d.shape[1]) : -1] 
            if max_auc < np.nanmean(np.nansum(d, axis = -1)):
                max_auc = np.nanmean(np.nansum(d, axis = -1))
            sensitivities[base_name].append([param, np.nanmean(np.nansum(d, axis = -1)), np.nanstd(np.nansum(d, axis = -1)) / np.sqrt(d.shape[0])]) # getting standard error over runs
    t_end = time.time()
    print("getting aucs took {}s".format(t_end - t_start))
    
    t_start = time.time()
    plt.figure(figsize = (13, 8))
    # print(sensitivities["HardReverseKL"], sensitivities["ForwardKL"])
    for ix, base_name in enumerate(sorted(sensitivities.keys())):
        s = sensitivities[base_name]
        s = np.array(s)
        # print(s.shape)
        x = s[:, 0]
        y = s[:, 1]
        std_error = s[:, 2]
        plt.plot(x, y, label = base_name, color = styles[base_name][0], linestyle = styles[base_name][1], marker = "o")
        # plt.fill_between(x, y - std_error, y + std_error, alpha = 0.3)
        plt.errorbar(x, y, yerr = std_error, color = styles[base_name][0], linestyle = styles[base_name][1])
        
    plt.ylim(top = max_auc)

    if params_to_fix:
        plt.savefig(write_dir + combined_base_names + "_auc{}_{}_{}_sensitivity_{}_UNLABELED.png".format(auc_fraction, env, x_param, fixed_params))
    else:
        plt.savefig(write_dir + combined_base_names + "_auc{}_{}_{}_sensitivity_UNLABELED.png".format(auc_fraction, x_param, env))

    plt.xlabel(x_param)
    plt.ylabel("Average {}-AUC".format(auc_fraction))
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    if suppress_title is False:
        plt.title("{} Sensitivity on {}, {}".format(x_param, env, fixed_params))
    if show_legend is True:
        plt.legend()
    if params_to_fix:
        plt.savefig(write_dir + combined_base_names + "_auc{}_{}_{}_sensitivity_{}.png".format(auc_fraction, env, x_param, fixed_params))
    else:
        plt.savefig(write_dir + combined_base_names + "_auc{}_{}_{}_sensitivity.png".format(auc_fraction, x_param, env))
    plt.close()
    t_end = time.time()
    print("plotting took {}s".format(t_end - t_start))
