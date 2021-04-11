import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import argparse
import numpy as np
import sys
from collections import OrderedDict, defaultdict
import json
import os
from plot_config import get_xyrange_with_p

from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%g'))
matplotlib.rcParams.update({'font.size': 50})

import ast

def tryeval(val):
    try:
        val = ast.literal_eval(val)
    except ValueError:
        val = None
    return val

# Usage
# python3 plot_sensitivity.py $STORE_DIR $ENV_NAME $OUTPUT_PLOT_DIR --agents $AGENTS

parser = argparse.ArgumentParser()

parser.add_argument('env_name', type=str)
parser.add_argument('agent', type=str,choices=('ForwardKL','ReverseKL'))
parser.add_argument('--store_dir', type=str, default="my_results/normal_sweeps/joint_rkl_fkl/_uncompressed_results")
parser.add_argument('--output_plot_dir', type=str, default="my_results/normal_sweeps/joint_rkl_fkl/_plots/sensitivity" )
parser.add_argument('--num_runs',type=int,default=30)

parser.add_argument('--best_setting_type',type=str,choices=('best','top20'),default='top20')

args = parser.parse_args()

# list of agent.json names
agents = [args.agent]
# sweep_params = ['pi_lr', 'qf_vf_lr','actor_critic_dim','n_hidden','batch_size', 'n_action_points']
sweep_params = ['pi_lr', 'qf_vf_lr']

last_N_ratio = 0.5
eval_last_N = True

# Handcoded temperature sweeps
temps = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0]
temps = sorted(temps)
store_dir = args.store_dir
env_name = args.env_name
output_plot_dir = os.path.join(args.output_plot_dir, args.best_setting_type)
if not os.path.isdir(output_plot_dir):
    os.makedirs(output_plot_dir,exist_ok=True)

best_setting_type = args.best_setting_type

num_runs = args.num_runs


param_dicts = {}
for p in sweep_params:
    param_dicts[p] = {}
    for ag in agents:
        param_dicts[p][ag] = {}

for agent_name in agents:

    # load json
    json_dir = '{}/merged{}results/{}_{}_agent_Params.json'.format(store_dir, env_name, env_name, agent_name)

    with open(json_dir, 'r') as agent_dat:

        json_data = json.load(agent_dat, object_pairs_hook=OrderedDict)

        agent_name = json_data['agent']
        agent_json = json_data['sweeps']

        idx_dict = {}
        t_idx_dict = {}

        parse_txt_fields = ['setting'] + list( json_data['sweeps'].keys() )

        param_parse_txt_idx = []
        for param in sweep_params:
            try:
                param_parse_txt_idx.append( parse_txt_fields.index(param) )
            except ValueError:
                print('Desired parameter not in json')
                exit()

        try:
            temperature_parse_txt_idx =  parse_txt_fields.index("entropy_scale")
        except ValueError:
            print('Entropy scale not in json')
            exit()

        params_txt_dir = '{}/merged{}results/{}_{}_agent_Params.txt'.format(store_dir, env_name, env_name, agent_name)
        settings_info = np.loadtxt(params_txt_dir, delimiter=',', dtype='str')

        for settings_info_row in settings_info:
            for param_tmp_idx, param in enumerate(sweep_params):
                value = tryeval(settings_info_row[ param_parse_txt_idx[param_tmp_idx] ])
                if param not in idx_dict:
                    idx_dict[param] = {}
                if value not in idx_dict[param]:
                    idx_dict[param][value] = []
                idx_dict[param][value].append(tryeval(settings_info_row[0]))

            setting_temp = tryeval(settings_info_row[temperature_parse_txt_idx])
            if setting_temp not in t_idx_dict:
                t_idx_dict[setting_temp] = []
            t_idx_dict[setting_temp].append(tryeval(settings_info_row[0]))


    train_mean_filename = '{}/merged{}results/{}_{}_TrainEpisodeMeanRewardsLC.txt'.format(store_dir, env_name, env_name, agent_name)
    train_mean_result = np.loadtxt(train_mean_filename, delimiter=',')


    params_max_idx = [len(agent_json[p]) for p in sweep_params]
    num_settings = np.prod(params_max_idx)


    print("agent: {}".format(agent_name))
    json_temps = agent_json['entropy_scale']

    for t in temps:
        print("== temp: {}".format(t))

        for p in sweep_params:
            print("cur sweep param: {}".format(p))

            plt_x = agent_json[p]
            plt_xticks = range(len(plt_x))
            plt_point_y = []
            plt_stderr_y = []

            # iterate through each parameter value
            for val in agent_json[p]:

                idx_arr = idx_dict[p][val]
                result_mean_array = []
                result_stderr_array = []
                result_all_array = []

                if t == 0 and agent_name == "ForwardKL":
                    # skip
                    plt_point_y.append(np.nan)
                    plt_stderr_y.append(np.nan)

                else:

                    for i in idx_arr: #Indexes (a.k.a settings) that contain the parameter

                        if i not in t_idx_dict[t]: #verify if they also contain the temperature
                            continue

                        each_run_avg_auc_arr = []
                        # load all train results for that setting
                        for n in range(num_runs):

                            train_rewards_filename = "{}/{}results/{}_{}_setting_{}_run_{}_EpisodeRewardsLC.txt".format(
                                store_dir, env_name, env_name, agent_name, i, n)

                            try:
                                lc_0 = np.loadtxt(train_rewards_filename, delimiter=',')

                                xmax = len(lc_0)
                                last_N = int(xmax * last_N_ratio) if eval_last_N else 0
                                lc_0 = lc_0[xmax - last_N:]
                                each_run_avg_auc_arr.append(np.mean(lc_0))

                            except:
                                print("missing {}.. skipping".format(train_rewards_filename))

                        result_mean_array.append(np.mean(each_run_avg_auc_arr)) #Between runs for same setting
                        if best_setting_type == 'best':
                            result_stderr_array.append(np.std(each_run_avg_auc_arr)/np.sqrt(len(each_run_avg_auc_arr))) #Between runs for same setting
                        else:
                            result_all_array.append(np.array(each_run_avg_auc_arr)) #Between runs for same setting                            
                    
                    if best_setting_type == 'best':
                        best_idx = np.nanargmax(result_mean_array)                    
                        assert(np.nanmax(result_mean_array) == result_mean_array[best_idx])
                        plt_point_y.append(result_mean_array[best_idx])
                        plt_stderr_y.append(result_stderr_array[best_idx])

                    else:
                        result_mean_array = np.array(result_mean_array)
                        result_all_array = np.array(result_all_array)
                        
                        sorted_idxs = np.argsort(result_mean_array)[::-1]
                        n_top_args = max(int(0.2 * len(sorted_idxs)),2)
                        best_idxs = sorted_idxs[:n_top_args]
                        plt_point_y.append(np.mean(result_mean_array[best_idxs]))

                        best_all_array = result_all_array[best_idxs]
                        unrolled_all_array = np.reshape(best_all_array, [-1])

                        combined_stderr = np.std(unrolled_all_array) / np.sqrt(unrolled_all_array.shape[0])
                        plt_stderr_y.append(combined_stderr)



            # plot result
            plt_x = plt_x[::-1]
            plt_point_y = plt_point_y[::-1]
            plt_stderr_y = plt_stderr_y[::-1]

            param_dicts[p][agent_name][t] = (p, agent_name, t, plt_xticks, plt_x, plt_point_y.copy(), plt_stderr_y.copy())


# Combined plots
import matplotlib.cm as cm

rkl_colors = [None for _ in range(len(temps))]
initial_rkl_color = np.array((0, 51, 26))/255.
final_rkl_color = np.array((204, 255, 204))/255.
for s_t_idx, s_temp in enumerate(sorted(temps)):
    t = float(s_t_idx)/(len(temps) - 1)
    color = initial_rkl_color*(1-t) + t*final_rkl_color
    rkl_colors[ temps.index(s_temp) ] = color  

fkl_colors = [None for _ in range(len(temps))]
initial_fkl_color = np.array((0, 26, 51))/255.
final_fkl_color = np.array((204, 230, 255))/255.
for s_t_idx, s_temp in enumerate(sorted(temps)):
    t = float(s_t_idx)/(len(temps) - 1)
    color = initial_fkl_color*(1-t) + t*final_fkl_color
    fkl_colors[ temps.index(s_temp) ] = color 

colours = {"ReverseKL": rkl_colors, "ForwardKL": fkl_colors}

figsize = (18, 12)
fig = plt.figure(figsize = figsize)
plt.subplots_adjust(bottom=0.17, left=0.2)

for p in param_dicts:
    for idx, a in enumerate(agents):

        if "Forward" in a:
            linestyle = "-"
            marker = "."
            dashes = (5, 0)
            name = "ForwardKL"
            mew = 3
            marker_size = 15

        elif "Reverse" in a:
            linestyle = "-"
            marker = "."
            dashes = (5, 0)
            name = "ReverseKL"
            mew = 3
            marker_size = 15
        else:
            raise ValueError("Invalid agent name")

        for t_idx, t in enumerate(temps):

            # skip Hard FKL
            if t == 0 and a == 'ForwardKL':
                continue

            plt_xticks = param_dicts[p][a][t][3]
            plt_x = param_dicts[p][a][t][4]
            plt_y = param_dicts[p][a][t][5]
            plt_y_stderr = param_dicts[p][a][t][6]
            plt.plot(plt_xticks, plt_y, label="{}, tau={}".format(a, t), color=colours[a][t_idx], linestyle=linestyle, marker=marker, mew=mew, markersize=marker_size, linewidth=5.0)
            plt.errorbar(plt_xticks, plt_y, yerr=plt_y_stderr, color=colours[a][t_idx], linestyle=linestyle, linewidth=3.0)

    plt.xlabel(p)
    translate_param = { "pi_lr": "Actor lr", "qf_vf_lr": "Critic lr"}
    if 'lr' in p:
        plt_x = np.log10(plt_x)
        plt.xlabel(r"$\log_{{10}}$" + "({})".format(translate_param[p]))

    plt.ylabel("Average {}-AUC".format(last_N_ratio), rotation=90)        

    plt.xticks(plt_xticks, plt_x)
    _, ymin, ymax, yticks = get_xyrange_with_p(args.env_name, p)
    plt.ylim(bottom=ymin[0], top=ymax[0])
    plt.yticks(ticks=yticks)
    plt.savefig("{}/{}_{}_combined_{}_sensitivity_curve_unlabeled.png".format(output_plot_dir, env_name, a, p))


    # full_agents = ['ForwardKL', 'ReverseKL']
    # markers = dict(zip( full_agents, ['.', '.'] ))
    # marker_sizes = dict(zip( full_agents, [20, 35] ))
    # mews = dict(zip( full_agents, [3, 7] ))    
    # legend_elements = [Line2D([0], [0], marker=markers["ForwardKL"], color='black', label='Forward KL',
    #                         markerfacecolor='black', markersize=marker_sizes["ForwardKL"], mew = mews["ForwardKL"]), Line2D([0], [0], marker=markers["ReverseKL"], color='black', label='Reverse KL',
    #                         markerfacecolor='black', markersize=marker_sizes["ReverseKL"], mew = mews["ReverseKL"])]


    # plt.xticks(plt_xticks, plt_x)
    # plt.title("{} sensitivity curve".format(translate_param[p]))
    # plt.legend(handles=legend_elements, frameon=False)
    # plt.savefig("{}/{}_{}_combined_{}_sensitivity_curve.png".format(output_plot_dir, env_name, a, p))
    # # plt.show()
    # plt.clf()
