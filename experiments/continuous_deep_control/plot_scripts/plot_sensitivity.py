import matplotlib.pyplot as plt

import argparse
import numpy as np
import sys
from collections import OrderedDict, defaultdict
import json
from matplotlib.lines import Line2D

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
parser.add_argument('--store_dir', type=str, default="my_results/normal_sweeps/joint_rkl_fkl/_results")
parser.add_argument('--output_plot_dir', type=str, default="my_results/normal_sweeps/joint_rkl_fkl/_plots/sensitivity" )
parser.add_argument('--agents',nargs='*',type=str,choices=('ForwardKL','ReverseKL'))
parser.add_argument('--num_runs',type=int,default=10)

parser.add_argument('--best_setting_type',type=str,choices=('best','top20'),default='top20')

args = parser.parse_args()

# list of agent.json names
agents = args.agents
# sweep_params = ['pi_lr', 'qf_vf_lr','actor_critic_dim','n_hidden','batch_size', 'n_action_points']
sweep_params = ['pi_lr', 'qf_vf_lr']


# Handcoded temperature sweeps
# temps = [1, 0.1, 0.01, 0]
temps = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0]
store_dir = args.store_dir
env_name = args.env_name
output_plot_dir = os.path.join(args.output_plot_dir, args.best_setting_type)
if not os.path.isdir(output_plot_dir):
    os.makedirs(output_plot_dir)

best_setting_type = args.best_setting_type

eval_last_N = True
last_N_ratio = 0.5
num_runs = args.num_runs
moving_avg_window = 20

show_label = True

_, ymin, ymax, yticks = get_xyrange(env_name)


def movingaverage (values, window):
    return [np.mean(values[max(0, i - (window-1)):i+1]) for i in range(len(values))]

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

    # 1, 0.1, 0.01
    for t in temps:
        print("== temp: {}".format(t))

        for p in sweep_params:
            print("cur sweep param: {}".format(p))

            plt_x = agent_json[p]
            plt_xticks = range(len(plt_x))
            plt_point_y = []
            plt_stderr_y = []

            # iterate through each parameter value: 1e-3, 1e-4, 1e-5
            for val in agent_json[p]:

                idx_arr = idx_dict[p][val]
                result_mean_array = []
                result_stderr_array = []

                if t == 0 and agent_name == "ForwardKL":
                    # skip
                    plt_point_y.append(np.nan)
                    plt_stderr_y.append(np.nan)

                else:

                    for i in idx_arr:

                        if i not in t_idx_dict[t]:
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
                                lc_0 = movingaverage(lc_0, moving_avg_window)[xmax - last_N:]
                                each_run_avg_auc_arr.append(np.mean(lc_0))

                            except:
                                print("missing {}.. skipping".format(train_rewards_filename))

                        result_mean_array.append(np.mean(each_run_avg_auc_arr)) #Between runs for same setting
                        result_stderr_array.append(np.std(each_run_avg_auc_arr)/np.sqrt(len(each_run_avg_auc_arr))) #Between runs for same setting
                    
                    if best_setting_type == 'best':
                        best_idx = np.nanargmax(result_mean_array)                    
                        assert(np.nanmax(result_mean_array) == result_mean_array[best_idx])
                        plt_point_y.append(result_mean_array[best_idx])
                        plt_stderr_y.append(result_stderr_array[best_idx])

                    else:
                        result_mean_array = np.array(result_mean_array)
                        result_stderr_array = np.array(result_stderr_array)
                        sorted_idxs = np.argsort(result_mean_array)[::-1]
                        n_top_args = max(int(0.2 * len(sorted_idxs)),2)
                        best_idxs = sorted_idxs[:n_top_args]
                        plt_point_y.append(np.mean(result_mean_array[best_idxs]))

                        var_best = np.square(result_stderr_array[best_idxs])
                        avg_var_best = np.sum(var_best, axis=0) / float(n_top_args**2)
                        plt_stderr_y.append(np.sqrt(avg_var_best))



            # plot result
            plt_x = plt_x[::-1]
            plt_point_y = plt_point_y[::-1]
            plt_stderr_y = plt_stderr_y[::-1]

            param_dicts[p][agent_name][t] = (p, agent_name, t, plt_xticks, plt_x, plt_point_y.copy(), plt_stderr_y.copy())


# Combined plots
import matplotlib.cm as cm

colours = [cm.jet(0.65 + (.99 - 0.65) * ix /float(len(temps)) ) for ix in range(len(temps))]
colours = list(reversed(colours))

for p in param_dicts:
    for idx, a in enumerate(agents):

        if "Forward" in a:
            linestyle = "-"
            marker = "o"
            dashes = (5, 0)
            name = "ForwardKL"
            mew = 3
            marker_size = 7

        elif "Reverse" in a:
            linestyle = "-"
            marker = "x"
            dashes = (5, 0)
            name = "ReverseKL"
            mew = 3
            marker_size = 7
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
            plt.plot(plt_xticks, plt_y, label="{}, tau={}".format(a, t), color=colours[t_idx], linestyle=linestyle, marker=marker, mew=mew, markersize=marker_size)
            plt.errorbar(plt_xticks, plt_y, yerr=plt_y_stderr, color=colours[t_idx], linestyle=linestyle)

    plt.ylim(ymin, ymax)
    if show_label:
        plt.xticks(plt_xticks, plt_x)
        plt.yticks(yticks, yticks)
        plt.legend()
        plt.title("{} sensitivity curve".format(p))
        plt.xlabel(p)
        plt.ylabel("0.5 AUC", rotation=90)
        plt.savefig("{}/{}_combined_{}_sensitivity_curve.png".format(output_plot_dir, env_name, p),dpi=30)
    else:

        legend_elements = [Line2D([0], [0], marker='o', color='black', label='Forward KL',
                                  markerfacecolor='black', markersize=10),
                           Line2D([0], [0], marker='x', color='black', label='Reverse KL',
                                  markerfacecolor='black', markersize=10, mew=4)]
        plt.legend(handles=legend_elements, frameon=False, prop={'size': 14})

        plt.xticks(plt_xticks, [])
        plt.yticks(yticks, [])
        plt.savefig("{}/{}_combined_{}_sensitivity_curve_unlabeled.png".format(output_plot_dir, env_name, p),dpi=30)

    # plt.show()
    plt.clf()
