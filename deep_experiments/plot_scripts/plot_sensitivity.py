import matplotlib.pyplot as plt

import numpy as np
import sys
from collections import OrderedDict
import json
from matplotlib.lines import Line2D

# Usage
# python3 plot_sensitivity.py $STORE_DIR $ENV_NAME

# list of agent.json names
agents = ['ForwardKL', 'ReverseKL']
sweep_params = ['pi_lr', 'qf_vf_lr']

# Handcoded temperature sweeps
temps = [1, 0.1, 0.01, 0]
store_dir = str(sys.argv[1])
env_name = str(sys.argv[2])

eval_last_N = True
last_N_ratio = 0.5
num_runs = 30
moving_avg_window = 20

show_label = False

if env_name == "Pendulum-v0":
    idx_dict= {
        "pi_lr": {
            1e-2: [0, 4, 8],
            1e-3: [1, 5, 9],
            1e-4: [2, 6, 10],
            1e-5: [3, 7, 11]
        },
        "qf_vf_lr": {
            1e-1: [0, 1, 2, 3],
            1e-2: [4, 5, 6, 7],
            1e-3: [8, 9, 10, 11]
        }
    }
    y_ticks = [-1600, -800,-200]
    ymin, ymax = -1600, -100

elif env_name == "Reacher-v2":
    idx_dict = {
            "pi_lr": {
                1e-3: [0, 3, 6],
                1e-4: [1, 4, 7],
                1e-5: [2, 5, 8]
            },
            "qf_vf_lr": {
                1e-2: [0, 1, 2],
                1e-3: [3, 4, 5],
                1e-4: [6, 7, 8]
            }
        }

    y_ticks = [-80, -40, 0]
    ymin, ymax = -120, 0

elif env_name == "Swimmer-v2":
    idx_dict = {
        "pi_lr": {
            1e-3: [0, 3, 6],
            1e-4: [1, 4, 7],
            1e-5: [2, 5, 8]
        },
        "qf_vf_lr": {
            1e-2: [0, 1, 2],
            1e-3: [3, 4, 5],
            1e-4: [6, 7, 8]
        }
    }
    y_ticks = [0, 20, 40]
    ymin, ymax = -10, 40
else:
    raise ValueError("Invalid env_name")


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

                        if t != 0:  # apply only for non-hard.
                            t_idx = json_temps.index(t)
                            i = num_settings * t_idx + i

                        each_run_avg_auc_arr = []
                        # load all train results for that setting
                        for n in range(num_runs):

                            if t == 0: # for Hard RKL load from diff. directory (it was swept separately)
                                train_rewards_filename = "{}/hardrkl/{}results/{}_{}_setting_{}_run_{}_EpisodeRewardsLC.txt".format(
                                    store_dir, env_name, env_name, agent_name, i, n)
                            else:
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

                        result_mean_array.append(np.mean(each_run_avg_auc_arr))
                        result_stderr_array.append(np.std(each_run_avg_auc_arr)/np.sqrt(len(each_run_avg_auc_arr)))

                    # best_idx = num_settings * t_idx + idx_arr[np.argmax(result_mean_array)]
                    best_idx = np.argmax(result_mean_array)
                    assert(np.max(result_mean_array) == result_mean_array[best_idx])

                    plt_point_y.append(result_mean_array[best_idx])
                    plt_stderr_y.append(result_stderr_array[best_idx])

            # plot result
            plt_x = plt_x[::-1]
            plt_point_y = plt_point_y[::-1]
            plt_stderr_y = plt_stderr_y[::-1]

            param_dicts[p][agent_name][t] = (p, agent_name, t, plt_xticks, plt_x, plt_point_y.copy(), plt_stderr_y.copy())


# Combined plots
import matplotlib.cm as cm
colours = [cm.jet(0.65 + (.99 - 0.65) * ix / 4) for ix in range(len(temps))]
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
        plt.yticks(y_ticks, y_ticks)
        plt.legend()
        plt.title("{} sensitivity curve".format(p))
        plt.xlabel(p)
        plt.ylabel("0.5 AUC", rotation=90)
        plt.savefig("{}/combined_{}_sensitivity_curve.png".format(store_dir, p))
    else:

        legend_elements = [Line2D([0], [0], marker='o', color='black', label='Forward KL',
                                  markerfacecolor='black', markersize=10),
                           Line2D([0], [0], marker='x', color='black', label='Reverse KL',
                                  markerfacecolor='black', markersize=10, mew=4)]
        plt.legend(handles=legend_elements, frameon=False, prop={'size': 14})

        plt.xticks(plt_xticks, [])
        plt.yticks(y_ticks, [])
        plt.savefig("{}/combined_{}_sensitivity_curve_unlabeled.png".format(store_dir, p))

    # plt.show()
    plt.clf()
