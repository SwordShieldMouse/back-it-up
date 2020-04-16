import matplotlib.pyplot as plt

import numpy as np
import glob
import sys
from pathlib import Path
from collections import OrderedDict
import json

# list of agent.json names
agents = ['ForwardKL', 'ReverseKL']
params = ['actor_critic_dim', 'pi_lr', 'qf_vf_lr', 'entropy_scale', 'q_update_type']

store_dir = str(sys.argv[1])
env_name = str(sys.argv[2])

eval_last_N = True
last_N_ratio = 0.5

# Pendulum
# plt_yticks = [-1600, -1200, -800, -400, -200, 0]

dicts = {}
for p in params:
    dicts[p] = {}

# for agent_filename in ['forward_kl_bandits', 'reverse_kl_bandits']:
for agent_name in agents:

    sweep_params = params

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

    # actor: 0, 5, 10
    # expert: 0,1,2,3,4

    x = 1
    y = 1

    print("agent: {}".format(agent_name))
    for p in sweep_params:
        print("cur sweep param: {}".format(p))

        plt_x = agent_json[p]
        plt_xticks = range(len(plt_x))

        cur_param_num = len(plt_x)

        x *= cur_param_num
        plt_cum_reward_y = []
        for i in range(0, x, y):

            idx_array = []
            for j in range(i, i + num_settings, x):
                for k in range(j, j+y, 1):
                    idx_array.append(k)

            # print("{}: {} - sweep idx: {}".format(p, i, idx_array))

            # find result and mean
            # param_sweep_array.append(idx_array)
            # print(len(idx_array))
            # if (len(idx_array)) == 108:
            #     print(idx_array)
            result_array = []
            for idx in idx_array:
                xmax = len(train_mean_result[idx])
                if eval_last_N:
                    last_N = int(xmax * last_N_ratio)
                else:
                    last_N = 0
                result_array.append(np.sum(train_mean_result[idx][xmax-last_N:]))

            plt_cum_reward_y.append(np.max(result_array))
        y *= cur_param_num

        # print("param {}: {}".format(p, param_sweep_array))

        # plot result
        plt_x = plt_x[::-1]
        # plt_xticks = plt_xticks[::-1]
        plt_cum_reward_y = plt_cum_reward_y[::-1]
        # plt.plot(plt_xticks, plt_cum_reward_y)

        dicts[p][agent_name] = (p, plt_xticks, plt_x, plt_cum_reward_y.copy())

        # if you want to show sensitivity for each agent individually
        # plt.title("{}: {} sensitivity curve".format(agent_name, p))
        # plt.xlabel("{}".format(p))
        # plt.ylabel("Avg. {} AUC ".format(last_N_ratio), rotation=90)
        # plt.xticks(plt_xticks, plt_x)
        #
        # plt.savefig("{}_{}_sensitivity_curve.png".format(agent_name, p))
        # plt.clf()

# Combined plots

show_label = True

colors = [ '#377eb8', '#4daf4a', '#ff7f00',
                  '#f781bf', '#984ea3', '#999999','#a65628',
                  '#e41a1c', '#999999', '#dede00']


# q-learning methods
# ae, ae_plus, qt_opt, naf, picnn, wirefitting, sql, optimalq

# Bimodal
# for idx, a in enumerate(['ActorExpert', 'ActorExpert_Plus', 'SoftQlearning', 'NAF', 'PICNN', 'QT_OPT', 'WireFitting', 'OptimalQ']):
# Pendulum

for p in dicts:
    for idx, a in enumerate(agents):

        plt_xticks = dicts[p][a][1]
        plt_x = dicts[p][a][2]
        plt_y = dicts[p][a][3]
        plt.plot(plt_xticks, plt_y, colors[idx], label="{}".format(a))
        # plt.ylim(0, 1.5)

        if show_label:
            plt.xticks(plt_xticks, plt_x)
            # plt.yticks(plt_yticks, plt_yticks)

        else:
            plt.xticks(plt_xticks, [])
            # plt.yticks(plt_yticks, [])

    if show_label:
        plt.legend()
        plt.title("{} sensitivity curve".format(p))
        plt.xlabel(p)
        plt.ylabel("0.5 AUC", rotation=90)

    plt.savefig("combined_{}_sensitivity_curve.png".format(p))
    # plt.show()
    plt.clf()

# # policy + value
# for idx, a in enumerate(['ActorExpert_Separate', 'ActorExpert_Plus_Separate', 'SoftQlearning', 'ActorCritic_Separate', 'SoftActorCritic', 'DDPG']):
#
#     plt_xticks = value_lr_curve_data[a][1]
#     plt_x = value_lr_curve_data[a][2]
#     plt_y = value_lr_curve_data[a][3]
#     plt.plot(plt_xticks, plt_y, colors[idx], label="{}: {}".format(a, value_lr_curve_data[a][0]))
#     plt.ylim(0, 1.5)
#
#     if show_label:
#         plt.xticks(plt_xticks, plt_x)
#         plt.yticks(plt_yticks, plt_yticks)
#
#     else:
#         plt.xticks(plt_xticks, [])
#         plt.yticks(plt_yticks, [])
#
# if show_label:
#     plt.legend()
#     plt.title("Policy + Value: Value LR sensitivity curve")
#     plt.xlabel("Value LR")
#     plt.ylabel("Cum reward", rotation=90)
# plt.show()
# plt.clf()
#
# for idx, a in enumerate(['ActorExpert_Separate', 'ActorExpert_Plus_Separate', 'SoftQlearning', 'ActorCritic_Separate', 'SoftActorCritic', 'DDPG']):
#
#     plt_xticks = policy_lr_curve_data[a][1]
#     plt_x = policy_lr_curve_data[a][2]
#     plt_y = policy_lr_curve_data[a][3]
#     plt.plot(plt_xticks, plt_y, colors[idx], label="{}: {}".format(a, policy_lr_curve_data[a][0]))
#     plt.ylim(0, 1.5)
#
#     if show_label:
#         plt.xticks(plt_xticks, plt_x)
#         plt.yticks(plt_yticks, plt_yticks)
#
#     else:
#         plt.xticks(plt_xticks, [])
#         plt.yticks(plt_yticks, [])
# if show_label:
#     plt.legend()
#     plt.title("Policy + Value: Policy LR sensitivity curve")
#     plt.xlabel("Policy LR")
#     plt.ylabel("Cum reward", rotation=90)
# plt.show()

# ac_separate, sac, sql, ddpg, ae, ae_plus
