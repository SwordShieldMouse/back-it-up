# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import sys
from pathlib import Path
import json
from matplotlib.lines import Line2D

from find_agent_best_setting import get_xyrange

from utils import get_agent_parse_info
from collections import OrderedDict, defaultdict
####################

### Usage

# arg1: root location
# arg2: env_name
# arg3: stored_dir (should have mergedENVNAMEResults, which should also contain npy/
# arg4: parse_type (should be the same when generating npys)

show_labels = False
show_plot = False
agents = ['ForwardKL', 'ReverseKL']

window_ma = 20

# Root loc
root_dir = str(sys.argv[1])

# Env name
env_filename = str(sys.argv[2])
with open('{}/jsonfiles/environment/{}.json'.format(root_dir, env_filename), 'r') as env_dat:
    env_json = json.load(env_dat)

# parse_type
parse_type = str(sys.argv[4])

env_name = env_json['environment']
TOTAL_MIL_STEPS = env_json['TotalMilSteps']
EVAL_INTERVAL_MIL_STEPS = env_json['EvalIntervalMilSteps']
EVAL_EPISODES = env_json['EvalEpisodes']

result_type = ['TrainEpisode']

# Stored Directory
stored_dir = str(sys.argv[3])+'merged{}results/'.format(env_name)

suffix = 'BestResult'
data_type = ['avg', 'se']

agent_results = defaultdict(list)
# agents_avg = []
# agents_se = []

max_length = 1

# Color scheme
import matplotlib.cm as cm
temps = [0, 0.01, 0.1, 1]
# start = 0.5
# end = 0.99
# colours = [cm.rainbow(start + (end-start) * ix / 4) for ix in range(len(temps))]
colours = [cm.jet(0.65 + (.99 - 0.65) * ix / 4) for ix in range(len(temps))]
# colours = list(reversed(colours[1:]))
colours = list(reversed(colours))

# temps = list(reversed(temps[1:]))
temps = list(reversed(temps))



for ag in agents:

    agent_jsonfilename = '{}_{}_agent_Params.json'.format(env_name, ag)

    with open(stored_dir + agent_jsonfilename, 'r') as agent_dat:
        agent_json = json.load(agent_dat, object_pairs_hook=OrderedDict)

    # type_arr, _, _, _, _ = get_agent_parse_info(agent_json, divide_type=parse_type)
    #
    # # make sure current temps match the experimented temps
    # print(temps, type_arr)
    # assert len(temps) == len(type_arr)
    # for m in range(len(type_arr)):
    #     assert type_arr[m] == temps[m]

    # load npy
    for t in temps:

        if ag == "ForwardKL" and t == 0:
            agent_results[ag].append((None, None))
            continue

        avg = np.load('{}npy/{}_{}_{}_{}_{}_{}.npy'.format(stored_dir, env_name, ag, result_type[0], parse_type, t, data_type[0]))
        se = np.load('{}npy/{}_{}_{}_{}_{}_{}.npy'.format(stored_dir, env_name, ag, result_type[0], parse_type, t, data_type[1]))

        agent_results[ag].append((avg, se))

        if max_length < len(avg):
            max_length = len(avg)
            print('agent: {}, max_length: {}'.format(ag, max_length))


plt.figure(figsize=(10, 6))

# xmax = int(max_length)
_, ymin, ymax = get_xyrange(env_name)

if env_name == 'Reacher-v2':
    xmax = int(6000)
elif env_name == 'Swimmer-v2':
    xmax = int(300)
else:
    xmax = int(max_length)

opt_range = range(0, xmax)
xlimt = (0, xmax - 1)
print('training xmax: {}'.format(xmax))

# Train Episode Rewards
ylimt = (ymin[0], ymax[0])

if show_labels:
    plt.title(env_name)
    plt.xlabel('Training steps (per 1000 steps)')
    plt.ylabel("Cum. Reward per episode").set_rotation(90)

# Set axes labels
if env_name == 'Pendulum-v0':

    ep_length = 200
    loc_arr = np.array([0, 19, 39, 59, 79, 99])
    val_arr = np.array([0, 4, 8, 12, 16, 20])

    # val_arr = np.array([ep_length * (window_ma + i)/1000 for i in loc_arr])

    if not show_labels:
        plt.xticks(loc_arr, [])  # val_arr)
        plt.yticks([-1600, -800,-200], [])  # [-1600, -1200, -800, -400, -200, 0])
    else:
        plt.xticks(loc_arr, val_arr)
        plt.yticks([-1600, -800, -200], [-1600, -800, -200])

elif env_name == 'Reacher-v2':

    ep_length = 50

    loc_arr = np.array([0, 2000-1, 4000-1, 6000-1]) #, 7980, 9980])
    val_arr = np.array([0, 100 , 200, 300])  # , 7980, 9980])
    # val_arr = np.array([ep_length * (window_ma + i) / 1000 for i in loc_arr])

    if not show_labels:
        plt.xticks(loc_arr, [])  # val_arr)
        plt.yticks([-80, -40, 0], [])
    else:
        plt.xticks(loc_arr, val_arr)
        plt.yticks([-80, -40, 0], [-80, -40, 0])

    # # 20000 episodes, 19991 points
    # loc_arr = np.array([0, 3991, 7991, 11991, 15991, 19991])
    # val_arr = np.array([45, 200, 400, 600, 800, 1000])
    #
    # if show_labels:
    #     plt.xticks(loc_arr, val_arr)
    #     plt.yticks([-40, -30, -20, -10, 0], [-40, -30, -20, -10, 0])
    # else:
    #     plt.xticks(loc_arr, [])
    #     plt.yticks([-40, -30, -20, -10, 0], [])  # [0, 2000, 4000, 6000, 7000])

# TODO: modify
elif env_name == 'Swimmer-v2':

    ep_length = 1000

    loc_arr = np.array([0, 100-1, 200-1, 300-1]) # , 380, 480])
    val_arr = np.array([0, 100, 200, 300])  # , 380, 480])
    # val_arr = np.array([ep_length * (window_ma + i) / 1000 for i in loc_arr])

    if not show_labels:
        plt.xticks(loc_arr, [])  # val_arr)
        plt.yticks([0, 20, 40], [])

        legend_elements = [Line2D([0], [0], marker='o', color='black', label='Forward KL',
                                  markerfacecolor='black', markersize=14),
                           Line2D([0], [0], marker='x', color='black', label='Reverse KL',
                                  markerfacecolor='black', markersize=14, mew=6)]
        plt.legend(handles=legend_elements, frameon=False, prop={'size': 22})
    else:
        plt.xticks(loc_arr, val_arr)
        plt.yticks([0, 20, 40], [0, 20, 40])



    # # 1000 episodes, 991 points
    # loc_arr = np.array([0, 191, 391, 591, 791, 991])
    # val_arr = np.array([45, 200, 400, 600, 800, 1000])
    #
    # if show_labels:
    #     plt.xticks(loc_arr, val_arr)
    #     plt.yticks([0, 10, 20, 30, 40, 50], [0, 10, 20, 30, 40, 50])
    # else:
    #     plt.xticks(loc_arr, [])
    #     plt.yticks([0, 10, 20, 30, 40, 50], [])  # [0, 2000, 4000, 6000, 7000])


else:
    plt.xticks(opt_range[::50], np.linspace(0.0, float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax - 1)),
                                            int(TOTAL_MIL_STEPS / EVAL_INTERVAL_MIL_STEPS) + 1)[::50])


plt.xlim(xlimt)
plt.ylim(ylimt)

handle_arr = []
# for idx in range(len(agents)):
#     pad_length = len(opt_range) - len(agents_avg[idx][:xmax])
#
#
#     plt.fill_between(opt_range,
#                      np.append(agents_avg[idx][:xmax] - agents_se[idx][:xmax], np.zeros(pad_length) + np.nan),
#                      np.append(agents_avg[idx][:xmax] + agents_se[idx][:xmax], np.zeros(pad_length) + np.nan),
#                      alpha=0.2, facecolor=colors[idx])
#     handle, = plt.plot(opt_range, np.append(agents_avg[idx][:xmax], np.zeros(pad_length) + np.nan), colors[idx],
#                        linewidth=1.2, label=agents[idx], linestyle=linestyles[idx], dashes=dashes[idx])
#     handle_arr.append(handle)


markers = ['o', 'x']
marker_sizes=[8, 14]
mews=[5, 3]

for jdx, ag in enumerate(agents):

    for idx, t in enumerate(temps):

        if ag == 'ForwardKL' and t == 0:
            continue

        lc = agent_results[ag][idx][0][:xmax]
        se = agent_results[ag][idx][1][:xmax]
        plt.fill_between(opt_range,  lc - se, lc + se, alpha=0.2, facecolor=colours[idx])

        if env_name == 'Pendulum-v0':
            mark_freq = 5 # 3
        elif env_name == 'Reacher-v2':
            mark_freq = 240 # 160
        elif env_name == 'Swimmer-v2':
            mark_freq = 15 # 10

        if jdx == 0: # Forward KL
            marker_size = 12
        else: # Reverse KL
            marker_size = 14
        handle, = plt.plot(opt_range, lc, color=colours[idx], linewidth=1.2, label='{}_{}_{}'.format(ag, parse_type, t), marker=markers[jdx], markevery=mark_freq, markersize=marker_sizes[jdx], mew=mews[jdx])
        handle_arr.append(handle)

if show_labels:
    plt.legend(handle_arr, )

if show_plot:
    plt.show()
else:
    if show_labels:
        plt.savefig("{}_{}_{}_comparison.png".format(env_name, 'train', parse_type))
    else:
        plt.savefig("{}_{}_{}_comparison_unlabeled.png".format(env_name, 'train', parse_type))
