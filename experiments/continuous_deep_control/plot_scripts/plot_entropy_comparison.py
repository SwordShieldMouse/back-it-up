# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys
import json
from matplotlib.lines import Line2D
from plot_config import get_xyrange
from collections import OrderedDict, defaultdict

# Usage
# python3 plot_entropy_comparison.py $ROOT_LOC $ENV_NAME $STORE_DIR $PARSE_TYPE
# ROOT_LOC: root location
# STORE_DIR: stored_dir (should have mergedENVNAMEResults, which should also contain npy/
# PARSE_TYPE: parse_type (should be the same when generating npys)

show_labels = True
show_plot = False

agents = ['ForwardKL']
markers = ['o', 'x']
marker_sizes = [8, 14]
mews = [5, 3]

# agents = ['ReverseKL']
# markers = ['x']
# marker_sizes = [14]
# mews = [3]

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

max_length = 1

# Color scheme
temps = [0, 0.01, 0.1, 1]
colours = [cm.jet(0.65 + (.99 - 0.65) * ix / 4) for ix in range(len(temps))]
colours = list(reversed(colours))
temps = list(reversed(temps))

for ag in agents:

    agent_jsonfilename = '{}_{}_agent_Params.json'.format(env_name, ag)

    with open(stored_dir + agent_jsonfilename, 'r') as agent_dat:
        agent_json = json.load(agent_dat, object_pairs_hook=OrderedDict)

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

    if not show_labels:
        plt.xticks(loc_arr, [])
        plt.yticks([-1600, -800,-200], [])
    else:
        plt.xticks(loc_arr, val_arr)
        plt.yticks([-1600, -800, -200], [-1600, -800, -200])

elif env_name == 'Reacher-v2':

    ep_length = 50

    loc_arr = np.array([0, 2000-1, 4000-1, 6000-1]) #, 7980, 9980])
    val_arr = np.array([0, 100 , 200, 300])  # , 7980, 9980])

    if not show_labels:
        plt.xticks(loc_arr, [])  # val_arr)
        plt.yticks([-80, -40, 0], [])
    else:
        plt.xticks(loc_arr, val_arr)
        plt.yticks([-80, -40, 0], [-80, -40, 0])

elif env_name == 'Swimmer-v2':

    ep_length = 1000

    loc_arr = np.array([0, 100-1, 200-1, 300-1]) # , 380, 480])
    val_arr = np.array([0, 100, 200, 300])  # , 380, 480])

    if not show_labels:
        plt.xticks(loc_arr, [])
        plt.yticks([0, 20, 40], [])

        legend_elements = [Line2D([0], [0], marker='o', color='black', label='Forward KL',
                                  markerfacecolor='black', markersize=14),
                           Line2D([0], [0], marker='x', color='black', label='Reverse KL',
                                  markerfacecolor='black', markersize=14, mew=6)]
        plt.legend(handles=legend_elements, frameon=False, prop={'size': 22})
    else:
        plt.xticks(loc_arr, val_arr)
        plt.yticks([0, 20, 40], [0, 20, 40])

else:
    plt.xticks(opt_range[::50], np.linspace(0.0, float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax - 1)),
                                            int(TOTAL_MIL_STEPS / EVAL_INTERVAL_MIL_STEPS) + 1)[::50])


plt.xlim(xlimt)
plt.ylim(ylimt)

handle_arr = []

for jdx, ag in enumerate(agents):
    for idx, t in enumerate(temps):

        # Skip HardForwardKL
        if ag == 'ForwardKL' and t == 0:
            continue

        lc = agent_results[ag][idx][0][:xmax]
        se = agent_results[ag][idx][1][:xmax]
        plt.fill_between(opt_range,  lc - se, lc + se, alpha=0.2, facecolor=colours[idx])

        if env_name == 'Pendulum-v0':
            mark_freq = 5  # 3
        elif env_name == 'Reacher-v2':
            mark_freq = 240  # 160
        elif env_name == 'Swimmer-v2':
            mark_freq = 15  # 10

        if jdx == 0:  # Forward KL
            marker_size = 12
        else:  # Reverse KL
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
