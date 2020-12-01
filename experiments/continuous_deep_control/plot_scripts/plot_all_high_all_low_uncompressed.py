
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
import json
from plot_config import get_xyrange
from collections import OrderedDict, defaultdict
import os
import argparse

import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter

plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%g'))
matplotlib.rcParams.update({'font.size': 50})

# Usage
# python3 plot_entropy_comparison.py $ROOT_LOC $ENV_NAME $STORE_DIR $PARSE_TYPE $OUTPUT_PLOT_DIR
# ROOT_LOC: root location
# STORE_DIR: stored_dir (should have mergedENVNAMEResults, which should also contain npy/
# PARSE_TYPE: parse_type (should be the same when generating npys)
# OUTPUT_PLOT_DIR: directory to dump plots

parser = argparse.ArgumentParser()
parser.add_argument("env_name", type=str)
parser.add_argument("--stored_dir", type=str, default="my_results/normal_sweeps/joint_rkl_fkl/_uncompressed_results")
parser.add_argument("--output_plot_dir", type=str, default="my_results/normal_sweeps/joint_rkl_fkl/_plots/all_high_all_low")
parser.add_argument("--parse_type", type=str, default="entropy_scale")
parser.add_argument("--root_dir", type=str, default="experiments/continuous_deep_control")
parser.add_argument("--best_setting_type", type=str, default="top20", choices=["best","top20"])

args = parser.parse_args()

show_plot = False

agents = ['ForwardKL', 'ReverseKL']
markers = dict(zip( agents, [None, None] ))
marker_sizes = dict(zip( agents, [15, 15] ))
mews = dict(zip( agents, [3, 3] ))

# Root loc
root_dir = args.root_dir

# Env name
env_filename = args.env_name
with open('{}/jsonfiles/environment/{}.json'.format(root_dir, env_filename), 'r') as env_dat:
    env_json = json.load(env_dat)

# parse_type
parse_type = args.parse_type

# out plot dir
output_plot_dir = os.path.join(args.output_plot_dir, args.best_setting_type)
if not os.path.exists(output_plot_dir):
    os.makedirs(output_plot_dir, exist_ok=True)


env_name = env_json['environment']
TOTAL_MIL_STEPS = env_json['TotalMilSteps']
EVAL_INTERVAL_MIL_STEPS = env_json['EvalIntervalMilSteps']
EVAL_EPISODES = env_json['EvalEpisodes']
X_AXIS_STEPS = env_json['XAxisSteps']
X_FORMATING_STEPS = env_json['XAxisFormatingSteps']

result_type = ['TrainEpisode']

# Stored Directory
stored_dir = os.path.join(args.stored_dir, 'merged{}results/'.format(env_name) )

data_type = ['avg', 'se']

agent_results = {}

max_length = 1

# Color scheme
temps = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0]
temps = sorted(temps)
auc_ratio = 0.5
colours = {"ForwardKL": None, "ReverseKL": None}

high_temps = [1, 0.5, 0.1]
low_temps = [0.05, 0.01, 0.005, 0.001, 0.0]

temp_types = ['high', 'low']

initial_rkl_color = cm.jet(0.65)
final_rkl_color = cm.jet(0.9)

initial_fkl_color = np.array((0, 26, 51))/255.
final_fkl_color = np.array((102, 181, 255))/255.

colours["ReverseKL"] = {'high': final_rkl_color, 'low': initial_rkl_color}
colours["ForwardKL"] = {'high': final_fkl_color, 'low': initial_fkl_color}

for ag in agents:

    agent_jsonfilename = '{}_{}_agent_Params.json'.format(env_name, ag)

    with open( os.path.join(stored_dir, agent_jsonfilename) , 'r') as agent_dat:
        agent_json = json.load(agent_dat, object_pairs_hook=OrderedDict)

    # load npy
    for t in temps:

        if ag == "ForwardKL" and t == 0:
            continue

        stored_npy_dir = os.path.join(stored_dir, 'npy')

        avg = np.load(os.path.join(stored_npy_dir, '{}_{}_{}_{}_{}_{}_{}.npy'.format(args.best_setting_type, env_name, ag, result_type[0], parse_type, t, data_type[0])))
        se = np.load(os.path.join(stored_npy_dir, '{}_{}_{}_{}_{}_{}_{}.npy'.format(args.best_setting_type, env_name, ag, result_type[0], parse_type, t, data_type[1])))

        high_low_t = 'high' if t in high_temps else 'low'

        if ag not in agent_results:
            agent_results[ag] = { 'high': [], 'low': []}
        agent_results[ag][high_low_t].append((avg, se))

        if max_length < len(avg):
            max_length = len(avg)
            print('agent: {}, max_length: {}'.format(ag, max_length))        

for ag in agents:
    for high_low_t in temp_types:
        mean = np.mean( list(map(lambda k: k[0],  agent_results[ag][high_low_t])), axis=0 )

        all_sterr = np.stack(list(map(lambda k: k[1],  agent_results[ag][high_low_t])), axis=0)
        all_var = np.square(all_sterr)
        combined_var = np.sum(all_var, axis=0) / np.square(all_var.shape[0])
        combined_sterr = np.sqrt(combined_var)

        agent_results[ag][high_low_t] = (mean, combined_sterr)

plt.figure(figsize=(18, 12))

_, ymin, ymax, yticks = get_xyrange(env_name)

xmax = int(max_length)

opt_range = range(0, xmax)
xlimt = (0, xmax - 1)
print('training xmax: {}'.format(xmax))

# Train Episode Rewards
ylimt = (ymin[0], ymax[0])


# Set axes labels
xtick_step = int( (TOTAL_MIL_STEPS*1e6/X_AXIS_STEPS)/5  )
tick = [o for o in opt_range[::xtick_step]]
ticklabel = [ int((x * X_AXIS_STEPS)/X_FORMATING_STEPS)  for x in tick]
plt.xticks(tick, ticklabel)

plt.yticks(ticks=yticks)
plt.xlim(xlimt)
plt.ylim(ylimt)

handle_arr = []

for jdx, ag in enumerate(agents):
    for high_low_t in temp_types:

        lc = agent_results[ag][high_low_t][0][:xmax]
        se = agent_results[ag][high_low_t][1][:xmax]

        mark_freq = xmax//30

        plt.plot(opt_range, lc, color=colours[ag][high_low_t], linewidth=1.0, label=ag, marker=markers[ag], markevery=mark_freq, markersize=marker_sizes[ag], mew=mews[ag])
        plt.fill_between(opt_range,  lc - se, lc + se, alpha=0.3, facecolor=colours[ag][high_low_t])


plt.subplots_adjust(bottom=0.17, left=0.2)

legend_elements = [Line2D([0], [0], linewidth = 6.0, marker=None, color=colours['ForwardKL']['high'], label='HIGH FKL'),
                   Line2D([0], [0], linewidth = 6.0, marker=None, color=colours['ForwardKL']['low'], label='LOW FKL'),
                   Line2D([0], [0], linewidth = 6.0, marker=None, color=colours['ReverseKL']['high'], label='HIGH RKL'),
                   Line2D([0], [0], linewidth = 6.0, marker=None, color=colours['ReverseKL']['low'], label='LOW RKL')
                  ]

if show_plot:
    plt.show()
else:
    map_xlabel = {100000: "Hundreds of Thousands", 1000: "Thousands"}
    plt.xlabel('{} of Timesteps'.format(map_xlabel[X_FORMATING_STEPS]))
    plt.ylabel("Average return").set_rotation(90)    
    #Unlabeled
    plt.savefig(os.path.join(output_plot_dir, "{}_{}_comparison_unlabeled.png".format(env_name, parse_type)))
    #Labeled
    # plt.title(env_name)
    plt.legend(handles=legend_elements, fontsize=30)
    plt.savefig(os.path.join(output_plot_dir, "{}_{}_comparison.png".format(env_name, parse_type)))
