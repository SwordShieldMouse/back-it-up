
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
parser.add_argument("--agent", type=str, choices=["ForwardKL","ReverseKL","both"],default="both")
parser.add_argument("--stored_dir", type=str, default="my_results/normal_sweeps/joint_rkl_fkl/_uncompressed_results")
parser.add_argument("--output_plot_dir", type=str, default="my_results/normal_sweeps/joint_rkl_fkl/_plots/entropy_comparison")
parser.add_argument("--parse_type", type=str, default="entropy_scale")
parser.add_argument("--root_dir", type=str, default="experiments/continuous_deep_control")
parser.add_argument("--best_setting_type", type=str, default="top20", choices=["best","top20"])

args = parser.parse_args()

show_plot = False

full_agents = ['ForwardKL', 'ReverseKL']
agents = full_agents if args.agent == 'both' else [args.agent]
markers = dict(zip( full_agents, ['o', 'x'] ))
marker_sizes = dict(zip( full_agents, [20, 35] ))
mews = dict(zip( full_agents, [3, 7] ))

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

suffix = 'Result_{}'.format(args.agent)
data_type = ['avg', 'se']

agent_results = defaultdict(list)

max_length = 1

# Color scheme
temps = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0]
temps = sorted(temps)
auc_ratio = 0.5
colours = {"ForwardKL": None, "ReverseKL": None}
rkl_colors = [cm.jet(0.65 + (.99 - 0.65) * ix / len(temps)) for ix in range(len(temps))]

fkl_colors = [None for _ in range(len(temps))]
initial_fkl_color = np.array((0, 26, 51))/255.
final_fkl_color = np.array((204, 230, 255))/255.
for s_t_idx, s_temp in enumerate(sorted(temps)):
    t = float(s_t_idx)/(len(temps) - 1)
    color = initial_fkl_color*(1-t) + t*final_fkl_color
    fkl_colors[ temps.index(s_temp) ] = color    

colours["ReverseKL"] = rkl_colors
colours["ForwardKL"] = fkl_colors


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

        agent_results[ag].append((avg, se, t))

        if max_length < len(avg):
            max_length = len(avg)
            print('agent: {}, max_length: {}'.format(ag, max_length))        

if len(agents) == 2: #Only top 3 temperatures
    for ag in agents:
        first_auc_idx = int(agent_results[ag][-1][0].shape[0] * (1.-auc_ratio))
        aucs = list(map( lambda k: np.mean(k[0][first_auc_idx:]) , agent_results[ag]))
        sorted_aucs = sorted(aucs,reverse=True)
        best_aucs = sorted_aucs[:3]
        deleted_idxs = []
        for auc_idx, auc in enumerate(aucs):
            if auc not in best_aucs:
                deleted_idxs.append(auc_idx)
        deleted_idxs = sorted(deleted_idxs, reverse=True)
        for didx in deleted_idxs:
            del agent_results[ag][didx]

plt.figure(figsize=(18, 12))

_, ymin, ymax = get_xyrange(env_name)

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


plt.xlim(xlimt)
plt.ylim(ylimt)

handle_arr = []

for jdx, ag in enumerate(agents):
    a_temps = list(map(lambda k: k[2], agent_results[ag]))
    for a_idx, t in enumerate(a_temps): #a_idx: temperature index in agent list

        idx = temps.index(t) #idx: temperature index in temperatures list (for colours)

        # Skip HardForwardKL
        if ag == 'ForwardKL' and t == 0:
            continue

        lc = agent_results[ag][a_idx][0][:xmax]
        se = agent_results[ag][a_idx][1][:xmax]

        mark_freq = xmax//30

        plt.plot(opt_range, lc, color=colours[ag][idx], linewidth=0.5, label=ag, marker=markers[ag], markevery=mark_freq, markersize=marker_sizes[ag], mew=mews[ag])
        plt.fill_between(opt_range,  lc - se, lc + se, alpha=0.3, facecolor=colours[ag][idx])


plt.subplots_adjust(bottom=0.17, left=0.2)

legend_elements = [Line2D([0], [0], marker=markers["ForwardKL"], color='black', label='Forward KL',
                        markerfacecolor='black', markersize=marker_sizes["ForwardKL"], mew = mews["ForwardKL"]), Line2D([0], [0], marker=markers["ReverseKL"], color='black', label='Reverse KL',
                        markerfacecolor='black', markersize=marker_sizes["ReverseKL"], mew = mews["ReverseKL"])]

if show_plot:
    plt.show()
else:
    map_xlabel = {100000: "Hundreds of Thousands", 1000: "Thousands"}
    plt.xlabel('{} of Frames'.format(map_xlabel[X_FORMATING_STEPS]))
    plt.ylabel("Average return").set_rotation(90)    
    #Unlabeled
    plt.savefig(os.path.join(output_plot_dir, "{}_{}_{}_comparison_unlabeled.png".format(args.agent, env_name, parse_type)))
    #Labeled
    plt.title(env_name)
    plt.legend(handles=legend_elements, frameon=False)
    plt.savefig(os.path.join(output_plot_dir, "{}_{}_{}_comparison.png".format(args.agent, env_name, parse_type)))
