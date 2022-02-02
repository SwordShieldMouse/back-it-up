
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
matplotlib.rcParams.update({'font.size': 70})

# Usage
# python3 plot_entropy_comparison.py $ROOT_LOC $ENV_NAME $STORE_DIR $PARSE_TYPE $OUTPUT_PLOT_DIR
# ROOT_LOC: root location
# STORE_DIR: stored_dir (should have mergedENVNAMEResults, which should also contain npy/
# PARSE_TYPE: parse_type (should be the same when generating npys)
# OUTPUT_PLOT_DIR: directory to dump plots

parser = argparse.ArgumentParser()
parser.add_argument("env_name", type=str)
parser.add_argument("--stored_dir", type=str, default="my_results/normal_sweeps/joint_rkl_fkl/_uncompressed_results")
parser.add_argument("--output_plot_dir", type=str, default="my_results/normal_sweeps/joint_rkl_fkl/_plots/all_high_all_low_bar")
parser.add_argument("--parse_type", type=str, default="entropy_scale")
parser.add_argument("--root_dir", type=str, default="")
parser.add_argument("--best_setting_type", type=str, default="top20", choices=["best","top20"])

args = parser.parse_args()

show_plot = False

agents = ['ReverseKL','ForwardKL']
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

# DIFFERENT FROM OTHER INITIAL AND FINAL COLORS!
initial_rkl_color = np.array((0, 128, 66))/255.
final_rkl_color = np.array((0, 204, 105))/255.

# DIFFERENT FROM OTHER INITIAL AND FINAL COLORS!
initial_fkl_color = np.array((0, 66, 128))/255.
final_fkl_color = np.array((0, 119, 230))/255.

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
        all_ = np.load(os.path.join(stored_npy_dir, '{}_{}_{}_{}_{}_{}_all.npy'.format(args.best_setting_type, env_name, ag, result_type[0], parse_type, t)))

        high_low_t = 'high' if t in high_temps else 'low'

        if ag not in agent_results:
            agent_results[ag] = { 'high': [], 'low': []}
        agent_results[ag][high_low_t].append((avg, all_))

        if max_length < len(avg):
            max_length = len(avg)
            print('agent: {}, max_length: {}'.format(ag, max_length))        

_, ymin, ymax, yticks = get_xyrange(env_name)

for ag in agents:
    for high_low_t in temp_types:
        # mean = np.mean( list(map(lambda k: k[0],  agent_results[ag][high_low_t])), axis=0 )

        all_lc = np.stack(list(map(lambda k: k[1],  agent_results[ag][high_low_t])), axis=0)

        unrolled_all_lc = np.reshape(all_lc, [-1, all_lc.shape[-1]])
        unrolled_all_lc = (unrolled_all_lc - ymin[0])/(ymax[0] - ymin[0])

        len_ov_2 = int(unrolled_all_lc.shape[-1]/2)
        all_auc = np.mean(unrolled_all_lc[:, len_ov_2:], axis=1)
        mean = np.mean(all_auc)
        combined_sterr = np.std(all_auc) / np.sqrt(all_auc.shape[0])

        agent_results[ag][high_low_t] = (mean, combined_sterr)

plt.figure(figsize=(10, 12))
width=0.05
intra_offset = 0.01
inter_offset = 0.1

# Train Episode Rewards

plt.ylim(bottom=0., top=1.)

x_axis = plt.gca().axes.get_xaxis()
plt.xticks([0.0 , intra_offset + 2* width + inter_offset])
x_axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: "RKL" if x == 0 else "FKL"))
plt.xlabel("")
plt.ylabel("AUC @ 0.5")
plt.locator_params(axis="y", nbins=4)

handle_arr = []

for idx, ag in enumerate(agents):
    avg_rewards = []
    std_error = []
    labels = np.array([-width/2. - intra_offset/2., width/2. + intra_offset/2.])
    labels += float(idx)*(inter_offset + 2*width + intra_offset)
    colors = []    
    for all_high_all_low_t in temp_types:
        el = agent_results[ag][all_high_all_low_t]
        if "Forward" in ag:
            basealg = "Forward"
            name = "ForwardKL"
        elif "Reverse" in ag:
            basealg = "Reverse"
            name = "ReverseKL"
        avg_rewards.append(el[0])
        std_error.append(el[1])
        colors.append(colours[name][all_high_all_low_t])        
    avg_rewards = np.array(avg_rewards)
    std_error = np.array(std_error)
    
    plt.bar(labels, avg_rewards, yerr=std_error, color=colors, width=width)

plt.subplots_adjust(bottom=0.17, left=0.2)

legend_elements = [Line2D([0], [0], linewidth = 6.0, marker=None, color=colours['ForwardKL']['high'], label='HIGH FKL'),
                   Line2D([0], [0], linewidth = 6.0, marker=None, color=colours['ForwardKL']['low'], label='LOW FKL'),
                   Line2D([0], [0], linewidth = 6.0, marker=None, color=colours['ReverseKL']['high'], label='HIGH RKL'),
                   Line2D([0], [0], linewidth = 6.0, marker=None, color=colours['ReverseKL']['low'], label='LOW RKL')
                  ]

if show_plot:
    plt.show()  
else:    
    #Unlabeled
    plt.savefig(os.path.join(output_plot_dir, "{}_{}_comparison_unlabeled.png".format(env_name, parse_type)),bbox_inches='tight',dpi=30)
    #Labeled
    # plt.title(env_name)
    plt.legend(handles=legend_elements, fontsize=30)
    plt.savefig(os.path.join(output_plot_dir, "{}_{}_comparison.png".format(env_name, parse_type)),bbox_inches='tight',dpi=30)
