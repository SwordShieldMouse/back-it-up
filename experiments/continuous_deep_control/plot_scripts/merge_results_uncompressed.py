import numpy as np
import re
import glob
from pathlib import Path
import os
import sys
import json
import statistics
import argparse

from utils import get_agent_parse_info
from collections import OrderedDict
from shutil import copyfile
from multiprocessing import Pool

## Usage:
# python3 merge_results.py $RESULT_DIR $ROOT_LOC $ENV_NAME $AGENT_NAME $NUM_RUNS $USE_MOVING_AVG
#
# Generates merged{$ENVNAME}results/ with merged results at $RESULT_DIR
#
# RESULT_DIR : where {$ENV_NAME}results is located
# ROOT_LOC : root directory of code (where nonlinear_run.py and experiment.py is located)


parser = argparse.ArgumentParser()
parser.add_argument("env_name", type=str)
parser.add_argument("agent_name", type=str, choices=["ForwardKL", "ReverseKL"])
parser.add_argument("--results_dir", type=str, default="my_results/normal_sweeps/joint_rkl_fkl/_uncompressed_results")
parser.add_argument("--num_runs", type=int, default=30)
parser.add_argument("--root_dir", type=str, default="experiments/continuous_deep_control")

args = parser.parse_args()

result_dir = args.results_dir
root_dir = args.root_dir
env_name = args.env_name
agent_name = args.agent_name
num_runs = args.num_runs


# load env info
with open('{}/jsonfiles/environment/{}.json'.format(root_dir, env_name), 'r') as env_dat:
    env_json = json.load(env_dat)

TOTAL_MIL_STEPS = env_json['TotalMilSteps']
X_AXIS_STEPS = env_json['XAxisSteps']
EVAL_INTERVAL_MIL_STEPS = env_json['EvalIntervalMilSteps']
EVAL_EPISODES = env_json['EvalEpisodes']

# location of results and future merged results
store_dir = '{}/{}results/'.format(result_dir, env_name)
merged_dir = '{}/merged{}results'.format(result_dir, env_name)

if not os.path.exists(merged_dir):
    os.makedirs(merged_dir)
merged_dir += '/'

# get num_settings from agent_json
agent_json_name = '{}_{}_agent_Params.json'.format(env_name, agent_name)
with open('{}/{}'.format(store_dir, agent_json_name), 'r') as agent_dat:
    agent_json = json.load(agent_dat, object_pairs_hook=OrderedDict)
_, _, _, _, num_settings = get_agent_parse_info(agent_json)

# copy json to mergedresult folder
try:
    copyfile('{}/{}'.format(store_dir, agent_json_name), '{}/{}'.format(merged_dir, agent_json_name))
    print("Copied... {} to merged dir".format(agent_json_name))
except:
    raise ValueError("Json not copied properly")

print("Environment: {}".format(env_name))
print("Agent: {}".format(agent_name))
print("Num settings: {}".format(num_settings))
print("Num runs: {}".format(num_runs))

# Disabled Evaluation
# suffix = ['_EpisodeRewardsLC.txt','_EvalEpisodeMeanRewardsLC.txt','_EvalEpisodeStdRewardsLC.txt','_Params.txt']
# save_suffix = ['_TrainEpisodeMeanRewardsLC.txt','_TrainEpisodeStdRewardsLC.txt','_EvalEpisodeMeanRewardsLC.txt','_EvalEpisodeStdRewardsLC.txt','_Params.txt']

suffix = ['_EpisodeRewardsLC.txt', '_Params.txt']
save_suffix = ['_TrainEpisodeMeanRewardsLC.txt', '_TrainEpisodeStdRewardsLC.txt', '_Params.txt', '_all.npy']


missingindexes = []
train_mean_rewards = []
train_std_rewards = []
train_all_rewards = []
eval_mean_rewards = []
eval_std_rewards = []

params = []
params_fn = None

# for each setting
for setting_num in range(num_settings):
    run_non_count = 0

    train_lc_arr = []
    train_lc_length_arr = []
    eval_mean_lc_arr = []
    

    def f(run_num):
        train_rewards_filename = store_dir + env_name + '_' + agent_name + '_setting_' + str(setting_num) + '_run_' + str(run_num) + suffix[0]

        # skip if file does not exist
        if not Path(train_rewards_filename).exists():
            run_non_count = 1
            # add dummy
            lc_0 = -1e10 * np.ones( int(TOTAL_MIL_STEPS/X_AXIS_STEPS)) # + np.nan  # will be padded
            train_lc = lc_0

            print(' setting ' + train_rewards_filename + ' does not exist')
            missingindex = num_settings * run_num + setting_num
            return (run_non_count, missingindex, train_lc)

        lc_0 = np.loadtxt(train_rewards_filename, delimiter=',')

        run_non_count = 0
        missingindex = None
        train_lc = lc_0        

        return (run_non_count, missingindex, train_lc)

    with Pool(10) as pool:
        mixed_arr = pool.map(f, range(num_runs))
        run_non_count += np.sum(list(map(lambda k: k[0], mixed_arr)))
        missingindexes.extend( filter(lambda k: k is not None, list(map(lambda k: k[1], mixed_arr))) )
        train_lc_arr = list(map(lambda k: k[2], mixed_arr))

    train_lc_arr = np.array(train_lc_arr)

    if run_non_count == num_runs:
        print('setting ' + str(setting_num) + ' does not exist')
        print(np.shape(train_lc_arr), train_lc_arr)

    # Need to have same size
    train_mean_rewards.append(np.nanmean(train_lc_arr, axis=0))
    train_std_rewards.append(np.nanstd(train_lc_arr, axis=0))
    train_all_rewards.append(train_lc_arr)

    '''read in params file'''
    paramfile = store_dir + env_name + '_' + agent_name + '_setting_' + str(setting_num) + '_run_*' + suffix[-1]
    files = glob.glob(paramfile)
    if len(files)<1:
        continue
    onefile = files[0]
    newfilename = re.sub(store_dir, '', files[0])
    newfilename = re.sub('_setting_[0-9]+_','_',newfilename)
    newfilename = merged_dir + re.sub('_run_[0-9]+_', '_', newfilename)
    params_fn = newfilename
    setting_params = np.loadtxt(onefile, delimiter=',', dtype='str')
    setting_params = np.insert(setting_params, 0, setting_num)
    params.append(setting_params)
params = np.array(params)


allres = [train_mean_rewards, train_std_rewards, params, train_all_rewards]
for i in range(len(save_suffix)):
    name = merged_dir + env_name + '_' + agent_name + save_suffix[i]

    if i == 2:
        name = params_fn

    print('Saving...' + name)
    if i == 3:
        train_all_rewards = np.array(train_all_rewards)
        np.save(name, allres[i])        
    else:
        np.savetxt(name, allres[i], fmt='%s', delimiter=',')
        

print('missing indexes are:  -- - - - - - - - - --')
missed = ''
for missid in missingindexes:
   missed += (str(missid)+',')
print(missed)


