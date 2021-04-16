import os
import numpy as np
import argparse
import pathlib
import shutil
import re
import json
from collections import OrderedDict
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument("env", type=str, choices=["HalfCheetah-v2", "Pendulum-v0", "Swimmer-v2", "Reacher-v2","ContinuousMaze"])
parser.add_argument("--input_dir", type=str, default="my_results/normal_sweeps/joint_rkl_fkl/_results")
parser.add_argument("--env_json_dir", type=str, default="experiments/continuous_deep_control/jsonfiles/environment")
parser.add_argument("--window",type=int,default=20)
args = parser.parse_args()

input_parent = str(pathlib.PurePath(args.input_dir).parent)
path = pathlib.PurePath(args.input_dir)
last_inputname_part = str(path.name)

output_dir = os.path.join(input_parent, '_uncompressed{}'.format(last_inputname_part))

env = args.env
input_dir_env = os.path.join(args.input_dir, env + 'results')
output_dir_env = os.path.join(output_dir, env + 'results')

with open(os.path.join(args.env_json_dir, args.env + '.json'),"r") as ff:
    agent_json = json.load(ff, object_pairs_hook=OrderedDict)

max_steps = int(agent_json["TotalMilSteps"] * 1e6)
x_axis_steps = int(agent_json["XAxisSteps"])

os.makedirs(output_dir_env, exist_ok=True)

agents = ["ForwardKL", "ReverseKL"]


def f(base):
    input_rewards =  os.path.join(input_dir_env, base + '_EpisodeRewardsLC.txt')
    input_steps =  os.path.join(input_dir_env, base + '_EpisodeStepsLC.txt')
    rewards = np.loadtxt(input_rewards, delimiter=',')
    steps = np.loadtxt(input_steps, delimiter=',')

    reward_per_step = np.zeros( int(max_steps/x_axis_steps), dtype=np.float64)
    if args.env == "ContinuousMaze":
        if steps.shape == ():
            steps = np.array([max_steps])
            rewards = np.array([rewards])            
        elif steps[-1] < max_steps:
            steps[-1] = max_steps

        assert np.max(rewards) < 100000

    current_episode = 0
    running_frame = 0
    for global_idx in range(int(max_steps/x_axis_steps)):
        while(running_frame >= steps[current_episode]):
            current_episode += 1
        reward_per_step[global_idx] = np.mean(rewards[current_episode:current_episode+args.window])
        running_frame += x_axis_steps

    out_rewards_filename = os.path.join(output_dir_env, base + '_EpisodeRewardsLC.txt')
    reward_per_step.tofile(out_rewards_filename, sep=',', format='%15.8f')

    input_base_param_filename = os.path.join(input_dir_env, base + '_agent_Params.txt')
    out_base_param_filename = os.path.join(output_dir_env, base + '_agent_Params.txt')
    shutil.copyfile(input_base_param_filename, out_base_param_filename)            

for ag in agents:
    main_params_file = "{}_{}_agent_Params.json".format(env, ag)
    src = os.path.join(input_dir_env, main_params_file)
    dst = os.path.join(output_dir_env, main_params_file)
    if os.path.isfile( src ):
        shutil.copyfile(src, dst)
    else:
        continue #Likely the sweep for this agent was not done
    
    txt_pattern = re.compile("^(?P<base>{}_{}_setting_\d+_run_\d+)_.*.txt".format(args.env, ag))

    all_base = [ txt_pattern.search(f).group("base") for f in os.listdir(input_dir_env) if txt_pattern.search(f) is not None ]

    filtered_base = [ f for f in all_base if os.path.isfile( os.path.join(input_dir_env, f + '_EpisodeRewardsLC.txt')) and os.path.isfile( os.path.join(input_dir_env, f + '_EpisodeStepsLC.txt')) ]

    filtered_base = list(set(filtered_base))

    with Pool(10) as pool:
        pool.map(f, filtered_base)

    # for a in filtered_base:
    #     f(a)
           
