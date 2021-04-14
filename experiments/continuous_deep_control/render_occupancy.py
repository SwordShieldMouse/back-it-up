# -*- encoding:utf8 -*-
import gym
import tensorflow as tf
import environments.environments as envs
from utils.config import Config
from experiment import Experiment
import shutil
from lockfile import LockFile
import torch

import numpy as np
import json
import os
import datetime
from collections import OrderedDict
import argparse
import subprocess

from utils.main_utils import get_sweep_parameters, create_agent
#from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

def main():
    # parse arguments

    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0)
    # ContinuousMaze arguments
    parser.add_argument('--agent', type=str, choices=("FKL","RKL"), default="RKL")
    parser.add_argument('--setting', type=int, default=0)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--timestep', type=int, default=23000)

    args = parser.parse_args()

    full_aname = "ForwardKL" if args.agent == "FKL" else "ReverseKL"

    netsave_data_file =  "results/ContinuousMazeresults/saved_nets/ContinuousMaze_{agent}_setting_{setting}_run_{run}/{timestep}/ContinuousMaze_{agent}_setting_{setting}_run_{run}.tar".format(setting=args.setting, run=args.run, timestep=args.timestep, agent=full_aname)

    # read env/agent json
    with open('experiments/continuous_deep_control/jsonfiles/environment/ContinuousMaze.json', 'r') as env_dat:
        env_json = json.load(env_dat, object_pairs_hook=OrderedDict)
        env_json["TimeoutSteps"] = 500000

    j_aname = "forward_kl_single_run" if args.agent == "FKL" else "reverse_kl_single_run"
    with open('experiments/continuous_deep_control/jsonfiles/agent/{}.json'.format(j_aname), 'r') as agent_dat:
        agent_json = json.load(agent_dat, object_pairs_hook=OrderedDict)
    

    # initialize env
    train_env = envs.create_environment(env_json, 'results/ContinuousMazeresults/WorkingDir')
    test_env = envs.create_environment(env_json, 'results/ContinuousMazeresults/WorkingDir')      

    # Create env_params for agent
    env_params = {
            "env_name": train_env.name,
            "state_dim": train_env.state_dim,
            "state_min": train_env.state_min,
            "state_max": train_env.state_max,

            "action_dim": train_env.action_dim,
            "action_min": train_env.action_min,
            "action_max": train_env.action_max
    }


    agent_params, total_num_sweeps = get_sweep_parameters(agent_json['sweeps'], args.index)
    print('Agent setting: ', agent_params)

    # get run idx and setting idx
    RUN_NUM = int(args.index / total_num_sweeps)
    SETTING_NUM = args.index % total_num_sweeps

    # set Random Seed (for training)
    RANDOM_SEED = RUN_NUM
    arg_params = {
        "write_plot": None,
        "write_log": None,
        "writer": None
    }
    arg_params['random_seed'] = RANDOM_SEED
    torch.manual_seed(RANDOM_SEED)    
 
    # init config and merge custom config settings from json
    config = Config()
    config.merge_config(env_params)
    config.merge_config(agent_params)
    config.merge_config(arg_params)


    # initialize agent
    agent = create_agent(agent_json['agent'], config)

    train_env.render = True
    

    if os.path.isfile(netsave_data_file):
        checkpoint = torch.load(netsave_data_file)   
    else:
        raise FileNotFoundError('network not found')
    
    agent.network_manager.network.pi_net.load_state_dict(checkpoint['pi_net'])
    agent.network_manager.network.q_net.load_state_dict(checkpoint['q_net'])
    agent.network_manager.network.v_net.load_state_dict(checkpoint['v_net'])

    obs = train_env.reset()
    Aold = agent.start(obs, True)
    oreturn = 0
    while(True):
        obs_n, reward, done, info = train_env.step(Aold)
        oreturn += reward
        if done == True:
            break
        Aold = agent.step(obs_n, True)        
        input('Return: {}. Press ENTER'.format(oreturn))

if __name__ == '__main__':
    main()


