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
    parser.add_argument('--agent', type=str, choices=("FKL","RKL"), default="FKL")
    parser.add_argument('--setting', type=int, default=0)
    parser.add_argument('--run', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timestep', type=int, default=23000)
    parser.add_argument('--from_results',action="store_true")
    parser.add_argument('--environment',type=str,choices=["EasyContinuousMaze","MediumContinuousMaze"],default="EasyContinuousMaze")

    args = parser.parse_args()

    full_aname = "ForwardKL" if args.agent == "FKL" else "ReverseKL"

    if args.from_results:
        netsave_data_file =  "results/{env}results/saved_nets/{env}_{agent}_setting_{setting}_run_{run}/{timestep}/{env}_{agent}_setting_{setting}_run_{run}.tar".format(setting=args.setting, run=args.run, timestep=args.timestep, agent=full_aname, env=args.environment)
    else:
        netsave_data_file =  "saved_nets/{env}_{agent}_setting_{setting}_run_{run}.tar".format(setting=args.setting, run=args.run, timestep=args.timestep, agent=full_aname, env=args.environment)        

    # read env/agent json
    with open('jsonfiles/environment/{}.json'.format(args.environment), 'r') as env_dat:
        env_json = json.load(env_dat, object_pairs_hook=OrderedDict)

    if args.from_results:
        j_aname = "forward_kl_maze" if args.agent == "FKL" else "reverse_kl_maze"
    else:
        if args.agent == "FKL":
            j_aname = "forward_kl_maze_image"
        else:
            raise NotImplementedError
    with open('jsonfiles/agent/{}.json'.format(j_aname), 'r') as agent_dat:
        agent_json = json.load(agent_dat, object_pairs_hook=OrderedDict)
    

    # initialize env
    wdir_name = "EasyWorkingDir" if args.environment == "EasyContinuousMaze" else "MediumWorkingDir"
    train_env = envs.create_environment(env_json, 'results/{env}results/{wdir}'.format(env=args.environment, wdir=wdir_name))
    test_env = envs.create_environment(env_json, 'results/{env}results/{wdir}'.format(env=args.environment,wdir = wdir_name))      

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
    # arg_params['random_seed'] = RANDOM_SEED
    # torch.manual_seed(RANDOM_SEED)    
    arg_params['random_seed'] = args.seed
    torch.manual_seed(args.seed)    
 
    # init config and merge custom config settings from json
    config = Config()
    config.merge_config(env_params)
    config.merge_config(agent_params)
    config.merge_config(arg_params)


    # initialize agent
    agent = create_agent(agent_json['agent'], config)

    # train_env.render = True
    

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
    i = 0
    while(True):
        i= i + 1
        obs_n, reward, done, info = train_env.step(Aold)
        oreturn += reward
        train_env
        if done == True:
            break
        Aold = agent.step(obs_n, True)  
        if i >= 1000:      
            train_env.instance.render(0.001, flagSave=False)
            i = 0
        # input('Return: {}. Press ENTER'.format(oreturn))
    train_env.instance.render(0, flagSave=False)

if __name__ == '__main__':
    main()


