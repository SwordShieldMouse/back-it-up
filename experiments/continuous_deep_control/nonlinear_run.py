# -*- encoding:utf8 -*-
import gym
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
from parsers.main_parser import MainParser

from utils.main_utils import get_sweep_parameters, create_agent
#from torch.utils.tensorboard import SummaryWriter


def main(args=None):
    # parse arguments
    print(torch.__version__)
    parser = MainParser()
    args = parser.parse_args(args)

    arg_params = {
        "write_plot": args.write_plot,
        "write_log": args.write_log
    }


    # read env/agent json
    with open(args.env_json, 'r') as env_dat:
        env_json = json.load(env_dat, object_pairs_hook=OrderedDict)

    with open(args.agent_json, 'r') as agent_dat:
        agent_json = json.load(agent_dat, object_pairs_hook=OrderedDict)

    # create save directory
    save_dir = './{}/'.format(args.out_dir) + env_json['environment'] + 'results/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    if 'ContinuousMaze' in args.env_json or 'ContinuousWorld' in args.env_json:
        f_lock = LockFile(os.path.join(save_dir,'f_lock.lock'))
        with f_lock:
            if 'EasyContinuousMaze' in args.env_json:
                wd_name = 'EasyMazeWorkingDir'
            elif 'MediumContinuousMaze' in args.env_json:
                wd_name = 'MediumMazeWorkingDir'
            elif 'HardContinuousMaze' in args.env_json:
                wd_name = 'HardMazeWorkingDir'
            elif 'EasyContinuousWorld' in args.env_json:
                wd_name = 'EasyWorldWorkingDir'
            elif 'MultimodalContinuousWorld' in args.env_json:
                wd_name = 'MultimodalWorldWorkingDir'
            elif 'DebugContinuousWorld' in args.env_json:
                wd_name = 'DebugContinuousWorldWorkingDir'
            else:
                raise NotImplementedError
            working_dir = os.path.join("environments/classes/GM",wd_name)
            netsave_data_bdir = os.path.join(save_dir, 'saved_nets')
            if not os.path.exists(netsave_data_bdir):
                os.makedirs(netsave_data_bdir, exist_ok=True)                
    else:
        netsave_data_bdir = None
        

    # initialize env
    if 'ContinuousMaze' in args.env_json or 'ContinuousWorld' in args.env_json:
        env_json['working_dir'] = working_dir
    env_json['render'] = args.render
    train_env = envs.create_environment(env_json)
    test_env = envs.create_environment(env_json)

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
    arg_params['random_seed'] = RANDOM_SEED
    if 'ContinuousMaze' in args.env_json or 'ContinuousWorld' in args.env_json:
        torch.manual_seed(RANDOM_SEED)    

    # save/resume params and dirs
    save_data_fname = env_json['environment'] + '_'+agent_json['agent'] + '_setting_' + str(SETTING_NUM) + '_run_'+str(RUN_NUM) + '.tar'

    save_data_endname = 'END_' + env_json['environment'] + '_'+agent_json['agent'] + '_setting_' + str(SETTING_NUM) + '_run_'+str(RUN_NUM) + '.txt'

    save_data_full_endname = os.path.join(args.save_data_bdir, save_data_endname)

    if args.resume_training:
        if os.path.isfile(save_data_full_endname):
            exit()
        os.makedirs(args.save_data_bdir, exist_ok=True)    


    resume_params = {"resume_training": args.resume_training,
                     "save_data_bdir": args.save_data_bdir,
                     "save_data_minutes": args.save_data_minutes,
                     "save_data_fname": save_data_fname,
                     "steps_per_netsave": args.steps_per_netsave,
                     "no_netsave": not args.cm_netsave,
                     "netsave_data_bdir": netsave_data_bdir
                     }

    # create log directory (for tensorboard, gym monitor/render)
    START_DATETIME = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    log_dir = './{}/{}results/log_summary/{}/{}_{}_{}'.format(args.out_dir, str(env_json['environment']), str(agent_json['agent']), str(SETTING_NUM), str(RUN_NUM), str(START_DATETIME))

    # writer = tf.summary.FileWriter(log_dir)
    #writer = tf.compat.v1.summary.FileWriter(log_dir)
    # writer = SummaryWriter(log_dir)
    writer = None
    agent_params["writer"] = writer

    # init config and merge custom config settings from json
    config = Config()
    config.merge_config(env_params)
    config.merge_config(agent_params)
    config.merge_config(arg_params)

    # initialize agent
    agent = create_agent(agent_json['agent'], config)

    # monitor/render
    if args.monitor or args.render:
        if 'ContinuousMaze' in args.env_json or 'ContinuousWorld' in args.env_json or 'GridWorld' in args.env_json:
            if args.monitor:
                raise NotImplementedError('Recording not implemented')
        else:
            monitor_dir = log_dir+'/monitor'
            if args.render:
                train_env.instance = gym.wrappers.Monitor(train_env.instance, monitor_dir, video_callable=(lambda x: True), force=True)
            else:
                train_env.instance = gym.wrappers.Monitor(train_env.instance, monitor_dir, video_callable=False, force=True)

    # initialize experiment
    experiment = Experiment(agent=agent, train_environment=train_env, test_environment=test_env, seed=RANDOM_SEED,
                            writer=writer, write_log=args.write_log, write_plot=args.write_plot,
                            resume_params = resume_params)
    
    # run experiment
    try:
        episode_rewards, eval_episode_mean_rewards, eval_episode_std_rewards, train_episode_steps = experiment.run()
    except KeyboardInterrupt:
        exit()

    # save to file
    prefix = save_dir + env_json['environment'] + '_'+agent_json['agent'] + '_setting_' + str(SETTING_NUM) + '_run_'+str(RUN_NUM)

    train_rewards_filename = prefix + '_EpisodeRewardsLC.txt'
    np.array(episode_rewards).tofile(train_rewards_filename, sep=',', format='%15.8f')

    eval_mean_rewards_filename = prefix + '_EvalEpisodeMeanRewardsLC.txt'
    np.array(eval_episode_mean_rewards).tofile(eval_mean_rewards_filename, sep=',', format='%15.8f')

    eval_std_rewards_filename = prefix + '_EvalEpisodeStdRewardsLC.txt'
    np.array(eval_episode_std_rewards).tofile(eval_std_rewards_filename, sep=',', format='%15.8f')

    train_episode_steps_filename = prefix + '_EpisodeStepsLC.txt'
    np.array(train_episode_steps).tofile(train_episode_steps_filename, sep=',', format='%15.8f')

    if 'ContinuousMaze' in args.env_json or 'ContinuousWorld' in args.env_json:
        right_exit_count_filename = prefix + '_RightExit.txt'
        np.array(experiment.right_exit_global_count).tofile(right_exit_count_filename, sep=',')

        bad_exit_count_filename = prefix + '_BadExit.txt'
        np.array(experiment.bad_exit_global_count).tofile(bad_exit_count_filename, sep=',')

    params = []
    # params_names = '_'
    for key in agent_params:
        # for Python 2 since JSON load delivers "unicode" rather than pure string
        # then it will produce problem at plotting stage
        if isinstance(agent_params[key], type(u'')):
            params.append(agent_params[key].encode('utf-8'))
        else:
            params.append(agent_params[key])
        # params_names += (key + '_')

    params = np.array(params)
    # name = prefix + params_names + 'Params.txt'
    name = prefix + '_agent_' + 'Params.txt'
    params.tofile(name, sep=',', format='%s')

    # save json file as well
    # Bimodal1DEnv_uneq_var1_ActorCritic_agent_Params
    with open('{}{}_{}_agent_Params.json'.format(save_dir, env_json['environment'], agent_json['agent']), 'w') as json_save_file:
        json.dump(agent_json, json_save_file)

    # generate video and delete figures
    if args.write_plot:
        subprocess.run(["ffmpeg", "-framerate", "24", "-i", "{}/figures/steps_%01d.png".format(log_dir), "{}.mp4".format(log_dir)])
        # subprocess.run(["mv", "{}.mp4".format(log_dir), "{}/../".format(log_dir)])
        subprocess.run(["rm", "-rf", "{}/figures".format(log_dir)])

    if args.resume_training:
        os.system('touch {}'.format(save_data_full_endname))
        if not args.save_last_net:
            os.system('rm {}'.format(os.path.join(args.save_data_bdir, save_data_fname)))


if __name__ == '__main__':
    main()


