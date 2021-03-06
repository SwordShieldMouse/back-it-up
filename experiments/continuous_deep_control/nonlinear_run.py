# -*- encoding:utf8 -*-
import gym
import tensorflow as tf
import environments.environments as envs
from utils.config import Config
from experiment import Experiment

import numpy as np
import json
import os
import datetime
from collections import OrderedDict
import argparse
import subprocess

from utils.main_utils import get_sweep_parameters, create_agent
#from torch.utils.tensorboard import SummaryWriter


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_json', type=str)
    parser.add_argument('--agent_json', type=str)
    parser.add_argument('--index', type=int)
    parser.add_argument('--monitor', default=False, action='store_true')
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--write_plot', default=False, action='store_true')
    parser.add_argument('--write_log', default=False, action='store_true')


    args = parser.parse_args()

    arg_params = {
        "write_plot": args.write_plot,
        "write_log": args.write_log
    }

    # read env/agent json
    with open(args.env_json, 'r') as env_dat:
        env_json = json.load(env_dat, object_pairs_hook=OrderedDict)

    with open(args.agent_json, 'r') as agent_dat:
        agent_json = json.load(agent_dat, object_pairs_hook=OrderedDict)

    # initialize env
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

    # create save directory
    save_dir = './results/' + env_json['environment'] + 'results/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # create log directory (for tensorboard, gym monitor/render)
    START_DATETIME = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    log_dir = './results/{}results/log_summary/{}/{}_{}_{}'.format(str(env_json['environment']), str(agent_json['agent']), str(SETTING_NUM), str(RUN_NUM), str(START_DATETIME))

    writer = tf.summary.FileWriter(log_dir)
    #writer = tf.compat.v1.summary.FileWriter(log_dir)
    # writer = SummaryWriter(log_dir)
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
        monitor_dir = log_dir+'/monitor'

        if args.render:
            train_env.instance = gym.wrappers.Monitor(train_env.instance, monitor_dir, video_callable=(lambda x: True), force=True)
        else:
            train_env.instance = gym.wrappers.Monitor(train_env.instance, monitor_dir, video_callable=False, force=True)

    # initialize experiment
    experiment = Experiment(agent=agent, train_environment=train_env, test_environment=test_env, seed=RANDOM_SEED,
                            writer=writer, write_log=args.write_log, write_plot=args.write_plot)
    
    # run experiment
    episode_rewards, eval_episode_mean_rewards, eval_episode_std_rewards, train_episode_steps = experiment.run()

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


if __name__ == '__main__':
    main()


