import argparse
import numpy as np
import json
from collections import OrderedDict
import environments.environments as envs
from utils.main_utils import get_sweep_parameters, create_agent
from utils.config import Config
from agents import ForwardKL, ReverseKL
import seaborn as sns
from itertools import product
import quadpy

from scipy.stats import norm as gauss
from datetime import datetime
import matplotlib.pyplot as plt

import math

INC = 0.001  # 0.005 # 0.01
MEAN_MIN, MEAN_MAX = -2, 2  # -0.7, -0.4  # -0.8, 0.8

# Orig. STD_MIN, STD_MAX = 0.01, 0.8
STD_MIN, STD_MAX = np.log(np.exp(0.002)-1), np.log(np.exp(0.9)-1)  # -4.6, 0.203
STD_INC = 0.01  # 0.0124 #  0.0304

clip_kl_upper_bound = False
KL_UPPER_LIMIT = 50

save_plot = True

env_name = 'ContinuousBanditsNormalized'

# dummy agent, just using params from this json
agent_params = {

    "entropy_scale": [0.01, 0.1, 1.0],
    "N_param": 1024
}


def compute_pi_logprob(mean_std_batch, action_arr):

    # mean_std_batch: (batch_size, 2)
    # action_arr: (batch_size, 1022)

    # (batch_size, )
    # using ln(1 + exp(param))
    logprob = np.array([gauss(mean_std[0], np.log(1+np.exp(mean_std[1]))).logpdf(np.arctanh(actions)) for mean_std, actions in zip(mean_std_batch, action_arr)])

    # term1 = np.array([dist[_].logpdf(np.arctanh(action_arr)) for _ in range(len(mean_std_batch))])
    logprob -= np.log(1 - action_arr**2)

    # logpdf = [dist.logpdf(math.atanh(a)) for a in arr]
    # logpdf -= np.log(1 - a**2 + epsilon).sum(dim=-1, keepdim=True)
    # pdf = [dist.logpdf(math.atanh(a)) - np.log(1 - a**2) for a in action_arr]

    # (batch_size, 1022)
    return logprob


def hard_forward_kl_loss(weights, actions, boltzmann_p, q_val, z, mean_std_batch):

    # intgrl_weights: (1022, )
    # intgrL_actions: (1022, )
    # boltzmann_p: (1022, )
    # mu_std_batch: (MEAN_NUM_POINTS * STD_NUM_POINTS, 2)

    batch_size = len(mean_std_batch)

    tiled_weights = np.tile(weights, [batch_size, 1])
    tiled_actions = np.tile(actions, [batch_size, 1])

    tiled_boltzmann_p = np.tile(boltzmann_p, [batch_size, 1])

    tiled_q_val = np.tile(q_val, [batch_size, 1])
    tiled_z = np.tile(z, [batch_size, ])

    # (batch_size, 1022)
    pi_logprob = compute_pi_logprob(mean_std_batch, tiled_actions)

    ### without simplification
    # computing full kl loss
    assert (np.shape(tiled_boltzmann_p) == np.shape(pi_logprob) == np.shape(tiled_weights))

    # (batch_size, 1022)
    integrands = tiled_boltzmann_p * (np.log(tiled_boltzmann_p) - pi_logprob)
    loss = np.sum(integrands * tiled_weights, -1)

    return loss


def forward_kl_loss(weights, actions, boltzmann_p, q_val, z, mean_std_batch):

    # intgrl_weights: (1022, )
    # intgrL_actions: (1022, )
    # boltzmann_p: (1022, )
    # mu_std_batch: (MEAN_NUM_POINTS * STD_NUM_POINTS, 2)

    batch_size = len(mean_std_batch)

    tiled_weights = np.tile(weights, [batch_size, 1])
    tiled_actions = np.tile(actions, [batch_size, 1])

    tiled_boltzmann_p = np.tile(boltzmann_p, [batch_size, 1])

    # tiled_q_val = np.tile(q_val, [batch_size, 1])
    # tiled_z = np.tile(z, [batch_size, ])

    # (batch_size, 1022)
    pi_logprob = compute_pi_logprob(mean_std_batch, tiled_actions)

    ### without simplification
    # computing full kl loss
    assert (np.shape(tiled_boltzmann_p) == np.shape(pi_logprob) == np.shape(tiled_weights))

    # (batch_size, 1022)
    integrands = tiled_boltzmann_p * (np.log(tiled_boltzmann_p) - pi_logprob)
    loss = np.sum(integrands * tiled_weights, -1)

    # del tiled_weights
    # del tiled_actions
    # del tiled_boltzmann_p
    # del pi_logprob
    # del integrands

    # (batch_size, )
    return loss

    # ### with simplification
    #
    # integrands = tiled_boltzmann_p * (tiled_q_val - pi_logprob)
    #
    # loss = np.sum(integrands * tiled_weights, -1)
    # loss -= np.log(tiled_z)
    #
    # return loss


def reverse_kl_loss(weights, actions, boltzmann_p, q_val, z, mean_std_batch):

    batch_size = len(mean_std_batch)
    tiled_weights = np.tile(weights, [batch_size, 1])
    tiled_actions = np.tile(actions, [batch_size, 1])
    tiled_boltzmann_p = np.tile(boltzmann_p, [batch_size, 1])
    # tiled_q_val = np.tile(q_val, [batch_size, 1])
    # tiled_z = np.tile(z, [batch_size, ])

    # (batch_size, 1022)
    pi_logprob = compute_pi_logprob(mean_std_batch, tiled_actions)

    # ### without simplification
    integrands = np.exp(pi_logprob) * (pi_logprob - np.log(tiled_boltzmann_p))

    assert (np.shape(integrands) == np.shape(tiled_weights))
    loss = np.sum(integrands * tiled_weights, -1)

    # del pi_logprob
    # del integrands
    # del tiled_weights
    # del tiled_actions
    # del tiled_boltzmann_p

    return loss

    ### with simplification
    # integrands = - np.exp(pi_logprob) * (tiled_q_val - pi_logprob)
    #
    # loss = np.sum(integrands * tiled_weights, -1)
    # loss += np.log(tiled_z)

    # del tiled_actions
    # del tiled_q_val
    # del pi_logprob
    # del tiled_z
    # del tiled_weights
    # del integrands

    # return loss


def compute_plot(kl_type, entropy_arr, x_arr, y_arr, kl_arr, save_dir):

    # kl_arr = np.log(np.swapaxes(kl_arr, 1, 2))
    kl_arr = np.swapaxes(kl_arr, 1, 2)

    # applying std = log(1+exp(param))
    y_arr = list(np.log(1+np.exp(np.array(y_arr))))

    # plot settings
    xticks = list(range(0, len(x_arr), 40)) + [len(x_arr)-1]
    xticklabels = np.around(x_arr[::40] + [MEAN_MAX], decimals=2)
    yticks = list(range(0, len(y_arr), 40)) + [len(y_arr)-1]

    # applying std = log(1+exp(param))
    yticklabels = np.around(y_arr[::40] + [np.log(1+np.exp(STD_MAX))], decimals=4)

    if clip_kl_upper_bound:
        kl_arr = np.clip(kl_arr, -np.inf, KL_UPPER_LIMIT)

    # Plot heatmap per entropy per kl
    for t_idx, tau in enumerate(entropy_arr):

        ax = sns.heatmap(kl_arr[t_idx])

        best_idx = np.argmin(kl_arr[t_idx])
        best_mean_idx = int(best_idx/len(x_arr))
        best_std_idx = best_idx%len(x_arr)
        best_param = (x_arr[best_std_idx], y_arr[best_mean_idx])
        print("tau {} best param - mean: {}, std: {}".format(tau, round(best_param[0], 2), round(best_param[1], 2)))

        # highlight minimum point
        # ax.add_patch(Rectangle((best_std_idx, best_mean_idx), 1, 1, fill=False, edgecolor='blue', lw=1))

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

        ax.set_title("{} KL Heatmap (truncated KL upper limit: {})\n best param - mean: {}, std: {}".format(kl_type, KL_UPPER_LIMIT if clip_kl_upper_bound else False, round(best_param[0], 4), round(best_param[1],4)))

        plt.savefig('{}/{}_kl_{}_tau={}.png'.format(save_dir, kl_type, t_idx, tau))
        plt.clf()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_kl_type', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--load_results', type=str)

    args = parser.parse_args()

    if args.load_results == 'True':
        args.load_results = True
    elif args.load_results == 'False':
        args.load_results = False
    else: 
        raise ValueError("Invalid --load_results value")

    env_json_path = './jsonfiles/environment/{}.json'.format(env_name)

    # read env/agent json
    with open(env_json_path, 'r') as env_dat:
        env_json = json.load(env_dat, object_pairs_hook=OrderedDict)


    # initialize env
    env = envs.create_environment(env_json)

    # Create env_params for agent
    env_params = {
        "env_name": env.name,
        "state_dim": env.state_dim,
        "state_min": env.state_min,
        "state_max": env.state_max,

        "action_dim": env.action_dim,
        "action_min": env.action_min,
        "action_max": env.action_max
    }

    config = Config()
    config.merge_config(env_params)
    config.merge_config(agent_params)

    # initialize kl params

    scheme = quadpy.line_segment.clenshaw_curtis(config.N_param)

    # agent_network = agent.network_manager.network

    intgrl_actions = np.array(scheme.points[1:-1])
    intgrl_weights = np.array(scheme.weights[1:-1])
    intgrl_actions_len = np.shape(intgrl_actions)[0]

    mean_candidates = list(np.arange(MEAN_MIN, MEAN_MAX+INC, INC))
    std_candidates = list(np.arange(STD_MIN, STD_MAX+STD_INC, STD_INC))

    MEAN_NUM_POINTS = len(mean_candidates)
    STD_NUM_POINTS = len(std_candidates)

    all_candidates = list(product(mean_candidates, std_candidates))

    print("mean: {} points".format(MEAN_NUM_POINTS))
    print("std: {} points".format(STD_NUM_POINTS))
    print("Total combinations: {}".format(len(all_candidates)))

    # batch_size = len(all_candidates)
    # tiled_intgrl_actions = np.tile(intgrl_actions, [batch_size, 1])
    # tiled_intgrl_weights = np.tile(intgrl_weights, [batch_size, 1])

    # assert(np.shape(tiled_intgrl_actions) == (batch_size, 1022))
    # assert (np.shape(tiled_intgrl_weights) == (batch_size, 1022))

    ## Forward KL
    if args.compute_kl_type == 'forward':
        print("== Forward KL ==")

        forward_kl_arr = np.zeros((len(config.entropy_scale), MEAN_NUM_POINTS, STD_NUM_POINTS))

        if not args.load_results:
            for t_idx, tau in enumerate(config.entropy_scale):

                start_run = datetime.now()
                print("--- tau = {} ::: {}".format(tau, start_run))

                ### Compute Boltzmann

                # (1022,)
                q_val = (env.reward_func(intgrl_actions)) / tau

                constant_shift = np.max(q_val, axis=-1)

                # (1022, )
                exp_q_val = np.exp(q_val - constant_shift)

                unshifted_z = (np.exp(q_val) * intgrl_weights).sum(-1)

                # (1,)
                z = (exp_q_val * intgrl_weights).sum(-1)

                # (1022, )
                tiled_z = np.tile(z, [intgrl_actions_len])

                # (1022, )
                boltzmann_prob = exp_q_val / tiled_z
                assert(np.shape(boltzmann_prob) == (1022, ))

                del tiled_z
                del constant_shift
                del q_val
                del exp_q_val
                del z

                # Loop over possible mean, std
                losses = forward_kl_loss(intgrl_weights, intgrl_actions, boltzmann_prob, None, None, all_candidates)
                # losses = forward_kl_loss(intgrl_weights, intgrl_actions, boltzmann_prob, q_val, unshifted_z, all_candidates)

                forward_kl_arr[t_idx] = np.reshape(losses, (MEAN_NUM_POINTS, STD_NUM_POINTS))

                del boltzmann_prob

                end_run = datetime.now()

                print("Time taken: {}".format(end_run - start_run))

                np.save('{}/forward_kl_mean[{},{}]_std[{},{}]_N_{}_tau_{}.npy'.format(args.save_dir, MEAN_MIN, MEAN_MAX, STD_MIN, STD_MAX, config.N_param, tau), forward_kl_arr[t_idx])

        else:
            for t_idx, tau in enumerate(config.entropy_scale):
                forward_kl_arr[t_idx] = np.load('{}/forward_kl_mean[{},{}]_std[{},{}]_N_{}_tau_{}.npy'.format(args.save_dir, MEAN_MIN, MEAN_MAX, STD_MIN, STD_MAX, config.N_param, tau))

        if save_plot:
            compute_plot(args.compute_kl_type, config.entropy_scale, mean_candidates, std_candidates, forward_kl_arr, args.save_dir)

    ## Reverse KL
    if args.compute_kl_type == 'reverse':
        print("== Reverse KL ==")
        reverse_kl_arr = np.zeros((len(config.entropy_scale), MEAN_NUM_POINTS, STD_NUM_POINTS))

        if not args.load_results:
            for t_idx, tau in enumerate(config.entropy_scale):

                start_run = datetime.now()
                print("--- tau = {} ::: {}".format(tau, start_run))

                # (1022, )
                q_val = (env.reward_func(intgrl_actions)) / tau

                constant_shift = np.max(q_val, axis=-1)

                exp_q_val = np.exp(q_val - constant_shift)

                # (1, )
                # unshifted_z = (np.exp(q_val) * intgrl_weights).sum(-1)

                z = (exp_q_val * intgrl_weights).sum(-1)
                tiled_z = np.tile(z, [intgrl_actions_len])
                boltzmann_prob = exp_q_val / tiled_z

                del tiled_z
                del constant_shift
                del q_val
                del exp_q_val
                del z

                losses = reverse_kl_loss(intgrl_weights, intgrl_actions, boltzmann_prob, None, None, all_candidates)
                # losses = reverse_kl_loss(intgrl_weights, intgrl_actions, boltzmann_prob, q_val, unshifted_z, all_candidates)

                reverse_kl_arr[t_idx] = np.reshape(losses, (MEAN_NUM_POINTS, STD_NUM_POINTS))

                del boltzmann_prob

                end_run = datetime.now()
                print("Time taken: {}".format(end_run - start_run))

                np.save('{}/reverse_kl_mean[{},{}]_std[{},{}]_N_{}_tau_{}.npy'.format(args.save_dir, MEAN_MIN, MEAN_MAX, STD_MIN, STD_MAX, config.N_param, tau), reverse_kl_arr[t_idx])

        else:
            for t_idx, tau in enumerate(config.entropy_scale):
                reverse_kl_arr[t_idx] = np.load('{}/reverse_kl_mean[{},{}]_std[{},{}]_N_{}_tau_{}.npy'.format(args.save_dir, MEAN_MIN, MEAN_MAX, STD_MIN, STD_MAX, config.N_param, tau))

        if save_plot:
            compute_plot(args.compute_kl_type, config.entropy_scale, mean_candidates, std_candidates,
                         reverse_kl_arr, args.save_dir)


if __name__ == '__main__':
    main()
