

import numpy as np
import json
from collections import OrderedDict
import environments.environments as envs
from utils.main_utils import get_sweep_parameters, create_agent
from utils.config import Config
import seaborn as sns
from itertools import product

from scipy.stats import norm
from datetime import datetime
import time

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# MEAN_NUM_POINTS = 200
# STD_NUM_POINTS = 200

INC = 0.01  # 0.01
MEAN_MIN, MEAN_MAX = -1, 1
STD_MIN, STD_MAX = 0.001, 1.2

KL_UPPER_LIMIT = 5
clip_kl_upper_bound = True

compute_forward = True
compute_reverse = True
save_plot = False
compute_custom = False

save_dir = './results/heatmaps'

agent_name = 'forward_kl'
agent_params = {

    "entropy_scale": [1.0, 0.1, 0.01], # 1.0, 0.5, 0.01
    "l_param": 6,
    "N_param": 1024,
    "optim_type": "intg",

    "actor_critic_dim": 200,
    "pi_lr": 0.0,
    "qf_vf_lr": 0.0,
    "sample_for_eval": "False",
    "use_true_q": "True",
    "q_update_type": "sac",
    "use_replay": False,
    "use_target": False,
    "batch_size": -1,

    "random_seed": 0,
    "write_plot": None,
    "writer": None,
    "write_log": None
}

def compute_pi_logprob(mean, std, action_arr):

    dist = norm(mean, std)

    result = []
    for arr in action_arr:
        result.append([dist.logpdf(a) for a in arr])
    return result


def forward_kl_loss(intgrl_weights, intgrl_actions, boltzmann_p, mu, std):
    pi_logprob = compute_pi_logprob(mu, std, intgrl_actions)  # (1, 62)

    # ignoring boltzmann entropy
    # integrands = -boltzmann_p * pi_logprob

    integrands = boltzmann_p * (np.log(boltzmann_p) - pi_logprob)
    loss = np.sum(integrands * intgrl_weights)

    return loss


def reverse_kl_loss(intgrl_weights, intgrl_actions, intgrl_q_val, mu, std, z):
    pi_logprob = np.array(compute_pi_logprob(mu, std, intgrl_actions))  # (1, 62)
    integrands = - np.exp(pi_logprob) * (intgrl_q_val - pi_logprob)
    loss = np.sum(integrands * intgrl_weights) + np.log(z)

    return loss


def main():
    env_json_path = './jsonfiles/environment/ContinuousBandits.json'
    agent_json_path = './jsonfiles/agent/{}.json'.format(agent_name)

    # read env/agent json
    with open(env_json_path, 'r') as env_dat:
        env_json = json.load(env_dat, object_pairs_hook=OrderedDict)

    with open(agent_json_path, 'r') as agent_dat:
        agent_json = json.load(agent_dat, object_pairs_hook=OrderedDict)

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

    # initialize agent
    agent = create_agent(agent_json['agent'], config)
    agent_network = agent.network_manager.network
    intgrl_actions = agent_network.intgrl_actions.numpy()
    intgrl_weights = agent_network.intgrl_weights.numpy()
    intgrl_actions_len = agent_network.intgrl_actions_len

    tiled_intgrl_actions = np.squeeze(agent_network.tiled_intgrl_actions.numpy(), -1)
    tiled_intgrl_weights = agent_network.tiled_intgrl_weights.numpy()

    mean_candidates = list(np.arange(MEAN_MIN, MEAN_MAX+INC, INC))
    std_candidates = list(np.arange(STD_MIN, STD_MAX+INC, INC))

    MEAN_NUM_POINTS = len(mean_candidates)
    STD_NUM_POINTS = len(std_candidates)

    all_candidates = list(product(mean_candidates, std_candidates))

    print("mean: {} points".format(MEAN_NUM_POINTS))
    print("std: {} points".format(STD_NUM_POINTS))
    print("Total combinations: {}".format(len(all_candidates)))

    forward_kl_arr = np.zeros((len(agent_network.entropy_scale), MEAN_NUM_POINTS, STD_NUM_POINTS))
    reverse_kl_arr = np.zeros((len(agent_network.entropy_scale), MEAN_NUM_POINTS, STD_NUM_POINTS))

    # plot settings
    xticks = range(0, len(std_candidates), 10)
    xticklabels = np.around(std_candidates[::10], decimals=2)
    yticks = range(0, len(mean_candidates), 10)
    yticklabels = np.around(mean_candidates[::10], decimals=2)

    ## Forward KL
    if compute_forward:
        print("== Forward KL ==")
        for t_idx, tau in enumerate(agent_network.entropy_scale):

            start_run = datetime.now()
            print("--- tau = {} ::: {}".format(tau, start_run))

            ### Compute Boltzmann
            # shape (1, 62)
            tiled_intgrl_q_val = (agent_network.predict_true_q(None, intgrl_actions)).reshape(-1, intgrl_actions_len) / tau

            constant_shift = np.amax(tiled_intgrl_q_val, axis=-1)[0]
            intgrl_exp_q_val = np.exp(tiled_intgrl_q_val - constant_shift)
            z = (intgrl_exp_q_val * agent_network.tiled_intgrl_weights.numpy()).sum(-1)
            tiled_z = np.repeat(np.expand_dims(z, -1), intgrl_actions_len, 1)
            boltzmann_prob = intgrl_exp_q_val / tiled_z

            # Loop over possible mean, std
            losses = np.array([forward_kl_loss(tiled_intgrl_weights, tiled_intgrl_actions, boltzmann_prob, mu, std) for (mu, std) in all_candidates])

            # best_idx = np.argmin(losses)
            # best_param = (mean_candidates[int(best_idx/STD_NUM_POINTS)], std_candidates[best_idx%STD_NUM_POINTS])
            # print("best param - mean: {}, std: {}".format(best_param[0], best_param[1]))

            forward_kl_arr[t_idx] = np.reshape(losses, (MEAN_NUM_POINTS, STD_NUM_POINTS))

            end_run = datetime.now()

            print("Time taken: {}".format(end_run - start_run))

            np.save('{}/forward_kl_mean[{},{}]_std[{},{}]_N_{}_tau_{}.npy'.format(save_dir, agent_network.N_param, MEAN_MIN, MEAN_MAX, STD_MIN, STD_MAX, tau), forward_kl_arr)


    ## Reverse KL
    if compute_reverse:
        print("== Reverse KL ==")
        for t_idx, tau in enumerate(agent_network.entropy_scale):

            start_run = datetime.now()
            print("--- tau = {} ::: {}".format(tau, start_run))

            tiled_intgrl_q_val = (agent_network.predict_true_q(None, intgrl_actions)).reshape(-1, intgrl_actions_len) / tau
            intgrl_exp_q_val = np.exp(tiled_intgrl_q_val)
            z = (intgrl_exp_q_val * agent_network.tiled_intgrl_weights.numpy()).sum(-1)

            # Loop over possible mean, std
            losses = np.array([reverse_kl_loss(tiled_intgrl_weights, tiled_intgrl_actions, tiled_intgrl_q_val, mu, std, z) for (mu, std) in all_candidates])

            # best_idx = np.argmin(losses)
            # best_param = (mean_candidates[int(best_idx / STD_NUM_POINTS)], std_candidates[best_idx % STD_NUM_POINTS])
            # print("best param - mean: {}, std: {}".format(best_param[0], best_param[1]))

            reverse_kl_arr[t_idx] = np.reshape(losses, (MEAN_NUM_POINTS, STD_NUM_POINTS))

            end_run = datetime.now()
            print("Time taken: {}".format(end_run - start_run))

            np.save('{}/reverse_kl_mean[{},{}]_std[{},{}]_N_{}_tau_{}.npy'.format(save_dir, agent_network.N_param, MEAN_MIN, MEAN_MAX, STD_MIN, STD_MAX, tau), reverse_kl_arr)

    if save_plot:
        if clip_kl_upper_bound:
            forward_kl_arr = np.clip(forward_kl_arr, -np.inf, KL_UPPER_LIMIT)
            reverse_kl_arr = np.clip(reverse_kl_arr, -np.inf, KL_UPPER_LIMIT)

        # Plot heatmap per entropy per kl
        for t_idx, tau in enumerate(agent_network.entropy_scale):

            # Forward KL
            if compute_forward:
                ax = sns.heatmap(forward_kl_arr[t_idx])

                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)

                plt.savefig('{}/forward_kl_{}_tau={}.png'.format(save_dir, t_idx, tau))
                plt.clf()

            # Reverse KL
            if compute_reverse:
                ax = sns.heatmap(reverse_kl_arr[t_idx])
                ax.set_xticks(xticks)
                ax.set_xticklabels(xticklabels)
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)

                plt.savefig('{}/reverse_kl_{}_tau={}.png'.format(save_dir, t_idx, tau))
                plt.clf()

    if compute_custom:

        forward_kl_arr = np.load('{}/forward_kl_201_points.npy'.format('results/heatmaps/3_201pts_heatmap_[-1,1]_[0.001,1.2]_N512'))
        reverse_kl_arr = np.load('./reverse_kl_201_points.npy')

        for t_idx, tau in enumerate(agent_network.entropy_scale):

            # print stats
            mean_idx, std_idx = np.unravel_index(forward_kl_arr[t_idx].argmin(), forward_kl_arr[t_idx].shape)
            mean, std = mean_candidates[mean_idx], std_candidates[std_idx]
            print("argmin forwardKL for tau {}: mean {}, std {}".format(tau, round(mean, 3), round(std, 3)))

            # mean_idx, std_idx = np.unravel_index(reverse_kl_arr[t_idx].argmin(), reverse_kl_arr[t_idx].shape)
            # mean, std = mean_candidates[mean_idx], std_candidates[std_idx]
            # print("argmin reverseKL for tau {}: mean {}, std {}".format(tau, round(mean, 3), round(std, 3)))


            # ax = sns.heatmap(np.clip(forward_kl_arr[t_idx], -np.inf, 5))
            # ax.set_xticks(xticks)
            # ax.set_xticklabels(xticklabels)
            # ax.set_yticks(yticks)
            # ax.set_yticklabels(yticklabels)
            #
            # ax.set(title="Forward KL heatmap",
            #        xlabel="std",
            #        ylabel="mean", )
            # plt.savefig('{}/forward_kl_{}_tau={}.png'.format(save_dir, t_idx, tau))
            # plt.clf()

            # ax = sns.heatmap(reverse_kl_arr[t_idx])
            # ax.set_xticks(xticks)
            # ax.set_xticklabels(xticklabels)
            # ax.set_yticks(yticks)
            # ax.set_yticklabels(yticklabels)
            #
            # ax.set(title="Reverse KL heatmap",
            #        xlabel="std",
            #        ylabel="mean", )
            #
            # plt.savefig('{}/reverse_kl_{}_tau={}.png'.format(save_dir, t_idx, tau))
            # plt.clf()

        #
        # mu = 1.0
        # std = 0.001
        # pi_logprob = np.array(compute_pi_logprob(mu, std, tiled_intgrl_actions))  # (1, 62)
        # integrands = - np.exp(pi_logprob) * (tiled_intgrl_q_val - tau * pi_logprob)
        # loss1 = np.sum(integrands * tiled_intgrl_weights)
        #
        # print('mu: {}, std:{} -- loss: {}'.format(mu, std, loss1))
        #
        # mu = 0.91457286
        # std = 0.001
        # pi_logprob = np.array(compute_pi_logprob(mu, std, tiled_intgrl_actions))  # (1, 62)
        # integrands = - np.exp(pi_logprob) * (tiled_intgrl_q_val - tau * pi_logprob)
        # loss2 = np.sum(integrands * tiled_intgrl_weights)
        #
        # print('mu: {}, std:{} -- loss: {}'.format(mu, std, loss2))
        #
        # mu = 0.99497487
        # std = 0.001
        # pi_logprob = np.array(compute_pi_logprob(mu, std, tiled_intgrl_actions))  # (1, 62)
        # integrands = - np.exp(pi_logprob) * (tiled_intgrl_q_val - tau * pi_logprob)
        # loss3 = np.sum(integrands * tiled_intgrl_weights)
        #
        # print('mu: {}, std:{} -- loss: {}'.format(mu, std, loss3))
        #

        # forward_kl_arr = []
        # reverse_kl_arr = []
        # tau = 0.01
        # std = 0.001
        #
        # tiled_intgrl_q_val = (agent_network.predict_true_q(None, intgrl_actions)).reshape(-1, intgrl_actions_len)
        # for mu in np.linspace(0.9, 1.1, 100):
        #     loss = reverse_kl_loss(tiled_intgrl_weights, tiled_intgrl_actions, tiled_intgrl_q_val, mu, std, tau)
        #     reverse_kl_arr.append(loss)
        #
        # plt.plot(np.linspace(0.9, 1.1, 100), reverse_kl_arr)
        # # plt.show()
        #
        # plt.savefig('reverse_kl_tau={}_mean_sensitivity.png'.format(tau))


if __name__ == '__main__':
    main()