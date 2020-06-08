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
from scipy.stats import norm

show_label = False


save_plot = True

env_name = 'ContinuousBanditsNormalized'

# dummy agent, just using params from this json
agent_params = {

"entropy_scale": [0, 0.01, 0.1, 0.4, 1.0],
    # "entropy_scale": [0.01, 0.1, 1.0],
    "N_param": 1024
}

optimal_params = {
    "fkl":{
        0: (0.55, 0.008),
        0.01: (0.55, 0.011),
        0.1: (0.54, 0.1067),
        0.4: (0.23, 0.6052),
        1.0: (0.06, 0.8013)
    },

    "rkl":{
        0: (0.55, 0.008),
        0.01: (0.55, 0.011),
        0.1: (0.55, 0.0364),
        0.4: (0.55, 0.0925),
        1.0: (0.04, 0.7794)
    }

    # "fkl":{
    #     0.2: (0.44, 0.337),
    #     0.4: (0.23, 0.5784),
    #     0.6: (0.13, 0.6712),
    #     0.8: (0.08, 0.7163),
    #     1.0: (0.06, 0.7421)
    # },
    #
    # "rkl":{
    #     0.2: (0.55, 0.0549),
    #     0.4: (0.55, 0.0925),
    #     0.6: (0.07, 0.7265),
    #     0.8: (0.05, 0.758),
    #     1.0: (0.04, 0.774)
    # }

}

def compute_pi_logprob(mean_std_batch, action_arr):
    # mean_std_batch: (batch_size, 2)
    # action_arr: (batch_size, 1022)

    # (batch_size, )
    # using ln(1 + exp(param))
    logprob = np.array(
        [gauss(mean_std[0], np.log(1 + np.exp(mean_std[1]))).logpdf(np.arctanh(actions)) for mean_std, actions in
         zip(mean_std_batch, action_arr)])

    # term1 = np.array([dist[_].logpdf(np.arctanh(action_arr)) for _ in range(len(mean_std_batch))])
    logprob -= np.log(1 - action_arr ** 2)

    # logpdf = [dist.logpdf(math.atanh(a)) for a in arr]
    # logpdf -= np.log(1 - a**2 + epsilon).sum(dim=-1, keepdim=True)
    # pdf = [dist.logpdf(math.atanh(a)) - np.log(1 - a**2) for a in action_arr]

    # (batch_size, 1022)
    return logprob


def compute_plot(kl_type, entropy_arr, x_arr, y_arr, kl_arr, save_dir):
    # kl_arr = np.log(np.swapaxes(kl_arr, 1, 2))
    kl_arr = np.swapaxes(kl_arr, 1, 2)

    # applying std = log(1+exp(param))
    y_arr = list(np.log(1 + np.exp(np.array(y_arr))))

    # plot settings
    xticks = list(range(0, len(x_arr), 50)) + [len(x_arr) - 1]
    xticklabels = np.around(x_arr[::50] + [MEAN_MAX], decimals=2)

    # Plot only first and last ticks
    yticks = list(range(0, len(y_arr), 80))[:-1] + [len(y_arr) - 1]
    # yticks = [0, len(y_arr)-1]

    # applying std = log(1+exp(param))
    yticklabels = np.around(y_arr[::80][:-1] + [np.log(1 + np.exp(STD_MAX))], decimals=3)
    # yticklabels = np.around([y_arr[0]] + [np.log(1 + np.exp(STD_MAX))], decimals=3)

    if clip_kl_upper_bound:
        kl_arr = np.clip(kl_arr, -np.inf, KL_UPPER_LIMIT)

    # Plot heatmap per entropy per kl
    for t_idx, tau in enumerate(entropy_arr):

        ax = sns.heatmap(kl_arr[t_idx])

        best_idx = np.argmin(kl_arr[t_idx])
        best_mean_idx = int(best_idx / len(x_arr))
        best_std_idx = best_idx % len(x_arr)
        best_param = (x_arr[best_std_idx], y_arr[best_mean_idx])
        print("tau {} best param - mean: {}, std: {}".format(tau, round(best_param[0], 2), round(best_param[1], 2)))

        # highlight minimum point
        # ax.add_patch(Rectangle((best_std_idx, best_mean_idx), 1, 1, fill=False, edgecolor='blue', lw=1))

        ax.set_xticks(xticks)

        ax.set_yticks(yticks)

        if show_label:
            ax.set_xticklabels(xticklabels)
            ax.set_yticklabels(yticklabels)
            ax.set_title("{} KL Heatmap (truncated KL upper limit: {})\n best param - mean: {}, std: {}".format(kl_type,
                                                                                                                KL_UPPER_LIMIT if clip_kl_upper_bound else False,
                                                                                                                round(
                                                                                                                    best_param[
                                                                                                                        0],
                                                                                                                    4),
                                                                                                                round(
                                                                                                                    best_param[
                                                                                                                        1],
                                                                                                                    4)))

        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        plt.savefig('{}/{}_kl_{}_tau={}.png'.format(save_dir, kl_type, t_idx, tau))
        plt.clf()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str)

    args = parser.parse_args()

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

    ixs = np.argwhere((np.abs(scheme.points) <= 0.98))  # for numerical stability

    # intgrl_actions = np.array(scheme.points[1:-1])
    # intgrl_weights = np.array(scheme.weights[1:-1])

    intgrl_actions = np.squeeze(np.array(scheme.points[ixs]))
    intgrl_weights = np.squeeze(np.array(scheme.weights[ixs]))

    intgrl_actions_len = np.shape(intgrl_actions)[0]

    print("== Compute target dist")


    for t_idx, tau in enumerate(config.entropy_scale):

        start_run = datetime.now()
        print("--- tau = {} ::: {}".format(tau, start_run))

        xticks = np.arange(-2.0, 2.1, 1.0)
        yticks = [0.0, 1.0, 2.0, 3.0]


        plt.figure(figsize=(5.5, 4.5))

        plt.xlim([-2.0, 2.0])
        plt.ylim(ymin=0, ymax=3)


        if tau == 0:
            plt.axvline(x=np.arctanh(0.5))

            plt.plot(np.arctanh(intgrl_actions),
                     norm.pdf(np.arctanh(intgrl_actions), optimal_params['fkl'][tau][0], optimal_params['fkl'][tau][1]),
                     label='forward kl')
            plt.plot(np.arctanh(intgrl_actions),
                     norm.pdf(np.arctanh(intgrl_actions), optimal_params['rkl'][tau][0], optimal_params['rkl'][tau][1]),
                     label='reverse kl')

            if show_label:
                plt.legend()
                plt.title("target tanh Boltzmann dist. with tau: {}".format(tau))
                plt.xticks(xticks, xticks)
                plt.yticks(yticks, yticks)
            else:
                plt.xticks(xticks, [])
                plt.yticks(yticks, [])


            plt.savefig('{}/target_BQ_tau_{}.png'.format(args.save_dir, tau))
            plt.clf()

        else:
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
            # assert(np.shape(boltzmann_prob) == (1022, ))


            # plt.plot(intgrl_actions, boltzmann_prob)
            plt.plot(np.arctanh(intgrl_actions), boltzmann_prob)


            # plot optimal dist
            plt.plot(np.arctanh(intgrl_actions), norm.pdf(np.arctanh(intgrl_actions), optimal_params['fkl'][tau][0], optimal_params['fkl'][tau][1]), label='forward kl')
            plt.plot(np.arctanh(intgrl_actions),norm.pdf(np.arctanh(intgrl_actions), optimal_params['rkl'][tau][0], optimal_params['rkl'][tau][1]), label='reverse kl')

            if show_label:
                plt.legend()
                plt.title("target tanh Boltzmann dist. with tau: {}".format(tau))
                plt.xticks(xticks, xticks)
                plt.yticks(yticks, yticks)
            else:
                plt.xticks(xticks, [])
                plt.yticks(yticks, [])


            plt.savefig('{}/target_BQ_tau_{}.png'.format(args.save_dir, tau))
            plt.clf()


if __name__ == '__main__':
    main()
