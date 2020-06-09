import argparse
import numpy as np
import json
from collections import OrderedDict
import environments.environments as envs
from utils.config import Config
import quadpy
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm

# Usage
# python3 compute_targetBQ.py --save_dir ./SAVE/DIR

show_label = True
save_plot = True

env_name = 'ContinuousBanditsNormalized'

agent_params = {
    "entropy_scale": [0, 0.01, 0.1, 0.4, 1.0],
    "N_param": 1024
}

# Computed from compute_heatmap.py
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

}


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

    intgrl_actions = np.array(scheme.points[1:-1])
    intgrl_weights = np.array(scheme.weights[1:-1])
    intgrl_actions_len = np.shape(intgrl_actions)[0]

    print("== Compute target distribution")

    for t_idx, tau in enumerate(config.entropy_scale):

        start_run = datetime.now()
        print("--- tau = {} ::: {}".format(tau, start_run))

        xticks = np.arange(-2.0, 2.1, 1.0)
        yticks = [0.0, 1.0, 2.0, 3.0]


        plt.figure(figsize=(5.5, 4.5))

        plt.xlim([-2.0, 2.0])
        plt.ylim(ymin=0, ymax=3)

        # dirac delta distribution
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
            # Compute Boltzmann
            q_val = (env.reward_func(intgrl_actions)) / tau
            constant_shift = np.max(q_val, axis=-1)

            exp_q_val = np.exp(q_val - constant_shift)

            z = (exp_q_val * intgrl_weights).sum(-1)
            tiled_z = np.tile(z, [intgrl_actions_len])

            boltzmann_prob = exp_q_val / tiled_z
            plt.plot(np.arctanh(intgrl_actions), boltzmann_prob)

            # plot optimal distribution
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
