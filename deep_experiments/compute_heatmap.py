import argparse
import numpy as np
import json
from collections import OrderedDict
import environments.environments as envs
from utils.config import Config
import seaborn as sns
from itertools import product
import quadpy

from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
from torch.distributions import Normal
import torch.nn.functional as F

## Usage:

### Compute results and generate heatmap
# python3 compute_heatmap.py --load_results False --save_dir ./SAVE/DIR --compute_kl_type forward
#
### Load results and generate heatmap
# python3 compute_heatmap.py --load_results True --save_dir ./SAVE/DIR --compute_kl_type forward

## FLAGS
show_label = True
save_plot = True

compute_grad = False
compute_log_kl_loss = False

MEAN_MIN, MEAN_MAX = -2, 2
MEAN_INC = 0.01

STD_MIN, STD_MAX = 0.008, 0.9
STD_INC = 0.01

STD_PARAM_MIN, STD_PARAM_MAX = np.log(np.exp(STD_MIN)-1), np.log(np.exp(STD_MAX)-1)

action_scale = 1
env_name = 'ContinuousBanditsNormalized'

agent_params = {

    "entropy_scale": [0, 0.01, 0.1, 0.4, 1],
    "N_param": 1024
}

dtype = torch.float


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
        raise ValueError("Invalid --load_results value (True/False)")

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

    intgrl_actions = torch.tensor(np.array(scheme.points[1:-1])*action_scale)
    intgrl_weights = torch.tensor(np.array(scheme.weights[1:-1])*action_scale)
    intgrl_actions_len = len(intgrl_actions)

    mean_candidates = list(np.arange(MEAN_MIN, MEAN_MAX + MEAN_INC, MEAN_INC))
    std_candidates = list(np.arange(STD_PARAM_MIN, STD_PARAM_MAX + STD_INC, STD_INC))

    MEAN_NUM_POINTS = len(mean_candidates)
    STD_NUM_POINTS = len(std_candidates)

    all_candidates = list(product(mean_candidates, std_candidates))

    print("mean: {} points".format(MEAN_NUM_POINTS))
    print("std: {} points".format(STD_NUM_POINTS))
    print("Total combinations: {}".format(len(all_candidates)))

    kl_loss_arr = np.zeros((len(config.entropy_scale), MEAN_NUM_POINTS, STD_NUM_POINTS))
    kl_grad_arr = np.zeros((len(config.entropy_scale), MEAN_NUM_POINTS, STD_NUM_POINTS, 2))

    ## Forward KL
    if args.compute_kl_type == 'forward':
        print("== Forward KL ==")

        if not args.load_results:
            for t_idx, tau in enumerate(config.entropy_scale):

                start_run = datetime.now()
                print("--- tau = {} ::: {}".format(tau, start_run))

                tensor_all_candidates = torch.tensor(all_candidates, requires_grad=True)

                if tau == 0:
                    loss = hard_forward_kl_loss(tensor_all_candidates)

                else:
                    ### Compute Boltzmann
                    q_val = (env.reward_func(intgrl_actions)) / tau
                    constant_shift = torch.max(q_val, axis=-1)[0]
                    exp_q_val = torch.exp(q_val - constant_shift)

                    z = (exp_q_val * intgrl_weights).sum(-1)
                    tiled_z = z.repeat([intgrl_actions_len])
                    boltzmann_prob = exp_q_val / tiled_z

                    # Loop over possible mean, std
                    loss = forward_kl_loss(intgrl_weights, intgrl_actions, boltzmann_prob, tensor_all_candidates)

                if compute_grad:
                    torch.sum(loss).backward()
                    kl_grad_arr[t_idx] = np.reshape(tensor_all_candidates.grad.numpy(), (MEAN_NUM_POINTS, STD_NUM_POINTS, 2))
                    np.save('{}/forward_kl_grad_mean[{},{},{}]_std[{},{},{}]_N_{}_tau_{}.npy'.format(args.save_dir, MEAN_MIN,
                                                                                               MEAN_MAX, MEAN_INC, STD_MIN,
                                                                                               STD_MAX, STD_INC, config.N_param,
                                                                                               tau), kl_grad_arr[t_idx])

                kl_loss_arr[t_idx] = np.reshape(loss.detach().numpy(), (MEAN_NUM_POINTS, STD_NUM_POINTS))
                np.save('{}/forward_kl_mean[{},{},{}]_std[{},{},{}]_N_{}_tau_{}.npy'.format(args.save_dir, MEAN_MIN, MEAN_MAX, MEAN_INC,
                                                                                      STD_MIN, STD_MAX, STD_INC, config.N_param,
                                                                                      tau), kl_loss_arr[t_idx])

                end_run = datetime.now()
                print("Time taken: {}".format(end_run - start_run))

        # load computed results
        else:
            for t_idx, tau in enumerate(config.entropy_scale):
                kl_loss_arr[t_idx] = np.load('{}/forward_kl_mean[{},{},{}]_std[{},{},{}]_N_{}_tau_{}.npy'.format(args.save_dir, MEAN_MIN, MEAN_MAX, MEAN_INC, STD_MIN, STD_MAX, STD_INC, config.N_param, tau))

                if compute_grad:
                    kl_grad_arr[t_idx] = np.load('{}/forward_kl_grad_mean[{},{},{}]_std[{},{},{}]_N_{}_tau_{}.npy'.format(args.save_dir, MEAN_MIN, MEAN_MAX, MEAN_INC, STD_MIN, STD_MAX, STD_INC, config.N_param,tau))

        if save_plot:
            compute_plot(args.compute_kl_type, config.entropy_scale, mean_candidates, std_candidates, kl_loss_arr, kl_grad_arr, args.save_dir)

    ## Reverse KL
    if args.compute_kl_type == 'reverse':
        print("== Reverse KL ==")
        if not args.load_results:
            for t_idx, tau in enumerate(config.entropy_scale):

                start_run = datetime.now()
                print("--- tau = {} ::: {}".format(tau, start_run))

                tensor_all_candidates = torch.tensor(all_candidates, requires_grad=True)

                if tau == 0:
                    q_val = (env.reward_func(intgrl_actions))
                    loss = hard_reverse_kl_loss(intgrl_weights, intgrl_actions, q_val, tensor_all_candidates)
                else:

                    # Compute Boltzmann
                    q_val = (env.reward_func(intgrl_actions)) / tau
                    constant_shift = torch.max(q_val, axis=-1)[0]
                    exp_q_val = torch.exp(q_val - constant_shift)

                    z = (exp_q_val * intgrl_weights).sum(-1)
                    tiled_z = z.repeat([intgrl_actions_len])
                    boltzmann_prob = exp_q_val / tiled_z

                    loss = reverse_kl_loss(intgrl_weights, intgrl_actions, boltzmann_prob, tensor_all_candidates)

                if compute_grad:
                    torch.sum(loss).backward()
                    kl_grad_arr[t_idx] = np.reshape(tensor_all_candidates.grad.numpy(), (MEAN_NUM_POINTS, STD_NUM_POINTS, 2))
                    np.save('{}/forward_kl_grad_mean[{},{},{}]_std[{},{},{}]_N_{}_tau_{}.npy'.format(args.save_dir,
                                                                                                     MEAN_MIN,
                                                                                                     MEAN_MAX, MEAN_INC,
                                                                                                     STD_MIN,
                                                                                                     STD_MAX, STD_INC,
                                                                                                     config.N_param,
                                                                                                     tau), kl_grad_arr[t_idx])

                kl_loss_arr[t_idx] = np.reshape(loss.detach().numpy(), (MEAN_NUM_POINTS, STD_NUM_POINTS))
                np.save('{}/reverse_kl_mean[{},{},{}]_std[{},{},{}]_N_{}_tau_{}.npy'.format(args.save_dir, MEAN_MIN,
                                                                                            MEAN_MAX, MEAN_INC, STD_MIN,
                                                                                            STD_MAX, STD_INC,
                                                                                            config.N_param, tau),kl_loss_arr[t_idx])

                end_run = datetime.now()
                print("Time taken: {}".format(end_run - start_run))

        # load computed results
        else:
            for t_idx, tau in enumerate(config.entropy_scale):
                kl_loss_arr[t_idx] = np.load('{}/reverse_kl_mean[{},{}]_std[{},{}]_N_{}_tau_{}.npy'.format(args.save_dir, MEAN_MIN, MEAN_MAX, MEAN_INC, STD_MIN, STD_MAX, STD_INC, config.N_param, tau))

        if save_plot:
            compute_plot(args.compute_kl_type, config.entropy_scale, mean_candidates, std_candidates,
                         kl_loss_arr, kl_grad_arr, args.save_dir)


def compute_pi_logprob(mean_std_batch, action_arr):

    # mean_std_batch: (batch_size, 2)
    # action_arr: (batch_size, 1022)

    # (batch_size, )
    # using ln(1 + exp(param))
    permuted_mean_std_batch = mean_std_batch.permute(1, 0)  # ( 2, batch_size)

    permuted_action_arr = action_arr.permute(1, 0) if len(action_arr.shape) > 1 else action_arr

    logprob = Normal(permuted_mean_std_batch[0], F.softplus(permuted_mean_std_batch[1])).log_prob(custom_atanh(permuted_action_arr))
    logprob = logprob.permute(1,0) if len(action_arr.shape) > 1 else logprob

    logprob -= torch.log(1 - torch.pow(action_arr, 2))

    assert not torch.isnan(logprob).any()

    # (batch_size, actions_in_batch)
    return logprob

def custom_atanh(x):
    return (torch.log(1 + x) - torch.log(1 - x)) / 2

def hard_forward_kl_loss(mean_std_batch):

    # mean_std_batch: (MEAN_NUM_POINTS * STD_NUM_POINTS, 2)

    batch_size = len(mean_std_batch)
    optimal_action = 0.5
    tiled_actions = torch.tensor(np.tile(optimal_action, [batch_size,]), requires_grad=True)  # (1022, )

    # (batch_size, 1022)
    pi_logprob = compute_pi_logprob(mean_std_batch, tiled_actions)

    # (batch_size, 1022)
    loss = - pi_logprob

    return loss


def forward_kl_loss(weights, actions, boltzmann_p, mean_std_batch):

    # intgrl_weights: (1022, )
    # intgrL_actions: (1022, )
    # boltzmann_p: (1022, )
    # mu_std_batch: (MEAN_NUM_POINTS * STD_NUM_POINTS, 2)

    batch_size = len(mean_std_batch)

    tiled_weights = weights.repeat([batch_size, 1])
    tiled_actions = actions.repeat([batch_size, 1])

    # (batch_size, 1022)
    pi_logprob = compute_pi_logprob(mean_std_batch, tiled_actions)
    tiled_boltzmann_p = boltzmann_p.repeat([batch_size, 1])

    # (batch_size, 1022)
    integrands = tiled_boltzmann_p * (torch.log(tiled_boltzmann_p) - pi_logprob)
    loss = torch.sum(integrands * tiled_weights, -1)

    # (batch_size, )
    return loss

def hard_reverse_kl_loss(weights, actions, q_val, mean_std_batch):

    batch_size = len(mean_std_batch)
    tiled_weights = weights.repeat([batch_size, 1])
    tiled_actions = actions.repeat([batch_size, 1])

    tiled_q_val = q_val.repeat([batch_size, 1])

    # (batch_size, 1022)
    pi_logprob = compute_pi_logprob(mean_std_batch, tiled_actions)

    # without simplification
    integrands = -torch.exp(pi_logprob) * tiled_q_val

    assert (pi_logprob.shape == tiled_q_val.shape)
    assert (integrands.shape == tiled_weights.shape)
    loss = torch.sum(integrands * tiled_weights, -1)

    return loss

def reverse_kl_loss(weights, actions, boltzmann_p, mean_std_batch):

    batch_size = len(mean_std_batch)
    tiled_weights = weights.repeat([batch_size, 1])
    tiled_actions = actions.repeat([batch_size, 1])
    tiled_boltzmann_p = boltzmann_p.repeat([batch_size, 1])

    # (batch_size, 1022)
    pi_logprob = compute_pi_logprob(mean_std_batch, tiled_actions)

    # without simplification
    integrands = torch.exp(pi_logprob) * (pi_logprob - torch.log(tiled_boltzmann_p))

    assert (integrands.shape == tiled_weights.shape)
    loss = torch.sum(integrands * tiled_weights, -1)

    return loss


def compute_plot(kl_type, entropy_arr, x_arr, y_arr, kl_arr, grad_arr, save_dir):

    kl_arr = np.swapaxes(kl_arr, 1, 2)

    if compute_log_kl_loss:
        print("computing log kl loss")
        for t_idx, tau in enumerate(entropy_arr):

            min_val = np.min(kl_arr[t_idx])
            if min_val < 0:
                print("kl loss at tau={} is negative, so shifting it up..".format(tau))
                kl_arr[t_idx] -= min_val - 1e-6
        kl_arr = np.log(kl_arr)

    # applying std = log(1+exp(param))
    y_arr = list(np.log(1+np.exp(np.array(y_arr))))

    # plot settings
    xticks = list(range(0, len(x_arr), 50)) + [len(x_arr)-1]
    xticklabels = np.around(x_arr[::50] + [MEAN_MAX], decimals=2)

    # Plot only first and last ticks
    yticks = list(range(0, len(y_arr), 100))[:-1] + [len(y_arr)-1]

    # applying std = log(1+exp(param))
    yticklabels = np.around(y_arr[::100][:-1] + [np.log(1 + np.exp(STD_PARAM_MAX))], decimals=3)

    # Plot heatmap per entropy per kl
    for t_idx, tau in enumerate(entropy_arr):

        try:
            if kl_type == 'forward':
                ax = sns.heatmap(kl_arr[t_idx], vmax=100)
            else:
                ax = sns.heatmap(kl_arr[t_idx])
        except:
            print("computing kl loss failed. skipping")
            continue

        best_idx = np.argmin(kl_arr[t_idx])
        best_std_idx = int(best_idx/len(x_arr))
        best_mean_idx = best_idx%len(x_arr)

        best_param = (x_arr[best_mean_idx], y_arr[best_std_idx])
        print("tau {} best param - mean: {}, std: {}, loss: {}".format(tau, round(best_param[0], 4), round(best_param[1], 4), kl_arr[t_idx][best_std_idx][best_mean_idx]))

        # highlight minimum point
        ax.add_patch(Rectangle((best_mean_idx, best_std_idx), 1, 1, fill=False, edgecolor='blue', lw=1))

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        if show_label:
            ax.set_xticklabels(xticklabels)
            ax.set_yticklabels(yticklabels)
            ax.set_title("{} KL Heatmap)\n best param - mean: {}, std: {}".format(kl_type, round(best_param[0], 4), round(best_param[1],4)))

        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        plt.savefig('{}/{}_kl_{}_tau={}.png'.format(save_dir, kl_type, t_idx, tau))
        plt.clf()

        # compute vector gradient map
        if compute_grad:
            vector = -np.swapaxes(grad_arr[t_idx], 0, 2)

            X, Y = np.meshgrid(x_arr, y_arr)
            plt.quiver(X, Y, vector[0], -vector[1])
            plt.gca().invert_yaxis()

            plt.title("{} kl loss gradient, temp={}".format(kl_type, tau))
            plt.xlabel("mean")
            plt.ylabel("std")

            plt.savefig('{}/grad_{}_kl_{}_tau={}.png'.format(save_dir, kl_type, t_idx, tau))
            plt.clf()


if __name__ == '__main__':
    main()
