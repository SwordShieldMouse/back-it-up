from __future__ import print_function

import numpy as np

from agents.base_agent import BaseAgent
from agents.network.base_network_manager import BaseNetwork_Manager
from agents.network import forwardkl_network
from utils.plot_utils import plotFunction


class ForwardKL_Network_Manager(BaseNetwork_Manager):
    def __init__(self, config):
        super(ForwardKL_Network_Manager, self).__init__(config)
        self.rng = np.random.RandomState(config.random_seed)

        self.sample_for_eval = config.sample_for_eval
        self.use_true_q = config.use_true_q
        self.use_target = config.use_target

        # define network
        self.network = forwardkl_network.ForwardKLNetwork(config)

    def take_action(self, state, is_train, is_start):

        # Train
        if is_train:
            if is_start:
                self.train_ep_count += 1
            self.train_global_steps += 1

            # Get action from network
            chosen_action = self.network.sample_action(np.expand_dims(state, 0))[0]

            if self.write_plot:
                if self.use_true_q:
                    q_func = self.network.getTrueQFunction(state)
                else:
                    q_func = self.network.getQFunction(state)

                pi_func = self.network.getPolicyFunction(state)
                greedy_action = self.network.predict_action(np.expand_dims(state, 0))[0]

                plotFunction("KLDiv", [q_func, pi_func], state,
                                              greedy_action, chosen_action,
                                              self.action_min, self.action_max,
                                              display_title='Forward KL, steps: ' + str(self.train_global_steps),
                                              save_title='steps_' + str(self.train_global_steps),
                                              save_dir=self.writer.get_logdir(), ep_count=self.train_ep_count,
                                              show=False)

        # Eval
        else:
            if self.sample_for_eval:
                # sample action
                chosen_action = self.network.sample_action(np.expand_dims(state, 0))[0]
            else:
                # greedy action (mean)
                chosen_action = self.network.predict_action(np.expand_dims(state, 0))[0]

            if is_start:
                self.eval_ep_count += 1
            self.eval_global_steps += 1

        return chosen_action

    def update_network(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):

        # Policy Update, Qf and Vf Update
        _ = self.network.update_network(state_batch, action_batch, next_state_batch, reward_batch, gamma_batch)

        # Update target networks
        # if not using target network, tau=1.0 in base_network.py
        if self.use_target:
            self.network.update_target_network()


class ForwardKL(BaseAgent):
    def __init__(self, config):
        network_manager = ForwardKL_Network_Manager(config)
        super(ForwardKL, self).__init__(config, network_manager)





