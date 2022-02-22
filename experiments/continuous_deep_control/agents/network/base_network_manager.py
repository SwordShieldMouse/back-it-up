
class BaseNetwork_Manager(object):
    def __init__(self, config):

        self.random_seed = config.random_seed

        # Env config
        self.state_dim = config.state_dim
        self.state_min = config.state_min
        self.state_max = config.state_max

        self.action_dim = config.action_dim
        self.action_min = config.action_min
        self.action_max = config.action_max

        # record step for tf Summary
        self.train_global_steps = 0
        self.eval_global_steps = 0
        self.train_ep_count = 0
        self.eval_ep_count = 0

    def take_action(self, state, is_train, is_start):
        raise NotImplementedError

    def update_network(self, state_batch, action_batch, next_state_batch, reward_batch, gamma_batch):
        raise NotImplementedError

    def reset(self):

        self.train_ep_count = 0
        self.eval_ep_count = 0
