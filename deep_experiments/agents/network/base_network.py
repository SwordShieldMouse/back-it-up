
class BaseNetwork(object):
    def __init__(self, config, learning_rate):
        """
        base network for actor and critic network.
        Args:
            sess: tf.Session()
            config: Configuration object
            learning_rate: learning rate for training (Could be an array if two-headed network)
        """

        # Env config
        self.state_dim = config.state_dim
        self.state_min = config.state_min
        self.state_max = config.state_max

        self.action_dim = config.action_dim
        self.action_min = config.action_min
        self.action_max = config.action_max

        self.learning_rate = learning_rate
        self.tau = config.tau


    def set_session(self, session):
        self.session = session

    def build_network(self, *args):
        """
        build network.
        """
        raise NotImplementedError("build network first!")

    def train(self, *args):
        raise NotImplementedError("train network!")

    def predict(self, *args):
        raise NotImplementedError("predict output for network!")

    def predict_target(self, *args):
        raise NotImplementedError("predict output for target network!")

    def update_target_network(self):
        raise NotImplementedError("update target network!")

    def get_num_trainable_vars(self):
        raise NotImplementedError("update target network!")



