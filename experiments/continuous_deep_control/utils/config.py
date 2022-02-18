class Config:
    # default setting
    def __init__(self):

        self.warmup_steps = 0
        self.buffer_size = 1e6

        self.tau = 0.01
        self.gamma = 0.99

    # add custom setting
    def merge_config(self, custom_config):

        for key in custom_config.keys():
            setattr(self, key, custom_config[key])

