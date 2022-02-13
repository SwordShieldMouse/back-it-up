import argparse

class MainParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--env_json', type=str)
        self.parser.add_argument('--agent_json', type=str)
        self.parser.add_argument('--index', type=int)
        self.parser.add_argument('--save_last_net', default=False, action='store_true')
        self.parser.add_argument('--monitor', default=False, action='store_true')
        self.parser.add_argument('--render', default=False, action='store_true')
        self.parser.add_argument('--write_plot', default=False, action='store_true')
        self.parser.add_argument('--write_log', default=False, action='store_true')
        self.parser.add_argument('--out_dir', type=str, default="results/_results")
        self.parser.add_argument('--resume_training', action="store_true")
        self.parser.add_argument('--save_data_bdir', type=str, default="saved_nets")
        self.parser.add_argument('--save_data_minutes', type=float, default=10.)
        # ContinuousMaze arguments
        self.parser.add_argument('--steps_per_netsave', type=int, default=1000)
        self.parser.add_argument('--cm_netsave', action='store_true')

    def parse_args(self, args):
        return self.parser.parse_args(args)