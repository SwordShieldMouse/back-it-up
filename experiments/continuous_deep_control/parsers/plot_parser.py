import argparse
import os

def _modify_arg(parser, which_arg, which_attr, new_v):
    for action in parser._actions:
        if action.dest == which_arg:
            setattr(action, which_attr, new_v)
            return
    else:
        raise AssertionError('argument {} not found'.format(which_arg))

class PlotParser:
    def __init__(self):
        self.cwd = os.getcwd()
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('env_name', type=str)
        self.parser.add_argument('--agent_names', type=str, nargs="+", choices=["ForwardKL","ReverseKL","GMMForwardKL","GMMReverseKL"], default=["ForwardKL","ReverseKL"])
        self.parser.add_argument('--root_dir', type=str, default=self.cwd, help="Base project directory (path to experiments/continuous_deep_control)")
        self.parser.add_argument('--how_to_group', type=str, default='best', choices=['best', 'top20'], help="How to group settings corresponding to the same point in the plotted curve")
        self.parser.add_argument('--results_root_dir', type=str, default=os.path.join(self.cwd, "results"), help="Where results are stored")
        self.parser.add_argument('--divide_type', type=str, default="entropy_scale", help="This parameter, together with the agent name, dictates the parameter which will be used to group runs")
        self.parser.add_argument('--hyperparam_for_sensitivity', type=str, default=None, help="X axis in the hyperparameter sensibility plots. Also dictates how runs will be grouped")
        self.parser.add_argument('--no_divide_type', action="store_true", help="Set to true to make divide_type equal to None")
        self.parser.add_argument('--separate_agent_plots', action="store_true", help="Whether to generate one plot for both agents or two separate plots")
        self.parser.add_argument('--config_class', type=str, default='PlotConfig', choices=['PlotConfig', 'BenchmarksPlotConfig','CMPlotConfig','BenchmarksBarPlotConfig','HyperSensPlotConfig'], help="Which class to use to get plot config (e.g linewidth, colors...)")
        self.parser.add_argument('--normalize', action="store_true", help="Whether to normalize Y axis")
        self.parser.add_argument('--bar', action="store_true", help="Set to true for the bar plots")
        self.parser.add_argument('--log_best_setting', action="store_true", help="Whether to log the best setting")

    def parse_args(self, args):
        args = self.parser.parse_args(args)
        # Where all json files are located
        args.json_dir = os.path.join(args.root_dir, 'jsonfiles')
        # Where all environment json files are located
        args.env_json_dir = os.path.join(args.json_dir, 'environment')
        # Where all agent json files are located
        args.agent_json_dir = os.path.join(args.json_dir, 'agent')
        # Json filename of current environment
        args.env_json_fname = os.path.join(args.env_json_dir, args.env_name + ".json")
        # Where raw results are stored
        args.results_dir = os.path.join(args.results_root_dir, '_results')
        # Where raw results for current env are stored
        args.plot_dir = os.path.join(args.results_root_dir, '_plots')
        # Where all plots are stored
        args.env_results_dir = os.path.join(args.results_dir, args.env_name)
        # Where desired info (e.g. best setting) is logged
        args.base_info_logdir = os.path.join(args.results_root_dir, "_info")
        args.info_logdir = os.path.join(args.base_info_logdir, args.env_name)
        if args.no_divide_type:
            args.divide_type = None
        return args


class CMPlotParser(PlotParser):
    def __init__(self):
        super().__init__()

    def parse_args(self, args):
        args = super().parse_args(args)
        # Where continuous maze plots are stored
        args.plot_dir = os.path.join(args.plot_dir, 'cm_plots')
        return args

class BenchmarksPlotParser(PlotParser):
    def __init__(self):
        super().__init__()

    def parse_args(self, args):
        args = super().parse_args(args)
        # Where benchmarks plots are stored
        args.plot_dir = os.path.join(args.plot_dir, 'benchmarks')
        # Where preprocessed files are stored (used to avoid repeated computation when unnecessary)
        args.preprocessed_dir = os.path.join(args.results_dir, args.env_name, 'preprocessed')
        if not os.path.isdir(args.preprocessed_dir):
            os.makedirs(args.preprocessed_dir, exist_ok=True)
        return args