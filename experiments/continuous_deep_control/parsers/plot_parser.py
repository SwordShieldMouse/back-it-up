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
        self.parser.add_argument('--agent_names', type=str, nargs="+", choices=["ForwardKL","ReverseKL"], default=["ForwardKL","ReverseKL"])
        self.parser.add_argument('--root_dir', type=str, default=self.cwd)
        self.parser.add_argument('--parse_type', type=str, default=None)
        self.parser.add_argument('--how_to_group', type=str, default='best', choices=['best', 'top20'])
        self.parser.add_argument('--results_root_dir', type=str, default=os.path.join(self.cwd, "results"))
        self.parser.add_argument('--divide_type', type=str, default="entropy_scale")
        self.parser.add_argument('--hyperparam_for_sensitivity', type=str, default=None)
        self.parser.add_argument('--no_divide_type', action="store_true")
        self.parser.add_argument('--separate_agent_plots', action="store_true")
        self.parser.add_argument('--config_class', type=str, default='PlotConfig', choices=['PlotConfig', 'BenchmarksPlotConfig','CMPlotConfig','BenchmarksBarPlotConfig','HyperSensPlotConfig'])
        self.parser.add_argument('--normalize', action="store_true")
        self.parser.add_argument('--bar', action="store_true")

    def parse_args(self, args):
        args = self.parser.parse_args(args)
        args.json_dir = os.path.join(args.root_dir, 'jsonfiles')
        args.env_json_dir = os.path.join(args.json_dir, 'environment')
        args.agent_json_dir = os.path.join(args.json_dir, 'agent')
        args.env_json_fname = os.path.join(args.env_json_dir, args.env_name + ".json")
        args.results_dir = os.path.join(args.results_root_dir, '_results')
        args.plot_dir = os.path.join(args.results_root_dir, '_plots')
        args.env_results_dir = os.path.join(args.results_dir, args.env_name)
        if args.no_divide_type:
            args.divide_type = None
        return args


class CMPlotParser(PlotParser):
    def __init__(self):
        super().__init__()

    def parse_args(self, args):
        args = super().parse_args(args)
        args.plot_dir = os.path.join(args.plot_dir, 'cm_plots')
        return args

class BenchmarksPlotParser(PlotParser):
    def __init__(self):
        super().__init__()

    def parse_args(self, args):
        args = super().parse_args(args)
        args.plot_dir = os.path.join(args.plot_dir, 'benchmarks')
        args.preprocessed_dir = os.path.join(args.results_dir, args.env_name, 'preprocessed')
        if not os.path.isdir(args.preprocessed_dir):
            os.makedirs(args.preprocessed_dir, exist_ok=True)
        return args