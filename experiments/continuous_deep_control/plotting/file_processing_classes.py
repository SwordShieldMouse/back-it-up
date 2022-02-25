import numpy as np
from .config import *
import re
import os
import json
import csv
from collections import OrderedDict

def follows_patt(i_str, patt):
    if patt.match(i_str) is not None:
        return True
    else:
        return False

def filter_files_by_patt(files, patt):
    return list(filter(lambda s: follows_patt(s, patt), files))

def get_high_low(temp):
    if temp in BenchmarksPlotConfig.high_temps:
        return 'high'
    elif temp in BenchmarksPlotConfig.low_temps:
        return 'low'
    else:
        raise ValueError

class FileProcessing:
    def __init__(self, args):
        self.args = args
        self.setting = None
        self.run = None
        self.agent_name = None
        self.base_name = None
        self.ag_params = None
        self.full_fname = None
        with open(self.args.env_json_fname, "r") as f:
            self.env_params = json.load(f, object_pairs_hook=OrderedDict)

    def iterate_input_files(self, input_file_patt_f, input_file_regex_groups, get_plot_id_f):
        args = self.args
        for self.agent_name in args.agent_names:
            patt = input_file_patt_f(e=args.env_name, a=self.agent_name)
            filenames = filter_files_by_patt(os.listdir(args.env_results_dir), patt)
            full_filenames = [os.path.join(args.env_results_dir, f) for f in filenames]

            with open(os.path.join(args.env_results_dir, "{e}_{a}_agent_Params.json".format(e=args.env_name, a=self.agent_name))) as f:
                json_param_names = json.load(f, object_pairs_hook=OrderedDict)
                param_names = json_param_names["sweeps"].keys()

            for self.full_fname in full_filenames:
                match = patt.search(self.full_fname)

                for re_group in input_file_regex_groups:
                    setattr(self, re_group, tryeval(match.group(re_group)))

                self.base_name = "{e}_{a}_setting_{s}_run_{r}".format(e=args.env_name, a=self.agent_name, s=self.setting, r=self.run)
                param_values_filename = self.base_name + "_agent_Params.txt"

                with open(os.path.join(args.env_results_dir, param_values_filename), "r") as p_f:
                    csv_reader = csv.reader(p_f)
                    param_values = [tryeval(v) for v in next(iter(csv_reader)) if 'tensorflow' not in v]
                self.ag_params = dict(zip(param_names, param_values))
                self.ag_params['all_high_all_low'] = get_high_low(self.ag_params['entropy_scale'])

                plot_id = get_plot_id_f(self)
                self.sync_idx = plot_id.split("_").index(args.env_name)

                yield plot_id


    def get_unrolled_data(self, data, steps):
        max_steps = self.env_params['TotalMilSteps'] * 1e6
        x_axis_steps = self.env_params['XAxisSteps']
        current_episode = 0
        running_frame = 0
        n_idxs = int(max_steps/x_axis_steps)
        output = np.zeros([n_idxs])
        for global_idx in range(n_idxs):
            while running_frame >= steps[current_episode]:
                current_episode += 1
            output[global_idx] = np.mean(data[current_episode:current_episode+PlotConfig.episodes_window])
            running_frame += x_axis_steps
        return output

    def load_and_unroll_current_file(self):
        base_dir = self.args.env_results_dir
        base_name = self.base_name
        rewards_fname = base_name + "_EpisodeRewardsLC.txt"
        steps_fname = base_name + "_EpisodeStepsLC.txt"
        unrolled_fname = base_name + "_UnrolledEpisodeStepsLC.txt"
        full_rewards_fname = os.path.join(base_dir, rewards_fname)
        full_steps_fname = os.path.join(base_dir, steps_fname)
        full_unrolled_fname = os.path.join(base_dir, unrolled_fname)
        if os.path.isfile(full_unrolled_fname):
            return np.loadtxt(full_unrolled_fname, delimiter=',')
        else:
            data = np.loadtxt(full_rewards_fname, delimiter=',')
            steps = np.loadtxt(full_steps_fname, delimiter=',')
            data = self.get_unrolled_data(data, steps)
            data.tofile(full_unrolled_fname, sep=',')
            return data
