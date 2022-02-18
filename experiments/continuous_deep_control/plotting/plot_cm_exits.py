from parsers.plot_parser import *
import os
import json
import csv
import re
import numpy as np
from collections import OrderedDict
import itertools
from aux_code import *
from config import *
from utils.main_utils import tryeval

def main(args=None):
    parser = CMPlotParser()
    args = parser.parse_args(args)
    with open(args.env_json_fname,"r") as f:
        env_params = json.load(f, object_pairs_hook=OrderedDict)

    manager = PlotManager(args.divide_type, args.how_to_group, env_params)

    for agent_name in args.agent_names:
        env_results_dir = os.path.join(args.results_dir, args.env_name)
        patt = re.compile("{e}_{a}_setting_(?P<setting>\d+)_run_(?P<run>\d+)_(?P<exit_type>BadExit|RightExit)\.txt".format(e=args.env_name, a=agent_name))

        filenames = filter_files_by_patt(os.listdir(env_results_dir), patt)
        full_filenames = [os.path.join(env_results_dir, f) for f in filenames]

        with open(os.path.join(env_results_dir, "{e}_{a}_agent_Params.json".format(e=args.env_name, a=agent_name))) as f:
            json_param_names = json.load(f, object_pairs_hook=OrderedDict)
            param_names = json_param_names["sweeps"].keys()

        for full_fname in full_filenames:
            match = patt.search(full_fname)
            setting = int(match.group('setting'))
            run = int(match.group('run'))
            exit_type = match.group('exit_type')

            param_values_filename = "{e}_{a}_setting_{s}_run_{r}_agent_Params.txt".format(e=args.env_name, a=agent_name, s=setting, r=run)
            with open(os.path.join(env_results_dir, param_values_filename), "r") as p_f:
                csv_reader = csv.reader(p_f)
                param_values = [tryeval(v) for v in next(iter(csv_reader))]
            params = dict(zip(param_names, param_values))

            if args.separate_agent_plots:
                key = {"agent_name": agent_name, "exit_type": exit_type}
            else:
                key = {"exit_type": exit_type}

            data = np.loadtxt(full_fname, delimiter=',')
            manager.add(key, agent_name, params, setting, data)

    manager.plot_and_save_all(args.plot_dir, synchronize_yaxis_on="exit_type", keep_ymin=True)

if __name__ == "__main__":
    main()
