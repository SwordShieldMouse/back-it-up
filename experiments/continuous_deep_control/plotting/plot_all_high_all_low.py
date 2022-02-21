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
    parser = BenchmarksPlotParser()
    args = parser.parse_args(args)
    manager = PlotManager(args)

    for agent_name in args.agent_names:
        patt = re.compile("{e}_{a}_setting_(?P<setting>\d+)_run_(?P<run>\d+)_EpisodeRewardsLC\.txt".format(e=args.env_name, a=agent_name))

        filenames = filter_files_by_patt(os.listdir(manager.env_results_dir), patt)
        full_filenames = [os.path.join(manager.env_results_dir, f) for f in filenames]

        with open(os.path.join(manager.env_results_dir, "{e}_{a}_agent_Params.json".format(e=args.env_name, a=agent_name))) as f:
            json_param_names = json.load(f, object_pairs_hook=OrderedDict)
            param_names = json_param_names["sweeps"].keys()

        for full_fname in full_filenames:
            match = patt.search(full_fname)
            setting = int(match.group('setting'))
            run = int(match.group('run'))

            base_name = "{e}_{a}_setting_{s}_run_{r}".format(e=args.env_name, a=agent_name, s=setting, r=run)
            param_values_filename = base_name + "_agent_Params.txt"

            with open(os.path.join(manager.env_results_dir, param_values_filename), "r") as p_f:
                csv_reader = csv.reader(p_f)
                param_values = [tryeval(v) for v in next(iter(csv_reader)) if 'tensorflow' not in v]
            ag_params = dict(zip(param_names, param_values))
            ag_params['all_high_all_low'] = get_high_low(ag_params['entropy_scale'])

            plot_id = args.env_name

            sync_idx = plot_id.split("_").index(args.env_name)
            if not manager.load_existing_data(plot_id):
                data = load_and_unroll_files(manager.env_results_dir, base_name, manager.env_params)
                manager.add(plot_id, agent_name, ag_params, setting, data)

    synchronize_yaxis_options = {
        "mode": "from_file",
        "target_call_id": "_".join(["SplitAgents", args.how_to_group, "entropy_scale"]),
        "sync_idx": sync_idx
    }
    manager.plot_and_save_all(synchronize_yaxis_options)
    manager.save_all_data() # To avoid processing again when plotting the next time

if __name__ == "__main__":
    main()
