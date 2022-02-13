from utils.main_utils import get_sweep_parameters
import json
import argparse
from collections import OrderedDict

def return_setting_and_run(agent_json, index):
    with open(agent_json, 'r') as agent_dat:
        agent_json = json.load(agent_dat, object_pairs_hook=OrderedDict)
    agent_params, total_num_sweeps = get_sweep_parameters(agent_json['sweeps'], index)
    setting = index % total_num_sweeps
    run = int(index / total_num_sweeps)
    return agent_params, setting, run

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('agent_json', type=str)
    parser.add_argument('index', type=int)
    args = parser.parse_args(args)

    agent_params, setting, run = return_setting_and_run(args.agent_json, args.index)
    print(agent_params)
    print("Setting: {}".format(setting))
    print("Run: {}".format(run))

if __name__ == '__main__':
    main()