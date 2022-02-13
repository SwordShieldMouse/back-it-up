from experiments.continuous_deep_control.utils.main_utils import get_sweep_parameters
import json
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('agent_json', type=str)
parser.add_argument('index', type=int)

args = parser.parse_args()

with open(args.agent_json, 'r') as agent_dat:
    agent_json = json.load(agent_dat, object_pairs_hook=OrderedDict)
agent_params, total_num_sweeps = get_sweep_parameters(agent_json['sweeps'], args.index)
print(agent_params)

print("Setting: {}".format(args.index % total_num_sweeps))
print("Run: {}".format(int(args.index / total_num_sweeps)))
