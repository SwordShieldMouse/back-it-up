import re
import os
import argparse
from parsers.main_parser import MainParser
from helper_scripts.debug_translate_setn_to_params import return_setting_and_run
import json

input_parser = argparse.ArgumentParser()
input_parser.add_argument("--path_to_command_file",type=str,default='helper_scripts/commands.txt')
input_parser.add_argument("--results_dir",type=str,default='results/_results')
input_args = input_parser.parse_args()

f = open(input_args.path_to_command_file, "r")
lines = f.readlines()

file_parser = MainParser()
get_fname = lambda full_fname : os.path.splitext(os.path.basename(full_fname))[0]


for line in lines:
    line = line.rstrip("\n")
    comm_patt = re.compile('(\\w|\\s)+\.py\\s+(?P<command_args>.+)')
    match = comm_patt.match(line)
    assert match is not None
    file_args = match.group('command_args')
    file_args = file_parser.parse_args(file_args.split())

    agent_args = json.load(open(file_args.agent_json,"r"))
    agent = agent_args["agent"]
    env = get_fname(file_args.env_json)
    index = file_args.index

    _, setting, run = return_setting_and_run(file_args.agent_json, index)

    results_subdir = os.path.join(input_args.results_dir, '{}'.format(env))
    fname = '{e}_{a}_setting_{s}_run_{r}_agent_Params.txt'.format(e=env, a=agent, s=setting, r=run )
    full_fname = os.path.join(results_subdir, fname)

    if os.path.isfile(full_fname):
        continue
    else:
        print(line)