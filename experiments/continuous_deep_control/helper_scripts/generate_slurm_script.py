import os
import pickle
import re

unimodal_nsettings = 125
gmm_nsettings = 375
##### Generate 5-seed sweep for FKL and RKL on EasyContinuousWorld and MultimodalContinuousWorld
for i in range(unimodal_nsettings * 5, unimodal_nsettings * 10):
    os.system("echo 'pipenv run python nonlinear_run.py --env_json jsonfiles/environment/EasyContinuousWorld.json --agent_json jsonfiles/agent/world/fkl.json --index {}' >> commands.txt".format(i))
    os.system("echo 'pipenv run python nonlinear_run.py --env_json jsonfiles/environment/EasyContinuousWorld.json --agent_json jsonfiles/agent/world/rkl.json --index {}' >> commands.txt".format(i))

    os.system("echo 'pipenv run python nonlinear_run.py --env_json jsonfiles/environment/MultimodalContinuousWorld.json --agent_json jsonfiles/agent/world/fkl.json --index {}' >> commands.txt".format(i))
    os.system("echo 'pipenv run python nonlinear_run.py --env_json jsonfiles/environment/MultimodalContinuousWorld.json --agent_json jsonfiles/agent/world/rkl.json --index {}' >> commands.txt".format(i))

##### Generate 5-seed sweep for FKL GMM and RKL GMM on EasyContinuousWorld and MultimodalContinuousWorld
for i in range(gmm_nsettings * 5, gmm_nsettings * 10):
    os.system("echo 'pipenv run python nonlinear_run.py --env_json jsonfiles/environment/EasyContinuousWorld.json --agent_json jsonfiles/agent/world/fkl_gmm.json --index {}' >> commands.txt".format(i))
    os.system("echo 'pipenv run python nonlinear_run.py --env_json jsonfiles/environment/EasyContinuousWorld.json --agent_json jsonfiles/agent/world/rkl_gmm.json --index {}' >> commands.txt".format(i))

    os.system("echo 'pipenv run python nonlinear_run.py --env_json jsonfiles/environment/MultimodalContinuousWorld.json --agent_json jsonfiles/agent/world/fkl_gmm.json --index {}' >> commands.txt".format(i))
    os.system("echo 'pipenv run python nonlinear_run.py --env_json jsonfiles/environment/MultimodalContinuousWorld.json --agent_json jsonfiles/agent/world/rkl_gmm.json --index {}' >> commands.txt".format(i))

###### Generate 30-seed sweep for FKL GMM, RKL GMM, FKL and RKL on EasyContinuousWorld and MultimodalContinuousWorld
# desired_temps = [10., 1., 0.1, 0.01]
# patt = re.compile("(?P<agent>\w+)_(?P<temp>.+)")
# envs = ['EasyContinuousWorld', 'MultimodalContinuousWorld']
# pkl_str = 'results/_info/{}/best_sett.pkl'
# tbr_settings = {}
# # Create dict with best settings for each temp
# for env in envs:
#     tbr_settings[env] = {}
#     pkl_f = pkl_str.format(env)
#     di = pickle.load(open(pkl_f, "rb"))
#     for plot_id, curve_dicts in di.items():
#         if 'RightExit' not in plot_id:
#             continue
#         for curve_id, sett in curve_dicts.items():
#             match = patt.match(curve_id)
#             agent = match.group('agent')
#             temp = float(match.group('temp'))
#             if temp in desired_temps:
#                 if agent not in tbr_settings[env]:
#                     tbr_settings[env][agent] = []
#                 tbr_settings[env][agent].append(sett)
#
# # Outputs commands for each seed
# ag_map = {  'ForwardKL': 'fkl',
#             'ReverseKL': 'rkl',
#             'GMMForwardKL': 'fkl_gmm',
#             'GMMReverseKL': 'rkl_gmm'
#           }
# for seed in range(30):
#     for env, agents_setts_dict in tbr_settings.items():
#         for agent, settings_list in agents_setts_dict.items():
#             for sett in settings_list:
#                 n_sett = unimodal_nsettings if 'GMM' not in agent else gmm_nsettings
#                 index = seed * n_sett + sett
#                 short_ag = ag_map[agent]
#                 os.system(
#                     "echo 'pipenv run python nonlinear_run.py --env_json jsonfiles/environment/{env}.json --agent_json jsonfiles/agent/world/{ag}.json --index {idx}' >> helper_scripts/commands.txt".format(
#                         ag=short_ag, env=env, idx=index))