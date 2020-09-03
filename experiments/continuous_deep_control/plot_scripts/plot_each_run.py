
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from plot_config import get_xyrange


# Needed
# BimodalEnv_NAF_setting_0_run_0_EpisodeRewardsLC


# Usage
# python3 ../plot_scripts/plot_each_run.py   DIR_RAW_RESULT(without / at the end)  ENV.json  NUM_RUNS AGENT_NAME  SETTING_NUM

show_plot = False
show_label = True


DIR = str(sys.argv[1])

env_filename = str(sys.argv[2])
with open(env_filename, 'r') as env_dat:
    env_json = json.load(env_dat)

ENV_NAME = env_json['environment']
# TOTAL_MIL_STEPS = env_json['TotalMilSteps']
# EVAL_INTERVAL_MIL_STEPS = env_json['EvalIntervalMilSteps']
# EVAL_EPISODES = env_json['EvalEpisodes']

NUM_RUNS = int(sys.argv[3])
AGENT_NAME = str(sys.argv[4])
SETTING_NUM = int(sys.argv[5])

custom_save_name = str(sys.argv[6])


#### Plot Settings #####
plt.figure(figsize=(12, 6))
#####

# Read each run

train_rewards_total_arr = []
train_action_total_arr = []
train_sigma_total_arr = []

xmax = None
for i in range(NUM_RUNS):

    # Filenames
    train_rewards_filename = DIR + '/' + ENV_NAME + '_' + AGENT_NAME + '_setting_' + str(SETTING_NUM) + '_run_' + str(i) + '_EpisodeRewardsLC.txt'

    train_rewards_arr = np.loadtxt(train_rewards_filename, delimiter=',')  # [:xmax]
    if not xmax:
        xmax = len(train_rewards_arr)
        print("assuming all training episodes have same length..")

    plt.plot(train_rewards_arr, color='b',alpha=0.1)
    train_rewards_total_arr.append(train_rewards_arr)

opt_range = range(0, xmax)
xlimt = (0, xmax-1)
_, ymin, ymax = get_xyrange(ENV_NAME)
ylimt = (ymin[0], ymax[0])
plt.xlim(xlimt)
plt.ylim(ylimt)

train_rewards_mean = np.nanmean(train_rewards_total_arr, axis=0)  # [:xmax]
plt.plot(opt_range, train_rewards_mean, color='b', linewidth=1.5)

if show_label:

    plt.title("{} \nEnv: {}, Agent: {} \n({} runs, setting: {})".format(custom_save_name, ENV_NAME, AGENT_NAME, NUM_RUNS, SETTING_NUM))
    plt.xlabel('Training Steps (per 1000 steps)')
    h = plt.ylabel("Cum. Reward per episode")
    h.set_rotation(90)


tick_interval = 50
# if show_label:
#     plt.xticks(opt_range[::50], np.linspace(0.0, float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax-1)), int(TOTAL_MIL_STEPS/EVAL_INTERVAL_MIL_STEPS)+1)[::50])
#     #plt.yticks([0,0.5,1,1.5], [0.0, 0.5, 1.0, 1.5])
# else:
#     plt.xticks(opt_range[::50], [])
#     plt.yticks([0,0.5,1,1.5], [])

if show_plot:
    plt.show()
else:
    plt.savefig("{}_{}_{}_runs.png".format(ENV_NAME, AGENT_NAME, custom_save_name))
plt.close()




