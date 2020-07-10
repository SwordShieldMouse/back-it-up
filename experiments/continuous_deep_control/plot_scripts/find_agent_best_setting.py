from collections import OrderedDict
import matplotlib.pyplot as plt

import numpy as np
import glob
import sys
import json

from utils import get_agent_parse_info

import os

# Usage:
# python3 find_agent_best_setting.py $RESULT_DIR $ROOT_LOC $ENV_NAME $AGENT_NAME $NUM_RUNS $CUSTOM_NAME $PARSE_TYPE
#
# generates plots and npy for the best settings according to $PARSE_TYPE
#
# RESULT_DIR : where merged{$ENV_NAME}results is located
# ROOT_LOC : root directory of code (where nonlinear_run.py and experiment.py is located)
# CUSTOM_NAME : name of th experiment (to differentiate different sweeps)
# PARSE_TYPE : type of params to parse. If 'entropy_scale', the script will compute best settings for each param value of entropy_scale

### CONFIG BEFORE RUNNING ###
show_plot = False
plot_each_runs = True
eval_last_N = True
last_N_ratio = 0.5
##############################


def get_xyrange(envname):
    xmax = None

    if envname == 'ContinuousBandits':
        ymin = [-0.5, -0.5]
        ymax = [2.0, 2.0]

    elif envname == 'Pendulum-v0':
        ymin = [-1600, -1600]
        ymax = [-100, -100]

    elif envname == 'Reacher-v2':
        ymin = [-80, -80]
        ymax = [0, 0]

    elif envname == "Swimmer-v2":
        ymin = [0, 0]
        ymax = [40, 40]

    else:
        raise ValueError("Invalid environment name")

    if xmax is None:
        return None, ymin, ymax
    else:
        return xmax, ymin, ymax


if __name__ == "__main__":

    if len(sys.argv) != 8:
        raise ValueError('Invalid input. \nCorrect Usage: find_agent_best_setting.py merged_result_loc, root_dir, env_name, agent_name, num_runs custom_save_name parse_type')

    root_dir = str(sys.argv[2])
    env_name = str(sys.argv[3])
    agent_name = str(sys.argv[4])

    merged_result_dir = '{}/merged{}results/'.format(str(sys.argv[1]), env_name)
    env_json_dir = '{}/jsonfiles/environment/{}.json'.format(root_dir, env_name)

    num_runs = int(sys.argv[5])

    custom_save_name = str(sys.argv[6])
    parse_type = str(sys.argv[7])

    with open(env_json_dir, 'r') as env_dat:
        env_json = json.load(env_dat, object_pairs_hook=OrderedDict)

    env_name = env_json['environment']

    # read agent json
    agent_jsonfile = '{}_{}_agent_Params.json'.format(env_name, agent_name)

    with open(merged_result_dir + agent_jsonfile, 'r') as agent_dat:
        agent_json = json.load(agent_dat, object_pairs_hook=OrderedDict)

    type_arr, pre_divider, num_type, post_divider, num_settings = get_agent_parse_info(agent_json, divide_type=parse_type)

    # To save npy files separately
    if not os.path.exists(merged_result_dir + '/npy'):
        os.makedirs(merged_result_dir + '/npy')

    type_idx_arr = []

    for i in range(num_type):
        arr = []

        for j in range(i*pre_divider, num_settings, pre_divider * num_type):
            for k in range(j, j+pre_divider, 1):
                arr.append(k)

        type_idx_arr.append(arr)

    print('Environment: ' + env_name)
    print('Agent: ' + agent_name)

    # ENV specific setting
    TOTAL_MIL_STEPS = env_json['TotalMilSteps']
    EVAL_INTERVAL_MIL_STEPS = env_json['EvalIntervalMilSteps']
    EVAL_EPISODES = env_json['EvalEpisodes']

     # Plot type
    # Disabled Evaluation
    result_type = ['TrainEpisode']

    title = "%s, %s: %s (%d runs)" % (env_name, agent_name, custom_save_name, num_runs)

    for result_idx, result in enumerate(result_type):

        plt.figure(figsize=(12, 6))
        plt.title(title)

        lcfilename = merged_result_dir + env_name + '_' + agent_name + '_' + result + 'MeanRewardsLC.txt'
        print('Reading lcfilename.. ' + lcfilename)
        lc = np.loadtxt(lcfilename, delimiter=',')

        stdfilename = merged_result_dir + env_name + '_' + agent_name + '_' + result + 'StdRewardsLC.txt'
        print('Reading stdfilename.. ' + stdfilename)
        lcstd = np.loadtxt(stdfilename, delimiter=',')

        paramfile = merged_result_dir + env_name + '_' + agent_name + '_*' + '_Params.txt'
        print('Reading paramfile.. ' + paramfile)

        files = glob.glob(paramfile)
        params = np.loadtxt(files[0], delimiter=',', dtype='str')

        print("{} lc dim: {} x {}".format(result, len(lc), len(lc[0])))
        # default xmax
        xmax = np.shape(lc)[-1]

        xmax_override, ymin, ymax = get_xyrange(env_name)
        if xmax_override is not None:
            xmax = xmax_override

        last_N = int(last_N_ratio * xmax)
        if result == 'TrainEpisode':
            plt.xlabel('Training Steps (per 1000 steps)')

        elif result == 'EvalEpisode':
            plt.xlabel('Training Steps (per 1000 steps)')

        h = plt.ylabel("Cum. Reward per episode")
        h.set_rotation(90)

        opt_range = range(0, xmax)
        plt.xticks(opt_range[::50], np.linspace(0.0, float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax - 1)), int(TOTAL_MIL_STEPS / EVAL_INTERVAL_MIL_STEPS) + 1)[::50])

        xlimt = (0, xmax - 1)
        ylimt = (ymin[result_idx], ymax[result_idx])
        plt.ylim(ylimt)
        plt.xlim(xlimt)

        # if only one line in _StepLC.txt)
        if not np.shape(lc[0]):
            bestlc = lc[:xmax]
            lcse = lcstd[:xmax] / np.sqrt(num_runs)

            print('The best param setting of ' + agent_name + ' is ')
            print(params[:])

        else:
            # sort total performance
            sort_performance_arr = []
            for i in range(len(lc)):
                if eval_last_N:
                    sort_performance_arr.append([i, np.nansum(lc[i, xmax - last_N:xmax])])
                else:
                    sort_performance_arr.append([i, np.nansum(lc[i, :xmax])])

            # sorted array in descending order
            sorted_performance_arr = sorted(sort_performance_arr, key=lambda x: x[1], reverse=True)

            type_best_arr = np.ones(num_type) * -1
            for idx, val in sorted_performance_arr:
                print('setting {}: {}'.format(idx, val))

                # find best index for each type
                for i in range(num_type):
                    if type_best_arr[i] == -1 and idx in type_idx_arr[i]:
                        type_best_arr[i] = idx

            # print result
            for i in range(num_type):
                print("*** best setting for {}: {} --- {}".format(parse_type, type_arr[i], int(type_best_arr[i])))

            print("\n total best setting {}".format(sorted_performance_arr[0][0]))

            if eval_last_N:
                BestInd = np.argmax(np.nansum(lc[:, xmax - last_N:xmax], axis=1))
            else:
                BestInd = np.argmax(np.nansum(lc[:, :xmax], axis=1))

            assert(BestInd == sorted_performance_arr[0][0])
            bestlc = lc[BestInd, :xmax]
            lcse = lcstd[BestInd, :xmax] / np.sqrt(num_runs)

            try:
                assert (BestInd == float(params[BestInd, 0]))
                print('The best param setting of ' + agent_name + ' is ')
                print(params[BestInd, :])
            except:
                # occurring because there aren't any results for some settings
                print('the best param setting of ' + agent_name + ' is ' + str(BestInd))

        legends = [agent_name + ', ' + str(num_runs) + ' runs']

        for i in range(num_type):
            plot_idx = int(type_best_arr[i])

            plot_lc = lc[plot_idx, :xmax]
            plot_lcse = lcstd[plot_idx, :xmax] / np.sqrt(num_runs)

            plt.fill_between(opt_range, plot_lc - plot_lcse, plot_lc + plot_lcse, alpha=0.2)
            plt.plot(opt_range, plot_lc, linewidth=1.0, label='best {}: {}'.format(type_arr[i], plot_idx))

            # save each best settings
            lc_name = '{}/npy/{}_{}_{}_{}_{}_avg.npy'.format(merged_result_dir, env_name, agent_name, result, parse_type, type_arr[i])
            lcse_name = '{}/npy/{}_{}_{}_{}_{}_se.npy'.format(merged_result_dir, env_name, agent_name, result, parse_type, type_arr[i])

            np.save(lc_name, plot_lc)
            np.save(lcse_name, plot_lcse)

        plt.legend(loc="best")

        if show_plot:
            plt.show()
        else:
            plt.savefig("{}_{}_{}.png".format(env_name, agent_name, result + '_' + custom_save_name))
        plt.close()

        savelc = bestlc
        savelcse = lcse

        # # save the best params
        savefilename_avg = merged_result_dir + 'npy/' + env_name + '_' + agent_name + '_' + result + '_BestResult_avg.npy'
        savefilename_se = merged_result_dir + 'npy/' + env_name + '_' + agent_name + '_' + result + '_BestResult_se.npy'
        np.save(savefilename_avg, savelc)
        np.save(savefilename_se, savelcse)

        # just call plot_Bimodal.py
        if plot_each_runs:

            for i in range(num_type):
                print("*** plotting each run for {}: {} --- {}".format(parse_type, type_arr[i], int(type_best_arr[i])))
                os.system("python3 {}/plot_scripts/plot_each_run.py {}results {}/jsonfiles/environment/{}.json {} {} {} {}_{}_{} {}".format(
                    root_dir, env_name, root_dir, env_name, num_runs, agent_name, int(type_best_arr[i]), custom_save_name, parse_type, type_arr[i], result))


