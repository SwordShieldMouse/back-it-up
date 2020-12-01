from collections import OrderedDict
import matplotlib.pyplot as plt

import numpy as np
import glob
import sys
import json

from utils import get_agent_parse_info
from plot_config import get_xyrange

import os

import argparse

# Usage:
# python3 find_agent_best_setting.py $RESULT_DIR $ROOT_LOC $ENV_NAME $AGENT_NAME $NUM_RUNS $CUSTOM_NAME $PARSE_TYPE $OUTPUT_PLOT_DIR
#
# generates plots and npy for the best settings according to $PARSE_TYPE
#
# RESULT_DIR : where merged{$ENV_NAME}results is located
# ROOT_LOC : root directory of code (where nonlinear_run.py and experiment.py is located)
# CUSTOM_NAME : name of th experiment (to differentiate different sweeps)
# PARSE_TYPE : type of params to parse. If 'entropy_scale', the script will compute best settings for each param value of entropy_scale
# OUTPUT_PLOT_DIR: directory to dump plots

### CONFIG BEFORE RUNNING ###
show_plot = False
plot_each_runs = True
eval_last_N = True
last_N_ratio = 0.5
##############################

if __name__ == "__main__":

    # if len(sys.argv) != 9:
    #     raise ValueError('Invalid input. \nCorrect Usage: find_agent_best_setting.py merged_result_loc, root_dir, env_name, agent_name, num_runs custom_save_name parse_type output_plot_dir')

    parser = argparse.ArgumentParser()

    parser.add_argument('env_name',type=str)
    parser.add_argument('agent_name',type=str)
    parser.add_argument('--root_dir',type=str, default="experiments/continuous_deep_control/")
    parser.add_argument('--merged_result_loc',type=str, default="my_results/normal_sweeps/joint_rkl_fkl/_uncompressed_results")
    parser.add_argument('--custom_save_name',type=str,default=None)
    parser.add_argument('--num_runs',type=int, default=30)
    parser.add_argument('--parse_type',type=str,default="entropy_scale")
    parser.add_argument('--output_plot_dir',type=str,default="my_results/normal_sweeps/joint_rkl_fkl/_plots/individual_performance")

    parser.add_argument('--best_setting_type',type=str,choices=('best','top20'),default='top20')

    args = parser.parse_args()

    if args.custom_save_name is None:
        args.custom_save_name = args.env_name + '_' + args.agent_name

    input_results_dir = args.merged_result_loc
    root_dir = args.root_dir
    env_name = args.env_name
    agent_name = args.agent_name

    non_merged_result_dir = '{}/{}results/'.format(input_results_dir, env_name)
    merged_result_dir = '{}/merged{}results/'.format(input_results_dir, env_name)
    env_json_dir = '{}/jsonfiles/environment/{}.json'.format(root_dir, env_name)

    num_runs = args.num_runs

    custom_save_name = args.custom_save_name
    parse_type = args.parse_type

    output_plot_dir = os.path.join(args.output_plot_dir, args.best_setting_type, args.agent_name, args.env_name)
    if not os.path.isdir(output_plot_dir):
        os.makedirs(output_plot_dir, exist_ok=True)
    best_setting_type = args.best_setting_type

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
    X_AXIS_STEPS = env_json['XAxisSteps']
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

        if args.best_setting_type == 'top20':
            allfilename = merged_result_dir + env_name + '_' + agent_name + '_'  + 'all.npy'
            all_lc = np.load(allfilename)

        paramfile = merged_result_dir + env_name + '_' + agent_name + '_*' + '_Params.txt'
        print('Reading paramfile.. ' + paramfile)

        files = glob.glob(paramfile)
        params = np.loadtxt(files[0], delimiter=',', dtype='str')

        print("{} lc dim: {} x {}".format(result, len(lc), len(lc[0])))
        # default xmax
        xmax = np.shape(lc)[-1] - 1

        xmax_override, ymin, ymax, yticks = get_xyrange(env_name)
        if xmax_override is not None:
            xmax = xmax_override

        last_N = int(last_N_ratio * xmax)
        if result == 'TrainEpisode':
            plt.xlabel('Training Steps (per {} steps)'.format(X_AXIS_STEPS))

        else:
            raise NotImplementedError

        h = plt.ylabel("Cum. Reward per episode")
        h.set_rotation(90)

        opt_range = range(0, xmax+1)
        # plt.xticks(opt_range[::50], np.linspace(0.0, float(EVAL_INTERVAL_MIL_STEPS * 1e3 * (xmax - 1)), int(TOTAL_MIL_STEPS / EVAL_INTERVAL_MIL_STEPS) + 1)[::50])

        xtick_step = int( (TOTAL_MIL_STEPS*1e6/X_AXIS_STEPS)/10  )
        tick = [o for o in opt_range[::xtick_step]]
        plt.xticks(tick, tick)

        xlimt = (0, xmax)
        ylimt = (ymin[result_idx], ymax[result_idx])
        plt.ylim(ylimt)
        plt.xlim(xlimt)
        plt.yticks(yticks)

        # if only one line in _StepLC.txt)
        if not np.shape(lc[0]):
            bestlc = lc[:(xmax+1)]
            lcse = lcstd[:(xmax+1)] / np.sqrt(num_runs)

            print('The best param setting of ' + agent_name + ' is ')
            print(params[:])

        else:
            # sort total performance
            sort_performance_arr = []
            for i in range(len(lc)):
                if eval_last_N:
                    sort_performance_arr.append([i, np.nansum(lc[i, (xmax+1) - last_N:(xmax+1)])])
                else:
                    sort_performance_arr.append([i, np.nansum(lc[i, :(xmax+1)])])

            # sorted array in descending order
            sorted_performance_arr = sorted(sort_performance_arr, key=lambda x: x[1], reverse=True)

            if best_setting_type == 'best':
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
                    BestInd = np.argmax(np.nansum(lc[:, (xmax+1) - last_N:(xmax+1)], axis=1))
                else:
                    BestInd = np.argmax(np.nansum(lc[:, :(xmax+1)], axis=1))

                assert(BestInd == sorted_performance_arr[0][0])
                bestlc = lc[BestInd, :(xmax+1)]
                lcse = lcstd[BestInd, :(xmax+1)] / np.sqrt(num_runs)

                try:
                    assert (BestInd == float(params[BestInd, 0]))
                    print('The best param setting of ' + agent_name + ' is ')
                    print(params[BestInd, :])
                except:
                    # occurring because there aren't any results for some settings
                    print('the best param setting of ' + agent_name + ' is ' + str(BestInd))                

            elif best_setting_type == 'top20':

                total_settings_per_parse = int( len(lc) / num_type )
                last_t20 = int( 0.8 * total_settings_per_parse)
                last_t20_elems = max(total_settings_per_parse - last_t20, 2)

                type_best_arr = [[ -1 for _ in range(last_t20_elems) ] for _ in range(num_type)]

                for idx, val in sorted_performance_arr:
                    print('setting {}: {}'.format(idx, val))

                    # find best index for each type
                    for i in range(num_type):
                        if -1 in type_best_arr[i] and idx in type_idx_arr[i]:
                            m1_idx = type_best_arr[i].index(-1)
                            type_best_arr[i][m1_idx] = idx

                # print result
                for i in range(num_type):
                    print("*** top 20%% settings for {}: {} --- {}".format(parse_type, type_arr[i], ', '.join( [str(a) for a in type_best_arr[i]] )))

                BestInd = sorted_performance_arr[0][0]
                bestlc = lc[BestInd, :(xmax+1)]
                lcse = lcstd[BestInd, :(xmax+1)] / np.sqrt(num_runs)                    


        legends = [agent_name + ', ' + str(num_runs) + ' runs']

        for i in range(num_type):
            if best_setting_type == 'best':
                plot_idx = int(type_best_arr[i])

                plot_lc = lc[plot_idx, :(xmax+1)]
                plot_lcse = lcstd[plot_idx, :(xmax+1)] / np.sqrt(num_runs)

                plt.fill_between(opt_range, plot_lc - plot_lcse, plot_lc + plot_lcse, alpha=0.2)
                plt.plot(opt_range, plot_lc, linewidth=1.0, label='best {}: {}'.format(type_arr[i], plot_idx))

            else:
                plot_idxs = type_best_arr[i]

                lc_separate_means = lc[plot_idxs][:, :(xmax+1)]
                plot_lc = np.mean(lc_separate_means, axis=0)

                filtered_lc = all_lc[plot_idxs][:, :, :(xmax+1)]
                filtered_lc = np.reshape(filtered_lc, [-1, (xmax+1)])

                plot_lcse = np.std(filtered_lc, axis=0) / np.sqrt(filtered_lc.shape[0])

                plt.fill_between(opt_range, plot_lc - plot_lcse, plot_lc + plot_lcse, alpha=0.2)
                plt.plot(opt_range, plot_lc, linewidth=1.0, label='top 20 {}'.format(type_arr[i]))                

            # save each best settings
            lc_name = '{m}/npy/{best_type}_{env}_{ag}_{res}_{pt}_{entr}_avg.npy'.format(m=merged_result_dir, env=env_name, ag=agent_name, res=result, pt=parse_type, entr=type_arr[i],best_type=best_setting_type)
            lcse_name = '{m}/npy/{best_type}_{env}_{ag}_{res}_{pt}_{entr}_se.npy'.format(m=merged_result_dir, env=env_name, ag=agent_name, res=result, pt=parse_type, entr=type_arr[i],best_type=best_setting_type)

            np.save(lc_name, plot_lc)
            np.save(lcse_name, plot_lcse)

        plt.legend(loc="best")

        if show_plot:
            plt.show()
        else:
            plt.savefig(os.path.join(output_plot_dir, "{best_type}_{env}_{ag}_{res}.png").format(env=env_name, ag=agent_name, res=result + '_' + custom_save_name, best_type=best_setting_type))
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
                if best_setting_type == 'best':
                    t = int(type_best_arr[i])
                elif best_setting_type == 'top20':
                    t = type_best_arr[i][0]

                print("*** plotting each run for {}: {} --- {}".format(parse_type, type_arr[i], t))
                os.system("python3 {root}/plot_scripts/plot_each_run.py {non_mer} {root}/jsonfiles/environment/{env}.json {n_runs} {ag} {t} {save}_{parse}_{entr} {res_type} {out}".format(
                    root=root_dir, non_mer=non_merged_result_dir, env=env_name, n_runs=num_runs, ag=agent_name, t=t, save=custom_save_name, parse=parse_type, entr=type_arr[i], res_type=result, out=output_plot_dir))


