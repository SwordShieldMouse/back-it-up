from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker

import numpy as np
import glob
import sys
import json

from utils import get_agent_parse_info
from plot_config import get_xyrange

import os

import argparse

matplotlib.rcParams.update({'font.size': 35})

cm_exit_count_interval = 100 #Duplicate, defined in utils.py, change if needed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('env_name',type=str,choices=['EasyContinuousMaze','MediumContinuousMaze',"HardContinuousMaze"])
    parser.add_argument('--root_dir',type=str, default="experiments/continuous_deep_control/")
    parser.add_argument('--result_loc',type=str, default="/media/data/SSD_data/back_it_up/_results")
    parser.add_argument('--num_runs',type=int, default=30)
    parser.add_argument('--output_plot_dir',type=str,default="/media/data/SSD_data/back_it_up/_plots/cm_exits")

    args = parser.parse_args()

    input_results_dir = args.result_loc
    root_dir = args.root_dir
    env_name = args.env_name

    result_dir = '{}/{}results/'.format(input_results_dir, env_name)
    env_json_dir = '{}/jsonfiles/environment/{}.json'.format(root_dir, env_name)

    num_runs = args.num_runs

    mean_good_final = {}
    mean_bad_final = {}
    stderr_good_final = {}
    stderr_bad_final = {}
    for agent_name in ['ForwardKL','ReverseKL']:
        output_plot_dir = os.path.join(args.output_plot_dir,agent_name, args.env_name)
        if not os.path.isdir(output_plot_dir):
            os.makedirs(output_plot_dir, exist_ok=True)

        with open(env_json_dir, 'r') as env_dat:
            env_json = json.load(env_dat, object_pairs_hook=OrderedDict)

        env_name = env_json['environment']

        # read agent json
        agent_jsonfile = '{}_{}_agent_Params.json'.format(env_name, agent_name)

        with open(result_dir + agent_jsonfile, 'r') as agent_dat:
            agent_json = json.load(agent_dat, object_pairs_hook=OrderedDict)

        type_arr, pre_divider, num_type, post_divider, num_settings = get_agent_parse_info(agent_json, divide_type="entropy_scale")

        assert num_type == num_settings

        print('Environment: ' + env_name)
        print('Agent: ' + agent_name)

        # ENV specific setting
        TOTAL_MIL_STEPS = env_json['TotalMilSteps']
        X_AXIS_STEPS = cm_exit_count_interval

        plt.figure(figsize=(12, 12))

        paramfile = result_dir + env_name + '_' + agent_name + '_*' + '_Params.txt'
        print('Reading paramfile.. ' + paramfile)

        files = glob.glob(paramfile)
        params = np.loadtxt(files[0], delimiter=',', dtype='str')    

        xmax = None
        entropy_good_data_mean = []
        entropy_good_data_stderr = []
        entropy_bad_data_mean = []
        entropy_bad_data_stderr = []
        for entropy_num, entropy in enumerate(type_arr):
            all_good_data = []
            all_bad_data = []
            for run in range(num_runs):

                pre_name = result_dir + env_name + '_' + agent_name + '_' + 'setting_{}_run_{}'.format(entropy_num,run)

                good_exit_name = pre_name + '_RightExit.txt'
                bad_exit_name = pre_name + '_BadExit.txt'

                good_data = np.loadtxt(good_exit_name, delimiter=',')
                bad_data = np.loadtxt(bad_exit_name, delimiter=',')

                if xmax is None:
                    xmax = np.shape(good_data)[-1] - 1

                all_good_data.append(good_data)
                all_bad_data.append(bad_data)

            all_good_data = np.array(all_good_data)
            all_bad_data = np.array(all_bad_data)        

            mean_good_data = np.mean(all_good_data, axis = 0)
            mean_bad_data = np.mean(all_bad_data, axis = 0)
            stderr_good_data = np.std(all_good_data, axis = 0) / np.sqrt( float(all_good_data.shape[0]) )
            stderr_bad_data = np.std(all_bad_data, axis = 0) / np.sqrt( float(all_bad_data.shape[0]) )

            entropy_good_data_mean.append(mean_good_data)
            entropy_bad_data_mean.append(mean_bad_data)
            entropy_good_data_stderr.append(stderr_good_data)
            entropy_bad_data_stderr.append(stderr_bad_data)

        entropy_good_data_mean = np.array(entropy_good_data_mean)
        entropy_bad_data_mean = np.array(entropy_bad_data_mean)
        entropy_good_data_stderr = np.array(entropy_good_data_stderr)
        entropy_bad_data_stderr = np.array(entropy_bad_data_stderr)

        if agent_name not in mean_good_final:
            mean_good_final[agent_name] = entropy_good_data_mean
            mean_bad_final[agent_name] = entropy_bad_data_mean
            stderr_good_final[agent_name] = entropy_good_data_stderr
            stderr_bad_final[agent_name] = entropy_bad_data_stderr

        entropy_color_dict = {
                            1000: 'blue',
                            100: 'orange',
                            10: 'green',
                            1: 'red',
                            0.1: 'purple',
                            }
        #Plot all entropies together
        for p_type in ["good","bad"]:
            ax = plt.gca()
            opt_range = range(0, xmax+1)
            h = plt.ylabel("Times reached",fontsize=50)
            h.set_rotation(90)
            plt.xlabel('Timesteps ($10^3$)',fontsize=50)
            xtick_step = int( (TOTAL_MIL_STEPS*1e6/X_AXIS_STEPS)/5  )
            tick = [o for o in opt_range[::xtick_step]]
            plt.xticks(tick, tick)
            ax.axes.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x,pos: '{:.0f}'.format( (x/10) )))

            if p_type == "good":
                plt.ylim((0, 500))
            xlimt = (0, xmax)
            plt.xlim(xlimt)


            legends = [agent_name + ', ' + str(num_runs) + ' runs']

            for i in range(num_type):

                mean = entropy_good_data_mean[i] if p_type == "good" else entropy_bad_data_mean[i]
                stderr = entropy_good_data_stderr[i] if p_type == "good" else entropy_bad_data_stderr[i]

                plt.fill_between(opt_range, mean - stderr, mean + stderr, alpha=0.2, color=entropy_color_dict[ type_arr[i] ])
                plt.plot(opt_range, mean, linewidth=1.0, color=entropy_color_dict[type_arr[i]])

            # plt.legend(loc="best")
            plt.savefig(os.path.join(output_plot_dir, "ALL_{good}_{env}_{ag}.png").format(good=p_type.upper() ,env=env_name, ag=agent_name), bbox_inches='tight')
            plt.clf()

    #Plot per entropy
    for i in range(num_type):
        for p_type in ["bad","good"]:
            ax = plt.gca()
            opt_range = range(0, xmax+1)
            plt.xlabel('Timesteps ($10^3$)',fontsize=50)
            xtick_step = int( (TOTAL_MIL_STEPS*1e6/X_AXIS_STEPS)/5  )
            tick = [o for o in opt_range[::xtick_step]]
            plt.xticks(tick, tick)
            ax.axes.get_xaxis().set_major_formatter(ticker.FuncFormatter(lambda x,pos: '{:.0f}'.format( (x/10) )))

            xlimt = (0, xmax)
            plt.xlim(xlimt)   

            if p_type == "good":
                ymax = np.max(np.stack([ mean_good_final['ReverseKL'][i], mean_good_final['ForwardKL'][i]]))
            else:
                ymax = np.max(np.stack([mean_bad_final['ReverseKL'][i],mean_bad_final['ForwardKL'][i]]) )
            if ymax > 1000:
                h = plt.ylabel("Times reached ($10^3$)",fontsize=50)
                ax.axes.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x,pos: '{:.0f}'.format( (x/1000) )))                
            else:
                h = plt.ylabel("Times reached",fontsize=50)                

            h.set_rotation(90)

            for agent_name in ['ReverseKL', 'ForwardKL']:
                color = (0, 179, 92) if agent_name == 'ReverseKL' else (0,92,179)
                if p_type == "good":
                    mean = mean_good_final[agent_name][i]
                    stderr = stderr_good_final[agent_name][i]
                    dash = ()                    
                else:
                    mean = mean_bad_final[agent_name][i]
                    stderr = stderr_bad_final[agent_name][i]                    
                    dash = (4,4)

                rgba_color = np.concatenate( [np.array(color)/255.,[1.] ])
                rgba_transp_color = np.concatenate( [np.array(color)/255.,[0.2] ])
                plt.fill_between(opt_range, mean - stderr, mean + stderr, color=rgba_transp_color)
                plt.plot(opt_range, mean, linewidth=6.0, color=rgba_color, dashes=dash)

            plt.savefig(os.path.join(args.output_plot_dir, "{good}_{env}_entropy_{entr}_BOTH.png").format(good=p_type.upper(), env=env_name, entr=type_arr[i]), bbox_inches='tight')
            plt.clf()

    



