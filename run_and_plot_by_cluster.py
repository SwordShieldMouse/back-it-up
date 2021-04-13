import os
import argparse

# run and plot by cluster
# parser = argparse.ArgumentParser()

# parser.add_argument("cluster",type=str,choices=('graham','cedar','beluga'))
# parser.add_argument("environment",type=str)

# args = parser.parse_args()

class Filler:
    def __init__(self):
        pass

args = Filler()

for cluster in ['graham','cedar','beluga']:
    for environment in ['Reacher-v2','Swimmer-v2','Pendulum-v0','HalfCheetah-v2']:

        args.cluster = cluster
        args.environment = environment

        if args.cluster == 'graham':
            # os.system('python3 experiments/continuous_deep_control/plot_scripts/merge_results.py my_results/normal_sweeps/graham/results_ experiments/continuous_deep_control {} ForwardKL 10 True'.format(args.environment))
            # os.system('python3 experiments/continuous_deep_control/plot_scripts/find_agent_best_setting.py my_results/normal_sweeps/graham/results_ experiments/continuous_deep_control {} ForwardKL 10 graham_{}_fkl_sweep entropy_scale my_results/normal_sweeps/graham/plots_'.format(args.environment, args.environment))
            os.system('python3 experiments/continuous_deep_control/plot_scripts/plot_sensitivity.py my_results/normal_sweeps/graham/results_ {} my_results/normal_sweeps/graham/plots_ --agent ForwardKL'.format(args.environment))
        elif args.cluster == 'cedar':
            # os.system('python3 experiments/continuous_deep_control/plot_scripts/merge_results.py my_results/normal_sweeps/cedar/results_ experiments/continuous_deep_control {} ReverseKL 10 True'.format(args.environment))
            # os.system('python3 experiments/continuous_deep_control/plot_scripts/find_agent_best_setting.py my_results/normal_sweeps/cedar/results_ experiments/continuous_deep_control {} ReverseKL 10 cedar_{}_rkl_rp_sweep entropy_scale my_results/normal_sweeps/cedar/plots_'.format(args.environment, args.environment))
            os.system('python3 experiments/continuous_deep_control/plot_scripts/plot_sensitivity.py my_results/normal_sweeps/cedar/results_ {} my_results/normal_sweeps/cedar/plots_ --agent ReverseKL'.format(args.environment))
        elif args.cluster == 'beluga':
            # os.system('python3 experiments/continuous_deep_control/plot_scripts/merge_results.py my_results/normal_sweeps/beluga/results_ experiments/continuous_deep_control {} ReverseKL 10 True'.format(args.environment))
            # os.system('python3 experiments/continuous_deep_control/plot_scripts/find_agent_best_setting.py my_results/normal_sweeps/beluga/results_ experiments/continuous_deep_control {} ReverseKL 10 beluga_{}_rkl_ll_sweep entropy_scale my_results/normal_sweeps/beluga/plots_'.format(args.environment, args.environment))
            os.system('python3 experiments/continuous_deep_control/plot_scripts/plot_sensitivity.py my_results/normal_sweeps/beluga/results_ {} my_results/normal_sweeps/beluga/plots_  --agent ReverseKL'.format(args.environment))