from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import json
from matplotlib import ticker
from .config import *
import os
import functools

def get_high_low(temp):
    if temp in BenchmarksPlotConfig.high_temps:
        return 'high'
    elif temp in BenchmarksPlotConfig.low_temps:
        return 'low'
    else:
        raise ValueError

def follows_patt(i_str, patt):
    if patt.match(i_str) is not None:
        return True
    else:
        return False

def filter_files_by_patt(files, patt):
    return list(filter(lambda s: follows_patt(s, patt), files))

def get_unrolled_data(data, steps, max_steps, x_axis_steps):
    current_episode = 0
    running_frame = 0
    n_idxs = int(max_steps/x_axis_steps)
    output = np.zeros([n_idxs])
    for global_idx in range(n_idxs):
        while(running_frame >= steps[current_episode]):
            current_episode += 1
        output[global_idx] = np.mean(data[current_episode:current_episode+PlotConfig.episodes_window])
        running_frame += x_axis_steps
    return output

def load_and_unroll_files(base_dir, base_name, env_params):
    rewards_fname = base_name + "_EpisodeRewardsLC.txt"
    steps_fname = base_name + "_EpisodeStepsLC.txt"
    unrolled_fname = base_name + "_UnrolledEpisodeStepsLC.txt"
    full_rewards_fname = os.path.join(base_dir, rewards_fname)
    full_steps_fname = os.path.join(base_dir, steps_fname)
    full_unrolled_fname = os.path.join(base_dir, unrolled_fname)
    if os.path.isfile(full_unrolled_fname):
        return np.loadtxt(full_unrolled_fname, delimiter=',')
    else:
        data = np.loadtxt(full_rewards_fname, delimiter=',')
        steps = np.loadtxt(full_steps_fname, delimiter=',')
        data = get_unrolled_data(data, steps, env_params['TotalMilSteps'] * 1e6, env_params['XAxisSteps'])
        data.tofile(full_unrolled_fname, sep=',')
        return data

def expand_limits(pct, low_lim, high_lim):
    delta = high_lim - low_lim
    mean_point = (high_lim + low_lim) / 2.
    new_delta = delta / pct
    new_high = mean_point + new_delta / 2.
    new_low = mean_point - new_delta / 2.
    return new_low, new_high

class ExtremePoint:
    def __init__(self, comparator_f):
        self.comparator_f = comparator_f
        self.v = None

    def update(self, new_v):
        if self.v is None:
            self.v = new_v
        else:
            self.v = self.comparator_f(self.v, new_v)

class PlotDataObj:
    def __init__(self, args):
        self.args = args
        self.divide_type = args.divide_type
        self.how_to_group = args.how_to_group
        self.all_data = OrderedDict()
        self.has_started_iterating = False
        self.loaded = False
        self.generator_data = []

    def add(self, agent, ag_params, index, data):
        if self.has_started_iterating:
            raise AssertionError("Trying to add data to a PlotDataObj that is already being iterated over")
        if self.loaded:
            raise AssertionError("Trying to add data to a PlotDataObj that was loaded from file")
        if agent not in self.all_data:
            self.all_data[agent] = OrderedDict()
        if self.divide_type is None:
            curr_dict = self.all_data[agent]
        else:
            if ag_params[self.divide_type] not in self.all_data[agent]:
                self.all_data[agent][ag_params[self.divide_type]] = OrderedDict()
            curr_dict = self.all_data[agent][ag_params[self.divide_type]]
        if index not in curr_dict:
            curr_dict[index] = []
        curr_dict[index].append(data)
        
    def load(self, pickle_data):
        self.loaded = True
        self.generator_data = pickle_data

    def iterate(self):
        self.has_started_iterating = True
        if self.loaded:
            for output in self.generator_data:
                yield output
        else:
            for agent in self.all_data.keys():
                if self.divide_type is None:
                    mean, stderr = self.group(self.all_data[agent])
                    curve_id = agent
                    self.generator_data.append((curve_id, mean, stderr))
                    yield curve_id, mean, stderr
                else:
                    for divide_param in self.all_data[agent].keys():
                        mean, stderr = self.group(self.all_data[agent][divide_param])
                        curve_id = "_".join([agent, str(divide_param)])
                        self.generator_data.append((curve_id, mean, stderr))
                        yield curve_id, mean, stderr

    def group(self, data_dict):
        auc_pct = 0.5
        def _get_means(k_AND_v_arr_list):
            k = k_AND_v_arr_list[0]
            v_arr_list = k_AND_v_arr_list[1]
            return k, np.mean(v_arr_list,axis = 0)
        def _get_auc(k_AND_mean_arr):
            k = k_AND_mean_arr[0]
            mean_arr = k_AND_mean_arr[1]
            size = mean_arr.size
            return k, np.sum(mean_arr[int(size * auc_pct):])

        all_means = OrderedDict(map(_get_means, data_dict.items()))
        size = next(iter(all_means.values())).size
        all_auc = OrderedDict(map(_get_auc, all_means.items()))
        all_auc_k, all_auc_v = np.array(list(all_auc.keys())), np.array(list(all_auc.values()))
        argsorted_auc_v = np.argsort(all_auc_v)
        sorted_auc_k, sorted_auc_v = all_auc_k[argsorted_auc_v], all_auc_v[argsorted_auc_v]

        if self.how_to_group == 'best':
            out_k = [sorted_auc_k[-1]]
        elif self.how_to_group == 'top20':
            out_k = sorted_auc_k[int(sorted_auc_k.size * 0.8):]

        nested_output_runs = list(map(lambda it: it[1], filter(lambda it: it[0] in out_k, data_dict.items())))
        output_runs = np.array(nested_output_runs).reshape((-1, size))
        if self.args.bar:
            output_auc = np.mean(output_runs[:, int(auc_pct * size):], axis=1)
            return np.mean(output_auc), np.std(output_auc) / np.sqrt(output_runs.shape[0])
        else:
            return np.mean(output_runs, axis=0), np.std(output_runs, axis=0) / np.sqrt(output_runs.shape[0])

class Plotter:
    def __init__(self, call_id, plot_id, args, env_params):
        self.args = args
        self.config = args.config_class()
        if hasattr(args, 'normalize'):
            self.config.normalize_formatter = args.normalize
        self.env_params = env_params
        self.plot_name = "_".join([call_id, plot_id])
        self.divide_type = args.divide_type
        self.max_y_plotted = ExtremePoint(max)
        self.min_y_plotted = ExtremePoint(min)
        self.call_buffer = []

    def initialize_plot(self):
        matplotlib.rcParams.update({'font.size': self.config.font_size})
        self.fig = plt.figure(figsize=self.config.figsize)
        self.ax = self.fig.add_subplot()

        if not self.args.bar:
            self.x_range = list(range(0, int(self.env_params['TotalMilSteps'] * 1e6), int(self.env_params['XAxisSteps'] )))
            self.config.x_lim = (0, (self.x_range[-1] + 1) )
            self.ax.set_xlim(self.config.x_lim)
        
        self.ax.set_ylabel(self.config.ylabel,fontsize=self.config.ylabel_fontsize, rotation=self.config.ylabel_rotation)
        self.ax.set_xlabel(self.config.xlabel, fontsize=self.config.xlabel_fontsize)

        if not hasattr(self.config, 'xticks'):
            xtick_max = int((self.env_params['TotalMilSteps'] * 1e6 ) )
            ticks = range(0, xtick_max+1, int(xtick_max/self.config.n_xticks))
            self.ax.set_xticks(ticks)
        else:
            self.ax.set_xticks(self.config.xticks)

        self.ax.get_xaxis().set_major_formatter(self.config.x_formatter)

        self.ax.set_yscale(self.config.yscale)
        self.ax.get_yaxis().set_major_formatter(self.config.y_formatter)

        if hasattr(self.config, "locator_params_kwargs"):
            self.ax.locator_params(**self.config.locator_params_kwargs)
        
    def plot_curve(self, curve_id, mean, stderr):
        self.max_y_plotted.update(np.max(mean))
        self.min_y_plotted.update(np.min(mean))
        color = self.config.get_color(curve_id, self.divide_type)
        if self.args.bar:
            x_pos = self.config.get_x_position(curve_id, self.divide_type)
            plot_partial_call = functools.partial(self.ax.bar, x=x_pos, height=mean, yerr=stderr, color=color, width=self.config.width)
            self.call_buffer.append(plot_partial_call)
        else:
            x_pos = np.array(self.x_range)
            self.ax.fill_between(x_pos, mean - stderr, mean + stderr, alpha=self.config.stderr_alpha, color=self.config.get_color(curve_id, self.divide_type))
            self.ax.plot(x_pos, mean, linewidth=self.config.linewidth, color=color)

    def save_plot(self, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        self.fig.savefig(os.path.join(save_dir, "{}.png".format(self.plot_name)), bbox_inches=self.config.savefig_bbox_in,dpi=self.config.savefig_dpi)
    
    def update_y_lim(self, new_ylim):
        new_ymin, new_ymax = new_ylim[0], new_ylim[1]
        self.config.y_lim = (new_ymin, new_ymax)
        self.ax.get_yaxis().set_major_formatter(self.config.y_formatter)
        self.ax.set_ylabel(self.config.ylabel)
        self.ax.set_ylim(bottom=new_ymin, top=new_ymax)
        if self.args.bar and self.args.normalize:
            for partial_call in self.call_buffer:
                partial_call.keywords['height'] = partial_call.keywords['height'] - new_ymin
                partial_call(bottom=new_ymin)
            delta = new_ymax - new_ymin
            self.ax.set_yticks([new_ymin + pct * delta for pct in self.config.yticks_pct])


class PlotManager:
    def __init__(self, args):
        self.args = args
        self.call_id = self.get_call_id()
        with open(self.args.env_json_fname, "r") as f:
            self.env_params = json.load(f, object_pairs_hook=OrderedDict)            
        self.env_results_dir = os.path.join(args.results_dir, args.env_name)
        self.args.config_class = eval(args.config_class)
        self.plot_dict = {}
        self.divide_type = args.divide_type
        self.how_to_group = args.how_to_group
        self.separate_agent_plots = args.separate_agent_plots
        self.sync_y_max_data = {}

    def get_call_id(self):
        sep_id = "SplitAgents" if self.args.separate_agent_plots else "JointAgents"
        div_id = self.args.divide_type if self.args.divide_type is not None else "NoDivide"
        bar_id = ["bar"] if self.args.bar else []
        return "_".join([sep_id, self.args.how_to_group, div_id] + bar_id)

    def add(self, plot_id, *f_args, **f_kwargs):
        if plot_id not in self.plot_dict:
            self.plot_dict[plot_id] = {'data': PlotDataObj(self.args)}
        self.plot_dict[plot_id]['data'].add(*f_args, **f_kwargs)
        
    def load_existing_data(self, plot_id):
        if hasattr(self.args, "preprocessed_dir"):
            full_fname = os.path.join(self.args.preprocessed_dir, "{}.pkl".format("_".join([self.call_id, plot_id])))
            if os.path.isfile(full_fname):
                if plot_id not in self.plot_dict:
                    self.plot_dict[plot_id] = {'data': PlotDataObj(self.args)}
                    self.plot_dict[plot_id]['data'].load(pickle.load(open(full_fname,'rb')))
                return True
        return False
            
    def save_all_data(self):
        preprocessed_dir = self.args.preprocessed_dir
        for this_plot_id, this_plot_dict in self.plot_dict.items():
            plot_obj = this_plot_dict['data']
            ofname = os.path.join(preprocessed_dir, "{}.pkl".format("_".join([self.call_id, this_plot_id])))
            if not os.path.isfile(ofname):
                pickle.dump(obj=plot_obj.generator_data, file=open(ofname, "wb"))

    def plot_and_save_all(self, synchronize_y_options=None):
        sync = synchronize_y_options is not None
        for plot_id in self.plot_dict.keys():
            self.plot_dict[plot_id]['plotter'] = Plotter(self.call_id, plot_id, self.args, self.env_params)
            if sync:
                if synchronize_y_options["mode"] == "y_idx":
                    sync_value = plot_id.split("_")[synchronize_y_options["sync_idx"]]
            plotter = self.plot_dict[plot_id]['plotter']
            plotter.initialize_plot()
            for curve_id, curve_mean, curve_stderr in self.plot_dict[plot_id]['data'].iterate():
                plotter.plot_curve(curve_id, curve_mean, curve_stderr)
                if sync:
                    if synchronize_y_options["mode"] == "y_idx":
                        if sync_value not in self.sync_y_max_data:
                            self.sync_y_max_data[sync_value] = {'max': ExtremePoint(max), 'min': ExtremePoint(min)}
                        self.sync_y_max_data[sync_value]['max'].update(plotter.max_y_plotted.v)
                        self.sync_y_max_data[sync_value]['min'].update(plotter.min_y_plotted.v)
            if not sync and self.args.normalize:
                new_ymin, new_ymax = expand_limits(0.8, plotter.min_y_plotted.v, plotter.max_y_plotted.v)
                plotter.update_y_lim((new_ymin, new_ymax))
        if sync:
            self.synchronize_y_axis(synchronize_y_options)
        for this_plot_dict in self.plot_dict.values():
            this_plot_dict['plotter'].save_plot(self.args.plot_dir)

    def synchronize_y_axis(self, synchronize_y_options):
        if "save_max" in synchronize_y_options:
            out_max = {}
        if synchronize_y_options["mode"] == "from_file":
            max_dict = pickle.load(open(os.path.join(self.args.preprocessed_dir, "{}_max_data.pkl".format(synchronize_y_options["target_call_id"])), "rb"))
        for plot_id, plot_dict in self.plot_dict.items():
            plotter = self.plot_dict[plot_id]['plotter']
            sync_value = plot_id.split("_")[synchronize_y_options["sync_idx"]]
            if synchronize_y_options["mode"] == "y_idx":
                if synchronize_y_options["keep_ymin"]:
                    new_ymin = self.sync_y_max_data[sync_value]['min'].v
                    new_ymax = self.sync_y_max_data[sync_value]['max'].v / 0.9
                else:
                    new_ymin, new_ymax = expand_limits(0.8, self.sync_y_max_data[sync_value]['min'].v, self.sync_y_max_data[sync_value]['max'].v)
                if "save_max" in synchronize_y_options:
                    out_max[sync_value] = (new_ymin, new_ymax)
            elif synchronize_y_options["mode"] == "from_file":
                new_ymin, new_ymax = max_dict[sync_value]
            plotter.update_y_lim((new_ymin, new_ymax))
        if "save_max" in synchronize_y_options:
            pickle.dump(out_max, open(os.path.join(self.args.preprocessed_dir, "{}_max_data.pkl".format(self.call_id)), "wb"))