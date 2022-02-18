from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import ticker
from config import CMPlotConfig
import os

def follows_patt(i_str, patt):
    if patt.match(i_str) is not None:
        return True
    else:
        return False

def filter_files_by_patt(files, patt):
    return list(filter(lambda s: follows_patt(s, patt), files))

class ExtremePoint:
    def __init__(self, comparator_f):
        self.comparator_f = comparator_f
        self.v = None

    def update(self, new_v):
        if self.v is None:
            self.v = new_v
        else:
            self.v = self.comparator_f(self.v, new_v)


class PlotDataTable:
    def __init__(self, divide_type, how_to_group='best'):
        self.divide_type = divide_type
        self.all_data = OrderedDict()
        self.how_to_group = how_to_group
        self.has_started_iterating = False

    def add(self, agent, params, index, data):
        if self.has_started_iterating:
            raise AssertionError("Trying to add data to a PlotDataTable that is already being iterated over")
        if agent not in self.all_data:
            self.all_data[agent] = OrderedDict()
        if self.divide_type is None:
            curr_dict = self.all_data[agent]
        else:
            if params[self.divide_type] not in self.all_data[agent]:
                self.all_data[agent][params[self.divide_type]] = OrderedDict()
            curr_dict = self.all_data[agent][params[self.divide_type]]

        if index not in curr_dict:
            curr_dict[index] = []

        curr_dict[index].append(data)

    def iterate(self):
        self.has_started_iterating = True
        for agent in self.all_data.keys():
            if self.divide_type is None:
                mean, stderr = self.group(self.all_data[agent])
                yield agent, mean, stderr
            else:
                for divide_param in self.all_data[agent].keys():
                    mean, stderr = self.group(self.all_data[agent][divide_param])
                    yield (agent, divide_param), mean, stderr

    def group(self, data_dict):
        auc_pct = 0.5
        def _get_means(k_AND_v_arr_list):
            k = k_AND_v_arr_list[0]
            v_arr_list = k_AND_v_arr_list[1]
            return k, np.mean(v_arr_list,axis = 0)
        def _get_auc(k_AND_v_arr):
            k = k_AND_v_arr[0]
            v_arr = k_AND_v_arr[1]
            size = v_arr.size
            return k, np.sum(v_arr[int(size * auc_pct):])

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
        return np.mean(output_runs, axis=0), np.std(output_runs, axis=0) / np.sqrt(output_runs.shape[0])

class Plotter:
    def __init__(self, plot_key, env_params, divide_type):
        self.config = CMPlotConfig()
        self.plot_key = plot_key
        self.env_params = env_params
        if isinstance(plot_key, tuple):
            plot_name = "_".join(plot_key)
        elif isinstance(plot_key, dict):
            plot_name = "_".join(plot_key.values())
        else:
            plot_name = plot_key
        self.plot_name = env_params["environment"] + "_" + plot_name
        self.divide_type = divide_type
        self.max_y_plotted = ExtremePoint(max)
        self.min_y_plotted = ExtremePoint(min)

    def initialize_plot(self, additional_calls=[]):
        matplotlib.rcParams.update({'font.size': self.config.font_size})
        self.fig = plt.figure(figsize=self.config.figsize)
        self.ax = self.fig.add_subplot()
        self.ax.set_ylabel(self.config.ylabel,fontsize=self.config.ylabel_fontsize, rotation=self.config.ylabel_rotation)
        self.ax.set_xlabel(self.config.xlabel, fontsize=self.config.xlabel_fontsize)

        self.xtick_step = int((self.env_params['TotalMilSteps'] * 1e6 / self.env_params['XAxisSteps']) / self.config.n_xticks)

        self.x_range = list(range(0, int(self.env_params['TotalMilSteps'] * 1e6 / self.env_params['XAxisSteps'])))

        ticks = [o for o in self.x_range[::self.xtick_step]]
        self.ax.set_xticks(ticks)
        self.ax.set_xlim((0, self.x_range[-1] + 1))

        x_formatter = ticker.FuncFormatter(lambda t, pos: '{:.0f}'.format(t / (self.config.x_formatter_divider / self.env_params['XAxisSteps'])))
        self.ax.get_xaxis().set_major_formatter(x_formatter)

        self.ax.set_yscale(self.config.yscale)
        self.ax.get_yaxis().set_major_formatter(self.config.y_formatter)

        for function, args in additional_calls:
            f = eval(function)
            if isinstance(args, list):
                f(*args)
            elif isinstance(args, dict):
                f(**args)
        
    def plot_curve(self, curve_key, mean, stderr):
        self.max_y_plotted.update(np.max(mean))
        self.min_y_plotted.update(np.min(mean))
        self.ax.fill_between(self.x_range, mean - stderr, mean + stderr, alpha=self.config.stderr_alpha, color=self.config.get_color(curve_key, self.divide_type))
        self.ax.plot(self.x_range, mean, linewidth=self.config.linewidth, color=self.config.get_color(curve_key, self.divide_type))

    def save_plot(self, save_dir):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        self.fig.savefig(os.path.join(save_dir, "{}.png".format(self.plot_name)), bbox_inches=self.config.savefig_bbox_in,dpi=self.config.savefig_dpi)
        # plt.clf()

class PlotManager:
    def __init__(self, divide_type, how_to_group, env_params):
        self.plot_data_dict = {}
        self.plotter_dict = {}
        self.divide_type = divide_type
        self.how_to_group = how_to_group
        self.env_params = env_params
        self.sync_y_max_values = {}
        self.sync_y_min_values = {}
        self.key_types = None

    def add(self, key, *args, **kwargs):
        if self.key_types is None:
            self.key_types = tuple(key.keys())
        else:
            assert self.key_types == tuple(key.keys())
        key_v = tuple(key.values())
        if key_v not in self.plot_data_dict:
            self.plot_data_dict[key_v] = PlotDataTable(self.divide_type, self.how_to_group)
        self.plot_data_dict[key_v].add(*args, **kwargs)

    def plot_and_save_all(self, plot_dir, synchronize_yaxis_on=None, keep_ymin=False):
        sync = synchronize_yaxis_on is not None
        for k in self.plot_data_dict.keys():
            self.plotter_dict[k] = Plotter(k, self.env_params, self.divide_type)
            if sync:
                sync_value = k[self.key_types.index(synchronize_yaxis_on)]
            plotter = self.plotter_dict[k]
            plotter.initialize_plot()
            for curve_key, curve_mean, curve_stderr in self.plot_data_dict[k].iterate():
                plotter.plot_curve(curve_key, curve_mean, curve_stderr)
                if sync:
                    if sync_value not in self.sync_y_max_values:
                        self.sync_y_max_values[sync_value] = ExtremePoint(max)
                        self.sync_y_min_values[sync_value] = ExtremePoint(min)
                    self.sync_y_max_values[sync_value].update(plotter.max_y_plotted.v)
                    self.sync_y_min_values[sync_value].update(plotter.min_y_plotted.v)
        if sync:
            self.synchronize_y_axis(synchronize_yaxis_on, keep_ymin)
        for plotter in self.plotter_dict.values():
            plotter.save_plot(plot_dir)

    def synchronize_y_axis(self, synchronize_yaxis_on, keep_ymin=False):
        for pl_k, plotter in self.plotter_dict.items():
            sync_value = pl_k[self.key_types.index(synchronize_yaxis_on)]
            if keep_ymin:
                new_ymin = self.sync_y_min_values[sync_value].v
                new_ymax = self.sync_y_max_values[sync_value].v / 0.9
            else:
                delta = self.sync_y_max_values[sync_value].v - self.sync_y_min_values[sync_value].v
                mean_point = (self.sync_y_max_values[sync_value].v + self.sync_y_min_values[sync_value].v) / 2.
                new_delta = delta / 0.8
                new_ymax = mean_point + new_delta / 2.
                new_ymin = mean_point - new_delta / 2.
            plotter.config.y_lim = (new_ymin, new_ymax)
            plotter.ax.get_yaxis().set_major_formatter(plotter.config.y_formatter)
            plotter.ax.set_ylabel(plotter.config.ylabel)
            plotter.ax.set_ylim(bottom=new_ymin, top=new_ymax)