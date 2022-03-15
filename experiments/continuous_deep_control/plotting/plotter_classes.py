from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import json
from .config import *
import os
import functools
import bisect


def expand_limits(pct, low_lim, high_lim):
    # Expands limits such that the old ones correspond to the central
    # pct portion of the new limits
    delta = high_lim - low_lim
    mean_point = (high_lim + low_lim) / 2.
    new_delta = delta / pct
    new_high = mean_point + new_delta / 2.
    new_low = mean_point - new_delta / 2.
    return new_low, new_high

class ExtremePoint:
    # Class to store maximum/minimum of many input values
    def __init__(self, comparator_f):
        self.comparator_f = comparator_f
        self.v = None

    def update(self, new_v):
        if self.v is None:
            self.v = new_v
        else:
            self.v = self.comparator_f(self.v, new_v)

class PlotDataObj:
    # Class responsible for storing the data to be plotted
    # First, the data is iteratively added by PlotManager using self.add,
    # alternatively, the data can be loaded from a preprocessed pickle using self.load.
    # After all the data is added (or loaded), the function self.iterate is used to
    # process (using self.group and/or self.group_sensitivity) and return the data
    # in plot-ready format (the x and y in matplotlib .plot functions).
    #
    # There is one PlotDataObj for each plot (with possibly many curves)
    def __init__(self, args, plot_id):
        self.args = args
        self.auc_pct = 0.5
        # Plot identifier
        self.plot_id = plot_id
        # Which parameter to use to group settings in curves
        self.divide_type = args.divide_type
        # Group settings using top20 or best
        self.how_to_group = args.how_to_group
        # X axis in hyperparameter plots
        self.hyperparam_for_sensitivity = args.hyperparam_for_sensitivity
        # Dictionary with all data in nested format
        # Example: { ReverseKL: {
        #               entr_0.1: { index_1: [run_1, run_2...], index_2: [run_1, run_2...]}
        #               entr_0.01: { index_5: [run_1, run_2...]}
        #               }
        #           }
        # (obs: has additional level hor hyperparameter sensitivity plots)
        self.all_data = OrderedDict()
        # Flags
        self.has_started_iterating = False
        self.loaded = False
        # Stores curve (i.e. processed) data to be saved to file (or which was loaded from file)
        self.generator_data = []

    def add(self, agent, ag_params, index, data):
        # Adds new data to self.all_data (see example dict in __init__)
        if self.has_started_iterating:
            raise AssertionError("Trying to add data to a PlotDataObj that is already being iterated over")
        if self.loaded:
            raise AssertionError("Trying to add data to a PlotDataObj that was loaded from file")
        # Creates agent dict if inexistent
        if agent not in self.all_data:
            self.all_data[agent] = OrderedDict()

        # These if statements create or select the dictionary where indexes and runs will be added
        # Option 1: No divide type, use only agent names
        if self.divide_type is None:
            if self.hyperparam_for_sensitivity is not None:
                raise NotImplementedError("Sensitivity plots only work with divide_type not None")
            curr_dict = self.all_data[agent]
        else:
            # Creates divide_type (i.e. entropy) dict if inexistent
            if ag_params[self.divide_type] not in self.all_data[agent]:
                self.all_data[agent][ag_params[self.divide_type]] = OrderedDict()
            # Option 2: Use agent names and divide_type (i.e. entropy)
            if self.hyperparam_for_sensitivity is None:
                curr_dict = self.all_data[agent][ag_params[self.divide_type]]
            else:
            # Option 3: Use agent names, divide_type and sensitivity hyperparameter
                # Creates sensitivity hyperparameter dict if non-existent
                if ag_params[self.hyperparam_for_sensitivity] not in self.all_data[agent][ag_params[self.divide_type]]:
                    self.all_data[agent][ag_params[self.divide_type]][ag_params[self.hyperparam_for_sensitivity]] = OrderedDict()
                curr_dict = self.all_data[agent][ag_params[self.divide_type]][ag_params[self.hyperparam_for_sensitivity]]
                # Stores new hyperparameters in shared list args.all_hyper (this is where the X
                # axis is obtained for sensitivity plots)
                if ag_params[self.hyperparam_for_sensitivity] not in self.args.all_hyper:
                    bisect.insort(self.args.all_hyper, ag_params[self.hyperparam_for_sensitivity])

        # Insert index and runs in selected dict
        if index not in curr_dict:
            curr_dict[index] = []
        curr_dict[index].append(data)
        
    def load(self, pickle_data):
        # Load preprocessed data from file, as opposed to using multiple self.add calls.
        # This function is called by PlotManager
        self.loaded = True
        self.generator_data = pickle_data['generator_data']
        self.args.all_hyper = pickle_data['all_hyper']

    def iterate(self):
        # Generator function
        # Processes data from self.all_data and returns it in plot-ready format
        self.has_started_iterating = True
        # If data was loaded from preprocessed file, simply read the list
        # and return iteratively
        if self.loaded:
            for output in self.generator_data:
                yield output
        # Else, process and return iteratively
        else:
            for agent in self.all_data.keys():
                # Option 1, only agent names are used for grouping settings and runs
                if self.divide_type is None:
                    self.cur_curve_id = agent
                    mean, stderr = self.group(self.all_data[agent])
                    self.generator_data.append((self.cur_curve_id, mean, stderr))
                    yield self.cur_curve_id, mean, stderr
                else:
                    # Option 2, agent names and divide_type (e.g. entropy) are used
                    # for grouping settings and runs
                    if self.hyperparam_for_sensitivity is None:
                        for divide_param in self.all_data[agent].keys():
                            self.cur_curve_id = "_".join([agent, str(divide_param)])
                            mean, stderr = self.group(self.all_data[agent][divide_param])
                            self.generator_data.append((self.cur_curve_id, mean, stderr))
                            yield self.cur_curve_id, mean, stderr
                    else:
                        # Option 3, agent names, divide_type (e.g. entropy) and sensitivity
                        # hyperparameter are used for grouping settings and runs
                        for divide_param in sorted(self.all_data[agent].keys()):
                            self.cur_curve_id = "_".join([agent, str(divide_param)])
                            # Only the [agent][divide_param] dict is passed, so that the entire
                            # X axis is generated at once
                            mean, stderr = self.group_sensitivity(self.all_data[agent][divide_param])
                            self.generator_data.append((self.cur_curve_id, mean, stderr))
                            yield self.cur_curve_id, mean, stderr

    def group(self, data_dict):
        # Receives data from settings and runs and groups it, returning mean and stderr
        # Example input: { index_1: [run_1, run_2...], index_2: [run_1, run_2...]}

        # Converts run list to np array
        for k,v in data_dict.items():
            data_dict[k] = np.stack(v, axis=0)
        auc_pct = self.auc_pct
        def _get_means(k_AND_v_arr_list):
            # Returns index and mean of all runs for that index
            k = k_AND_v_arr_list[0]
            v_arr_list = k_AND_v_arr_list[1]
            return k, np.mean(v_arr_list,axis = 0)
        def _get_auc(k_AND_mean_arr):
            # Returns index and last (self.auc)% of AUC of mean of all runs for that index
            k = k_AND_mean_arr[0]
            mean_arr = k_AND_mean_arr[1]
            size = mean_arr.size
            return k, np.mean(mean_arr[int(size * auc_pct):])

        # Get dict with (setting: mean of runs) entries
        all_means = OrderedDict(map(_get_means, data_dict.items()))
        size = next(iter(all_means.values())).size
        # Get dict with (setting: (self.auc)% of AUC of mean of runs) entries
        all_auc = OrderedDict(map(_get_auc, all_means.items()))
        all_auc_k, all_auc_v = np.array(list(all_auc.keys())), np.array(list(all_auc.values()))
        # Sorts (to get top 20% or best)
        argsorted_auc_v = np.argsort(all_auc_v)
        sorted_auc_k, sorted_auc_v = all_auc_k[argsorted_auc_v], all_auc_v[argsorted_auc_v]

        # Selects output settings
        if self.how_to_group == 'best':
            out_k = [sorted_auc_k[-1]]
        elif self.how_to_group == 'top20':
            out_k = sorted_auc_k[int(sorted_auc_k.size * 0.8):]

        # Logs best setting
        if self.args.log_best_setting:
            self.log_best_setting(out_k[-1])

        # Filters the input data_dict with out_k
        nested_output_runs = list(map(lambda it: it[1], filter(lambda it: it[0] in out_k, data_dict.items())))
        # Converts to (-1, size) format
        output_runs = np.concatenate(nested_output_runs, axis=0)
        # If bar plots or sensitivity plots, get the mean of the (self.auc)% of AUC of all settings and runs
        # corresponding to out_k
        if self.args.bar or self.hyperparam_for_sensitivity is not None:
            output_auc = np.mean(output_runs[:, int(auc_pct * size):], axis=1)
            return np.mean(output_auc), np.std(output_auc) / np.sqrt(output_runs.shape[0])
        # Else (e.g. benchmark plots), just return the mean of the returns
        # (each step corresponds to returns of nearby episodes)
        else:
            return np.mean(output_runs, axis=0), np.std(output_runs, axis=0) / np.sqrt(output_runs.shape[0])

    def group_sensitivity(self, ag_AND_scale_all_hypers_dict):
        # Receives dict corresponding to current (agent, divide_type) and, for each hyperparameter,
        # obtains the AUC
        out_auc = np.zeros([len(self.args.all_hyper)])
        out_stderr = np.zeros([len(self.args.all_hyper)])
        for h_idx, h in enumerate(self.args.all_hyper):
            dict_all_sett = ag_AND_scale_all_hypers_dict[h]
            auc, stderr = self.group(dict_all_sett)
            out_auc[h_idx] = auc
            out_stderr[h_idx] = stderr
        return out_auc, out_stderr

    def log_best_setting(self, best_setting):
        # Saves the best setting to results/_info directory
        if not os.path.isdir(self.args.info_logdir):
            os.makedirs(self.args.info_logdir, exist_ok=True)
        ofname = os.path.join(self.args.info_logdir, 'best_sett.pkl')
        if os.path.isfile(ofname):
            r_f = open(ofname, "rb")
            o_d = pickle.load(r_f)
            r_f.close()
        else:
            o_d = {}
        if self.plot_id not in o_d:
            o_d[self.plot_id] = {}
        o_d[self.plot_id][self.cur_curve_id] = best_setting
        w_f = open(ofname, "wb")
        pickle.dump(obj=o_d, file=w_f)
        w_f.close()

class Plotter:
    # Class responsible for plotting, there is one Plotter for each plot (with
    # possibly many curves)
    def __init__(self, call_id, plot_id, args, env_params):
        self.args = args
        # Plot identifier
        self.plot_id = plot_id
        # self.config is an object with information such as fontsizes, colors...
        self.config = args.config_class(args)
        # The bar plots are normalized
        if hasattr(args, 'normalize'):
            self.config.normalize_formatter = args.normalize
        self.env_params = env_params
        # The name to be saved from file, the combination of call_id and
        # plot_id uniquely identifies the plot
        self.plot_name = "_".join([call_id, plot_id])
        self.divide_type = args.divide_type # Usually entropy_scale
        # For all curves plotted, these variables store the maximum and minimum
        # overall Y values. This can be used to synchronize the Y axis between
        # different plots
        self.max_y_plotted = ExtremePoint(max)
        self.min_y_plotted = ExtremePoint(min)
        # The bar plots require normalization, which can only be done using self.max_y_plotted
        # and self.min_y_plotted, which in turn are only known after all curves have been plotted.
        # The bar plots' bottom cannot be adjusted after ax.bar() is called. So, instead of calling it
        # when self.plot_curve is called (i.e. the regular behavior), the functions are stored in
        # self.call_buffer and all functions from this list are called at the end, in self.update_y_lim
        # (with the correct information of bar bottoms now available)
        self.call_buffer = []

    def initialize_plot(self):
        # Reads self.config and calls functions to initialize the plots
        matplotlib.rcParams.update({'font.size': self.config.font_size})
        self.fig = plt.figure(figsize=self.config.figsize)
        self.ax = self.fig.add_subplot()

        # This has to be called before ax.set_major_formatter(), otherwise it overwrites
        # that function's behavior
        self.ax.set_yscale(self.config.yscale)
        self.ax.set_xscale(self.config.xscale)

        # When the X axis is timesteps, the X coordinates of the subsequent points are
        # separated by self.env_params['XAxisSteps'] steps
        if (not self.args.bar) and (self.args.hyperparam_for_sensitivity is None):
            # self.x_range example: [0, 100, 200, 300, ....] for self.env_params['XAxisSteps'] = 100
            self.x_range = list(range(0, int(self.env_params['TotalMilSteps'] * 1e6), int(self.env_params['XAxisSteps'] )))
            self.config.x_lim = (0, (self.x_range[-1] + 1) )
            self.ax.set_xlim(self.config.x_lim)

        self.ax.set_ylabel(self.config.ylabel, fontsize=self.config.ylabel_fontsize, rotation=self.config.ylabel_rotation)
        self.ax.set_xlabel(self.config.xlabel, fontsize=self.config.xlabel_fontsize)

        if self.args.hyperparam_for_sensitivity is None:
            if not hasattr(self.config, 'xticks'):
                xtick_max = int((self.env_params['TotalMilSteps'] * 1e6 ) )
                ticks = range(0, xtick_max+1, int(xtick_max/self.config.n_xticks))
                self.ax.set_xticks(ticks)
            else:
                self.ax.set_xticks(self.config.xticks)

        self.ax.get_yaxis().set_major_formatter(self.config.y_formatter)
        self.ax.get_xaxis().set_major_formatter(self.config.x_formatter)

        if hasattr(self.config, "locator_params_kwargs"):
            self.ax.locator_params(**self.config.locator_params_kwargs)
        
    def plot_curve(self, curve_id, mean, stderr):
        # Plots the curve on self.ax
        #
        # Updates maximum/minimum information
        self.max_y_plotted.update(np.max(mean))
        self.min_y_plotted.update(np.min(mean))
        color = self.config.get_color(curve_id)
        # Option 1: bar plots (e.g. for all_high_all_low)
        if self.args.bar:
            # X axis has only 2 points relative to the centers of the 2 bar
            # clusters: RKL and FKL
            x_pos = self.config.get_x_position(curve_id)
            # Don't call ax.bar before knowing the normalization limits,
            # since the bottom of the bar cannot be changed or known at
            # this point.
            # Append the function to self.call_buffer and call all of
            # them later
            plot_partial_call = functools.partial(self.ax.bar, x=x_pos, height=mean, yerr=stderr, color=color, width=self.config.width)
            self.call_buffer.append(plot_partial_call)
        # Option 2: sensitivity plots
        elif self.args.hyperparam_for_sensitivity is not None:
            # X axis are the logs of the hyperparameters (e.g. learning rates)
            x_axis = np.log10(self.args.all_hyper)
            self.ax.plot(x_axis, mean, color=self.config.get_color(curve_id), linestyle=self.config.linestyle, marker=self.config.marker, mew=self.config.mew, markersize=self.config.marker_size, linewidth=self.config.linewidth)
            self.ax.errorbar(x_axis, mean, yerr=stderr, color=self.config.get_color(curve_id), linestyle=self.config.linestyle, linewidth=self.config.linewidth_err)
        # Option 3: benchmark plots
        else:
            x_pos = np.array(self.x_range)
            self.ax.fill_between(x_pos, mean - stderr, mean + stderr, alpha=self.config.stderr_alpha, color=self.config.get_color(curve_id))
            self.ax.plot(x_pos, mean, linewidth=self.config.linewidth, color=color)

    def save_plot(self, save_dir):
        # Save to .png images
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        self.fig.savefig(os.path.join(save_dir, "{}.png".format(self.plot_name)), bbox_inches=self.config.savefig_bbox_in,dpi=self.config.savefig_dpi)
    
    def update_y_lim(self, new_ylim):
        # Change the old self.config ymax and ymin and update the plots accordingly.
        # This function is called after all self.plot_curve() calls are finished.
        # For the bar plots, this is where the plotting is done (read more on
        # self.plot_curve())
        new_ymin, new_ymax = new_ylim[0], new_ylim[1]
        if self.config.yscale == "log" and new_ymin <= 0:
            new_ymin = 1
        self.config.y_lim = (new_ymin, new_ymax)
        self.ax.get_yaxis().set_major_formatter(self.config.y_formatter)
        # Change y label depending on the new scale (e.g. change it to "Rewards (10^3)")
        # instead of "Rewards"
        self.ax.set_ylabel(self.config.ylabel)
        self.ax.set_ylim(bottom=new_ymin, top=new_ymax)
        # If we are using bar plots, this is the moment to plot them
        if self.args.bar and self.args.normalize:
            for partial_call in self.call_buffer:
                partial_call.keywords['height'] = partial_call.keywords['height'] - new_ymin
                partial_call(bottom=new_ymin)
            delta = new_ymax - new_ymin
            self.ax.set_yticks([new_ymin + pct * delta for pct in self.config.yticks_pct])


class PlotManager:
    # Class responsible for coordinating and synchronizing different Plotter and
    # PlotterDataObj elements. It is responsible for plotting, synchronizing Y axis,
    # saving processed data to be reused in subsequent calls...
    def __init__(self, args):
        self.args = args
        self.call_id = self.get_call_id()
        self.args.config_class = eval(args.config_class)
        # Dictionary with all Plotter and PlotDataObj entries
        self.plot_dict = {}
        self.env_results_dir = self.args.env_results_dir
        self.divide_type = args.divide_type
        self.how_to_group = args.how_to_group
        self.separate_agent_plots = args.separate_agent_plots
        # This dict stores the maximum and minimum between different plots
        # (obtained by accessing their .<max/min>_y_plotted info) and is
        # used to synchronize the Y axis of different plots
        self.sync_y_max_data = {}
        # This list is used in the sensitivity plots and stores all possible
        # X axis values
        self.args.all_hyper = []
        with open(self.args.env_json_fname, "r") as f:
            self.env_params = json.load(f, object_pairs_hook=OrderedDict)

    def get_call_id(self):
        # Uses args to identify the call (the call is unique within the environment)
        sep_id = "SplitAgents" if self.args.separate_agent_plots else "JointAgents"
        div_id = self.args.divide_type if self.args.divide_type is not None else "NoDivide"
        bar_id = ["bar"] if self.args.bar else []
        hypersens_id = ["hsens_{}".format(self.args.hyperparam_for_sensitivity)] if self.args.hyperparam_for_sensitivity is not None else []
        return "_".join([sep_id, self.args.how_to_group, div_id] + bar_id + hypersens_id)

    def add(self, plot_id, *f_args, **f_kwargs):
        # Creates new dictionary entry, unique to each plot (identified via plot_id)
        # this entry will have a 'data' element with the PlotDataObj and a 'plotter'
        # element, with the Plotter (the second one is created during plotting)
        if plot_id not in self.plot_dict:
            self.plot_dict[plot_id] = {'data': PlotDataObj(self.args, plot_id)}
        self.plot_dict[plot_id]['data'].add(*f_args, **f_kwargs)
        
    def load_existing_data(self, plot_id):
        # Called by the main function. For each plot, loads preprocessed data, so that
        # grouping runs does not have to be performed again by the PlotDataObj. Output
        # informs the caller whether loading was successful or if reading the files and
        # grouping runs will be necessary
        if hasattr(self.args, "preprocessed_dir"):
            full_fname = os.path.join(self.args.preprocessed_dir, "{}.pkl".format("_".join([self.call_id, plot_id])))
            if os.path.isfile(full_fname):
                if plot_id not in self.plot_dict:
                    self.plot_dict[plot_id] = {'data': PlotDataObj(self.args, plot_id)}
                    self.plot_dict[plot_id]['data'].load(pickle.load(open(full_fname,'rb')))
                return True
        return False
            
    def save_all_data(self):
        # For all PlotDataObj, saves the grouped (i.e. plot ready) data and, for sensitivity plots,
        # all posssible X values
        preprocessed_dir = self.args.preprocessed_dir
        for this_plot_id, this_plot_dict in self.plot_dict.items():
            plot_obj = this_plot_dict['data']
            ofname = os.path.join(preprocessed_dir, "{}.pkl".format("_".join([self.call_id, this_plot_id])))
            if not os.path.isfile(ofname):
                pickle.dump(obj={'generator_data': plot_obj.generator_data, 'all_hyper': self.args.all_hyper}, file=open(ofname, "wb"))

    def plot_and_save_all(self, synchronize_y_options=None):
        # Iterate through all plots, creates Plotter objects and uses them for
        # plotting
        sync = synchronize_y_options is not None
        for plot_id in self.plot_dict.keys():
            self.plot_dict[plot_id]['plotter'] = Plotter(self.call_id, plot_id, self.args, self.env_params)
            if sync:
                # Sync can be either y_idx or from_file (the second simply uses a file to
                # set the limits)
                if synchronize_y_options["mode"] == "y_idx":
                    # Obtains the value used for syncing from the plot_id (i.e. the name
                    # of the environment to sync FKL and RKL plots)
                    sync_value = plot_id.split("_")[synchronize_y_options["sync_idx"]]
            # Initializes plots
            plotter = self.plot_dict[plot_id]['plotter']
            plotter.initialize_plot()
            # Obtain plot-ready information from the PlotDataObj generator
            for curve_id, curve_mean, curve_stderr in self.plot_dict[plot_id]['data'].iterate():
                # Plots the curve (or caches the plot call if it is a bar plot)
                plotter.plot_curve(curve_id, curve_mean, curve_stderr)
                if sync:
                    if synchronize_y_options["mode"] == "y_idx":
                        # Accumulates max and min values accross plots
                        if sync_value not in self.sync_y_max_data:
                            self.sync_y_max_data[sync_value] = {'max': ExtremePoint(max), 'min': ExtremePoint(min)}
                        self.sync_y_max_data[sync_value]['max'].update(plotter.max_y_plotted.v)
                        self.sync_y_max_data[sync_value]['min'].update(plotter.min_y_plotted.v)
            # If the plot is normalized (e.g. all_high_all_low_bar), set the new Y limits
            # such that the max and min plotted values correspond to the central 80% portion
            # of the new limits
            if not sync and self.args.normalize:
                new_ymin, new_ymax = expand_limits(0.8, plotter.min_y_plotted.v, plotter.max_y_plotted.v)
                plotter.update_y_lim((new_ymin, new_ymax))
        # Synchronize Y values between plots (and calls the cached plots if it is a bar plot)
        if sync:
            self.synchronize_y_axis(synchronize_y_options)
        # Save the plot to file
        for this_plot_dict in self.plot_dict.values():
            this_plot_dict['plotter'].save_plot(self.args.plot_dir)

    def synchronize_y_axis(self, synchronize_y_options):
        # Option to save max per sync_value (e.g. environment name or exit type)
        # for syncing from file in subsequent calls
        if "save_max" in synchronize_y_options:
            out_max = {}
        # If syncing from file, load the preprocessed max data, which will be a dict
        # Example: {RightExit: (min, max), BadExit: (min, max)}
        if synchronize_y_options["mode"] == "from_file":
            max_dict = pickle.load(open(os.path.join(self.args.preprocessed_dir, "{}_max_data.pkl".format(synchronize_y_options["target_call_id"])), "rb"))

        # Iterate through all plots
        for plot_id, plot_dict in self.plot_dict.items():
            plotter = self.plot_dict[plot_id]['plotter']
            # Gets value used for synchronization
            sync_value = plot_id.split("_")[synchronize_y_options["sync_idx"]]
            # Option 1: sync the y axis of plots from current call
            # Expand the ymin and ymax (allows for some space between the plot limits and the curves)
            if synchronize_y_options["mode"] == "y_idx":
                # Keep the minimum value (if zero for example, it may not make sense to allow
                # the new ymin to be a negative value)
                if synchronize_y_options["keep_ymin"]:
                    new_ymin = self.sync_y_max_data[sync_value]['min'].v
                    new_ymax = self.sync_y_max_data[sync_value]['max'].v / 0.9
                # Expansion done on both sides
                else:
                    new_ymin, new_ymax = expand_limits(0.8, self.sync_y_max_data[sync_value]['min'].v, self.sync_y_max_data[sync_value]['max'].v)
                # Save max info for syncing of posterior calls
                if "save_max" in synchronize_y_options:
                    out_max[sync_value] = (new_ymin, new_ymax)
            # Option 2: sync the y axis using values read from a file
            elif synchronize_y_options["mode"] == "from_file":
                new_ymin, new_ymax = max_dict[sync_value]
            plotter.update_y_lim((new_ymin, new_ymax))
        # Save max info for syncing of posterior calls
        if "save_max" in synchronize_y_options:
            pickle.dump(out_max, open(os.path.join(self.args.preprocessed_dir, "{}_max_data.pkl".format(self.call_id)), "wb"))