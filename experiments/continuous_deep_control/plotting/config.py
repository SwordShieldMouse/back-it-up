import numpy as np
from matplotlib import ticker
from utils.main_utils import tryeval

def split_key(key):
    key = key.split("_")
    agent = key[0]
    if len(key) > 1:
        divider = tryeval(key[1])
    else:
        divider = None
    return agent, divider

def interpolate_colors(initial, final, n_idxs):
    output = []
    for idx in range(n_idxs):
        t = float(idx) / (n_idxs - 1)
        color = initial * (1 - t) + t * final
        output.append(color)
    return output

class PlotConfig:
    font_size = 35
    ylabel_fontsize = 50
    xlabel_fontsize = 50
    figsize = (12, 12)
    n_xticks = 5
    stderr_alpha = 0.2
    linewidth = 1.0
    savefig_bbox_in = 'tight'
    savefig_dpi = 100
    ylabel_rotation = 90
    episodes_window = 20
    y_lim = None
    x_lim = None
    yscale = "linear"
    xscale = "linear"
    x_str = "Timesteps"
    y_str = "Reward"
    normalize_formatter = False

    def __init__(self, args):
        self.args = args

    @property
    def y_formatter(self):
        return self.formatter('y_lim')

    @property
    def ylabel(self):
        return self.label(self.y_str, 'y_lim')

    @property
    def x_formatter(self):
        return self.formatter('x_lim')

    @property
    def xlabel(self):
        return self.label(self.x_str, 'x_lim')

    def formatter(self, attr):
        def _normalized(x, pos):
            if getattr(self, attr) is not None:
                _max = getattr(self, attr)[1]
                _min = getattr(self, attr)[0]
                n = float(x - _min)/(_max-_min)
                return '{:.2f}'.format(n)
            else:
                return '{}'.format(int(x))
        regular = ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x))
        thousands = ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x / 1000))
        hundreds_of_thousands = ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x / 1e5))
        millions = ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x / 1e6))
        normalized = ticker.FuncFormatter(_normalized)
        if self.normalize_formatter is False:
            if getattr(self, attr) is not None:
                dif = getattr(self, attr)[1] - getattr(self, attr)[0]
                if dif > 1000 and dif < 1e5:
                    return thousands
                elif dif > 1e5 and dif < 1e6:
                    return hundreds_of_thousands
                elif dif > 1e6:
                    return millions
        else:
            return normalized
        return regular

    def label(self, id_str, attr):
        regular = id_str
        thousands = "{} ($10^3$)".format(id_str)
        hundreds_of_thousands = "{} ($10^5$)".format(id_str)
        millions = "{} ($10^6$)".format(id_str)
        if self.normalize_formatter is False:
            if getattr(self, attr) is not None:
                dif = getattr(self, attr)[1] - getattr(self, attr)[0]
                if dif > 1000 and dif < 1e5:
                    return thousands
                elif dif > 1e5 and dif < 1e6:
                    return hundreds_of_thousands
                elif dif > 1e6:
                    return millions
        return regular

    kl_color_dict = {
                    'HardReverseKL': np.array((0, 128, 66))/255.,
                    'ReverseKL': np.array((0, 204, 105))/255.,
                    'HardForwardKL': np.array((0, 66, 128))/255.,
                    'ForwardKL': np.array((0, 119, 230))/255.
                    }


class CMPlotConfig(PlotConfig):
    x_str = "Timesteps"
    ylabel = "Times Reached"

    yscale = "log"
    y_formatter = ticker.FuncFormatter(lambda x, pos: '{:.2e}'.format(x))

    entropy_color_dict = {
                        1000.: 'blue',
                        100.: 'orange',
                        10.: 'green',
                        1.: 'red',
                        0.1: 'purple',
                        0.01: 'yellow',
                        0.001: 'magenta',
                        }

    def get_color(self, key):
        divide_type = self.args.divide_type
        agent, divider = split_key(key)

        if divide_type is None:
            return self.kl_color_dict[agent]
        else:
            if divide_type == 'entropy_scale':
                return self.entropy_color_dict[divider]
            else:
                raise NotImplementedError

class BenchmarksPlotConfig(PlotConfig):
    font_size = 50
    figsize = (18, 12)
    stderr_alpha = 0.2
    
    high_temps = [100., 10., 1., 0.5, 0.1]
    low_temps = [0.05, 0.01, 0.005, 0.001, 0.0]
    @property
    def entropy_and_kl_color_dict(self):
        temps = [0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.]
        temps = sorted(temps)
        colors = {}

        initial_rkl_color = np.array((0, 51, 26))/255.
        final_rkl_color = np.array((204, 255, 204))/255.
        rkl_colors = interpolate_colors(initial_rkl_color, final_rkl_color, len(temps))

        initial_fkl_color = np.array((0, 26, 51))/255.
        final_fkl_color = np.array((204, 230, 255))/255.
        fkl_colors = interpolate_colors(initial_fkl_color, final_fkl_color, len(temps))

        colors["ReverseKL"] = dict(zip(temps, rkl_colors))
        colors["ForwardKL"] = dict(zip(temps, fkl_colors))
        return colors
    
    high_low_color_dict = {
        "ReverseKL": {
            "low": np.array((0, 77, 0))/255.,
            "high": np.array((0, 204, 0))/255.
        },
        "ForwardKL": {
            "low": np.array((0, 26, 51))/255.,
            "high": np.array((102, 181, 255))/255.
        }
    }

    def get_color(self, key):
        divide_type = self.args.divide_type
        agent, divider = split_key(key)
        if divide_type is None:
            return self.kl_color_dict[agent]
        else:
            if divide_type == 'entropy_scale':
                return self.entropy_and_kl_color_dict[agent][divider]
            elif divide_type == 'all_high_all_low':
                return self.high_low_color_dict[agent][divider]
            else:
                raise NotImplementedError

class BenchmarksBarPlotConfig(BenchmarksPlotConfig):
    high_low_color_dict = {
        "ReverseKL": {
            "low": np.array((0, 128, 66))/255.,
            "high": np.array((0, 204, 105))/255.
        },
        "ForwardKL": {
            "low": np.array((0, 66, 128))/255.,
            "high": np.array((0, 119, 230))/255.
        }
    }

    width = 0.05
    intra_offset = 0.01
    inter_offset = 0.1

    ylabel = "AUC @ 0.5"
    xlabel = ""

    locator_params_kwargs = {
        "axis": "y",
        "nbins": 5,
    }

    @property
    def x_formatter(self):
        return ticker.FuncFormatter(lambda x, pos: "RKL" if x == 0 else "FKL")

    xticks = [0.0, intra_offset + 2* width + inter_offset]
    yticks_pct = [0.0, 0.25, 0.5, 0.75, 1.0]

    x_axis_dict = {
        "ReverseKL": {
            "low": width/2. + intra_offset/2.,
            "high": -width/2. - intra_offset/2.
        },
        "ForwardKL": {
            "low": (width/2. + intra_offset/2.) + (inter_offset + 2*width + intra_offset),
            "high": (-width/2. - intra_offset/2.) + (inter_offset + 2*width + intra_offset)
        }
    }

    def get_x_position(self, key):
        assert self.args.divide_type == 'all_high_all_low'
        agent, divider = split_key(key)
        return self.x_axis_dict[agent][divider]

class HyperSensPlotConfig(BenchmarksPlotConfig):
    param = None
    translate_param = {"pi_lr": "Actor lr", "qf_vf_lr": "Critic lr"}

    linewidth = 5.0
    linewidth_err = 5
    figsize = (18, 12)
    mew = 3
    marker_size = 15
    dashes = (5, 0)
    marker = "."
    linestyle = "-"

    y_str = "Average 0.5-AUC"

    @property
    def xlabel(self):
        return r"$\log_{{10}}$" + "({})".format(self.translate_param[self.args.hyperparam_for_sensitivity])