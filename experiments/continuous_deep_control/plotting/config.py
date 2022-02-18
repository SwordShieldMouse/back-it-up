import numpy as np
from matplotlib import ticker

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

class CMPlotConfig(PlotConfig):
    xlabel = 'Timesteps ($10^3$)'
    x_formatter_divider = 1000
    y_lim = None

    yscale = "linear"

    @property
    def y_formatter(self):
        regular = ticker.FuncFormatter(lambda x, pos: '{}'.format(int(x)))
        thousands = ticker.FuncFormatter(lambda x, pos: '{}'.format(int(x)/ 1000))
        if self.y_lim is not None:
            if self.y_lim[1] - self.y_lim[0] > 1000:
                return thousands
        return regular

    @property
    def ylabel(self):
        regular = "Times reached"
        thousands = "Times reached ($10^3$)"
        if self.y_lim is not None:
            if self.y_lim[1] - self.y_lim[0] > 1000:
                return thousands
        return regular

    entropy_color_dict = {
                        1000: 'blue',
                        100: 'orange',
                        10: 'green',
                        1: 'red',
                        0.1: 'purple',
                        0.01: 'yellow',
                        0.001: 'magenta',
                        }
    kl_color_dict = {
                    'HardReverseKL': np.array((0, 128, 66))/255.,
                    'ReverseKL': np.array((0, 204, 105))/255.,
                    'HardForwardKL': np.array((0, 66, 128))/255.,
                    'ForwardKL': np.array((0, 119, 230))/255.
                    }

    def get_color(self, key, divide_type):
        if isinstance(key, tuple):
            agent = key[0]
            divider = key[1]

        if divide_type is None:
            return self.kl_color_dict[key]
        else:
            if divide_type == 'entropy_scale':
                return self.entropy_color_dict[divider]
            else:
                raise NotImplementedError