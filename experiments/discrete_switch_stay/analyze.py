import numpy as np 
from matplotlib import pyplot as plt
import os 
import glob
import re
import sys
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from colour import Color
matplotlib.rcParams.update({'font.size': 30})

# matplotlib.rcParams["axes.labelsize"] = 15
# matplotlib.rcParams["axes.titlesize"] = 15


env = "switch-stay"
foldername = "polytope/notlearnQ" ## MODIFY THIS
read_dir = "data/{}/{}/".format(env, foldername)
write_dir = "figs/{}/{}/".format(env, foldername)
basenames = [os.path.basename(name).rsplit('_', maxsplit=1)[0] for name in glob.glob(read_dir + "*.npy")]


colours = ['r', 'g', 'b', 'm']
figsize = (40, 20)
nth_point = 50

# get names of all the optimizers
optim_names = set([re.search(r'(optim=)([^_]+)', s).group() for s in basenames])

# get all possible lrs and temps
# lrs = sorted(set([float(re.search(r'(lr=)([^_]+)', s).group(2)) for s in basenames]))
# temps = sorted(set([float(re.search(r'(temp=)([^_]+)', s).group(2)) for s in basenames]))
lrs = [0.01]
temps = [0, 0.01, 0.1, 0.5, 1]


polytope_boundary = np.load("data/switch-stay/{}/".format(foldername) + basenames[0] + "_polytope.npy") # read in the boundary at the beginning from switch-stay, not neccessarily continuous switch-stay

alpha = 0.01

for optim_name in optim_names:
    # separate by direction as well
    for direction in ["reverse", "forward"]:
        polytope_fig, polytope_axs = plt.subplots(len(lrs), len(temps), sharex=True, sharey=True, figsize = figsize, gridspec_kw={'hspace': 0.1, "wspace": 0.1})
        # plt.tight_layout(h_pad = 1.0)
        # go through lrs and temps in same order
        figname = direction + "_{}".format(optim_name)
        print("plotting " + figname)
        for i in range(len(lrs)):
            for j in range(len(temps)):
                polytope_fig.suptitle("polytope_" + figname, size = 20)
                curr_write_dir = write_dir + optim_name + "/"
                os.makedirs(curr_write_dir, exist_ok=True)
                for basename in basenames:
                    # match strings to make sure there aren't any precision/conversion issues
                    str_lr = re.search(r'(lr=)([^_]+)', basename).group(2)
                    str_temp = re.search(r'(temp=)([^_]+)', basename).group(2)
                    if not all([lrs[i] == float(str_lr), temps[j] == float(str_temp), direction in basename, optim_name in basename]):
                        continue
                    
                    V_values = np.load(read_dir + basename + "_V.npy")[-1, :, :]

                    # V_values = np.load(read_dir + basename + "_V.npy")[::nth_point, :, :]
                    # V_colors = list(Color("blue").range_to(Color("red"), V_values.shape[0]))
                    # print(polytope_boundary.shape)
                    polytope_axs[i, j].scatter(x = polytope_boundary[:, 0].flatten(), y=polytope_boundary[:, 1], alpha = 0.5, color = 'black', s = 1)
                    polytope_axs[i, j].scatter(x = V_values[:, 0], y=V_values[:, 1], alpha = alpha, color = "red", s = 700)
                    # for v in range(V_values.shape[0]):
                        # print(V_colors)
                        # polytope_axs[i, j].scatter(x = V_values[v, :, 0], y=V_values[v, :, 1], alpha = alpha, color = cm.jet(v / V_values.shape[0]))
                    # polytope_axs[i, j].scatter(x = V_values[:, :, 0].flatten(), y=V_values[:, :, 1].flatten(), alpha = 0.05, color = colours[0])
                    
                    for axs in [polytope_axs]:
                        if j == 0:
                            axs[i, j].set_ylabel(ylabel="lr = {}".format(lrs[i]), fontsize = 65)
                        if i == (len(lrs) - 1):
                            axs[i, j].set_xlabel(xlabel=r"$\tau$ = {}".format(temps[j]), fontsize = 65)
        
        for ax in polytope_axs.flat:
            ax.label_outer()
        plt.margins(0.3)
        
        # legend colour bar
        # normalize = mcolors.Normalize(vmin=0, vmax=500)
        # scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cm.jet)
        # scalarmappaple.set_array([])
        # cbar = polytope_fig.colorbar(scalarmappaple)
        # cbar.set_label("gradient steps")

        # plt.legend()

        polytope_fig.savefig(curr_write_dir + "polytope_" + figname + ".png", bbox_inches="tight", pad_inches=0.2)
        plt.clf()
        plt.close("all")