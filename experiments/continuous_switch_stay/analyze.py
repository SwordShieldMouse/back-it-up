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
matplotlib.rcParams.update({'font.size': 40})

std_colours = {"std-0.1,0.3": "red", "std-0.3,0.5": "blue", "std-0.5,1": "green", "std-0.3,1.0": "blue", "std-0.1,1": "black"}
# std_names = {"std-0.1,0.3": r"$0.1 < \sigma_0 < 0.3$", "std-0.3,0.5": r"$0.3 < \sigma_0 < 0.5$", "std-0.5,1": r"$0.5 < \sigma_0 < 1$"}
# std_names = {"std-0.1,0.3": r"$0.1 < \sigma_0 < 0.3$", "std-0.3,1.0": r"$\sigma_0 > 0.3$"}
std_names = {"std-0.1,1": r"$0.1 < \sigma_0 < 1.0$"}
env = "cont-switch-stay"
std_partition = list(std_names.keys())[0]
foldername = "polytope/500samples/{}/notlearnQ".format(std_partition) ## MODIFY THIS
read_dir = "data/{}/{}/".format(env, foldername)
write_dir = "figs/{}/{}/".format(env, foldername)
basenames = [os.path.basename(name).rsplit('_', maxsplit=1)[0] for name in glob.glob(read_dir + "*.npy")]


all_time = False

colours = ['r', 'g', 'b', 'm']
figsize = (40, 20)
nth_point = 50

# get names of all the optimizers
# optim_names = set([re.search(r'(optim=)([^_]+)', s).group() for s in basenames])
optim_names = ["optim=adam", "optim=rmsprop"]
# get all possible lrs and temps
# lrs = sorted(set([float(re.search(r'(lr=)([^_]+)', s).group(2)) for s in basenames]))
# temps = sorted(set([float(re.search(r'(temp=)([^_]+)', s).group(2)) for s in basenames]))
lrs = [0.005]
temps = [0, 0.01, 0.1, 0.4, 1]
# print(read_dir)
polytope_file = glob.glob(read_dir + "*_polytope.npy")[0]
polytope_boundary = np.load(polytope_file) # read in the boundary at the beginning from switch-stay, not neccessarily continuous switch-stay

alpha = 0.01

for optim_name in optim_names:
    # separate by direction as well
    for direction in ["reverse", "forward"]:
        polytope_fig, polytope_axs = plt.subplots(len(list(std_names.keys())), len(temps), sharey=True, sharex=True, figsize = figsize, gridspec_kw={'hspace': 0.1, "wspace": 0.1})
        # go through lrs and temps in same order
        figname = direction + "_{}_lr={}".format(optim_name, lrs[-1])
        print("plotting " + figname)
        for i in range(len(list(std_names.keys()))):
        # for i in range(len(lrs)):
            for j in range(len(temps)):
                curr_write_dir = write_dir + optim_name + "/"
                os.makedirs(curr_write_dir, exist_ok=True)
                for basename in basenames:
                    # match strings to make sure there aren't any precision/conversion issues
                    str_lr = re.search(r'(lr=)([^_]+)', basename).group(2)
                    str_temp = re.search(r'(temp=)([^_]+)', basename).group(2)
                    if not all([lrs[-1] == float(str_lr), temps[j] == float(str_temp), direction in basename, optim_name in basename]):
                        continue
                    
                    if len(std_names.keys()) > 1:
                        polytope_axs[i, j].scatter(x = polytope_boundary[:, 0].flatten(), y=polytope_boundary[:, 1], alpha = 1, color = "black", s = 1)
                        
                        std_partition = list(std_names.keys())[i]
                        curr_read_dir = "data/cont-switch-stay/polytope/{}/notlearnQ/".format(std_partition)
                        if all_time is False:
                            V_values = np.load(curr_read_dir + basename + "_V.npy")[-1, :, :]
                            polytope_axs[i, j].scatter(x = V_values[:, 0], y = V_values[:, 1], alpha = alpha, color = std_colours[std_partition], s = 700)
                        # polytope_axs[i, j].set_xlim(left = -7, right = 20)
                        # polytope_axs[i, j].set_ylim(bottom=-5, top = 22)
                        else:
                            V_values = np.load(curr_read_dir + basename + "_V.npy")
                            for v in range(V_values.shape[0]):
                                polytope_axs[i, j].scatter(x = V_values[v, :, 0], y=V_values[v, :, 1], alpha = alpha, color = cm.jet(v / V_values.shape[0]), s = 200)
                        
                        
                        for axs in [polytope_axs]:
                            # if j == 0:
                            #     axs[i, j].set_ylabel(ylabel = std_names[list(std_names.keys())[i]], fontsize = 55)
                                # axs[i, j].set(ylabel="lr={}".format(lrs[i]))
                            if i == (len(list(std_names.keys())) - 1):
                                axs[i, j].set_xlabel(xlabel=r"$\tau$ = {}".format(temps[j]), fontsize = 65)
                    else:
                        polytope_axs[j].scatter(x = polytope_boundary[:, 0].flatten(), y=polytope_boundary[:, 1], alpha = 1, color = "black", s = 1)
                        
                        std_partition = list(std_names.keys())[i]
                        curr_read_dir = read_dir
                        if all_time is False:
                            V_values = np.load(curr_read_dir + basename + "_V.npy")[-1, :, :]
                            polytope_axs[j].scatter(x = V_values[:, 0], y = V_values[:, 1], alpha = alpha, color = std_colours[std_partition], s = 700)
                        # polytope_axs[i, j].set_xlim(left = -7, right = 20)
                        # polytope_axs[i, j].set_ylim(bottom=-5, top = 22)
                        else:
                            V_values = np.load(curr_read_dir + basename + "_V.npy")
                            for v in range(V_values.shape[0]):
                                polytope_axs[j].scatter(x = V_values[v, :, 0], y=V_values[v, :, 1], alpha = alpha, color = cm.jet(v / V_values.shape[0]), s = 200)
                        
                        
                        for axs in [polytope_axs]:
                            # if j == 0:
                            #     axs[i, j].set_ylabel(ylabel = std_names[list(std_names.keys())[i]], fontsize = 55)
                                # axs[i, j].set(ylabel="lr={}".format(lrs[i]))
                            if i == (len(list(std_names.keys())) - 1):
                                axs[j].set_xlabel(xlabel=r"$\tau$ = {}".format(temps[j]), fontsize = 65)
        
        plt.margins(0.3)
        for ax in polytope_axs.flat:
            ax.label_outer()

                
        # legend_elements = [Line2D([0], [0], marker='o', color='g', label='Reverse KL',
        #                   markerfacecolor='g', markersize=5), Line2D([0], [0], marker='o', color='b', label='ForwardKL',
        #                   markerfacecolor='b', markersize=5)]
        # dist_axs[-1, -1].legend(handles = legend_elements)
        # plt.legend()
        if all_time is False:
            polytope_fig.savefig(curr_write_dir + "polytope_" + figname + ".png", bbox_inches="tight", pad_inches=0.2)
        else:
            polytope_fig.savefig(curr_write_dir + "alltime_polytope_" + figname + ".png", bbox_inches="tight", pad_inches=0.2)
        plt.clf()
        plt.close("all")