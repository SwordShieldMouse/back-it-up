import numpy as np 
from matplotlib import pyplot as plt
import os 
import glob
import re
import sys
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.colors

matplotlib.rcParams.update({'font.size': 30})

n_modes = 1
foldername = "sample10_notlearnQ" ## MODIFY THIS
read_dir = "data/continuous_bandit/{}/nmodes={}/".format(foldername, n_modes)
write_dir = "figs/continuous_bandit/{}/nmodes={}/".format(foldername, n_modes)
basenames = [os.path.basename(name).rsplit('_', maxsplit=1)[0] for name in glob.glob(read_dir + "*.npy")]

def std_transform(param):
    return np.log(1 + np.exp(param))

COLOURS = ['r', 'g', 'b']
figsize = (40, 20)
nth_point = 50

# get names of all the optimizers
optim_names = set([re.search(r'(optim=)([^_]+)', s).group() for s in basenames])

# get all possible lrs and temps
# lrs = sorted(set([float(re.search(r'(lr=)([^_]+)', s).group(2)) for s in basenames]))
# temps = sorted(set([float(re.search(r'(temp=)([^_]+)', s).group(2)) for s in basenames]))
lr = 0.005
# temps = [0, 0.01, 0.1, 1]
temps = [0, 0.01, 0.1, 0.4, 1]

temp = np.load(read_dir + basenames[0] + "_mu.npy")
STEPS = (np.arange(1000) + 1)[::nth_point] # plot every n-th point
del temp

alpha = 0.01

std_colours = ["red", "blue", "green"]
# std_names = [r"$\sigma_0 < 0.3$", r"$0.3 \leq \sigma_0 \leq 0.5$", r"$0.5 < \sigma_0$"]
std_names = [r"$\sigma_0 < 0.3$", r"$\sigma_0 > 0.3$"]

for optim_name in optim_names:
    # separate by direction as well
    for direction in ["reverse", "forward"]:
        loss_fig, loss_axs = plt.subplots(len(std_names), len(temps), sharex=True, sharey = True, figsize = figsize, gridspec_kw={'hspace': 0.1, "wspace": 0.1})
        # plt.tight_layout(h_pad = 1.0)
        # plt.subplots_adjust(left=0.1, right = 0.2)
        mean_fig, mean_axs = plt.subplots(len(std_names), len(temps), sharex=True, sharey=True, figsize = figsize, gridspec_kw={'hspace': 0.1, "wspace": 0.1})
        # plt.tight_layout(h_pad = 1.0)
        std_fig, std_axs = plt.subplots(len(std_names), len(temps), sharex=True, sharey = True, figsize = figsize, gridspec_kw={'hspace': 0.1, "wspace": 0.1})
        # plt.tight_layout(h_pad = 1.0)
        pdf_fig, pdf_axs = plt.subplots(len(std_names), len(temps), sharex=True, figsize = figsize, gridspec_kw={'hspace': 0.1, "wspace": 0.1})
        # plt.tight_layout(h_pad = 1.0)
        figname = direction + "_{}_modes={}_lr={}".format(optim_name, n_modes, lr)
        print("plotting " + figname)
        for i in range(len(std_names)):
            for j in range(len(temps)):
                # loss_fig.suptitle("loss_" + figname, size = 20)
                # mean_fig.suptitle("mean_" + figname, size = 20)
                # std_fig.suptitle("std_" + figname, size = 20)
                # pdf_fig.suptitle("pdf_" + figname, size = 20)
                curr_write_dir = write_dir + optim_name + "/"
                os.makedirs(curr_write_dir, exist_ok=True)
                for basename in basenames:
                    # match strings to make sure there aren't any precision/conversion issues
                    str_lr = re.search(r'(lr=)([^_]+)', basename).group(2)
                    str_temp = re.search(r'(temp=)([^_]+)', basename).group(2)
                    if not all([lr == float(str_lr), temps[j] == float(str_temp), direction in basename, optim_name in basename]):
                        continue
                    # curr_write_dir = write_dir + optim_name + "/"

                    mu_values = np.load(read_dir + basename + "_mu.npy")[::nth_point, :]
                    logstd_values = np.load(read_dir + basename + "_logstd.npy")[::nth_point, :]
                    std_values = std_transform(logstd_values)
                    loss_values = np.load(read_dir + basename + "_loss.npy")[::nth_point, :]
                    pdf = np.load(read_dir + basename + "_pdf.npy")[::5, :]
                    points = np.load(read_dir + basename + "_points.npy")[::5].reshape((-1, 1))
                    # BQ = np.load(read_dir + basename + "_BQ.npy")[::20] 
                    BQ = np.load(read_dir + basename + "_BQ.npy")[::5, :] 
                    if i == 0:
                        init_ixs = np.where(std_values[0, :] < 0.3)[0]
                    elif i == 1:
                        init_ixs = np.where(std_values[0, :] > 0.3)[0]
                    # elif i == 1:
                    #     init_ixs = np.where(np.logical_and(std_values[0, :] >= 0.3, std_values[0, :] <= 0.5))[0]
                    # else:
                    #     init_ixs = np.where(std_values[0, :] > 0.5)[0]
                    
                    points = np.repeat(points, repeats = len(init_ixs), axis = 1)
                    steps_repeat = np.repeat(STEPS.reshape((-1, 1)), repeats = len(init_ixs), axis = 1)
                    pdf_axs[i, j].scatter(points, BQ[:, init_ixs], color = COLOURS[0], alpha = alpha / 2)
                    pdf_axs[i, j].scatter(points, pdf[:, init_ixs], color = COLOURS[1], alpha = alpha / 2)
                    mean_axs[i, j].scatter(x=steps_repeat, y=mu_values[:, init_ixs], alpha = alpha, color = std_colours[i])
                    std_axs[i, j].scatter(x=steps_repeat, y=std_transform(logstd_values[:, init_ixs]), alpha = alpha, color = std_colours[i])
                    loss_axs[i, j].scatter(x = steps_repeat, y=loss_values[:, init_ixs], alpha = alpha, color = std_colours[i])
                    
                    pdf_axs[i, j].set_xlim(left = -1, right = 1)
                    mean_axs[i, j].set_xlim(left = 0, right = 1000)
                    std_axs[i, j].set_xlim(left = 0, right = 1000)
                    loss_axs[i, j].set_xlim(left = 0, right = 1000)
                    mean_axs[i, j].set_ylim(top = 1.5, bottom = -1.5)
                    mean_axs[i, j].set_yticks([1.5, 1, 0.5, 0, -0.5, -1, -1.5])
                    # mean_axs[i, j].set_yticks([-3, -2, -1, 0, 1, 2, 3])
                    std_axs[i, j].set_ylim(top = 2., bottom = 0)
                    # loss_axs[i, j].set_ylim(bottom = 0)
                    
                    
                    # pdf_axs[i, j].scatter(points.flatten(), BQ.flatten(), color = COLOURS[0], alpha = alpha)
                    # # pdf_axs[i, j].plot(points, BQ, color = colours[0])
                    # pdf_axs[i, j].scatter(points.flatten(), pdf.flatten(), alpha = alpha, color = COLOURS[1])
                    # set labels
                    # mean_axs[i, j].set_ylim(bottom = -1, top = 1.)
        
        for i in range(len(std_names)):
            for j in range(len(temps)):
                for axs in [loss_axs, mean_axs, std_axs, pdf_axs]:
                    if j == 0:
                        axs[i, j].set_ylabel(ylabel=std_names[i], fontsize = 60, rotation = 90)
                    if i == (len(std_names) - 1):
                        axs[i, j].set_xlabel(xlabel=r"$\tau$ = {}".format(temps[j]), fontsize = 65)
                    if j== 0 or i == (len(std_names) - 1):
                        if axs is not pdf_axs:
                            a = axs[i, j].get_xticks().tolist()
                            # a[-1] = str(a[-1]) + " steps"
                            for k in range(len(a)):
                                if k == len(a) - 1:
                                    a[k] = str(int(a[k])) + " steps"
                                else:
                                    a[k] = str(int(a[k]))
                            axs[i, j].set_xticklabels(a, rotation = 45, ha = "right")
                            # axs[i, j].xticks(rotation = -30)
        # for ax in mean_axs.flat:
        #     ax.label_outer()
        # # make labels outer only 
        for axs in [loss_axs, mean_axs, std_axs]:
            for ax in axs.flat:
                ax.label_outer()
        plt.margins(0.3)

        # legend for mean axis
        # legend_elements = [Line2D([0], [0], marker='o', color='r', label='std dev < 0.3',
        #                   markerfacecolor='r', markersize=5), Line2D([0], [0], marker='o', color='g', label='std > 0.5', markersize=5), Line2D([0], [0], marker='o', color='b', label='0.3 <= std <= 0.5', markersize=5)]
        # mean_axs[-1, -1].legend(handles = legend_elements)

        loss_fig.savefig(curr_write_dir + "loss_" + figname + ".png", bbox_inches="tight", pad_inches=0.2)
        mean_fig.savefig(curr_write_dir + "mean_" + figname + ".png", bbox_inches="tight", pad_inches=0.2)        
        std_fig.savefig(curr_write_dir + "std_" + figname + ".png" , bbox_inches="tight", pad_inches=0.2)        
        pdf_fig.savefig(curr_write_dir + "pdf_" + figname + ".png", bbox_inches="tight", pad_inches=0.2)
        plt.clf()
        plt.close("all")