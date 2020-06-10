import numpy as np 
from matplotlib import pyplot as plt
import os 
import glob
import re
import sys
import matplotlib as mpl
mpl.rcParams.update({'font.size': 30})

# mpl.rcParams["axes.labelsize"] = 15
# mpl.rcParams["axes.titlesize"] = 15

foldername = "notlearnQ" ## MODIFY THIS
read_dir = "data/discrete_bandit/{}/".format(foldername)
write_dir = "figs/discrete_bandit/{}/".format(foldername)
basenames = [os.path.basename(name).rsplit('_', maxsplit=1)[0] for name in glob.glob(read_dir + "*.npy")]

colours = ['r', 'g', 'b', 'm']
figsize = (40, 20)
nth_point = 50

# get names of all the optimizers
optim_names = set([re.search(r'(optim=)([^_]+)', s).group() for s in basenames])

# get all possible lrs and temps
lrs = sorted(set([float(re.search(r'(lr=)([^_]+)', s).group(2)) for s in basenames]))
temps = sorted(set([float(re.search(r'(temp=)([^_]+)', s).group(2)) for s in basenames]))

temp = np.load(read_dir + basenames[0] + "_prob.npy")
n_actions = temp.shape[1]
n_actions = 3
STEPS = (np.arange(1000) + 1)[::nth_point] # plot every n-th point
del temp

alpha = 0.01

for optim_name in optim_names:
    # separate by direction as well
    for direction in ["reverse", "forward"]:
        loss_fig, loss_axs = plt.subplots(len(lrs), len(temps), sharex=True, sharey=True, figsize = figsize, gridspec_kw={'hspace': 0.1, "wspace": 0.1})
        # plt.tight_layout(h_pad = 1.0)
        # plt.subplots_adjust(left=0.1, right = 0.2)
        prob_plots = [plt.subplots(len(lrs), len(temps), sharex=True, sharey=True, figsize = figsize, gridspec_kw={'hspace': 0.1, "wspace": 0.1}) for _ in range(n_actions)]
        # plt.tight_layout(h_pad = 1.0)
        pdf_fig, pdf_axs = plt.subplots(len(lrs), len(temps), sharex=True, sharey=True, figsize = figsize, gridspec_kw={'hspace': 0.1, "wspace": 0.1})
        # plt.tight_layout(h_pad = 1.0)
        # go through lrs and temps in same order
        figname = direction + "_{}".format(optim_name)
        print("plotting " + figname)
        for i in range(len(lrs)):
            for j in range(len(temps)):
                loss_fig.suptitle("loss_" + figname, size = 20)
                for prob_fig, prob_axs in prob_plots:
                    prob_fig.suptitle("prob_" + figname, size = 20)
                pdf_fig.suptitle("pdf_" + figname, size = 20)
                curr_write_dir = write_dir + optim_name + "/"
                os.makedirs(curr_write_dir, exist_ok=True)
                for basename in basenames:
                    # match strings to make sure there aren't any precision/conversion issues
                    str_lr = re.search(r'(lr=)([^_]+)', basename).group(2)
                    str_temp = re.search(r'(temp=)([^_]+)', basename).group(2)
                    if not all([lrs[i] == float(str_lr), temps[j] == float(str_temp), direction in basename, optim_name in basename]):
                        continue
                    
                    prob_values = np.load(read_dir + basename + "_prob.npy").squeeze()
                    loss_values = np.load(read_dir + basename + "_loss.npy").squeeze()
                    pdf = np.load(read_dir + basename + "_pdf.npy")
                    
                    # BQ will be different for different inits if learning Q
                    # will be of shape (n_points, n_inits)
                    # print(np.load(read_dir + basename + "_BQ.npy").shape)
                    BQ = np.load(read_dir + basename + "_BQ.npy")
                    
                    loss_scatter = []
                    prob_scatter = []
                    for step in STEPS:
                        for k in range(loss_values.shape[1]):
                            loss_scatter.append([step, loss_values[step - 1, k]])
                            prob_scatter.append(np.concatenate(([step], prob_values[step - 1, :, k])))
                    # print(np.min(loss_values))
                    # print(loss_values.shape)
                    loss_scatter = np.array(loss_scatter)
                    prob_scatter = np.array(prob_scatter)
                    # print(loss_scatter.shape)
                    loss_axs[i, j].scatter(x = loss_scatter[:, 0], y=loss_scatter[:, 1], alpha = alpha)
                    for k in range(n_actions):
                        prob_axs[i, j].scatter(x= prob_scatter[:, 0], y=prob_scatter[:, 1 + k], alpha = alpha, color = colours[k])
                    
                    width = 0.35
                    # print(BQ.shape, BQ)
                    x = np.arange(n_actions)
                    # print(BQ, x)
                    if len(BQ.shape) == 1:
                        pdf_axs[i, j].bar(x - width / 2, BQ, width = width, color = colours[0])
                    else:
                        pdf_axs[i, j].bar(x - width / 2, BQ[:, k], width = width, color = colours[0], alpha = alpha)
                    for k in range(pdf.shape[-1]):
                        pdf_axs[i, j].bar(x + width / 2, pdf[:, k], width = width, alpha = alpha, color = colours[1])
                    # set labels
                    # mean_axs[i, j].set_ylim(bottom = -1, top = 1.)
                    loss_axs[i, j].set_ylim(bottom = 0)
                    pdf_axs[i, j].set_ylim(top = 1, bottom = 0)
                    prob_axs[i, j].set_ylim(top = 1, bottom = 0)
                    for axs in [loss_axs, pdf_axs] + [prob_axs for _, prob_axs in prob_plots]:
                        if j == 0:
                            axs[i, j].set_ylabel(ylabel="lr = {}".format(lrs[i]), fontsize = 65)
                        if i == (len(lrs) - 1):
                            axs[i, j].set_xlabel(xlabel=r"$\tau$ = {}".format(temps[j]), fontsize = 65)
        
        # make labels outer only 
        for axs in [loss_axs, prob_axs, pdf_axs]:
            for ax in axs.flat:
                ax.label_outer()
        plt.margins(0.3)
        loss_fig.savefig(curr_write_dir + "loss_" + figname + ".png", bbox_inches="tight", pad_inches=0.2)
        prob_fig.savefig(curr_write_dir + "prob_" + figname + ".png", bbox_inches="tight", pad_inches=0.2)        
        pdf_fig.savefig(curr_write_dir + "pdf_" + figname + ".png", bbox_inches="tight", pad_inches=0.2)
        plt.clf()
        plt.close("all")