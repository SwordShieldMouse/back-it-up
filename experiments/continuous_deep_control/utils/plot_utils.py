import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('TkAgg')
# mpl.use('Agg')
from matplotlib import pyplot as plt
import numpy as np


# Plot function for Continuous Bandits environment
def plotFunction(agent_name, func_list, state, greedy_action, expl_action, x_min, x_max, resolution=1e3, display_title='', save_title='',
                 save_dir='', linewidth=2.0, ep_count=0, grid=True, show=False, equal_aspect=False):
    fig, ax = plt.subplots(2, sharex=True)

    x = np.linspace(x_min, x_max, int(resolution))
    y1 = []
    y2 = []

    max_point_x = x_min
    max_point_y = np.float('-inf')

    if agent_name == 'KLDiv':
        func1, func2 = func_list[0], func_list[1]


        for point_x in x:
            point_y1 = np.squeeze(func1(point_x))  # reduce dimension
            point_y2 = np.squeeze(func2(point_x))

            if point_y1 > max_point_y:
                max_point_x = point_x
                max_point_y = point_y1

            y1.append(point_y1)
            y2.append(point_y2)

        ax[0].plot(x, y1, linewidth=linewidth)
        ax[1].plot(x, y2, linewidth=linewidth)

        if grid:
            ax[0].grid(True)
            ax[1].grid(True)
            ax[1].axhline(y=0, linewidth=1.5, color='darkslategrey')


            ax[1].axvline(x=greedy_action[0], linewidth=1.5, color='grey')
            ax[1].axvline(x=expl_action[0], linewidth=1.5, color='blue')

        if display_title:
            fig.suptitle(display_title, fontsize=11, fontweight='bold')

            mode_string = ""
            for i in range(len(greedy_action)):
                mode_string += "{:.2f}".format(np.squeeze(greedy_action[i]))

            ax[0].set_title("Action-values, argmax Q(S,A): {:.2f}".format(max_point_x[0]))
            ax[1].set_title("Policy, mean: {}".format(mode_string))

    # common
    if equal_aspect:
        ax.set_aspect('auto')

    if show:
        plt.show()

    else:
        # print(save_title)
        save_dir = save_dir + '/figures/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_dir + save_title)
        plt.close()
