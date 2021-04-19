#!python
from pylab import *
from numpy import outer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

##### Line All high All low plots
# initial_rkl_color = np.array((0, 77, 0))/255.
# final_rkl_color = np.array((0, 204, 0))/255.

# initial_fkl_color = np.array((0, 26, 51))/255.
# final_fkl_color = np.array((102, 181, 255))/255.

#### Summary bar plots
initial_rkl_color = np.array((0, 128, 66))/255.
final_rkl_color = np.array((0, 204, 105))/255.

initial_fkl_color = np.array((0, 66, 128))/255.
final_fkl_color = np.array((0, 119, 230))/255.

colours = {}

colours["ReverseKL"] = {'high': final_rkl_color, 'low': initial_rkl_color}
colours["ForwardKL"] = {'high': final_fkl_color, 'low': initial_fkl_color}

legend_elements = [
                   Line2D([0], [0], linewidth = 6.0, marker=None, color=colours['ReverseKL']['high'], label='High RKL'),
                   Line2D([0], [0], linewidth = 6.0, marker=None, color=colours['ReverseKL']['low'], label='Low RKL'),
                   Line2D([0], [0], linewidth = 6.0, marker=None, color=colours['ForwardKL']['high'], label='High FKL'),
                   Line2D([0], [0], linewidth = 6.0, marker=None, color=colours['ForwardKL']['low'], label='Low FKL')
                  ]

plt.axis('off')
plt.legend(handles=legend_elements, fontsize=30, ncol=2)
plt.savefig('general_figs/legend__.png',bbox_inches='tight')
# plt.show()