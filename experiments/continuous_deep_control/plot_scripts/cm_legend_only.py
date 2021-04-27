#!python
from pylab import *
from numpy import outer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

entropy_color_dict = {
                    1000: 'blue',
                    100: 'orange',
                    10: 'green',
                    1: 'red',
                    0.1: 'purple',
                    }

legend_elements = [
                   Line2D([0], [0], linewidth = 6.0, marker=None, color=value, label='$\\tau = {}$'.format(key) )
                   for key,value in entropy_color_dict.items()
                  ]

plt.axis('off')
plt.legend(handles=legend_elements, fontsize=30, ncol=3)
plt.savefig('general_figs/cm_all_legend__.png',bbox_inches='tight')

legend_elements = [
                    Line2D([0], [0], linewidth = 6.0, marker=None, color='black', label='Misleading exit', linestyle='--' ),
                   Line2D([0], [0], linewidth = 6.0, marker=None, color='black', label='Good exit'),
                  ]

plt.axis('off')
plt.legend(handles=legend_elements, fontsize=30, ncol=2)
plt.savefig('general_figs/cm_APPENDIX_all_legend__.png',bbox_inches='tight')                  