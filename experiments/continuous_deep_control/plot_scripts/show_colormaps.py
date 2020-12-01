#!python
from pylab import *
from numpy import outer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

cdict = {'red': ((0.0, 0.0, 0.0),
                 (1.0, 204./255., 204./255.)),
         'green': ((0.0, 26./255., 26./255.),
                   (1.0, 230./255., 230./255.)),
         'blue': ((0.0, 51./255., 51./255.),
                  (1.0, 255./255., 255./255.))}

cmap_dict = {'ForwardKL': None, 'ReverseKL': None}

my_cmap = matplotlib.colors.LinearSegmentedColormap('fkl_colormap',cdict,256)

cmap_dict['ForwardKL'] = my_cmap

def truncate_colormap(cmap, minval=0.65, maxval=.99, n=1000):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

jet_cmap = plt.get_cmap('jet')
trunc_cmap = truncate_colormap(jet_cmap)

cmap_dict['ReverseKL'] = trunc_cmap

a=np.transpose(outer(arange(0,1,0.01),ones(5)), [1,0])

xticks_f = ticker.FuncFormatter(
    lambda x, pos: '{}'.format((x - 0) / (a.shape[1]-1)))

matplotlib.rcParams.update({'font.size': 14})

plt.figure(figsize=(3.5,0.7))
for i, (kl, m) in enumerate(cmap_dict.items()):
    paxis = plt.gca()
    paxis.yaxis.set_visible(False)
    paxis.set_xlabel('$\\tau$', labelpad=-15)
    paxis.xaxis.set_major_formatter(xticks_f)
    paxis.set_xticks([0, (a.shape[1]-1)])    
    imshow(a,cmap=m,origin="lower")
    title(kl,rotation=0,fontsize=14)
    savefig("general_figs/colormaps_{}.png".format(kl),dpi=100)
    # plt.show()
