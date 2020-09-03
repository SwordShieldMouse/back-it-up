# Contents
`compute_heatmap.py` generates the Forward KL and the Reverse KL loss on a heatmap.
`compute_targetBQ.py` generates the target Boltzmann distribution with varying temperature.

# To run the experiment and plot
Modify the settings at the beginning of `compute_heatmap.py` or `compute_targetBQ.py` as desired.  

Example:

`python3 compute_heatmap.py --load_results False --save_dir ./SAVE/DIR --compute_kl_type forward` generates the forward KL heatmaps to the specified directory.
`python3 compute_targetBQ.py --save_dir ./SAVE/DIR` generates the plots showing target distribution and best distributions of forward KL and reverse KL to the specified directory.
