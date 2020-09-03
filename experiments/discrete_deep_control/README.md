# Contents
This experiment contains the KL experiments for discrete action space environments and non-linear function approximation. 

# To run experiments
Modify `deep_sweep.py` for desired hyperparameters and environment. Then, either:
1. Call `python3 experiments/deep_control/deep_sweep.py experiment_index max_frames max_frames_per_episode`
2. Call `./experiments/deep_control/deep_script.sh`

`deep_sweep.py` will save data to the directory `data/deep_control/env_name/`. The directory will be created automatically if it does not exist.

# To plot the data
Call `plotting/kl_results.py desired_auc_fraction env_index`, where `env_index` corresponds to the array `ENV_NAMES` in `kl_results.py`.

# Sweeping hyperparameters
`deep_sweep.py` uses auxiliary functions to generate a unique experiment index for every algorithm + hyperparameter combination. One of the first lines of the output of `deep_sweep.py` is the current experiment index out of the total number of experiments. This feature allows for parallelizing the runs; a slurm script was used for the parallelization of our runs and is not included here. 

# NB
To run the MinAtar experiments, you will have to clone the repo: https://github.com/kenjyoung/MinAtar.