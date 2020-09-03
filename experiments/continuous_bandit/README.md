# Contents
This experiment compares Forward KL and Reverse KL on a continuous bandit. 

# To run the experiment
Modify the hyperparameters in `run.py` as desired. Call either `./experiments/continuous_bandit/run.sh` or `python3 experiments/continuous_bandit/run.py experiment_index`. To run the sample-based experiments, modify the variable `n_action_points` in `sample_run.py` with the desired number of sample points, and then run `sample_run.py` or `sample_run.sh` as above. 

# To plot the data
Modify `analyze.py` to point to the correct folder where the data is stored, then call `python3 experiments/continuous_bandit/analyze.py`.

# Sweeping hyperparameters
A unique index is generated for each algorithm + hyperparameter combination. Passing in `experiment_index` allows you to specify which combination to run. 