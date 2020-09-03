# Contents
Records the value functions of learned Tanh-Gaussian policies on the continuous version of Switch-Stay. 

# To run the experiments
To run the parameter sweeps, call `./experiments/continuous-switch-stay/run.sh`. To run an individual experiment, call `python3 experiments/continuous-switch-stay/run.py experiment_index`. To run the sample-based experiments, modify `n_action_points` in `sample_run.py` to the desired number of sample points and then run either `sample_run.py` or `sample_run.sh` as above. 

# To plot the data
Call `python3 experiments/continuous-switch-stay/analyze.py`. An appropriate subfolder will be created in the `figs/` folder at the root. 