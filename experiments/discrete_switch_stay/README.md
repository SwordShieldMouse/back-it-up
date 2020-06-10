# Contents
This experiment records the value function of learned policies on switch-stay for later visualization on the value function polytope.

# To run the experiments
To run the parameter sweeps, call `./experiments/switch-stay/run.sh`. 
To run an individual experiment, call `python3 experiments/switch-stay/run.py experiment_index`.

# To plot the data
Call `python3 experiments/switch-stay/analyze.py`. An appropriate subfolder will be created in the `figs/` folder at the root. 