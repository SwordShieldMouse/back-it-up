# Contents
This experiment contains the KL experiments for continuous action space environments and non-linear function approximation. 

# To install dependencies
Dependencies are managed by pipenv and versions are all stored in `Pipfile` and `Pipfile.lock`. To automatically install them, first install pipenv, then run:

```
pipenv install
```

To install all dependencies. In Compute Canada, it may be the case that the exact versions used in our virtual environment are not available, in which case the recommendation is to use a base Singularity environment with python and pipenv installed and then run pipenv from there.

# To run experiments
- Modify agent sweep settings in `jsonfiles/agent/<path_to_agent_json>.json`
- Modify environment settings in `jsonfiles/environment/<path_to_env_json>.json`.
- Run: 
```
python3 nonlinear_run.py --env_json <full_path_to_env_json> --agent_json <full_path_to_agent_json> --index <idx>
```

Where `idx` is a unique number defining the exact setting from the agent json. In case this number is larger than the number of settings, the modulo of the division by the number of settings is used to get the corresponding setting, but with a different random seed.

After the experiment is run, results are stored in `--out_dir`, which defaults to `results/_results`. A directory with the environment name is created there and the following files are generated:
- `<env>_<agent>_agent_Params.json`: dictionary with all parameters and values used in the sweep, 
- `<env>_<agent>_setting_<sett_n>_run_<seed_n>_agent_Params.txt`: parameter values of the specific setting used in the run writen in a CSV-like format,
- `<env>_<agent>_setting_<sett_n>_run_<seed_n>_agent_EpisodeRewardsLC.txt`: array where i-th entry is the reward of the i-th episode. Written in numpy format, to open it, use `np.loadtxt(<file_name.txt>, delimiter=',')`
- `<env>_<agent>_setting_<sett_n>_run_<seed_n>_agent_EpisodeStepsLC.txt`: array where i-th entry is the cumulative count of all episode steps by the end of episode i. Also written in numpy format,
- `<env>_<agent>_setting_<sett_n>_run_<seed_n>_agent_EvalEpisodeMeanRewardsLC.txt`: not used,
- `<env>_<agent>_setting_<sett_n>_run_<seed_n>_agent_EvalEpisodeStdRewardsLC.txt`: not used

Mujoco must be installed in order to run Reacher-v2 and Swimmer-v2 environments.

# To plot the data

Refer to the `README.md` in `plotting`

# Helper scripts

Auxiliary scripts to do things such as finding missing/error Compute Canada runs given a `.txt` file with all commands that were run and also generating scripts to be run on Compute Canada. Refer to the `README.md` in `helper_scripts` for more information.