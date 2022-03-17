# Contents

This directory contains scripts that help find runs that failed on Compute Canada and that generate the commands to be run. Scripts should be run from the root directory with the command:

```
pipenv run python3 -m helper_scripts.<name>
```

The scripts are:

- `debug_find_error_runs.py`: reads a command file, given by the optional command argument `--path_to_command_file`, and a directory, given by the optional command argument `--results_dir`. Based on the contents of the result directory, it prints out the runs from the input commands that failed to complete
- `debug_translate_setn_to_params.py`: receives as inputs the arguments `agent_json` and `index`, this last one corresponding to the input number passed as a command argument to `nonlinear_run.py` and equal to `SETTING + NUM_SETTINGS * RUN_NUMBER`. Prints the agent parameters, the setting and the run number.
- `generate_slurm_script.py`: code used to generate the commands that will be passed to slurm. Each command is a line in the output file `helper_scripts/commands.txt` and lines are commented and uncommented depending on what the user wants to run.