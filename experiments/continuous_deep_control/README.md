# Contents
This experiment contains the KL experiments for continuous action space environments and non-linear function approximation. 

# To run experiments
Modify agent sweep settings in `jsonfiles/agent/$AGENT_NAME.json`.
Modify environment settings in `jsonfiles/environment/$ENV_NAME.json`.
Run `python3 nonlinear_run.py --env_json jsonfiles/environment/$ENV_NAME.json --agent_json jsonfiles/agent/$AGENT_NAME.json --index $IDX`

IDX is unique identification of the run number and setting number, depending on the sweep settings in `$AGENT_NAME.json`.
Running the command will generate a directory `results/$ENV_NAMEresults/` saving 6 files per run, of which the current setting file (`*_agent_Params.json`) would be identical for different runs and settings.

Other five files are:
`*_setting*_run_*_agent_Params.txt`,
`*_setting*_run_*_agent_EpisodeRewardsLC.txt`,
`*_setting*_run_*_agent_EpisodeStepsLC.txt`,
`*_setting*_run_*_agent_EvalEpisodeMeanRewardsLC.txt`,
`*_setting*_run_*_agent_EvalEpisodeStdRewardsLC.txt`

(Currently evaluation is not used; only training data is used)
Mujoco must be installed in order to run Reacher-v2 and Swimmer-v2 environments.

# To plot the data
After all runs are complete, run the following:

1. Merge generated results: `python3 merge_results.py $RESULT_DIR $ROOT_LOC $ENV_NAME $AGENT_NAME $NUM_RUNS $USE_MOVING_AVG`. This command will create `merged$ENV_NAMEresults/` directory and the combined results will be stored there.

2. Find best settings: `python3 find_agent_best_setting.py $RESULT_DIR $ROOT_LOC $ENV_NAME $AGENT_NAME $NUM_RUNS $CUSTOM_NAME $PARSE_TYPE`. Here the best settings will be found along `$PARSE_TYPE` params (in our case entropy_scale). Along with generating plots of the best settings, it will also save in `merged$ENV_NAMEresults/npy` the npy data of the best settings found for each $PARSE_TYPE.

3. Generate entropy scale comparison plot: `python3 plot_entropy_comparison.py $ROOT_LOC $ENV_NAME $STORE_DIR $PARSE_TYPE`

4. Generate sensitivity curves: `python3 plot_sensitivity.py $STORE_DIR $ENV_NAME`
