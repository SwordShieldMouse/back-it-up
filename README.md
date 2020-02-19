# Tabular experiments
run with `python3 tabular_run.py seed max_frames max_frames_per_ep`

# Linear experiments:
run with `python3 linear_run.py experiment_index max_frames max_frames_per_ep`

# Non-linear experiments (Pendulum):
`cd deep_experiments`

run with `python3 nonlinear_run.py --env_json jsonfiles/environment/Pendulum-v0.json --agent_json jsonfiles/agent/agent_name.json --index index_num` 

(environment and agent sweep settings are saved in respective jsons)

# Continuous Bandits experiments:
`cd deep_experiments`

run with `python3 nonlinear_run.py --env_json jsonfiles/environment/ContinuousBandits.json --agent_json jsonfiles/agent/agent_name_bandits.json --index index_num --write_plot`

(With the `--write_plot` flag, plots at each step are saved and later compiled into a video, as shown in the Supplementary webpage.)
