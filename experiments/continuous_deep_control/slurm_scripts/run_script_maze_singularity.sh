#!/bin/bash
ENV_NAME=$1
AGENT_NAME=$2
INDEX=$3

# pip3 uninstall 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mujoco/mujoco200/bin
python3 experiments/continuous_deep_control/nonlinear_run.py --env_json experiments/continuous_deep_control/jsonfiles/environment/$ENV_NAME.json --agent_json experiments/continuous_deep_control/jsonfiles/agent/$AGENT_NAME.json --index $3 --no_netsave
