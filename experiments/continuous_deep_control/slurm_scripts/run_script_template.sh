#!/bin/bash
ENV_NAME=$1
AGENT_NAME=$2
INDEX=$3

# pip3 uninstall 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sungsu/.mujoco/mjpro150/bin
python3 ../nonlinear_run.py --env_json ../jsonfiles/environment/$ENV_NAME.json --agent_json ../jsonfiles/agent/$AGENT_NAME.json --index $3
