#!/bin/bash

ENV_NAME=$1
AGENT_NAME=$2

conda activate back-it-up

# Inclusive
start_idx=$3
increment=$4
end_idx=$5

for i in $(seq ${start_idx} ${increment} ${end_idx})
do
  echo Running.. $i
  python nonlinear_run.py --env_json jsonfiles/environment/$ENV_NAME.json --agent_json jsonfiles/agent/$AGENT_NAME.json --index "$i" --write_plot
done
