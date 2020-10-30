#!/bin/bash
#SBATCH --job-name=Swimmer_rkl_rpm_mul
#SBATCH --output=./logs/Swimmer/rkl/%A%a.out
#SBATCH --error=./logs/Swimmer/rkl/%A%a.err

#SBATCH --array=449,2879,5309,7739,10169,12599,15029,17459,19889,22319

#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=8000M

#SBATCH --account=rrg-whitem
#SBATCH --gres=gpu:v100l:1

ENV_NAME=Swimmer-v2
AGENT_NAME=swimmer_huge

echo Running..$ENV_NAME $AGENT_NAME $SLURM_ARRAY_TASK_ID

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hugoluis/.mujoco/mjpro150/bin
python3 experiments/continuous_deep_control/nonlinear_run.py --env_json experiments/continuous_deep_control/jsonfiles/environment/$ENV_NAME.json --agent_json experiments/continuous_deep_control/jsonfiles/agent/$AGENT_NAME.json --index $SLURM_ARRAY_TASK_ID



