#!/bin/bash
#SBATCH --job-name=Reacher_rpm_mul
#SBATCH --output=./logs/Reacher/rkl/%A%a.out
#SBATCH --error=./logs/Reacher/rkl/%A%a.err

#SBATCH --array=0-3

#SBATCH --cpus-per-task=1
#SBATCH --time=4:30:00
#SBATCH --mem-per-cpu=8000M

#SBATCH --account=def-whitem

ENV_NAME=Reacher-v2
AGENT_NAME=reverse_kl_rpm_big

ERROR_RUNS=(3125 2174 2976)

echo Running..$ENV_NAME $AGENT_NAME $SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

increment=1

source ~/sungsu_env/bin/activate
bash experiments/continuous_deep_control/slurm_scripts/run_script_template.sh $ENV_NAME $AGENT_NAME ${ERROR_RUNS[$SLURM_ARRAY_TASK_ID]}


