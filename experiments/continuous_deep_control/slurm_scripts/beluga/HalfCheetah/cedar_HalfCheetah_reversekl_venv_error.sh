#!/bin/bash
#SBATCH --job-name=HalfCheetah_rkl_rpm_mul
#SBATCH --output=./logs/HalfCheetah/rkl/%A%a.out
#SBATCH --error=./logs/HalfCheetah/rkl/%A%a.err

#SBATCH --array=1022,1023,925

#SBATCH --cpus-per-task=1
#SBATCH --time=15:30:00
#SBATCH --mem-per-cpu=6000M

#SBATCH --account=def-whitem

ENV_NAME=HalfCheetah-v2
AGENT_NAME=reverse_kl_rpm_big

echo Running..$ENV_NAME $AGENT_NAME $SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

source ~/sungsu_env/bin/activate; bash experiments/continuous_deep_control/slurm_scripts/run_script_template.sh $ENV_NAME $AGENT_NAME ${SLURM_ARRAY_TASK_ID}


