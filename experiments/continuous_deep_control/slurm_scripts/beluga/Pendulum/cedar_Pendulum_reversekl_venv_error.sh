#!/bin/bash
#SBATCH --job-name=Pendulum_rkl_rpm_mul
#SBATCH --output=./logs/Pendulum/rkl/%A%a.out
#SBATCH --error=./logs/Pendulum/rkl/%A%a.err

#SBATCH --array=484,485,486,487,488,489,490,491,492,493,494,495,496,497,498,499,500,501,502,503,564,565,566,567,568,569,570,571,572,573,574,575

#SBATCH --cpus-per-task=1
#SBATCH --time=12:30:00
#SBATCH --mem-per-cpu=6000M

#SBATCH --account=def-whitem

ENV_NAME=Pendulum-v0
AGENT_NAME=reverse_kl_rpm_big

echo Running..$ENV_NAME $AGENT_NAME $SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

source ~/sungsu_env/bin/activate; bash experiments/continuous_deep_control/slurm_scripts/run_script_template.sh $ENV_NAME $AGENT_NAME ${SLURM_ARRAY_TASK_ID}


