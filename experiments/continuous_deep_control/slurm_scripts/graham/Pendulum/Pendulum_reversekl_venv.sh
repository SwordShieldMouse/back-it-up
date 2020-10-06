#!/bin/bash
#SBATCH --job-name=Pendulum_rkl_rpm_mul
#SBATCH --output=/home/hugoluis/scratch/back-it-up/logs/Pendulum/rkl/%A%a.out
#SBATCH --error=/home/hugoluis/scratch/back-it-up/logs/Pendulum/rkl/%A%a.err

#SBATCH --array=0-2429%200

#SBATCH --cpus-per-task=1
#SBATCH --time=8:30:00
#SBATCH --mem-per-cpu=6000M

#SBATCH --account=def-whitem
#SBATCH --gres=gpu:1

ENV_NAME=Pendulum-v0
AGENT_NAME=reverse_kl_rpm_big

echo Running..$ENV_NAME $AGENT_NAME $SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

source ~/sungsu_env/bin/activate; bash experiments/continuous_deep_control/slurm_scripts/run_script_gpu_template_error.sh $ENV_NAME $AGENT_NAME ${SLURM_ARRAY_TASK_ID}


