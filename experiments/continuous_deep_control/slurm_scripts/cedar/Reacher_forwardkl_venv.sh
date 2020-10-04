#!/bin/bash
#SBATCH --job-name=Reacher_fkl
#SBATCH --output=/home/hugoluis/scratch/back-it-up/logs/Reacher/fkl/%A%a.out
#SBATCH --error=/home/hugoluis/scratch/back-it-up/logs/Reacher/fkl/%A%a.err

#SBATCH --array=0-1620:4

#SBATCH --cpus-per-task=4
#SBATCH --time=16:30:00
#SBATCH --mem-per-cpu=6000M

#SBATCH --account=rrg-whitem
#SBATCH --gres=gpu:v100l:2

ENV_NAME=Reacher-v2
AGENT_NAME=forward_kl_big

echo Running..$ENV_NAME $AGENT_NAME $SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

increment=1
let "end_idx=$SLURM_ARRAY_TASK_ID+3"

parallel --jobs 4 "source ~/sungsu_env/bin/activate; bash experiments/continuous_deep_control/slurm_scripts/run_script_gpu_template.sh $ENV_NAME $AGENT_NAME {}" ::: $(seq ${SLURM_ARRAY_TASK_ID} ${increment} ${end_idx})


