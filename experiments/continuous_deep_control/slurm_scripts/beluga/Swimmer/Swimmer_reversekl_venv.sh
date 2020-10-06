#!/bin/bash
#SBATCH --job-name=Swimmer_rkl_ll_mul
#SBATCH --output=/home/hugoluis/scratch/back-it-up/logs/Swimmer/rkl/%A%a.out
#SBATCH --error=/home/hugoluis/scratch/back-it-up/logs/Swimmer/rkl/%A%a.err

#SBATCH --array=0-2429:3%200

#SBATCH --cpus-per-task=3
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=6000M

#SBATCH --account=def-whitem

#SBATCH --gres=gpu:3

ENV_NAME=Swimmer-v2
AGENT_NAME=reverse_kl_ll_big

echo Running..$ENV_NAME $AGENT_NAME $SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

n_gpus=3
increment=1
let "end_idx=$SLURM_ARRAY_TASK_ID+2"

parallel --jobs 3 "source ~/sungsu_env/bin/activate; bash experiments/continuous_deep_control/slurm_scripts/run_script_gpu_template.sh $ENV_NAME $AGENT_NAME {}" ::: $(seq ${SLURM_ARRAY_TASK_ID} ${increment} ${end_idx} ${n_gpus})


