#!/bin/bash
#SBATCH --job-name=Swimmer_rkl_ll_mul
#SBATCH --output=./logs/Swimmer/rkl/%A%a.out
#SBATCH --error=./logs/Swimmer/rkl/%A%a.err

#SBATCH --array=0-1599%200

#SBATCH --cpus-per-task=4
#SBATCH --time=8:30:00
#SBATCH --mem-per-cpu=6000M

#SBATCH --account=def-whitem

ENV_NAME=Swimmer-v2
AGENT_NAME=reverse_kl_ll_big

echo Running..$ENV_NAME $AGENT_NAME $SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

increment=1
let "end_idx=$SLURM_ARRAY_TASK_ID+3"

parallel --jobs 4 "source ~/sungsu_env/bin/activate; bash experiments/continuous_deep_control/slurm_scripts/run_script_template.sh $ENV_NAME $AGENT_NAME {}" ::: $(seq ${SLURM_ARRAY_TASK_ID} ${increment} ${end_idx})


