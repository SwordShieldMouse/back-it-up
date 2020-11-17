#!/bin/bash
#SBATCH --job-name=HalfCheetah_fkl_mul
#SBATCH --output=./logs/HalfCheetah/fkl/%A%a.out
#SBATCH --error=./logs/HalfCheetah/fkl/%A%a.err

#SBATCH --array=1400-4199:4

#SBATCH --cpus-per-task=4
#SBATCH --time=10:30:00
#SBATCH --mem-per-cpu=8000M

#SBATCH --account=def-whitem

ENV_NAME=HalfCheetah-v2
AGENT_NAME=forward_kl_big

echo Running..$ENV_NAME $AGENT_NAME $SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

increment=1
let "end_idx=$SLURM_ARRAY_TASK_ID+3"

parallel --jobs 4 "source ~/sungsu_env/bin/activate; bash experiments/continuous_deep_control/slurm_scripts/run_script_template.sh $ENV_NAME $AGENT_NAME {}" ::: $(seq ${SLURM_ARRAY_TASK_ID} ${increment} ${end_idx})


