#!/bin/bash
#SBATCH --job-name=ContinuousMaze_fkl
#SBATCH --output=./logs/ContinuousMaze/fkl/%A%a.out
#SBATCH --error=./logs/ContinuousMaze/fkl/%A%a.err

#SBATCH --array=0-287:4

#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=3000M

#SBATCH --account=def-whitem

ENV_NAME=ContinuousMaze
AGENT_NAME=forward_kl_maze

echo Running..$ENV_NAME $AGENT_NAME $SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1


increment=1
let "end_idx=$SLURM_ARRAY_TASK_ID+3"

parallel --jobs 4 "source ~/sungsu_env/bin/activate; bash experiments/continuous_deep_control/slurm_scripts/run_script_maze.sh $ENV_NAME $AGENT_NAME {}" ::: $(seq ${SLURM_ARRAY_TASK_ID} ${increment} ${end_idx})


