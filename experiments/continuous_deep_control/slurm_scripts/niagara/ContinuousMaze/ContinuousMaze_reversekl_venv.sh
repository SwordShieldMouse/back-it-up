#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --job-name=cm_rkl
#SBATCH --output=./logs/MediumContinuousMaze_rkl_%A%a.out
#SBATCH --error=./logs/MediumContinuousMaze_rkl_%A%a.err
#SBATCH --array=0-59:10
#SBATCH --time=6:00:00
#SBATCH --account=def-whitem
#SBATCH --dependency=singleton

#Each takes +- 400MB of RAM if run only on CPU

ENV_NAME=MediumContinuousMaze
AGENT_NAME=reverse_kl_medium_maze

export OMP_NUM_THREADS=4

increment=1
start_idx=$SLURM_ARRAY_TASK_ID
let "end_idx=$start_idx+9"

module load NiaEnv/2019b gnu-parallel 

parallel "singularity exec --cleanenv --no-home --writable --bind .:/scratch --pwd /scratch $HOME/sungsu_test_sandbox bash experiments/continuous_deep_control/slurm_scripts/run_script_maze_singularity.sh $ENV_NAME $AGENT_NAME {}"  ::: $(seq ${SLURM_ARRAY_TASK_ID} ${increment} ${end_idx})