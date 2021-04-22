#!/bin/bash
#SBATCH --job-name=MediumContinuousMaze_fkl
#SBATCH --output=./logs/MediumContinuousMaze_fkl_%A%a.out
#SBATCH --error=./logs/MediumContinuousMaze_fkl_%A%a.err

#SBATCH --array=0-59:4
#SBATCH --dependency=singleton
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=5G

#SBATCH --account=rrg-whitem

ENV_NAME=MediumContinuousMaze
AGENT_NAME=forward_kl_medium_maze

echo Running..$ENV_NAME $AGENT_NAME $SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

module load singularity

increment=1
let "end_idx=$SLURM_ARRAY_TASK_ID+3"

parallel --jobs 4 "singularity exec --cleanenv --no-home --writable --bind .:/scratch --pwd /scratch ~/singularity_environment/sungsu_test_sandbox bash experiments/continuous_deep_control/slurm_scripts/run_script_maze_singularity.sh $ENV_NAME $AGENT_NAME {}"  ::: $(seq ${SLURM_ARRAY_TASK_ID} ${increment} ${end_idx})


