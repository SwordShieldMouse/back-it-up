#!/bin/bash
#SBATCH --job-name=EasyContinuousMaze_fkl
#SBATCH --output=./logs/EasyContinuousMaze_fkl_%A%a.out
#SBATCH --error=./logs/EasyContinuousMaze_fkl_%A%a.err

#SBATCH --array=0-149

#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=1G
#SBATCH --dependency=singleton
#SBATCH --account=def-whitem

ENV_NAME=EasyContinuousMaze
AGENT_NAME=forward_kl_maze

module load singularity

echo Running..$ENV_NAME $AGENT_NAME $SLURM_ARRAY_TASK_ID

singularity exec --cleanenv --no-home --writable --bind .:/scratch --pwd /scratch ~/singularity_environment/sungsu_test_sandbox bash experiments/continuous_deep_control/slurm_scripts/run_script_maze_singularity.sh $ENV_NAME $AGENT_NAME ${SLURM_ARRAY_TASK_ID}

