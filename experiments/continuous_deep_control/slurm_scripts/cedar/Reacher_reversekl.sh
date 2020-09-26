#!/bin/bash
#SBATCH --job-name=Reacher_rkl_rpm_mul
#SBATCH --output=/home/hugoluis/scratch/back-it-up/logs/Reacher/rkl/%A%a.out
#SBATCH --error=/home/hugoluis/scratch/back-it-up/logs/Reacher/rkl/%A%a.err

#SBATCH --array=0-15:4

#SBATCH --cpus-per-task=4
#SBATCH --time=6:30:00
#SBATCH --mem-per-cpu=6000M

#SBATCH --account=rrg-whitem

ENV_NAME=Reacher-v2
AGENT_NAME=reverse_kl_reduced

module load singularity/3.5
echo Running..$ENV_NAME $AGENT_NAME $SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

increment=1
let "end_idx=$SLURM_ARRAY_TASK_ID+3"

parallel --jobs 4 "singularity exec -B /scratch /home/hugoluis/singularity_environment/torch_rlcontrol.simg bash experiments/continuous_deep_control/slurm_scripts/run_script_template.sh $ENV_NAME $AGENT_NAME {}" ::: $(seq ${SLURM_ARRAY_TASK_ID} ${increment} ${end_idx})


