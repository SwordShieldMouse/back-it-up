#!/bin/bash
#SBATCH --job-name=video_runs
#SBATCH --output=./logs/%A%a.out
#SBATCH --error=./logs/%A%a.err

#SBATCH --array=1-8

#SBATCH --cpus-per-task=1
#SBATCH --time=8:30:00
#SBATCH --mem-per-cpu=6000M

#SBATCH --account=rrg-whitem

bash -c "source ~/sungsu_env/bin/activate; export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hugoluis/.mujoco/mjpro150/bin; `sed "${SLURM_ARRAY_TASK_ID}q;d" experiments/continuous_deep_control/slurm_scripts/video_runs/video_runs.txt`"