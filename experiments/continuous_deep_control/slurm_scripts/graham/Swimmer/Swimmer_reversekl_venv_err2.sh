#!/bin/bash
#SBATCH --job-name=Swimmer_rkl_rpm_mul
#SBATCH --output=./logs/Swimmer/rkl/%A%a.out
#SBATCH --error=./logs/Swimmer/rkl/%A%a.err

#SBATCH --array=0-6:4

#SBATCH --cpus-per-task=4
#SBATCH --time=8:30:00
#SBATCH --mem-per-cpu=12G

#SBATCH --account=def-whitem

ENV_NAME=Swimmer-v2
AGENT_NAME=reverse_kl_rpm_big

ERROR_RUNS=(4405,4565,4725,1686,1884,2204)

echo Running..$ENV_NAME $AGENT_NAME $SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

THESE_ERROR_RUNS=()

let "end_idx=$SLURM_ARRAY_TASK_ID+3"
for ((i=$SLURM_ARRAY_TASK_ID; i<= $end_idx; i++)); do
    THESE_ERROR_RUNS+=( ${ERROR_RUNS[$i]} )
done

increment=1

parallel --jobs 4 "source ~/sungsu_env/bin/activate; bash experiments/continuous_deep_control/slurm_scripts/run_script_template.sh $ENV_NAME $AGENT_NAME {}" ::: ${THESE_ERROR_RUNS[@]}
