#!/bin/bash
#SBATCH --job-name=Reacher_fkl
#SBATCH --output=./logs/Reacher/fkl/%A%a.out
#SBATCH --error=./logs/Reacher/fkl/%A%a.err

#SBATCH --array=0-43:4

#SBATCH --cpus-per-task=4
#SBATCH --time=9:30:00
#SBATCH --mem-per-cpu=8000M

#SBATCH --account=def-whitem

ENV_NAME=Reacher-v2
AGENT_NAME=forward_kl_big

ERROR_RUNS=(4156 4157 4158 4159 4160 4161 4162 4163 4164 4165 4166 4167 4168 4169 4170 4171 4172 4173 4174 4175 4176 4177 4178 4179 4180 4181 4182 4183 4184 4185 4186 4187 4188 4189 4190 4191 4192 4193 4194 4195 4196 4197 4198 4199)

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


