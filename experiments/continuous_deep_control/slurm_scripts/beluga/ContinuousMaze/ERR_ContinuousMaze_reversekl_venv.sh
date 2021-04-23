#!/bin/bash
#SBATCH --job-name=EasyContinuousMaze
#SBATCH --output=./logs/EasyContinuousMaze_%A%a.out
#SBATCH --error=./logs/EasyContinuousMaze_%A%a.err

#SBATCH --array=0-104:4
#SBATCH --dependency=singleton
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=3000M

#SBATCH --account=def-whitem

ENV_NAME=EasyContinuousMaze
AGENT_NAME=reverse_kl_maze

ERROR_RUNS=(1 145 4 5 149 7 9 153 13 157 159 161 163 21 165 167 25 169 170 27 171 28 29 173 31 175 32 176 33 177 35 37 39 183 41 45 189 49 193 51 195 57 201 61 205 63 207 65 209 67 211 69 213 73 217 75 219 77 79 223 81 225 83 85 229 87 231 89 237 97 241 99 243 101 245 103 247 105 249 109 253 111 255 113 115 259 117 119 263 121 265 123 125 269 127 129 275 133 135 279 137 281 283 141 285)

echo Running..$ENV_NAME $AGENT_NAME $SLURM_ARRAY_TASK_ID

module load singularity

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

THESE_ERROR_RUNS=()

let "end_idx=$SLURM_ARRAY_TASK_ID+3"
for ((i=$SLURM_ARRAY_TASK_ID; i<= $end_idx; i++)); do
    THESE_ERROR_RUNS+=( ${ERROR_RUNS[$i]} )
done

increment=1

parallel --jobs 4 "singularity exec --cleanenv --no-home --writable --bind .:/scratch --pwd /scratch ~/singularity_environment/sungsu_test_sandbox bash experiments/continuous_deep_control/slurm_scripts/run_script_maze_singularity.sh $ENV_NAME $AGENT_NAME {}" ::: ${THESE_ERROR_RUNS[@]}


