#!/bin/bash
#SBATCH --job-name=ContinuousMaze_rkl
#SBATCH --output=./logs/ContinuousMaze/rkl/%A%a.out
#SBATCH --error=./logs/ContinuousMaze/rkl/%A%a.err

#SBATCH --array=0-173:4

#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=8000M

#SBATCH --account=def-whitem

ENV_NAME=ContinuousMaze
AGENT_NAME=reverse_kl_maze

ERROR_RUNS=(0 1 145 3 4 5 149 7 151 152 9 153 11 156 13 157 15 159 17 161 19 163 21 165 23 167 25 169 170 27 171 28 29 173 31 175 32 176 33 177 35 179 37 181 39 183 40 184 41 185 43 187 188 45 189 190 47 191 49 193 51 195 52 53 197 55 199 200 57 201 59 203 60 204 61 205 206 63 207 208 65 209 67 211 212 69 213 214 71 215 72 73 217 75 219 77 221 79 223 80 81 225 83 227 84 228 85 229 230 87 231 88 89 233 91 235 236 93 237 238 95 239 96 240 97 241 99 243 101 245 103 247 105 249 107 251 109 253 111 255 112 256 113 257 258 115 259 117 261 119 263 121 265 123 125 269 127 272 129 273 274 131 275 276 133 277 135 279 137 281 283 141 285 287)

echo Running..$ENV_NAME $AGENT_NAME $SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

THESE_ERROR_RUNS=()

let "end_idx=$SLURM_ARRAY_TASK_ID+3"
for ((i=$SLURM_ARRAY_TASK_ID; i<= $end_idx; i++)); do
    THESE_ERROR_RUNS+=( ${ERROR_RUNS[$i]} )
done

increment=1

parallel --jobs 4 "source ~/sungsu_env/bin/activate; bash experiments/continuous_deep_control/slurm_scripts/run_script_maze.sh $ENV_NAME $AGENT_NAME {}" ::: ${THESE_ERROR_RUNS[@]}


