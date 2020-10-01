#!/bin/bash
#SBATCH --job-name=Reacher_rkl_rpm_mul
#SBATCH --output=/home/hugoluis/scratch/back-it-up/logs/Reacher/rkl/%A%a.out
#SBATCH --error=/home/hugoluis/scratch/back-it-up/logs/Reacher/rkl/%A%a.err

#SBATCH --array=2165,2171,2177,2183,2189,2195,2201,2207,2213,2219,2225,2231,2237,2243,2249,2255,2261,2267,2273,2279,2285,2291,2297,2303,2309,2315,2321,2327,2333,2339,2345,2351,2357,2363,2369,2375,2381,2387,2393,2399,2405,2411,2417,2423,2429

#SBATCH --cpus-per-task=1
#SBATCH --time=16:30:00
#SBATCH --mem-per-cpu=6000M

#SBATCH --account=rrg-whitem

#SBATCH --gres=gpu:v100l:1

ENV_NAME=Reacher-v2
AGENT_NAME=reverse_kl_rpm_big

echo Running..$ENV_NAME $AGENT_NAME $SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

source ~/sungsu_env/bin/activate; bash experiments/continuous_deep_control/slurm_scripts/run_script_gpu_template_error.sh $ENV_NAME $AGENT_NAME ${SLURM_ARRAY_TASK_ID}
