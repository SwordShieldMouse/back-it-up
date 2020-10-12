#!/bin/bash
#SBATCH --job-name=HalfCheetah_rkl_rpm_mul
#SBATCH --output=./logs/HalfCheetah/rkl/%A%a.out
#SBATCH --error=./logs/HalfCheetah/rkl/%A%a.err

#SBATCH --array=1016,1017,1018,1019,1020,1021,1022,1023,880,881,882,883,1368,1369,1370,1371,1372,1373,1374,1375,1376,1377,1378,1379,1068,1069,1070,1071,1072,1073,1074,1075,1076,1077,1078,1079,924,925,926,927,928,929,930,931,1420,1421,1422,1423,1424,1425,1426,1427

#SBATCH --cpus-per-task=1
#SBATCH --time=12:30:00
#SBATCH --mem-per-cpu=6000M

#SBATCH --account=def-whitem

ENV_NAME=HalfCheetah-v2
AGENT_NAME=reverse_kl_rpm_big

echo Running..$ENV_NAME $AGENT_NAME $SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

source ~/sungsu_env/bin/activate; bash experiments/continuous_deep_control/slurm_scripts/run_script_template.sh $ENV_NAME $AGENT_NAME ${SLURM_ARRAY_TASK_ID}


