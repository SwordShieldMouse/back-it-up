#!/bin/bash
#SBATCH --job-name=Reacher_rkl_rpm_mul
#SBATCH --output=./logs/Reacher/rkl/%A%a.out
#SBATCH --error=./logs/Reacher/rkl/%A%a.err

#SBATCH --array=322,323,325,326,486,327,487,1451,333,334,335,23,504,27,191,1311,194,1155,356,197,198,1159,527,1490,371,1177,699,540,543,1183,866,1346,867,710,1191,1036,1039,1360,401,1362,1526,568,729,1529,570,251,1531,572,899,1062,103,1063,584,904,1384,425,907,748,749,750,751,914,915,1557,438,1078,123,1244,287,288,290,291,931,932,773,138,618,619,1579,620,940,461,942,623,1423,305,948,310,790,797,478,1119

#SBATCH --cpus-per-task=4
#SBATCH --time=10:30:00
#SBATCH --mem-per-cpu=12000M

#SBATCH --account=def-whitem
#SBATCH --gres=gpu:1

ENV_NAME=Reacher-v2
AGENT_NAME=reverse_kl_rpm_big

echo Running..$ENV_NAME $AGENT_NAME $SLURM_ARRAY_TASK_ID

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

increment=1
let "end_idx=$SLURM_ARRAY_TASK_ID+3"

parallel --jobs 4 "source ~/sungsu_env/bin/activate; bash experiments/continuous_deep_control/slurm_scripts/run_script_template.sh $ENV_NAME $AGENT_NAME {}" ::: $(seq ${SLURM_ARRAY_TASK_ID} ${increment} ${end_idx})


