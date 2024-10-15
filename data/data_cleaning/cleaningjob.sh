#!/bin/bash
#SBATCH --job-name=log_15run
#SBATCH --nodelist=login01
#SBATCH --mail-user=dario.massa@ideas-ncbr.pl
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --output=log_15run_%j.out
#SBATCH --error=log_15run_%j.err

start_time=$(date +%s)

echo "Running on $(nproc) CPUs"
eval "$(conda shell.bash hook)"
conda activate ~/anaconda3/envs/tfgpu1/

python -u cleaning.py

end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Execution time: ${execution_time} seconds"

