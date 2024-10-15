#!/bin/bash 

#SBATCH --job-name=g_vrh_stat_3_s621
#SBATCH --output=g_vrh_stat_3_s621_out.txt
#SBATCH --error=g_vrh_stat_3_s621_err.txt

#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --nodelist=gpu01
#SBATCH --mail-user=dario.massa@ideas-ncbr.pl
#SBATCH --mail-type=ALL

eval "$(conda shell.bash hook)"
conda activate ~/anaconda3/envs/tfgpu1/
cd /raid/NFS_SHARE/home/dario.massa/PROJECT_3_corrected/CLIP

python -u main.py --config ./g_vrh_confs/statistics/config_color_3_s621.yaml 