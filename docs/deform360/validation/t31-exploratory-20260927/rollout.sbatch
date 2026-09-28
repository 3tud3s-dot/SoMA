#!/bin/bash
#SBATCH -p 5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=d360-t31-explore
#SBATCH --output=/data1/userdata/tcweng/projects/tcgs/outputs/deform360/t31-exploratory-control-20260927/%j.log
set -eo pipefail
source /data1/userdata/tcweng/projects/tcgs/environment/soma-20260923/activate-soma.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python -u /data1/userdata/tcweng/projects/tcgs/outputs/deform360/t31-exploratory-control-20260927/check_continuous_exploratory_v2.py
