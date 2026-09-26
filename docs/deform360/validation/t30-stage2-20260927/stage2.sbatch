#!/bin/bash
#SBATCH --partition=5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --job-name=d360-stage2-seed5
#SBATCH --output=/data1/userdata/tcweng/projects/tcgs/outputs/deform360/t30-stage2-control-20260927/%x-%j.stdout.log
#SBATCH --error=/data1/userdata/tcweng/projects/tcgs/outputs/deform360/t30-stage2-control-20260927/%x-%j.stderr.log
set -eo pipefail
source /data1/userdata/tcweng/projects/tcgs/environment/soma-20260923/activate-soma.sh
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
TASK_ROOT=/data1/userdata/tcweng/projects/tcgs
CONTROL_DIR="$TASK_ROOT/outputs/deform360/t30-stage2-control-20260927"
cd "$TASK_ROOT/SoMA"
test "$(git rev-parse HEAD)" = 1d54006ed2a7a8d3f1c247232b484186f9a16363
test -z "$(git status --porcelain)"
test ! -e "$TASK_ROOT/outputs/deform360/stage2/config_a_seed5"
date --iso-8601=seconds
hostname
nvidia-smi
set +e
python -u "$CONTROL_DIR/run_stage2_observed.py" configs/SoMA/deform360_v0_stage2.py --seed 5 --gpus 1 --launcher none --work_dir "$TASK_ROOT/outputs/deform360/stage2/config_a_seed5/"
TRAIN_EXIT=$?
set -e
printf '%s\n' "$TRAIN_EXIT" > "$CONTROL_DIR/exit_code.txt"
exit "$TRAIN_EXIT"
