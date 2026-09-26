#!/bin/bash
#SBATCH --partition=5090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=d360-stage1-normfix
#SBATCH --output=/data1/userdata/tcweng/projects/tcgs/outputs/deform360/t26-normalizer-fix-control-20260926/%x-%j.stdout.log
#SBATCH --error=/data1/userdata/tcweng/projects/tcgs/outputs/deform360/t26-normalizer-fix-control-20260926/%x-%j.stderr.log
set -eo pipefail
source /data1/userdata/tcweng/projects/tcgs/environment/soma-20260923/activate-soma.sh
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
TASK_ROOT=/data1/userdata/tcweng/projects/tcgs
CONTROL_DIR="$TASK_ROOT/outputs/deform360/t26-normalizer-fix-control-20260926"
cd "$TASK_ROOT/SoMA"
test "$(git rev-parse HEAD)" = 0a628fa4dec9d8d267b131187f06b9582bc18c7b
test -z "$(git status --porcelain)"
test ! -e "$TASK_ROOT/outputs/deform360/stage1/config_a_seed5_normalizer_fix"
date --iso-8601=seconds
hostname
nvidia-smi
set +e
python -u "$CONTROL_DIR/run_stage1_observed.py" configs/SoMA/deform360_v0_stage1.py --seed 5 --gpus 1 --launcher none --work_dir "$TASK_ROOT/outputs/deform360/stage1/config_a_seed5_normalizer_fix/"
TRAIN_EXIT=$?
set -e
date --iso-8601=seconds
printf 'TRAIN_EXIT=%s\n' "$TRAIN_EXIT"
printf '%s\n' "$TRAIN_EXIT" > "$CONTROL_DIR/exit_code.txt"
exit "$TRAIN_EXIT"
