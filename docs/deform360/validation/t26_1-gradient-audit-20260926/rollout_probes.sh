#!/bin/bash
set -eo pipefail
source /data1/userdata/tcweng/projects/tcgs/environment/soma-20260923/activate-soma.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK_AUDIT=/data1/userdata/tcweng/projects/tcgs/outputs/deform360/t26_1-audit-20260926
for N in 1 2 3; do
 python -u "$TASK_AUDIT/audit.py" --out "$TASK_AUDIT/r$N" --tag "r$N" --rollout "$N" > "$TASK_AUDIT/r$N.txt" 2>&1
done
