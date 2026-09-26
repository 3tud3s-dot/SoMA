#!/bin/bash
set -eo pipefail
source /data1/userdata/tcweng/projects/tcgs/environment/soma-20260923/activate-soma.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK_AUDIT=/data1/userdata/tcweng/projects/tcgs/outputs/deform360/t26_1-audit-20260926
for C in l2 ssim; do
 python -u "$TASK_AUDIT/audit_native.py" --out "$TASK_AUDIT/native_$C" --tag "native_$C" --rollout 1 --step 1 --component "$C" --native-trace > "$TASK_AUDIT/native_$C.txt" 2>&1
done
