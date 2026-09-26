#!/bin/bash
set -eo pipefail
source /data1/userdata/tcweng/projects/tcgs/environment/soma-20260923/activate-soma.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK_AUDIT=/data1/userdata/tcweng/projects/tcgs/outputs/deform360/t26_1-audit-20260926
for C in l2 ssim; do
 # Anomaly exception is an expected diagnostic result; run independent second component.
 set +e
 python -u "$TASK_AUDIT/audit_anomaly.py" --out "$TASK_AUDIT/anomaly_$C" --tag "anomaly_$C" --rollout 1 --step 1 --component "$C" --anomaly > "$TASK_AUDIT/anomaly_$C.txt" 2>&1
 CODE=$?
 set -e
 if [ "$CODE" -ne 0 ] && [ "$CODE" -ne 2 ]; then exit "$CODE"; fi
done
