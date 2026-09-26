#!/usr/bin/env bash
set -eu
source /data1/userdata/tcweng/projects/tcgs/environment/soma-20260923/activate-soma.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
out=/data1/userdata/tcweng/projects/tcgs/outputs/deform360/t26_2-audit-20260926
for probe in original exclude mean cov both; do
 /data1/userdata/tcweng/miniconda3/envs/soma/bin/python "$out/audit_renderer_degeneracy.py" --out "$out/verified_$probe" --probe "$probe" > "$out/verified_$probe.log" 2>&1
done
/data1/userdata/tcweng/miniconda3/envs/soma/bin/python "$out/audit_renderer_degeneracy.py" --out "$out/verified_control009" --probe original --camera 1 > "$out/verified_control009.log" 2>&1
