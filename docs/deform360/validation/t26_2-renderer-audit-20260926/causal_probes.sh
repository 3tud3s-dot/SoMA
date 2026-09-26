#!/usr/bin/env bash
set -eu
source /data1/userdata/tcweng/projects/tcgs/environment/soma-20260923/activate-soma.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
out=/data1/userdata/tcweng/projects/tcgs/outputs/deform360/t26_2-audit-20260926
for probe in exclude mean cov both; do
 /data1/userdata/tcweng/miniconda3/envs/soma/bin/python "$out/audit_renderer_degeneracy.py" --out "$out/$probe" --probe "$probe" > "$out/$probe.log" 2>&1
done
/data1/userdata/tcweng/miniconda3/envs/soma/bin/python "$out/audit_renderer_degeneracy.py" --out "$out/control009" --probe original --camera 1 > "$out/control009.log" 2>&1
