# T26 — frozen Stage-1 run, 2026-09-26

**FAIL/nonfinite; stopped at the first backward, before clipping or optimizer.step.**

## Preserved infrastructure history

The original [preflight.json](preflight.json) remains unchanged: SSH banner timeout before job submission. This was an infrastructure transient block, not training FAIL, OOM, or model/config failure. This run was **RESUMED after SSH recovery**. Server clean checkout fast-forwarded from 92107be to `ff02eaf9fceaf6f9bf1f53ebd615eef545ddd922`, matching Mac/origin (0/0). [Resumed preflight](resumed_preflight.json) checked the real MMCV-expanded configuration and 794 existing asset hashes; no assets regenerated.

## Actual execution

| Item | Result |
|---|---|
| Slurm job / node | 25850 / amax |
| GPU | NVIDIA GeForce RTX 5090, GPU-be898c18-3979-84e7-e75a-ac09ad91a7d7, 32607 MiB, 580.126.18 |
| Git checkpoint | `ff02eaf9fceaf6f9bf1f53ebd615eef545ddd922` |
| Config SHA256 | `e65d1c7848bd222f55c0e18050c464eec5922aef76c1bc62a06813359cca4930` |
| Seed / config | 5 / Config A, official inherited frozen protocol |
| Attempted epoch / iteration | 1 / 1 |
| Completed epochs / optimizer steps | 0 / 0 |
| Rollout | 3; initial source113, targets123/133/143 |
| LR | 0.000404 (Hood first epoch; base LR0.0004) |
| Forward total loss | 40912.95703125 (finite) |
| Momentum / L2 render / SSIM | 7150.60302734375 / 28851.828125 / 4910.5263671875 |
| First nonfinite | gradients of 275 parameter tensors after combined backward |
| Stop boundary | original OptimizerHook → observed clip_grads precheck; before clip and optimizer.step |
| Model / normalizer at stop | all registered tensors finite |
| Optimizer | existing state finite; optimizer.step never reached, no successful Adam update |
| Peak allocated / reserved | 3832729088 / 4066377728 bytes (3.570 / 3.787 GiB) |
| Process GPU memory at stop | 23184, python, 4578 MiB |
| Memory at stop | allocated 57697792 B; reserved 4066377728 B; free 28859891712 B |
| Wall time | Slurm25s; observed invocation 11.675s (excludes prior interpreter imports/activation) |
| OOM | no |
| Checkpoints / last-good / final | none; no completed epoch, no checkpoint path/hash available |
| Exit | 1:0 |

`failure_context.step=3` means the last completed forward step. All three step outputs/losses were finite. Backward operated on the aggregate loss; this experiment does **not** identify which step or operation caused nonfinite gradients. No stability investigation or repair was performed.

Both camera readers loaded only local0,10,…,150 / source113,123,…,263; actual loss targets were123,133,143. No source≥268 supervision. Config/source assets and in-memory Gaussian fields remained unchanged. Frozen rollout/LR/loss/detach/batch/resolution/graph/allocator settings were not changed. No warmup, retry, resume, or T27.

## Command and paths

Submission: `sbatch --parsable /data1/userdata/tcweng/projects/tcgs/outputs/deform360/t26-control-20260926/stage1.sbatch`

```bash
/data1/userdata/tcweng/miniconda3/envs/soma/bin/python /data1/userdata/tcweng/projects/tcgs/outputs/deform360/t26-control-20260926/run_stage1_observed.py configs/SoMA/deform360_v0_stage1.py --seed 5 --gpus 1 --launcher none --work_dir /data1/userdata/tcweng/projects/tcgs/outputs/deform360/stage1/config_a_seed5/
```

The observer delegates to the unmodified `tools/train.py` and original runner/optimizer hook; observational wrappers record frame ranges, state/memory/loss, and fail fast on NaN/Inf before an update. It does not insert extra forward passes or change values/gradients/config. The exact observer and sbatch are included here; their executed hashes are in [training_report.json](training_report.json).

- Training directory (server only): `/data1/userdata/tcweng/projects/tcgs/outputs/deform360/stage1/config_a_seed5/`
- Control/log directory: `/data1/userdata/tcweng/projects/tcgs/outputs/deform360/t26-control-20260926/`
- [stdout](d360-stage1-seed5-25850.stdout.txt), [stderr / full traceback](d360-stage1-seed5-25850.stderr.txt), [events](events.jsonl), [Slurm accounting](slurm_accounting.txt), [summary](summary.json).

Small evidence files are uncommitted in SoMA/deform360-adaptation on Mac. Server checkout remains clean at the T25 commit; the executed observer lives outside the server repository. No T26 commit/push. No T27.
