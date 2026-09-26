# T30 formal launch — BLOCKED / entry-config

2026-09-27. T30.1 checkpoint: `1d54006ed2a7a8d3f1c247232b484186f9a16363`; Mac/origin/server synchronized and clean before observer/evidence creation. No unrelated files committed.

User limited this allocation to12h. Frozen budget61epochs/36600steps unchanged. No automatic resubmission at12h without further user confirmation.

Slurm25874, amax, RTX5090, FAILED exit1:0, elapsed7s, timelimit12h. **0 epochs / 0 optimizer steps**, no checkpoint. Actual tools/train.py failed at line71 before dataset/model construction, because formal config has no top-level `max_seq`. Error: `AttributeError: 'ConfigDict' object has no attribute 'max_seq'`. This is a missed entrypoint requirement in T30.1's generated config/static validation, not a training numerical failure, OOM, or initialization-resume failure.

No training loss/gradient/LR trajectory exists; no cache was consumed by training. CPU preflight verified16cache hashes/schema/finite, T26checkpoint hash,12FP64statistics and source hashes. Frozen config unchanged, source assets unchanged. Server clean. Original T30 BLOCKED and T30.1 PASS historical records retained; static PASS did not prove entrypoint completeness.

## Artifacts

- `run_stage2_observed.py`: observer based on tested Stage1 serializer/guards; no scheduler/optimizer algorithm replacement; original entrypoint/OptimizerHook calls. Variable-window metric-key aggregation handles9-vs12-step windows. Syntax, NumPy sanitizer and mixed-key mean tests ran CPU-only before submission. Its training callbacks were not reached in this failed job.
- `stage2.sbatch`: exact12h launch.
- `preflight.py`, `preflight.json`: CPU preflight; no CUDA initialized.
- `training_report.json`: original server failure report, not rewritten to hide its generic FAIL status.

Server control: `/data1/userdata/tcweng/projects/tcgs/outputs/deform360/t30-stage2-control-20260927/`
stdout: `d360-stage2-seed5-25874.stdout.log`
stderr: `d360-stage2-seed5-25874.stderr.log`
Intended workdir: `/data1/userdata/tcweng/projects/tcgs/outputs/deform360/stage2/config_a_seed5/`
No checkpoint exists from this job. Prior T26 epoch46 remains unchanged; it is an initialization source, not a Stage2 last-good checkpoint.

## Minimal next step (not applied)

Add explicit top-level `max_seq` to formal config (consistent with15 subwindows), then CPU-audit required tools/train.py fields and default CLI path before resubmission. Default --max_seq=100 must not truncate15 windows. Update config hash/protocol evidence explicitly; do not silently change frozen config or reuse failed control directory. No model/loader/loss/algorithm change is indicated. No automatic fix/retry performed. No T31.
