# T30.1 — Stage-2 protocol freeze (PASS)

2026-09-27. Only config/protocol and CPU static validation; no model/dataset instantiation, CUDA, sbatch or training. T30 is not a training PASS.

## Official/fresh equivalence

All counters in formulas are zero-based. Resume meta epoch15 starts training at runner.epoch15 (human epoch16); MMCV `while self.epoch < self._max_epochs`, increment after each train epoch, gives epochs15..75 inclusive =61. Fresh runner executes0..60, never forged15.

| Official runner epoch | Fresh / relative epoch | Requested rollout | Effective first14 / last window | LR |
|---|---:|---:|---:|---:|
| 15 (display 16) | 0 | 3 | 3 / 3 | 0.000404 |
| 16 (display 17) | 1 | 6 | 6 / 6 | 0.000204 |
| 17 (display 18) | 2 | 9 | 9 / 9 | 0.000204 |
| 18 (display 19) | 3 | 12 | 12 / 9 | 0.000104 |
| 19 (display 20) | 4 | 15 | 12 / 9 | 0.000104 |
| 20 (display 21) | 5 | 18 | 12 / 9 | 5.4e-05 |
| 21 (display 22) | 6 | 21 | 12 / 9 | 5.4e-05 |
| 45 (display 46) | 30 | 93 | 12 / 9 | 4.01220703125e-06 |
| 75 (display 76) | 60 | 183 | 12 / 9 | 4.00000037252903e-06 |

Official requested=`min(3*(15+k)-42,1000)`; fresh=`min(3*k+3,1000)`. Official Hood `.0004*(.5**(max(0,(15+k)-14)//2)+.01)` equals fresh Hood with `step_start=-1`. The original +.01 is an additive floor. All61 rows tested with AST-extracted actual `_rollout_steps` and `_sched_fun`, not a reimplemented scheduler. Training effective count is `min(gt_label length-1, requested)` (simulator line755); evaluation's existing one-step recurrent API is unchanged.

## Budget and data

Actual `_parse_idx` gives15 dataset entries for15 sequences and2cameras/max_cam2. RepeatDataset `__len__` multiplies by40:600 samples/epoch, batch1 × GPU1 ->600 updates, 61epochs ->36600. Windows `[i,min(i+13,150))`, i=0,10,...,140. Dense gap1/coarse10. All train/val/test diagnostic windows stay source113..262, within allowed [113,268). Source263..267 is unused under inherited window policy. T27 contains key150 but it is not a training window start. No test split or checkpoint selection by test.

## Initialization, Normalizer, optimizer

`load_from` is T26 epoch46; SHA256 `d4dbea360fa033bfbf0b1ec338ce3107d49c5549675effb481f817eb672932c4`. `resume_from=None`. `mmgs/apis/train.py:76,141-150` builds fresh Adam and uses runner.load_checkpoint; its installed implementation is recorded in static_validation.json. This loads model state including12 learned FP64 statistics, not optimizer/runner counters. Train mode retains cumulative Normalizer updates (normalization.py:126), until existing accumulation cap. No zero reset/warm-up. Runner0/0; fresh Adam(.9,.999), wd0, amsgradFalse, baseLR.0004, clip1; first actualLR.000404. CLI must supply `--seed 5` because tools/train.py overwrites cfg.seed.

## Fixed cache / evaluation policy

model.data_dir points to canonical gap10 root; cache files under config_a_2cam/pred_stage1. Training reads cached starts at simulator:729-731. Both update flags false. Training update call:798-799 is disabled; eval update guard:1051 is false because mode is segmented, not continuous. Consequently train-side evaluation cannot change later start states through update_gaussian. Within each window positions remain autoregressive; existing window-start covariance semantics unchanged. No future reconstructed PLY reset. This config-only fixed-cache choice intentionally differs from official evaluation flag true. No model source changed.

## Checkpoints / evaluation

Every epoch checkpoint and train-side eval; no save_best. Final selected `epoch_61.pth` (meta epoch61/iter36600), work_dir `/data1/userdata/tcweng/projects/tcgs/outputs/deform360/stage2/config_a_seed5/`. `data.test` deliberately points to same train-only segmented windows, not T31 continuous test. Preserve current loss and detach semantics. T24 numerical repeatability limitation remains non-gating.

## Validation and provenance

- [expanded config, 61-epoch equality, actual installed runner source](static_validation.json)
- [CPU cache/checkpoint hash/schema/finite verification](asset_validation.json):16 canonical files;12 learned FP64 stats; CUDA uninitialized.
- [machine-readable protocol](../../contracts/008-pink-cloth/episode_0/stage2_protocol_contract.json)
- [formal config](../../../../configs/SoMA/deform360_v0_stage2.py), SHA256 `6a8b07fcfd1d4322508d1546ebd06ba0c292deb638d5580ea629ae83b9887df6`.
- Official config/source hashes recorded; approved T29 expanded model differs only in train_cfg/test_cfg schedule and flag_update_gaussian. Model architecture/loss/renderer/Normalizer unchanged.

T30 BLOCKED audit preserved at commit db8b90ac46ba22731a9862728126036a6c97f023. T30.1 resolves protocol ambiguity only; no T30 training or T31 executed. New files remain uncommitted on Mac; server repository stays clean at audit checkpoint, pending future Git synchronization.
