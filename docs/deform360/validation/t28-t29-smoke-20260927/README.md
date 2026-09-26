# T28 / T29 independent gated smoke results

T27 checkpoint `3683f12e938ad6893493bab06d36154ae2abe992` pushed; Mac/origin/server clean and0/0 before tests. Slurm25873 COMPLETED/exit0:0,8s. All CUDA confined to allocation.

## T28 — PASS

Actual EmbodiedDataset + GsSimulatorEmbodiedS2 constructor/cache initializer. Official cloth_lift_stage2.py first window [0,13), dense gap1, coarse gap10. Local0…12/source113…125; targets local1…12/source114…125 (12 possible transitions). Selected cache frame_0, N12861, pred_pos[12861,3]/pred_cov[12861,6] float32. No tail truncation, no extra113 offset or repeated ×10. Full13controller frames exact against T11, all13masked GT frames/camera exact against canonical RGB/mask. Config A023/009order checked. Cache initializer tensors exact against file; input tensors finite.

## T29 — PASS

Same batch object retained after T28 gate. Actual Stage2 forward_train through EpochRunner.run_iter / MMDataParallel.train_step, real OptimizerHook backward/clip/step. Preprocess observer proves current position/covariance exactly equal selected cache; controller frames0→1; source113→114. No future-Ply reset, no Stage1 simulator substitute.

One-dense-step smoke override only: max_rollout_step1 / step_increase_interval0. This is not the Stage2 training curriculum. Official config is loaded with sample path/name substitutions; only approved D360 geometry/environment entries replace sample-specific values. Model cache root points to T27; dataset stays T17. No new data copying/symlinks or core-source patch.

Weights: T26epoch46 learned weights. Fresh optimizer (not a formal resume). Official Stage2 Adam baseLR0.0004 (.0001 × official num_sample1 × n_gpu4); unchanged official LR is retained, not recomputed for smoke GPU count. Official Stage2 entry epoch_stage1=15 used for scheduler counter; Hood yieldsLR0.000404. This counter is a smoke choice following official entry, not a claim that T26 ended at15 or a frozen Stage2 resume protocol. Clip1 from official schedule. Model dt1/15 and datasetcompdt1/30/real_dt1/15 are inherited; no gravity/time change.

Loss total5406.4892578125; momentum5.5512504578, SSIM1869.05859375, L23531.87939453125, static0. 275/275gradients finite, NaN/+Inf/-Inf0. Preclipnorm152467.8164, clipreturn152467.8125, postclip1.0000000411. Exactly1optimizerstep;275parameter tensors updated. Model/Adam/runtime finite;12Normalizer statistics FP64. Training-mode statistics may update by original semantics; no reset or Normalizer patch change.

Peak allocated1447921664bytes / reserved1568669696bytes. Registered source assets plus dense RGB/masks and everyT27cache hash unchanged. Serverrepo clean. No checkpoint saved, no extra batch, no formal Stage2 training/T30.

## Evidence

- report.json: expanded smoke config, tensor/file hashes, independent T28/T29 results
- run.txt: Slurm process log
- tools/deform360_adapter/check_stage2_smoke.py: config glue and observers; algorithm delegated to original classes
- contracts/008-pink-cloth/episode_0/stage2_subwindow_contract.json
- contracts/008-pink-cloth/episode_0/stage2_optimizer_smoke_contract.json

Pending Mac SoMA/deform360-adaptation commit; server execution copy remains under outputs. No commit/push for T28/T29.
