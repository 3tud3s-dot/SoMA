# T27 canonical gap10 cache — PASS

Failure evidence checkpoint: b7bb92a0de8437796a6a2c8b832db5e8b580a2a9; Mac/origin/server clean 0/0 before helper edit. Job25872 COMPLETED/0:0,11s,Slurm GPU allocation; no training/optimizer or Stage2 instance.

## Minimal diff / code semantics

Only helper change: `mc.frame_gap = cfg.data.test.env_cfg.frame_gap` plus provenance comment. Frozen config, model, loader, Normalizer and checkpoint unchanged. Stage1 constructor default is1 (:71); only read of self.frame_gap is save_gaussian:447; filename at450. simple_test controller indexing894-895 and autoregressive updates940-945 do not use it. Official single_gpu_test/single_gpu_rollout -> simple_test -> save_gaussian unchanged. Stage2 loader219-241 expects filename stems and pred_pos/pred_cov;729-731 indexes gs_aligned_frame. No Stage2 test performed.

## Artifact

Server: `/data1/userdata/tcweng/projects/tcgs/datasets/deform360/derived/008-pink-cloth/episode_0/t27_stage1_cache_gap10/config_a_2cam/pred_stage1`

16 files / 7454328 bytes. Required fields only pred_pos [12861,3] float32 and pred_cov [12861,6] float32, packed upper triangle. Initial+15 predictions; no Gaussian reordering, no future-Ply reset. Initial covariance reuse unchanged. All tensors finite; every state12861/12861PD. Per-file SHA256, eigenspectrum/condition summaries in canonical_report.json and final contract. Old wrongly named files remain unchanged and are not canonical.

Checkpoint `d4dbea360fa033bfbf0b1ec338ce3107d49c5549675effb481f817eb672932c4`, epoch46/iter2300; model exact loaded, learned12FP64Normalizer statistics unchanged before/after, no warmup/accumulation.

## Mapping and displacement (meters)

| Key | Local | Source | Provenance | displacement p50 / p95 / p99 / max | PD |
|---|---|---|---|---|---|
| frame_0 | 0 | 113 | initial | 0 / 0 / 0 / 0 | 12861/12861 |
| frame_10 | 10 | 123 | predicted | 0.00319155 / 0.0372422 / 0.0479966 / 0.0503745 | 12861/12861 |
| frame_20 | 20 | 133 | predicted | 0.00298919 / 0.011252 / 0.018903 / 0.0301703 | 12861/12861 |
| frame_30 | 30 | 143 | predicted | 0.00290867 / 0.0125972 / 0.0157249 / 0.0498497 | 12861/12861 |
| frame_40 | 40 | 153 | predicted | 0.00439936 / 0.023654 / 0.0267501 / 0.0420938 | 12861/12861 |
| frame_50 | 50 | 163 | predicted | 0.00416086 / 0.0139652 / 0.0165371 / 0.0364054 | 12861/12861 |
| frame_60 | 60 | 173 | predicted | 0.00523978 / 0.01908 / 0.0208389 / 0.0370917 | 12861/12861 |
| frame_70 | 70 | 183 | predicted | 0.00526184 / 0.0174288 / 0.0192099 / 0.0310494 | 12861/12861 |
| frame_80 | 80 | 193 | predicted | 0.00532829 / 0.0191955 / 0.0213043 / 0.0318842 | 12861/12861 |
| frame_90 | 90 | 203 | predicted | 0.00755529 / 0.0197441 / 0.0210724 / 0.0294648 | 12861/12861 |
| frame_100 | 100 | 213 | predicted | 0.00560565 / 0.00965183 / 0.0128734 / 0.0300542 | 12861/12861 |
| frame_110 | 110 | 223 | predicted | 0.00231734 / 0.00591931 / 0.0102043 / 0.0358635 | 12861/12861 |
| frame_120 | 120 | 233 | predicted | 0.00874917 / 0.0262548 / 0.02927 / 0.031995 | 12861/12861 |
| frame_130 | 130 | 243 | predicted | 0.00808477 / 0.0261419 / 0.0291327 / 0.0432497 | 12861/12861 |
| frame_140 | 140 | 253 | predicted | 0.0153244 / 0.0336636 / 0.0370689 / 0.0429082 | 12861/12861 |
| frame_150 | 150 | 263 | predicted | 0.00902486 / 0.0274442 / 0.0337963 / 0.0359718 | 12861/12861 |

## Old/new regression

Old frame_i matches new frame_10i by schema/shape/dtype/count and semantic frame. All16 read back and hashes stable. Not all bitwise equal: maximum pred_pos difference2.6747584342956543e-6m, pred_cov2.5756889954209328e-9. No evident prediction-scale change; no claim of bitwise determinism or new numerical attribution. Full32 tensor comparisons in comparison.json. Maximum step displacement remains0.0503744541m; no tens-of-meters explosion.

Source/config/Normalizer code unchanged; server repo clean. T27 small evidence/helper/roadmap pending commit and server Git sync. No commit/push of successful T27, no T28.
