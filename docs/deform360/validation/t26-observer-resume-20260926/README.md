# T26 resumed Stage-1 — PASS

## Final result

46 epochs / 2300 optimizer steps completed (50 inherited + 2250 resumed). Job25868 on amax / RTX5090: COMPLETED, exit0:0, Slurm elapsed 02:42:04; observer wall 9717.345s. Previous epoch1 job25867 elapsed83s; interruption/downtime is separate. Seed5, Config A, frozen rollout3→6→9→12→15. No NaN/Inf or OOM, no algorithm/protocol changes. Train sampled source113…263, no supervision source≥268 and no future reconstructed Gaussian reset. Positions remain autoregressive with original detach; original initial-covariance reuse semantics unchanged.

Original nonfinite failure, root-cause audits and observer interruption are preserved in roadmap. PASS means budget/state integrity, not prediction quality.

## Provenance and checkpoint

Git: `c8ec65ab672277b5bcb8f9d1bbb18313ff6feccd`. Config SHA256: `e65d1c7848bd222f55c0e18050c464eec5922aef76c1bc62a06813359cca4930`. Normalizer SHA256: `f78712dd0760fcf16a58431d7fdfd240a2e38cba89c400b405df93798f364220`.

Final server-only checkpoint: `/data1/userdata/tcweng/projects/tcgs/outputs/deform360/stage1/config_a_seed5_normalizer_fix_resume_epoch2/epoch_46.pth`

Size: 30042493 bytes. SHA256: `d4dbea360fa033bfbf0b1ec338ce3107d49c5549675effb481f817eb672932c4`.

Independent CPU-only torch.load: epoch46/iter2300, 289 model state entries finite;275 Adam states finite, all step2300;12 normalizer statistics FP64/finite. CUDA not initialized. Hash agrees with observer. Original epoch1 hash unchanged. All46 epoch checkpoints recorded with path/size/hash in training_report.json; each was read for metadata at save time. This is a CPU integrity check, not another full T24 inference audit.

Resume exact state gate passed; first batch epoch2/iter51/rollout6/LR0.000404, Adam50→51. Observer JSON fix only converts NumPy scalar/array types recursively; CPU tests passed; no extra training forward. Source asset/config/Normalizer hash guards passed; server Git clean.

## Loss / LR / memory trajectory

All epoch gradients finite. Preclip norm range 2622.88–430046.29; original clip=1 retained. Overall peak allocated 16.918GiB / reserved 18.420GiB. Loss across different rollout lengths is not directly comparable.

| Epoch | Rollout | LR | Mean loss | Peak allocated GiB |
|---|---|---|---|---|
| 1 | 3 | 0.000404 | 8077.148 | 3.703 |
| 2 | 6 | 0.000404 | 6615.710 | 6.651 |
| 3 | 9 | 0.000404 | 6401.460 | 9.863 |
| 4 | 12 | 0.000404 | 7800.296 | 13.111 |
| 5 | 15 | 0.000404 | 7870.439 | 16.359 |
| 6 | 15 | 0.000404 | 6273.882 | 16.206 |
| 7 | 15 | 0.000404 | 5662.445 | 16.201 |
| 8 | 15 | 0.000404 | 5425.958 | 16.294 |
| 9 | 15 | 0.000404 | 4962.543 | 16.299 |
| 10 | 15 | 0.000404 | 5534.233 | 16.272 |
| 11 | 15 | 0.000404 | 5726.207 | 16.326 |
| 12 | 15 | 0.000404 | 5093.440 | 16.315 |
| 13 | 15 | 0.000404 | 4821.468 | 16.612 |
| 14 | 15 | 0.000404 | 5239.315 | 16.454 |
| 15 | 15 | 0.000404 | 4551.825 | 16.671 |
| 16 | 15 | 0.000404 | 4817.401 | 16.789 |
| 17 | 15 | 0.000204 | 3662.512 | 16.820 |
| 18 | 15 | 0.000204 | 3628.854 | 16.705 |
| 19 | 15 | 0.000104 | 3315.712 | 16.766 |
| 20 | 15 | 0.000104 | 3197.639 | 16.896 |
| 21 | 15 | 5.4e-05 | 3113.875 | 16.918 |
| 22 | 15 | 5.4e-05 | 3014.752 | 16.891 |
| 23 | 15 | 2.9e-05 | 2962.903 | 16.856 |
| 24 | 15 | 2.9e-05 | 2979.413 | 16.885 |
| 25 | 15 | 1.65e-05 | 2921.779 | 16.876 |
| 26 | 15 | 1.65e-05 | 2896.345 | 16.855 |
| 27 | 15 | 1.025e-05 | 2866.114 | 16.843 |
| 28 | 15 | 1.025e-05 | 2854.407 | 16.849 |
| 29 | 15 | 7.125e-06 | 2858.554 | 16.844 |
| 30 | 15 | 7.125e-06 | 2899.558 | 16.843 |
| 31 | 15 | 5.5625e-06 | 2869.538 | 16.842 |
| 32 | 15 | 5.5625e-06 | 2856.757 | 16.843 |
| 33 | 15 | 4.78125e-06 | 2831.944 | 16.828 |
| 34 | 15 | 4.78125e-06 | 2820.031 | 16.829 |
| 35 | 15 | 4.390625e-06 | 2816.445 | 16.831 |
| 36 | 15 | 4.390625e-06 | 2810.941 | 16.833 |
| 37 | 15 | 4.1953125e-06 | 2810.982 | 16.831 |
| 38 | 15 | 4.1953125e-06 | 2807.491 | 16.833 |
| 39 | 15 | 4.09765625e-06 | 2813.514 | 16.836 |
| 40 | 15 | 4.09765625e-06 | 2809.983 | 16.836 |
| 41 | 15 | 4.04882813e-06 | 2795.604 | 16.837 |
| 42 | 15 | 4.04882813e-06 | 2790.178 | 16.841 |
| 43 | 15 | 4.02441406e-06 | 2781.723 | 16.840 |
| 44 | 15 | 4.02441406e-06 | 2788.890 | 16.837 |
| 45 | 15 | 4.01220703e-06 | 2799.880 | 16.842 |
| 46 | 15 | 4.01220703e-06 | 2783.460 | 16.848 |

Full component trajectory: [epoch_trajectory.csv](epoch_trajectory.csv). Checkpoint integrity: [final_checkpoint_verification.json](final_checkpoint_verification.json). Source report: [training_report.json](training_report.json).

## Delivery

Automation paused by user request; no next training run/T27. No commit/push. New small evidence belongs to SoMA/deform360-adaptation; Mac roadmap/evidence pending Git checkpoint and subsequent server sync. Dataset/checkpoints/full events log remain server-only. Historical numerical reproducibility limitation remains unchanged.
