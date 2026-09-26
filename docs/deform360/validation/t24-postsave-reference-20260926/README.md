# T24 post-save reference S / fresh A / fresh B

**Result: Case B; T24 remains FAIL under unchanged atol=rtol=1e-5.**

S was computed after a new finite checkpoint was written, in eval/no_grad, with the same direct `one_step()` function and arguments as A/B. S still differs from A/B; A and B are bitwise identical. This rules out simply reusing a pre-step/train-mode intermediate reference. It does not identify a missing learned checkpoint tensor or a proven causal mutable Python state.

Old R is preserved at the original server path. Historical tool code already generated R after `runner.run` and checkpoint save; calling it “pre-save reference” was imprecise. This audit records the exact event order instead.

## Event order

Save job 25842: instantiate dataset/model/optimizer/real runner → one T21-style no-update warmup → train forward (rollout=1, source113→123) → backward → pre-step finite gradient guard → one Adam step → finite model/Adam/normalizer guard → real CheckpointHook save → runner completion/file hash → eval → no_grad fixed forward S → verify state unchanged → save S → exit.

New process A job 25843 and B job 25844 each: instantiate independently → real runner.resume same checkpoint → exact model/optimizer/normalizer/counter checks → eval → one no_grad fixed forward → verify state unchanged → save output → exit. No optimizer step in A/B.

All three use the same direct `_preprocess → encode_decode → _encode_decode_train` fixed-output routine, register_norm=True, rollout=1, fixed initial113/target123, same input tensors/cameras. That routine accepts no epoch/iter arguments. Runner counters are epoch=1/iter=1 on both sides; model.num_iter/num_epoch mirrors differ but are not read by this direct routine. No substitution of future reconstructed Gaussian.

## Targeted mutable-state audit

- Registered model tensors and optimizer restore exactly. 12 normalizer accumulators exact; eval mode flags equal and all inspected pre/post-forward state unchanged.
- Input/camera tensor hashes, parameter values/layout/flags, Gaussian scene dictionaries, render pipeline, preprocessing configuration and backend flags match S/A/B.
- scene_init_pos/cov/prev_pos and scene_initn1_pos/scene_init0_pos are absent in this actual simulator configuration; no invented restoration fields.
- Only inspected differences: model.num_epoch/num_iter mirrors (S=1; A/B=0), plus Python random state. Mirrors are assigned at `gs_simulator_embodied.py:638` and not read by fixed inference; Python RNG differs even A vs B although outputs equal, and RNG states do not advance during any fixed forward. Torch CPU/CUDA and NumPy RNG states match. These differences are not demonstrated causes.
- Scene/Gaussian and Normalizer state did not change during S/A/B inference. No direct mutable forward state with a proven causal difference was found. Unmeasured native backend execution state/intermediate arithmetic remains an uncertainty; do not equate Case B with proof of a missing model field.

## R versus S limitation

The old R and new S belong to separate one-step training runs. Their resulting checkpoint model tensors are not all bitwise equal. Thus R vs S is reported as requested but is confounded by training-result differences and cannot establish reference-capture timing as the cause. No attempt was made to investigate training stability or change algorithms.

## Checkpoint

New server-only: `outputs/deform360/t24-postsave-reference-20260926/epoch_1.pth`, 30024957 bytes, SHA256 `95e54bd6f97fd3508fb5e3dd75129a3b5a978a3dc28376c504c01e68b2e39723`.

Old checkpoint/R preserved. New S/A/B tensor outputs and reference tensors stay server-only. Local JSON snapshots and logs are small evidence; no checkpoint/model/runner logic edits, no tolerance change, no commit/push, no T25–T28.

## All requested comparisons

Original test is elementwise `abs(a-b) <= 1e-5 + 1e-5*abs(b)`; a scalar loss may exceed 1e-5 absolute while still passing the relative term.

### R_vs_S

| Output | Max abs diff | Mean abs diff | Bitwise equal | Original tolerance |
|---|---:|---:|---|---|
| /steps/0/pred_pos | 6.57513737679e-07 | 1.18792218935e-07 | False | True |
| /steps/0/pred_cov | 5.89352566749e-10 | 2.72375465729e-11 | False | True |
| /steps/0/render/0 | 0.00231321156025 | 5.22895140623e-07 | False | False |
| /steps/0/render/1 | 0.00555866956711 | 9.698331102e-07 | False | False |
| /loss/decode.loss_mse_momentum | 2.8133392334e-05 | 2.8133392334e-05 | False | True |
| /loss/decode.loss_ssim_render | 0.0029296875 | 0.0029296875 | False | True |
| /loss/decode.loss_l2_render | 0.04296875 | 0.04296875 | False | True |
| /loss/decode.acc_l2_render.acc | 1.78813934326e-07 | 1.78813934326e-07 | False | True |
| /loss/decode.acc_l1_render.acc | 2.60770320892e-07 | 2.60770320892e-07 | False | True |
| /loss/decode.acc_ssim_render.acc | 0 | 0 | True | True |
| /loss/decode.acc_psnr_render.acc | 1.33514404297e-05 | 1.33514404297e-05 | False | True |

### S_vs_A

| Output | Max abs diff | Mean abs diff | Bitwise equal | Original tolerance |
|---|---:|---:|---|---|
| /steps/0/pred_pos | 8.94069671631e-08 | 5.95814177016e-09 | False | True |
| /steps/0/pred_cov | 3.63797880709e-11 | 8.79617850328e-13 | False | True |
| /steps/0/render/0 | 0.000886857509613 | 3.74607928414e-08 | False | False |
| /steps/0/render/1 | 0.000533521175385 | 4.59834596869e-08 | False | False |
| /loss/decode.loss_mse_momentum | 0.000170826911926 | 0.000170826911926 | False | False |
| /loss/decode.loss_ssim_render | 0.0009765625 | 0.0009765625 | False | True |
| /loss/decode.loss_l2_render | 0.0009765625 | 0.0009765625 | False | True |
| /loss/decode.acc_l2_render.acc | 0 | 0 | True | True |
| /loss/decode.acc_l1_render.acc | 0 | 0 | True | True |
| /loss/decode.acc_ssim_render.acc | 0 | 0 | True | True |
| /loss/decode.acc_psnr_render.acc | 0 | 0 | True | True |

### S_vs_B

| Output | Max abs diff | Mean abs diff | Bitwise equal | Original tolerance |
|---|---:|---:|---|---|
| /steps/0/pred_pos | 8.94069671631e-08 | 5.95814177016e-09 | False | True |
| /steps/0/pred_cov | 3.63797880709e-11 | 8.79617850328e-13 | False | True |
| /steps/0/render/0 | 0.000886857509613 | 3.74607928414e-08 | False | False |
| /steps/0/render/1 | 0.000533521175385 | 4.59834596869e-08 | False | False |
| /loss/decode.loss_mse_momentum | 0.000170826911926 | 0.000170826911926 | False | False |
| /loss/decode.loss_ssim_render | 0.0009765625 | 0.0009765625 | False | True |
| /loss/decode.loss_l2_render | 0.0009765625 | 0.0009765625 | False | True |
| /loss/decode.acc_l2_render.acc | 0 | 0 | True | True |
| /loss/decode.acc_l1_render.acc | 0 | 0 | True | True |
| /loss/decode.acc_ssim_render.acc | 0 | 0 | True | True |
| /loss/decode.acc_psnr_render.acc | 0 | 0 | True | True |

### A_vs_B

| Output | Max abs diff | Mean abs diff | Bitwise equal | Original tolerance |
|---|---:|---:|---|---|
| /steps/0/pred_pos | 0 | 0 | True | True |
| /steps/0/pred_cov | 0 | 0 | True | True |
| /steps/0/render/0 | 0 | 0 | True | True |
| /steps/0/render/1 | 0 | 0 | True | True |
| /loss/decode.loss_mse_momentum | 0 | 0 | True | True |
| /loss/decode.loss_ssim_render | 0 | 0 | True | True |
| /loss/decode.loss_l2_render | 0 | 0 | True | True |
| /loss/decode.acc_l2_render.acc | 0 | 0 | True | True |
| /loss/decode.acc_l1_render.acc | 0 | 0 | True | True |
| /loss/decode.acc_ssim_render.acc | 0 | 0 | True | True |
| /loss/decode.acc_psnr_render.acc | 0 | 0 | True | True |

See `*_report.json` for ordered events and source/tool hashes; `*_before_forward_state.json` and `*_after_forward_state.json` for targeted state; `comparison.json` for exact measurements.
