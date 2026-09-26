# T24 same-process fresh-instance isolation

**Case 3; T24 remains FAIL.** S1 differs from both S0 and fresh-process A under the original atol=rtol=1e-5 gate. No causal claim is made about an individual model field or process-global backend state. Stop here; first-divergence tracing is the next separately authorized task, not part of this run.

## Execution

- Slurm 25845 / PID 1148289 / RTX 5090: instantiate M0/optimizer/real EpochRunner, canonical seed 5 / Config A. Retain the existing T22 finite path's one no-update warmup forward; one rollout=1 training forward (source113→123), backward, finite gradients before Adam.step, finite model/Adam/normalizers after step, original CheckpointHook save C.
- M0 eval/no_grad → S0. In the **same** Python/CUDA process create new M1/optimizer/runner, real resume C, no training → S1. Then switch references back to M0 → S0_repeat; switch back to M1 → S1_repeat. Both model objects remain alive, with distinct object IDs. No RNG reset, empty_cache, backend flag intervention or state edits to force agreement.
- Process exits. Slurm 25846 / PID 1148551: fresh dataset/model/optimizer/runner, real resume C, eval/no_grad → A.
- New C only; prior checkpoints/references preserved. No model/runner/checkpoint algorithm change, no full-layer instrumentation, no forced deterministic algorithms, new CUBLAS flags, TF32 changes or tolerance changes.

## Required state checks

Before every S0/S1/repeat/A inference: checkpoint SHA256 unchanged; registered model/normalizer tensors exact to saved live reference; optimizer state exact; runner epoch=1/iter=1; parameter-name mapping exact; Gaussian fields exact; input/camera/controller hashes, module eval flags, dtype/device/strides/storage offsets/require-grad flags and direct fixed-forward arguments recorded and matching. The direct fixed routine takes no epoch/iter arguments. All registered state and observed runtime remain unchanged during each inference.

The same seed initializes each independent process. M1 is newly constructed without rewinding the shared process RNG, then overwritten by exact checkpoint resume; this preserves the requested process-history isolation. Snapshots record RNG state, without changing it.

## Findings

- S1 vs S1_repeat: every output/metric bitwise equal; every original gate passes.
- S0 vs S0_repeat: not bitwise equal, both renders exceed original gate. The repeat occurs after M1 construction/inference, as prescribed; it is not an immediate consecutive M0 repeat. Do not infer a permanent object-local defect from it.
- S0 vs S1, S1 vs A, S0 vs A: non-equal, render gate failures. S1/A and S0/A also fail momentum-loss gate.
- The simple object-local versus process-global dichotomy is insufficient here. This is Case 3 (possible object/layout/backend interaction, exact cause unresolved); no further guessing or backend diagnostics performed.

## Server-only artifact

`/data1/userdata/tcweng/projects/tcgs/outputs/deform360/t24-same-process-20260926/epoch_1.pth`

Bytes: 30024957. SHA256: `421070e708ca8920de7b661a2708ec974ec4008c7469e80f7585e17a571a7979`.

S0/S1/repeats/A output tensors and live reference tensors remain beside C on server. JSON state snapshots and logs here are lightweight evidence. SoMA/deform360-adaptation; no commit/push, no T25–T28. Earlier BLOCKED/FAIL history retained.

## All comparison values

Elementwise original gate: `abs(a-b) <= 1e-5 + 1e-5*abs(b)`. Bitwise equality and gate results are separate.

### S0_vs_S0_repeat

| Output | Max abs diff | Mean abs diff | Bitwise equal | Original gate |
|---|---:|---:|---|---|
| /steps/0/pred_pos | 2.98023223877e-08 | 1.5300067221e-09 | False | True |
| /steps/0/pred_cov | 2.18278728426e-11 | 5.7536319549e-13 | False | True |
| /steps/0/render/0 | 0.000756442546844 | 2.3315916493e-08 | False | False |
| /steps/0/render/1 | 0.00134861469269 | 2.32112840961e-08 | False | False |
| /loss/decode.loss_mse_momentum | 1.31130218506e-06 | 1.31130218506e-06 | False | True |
| /loss/decode.loss_ssim_render | 0.00048828125 | 0.00048828125 | False | True |
| /loss/decode.loss_l2_render | 0.001953125 | 0.001953125 | False | True |
| /loss/decode.acc_l2_render.acc | 7.45058059692e-09 | 7.45058059692e-09 | False | True |
| /loss/decode.acc_l1_render.acc | 7.45058059692e-09 | 7.45058059692e-09 | False | True |
| /loss/decode.acc_ssim_render.acc | 0 | 0 | True | True |
| /loss/decode.acc_psnr_render.acc | 0 | 0 | True | True |

### S1_vs_S1_repeat

| Output | Max abs diff | Mean abs diff | Bitwise equal | Original gate |
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

### S0_vs_S1

| Output | Max abs diff | Mean abs diff | Bitwise equal | Original gate |
|---|---:|---:|---|---|
| /steps/0/pred_pos | 2.98023223877e-08 | 1.5300067221e-09 | False | True |
| /steps/0/pred_cov | 2.18278728426e-11 | 5.7536319549e-13 | False | True |
| /steps/0/render/0 | 0.000756442546844 | 2.3315916493e-08 | False | False |
| /steps/0/render/1 | 0.00134861469269 | 2.32112840961e-08 | False | False |
| /loss/decode.loss_mse_momentum | 1.31130218506e-06 | 1.31130218506e-06 | False | True |
| /loss/decode.loss_ssim_render | 0.00048828125 | 0.00048828125 | False | True |
| /loss/decode.loss_l2_render | 0.001953125 | 0.001953125 | False | True |
| /loss/decode.acc_l2_render.acc | 7.45058059692e-09 | 7.45058059692e-09 | False | True |
| /loss/decode.acc_l1_render.acc | 7.45058059692e-09 | 7.45058059692e-09 | False | True |
| /loss/decode.acc_ssim_render.acc | 0 | 0 | True | True |
| /loss/decode.acc_psnr_render.acc | 0 | 0 | True | True |

### S1_vs_A

| Output | Max abs diff | Mean abs diff | Bitwise equal | Original gate |
|---|---:|---:|---|---|
| /steps/0/pred_pos | 8.94069671631e-08 | 6.15412923569e-09 | False | True |
| /steps/0/pred_cov | 3.27418092638e-11 | 9.64998222428e-13 | False | True |
| /steps/0/render/0 | 0.000756487250328 | 4.54709398852e-08 | False | False |
| /steps/0/render/1 | 0.00274060666561 | 5.14803990107e-08 | False | False |
| /loss/decode.loss_mse_momentum | 4.12464141846e-05 | 4.12464141846e-05 | False | False |
| /loss/decode.loss_ssim_render | 0.00244140625 | 0.00244140625 | False | True |
| /loss/decode.loss_l2_render | 0.00390625 | 0.00390625 | False | True |
| /loss/decode.acc_l2_render.acc | 7.45058059692e-09 | 7.45058059692e-09 | False | True |
| /loss/decode.acc_l1_render.acc | 7.45058059692e-09 | 7.45058059692e-09 | False | True |
| /loss/decode.acc_ssim_render.acc | 0 | 0 | True | True |
| /loss/decode.acc_psnr_render.acc | 1.90734863281e-06 | 1.90734863281e-06 | False | True |

### S0_vs_A

| Output | Max abs diff | Mean abs diff | Bitwise equal | Original gate |
|---|---:|---:|---|---|
| /steps/0/pred_pos | 8.94069671631e-08 | 6.2127568262e-09 | False | True |
| /steps/0/pred_cov | 3.63797880709e-11 | 9.61611631679e-13 | False | True |
| /steps/0/render/0 | 0.00029718875885 | 4.27172432393e-08 | False | False |
| /steps/0/render/1 | 0.00274068117142 | 5.58954181745e-08 | False | False |
| /loss/decode.loss_mse_momentum | 4.25577163696e-05 | 4.25577163696e-05 | False | False |
| /loss/decode.loss_ssim_render | 0.001953125 | 0.001953125 | False | True |
| /loss/decode.loss_l2_render | 0.001953125 | 0.001953125 | False | True |
| /loss/decode.acc_l2_render.acc | 0 | 0 | True | True |
| /loss/decode.acc_l1_render.acc | 0 | 0 | True | True |
| /loss/decode.acc_ssim_render.acc | 0 | 0 | True | True |
| /loss/decode.acc_psnr_render.acc | 1.90734863281e-06 | 1.90734863281e-06 | False | True |

## Exact event log


same-process M0/M1:

1. instantiate_begin
2. instantiate_complete
3. runner_optimizer_instantiated
4. warmup_forward_begin
5. warmup_forward_end_no_optimizer
6. runner_one_batch_begin
7. train_forward_complete
8. backward_complete
9. pre_step_gradient_finite
10. optimizer_step_complete
11. post_step_all_state_finite
12. checkpoint_write_completed
13. switch_eval (S0)
14. fixed_forward_begin
15. fixed_forward_end
16. S0_saved
17. fresh_M1_instantiation_begin_same_process
18. M1_resume_complete_same_process
19. switch_eval (S1)
20. fixed_forward_begin
21. fixed_forward_end
22. S1_saved
23. switch_eval (S0_repeat)
24. fixed_forward_begin
25. fixed_forward_end
26. S0_repeat_saved
27. switch_eval (S1_repeat)
28. fixed_forward_begin
29. fixed_forward_end
30. S1_repeat_saved
31. verification_complete_process_exit_next

fresh process A:

1. instantiate_begin
2. instantiate_complete
3. runner_optimizer_instantiated
4. resume_begin
5. resume_complete
6. switch_eval (A)
7. fixed_forward_begin
8. fixed_forward_end
9. fresh_reload_reference_saved
10. verification_complete_process_exit_next
