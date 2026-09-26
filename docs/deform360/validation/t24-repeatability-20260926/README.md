# T24 independent reload determinism baseline audit

Result: **Case 2; T24 remains FAIL**. A (Slurm 25840, PID 1143490) and B (Slurm 25841, PID 1143618) are completely independent processes, each one eval/no_grad forward after real runner.resume of the same finite checkpoint; zero optimizer steps.

All A/B predicted position, covariance, both camera renders and every loss/metric are bitwise equal; max abs diff=0 and mean abs diff=0. This is evidence for this pair only, not a universal GPU determinism guarantee. Seed, model/config/input/initial Gaussian/checkpoint hashes, loaded optimizer hash, backend flags and runner counters match. Both independently match original saved model/optimizer/normalizer reference states exactly. Checkpoint, inputs and core source hashes unchanged.

Pre-save vs A and B still differ identically: pred_pos max=5.960464477539063e-08 / mean=4.309909814614748e-09; pred_cov max=4.3655745685100555e-11 / mean=9.115150849039588e-13; 023 render max=0.00018256902694702148 / mean=3.2954325416977994e-08; 009 render max=8.32974910736084e-05 / mean=3.2147387695726205e-08. Momentum loss diff=0.00011563301086425781. All outputs finite. Original atol=rtol=1e-5 retained, not replaced by a looser threshold.

## Read-only localization / remaining uncertainty

The discrepancy is between a post-training process and fresh resumed processes, not between these two fresh resumes. It already exists in pred_pos/pred_cov before rendering, so it cannot be localized exclusively to the rasterizer. The experiment does not establish a missing learned checkpoint tensor: state_dict and optimizer/normalizers restore exactly.

`gs_simulator_embodied.py:638` writes model.num_iter/num_epoch at forward_train entry; the fixed evaluation path directly calls `_preprocess` and `encode_decode` (around lines 231/276), neither consumes those old counter mirrors. Their 1→0 difference is documented but is not a demonstrated cause. `normalization.py:21–27,120–135` has registered accumulator parameters and an unregistered mode flag; eval is applied on both paths, accumulators are exact and frozen. Gaussian inputs are plain scene objects (`gs_simulator_embodied.py:188+`), independently loaded and hash/equality checked. No missing runtime field with a proven causal link has been found.

Further isolation would require comparing graph/feature intermediates, non-state_dict tensor layout/flags, RNG state and execution/backend state between a finite post-training process and a fresh resumed process. A/B identical initial seed does not prove equality of historical RNG state after training. Those original pre-save details were not captured. No new training, speculative model/runner fix, tolerance change or additional diagnostic forward beyond authorized A/B was performed.

## Artifacts and Git

A_output.pt and B_output.pt remain server-only at `outputs/deform360/t24-repeatability-20260926/`; small reports/logs are here. Tools: `tools/deform360_adapter/audit_checkpoint_repeatability.py` and `compare_checkpoint_repeatability.py`. CPU comparator only reads tensors; no CUDA initialized on login node. Prior recovery tool and checkpoint unchanged. No commit/push, no T25–T28. Old BLOCKED/FAIL history retained.
