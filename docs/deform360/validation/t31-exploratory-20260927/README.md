# T31 exploratory — SUCCESS (intermediate checkpoint only)

T30 stays INCOMPLETE/TIMEOUT; frozen61epochs not completed. Evidence commit9cd6fe0706f1b1ad4f4f1b0430bbaafb5dccd16e pushed, Mac/origin/server clean0/0 before this task. That checkpoint includes the previously authorized max_seq=1 fix, without further config change.

## Input and execution

Stage2 epoch22/iter13200, SHA25631223b8ca972db83a78412040ddee9dcf7e5836f63fb11cae54b5f91aec9380c. This is NOT final Stage2 checkpoint. Actual single_gpu_rollout + simple_test continuous mode, ConfigA, T11controller, dense local0..193/source113..306 (193transitions). Eval-only split override[0,194) according to T31; training config/dataset source unchanged. No backward/optimizer or T30 resume. Job25883 COMPLETED/0:0,76s;Python69.35s.

Continuous constructor uses canonical initialSH0 raw Gaussian, exact equal to T27frame0 pos/cov verified. Only checkpoint and frame0 were read by torch.load; future coarse caches rejected by guard. Constructor has onlyframe0; each later template originates from traced update_gaussian. No future reconstructed PLY state injected. Initial PLY path is recorded in report model.gs_scene; runtime position/covariance equality guards prove each consumed state origin, not a general OS file-access trace.

## Provenance and boundary finding

All193 current positions exact equal to the immediately preceding prediction (initial canonical forstep1). All19 online template updates atlocal10,20,...,190 verified against actual prediction pos/cov. Source267→268 consumed model prediction; source306 reached. Per-step hashes/origins in report.json and step_provenance.csv.

**There IS a previous-state boundary reassignment:** current position remains continuous, but at every10frame switch, prev_state equals current boundary prediction instead of previous dense frame. Existing simulator:1037-1053 first advances cur_state, then passes it as prev_pos to update_gaussian; :973-976 later reloads this value. Thus the boundary finite-difference velocity history is reset to zero. This is not GT/future cache injection, but it is a real temporal-state limitation; no fix performed. Covariance also follows existing template-block semantics, not unrestricted one-frame covariance recurrence.

## Numerical / exploratory observations

All steps predicted pos/cov finite;12861/12861 covariance matrices positive definite at every step (CPU float64 eigvalsh of actual packed covariance); both camera renders finite. No numerical explosion. Max single-step displacement0.044981m. Final from-initial displacement p50=.035024m,p99=.106821m,max=.179328m: motion amounts, NOT3D ground-truth errors.

Camera-aggregated existing-path PSNR first25.776→final17.646dB; L2 first.019414→final.073415. Images/GT differ by time, so this is an exploratory mismatch observation, not a benchmark or causal drift measurement. No hyperparameter/training continuation decision made from test observations. Registered model/Normalizer state unchanged. Full metrics per step retained.

## Launch history

25881 failed before Python because sbatch --wrap used sh/source. Corrected launcher to explicit Bash.25882 failed in diagnostic comparison only (cached tensor requires_grad, NumPy conversion lacked detach);0rollout steps. Corrected read-only diagnostic conversion; original report retained.25883 completed. No model/cache/renderer/loss changes.

## Artifacts

- report.json: complete per-step numerical and provenance evidence.
- step_provenance.csv: compact step/origin/boundary table.
- diagnostic_conversion_failure.json: preserved25882 report.
- ../../contracts/008-pink-cloth/episode_0/continuous_exploratory_contract.json: classification and limitations.
- ../../../../tools/deform360_adapter/check_continuous_exploratory.py: standalone observation tool.

Server output: /data1/userdata/tcweng/projects/tcgs/outputs/deform360/t31-exploratory-epoch22-v2-20260927/.
Server launch logs: outputs/deform360/t31-exploratory-control-20260927/{25881,25882,25883}.log.
New T31 evidence/tool uncommitted; T32 not executed; no final baseline PASS claimed.
