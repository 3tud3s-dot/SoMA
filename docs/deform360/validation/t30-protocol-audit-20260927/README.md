# T30 protocol audit — BLOCKED before job submission

T28/T29 checkpoint033f47ff2731d657681fd9ba41da10a838047f05 pushed; Mac/origin/server matched,0/0,clean before audit. No formal deform360_v0_stage2.py exists; T29 glue is a one-step smoke, not a training protocol. No sbatch/CUDA/optimizer/experiment launched.

## Initialization and budget mismatch

Official README:108 uses --resume-from epoch_15.pth. train.py dispatcher mmgs/apis/train.py:141-150 distinguishes runner.resume (optimizer/counters) from load_checkpoint (weights). Installed MMCV BaseRunner.resume source captured in cpu_preflight.json; restores model/Normalizer, Adam,epoch/iter. EpochRunner.run_iter:19 forwards actual runner counters to model. cloth_lift_stage2.py:11,29,38 uses epoch_stage1=15, total maxepochs76, step_initial=-42. This constant is only rollout-offset configuration, not automatic conversion of checkpoint counters.

Official intended resume15 has61additionalepochs. Direct resume of actual T26epoch46 means30additionalepochs, first requestedrollout96, clamped to12or9available targets; first LR4.006103515625e-6. With15windows×repeat40×batch1/oneGPU,600steps/epoch:18000additionalsteps, globaliter20300, finalepoch76. Cannot silently describe this as rollout3 warm-start.

Weight-only loading retains learned Normalizer but starts runner0/freshAdam. Unchanged official step_initial=-42 then produces negative rollout/empty loop; adopting fresh-start requires explicit rollout/scheduler/budget decisions. T29 epoch15 + single-step settings do not resolve this.

## Evaluation mutates future training initialization

Official flag_update_gaussian=True, flag_update_gaussian_train=False, segmented evaluation. EvalHook:24-29 calls single_gpu_test on the same model; Stage2 simple_test:1051-1053 calls update_gaussian at coarse boundaries even in segmented mode. update_gaussian:502-510 replaces scene_init_pos/cov/prev_pos dictionary entries; forward_train:729-731 consumes those entries next time. Example first [0,13) validation subwindow can replace key10, later key10 training no longer exact T27 state. Disk hash preservation does not prove fixed cache provenance. These dictionaries are not registered model state (also relevant to future resume semantics if updates are retained).

Current task requires corresponding T27 start states. Suggested config-only resolution: flag_update_gaussian=False for segmented train-side evaluation; keep flag_update_gaussian_train=False. Not applied without confirming this departure from official behavior. Alternative is explicitly authorize evolving model-predicted start states; no GT/futurePLY injection either way.

## Static inputs

CPU-only:16T27file hashes exact, pred_pos[12861,3]/pred_cov[12861,6]float32/finite; T26checkpointhash verified epoch46/iter2300 with12FP64finiteNormalizer states. No CUDA initialized. Official windows start0,10,…140, ends min(start+13,150), exclusive. Controller/GT use same dataset video_range; sources≤262, inside approved≤267, no test data. Canonical key150 exists but unused under official15-window scheme. Extending tofulltrainend155/start150 would change window count/steps; not assumed.

## Fixed candidates versus unresolved

Seed5 candidate; gaps1dense/10coarse; modeldt1/15, datasetcompdt1/30 and externalgravity−39.2 from contracts. OfficialAdam baseLR.0004,betas.9/.999,weightdecay0,amsgradFalse;Hood(.5,2,14),clip1;lossmomentum1/L2.9/SSIM.1;per-epochcheckpoint/evaluation;positiondetach, no cross-batch accumulation. Initialization/budget/curriculum/LR start and in-memory cache-update policy are not frozen. Final-checkpoint selection only, no test selection. No executable formal config generated.

T30 staysBLOCKED; no trainingjob/GPU/walltime/loss/newcheckpoint to report. No protocol silently changed. T31/T32 untouched. Audit/roadmap pendingcommit/push.
