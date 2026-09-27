# T30 entry-config fix / first batch PASS

job25874 history remains BLOCKED/entry-config,0epochs/0steps; evidence commit c6fd7845aa369d700b836b341ec2150c7f316989 was pushed and Mac/origin/server clean0/0 before this fix.

## max_seq semantics

Official Stage2 configs/SoMA/cloth_lift_stage2.py:44 and Stage1:44 use max_seq=1. tools/train.py:30 default CLI100; :71-78 compares CLI cap against cfg.max_seq, and only when smaller adjusts env max_seq and max_cam_total. tools/test.py:121-128 has same optional cap. It is not rollout limit. Official config also uses it as initial scene num_seq and camera budget multiplier (:106,223-224); D360's existing nested values remain unchanged. EmbodiedDataset:289 reads env max_seq then :307 overrides it from len(split_list), yielding15; :324 passes that to scene/camera parsing. Actual curriculum is model.train_cfg, independent of this entry sentinel. DatasetAdjuster has a separate max_seq used for adaptive sequence length, not enabled in frozen config.

Production diff: only top-level `max_seq=1` plus comment. Default CLI100 does not truncate anything. Windows15/repeat40/batch1/GPU1 ->600steps/epoch,36600total,61epochs unchanged. T29 check_stage2_smoke.py:62 builds model directly, never calls train.py main; hence it did not exercise the missing top-level access.

## CPU entry preflight

check_entry.py compiles actual train.py parse_args/main AST and executes real config/CLI branches. File writes/logger/collect_env/seed side effects are mocked; build_simulator raises a sentinel at the boundary before model/CUDA. This is not a model smoke test. Config.fromfile/_base_ expansion and types checked, all61schedule rows unchanged; cfg data exact before/after entry; only max_seq differs from prior expanded config. Additional checks cover load_from/resume_from, workflow, model/data, optimizer/optimizer_config, lr_config/runner, checkpoint/evaluation/log config, log_level/dist_params, CLI seed/gpu_ids. CPU torch.cuda remains uninitialized.

## Formal launch

New job25875 on amax / RTX5090,12h. Frozen config hash c4042fa409bdb4bc2746629a390f9763aa7f277a8ec54d2e43e314b30566e16a. HEAD c6fd784 plus the explicitly authorized single config diff on both machines; config not committed. Observer records diff/hash and permits only this exact server diff. Prior job/control files preserved. Formal checkpoint workdir unchanged; it did not exist after25874.

Control/logs: /data1/userdata/tcweng/projects/tcgs/outputs/deform360/t30-entry-fixed-control-20260927/
Workdir: /data1/userdata/tcweng/projects/tcgs/outputs/deform360/stage2/config_a_seed5/

First batch PASS (see first_batch_gate.json): exact T26 model/statistics load; runner0/0; empty fresh Adam;12FP64stats. Canonical16cache states loaded exact. Shuffled first window start70/source183, targets184/185/186;requested/effective3,LR.000404;loss9431.357421875;275gradients finite,preclip192872.98352180913/postclip.9999999711029137;Adam.step completed with model/Adam/Normalizer finite. No future PLY reset. Peak allocated3830231552B (~3.57GiB). This is not full T30 PASS.

Normal TIMEOUT/preemption infrastructure resume is authorized by latest user instructions, only from last complete healthy Stage2 checkpoint, preserving Adam/counters/schedule. No numerical/OOM/schema auto-fix or retry; no T31. Training continues; final integrity/budget checks pending. New results/config remain uncommitted.
