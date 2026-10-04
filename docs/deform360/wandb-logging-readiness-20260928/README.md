# W&B historical import 与 Stage-2 logging readiness

本轮只新增独立 CPU importer，不修改训练算法、训练入口、optimizer、scheduler、checkpoint hook 或 frozen config。未停止/重启当前 Stage-2，未运行 CUDA。历史 run 的 `source=historical_import_from_events`，不代表原训练实时连接了 W&B。

## Stage-1 已完成导入

- Entity：`3tud3s-carnegie-mellon-university`
- Project：`SoMA-D360`
- Name：`stage1_pinkcloth_seed5_historical`
- Group：`stage1_baseline`
- Tags：`deform360`, `pink-cloth`, `no-tactile`, `historical`
- Run：<https://wandb.ai/3tud3s-carnegie-mellon-university/SoMA-D360/runs/historical-3382d747651c28499bd7>
- Run ID：`historical-3382d747651c28499bd7`
- 已核验云端 2300 条完整训练记录，step 1–2300 无缺失/重复，epoch=(step−1)//50+1，最终 epoch46。
- Final checkpoint SHA256：`d4dbea360fa033bfbf0b1ec338ce3107d49c5549675effb481f817eb672932c4`。
- 46 个 checkpoint 的 path/epoch/iter/size/hash 存在 run config；未上传 checkpoint 文件。

输入为服务器 `outputs/deform360/` 下：

1. `t26-normalizer-fix-control-20260926/events.jsonl`：step1–50。
2. `t26-resume-control-20260926/events.jsonl`：step51–2300。
3. `t26-resume-control-20260926/training_report.json`：训练 Git/config provenance、checkpoint metadata。

本轮离线包及机器可读校验留在服务器：
`outputs/deform360/wandb-stage1-historical-20260928/`。

| W&B field | 真实来源 |
|---|---|
| train/loss_total | optimizer_step_complete.losses.loss |
| train/loss_momentum | decode.loss_mse_momentum_config_a_2cam |
| train/loss_l2 | decode.loss_l2_render_config_a_2cam |
| train/loss_ssim | decode.loss_ssim_render_config_a_2cam |
| optimization/lr | 同一 iteration_start 的 LR |
| optimization/grad_norm_preclip | grad_norm / preclip_norm；原记录的数值精度保持不变 |
| optimization/epoch、iter | 原 observer 计数器 |
| runtime/gpu_*_bytes | 原 memory 字段，单位 bytes |
| runtime/source_elapsed_seconds | 各原进程开始后的秒数，不合成为虚构统一 wall clock |

Loss 保留原权重和聚合语义，未重算或重加权。未生成原日志中不存在的梯度直方图或历史系统遥测。SDK 导入期间的系统统计/code/git 自动采集禁用，避免将导入机器状态误当训练状态。

## Importer 与重复保护

工具：`tools/deform360_adapter/import_wandb_events.py`。

默认只解析校验并生成服务器本地 `metrics.jsonl` / `validation.json`；`--emit` 才生成 offline W&B run，随后明确 sync 指定目录。无需 torch/MMCV/GPU；只在 emit 时导入 wandb。

- Stage-1 必须覆盖1–2300；重复、缺失、非有限核心标量、epoch映射异常均拒绝。
- 一个输出目录只能创建一次；失败也保留目录，避免无意重复执行。
- Run ID 从 source hashes / metadata 派生。Stage-1 已上传，不要再次生成/重复 sync 为新 run。
- Stage-2 只允许每次一个 event file，每个 Slurm attempt 独立 run。
- 对活动日志固定读取字节上限，忽略未写完的末行；显式 `--last-step` 限定已完成 optimizer step，不追逐增长文件。
- 新的 Stage-2 snapshot 会形成不同 fingerprint；本工具不是增量实时 tailer。不要反复 emit 不同 snapshot 来冒充同一个 attempt 的实时更新。

运行 CPU 测试：
`python3 tools/deform360_adapter/test_import_wandb_events.py`

## Stage-2 真实日志转换验证

| Attempt | 本轮已校验 step 范围 | 条数 | 处理 |
|---|---:|---:|---|
| job25875 timeout | 1–13388 | 13388 | 验证转换，不创建 W&B run |
| job25884 resume | 13201–15000 | 1800 | 固定快照验证，不创建 W&B run |

这些是本次快照范围，不代表 Stage-2 最终进度或完成状态。原 job25875 report 的 IN_PROGRESS 字段因 TIMEOUT 未正常 finalize，不作为训练仍在运行的证据。

job25875 的最后完整 checkpoint 在13200；其后188次更新未纳入续跑起点。job25884重新从13201开始。因此两段日志不可按 iteration 合并去重：它们表示不同的实际更新。

建议独立命名：`stage2_timeout_attempt_25875`、`stage2_resume_attempt_25884`，共享 group `stage2_pinkcloth_seed5`。未来完成预算的 attempt 仍保留 job身份，summary 标注最终 checkpoint；不要额外复制一套曲线为虚构 final run。

已支持：requested/effective rollout、cache key/hash、window local/source start/end、dense target start/end、pre/post-clip norm、loss、LR、显存。窗口末端来自真实 cache_provenance.source_window，使用 inclusive end。checkpoint metadata来自对应 report，可能包含继承的旧 checkpoint，须保留该 provenance。

本轮验证输出（server-only）：
- `outputs/deform360/wandb-stage2-readiness-25875-20260928/`
- `outputs/deform360/wandb-stage2-readiness-25884-20260928/`

## 未来实时接入方案（未启用）

当前 `CusWandbLoggerHook` 会调用 `wandb.watch(model)`，本轮不启用。优先独立 CPU observer adapter：消费既有 events，不进入 forward/backward，不增加 renderer 调用。

第一版采用 offline：adapter 的 W&B run 使用 `WANDB_MODE=offline`（本 importer显式mode=offline），结束后对精确 run目录执行 `wandb sync --entity 3tud3s-carnegie-mellon-university --project SoMA-D360 <offline-run-directory>`。不要全目录盲目sync。服务器已安装wandb0.30.0，本轮使用既有认证，未读取或显示token内容。

真正实时 tailer 尚需独立实现 cursor/断点与重复保护；当前工具只能转换有限快照。无需为此重启训练。若后续改用进程内hook，应默认关闭watch，验证与TextLogger共享log_buffer的reset顺序，明确eval_train命名，保持训练数值与checkpoint路径不变。

风险：JSON读取/SDK写入有CPU与磁盘成本；online网络可能阻塞，因此不放入训练关键路径。离线sync仍需网络/账号权限。跨attempt的iteration回退必须分run表示，不能用W&B resume替代真实runner.resume。

本轮未commit/push；既有 `docs/deform360/reports/` 未触碰。
