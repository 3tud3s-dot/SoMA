# T26.5 formal first-step regression — PASS

2026-09-26。仅执行 frozen T25 第一个 batch，未恢复完整 T26，未执行 T27。

## 前置 checkpoint

T26.4 commit `8b7c4507bc47eb5a033b9f707cbe898c7d76af5c`（`Fix Normalizer numerical stability`）已 push。
执行前 Mac/origin/server HEAD 相同，ahead/behind=0/0，均 clean。服务器原未提交 Normalizer 与远端字节一致；先备份至仓库外，移除已核验的同一 patch 后 fast-forward，再逐字节核对，未依赖未提交 production patch。

## 实际路径与停止边界

Slurm job **25866**，amax / 1× RTX 5090，COMPLETED exit0:0；Slurm 21s，observer 14.650s。
原始 `tools/train.py` + frozen `deform360_v0_stage1.py`，seed5、fresh model / zero-history FP64 normalizer、Config A、epoch1/iter1、rollout3、LR0.000404、batch1。
observer 没有覆盖 rollout/config、forward 参数或 backend flags。直接委托真实 `OptimizerHook.after_train_iter`；原方法执行 zero_grad → combined backward → clip_grads → Adam.step。
clip wrapper 在原 clip 之前检查全部梯度，在原 clip 返回后检查梯度，任何非有限值抛出异常以阻止后续操作。六次 native backward 返回也逐项检查。真实 hook 返回后检查模型/Adam/normalizer，立即结束进程；没有第二个 batch、额外 inference 或 checkpoint。
runner 内部 epoch/iter 在此停止点仍为0/0，因为迭代计数递增发生在 hook 之后；实际已经完成 epoch1/iteration1 的一次 Adam 更新。

## 三步 forward

位移是相对该 step 输入位置的 L2 距离，单位 m。F max 为本次实际 deformation matrices 中的最大奇异值。

| Step | Target source | Displacement p50 / p99 / max (m) | F max | Covariance PD |
|---|---|---|---|---|
| 1 | 123 | 0.043054 / 0.127785 / 0.140789 | 1.087950 | 12861/12861 |
| 2 | 133 | 0.004358 / 0.017818 / 0.019395 | 1.089396 | 12861/12861 |
| 3 | 143 | 0.006815 / 0.017332 / 0.018546 | 1.093408 | 12861/12861 |

初始 bbox diagonal≈0.524m；三步相对初始位置 max=0.140789 / 0.156368 / 0.146973m，没有几十米尺度爆炸。诊断有一个不改变结果的 gross-scale tripwire（max displacement from initial < 10×initial bbox）；PASS 判断同时依据上述实际分布、F、PD、render/backward，不以这个宽松 tripwire 独自证明合理性。

step2/3 输入位置与上一预测逐值相同（沿用原 detach）。**原源码每步复用 initial covariance，未改为预测 covariance 递推**。template Gaussian 仍为 source113；没有 future reconstructed PLY reset。
三个 target GT tensor hashes 与 dataset 中 local10/20/30 精确对应，source123/133/143。读取范围 source113…263，没有 source≥268。controller 与 T11 trajectory 逐值一致，grouping=30→10→2→1。
Config A 的 scene camera index0/1 沿用已核验 T17/T19 契约，对应023_cam0/009_cam1；底层 camera assets hashes 未变。

两camera × 三step render 全 finite，全部正radius Gaussian 的投影协方差 CPU FP32复算无非法项。这不是 kernel intermediate 直接读回。两camera positive-radius counts：step1=12861/12861，step2=12861/12826，step3=12861/12860。Native backward 六次返回的 means/cov/color/opacity/SH/scale/rotation 梯度全 finite。

## Combined backward / clipping / Adam

- 总 loss=23798.42578125；momentum=7.86843681，L2=19622.82617188，SSIM=4167.73144531，static=0，均 finite（原 avg_loss 语义）。
- 275/275 实际 gradient tensor finite；NaN / +Inf / -Inf = **0 / 0 / 0**。277 trainable 中原有两项 `backbone.encoder.emb_norm.weight/bias` 无梯度，并非本轮丢失梯度。
- FP64诊断 global norm：pre=**77384.46931221339**；最大单参数 norm=48978.27493740703（dynamic_proj.layers.2.bias）。
- 原 clip max_norm=1，默认 norm_type=2；原返回值=**77384.46875**；post=**1.000000029656491**。约3e-8超过1是浮点舍入，原阈值没有修改。
- 真实 Adam.step 一次，275个 parameter tensors 发生有限更新；275组 exp_avg/exp_avg_sq 全 finite，全部 step counter=1。
- 全 model state finite。3组×4个统计参数仍 FP64、finite；每组6次累计。node/anchor count1977，edge count32361；anchor mean≈[0.00133,0.10147,-39.25369]、std≈[0.66603,6.34259,1.08616]。累计 squared sum 的绝对大小包含 count 和 gravity²，不能据此误判尺度爆炸。
- Peak allocated=3946289152B (**3.6753GiB**)，peak reserved=4185915392B (**3.8984GiB**)，无 OOM。
- 794项资产 hash、Gaussian fields、frozen config均未变；Gaussian 参数不在 optimizer 内。

## 范围和证据

**原正式 T26 首 batch failure condition 已在 Normalizer patch 后通过。** T26 原 FAIL 历史保留；这不是46epoch/2300step训练成功，后续正式训练需单独授权。T24.5 limitation不变。

- [原始逐step报告](report.json)
- [Slurm/real runner log](slurm.txt)
- [机器可读contract](../../contracts/008-pink-cloth/episode_0/formal_first_step_contract.json)
- [观察工具](../../../../tools/deform360_adapter/check_formal_first_step.py)

本项未 commit/push。工具在服务器 outputs 中执行，server repository 保持clean；新的小型artifact需后续经Git同步，不同步训练产物。
