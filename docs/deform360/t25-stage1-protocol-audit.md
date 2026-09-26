# T25：Stage-1 baseline protocol audit

2026-09-26 · **PASS — 完整 Stage-1 protocol 已冻结** · `frozen=true`

用户已明确批准 seed=5；沿用诊断种子，不依据 test/performance 选择。第一版单 seed，未来 multi-seed 另行授权。所有关键参数现已有来源；保留官方 random-init rollout=3，无 rollout=1 warm-up，不提前规避 rollout9 显存风险。

可执行配置：[deform360_v0_stage1.py](../../configs/SoMA/deform360_v0_stage1.py)。权威冻结记录：[stage1_protocol_contract.json](contracts/008-pink-cloth/episode_0/stage1_protocol_contract.json)。必须经 sbatch，以 `--seed 5 --gpus 1 --launcher none` 运行现有训练入口；输出固定 `outputs/deform360/stage1/config_a_seed5/`。T25 不启动训练。

**历史（保留）：初次审计 BLOCKED，唯一缺口为正式训练 seed 未批准。** 以下原始审计叙述中的“本轮/未冻结/待批准”指初次审计时点，已由上方人工决定解决。

本轮只读审计并计算协议参数；未实例化 dataset/model/runner、未初始化 CUDA、未训练。没有创建 `configs/SoMA/deform360_v0_stage1.py`：本地 `configs/` 中无任何 Deform360 配置，既有工具只是 smoke/validation harness。以下均为有据候选值，不能把本文件视为已冻结的启动授权。

## 执行前置

已按明确授权删除未跟踪 `t24-checkpoint-20260926.zip`，没有移动或计算其 hash。进入审计前 `deform360-adaptation` 工作树 clean，HEAD=`7f0fa94116aa60cef3b333092b45e3a4ffdbefda`，ahead/behind=0/0。T24+T24.5 checkpoint 已在上一轮正常 push；所有原历史保留。

T24 仍 PASS — checkpoint state recovery；T24.5 仍 non-gating PARTIALLY_ATTRIBUTED。本轮 BLOCKED 与其数值重复性无关。

## 唯一缺少授权的训练参数

| 参数 | 现有证据 | 可选值 | 为什么不能替用户定 |
|---|---|---|---|
| training seed | 官方 config/README 未固定；`tools/train.py:47` 默认 None；T19/T20–T24.5 使用 5 做诊断 | 建议明确批准 `5`；或用户指定另一个固定整数 | 诊断种子不等于正式训练协议；T19 contract 明确未锁定未来训练顺序 |

`tools/train.py:139–145` 仅在传入 CLI `--seed` 时调用 `set_random_seed`，随后以 `args.seed` 覆盖 `cfg.seed`。将来只在 config 写 `seed=5`、却不传 `--seed 5`，不能保证生效。

**没有要求重新决定所有已有来源的参数。** 特别是官方 rollout=3 开始有明确依据，因此保留为候选；若用户希望采用 rollout=1 warm-up、不同增长速度或上限，必须另外明确完整 schedule。本轮没有自动实施这些变化。

## 参数展开

Provenance：O = official inherited，D = Deform360 contract，I = implementation-derived；U = 用户当前明确的 checkpoint policy。

| Parameter | Frozen value | Provenance / source |
|---|---|---|
| seed | **5** | 用户明确批准；非 test/performance 选择 |
| runtime | 单 GPU；Slurm allocation 内使用非分布式 launcher=none | O：`tools/train.py:55–57,99–106`；workspace GPU rule |
| epochs | 46 | O：`cloth_lift_stage1.py:28,281` |
| dataset length / steps | 原始 1 sample，repeat 后 50；50 steps/epoch；46×50=2300 optimizer steps | O+I：dataset `_parse_idx`、RepeatDataset、builder |
| batch / workers | 每 GPU 1；global batch=1；workers=0；shuffle=True | O：config:13–15、builder:53–63,125–139 |
| repetition | 50；反复读取同一 scene/sequence 的完整 coarse clip | O：config:29,191；不是随机时间裁剪 |
| rollout requested | zero-based epoch `e`：`min(3+3e,1000)` | O：config:35–39；simulator:584–612 |
| effective rollout | epoch 1–4：3/6/9/12；epoch 5–46：15 | I：16 states 提供 15 transitions；simulator:689 |
| configured max rollout | 1000；本数据实际最多 15 | O+I；不能混淆配置上限和真实执行长度 |
| frame_gap | 10 | O+D：T6/T19 |
| dt | model/backbone/decoder=1/15；dataset comp_dt=1/30；real_dt=1/15 | O+D：T6；dataset:921–934 |
| gravity | raw `[0,0,-9.8]`，identity rot_est；仅一次 ×4 → `[0,0,-39.2]` | D：T13；不是独立测量的物理竖直 |
| optimizer | Adam；betas=(0.9,0.999)，amsgrad=False | O：config:277 |
| base LR | `1e-4×num_sample(1)×n_gpu(4)=4e-4` | O：config:13,277；字面 n_gpu 不是 allocation 数量 |
| effective LR | `4e-4×(0.5**floor(max(0,e−14)/2)+0.01)` | O+I：HoodLrUpdaterHook:123–177 |
| LR examples | epochs 1–16=4.04e-4；epoch17=2.04e-4；epoch46=4.01220703125e-6 | I；不是简单对 4e-4 直接二分 |
| weight decay | 0 | O：config:277 |
| scheduler | Hood；by_epoch=True；decay_rate=.5；decay_steps=2；step_start=14；无配置 warmup | O：config:280 |
| active loss weights | momentum=1.0；L2 render=.9；SSIM render=.1，kernel=5；均配置 sum reduction | O：config:173–181；decoder/filter 实际调用 |
| other configured losses | static=1.0（实际零占位），spatial=1.0、L2 bounding=.9、SSIM bounding=.1（本 train path 未调用） | O+I：simulator:325–329,721–773；不能当成实际非零 loss |
| loss aggregation | camera loss 按各 loss 实现平均，rollout 各项损失取 mean，再 sum loss keys；selfsup_loss=False | O+D：avg_loss=True、T16 |
| detach / accumulate | accumulate_gradient=False；step 间 detach prev/cur position；预测继续自回归；cur_cov update 保持原注释状态 | O：base model:27；simulator:726–734 |
| optimizer accumulation / clipping | 不做跨 batch accumulation；每 batch 聚合 step losses 后一次 backward/step；grad_clip max_norm=1.0 | O：adam_hood:3；标准 optimizer hook，未选 CumulativeHook |
| logging | interval=100 iterations，CusTextLoggerHook | O：default_runtime.py:4–8 |
| checkpoint interval | 每 epoch；max_keep_ckpts=10000；真实 EpochRunner/MMCV checkpoint | O：config:281–282；T24 |
| checkpoint choice | 固定预算结束的 `epoch_46.pth`；不取 test 最优 | U+I；README epoch15 仅使用示例 |
| validation | 每 epoch；仅 train split；无独立 validation split；不接触 holdout | O+D：config:219–246,283 |
| validation actual horizon | 每次 rollout 全部15个 train transitions；`eval_start_frame=-1` 导致只记录最后 source263 的 metric | I：apis/test.py:164–239；早期也不按训练的3步截短 |
| train/test | source train[113,268)，test[268,307)；local train[0,155)，test[155,194) | D：T4/T5 |
| output path | server `outputs/deform360/stage1/config_a_seed5/` | I：server-only output rule；CLI `--work_dir` |
| initial load/resume | 正式 baseline 默认 random init，load_from/resume_from=None；不从 T24 diagnostic checkpoint 起训 | O：default_runtime.py:12–13 |

### 预算和 sample 的具体含义

Config A 的一条序列与两台 camera 构成一个 sample，`_parse_idx` 按 scene/sequence/camera batch 建 index，不按每帧建 index。Stage-1 sample 一次读取 source `[113,123,133,143,153,163,173,183,193,203,213,223,233,243,253,263]`。每次 train forward 都从同一 canonical source113 Gaussian 开始；后续位置来自上一步 prediction，没有 future reconstructed PLY reset。

计划继承值对应 **2300 次 optimizer step、33000 次训练 transition prediction、66000 次训练 camera render**，不含每 epoch 的 train-only evaluation。这只是静态展开，不是已运行计数。每 epoch 的 requested/effective rollout、实际 targets、LR 和 step 区间完整列于 JSON `single_gpu_inherited_budget.epoch_rows`。

`n_gpu=4` 在官方配置中只是 LR 表达式的一项；官方 README 的非分布式训练命令默认实际用 1 GPU。因此不因服务器只申请 1 GPU 就自动把 LR 改成 1e-4。若改成多 GPU DDP，采样 padding 与每 rank steps 都会改变；本候选不做这种改变。

### Rollout 稳定性限制

官方 `step_initial=3`，按 runner **zero-based epoch** 增长。代码中 `if self.is_est_vel: rollout_size=1` 已注释，不生效，无法解释为自动 warm-up。若继承官方，第一步确实是 random-init→rollout3。

T22 rollout1 finite、T23 已更新状态 rollout3 finite、T24 random-init rollout3 曾 nonfinite，均保留为不同实验条件。它们没有证明某个替代训练 curriculum 正确，也没有授权 cap=3/6 或 warm-up=1。rollout9 的历史 sample OOM 也不授权在本轮改显存策略。若后续实际训练 nonfinite/OOM，按当时任务约定停止。

## 官方与 Deform360 的差异

| 项目 | 官方 cloth_lift | Deform360 candidate | 依据 |
|---|---|---|---|
| scene/path/initial PLY | left_lift_1、sample 路径、初始 SH0 | T17 config_a_2cam、source113 SH0、12861 Gaussians；共享 canonical 资产 | T2/T4/T17 |
| cameras / calibration | sample 的3 camera | 023_cam0、009_cam1；T8/T9 calibration；640×360 | T7–T9/T19 |
| resolution 配置 | base env literal [960,540]；embodied camera 实际尺寸还读 calibrate WH | 明确 env [360,640]、WH [640,360]，已由 T19 验证 | T9/T17/T19；不能把 base literal 当真实 sample 分辨率 |
| controller / grouping | sample trajectory，首层 downsample_rate=.5 | candidate_7mm；0–14左/15–29右；30→10→2→1 | T10–T12 |
| gravity world rotation | sample gravity_rot_quat 的逆旋转 | identity；-Z user-approved engineering convention；×4一次 | T6/T13 |
| mask roles | cloth + hand/robotic arm obstacles | cloth；obstacle=[]；无 robot exclusion；全图 bbox | T14–T17 |
| train split | [0,150)，15 states/14 transitions | [0,155)，16 states/15 transitions | T4/T5/T19 |
| val / test routing | train、val、test 均指同一 sample split | train-only val；holdout单独记录，T25不建立test-reset路径 | T5；test绝不进入训练/选择 |
| effective rollout | 数据截断最多14 | 最多15；requested schedule 未变 | 数据接口推导 |
| checkpoint selection | 周期保存；epoch15示例，无best选择实现 | fixed budget final epoch46 | 用户明确要求 |
| seed | 未指定 | 待批准；不将诊断5冒充训练约定 | 人工决策 |
| output | CLI或work_dirs/config-stem | server outputs 目录模板 | workspace规则+实现推导 |

架构、graph radii、object cluster rates(.02/.2)、volume_scalar=512、rope attribute label、optimizer/LR/scheduler、loss、detach、logging/checkpoint/eval频率均有明确继承来源；不因适配而重构。`checkpoint_rollout=20` 虽在 config 中，实际当前模型只存储该字段，未找到使用它启用 activation checkpoint 的调用，不能据变量名宣称显存保护。

训练 sample 的相机 batch 顺序可能由现有 loader 在构造时 permutation；camera identity 与对应 GT/标定始终一起索引。T19 使用 seed5 得到的顺序不能当成所有未来训练的永久保证。本轮未修改 loader 或 camera selection。

## Checkpoint / evaluation policy

真实 runner checkpoint 保存 model、Adam buffers、12个normalizer统计tensor、epoch/iter和meta/config/seed；使用真实 resume 路径。T24已有exact-state证据。T24.5数值重复性不是training gate；不保证fixed outputs逐位一致。resume 后出现系统性loss/metric trajectory discontinuity时，才重新开启独立follow-up。NaN/Inf/OOM的运行安全要求仍保留。

现有 EvalHook 只计算并写入 metric，没有 best-checkpoint 选择逻辑。train-only evaluation 每 epoch 执行完整 coarse 序列，默认记录末帧；不创造新的验证划分，不将 test 结果用于超参或checkpoint选择。T25不为holdout直接套一个source268初始态；连续test rollout仍属于后续TODO。

## 验证和产物

- [机器可读审计](contracts/008-pink-cloth/episode_0/stage1_protocol_audit.json)：Status=BLOCKED，frozen=false；官方表达式行号、逐epoch展开、合同hash和源码SHA256。
- 静态解析真实config AST中的数值表达式；没有用假的loader或模型证明可运行，也没有完整实例化MMCV config（其文件加载依赖server sample路径）。已有environment config-summary只作历史展开交叉核对。
- 本轮审计涉及的17个核心config/source文件均逐字等于官方基线 `8e8772a`；本轮未修改源码。
- 可选服务器只读查询返回 `Connection closed by 10.157.195.126 port 22`；没有声称读取成功、服务器已同步或配置不存在于所有服务器位置。已确认**本地**该candidate文件不存在。
- 待确认seed后，才可将候选协议整体冻结并生成可执行config；本轮停止，不执行T26–T28。

SoMA/deform360-adaptation：本次只新增审计JSON与本说明、更新roadmap T25。未commit/push；待未来通过Git同步服务器。

## 最终冻结验证（2026-09-26）

- 新 config 无 sample 文件加载副作用；base 模型、scheduler、runtime 保持官方继承。静态递归展开后，model 除 scene/cluster/controller 映射外与官方逐项相等。
- env 完整替换为 T19 已验证 Config A，避免 base 中旧 scene key 混入；train/val/test 配置路由均为 train-only [0,155)，test 字段在此仅为官方接口兼容，不是 holdout 评价入口。
- 17 个审计源码/config hash 和全部输入 contract hash 均与审计记录一致；模型/dataset/loss 源码未修改。
- 46 epochs、50 steps/epoch、2300 optimizer steps、configured max=1000/effective max=15、完整逐 epoch LR/targets 已冻结。没有新增关键未决参数。
- T24 PASS / T24.5 non-gating 不变。原 BLOCKED 历史保留在本页和 audit JSON history；T26 运行结果另行记录。
