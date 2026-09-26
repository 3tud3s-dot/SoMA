# SoMA-D360-v0 implementation TODO

最后更新时间：2026-09-24。

本文件是 **SoMA-D360-v0 implementation 的唯一 TODO source of truth**。后续执行任何 TODO 前，必须先读取本文件，确认当前任务、前置依赖、允许修改的范围、验证方法和 PASS / FAIL 条件，不依赖聊天记录重新猜测任务内容。

## 当前项目状态

```text
Project:
SoMA-D360-v0

Goal:
Deform360 → SoMA no-tactile baseline
之后再加入 tactile conditioning

Dataset:
008-pink-cloth / episode_0

SoMA branch:
deform360-adaptation

SoMA HEAD:
8e8772a98f5eeb332745d9b6dd008e6927f050af

Deform360 HEAD:
d8522a4403b766aeb387510c04e89032a56fdf35

Current task:
T24.5

Current status:
PARTIALLY_ATTRIBUTED
```

以上 HEAD 为本 roadmap 建立时的源码基线。T0–T11 已 PASS，具体结论及证据保留于各项历史 Result / Evidence。T11 checkpoint 已提交并推送；T12 已按人工确认采用 30→10→2→1 grouping contract 并验证 PASS；此前默认分组 FAIL 历史保留，不进入 T13。

## 执行规则

- 一次只允许执行一个 TODO。只有明确收到“执行 Tn”的指令，才执行对应任务。
- 当前 TODO PASS / FAIL 并汇报以后必须停止。未经明确指令不得自动进入下一项。
- 执行前读取本文件和适用的 AGENTS.md，确认当前任务及前置依赖。前置依赖以既定顺序、各项输入文件和正文引用的 TODO 产物为准。
- 每次只解决对应 TODO 的问题，遵守该项允许修改的范围；FAIL 时仅定位当前问题，不自动进入下一项。
- 不因执行一个 TODO 而删减、合并、重排或重写 roadmap，不提前改变技术方案。
- 主线状态只允许使用：`TODO`、`IN_PROGRESS`、`PASS`、`FAIL`、`BLOCKED`。用户授权的独立 Non-gating diagnostic T24.5 使用 `ATTRIBUTED`、`PARTIALLY_ATTRIBUTED`、`UNRESOLVED`；其分类不改变 T24 PASS，也不阻塞 T25+。
- 历史审计和聊天中的分析不是执行结果。T0–T6 的执行结果见对应 Result / Evidence；后续任务以各自 Status / Result / Evidence 为准。

### 后续更新规则

执行某个 TODO 后，只允许修改：

1. 对应 TODO 的 Status。
2. 对应 TODO 下的 Result / Evidence。
3. 顶部 Progress 表格。
4. Current task。

不得因为执行一个 TODO 而重写整个 roadmap。执行后按以下格式汇报，并停止：

```text
Tn: <名称>

检查/修改：
...

验证：
...

结果：
PASS / FAIL

证据：
...

Git diff：
...

下一步：
下一 TODO 的 ID 和名称；仅列出，不执行。
```

## Slurm GPU execution rule

在本项目中，任何需要 CUDA 初始化的操作均不允许直接在 SSH login node 执行，包括但不限于：

- `torch.cuda` 调用；
- GaussianModel GPU load；
- CUDA tensor 创建；
- 使用 `nvidia-smi` 验证 GPU 状态。

必须通过 Slurm GPU allocation 获得 GPU 后再执行：

- 短任务：`srun -p <partition> --gres=gpu:<num> ...`。
- 长任务：`sbatch`，在作业脚本中申请 GPU，并在获配 GPU 的作业内执行。

如果 CUDA 初始化失败，必须首先依次确认：

1. 当前 shell 是否位于 GPU allocation 内。
2. `nvidia-smi` 是否在该 allocation 内正常。
3. `torch.cuda.is_available()` 是否在该 allocation 内正常。

不要把 login node 的 CUDA failure 误判为数据错误、PLY schema 错误、CUDA 环境损坏或 GPU 故障。

## Artifact and synchronization rule

1. **小型定义性 artifact 必须进入 Git 管理。** 包括 `*.md`、`*.json`、`*.yaml`、`*.py`、`*.sh`，以及 manifest、contract 文件。例如 `frame_manifest.json`、`split_contract.json`、`camera_manifest.json` 和 scene metadata contract。
2. **大型数据文件不进入 Git。** 包括 `*.ply`、`*.mp4`、`*.h5`、`*.npy`、checkpoints 和 cache。大型数据保留在服务器，但必须有纳入 Git 管理的可追踪 manifest/hash 记录。
3. **Mac 与服务器的同步分工：** 代码、文档和小型 artifact 使用 Git 同步；数据集和训练产物保留服务器。不允许只在服务器生成长期需要的小型 JSON 而不将其纳入 Git 管理。纳入独立源码仓库管理，不在 tcgs root 初始化 Git 仓库；同步时仍须遵守 dirty working tree 保护规则。
4. **每个 TODO 完成后必须汇报：** 新增/修改文件、各文件是否 tracked by Git，以及是否需要同步到另一端。明确区分已纳入 Git 管理与尚未跟踪、已同步与待同步，不把服务器文件存在视为已进入版本管理。

## Git repository and commit rule

SoMA repo 是 SoMA-D360-v0 的默认开发仓库。后续涉及 SoMA docs、SoMA configs、SoMA tools 和 SoMA source code 的变更，默认在 `SoMA/` 仓库的 `deform360-adaptation` branch 进行 Git commit/push。

tcgs root Git 仅用于 workspace-level 资产。除非明确说明，不在 tcgs root 提交 SoMA-D360-v0 日常开发文件。不改变现有 SoMA、deform360、PhysTwin 独立 Git repo 的边界。

每个 TODO 完成后的 Git 汇报必须包含：

- 修改文件所属 repository；
- branch；
- commit/push 状态；
- 是否需要另一端同步。

## Contract 权威版本

长期 contract 的权威版本位于 `SoMA/docs/deform360/contracts/008-pink-cloth/episode_0/`：`frame_manifest.json` 与 `split_contract.json`。本次归档保留 JSON 原始字节和 provenance/hash；路径解析及旧服务器路径与仓库路径的对应关系见该目录的 `README.md`。后续任务应读取仓库内版本，不将服务器生成位置作为唯一引用。T0–T5 历史 Result / Evidence 中的路径保留为当时的执行记录。

## Progress

| ID | Task | Status |
|----|------|--------|
| T0 | 确认 frame 113 是否应该作为 SoMA-D360-v0 的 initial frame | PASS |
| T1 | 定位原始 PLY 与 SoMA loader 的兼容性阻塞 | PASS |
| T2 | 仅解决零高阶 SH 的文件兼容 | PASS |
| T3 | 确认 scale、opacity、rotation 的解释保持一致 | PASS |
| T4 | 固定原始 frame 与本地 frame 的一一映射 | PASS |
| T5 | 固定训练／测试边界 | PASS |
| T6 | 明确采样间隔与模型内部时间尺度 | PASS |
| T7 | 选定两个固定 camera ID | PASS |
| T8 | 只解决相机外参转换 | PASS |
| T9 | 只解决目标分辨率对应的内参 | PASS |
| T10 | 定义单帧 30 点 controller representation | PASS |
| T11 | 将固定 controller representation 扩展为 trajectory | PASS |
| T12 | 确认 controller 层级分组不跨手指 | PASS |
| T13 | 确认重力方向与世界坐标 | PASS |
| T14 | 只导出 RGB | PASS |
| T15 | 只导出 object mask | PASS |
| T16 | 明确 object mask 与遮挡 loss 的边界 | PASS |
| T16.5 | 批量生成 SH0 compatible Gaussian sequence | PASS |
| T17 | 只组装 SoMA scene metadata 与目录契约 | PASS |
| T18 | 验证初始静态渲染的几何对齐 | PASS |
| T19 | 让 EmbodiedDataset 读取一个 sample | PASS |
| T20 | 只验证 graph construction | PASS |
| T21 | 只运行一次 forward/render/loss | PASS |
| T22 | 只验证一次 backward 与 optimizer step | PASS |
| T23 | 只验证 rollout=3 | PASS |
| T24 | 验证 checkpoint 保存与恢复 | PASS |
| T24.5 | 数值重复性 first-divergence audit（Non-gating） | PARTIALLY_ATTRIBUTED |
| T25 | 冻结首个 baseline 的运行协议 | TODO |
| T26 | 执行约定预算的 Stage 1 训练 | TODO |
| T27 | 只生成并核验 Stage-1 cache | TODO |
| T28 | 只串联一个 Stage 2 子窗口 | TODO |
| T29 | 只验证 Stage 2 一个优化步骤 | TODO |
| T30 | 执行固定预算的 Stage 2 训练 | TODO |
| T31 | 验证真正的 continuous rollout | TODO |
| T32 | 汇总首个 no-tactile baseline 结果 | TODO |

## 路径与边界

所有路径均相对于 **tcgs root**，路径本身不再添加 workspace 目录名前缀。路径以行内代码展示，不使用依赖本文档所在目录的相对超链接。

- 输入 episode：`datasets/deform360/processed/008-pink-cloth/episode_0`。
- 拟新增的单组件工具目录：`SoMA/tools/deform360_adapter/`；目前不创建，具体工具文件名尚未确定，不预设文件名。
- 拟新增的 Stage 1 配置：`SoMA/configs/SoMA/deform360_v0_stage1.py`。
- 拟新增的 Stage 2 配置：`SoMA/configs/SoMA/deform360_v0_stage2.py`。
- Gaussian loader 的 workspace 内路径：`SoMA/gaussian-splatting/scene/gaussian_model.py`；该依赖实际在服务器执行，不要求 Mac 上存在或运行 CUDA 环境。
- `<camera_id>` 表示对应 TODO 中选定或核验的真实相机目录名。派生 scene、cache、checkpoint 等输出路径未确定时，明确引用其产生 TODO 的产物，不虚构具体文件名。

后续派生数据写入服务器的**独立新目录**，保留 `datasets/deform360/processed/008-pink-cloth/episode_0` 不变。原则上通过派生文件和独立配置满足现有接口；源码修改必须有当前 TODO 的失败证据支持。

---

## 第一阶段：初始状态与时间定义

### T0：确认 frame 113 是否应该作为 SoMA-D360-v0 的 initial frame

Status: PASS

**问题：** 113 是否在局部检查中确实对应官方 contact-start？

**为什么现在解决：** initial Gaussian、controller 初态和时间编号都依赖这个原点。

**输入文件：**

- `datasets/deform360/processed/008-pink-cloth/episode_0/brics-odroid_tactilel_left/synced_tactile.npy`
- `datasets/deform360/processed/008-pink-cloth/episode_0/brics-odroid_tactilel_right/synced_tactile.npy`
- `datasets/deform360/processed/008-pink-cloth/episode_0/brics-odroid_tactiler_left/synced_tactile.npy`
- `datasets/deform360/processed/008-pink-cloth/episode_0/brics-odroid_tactiler_right/synced_tactile.npy`
- `datasets/deform360/processed/008-pink-cloth/episode_0/metadata.json`
- `datasets/deform360/processed/008-pink-cloth/episode_0/split.json`
- 可辅助读取 `datasets/deform360/processed/008-pink-cloth/episode_0/robot/robot.npy`。

**预计涉及文件：**

- 无。

**最小修改：** 无；只读检查 108–118，不调用 Deform360 processing pipeline。

**验证方法：** 严格复用 `deform360/deform360/processing/control_points_stage.py` 中的 `_active_taxel_counts`、`_episode_active_frames`（第 131 行附近）的规则：

- 只统计 sensor 的前 12 行，值 **`>0`** 的 taxel；
- 同一 gripper 的左右 sensor 计数相加，**合计 `>1`** 才 active；
- 各 gripper 的 active 取 OR；
- `CONTACT_PATIENCE=5` 用于 contact 结束判定，不要求开始前连续 active 五帧。

输出 108–118 的四 sensor 计数、两个 gripper 的聚合结果、episode active、窗口内首个 active frame。可附 opening 和相邻 EEF translation delta。

**PASS：** 108–112 均 inactive，113 active，窗口内首个 active 为 113，且与发布 metadata/split 起点一致。

**FAIL 后检查：** sensor 分组、12 行截取、数组索引和发布版本差异；不改阈值、不换 initial frame、不进入 T1。

**证据边界：** 这验证局部 contact-start sanity。仅检查 108–118，不能证明整个 episode 在 108 之前从未 active；若必须证明全局最早起点，需要另行明确扩大只读范围。

#### Result

2026-09-24：PASS — adopt official processed start frame。根据用户明确确认，T0 的最终目标是确定数据时间原点，而不是重新定义 contact onset：**source frame 113 = SoMA-D360-v0 local frame 0**。

- detector first active in checked range: **108**（检查范围 108–118；不代表全局首次物理接触）。
- official processed start frame: **113**。
- adopted initial frame: **113**。

采用原因：官方 split/metadata 已以 113 作为有效 dynamics window 起点；按用户确认的项目约定，Gaussian、robot、tactile、train/test split 均围绕 113–306 对齐。本 PASS 表示采用官方 processed 时间原点，不表示 113 是 first physical contact，也不表示先前 detector 的结果被推翻。

上述用户确认取代本项原始“108–112 inactive、113 首次 active”的验收要求；原始条件与实测记录保留以追溯差异。无需为通过 T0 调整 detector、阈值或 sensor 分组。下一步准备 T1，本次不执行 T1。

#### Evidence

- 用户于 2026-09-24 明确确认采用官方 processed start frame：detector first active in checked range = 108；official processed start frame = 113；adopted initial frame = 113；source frame 113 映射为 local frame 0。
- 以下为此前真实检查证据，本次仅记录时间原点决策，未重新计算数据。当前源码 detector 与发布窗口的差异保留，不再作为采用官方起点的阻塞。
- 运行位置：服务器 tcgs root；解释器为现有 soma 环境 Python，以 `-B` 禁止写入字节码。NumPy 以 `mmap_mode='r'` 打开四份输入，只计算 `[108:119]`，未调用 processing pipeline、未生成数据。
- 通过 AST 从服务器当前 `deform360/deform360/processing/control_points_stage.py` 提取原始纯函数及常量执行，没有重新实现判据：`_active_taxel_counts`（131）、`_contact_range_from_active`（140）、`_gripper_group`（193）、`_episode_active_frames`（202）。源码 SHA256：`9ff82c86c22e38c56dd2ce5d872850afb6ffeb502da7338baf0b55108afb7373`。
- 判据：行 0–11 的值 > 0；同一 gripper 两 sensor 计数相加 > 1；gripper 间 OR；CONTACT_PATIENCE=5 仅影响结束判定。
- 四份输入 shape 均为 `[357,16,32]`，dtype 均为 `float32`。下表 LL/LR/RL/RR 依次对应输入列表中的 tactilel_left、tactilel_right、tactiler_left、tactiler_right；L/R 仅表示文件名分组，不推断实际执行机械臂身份。

| Frame | LL | LR | RL | RR | L sum | L active | R sum | R active | Episode active |
|---|---|---|---|---|---|---|---|---|---|
| 108 | 18 | 14 | 0 | 1 | 32 | true | 1 | false | true |
| 109 | 18 | 14 | 0 | 1 | 32 | true | 1 | false | true |
| 110 | 18 | 14 | 0 | 1 | 32 | true | 1 | false | true |
| 111 | 18 | 12 | 0 | 1 | 30 | true | 1 | false | true |
| 112 | 17 | 12 | 0 | 1 | 29 | true | 1 | false | true |
| 113 | 17 | 12 | 0 | 3 | 29 | true | 3 | true | true |
| 114 | 17 | 12 | 0 | 3 | 29 | true | 3 | true | true |
| 115 | 17 | 12 | 0 | 3 | 29 | true | 3 | true | true |
| 116 | 17 | 12 | 0 | 3 | 29 | true | 3 | true | true |
| 117 | 17 | 12 | 1 | 4 | 29 | true | 5 | true | true |
| 118 | 17 | 13 | 0 | 4 | 30 | true | 4 | true | true |

- 实际 `datasets/deform360/processed/008-pink-cloth/episode_0/metadata.json`：start_frame=113、end_frame=306、frame_num=194；同目录 `split.json`：frame_len=194、train=[113,268]、test=[268,307]。
- 先前 detector 验收 FAIL 的定位（实测不变）：sensor 按 `_left`/`_right` 后缀剥离分组，截取原始数组的 108–118 并统计前 12 行；当前源码第 293、322–328 行加载全部 streams 并取 OR，未按 bimanual 过滤 contact sensor。`deform360/deform360/layout.py` 第 56–61 行只按目录名称列出 sensor，不赋予机械臂语义。发布窗口采用的生成版本/选组规则仍未确认，不能擅自把 tactilel 组排除。
- 首次临时诊断命令因 AST 新节点缺少 lineno 在加载数据前退出；补上 `ast.fix_missing_locations` 后成功运行，未改动源码或判据。
- 两端 SoMA HEAD：`8e8772a98f5eeb332745d9b6dd008e6927f050af`，branch `deform360-adaptation`；两端 deform360 HEAD：`d8522a4403b766aeb387510c04e89032a56fdf35`。执行前服务器两仓库 clean；本地仅本 roadmap 未跟踪，tracked/staged diff 均为空。
- 本轮仅更新本地 roadmap 的 T0 Status、Result / Evidence、Progress 行。未同步文档、未修改源码/config、未运行训练、未 commit/push，T1–T32 未执行。

### T1：定位原始 PLY 与 SoMA loader 的兼容性阻塞

Status: PASS

**问题：** 原始 `datasets/deform360/processed/008-pink-cloth/episode_0/splatfacto/splat_113.ply` 能被哪些实际 loader 设置读取？

**为什么现在解决：** 区分文件损坏、SH 配置不匹配与其他属性问题。

**输入文件：**

- `datasets/deform360/processed/008-pink-cloth/episode_0/splatfacto/splat_113.ply`

**预计涉及文件：**

- `SoMA/gaussian-splatting/scene/gaussian_model.py`：只读检查 `load_ply`。
- `SoMA/mmgs/datasets/embodied_dataset.py`：只读检查 `_load_cluster_mask`。

**最小修改：** 无模型修改；仅建立隔离的加载诊断。

**验证方法：** 服务器上分别检查 `GaussianModel(3)` 和当前 `GaussianModel(0)` 路径，记录完整异常与 tensor shape。

**PASS：** SH3 正常读取 12,861 点；SH0 的预期阻塞被准确定位到 45 个 `f_rest_*` 与 degree 不匹配，没有混入路径或环境错误。

**FAIL 后检查：** 实际 import 来源、PLY 属性数量、loader 参数和异常栈；不同时调整 SH、scale 或环境。

这里的 PASS 指**诊断完成**，不表示原始 PLY 已兼容当前 SH0 配置。

#### Result

2026-09-24：PASS（加载诊断完成）。通过 Slurm 获配 GPU 的 job 25620 实际测试同一原始 PLY：GaussianModel(3) 成功读取 12,861 个 Gaussian；GaussianModel(0) 在 load_ply 第 288 行因 45 个 f_rest_* 与 SH0 要求的 0 个不匹配而触发 AssertionError。SH3 要求的数量为 45，与文件一致。

同一文件和真实 loader 在同一获配 GPU 的进程中 SH3 成功，排除了路径错误和本次 GPU 环境不可用。PASS 表示诊断完成，不表示原始文件兼容 SH0。未修改 PLY、loader 或环境，未执行 T2。

#### Evidence

- 命令：在服务器 SoMA 目录运行 `srun -p 5090 --gres=gpu:1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:02:00 --job-name=tcgs-t1-load <soma-env>/bin/python -B -`，诊断脚本经标准输入执行，未调用训练、dataset pipeline 或写文件方法。
- Job 25620，host amax，CUDA_VISIBLE_DEVICES=0，CUDA_AVAILABLE=True，RTX 5090，PyTorch 2.7.1+cu128；进程退出码 0（捕获并记录 SH0 预期异常）。
- 纠正此前 BLOCKED 的解释：此前未通过 Slurm 申请 GPU，直接 SSH 进程无法访问 GPU，不能据此断言服务器 GPU 故障。本次获配 GPU 后 SH3 成功，无需修改驱动或 CUDA 环境。
- 两端 SoMA branch deform360-adaptation，HEAD 8e8772a98f5eeb332745d9b6dd008e6927f050af；执行前服务器 clean，本地仅 roadmap 未跟踪。没有干预其他运行作业。
- `SoMA/mmgs/datasets/embodied_dataset.py` 第 515 行附近首选 GaussianModel(0)，异常后改用另一文件的 GaussianModel(3)；本诊断两种 degree 使用同一个指定 PLY，未依赖备用路径。
- PLY 和 loader 执行前后 SHA256 一致。以下保存实际 tensor shape 和完整异常栈；仅将 workspace 前缀移除、环境前缀替换为 <soma-env>。

```text
JOB 25620 HOST amax CUDA_VISIBLE_DEVICES 0
PYTHON <soma-env>/bin/python TORCH 2.7.1+cu128
LOADER SoMA/gaussian-splatting/scene/gaussian_model.py CUDA_AVAILABLE True
GPU NVIDIA GeForce RTX 5090
SHA256_BEFORE {"datasets/deform360/processed/008-pink-cloth/episode_0/splatfacto/splat_113.ply": "58f684ff02881821e7d2c91ae69983a5ec7bd88ad1949e41a4994505e81020c7", "SoMA/gaussian-splatting/scene/gaussian_model.py": "546ba41528a879af4d83bd280a3ef3cbcca9a5e9132204185aaa8f8ee9ffa310"}
VERTEX_COUNT 12861 REST_COUNT 45
DEGREE 3
SUCCESS
_xyz (12861, 3) torch.float32 cuda:0
_features_dc (12861, 1, 3) torch.float32 cuda:0
_features_rest (12861, 15, 3) torch.float32 cuda:0
_opacity (12861, 1) torch.float32 cuda:0
_scaling (12861, 3) torch.float32 cuda:0
_rotation (12861, 4) torch.float32 cuda:0
DEGREE 0
FAILURE
Traceback (most recent call last):
  File "<stdin>", line 22, in <module>
  File "SoMA/gaussian-splatting/scene/gaussian_model.py", line 288, in load_ply
    assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
AssertionError

_xyz (0,) torch.float32 cpu
_features_dc (0,) torch.float32 cpu
_features_rest (0,) torch.float32 cpu
_opacity (0,) torch.float32 cpu
_scaling (0,) torch.float32 cpu
_rotation (0,) torch.float32 cpu
SHA256_AFTER {"datasets/deform360/processed/008-pink-cloth/episode_0/splatfacto/splat_113.ply": "58f684ff02881821e7d2c91ae69983a5ec7bd88ad1949e41a4994505e81020c7", "SoMA/gaussian-splatting/scene/gaussian_model.py": "546ba41528a879af4d83bd280a3ef3cbcca9a5e9132204185aaa8f8ee9ffa310"}
UNCHANGED True
```

### T2：仅解决零高阶 SH 的文件兼容

Status: PASS

**问题：** 当前文件有 45 个全零 `f_rest_*`，而 dataset 聚类入口要求 SH0。

**为什么现在解决：** 这是已知最小文件层阻塞。

**输入文件：**

- `datasets/deform360/processed/008-pink-cloth/episode_0/splatfacto/splat_113.ply`
- T1 的加载诊断结果。

**预计涉及文件：**

- `SoMA/tools/deform360_adapter/`：拟新增的独立 Gaussian 导出工具，文件名尚未确定。
- `SoMA/gaussian-splatting/scene/gaussian_model.py`：SH0 loader 验证入口。
- `SoMA/mmgs/datasets/embodied_dataset.py`：dataset 加载验证入口。

**最小修改：** 生成独立 SH0 派生 PLY，仅移除已确认全零的 `f_rest_*`；保留原文件。

**验证方法：** 逐字段比较保留属性、点数和行顺序，再用 `SoMA/gaussian-splatting/scene/gaussian_model.py` 的 SH0 loader 读取。

**PASS：** 12,861 点及全部保留属性数值不变，SH0 加载成功。

**FAIL 后检查：** 是否存在非零高阶项、PLY writer 精度或 property 顺序问题；不修改 loader 来掩盖导出错误。

#### Result

2026-09-24：PASS。已生成独立 SH0 派生文件 `datasets/soma_d360_v0/008-pink-cloth/episode_0/t2_sh0/splat_113_sh0.ply`（875025 bytes，server-only），未覆盖任何已有文件。

仅移除经实际检查全部严格等于 0 的 45 个 f_rest_0…f_rest_44 属性。保留 12,861 点、点顺序，以及全部其他属性的名称、顺序、dtype 和逐字段逐行数值；重新读取派生文件后按字段字节比较全部一致。原始 PLY 和 Gaussian loader 保持不变。

在 Slurm job 25623 中，原始 PLY 的 GaussianModel(3) 和派生 PLY 的 GaussianModel(0) 均成功加载。不执行 T3。

#### Evidence

- 转换输入：`datasets/deform360/processed/008-pink-cloth/episode_0/splatfacto/splat_113.ply`。通过现有 soma 环境 Python/plyfile/NumPy 的一次性标准输入脚本执行；未新增或修改源码工具，未安装环境。
- 转换前断言：vertex=12861，f_rest 属性数=45，每一个值严格等于 0（非零或 NaN 均会中止）。按原 property 顺序复制全部非 f_rest 属性，保持 PLY text/byte_order/comments/obj_info；输出以独占创建模式写入，不覆盖已有文件。
- 保留属性：x/y/z、nx/ny/nz、f_dc_0…2、opacity、scale_0…2、rot_0…3。重新读取输出，逐字段核验 dtype 和原始行序下 tobytes 完全一致；没有排序、筛点、重新量化或改变任何保留值。
- 原始 SHA256：`58f684ff02881821e7d2c91ae69983a5ec7bd88ad1949e41a4994505e81020c7`；派生 SHA256：`fd79797c7b7da6ba5db02a4de5fe3c5e463cb7db1eabbe6a6a57bc283e51f7de`；loader SHA256：`546ba41528a879af4d83bd280a3ef3cbcca9a5e9132204185aaa8f8ee9ffa310`。原文件与 loader 转换前后 hash 一致；三者 GPU 验证前后 hash 也一致。
- CUDA 验证命令：在服务器 SoMA 目录执行 `srun -p 5090 --gres=gpu:1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:02:00 --job-name=tcgs-t2-load <soma-env>/bin/python -B -`。job 25623，进程退出码 0；真实 loader 无 mock/改写。没有启动 dataset pipeline、训练或 T3 数值解释检查。
- 两端 SoMA branch deform360-adaptation，HEAD 8e8772a98f5eeb332745d9b6dd008e6927f050af；仅本地 roadmap 更新，派生 PLY 保留服务器。此前 SSH 阻塞已解除。
- 完整加载输出（workspace 前缀省略）：

```text
JOB 25623 CUDA_AVAILABLE True GPU NVIDIA GeForce RTX 5090 TORCH 2.7.1+cu128
LOADER SoMA/gaussian-splatting/scene/gaussian_model.py
SUCCESS 3 datasets/deform360/processed/008-pink-cloth/episode_0/splatfacto/splat_113.ply
_xyz (12861, 3) torch.float32 cuda:0
_features_dc (12861, 1, 3) torch.float32 cuda:0
_features_rest (12861, 15, 3) torch.float32 cuda:0
_opacity (12861, 1) torch.float32 cuda:0
_scaling (12861, 3) torch.float32 cuda:0
_rotation (12861, 4) torch.float32 cuda:0
SUCCESS 0 datasets/soma_d360_v0/008-pink-cloth/episode_0/t2_sh0/splat_113_sh0.ply
_xyz (12861, 3) torch.float32 cuda:0
_features_dc (12861, 1, 3) torch.float32 cuda:0
_features_rest (12861, 0, 3) torch.float32 cuda:0
_opacity (12861, 1) torch.float32 cuda:0
_scaling (12861, 3) torch.float32 cuda:0
_rotation (12861, 4) torch.float32 cuda:0
UNCHANGED_SHA256 ['58f684ff02881821e7d2c91ae69983a5ec7bd88ad1949e41a4994505e81020c7', 'fd79797c7b7da6ba5db02a4de5fe3c5e463cb7db1eabbe6a6a57bc283e51f7de', '546ba41528a879af4d83bd280a3ef3cbcca9a5e9132204185aaa8f8ee9ffa310']
```

### T3：确认 scale、opacity、rotation 的解释保持一致

Status: PASS

**问题：** 文件能读取，不代表 Gaussian 的数值含义正确。

**为什么现在解决：** 避免重复 exp/sigmoid、错误 quaternion 顺序或几何缩放。

**输入文件：**

- 原始 PLY：`datasets/deform360/processed/008-pink-cloth/episode_0/splatfacto/splat_113.ply`
- T2 生成的 SH0 派生 PLY；实际输出路径以 T2 记录为准。

**预计涉及文件：**

- `SoMA/gaussian-splatting/scene/gaussian_model.py`：属性 activation、covariance 计算。
- `SoMA/tools/deform360_adapter/`：Gaussian 数值验证部分，文件名尚未确定。

**最小修改：** 仅新增这一组数值验证，不再转换属性。

**验证方法：** 比较 xyz、SH DC、activated scale/opacity、rotation 和 covariance。

**PASS：** xyz/点序一致；scale 为正、opacity 有限且在合法范围；两份文件产生一致的旋转和 covariance。

**FAIL 后检查：** log/raw 值混淆、wxyz/xyzw、零 quaternion、单位或序列化精度。

#### Result

2026-09-24：PASS。原始 SH3 与 T2 派生 SH0 在真实 SoMA GaussianModel 加载后，12,861 点的 xyz/点序、SH DC、raw/activated scale、raw/activated opacity、raw/normalized quaternion、rotation matrix 和 covariance 全部逐元素完全相等（max absolute difference=0）。

scale 经 exp 后均为正且有限；opacity 经 sigmoid 后有限且在 [0,1] 内；无零 quaternion，SoMA 按 wxyz 解释并归一化；默认 scaling_modifier=1 的 covariance 一致。原始和派生文件、loader 均未改动；未执行 T4。

#### Evidence

- 输入：`datasets/deform360/processed/008-pink-cloth/episode_0/splatfacto/splat_113.ply` 与 `datasets/soma_d360_v0/008-pink-cloth/episode_0/t2_sh0/splat_113_sh0.ply`。
- 运行命令：服务器 SoMA 目录，`srun -p 5090 --gres=gpu:1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:02:00 --job-name=tcgs-t3-values <soma-env>/bin/python -B -`；一次性标准输入脚本，torch.no_grad()，未新增源码工具。Job 25624，RTX 5090，退出码 0。
- 源码依据：`SoMA/gaussian-splatting/scene/gaussian_model.py` setup_functions 第 32 行（exp、sigmoid、normalize），load_ply 第 263 行（原始字段直接载入），get_covariance 第 142 行；`SoMA/gaussian-splatting/utils/general_utils.py` build_rotation 第 78 行（wxyz）与 build_scaling_rotation 第 101 行（R @ diag(scale)）。
- 两份 PLY 分别用 GaussianModel(3)/GaussianModel(0) 加载；逐行核验模型 raw xyz/DC/scale/opacity/rotation 与各自 PLY 字段完全相等，排除重排和加载时重复激活。再核验 get_scaling=exp(raw_scale)、get_opacity=sigmoid(raw_opacity)，两模型输出逐元素比较使用 torch.equal。
- activated scale 范围约 [9.80978075e-05, 0.0116145378]；opacity 范围 [0.0123573467,1.0]（float32 sigmoid 可饱和到 1）；raw quaternion norm 范围 [0.9999999404,1.0]，归一化后 norm 最大误差 1.1920929e-07。
- 默认 covariance 为 R diag(exp(raw_scale)^2) R^T 的压缩表示；两模型的 [12861,6] covariance 完全相等。使用另一乘法顺序核验该公式，最大绝对误差 1.4551915e-11，满足 rtol=1e-5、atol=1e-10。这里仅确认 T2 转换前后在 SoMA 中的解释一致，不扩展为外部坐标系/物理单位的验证。
- 两端 SoMA branch deform360-adaptation，HEAD 8e8772a98f5eeb332745d9b6dd008e6927f050af。以下输出包含输入与实现文件执行前后不变的 SHA256。仅更新本地 T3 记录与 Progress，未训练、未 commit/push。

```text
JOB 25624 CUDA True GPU NVIDIA GeForce RTX 5090
SOURCE_LINES {'setup_functions': 32, 'load_ply': 263, 'get_covariance': 142, 'build_rotation': 78, 'build_scaling_rotation': 101}
DEGREE 3 SCALE_RANGE 9.809780749492347e-05 0.011614537797868252 OPACITY_RANGE 0.012357346713542938 1.0 RAW_Q_NORM_RANGE 0.9999999403953552 1.0 NORMALIZED_Q_NORM_MAX_ERROR 1.1920928955078125e-07
DEGREE 0 SCALE_RANGE 9.809780749492347e-05 0.011614537797868252 OPACITY_RANGE 0.012357346713542938 1.0 RAW_Q_NORM_RANGE 0.9999999403953552 1.0 NORMALIZED_Q_NORM_MAX_ERROR 1.1920928955078125e-07
EQUAL xyz shape [12861, 3] max_abs_diff 0.0
EQUAL SH_DC shape [12861, 1, 3] max_abs_diff 0.0
EQUAL raw_scale shape [12861, 3] max_abs_diff 0.0
EQUAL activated_scale shape [12861, 3] max_abs_diff 0.0
EQUAL raw_opacity shape [12861, 1] max_abs_diff 0.0
EQUAL activated_opacity shape [12861, 1] max_abs_diff 0.0
EQUAL raw_quaternion_wxyz shape [12861, 4] max_abs_diff 0.0
EQUAL normalized_quaternion shape [12861, 4] max_abs_diff 0.0
EQUAL rotation_matrix shape [12861, 3, 3] max_abs_diff 0.0
EQUAL covariance_packed_modifier_1 shape [12861, 6] max_abs_diff 0.0
COVARIANCE_FORMULA_CHECK_MAX_ABS 1.4551915228366852e-11
COVARIANCE_FORMULA_CHECK_MAX_ABS 1.4551915228366852e-11
UNCHANGED_SHA256 {"datasets/deform360/processed/008-pink-cloth/episode_0/splatfacto/splat_113.ply": "58f684ff02881821e7d2c91ae69983a5ec7bd88ad1949e41a4994505e81020c7", "datasets/soma_d360_v0/008-pink-cloth/episode_0/t2_sh0/splat_113_sh0.ply": "fd79797c7b7da6ba5db02a4de5fe3c5e463cb7db1eabbe6a6a57bc283e51f7de", "SoMA/gaussian-splatting/scene/gaussian_model.py": "546ba41528a879af4d83bd280a3ef3cbcca9a5e9132204185aaa8f8ee9ffa310", "SoMA/gaussian-splatting/utils/general_utils.py": "97553507caa4f3e6849919d9658b0b0da064e6c8f19142045da79072171ffa51"}
PASS
```

### T4：固定原始 frame 与本地 frame 的一一映射

Status: PASS

**问题：** 原始 113 应如何成为 SoMA scene 的 frame 0？

**为什么现在解决：** 所有导出组件必须共享同一编号。

**输入文件：**

- T0 结论。
- `datasets/deform360/processed/008-pink-cloth/episode_0/metadata.json`
- `datasets/deform360/processed/008-pink-cloth/episode_0/<camera_id>/aligned_timestamps.txt`

**预计涉及文件：**

- `SoMA/tools/deform360_adapter/`：拟新增的独立 frame manifest 工具，文件名尚未确定。
- `SoMA/mmgs/datasets/embodied_dataset.py`：`__getitem__`。

**最小修改：** 建立 `local=source−113` 的映射，不导出图像或 controller。

**验证方法：** 检查正反映射和 timestamps；明确保留窗口内每一帧，不按 tactile active 再删帧。

**PASS：** `113↔0`、`306↔193`，194 条连续、无重复、可逆。

**FAIL 后检查：** inclusive/exclusive 边界、额外减偏移、按 contact-active 压缩帧的问题。

#### Result

2026-09-24：PASS。建立并持久化 source_frame ↔ local_frame 映射：local=source−113，source=local+113；113↔0，306↔193，共 194 条连续、无重复、可逆记录。

产物：`datasets/soma_d360_v0/008-pink-cloth/episode_0/t4_frame_mapping/frame_manifest.json`（服务器，528101 bytes）。每条记录包含 source/local frame 及 36 个 camera 的原始 timestamp token。保留窗口内每一帧；没有读取 tactile 数值、按 active 筛帧或压缩时间轴。未执行 T5。

#### Evidence

- 以实际 metadata 的 start_frame=113、end_frame=306、frame_num=194 和 T0 已确认的官方 processed 起点为依据；未解析或修改 train/test 划分。
- 输入为 `datasets/deform360/processed/008-pink-cloth/episode_0/metadata.json` 与该目录中 36 个 camera 的 aligned_timestamps.txt；每个文件 357 行。source_frame 明确定义为从 0 起算的行号。
- 原始行格式为 frame_<integer_timestamp>_<frame_id>。逐 camera 核验选中窗口中 token 和整数 timestamp 均无重复、严格递增，内嵌 frame_id 与 source 行号一致（36/36 cameras）。不假定各相机 timestamp 完全相同，不插值、不重新计时，也不推断未核验的时间单位。
- 建立 194 条连续 source=113…306/local=0…193；对每个 camera 每条记录验证 (camera_id,timestamp token)→(source,local) 的反向查找；原始 timestamp 字符串精确保留。manifest 序列化后重新读取并与内存对象全量比较一致。
- `brics-odroid-001_cam0` 示例：local 0/source 113 = frame_1766008308603525_000000000113；local 193/source 306 = frame_1766008315035710_000000000306。
- manifest 保存全部 36 个 timestamp 输入文件 SHA256，运行前后均一致；输出 SHA256 为 d48338cf7a606b1ee4bc2e5a560b5a53c017efdc9880a1db282a8494b1b013db。独占创建输出，没有覆盖已有文件。
- 通过既有 soma 环境 Python -B 的一次性标准输入脚本完成纯 CPU manifest 操作；未初始化 CUDA、未修改 Gaussian/controller/camera 或任何源码，未导出图像。没有新增独立工具源码。
- 只读查看 `SoMA/mmgs/datasets/embodied_dataset.py` __getitem__ 第 798 行及 video_range 第 826 行；本轮不修改 loader、split 或 frame_gap，不进行 T5。
- 两端 SoMA branch deform360-adaptation，HEAD 8e8772a98f5eeb332745d9b6dd008e6927f050af。本轮仅新增服务器 manifest 并更新本地 T4 记录/Progress；不 commit/push。

```json
{
  "output": "datasets/soma_d360_v0/008-pink-cloth/episode_0/t4_frame_mapping/frame_manifest.json",
  "bytes": 528101,
  "sha256": "d48338cf7a606b1ee4bc2e5a560b5a53c017efdc9880a1db282a8494b1b013db",
  "count": 194,
  "cameras": 36,
  "endpoints": [
    [
      113,
      0
    ],
    [
      306,
      193
    ]
  ],
  "all_cameras_timestamps_unique_strictly_increasing_reversible": true,
  "source_files_unchanged": true,
  "embedded_id_matches_row_camera_count": 36,
  "first_camera_example": {
    "brics-odroid-001_cam0": {
      "first": "frame_1766008308603525_000000000113",
      "last": "frame_1766008315035710_000000000306",
      "embedded_id_equals_source_row": true
    }
  }
}
```

### T5：固定训练／测试边界

Status: PASS

**问题：** 官方 split 如何映射，如何避免测试监督进入训练？

**为什么现在解决：** 后续采样、cache 和 Stage 2 子窗口都受其约束。

**输入文件：**

- `datasets/deform360/processed/008-pink-cloth/episode_0/split.json`
- T4 生成的 frame manifest；实际输出路径以 T4 记录为准。

**预计涉及文件：**

- `SoMA/configs/SoMA/deform360_v0_stage1.py`（拟新增）：split 定义。
- `SoMA/configs/SoMA/deform360_v0_stage2.py`（拟新增）：split 定义。
- 本项先形成 split contract。

**最小修改：** 只定义边界：训练本地 `[0,155)`，测试 `[155,194)`。

**验证方法：** 枚举 frame_gap=10 的训练 target，以及 gap=1 的目标集合；明确评估 warm-up 与评分窗口。

**PASS：** 训练监督不含 source frame ≥268；测试评分仅对应 268–306；不把 268 的重建 Gaussian 当测试初态。

**FAIL 后检查：** Python range 端点、最后一个子窗口、validation/test 配置混用。

#### Result

2026-09-24：PASS（split contract 与索引集合验证）。产物：`datasets/soma_d360_v0/008-pink-cloth/episode_0/t5_split_contract/split_contract.json`，保存在服务器。

官方 source train=[113,268)、test=[268,307)，分别对应 local train=[0,155)、test=[155,194)。训练 155 帧、测试 39 帧，互不重叠且覆盖 T4 的全部 194 帧。合同规定任何训练监督不得使用 source≥268；测试评分仅为 source 268–306。

评估从 source 113/local 0 的 T2 initial Gaussian 连续推进；local [0,155) 为不计分 warm-up，local [155,194) 为评分窗口。不得把 source 268 的重建 Gaussian 用作测试初态，也不得在测试边界重置或注入重建/GT Gaussian。

本 PASS 仅证明 contract 和枚举集合满足约束；未集成 loader/config，因此不声称已验证未来训练运行时的数据隔离。未执行 T6。

#### Evidence

- 读取官方 `datasets/deform360/processed/008-pink-cloth/episode_0/split.json`：frame_len=194、train=[113,268]、test=[268,307]，按左闭右开解释。
- 读取并逐条核验 T4 的 `datasets/soma_d360_v0/008-pink-cloth/episode_0/t4_frame_mapping/frame_manifest.json`：194 条，source=local+113；训练/测试集合交集为空，合集为 local 0…193，source 测试端点为 268、306。
- frame_gap=10，从 local 0 起采样的训练序列为 0,10,…,150；0 为初态，预测 target 为 local 10,20,…,150，对应 source 123,133,…,263（15 个），没有 source≥268。
- gap=1，从 local 0 起采样的预测 target 为 local 1…154，对应 source 114…267（154 个）。local 0 属于训练允许范围，但在上述 rollout 枚举中是初态而非未来预测 target。
- contract 保存完整 sampled frame/target 集合及边界断言；后续任意训练子窗口、Stage-1 cache 与 Stage-2 训练窗口均必须遵守训练边界，跨 local 155 的末尾窗口必须拒绝，test 评分不能混入训练 loss/gradient。这些是后续集成要求，不是本轮已运行的训练验证。
- 输出独占创建并重新读回全量核验；输入 split/manifest 的 SHA256 已写入 contract，执行前后均不变；输出 SHA256：fd1370b2202836c2c8be2589955bca33bbe88e38a067fb00bb937e48500debd1。
- 纯 CPU Python -B 一次性标准输入脚本，没有 CUDA 初始化、没有读取 source 268 的 Gaussian、没有修改 dataset loader/config 或新增工具源码。
- 两端 SoMA branch deform360-adaptation，HEAD 8e8772a98f5eeb332745d9b6dd008e6927f050af；本轮仅创建服务器 contract 并更新本地 T5 记录/Progress，未 commit/push。

```json
{
  "output": "datasets/soma_d360_v0/008-pink-cloth/episode_0/t5_split_contract/split_contract.json",
  "sha256": "fd1370b2202836c2c8be2589955bca33bbe88e38a067fb00bb937e48500debd1",
  "counts": {
    "train": 155,
    "test": 39
  },
  "local_split": {
    "train": [
      0,
      155
    ],
    "test": [
      155,
      194
    ]
  },
  "gap10_prediction_targets_local": [
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
    130,
    140,
    150
  ],
  "gap10_prediction_targets_source": [
    123,
    133,
    143,
    153,
    163,
    173,
    183,
    193,
    203,
    213,
    223,
    233,
    243,
    253,
    263
  ],
  "gap1_prediction_target_count": 154,
  "gap1_prediction_source_endpoints": [
    114,
    267
  ],
  "test_source_endpoints": [
    268,
    306
  ],
  "overlap": false,
  "inputs_unchanged": true,
  "PASS": true
}
```

### T6：明确采样间隔与模型内部时间尺度

Status: PASS

**问题：** 数据间隔、模型 `dt`、dataset `real_dt/env_cfg.dt` 不是同一个量。

**为什么现在解决：** 官方配置包含 `/5` 和重力时间缩放，不能直接套 FPS。

**输入文件：**

- T4 确认的时间映射和 `datasets/deform360/processed/008-pink-cloth/episode_0/<camera_id>/aligned_timestamps.txt`。
- 官方 Stage 1 时间配置：`SoMA/configs/SoMA/cloth_lift_stage1.py`
- 官方 Stage 2 时间配置：`SoMA/configs/SoMA/cloth_lift_stage2.py`

**预计涉及文件：**

- `SoMA/configs/SoMA/deform360_v0_stage1.py`（拟新增）：时间参数。
- `SoMA/configs/SoMA/deform360_v0_stage2.py`（拟新增）：时间参数。
- `SoMA/mmgs/datasets/embodied_dataset.py`：重力缩放。
- `SoMA/mmgs/models/heads/acc_decoder.py`：decoder 的 `dt` 使用。

**最小修改：** 单独形成时间参数约定；必要时只设置对应配置项。

**验证方法：** 列出一个 Stage 1 步和一个 Stage 2 步对应的 source frame、观测时间间隔及模型内部参数。

**PASS：** 各参数用途明确、无重复缩放；Stage 2 gap=1 没有被误解成 Stage 1 gap=10。

**FAIL 后检查：** 配置继承、decoder/backbone 时间参数和 gravity scaling；不同时调整重力方向。

#### Result

2026-09-25：PASS（只读时间参数审计与 v0 约定；未改配置、未训练）。数据采样间隔与模型内部时间不是同一个量。为保持官方 backbone 行为，本阶段约定先保留官方有效内部时间参数，不按 Deform360 FPS 或 Stage 2 dense gap 自动重新缩放模型 dt；此为 baseline 的内部数值约定，不将其宣称为真实物理秒。

| 参数 | Stage 1 | Stage 2 |
|---|---|---|
| dataset frame_gap | 10 | 1 |
| source 示例 | 113→123→133 | 113→114→115；任意子窗口 i 对应 source 113+i |
| 标称观测间隔 | 10/30 秒 | 1/30 秒 |
| 实测示例（001_cam0） | 113→123 = 0.336055 秒 | 113→114 = 0.036402 秒 |
| model.dt / backbone.dt / decoder.dt | 1/15 | 1/15（官方现状；不是 dense 观测秒数） |
| dataset env_cfg.dt（comp_dt） | 1/30 | 1/30 |
| dataset real_dt[scene] | 1/15 | 1/15 |
| dataset gravity 时间倍率 | (real_dt/comp_dt)^2 = 4 | 同样为 4 |
| Stage 2 model.frame_gap | 不适用 | 10（coarse cache/update 间隔，不是 dataset dense gap） |

不额外把 model.dt 乘/除 10 或 5，不把实测 timestamps 直接填入 real_dt。输入 gravity 保持未做时间预缩放的基准值，dataset 仅一次乘 4，再执行已有坐标旋转；不在 adapter/config 预乘 4，也不再次按 gap/FPS 缩放。若基准模长为 9.8，则时间缩放后的模长为 39.2；本项不确定或修改重力方向。

Stage 1 的训练采样延续 T5：local 0,10,…,150 ↔ source 113,123,…,263；Stage 2 dense 训练范围 local 0…154 ↔ source 113…267。后续 gap、dt 或 gravity 若要改为物理时间一致的另一套约定，必须独立验证，不作为本 T6 的隐式修正。

#### Evidence

- 官方配置：`SoMA/configs/SoMA/cloth_lift_stage1.py` 第 21–23 行与 `SoMA/configs/SoMA/cloth_lift_stage2.py` 第 21–23 行均定义 frame_gap=10、frame_gap_stage2=1、dt=(1/30)*10/5=1/15；两文件第 104/105 行将 real_dt[scene] 设为该 dt。Stage 1 dataset 第 198 行使用 gap=10；Stage 2 第 205 行使用 gap=1，但 model 第 126、132 行仍为 frame_gap=10、dt=1/15。
- 配置继承：两阶段均继承 `SoMA/configs/_base_/datasets/gs_soma_dataloader.py`，第 27 行 env_cfg.dt=1/30；场景配置只覆盖 real_dt，没有覆盖此 env_cfg.dt。不可把顶层 Python dt 变量误认为自动修改了 dataset comp_dt。
- `SoMA/mmgs/datasets/embodied_dataset.py` 第 280–294 行读取 gravity、real_dt、comp_dt、frame_gap；第 826 行以 range(start,end,frame_gap) 选帧，第 918 行按同一 video_range 选 controller。第 921–934 行只将 gravity 乘 (real_dt/comp_dt)^2，然后执行 scene rotation；没有依据 timestamps 自动校正步长。
- `SoMA/mmgs/models/simulators/gs_simulator_embodied.py` 第 84–87 行与 `SoMA/mmgs/models/simulators/gs_simulator_embodied_stage2.py` 第 87–89 行将 model.dt 注入 backbone/decode_head。Stage 2 第 798–800、944–945 行的 model.frame_gap 用于 coarse 更新/cache 对齐，不能当作 dense 视频步长。
- `SoMA/mmgs/models/backbones/meshgraphnet_embodied.py` 第 330、374、382 行分别用 dt/dt² 计算相对速度、anchor acceleration 和速度；external gravity 与 anchor acceleration 结合，没有再次按采样 gap 做时间换算。
- `SoMA/mmgs/models/heads/acc_decoder.py` 第 252 行 kinetic 使用 ((pred_pos−2*cur_state+prev_state)/dt)^2；第 253–254、264–270 行直接使用 external 与位置形成重力项，没有第二次 gravity 时间倍率。dt 会影响内部特征/能量，不能仅因 dense gap=1 就无验证地改它。
- 真实时间证据来自 Git 已归档的 `SoMA/docs/deform360/contracts/008-pink-cloth/episode_0/frame_manifest.json`，SHA256=d48338cf7a606b1ee4bc2e5a560b5a53c017efdc9880a1db282a8494b1b013db，与 T4 原始产物一致。保留 36 个 camera 的原始 aligned_timestamps token 及输入 hash；核验 source=local+113。`deform360/deform360/timestamps.py` 第 10、31 行规定 Unix microseconds，因此差值除以 1e6 得秒。
- 36 cameras 的 dense 差分共 6948 个：min=0.030940、median=0.032047、max=0.037014、mean=0.0333273834 秒。按 local 0,10,…,190 的 coarse 差分共 684 个：min=0.330925、median=0.332057、max=0.337027、mean=0.3332729474 秒。统计整段只是核对时间间隔，不改变 T5 训练/测试边界；这些并非额外训练监督。
- 证据限制：本次 SSH 到 amax 超时（exit 255），未重新读取服务器 aligned_timestamps；以上数值来自 T4 已实际读取并归档的原始 timestamp token。本地 SoMA HEAD=12150096ee1d3759adb0a0d724392e54bf4888d6，branch=deform360-adaptation，执行前 clean。未运行 CUDA、未修改配置或训练代码。
- 本轮只修改本 roadmap（已 tracked），T6 Status/Result/Evidence 与 Progress 更新；尚未 commit/push，后续需通过 Git 同步服务器。未执行 T7。

---

## 第二阶段：分别实现空间与数据组件

### T7：选定两个固定 camera ID

Status: PASS

**问题：** 哪两个训练视角适合作为 v0 的固定监督？

**为什么现在解决：** calibration、RGB、mask 必须使用同一份相机名单。

**输入文件：**

- 现有 inventory：`datasets/deform360/inventory_episode_0/`
- 训练区间少量 RGB：`datasets/deform360/processed/008-pink-cloth/episode_0/<camera_id>/undistorted.mp4`
- 训练区间对应 mask：`datasets/deform360/processed/008-pink-cloth/episode_0/<camera_id>/mask_refined.h5`
- `datasets/deform360/processed/008-pink-cloth/episode_0/metadata.json`
- `datasets/deform360/processed/008-pink-cloth/episode_0/metric_params_refined_undistorted.txt`

**预计涉及文件：**

- `SoMA/tools/deform360_adapter/`：camera manifest，具体文件名尚未确定；暂不改 loader。

**最小修改：** 固定两个 ID 和顺序，排除 `016_cam0`。

**验证方法：** 检查训练区间 mask 有效性、代表帧遮挡及视角互补性；不利用测试表现挑相机。

**PASS：** 两个不同且标定齐全的相机，训练窗口 mask 非空，选择依据和顺序可复核。

**FAIL 后检查：** 空 mask、遮挡、裁边或视角重复；只重新选相机。

#### Result

2026-09-25：PASS。用户已完成人工确认，正式采用以下两组配置并保留其顺序：

- Config A / v0 minimal baseline（默认最小配置）：[brics-odroid-023_cam0, brics-odroid-009_cam1]。
- Config B / v0 auxiliary 3-camera comparison（扩展对比配置）：[brics-odroid-023_cam0, brics-odroid-009_cam1, brics-odroid-014_cam1]。

后续 T8 及之后的 camera-related steps 必须支持这两组配置及各自相机顺序，不能只按 Config A 硬编码。本次仅确认选择，不执行 T8。

三份文件角色：camera_manifest.json 为总体候选评估与人工确认记录；camera_config_2cam.json 为默认最小配置；camera_config_3cam.json 为扩展对比配置。三者均保存在 `SoMA/docs/deform360/contracts/008-pink-cloth/episode_0/`，已标记 human_confirmed=true；保留原验证数据与 provenance。

此前候选阶段记录（历史状态，不代表当前结论）：

2026-09-25：IN_PROGRESS — waiting for human confirmation。此前自动选择不再视为最终 PASS。当前候选组合仍为 [brics-odroid-023_cam0, brics-odroid-009_cam1]，尚未锁定；需由用户结合三维布局与代表帧确认后，才能正式固定 camera manifest。

补充可视化位于 `SoMA/docs/deform360/visualizations/t7-review-20260925/`：interactive-layout.html（可拖拽旋转、缩放、点选 ID、切换 XY/XZ 的交互图）、soma_layout.png、deform360_layout.png（各含 3D/XY/XZ），soma_representative.jpg、deform360_candidates.jpg，以及 layout.json。图中 XY/XZ 是原生坐标投影，不是已验证的重力对齐视图；两数据集世界坐标独立。紫色星标仅是光轴最小二乘汇聚点估计，不是实测物体中心。未执行 T8。

新增候选配置比较（2026-09-25）：Config A 为 [023_cam0,009_cam1] minimal baseline；Config B 为 [023_cam0,009_cam1,014_cam1]。两份 JSON 分别位于 `SoMA/docs/deform360/contracts/008-pink-cloth/episode_0/camera_config_2cam.json` 与 `camera_config_3cam.json`，全部 human_confirmed=false。原 camera_manifest.json 内容保持不变，T7 保持 IN_PROGRESS。

#### Evidence

- 最终确认依据：用户于 2026-09-25 明确确认 Config A 和 Config B 的全部 camera ID、顺序及用途。本轮没有重新筛选相机或使用测试表现；下列 IN_PROGRESS/candidate/未确认描述属于此前阶段记录，当前结论以本项最新 Result 与三份 JSON 的确认字段为准。
- 四个本轮修改文件均属于 SoMA/deform360-adaptation：roadmap（tracked）及三份 camera JSON（目前 untracked）。未修改源码、未执行 T8、未 commit/push；建议将 T7 相关 contracts 与必要 review 资产一起提交，然后通过 Git 同步另一端。

- 三相机候选选择：重新扫描除原两候选及排除项外的相机训练 mask 113–267，014_cam1 的 155 帧均非空、0 帧触边，foreground min/median/max=91269/113535/131148。它与 023_cam0/009_cam1 光轴夹角为 79.44155°/77.55762°；结合三维布局及 source 113/200/267 的 RGB 提供另一侧形变轮廓，不以测试表现选择。
- 014_cam1 calibration camera_id=20、metadata index=17，完整 19 列且有限，K 与 metadata 一致；原标定行已保存于 3cam 配置。代表帧仍有夹爪局部遮挡，候选不等于最终确认。
- 快照：`SoMA/docs/deform360/visualizations/t7-config-comparison-20260925/config_a_2cam.jpg`（2 行相机 × 3 列 source 113/200/267）；同目录 `config_b_3cam.jpg`（3 行 × 3 列，末行为014_cam1）。仅生成两张缩略拼图，未导出完整 sequence。
- 本补充所属 SoMA/deform360-adaptation：新增两份 candidate JSON 和两张 JPEG（尚未 tracked），修改已 tracked roadmap；未 commit/push，待通过 Git 同步另一端。原 camera_manifest.json 未改动，未修改 loader，未执行 T8。

- 本轮可视化：SoMA calibrate.pkl 按现有 readEmbodiedCameras/extract_extrinsics 的 c2w 约定；Deform360 metric qvec 按 wxyz world-to-camera 绘图，C=-R.T@t，forward=R.T[:,2]。仅为诊断绘图，没有导出 T8 calibration。layout.json 保存全部 ID/位置/光轴与 SoMA serial 对应。
- SoMA sample 展示全部 3 个相机、帧 0/60/140；Deform360 展示 metadata 中全部 36 个相机，016_cam0 红色排除，023_cam0 橙色、009_cam1 青色候选；代表帧仅 source 113/200/267（训练窗口），共 6 张缩略图。未导出大规模 RGB/mask。
- camera_manifest.json 已标记 selection_status=IN_PROGRESS、human_confirmed=false、candidate_camera_ids；保留之前诊断信息但不作为最终锁定名单。下列为此前检查证据，不能替代人类确认。
- SSH 已恢复。读取 `datasets/deform360/inventory_episode_0/cameras.inventory.json` 与 robot_tactile_calibration.inventory.json，候选筛选仅使用 mask 计数的 [113:268]；未用测试指标或测试 RGB 挑选。
- 实际读取候选 undistorted.mp4 的 source 113、200、267，内存解码检查 RGB 与 mask 边界叠加，不写出 RGB/mask 图像文件。两者夹爪接触附近仍有局部遮挡，未宣称全帧无遮挡。
- 对选定两相机逐帧重新读取 mask_refined.h5 的训练窗口：023_cam0 非零像素 min/median/max=158227/169436/178730；009_cam1=71141/78212/123345。均 155/155 非空，0/155 触边；数据为 [357,720,1280] uint8。视频均 1280×720、357 frames，三个代表帧解码成功。
- 023_cam0 metadata index=28、calibration camera_id=36；009_cam1 metadata index=9、calibration camera_id=11。metric_params_refined_undistorted.txt 中都有完整 19 列、有限 fx/fy/cx/cy/distortion/qvec/tvec，内参与 metadata 3×3 K 一致；原始标定行保存于 camera manifest。
- 以标定 qvec 的相机光轴计算视角分离约 117.26045°，结合训练 RGB 确认互补；没有生成或修改 T8 相机外参。015_cam0 虽有侧面形变信息，但 155 帧均触底边，已拒绝；001_cam0、017_cam0 也触边；008_cam1、013_cam0、027_cam1 等代表帧可见裁边，不以面积最大作为唯一依据。
- inventory 已确认 016_cam0 的 mask 全零；本次排除。metadata/calibration SHA256 与选定原始标定行记录于 camera_manifest.json，可复核。
- 本地 SoMA HEAD=12150096ee1d3759adb0a0d724392e54bf4888d6，服务器 SoMA HEAD=8e8772a98f5eeb332745d9b6dd008e6927f050af，服务器源码仓库 clean；未同步或覆盖。本地已有 roadmap 未提交修改全部保留。
- 所属 repository：SoMA；branch：deform360-adaptation。修改已 tracked roadmap；新增 camera_manifest.json 尚未 tracked，待纳入 Git。未 commit/push；文档和 manifest 后续需通过 Git 同步服务器。未修改 loader、未导出 RGB/mask、未训练、未执行 T8。

### T8：只解决相机外参转换

Status: PASS

**问题：** 实际发布的是 qvec/tvec 文本，而非现成 c2w `.npy`。

**为什么现在解决：** SoMA embodied reader 明确接收 c2w。

**输入文件：**

- `datasets/deform360/processed/008-pink-cloth/episode_0/metric_params_refined_undistorted.txt`
- T7 固定的相机名单。

**预计涉及文件：**

- `SoMA/tools/deform360_adapter/`：calibration 工具，具体文件名尚未确定。
- `SoMA/mmgs/datasets/embodied_dataset.py`：`extract_extrinsics`。

**最小修改：** 按 camera name 解析外参并形成选定相机的 c2w；不改内参或图像。

**验证方法：** 确认发布格式的 quaternion/translation convention，检查逆矩阵、旋转正交性和投影方向。

**PASS：** 两个 `[4,4]` c2w 与相机身份一致，变换往返正确，投影方向有证据支持。

**FAIL 后检查：** w2c/c2w、quaternion 顺序、矩阵转置、camera 行匹配；不靠试翻轴选择“看起来对”的结果。

#### Result

2026-09-25：PASS。仅完成外参转换 contract 和 CPU 几何验证；未修改源码、loader、原始 calibration、相机选择或配置，未导出 RGB/mask，未执行训练或 T9。

- 发布文本 header 明确四元数为 `qvecw qvecx qvecy qvecz`，采用 Hamilton `w,x,y,z` 的 world-to-camera 旋转 `R`；`tvec` 是 `X_camera = R @ X_world + t` 中的平移，不是世界坐标下的相机中心。
- 对发布四元数做单位化以消除文本舍入误差（最大模长误差 `5.052e-13`），形成 `w2c = [[R,t],[0,0,0,1]]`，解析求逆得到 `c2w = [[R.T,-R.T@t],[0,0,0,1]]`。保持原始世界坐标和长度尺度，不翻轴、不重新定位。
- 新增权威小型 artifact：[camera_extrinsics_contract.json](contracts/008-pink-cloth/episode_0/camera_extrinsics_contract.json)。其中 `configurations.A/B` 分别提供按原顺序排列的 `[2,4,4]` / `[3,4,4]` c2w，`cameras` 保存三个唯一相机的原始行、qvec/tvec、w2c/c2w 和逐相机验证结果，可作为 T9 输入。
- Config A（默认 minimal baseline）：`023_cam0 → 009_cam1`；Config B（auxiliary comparison）：`023_cam0 → 009_cam1 → 014_cam1`，均保留完整 `brics-odroid-` 前缀。camera manifest 和两个 camera config 的字节及 SHA256 均未改变。
- 三个 c2w 均为有限的 `[4,4]` 矩阵，末行 `[0,0,0,1]`；旋转正交性与双向互逆最大误差均为 `1.111e-15`，det(R) 均约为 `+1`。SoMA 纯 NumPy 外参函数往返通过，包含 float32 转换后的 w2c 最大误差 `2.746e-8`。
- 朝向与 T7 布局一致；真实 `splat_113.ply` 的 12,861 个 Gaussian 中心在每个相机下均为正深度且落在原始 1280×720 图像范围内。没有通过试翻轴选择结果。

范围说明：当前 Deform360 `calibration.py` 说明的是另一种 `.npy` c2w 发布格式，不能直接作为本 legacy 文本的 exporter 证明；未找到该文本的 exporter。本次 w2c 判断由发布 header、COLMAP 标准约定及真实点云/图像几何对应共同支持。T7 layout 使用相同约定，因此布局一致性只是回归检查，不单独当作独立证明。

#### Evidence

- 输入：服务器 `datasets/deform360/processed/008-pink-cloth/episode_0/` 下真实 metadata、metric calibration、`splatfacto/splat_113.ply`，以及三个 camera 的 `mask_refined.h5:data[113]`。只读取三个 mask 单帧用于投影 sanity check，没有导出图像。41 行 calibration 与 36 个 metadata camera 按精确名称关联，未按行号直接配对。
- 文本行号 / metadata index：`023_cam0 = 31 / 28`、`009_cam1 = 12 / 9`、`014_cam1 = 21 / 17`；具体原始 token 和实际行号以 contract 的 `cameras` 记录为准。
- 约定参考：[COLMAP images.txt format](https://colmap.github.io/format.html#images-txt)：Hamilton `(QW,QX,QY,QZ)` 表示 world-to-camera，camera center 为 `-R.T @ t`，相机轴为右、下、前。
- SoMA 源码：`mmgs/utils/colmap_utils.py:43` 的 `qvec2rotmat`；`mmgs/datasets/embodied_dataset.py:184` 的 `extract_extrinsics`；`mmgs/datasets/utils/cameras.py:137` 的 `getWorld2View2`。通过 AST 提取并在服务器 CPU 执行这些未修改的纯 NumPy 函数，与 SciPy 四元数转换交叉核验；没有调用 CUDA 或完整 GPU Camera loader。
- 三个相机的正深度及图像内中心数均为 `12861/12861`；中心投影落入 frame 113 foreground mask 的数量依次为 `12303 / 10192 / 12064`（约 `95.66% / 79.25% / 93.80%`）。这是中心投影 sanity check，不是渲染质量、遮挡或 test performance 指标，也未用于重新选择 camera。
- 光轴与指向 Gaussian centroid 方向的点积依次为 `0.93710 / 0.98202 / 0.97123`；与 T7 layout 的 camera position 最大差 `2.221e-16`，forward 最大差 `3.331e-16`。
- contract 记录输入 SHA256、各相机原始 calibration tokens、原始 mask 单帧数组 hash、源码函数文件 hash、数值容差和完整验证结果。metadata、metric calibration、原始 PLY 的检查前后 SHA256 一致；所有输入均未写入。
- Artifact：`camera_extrinsics_contract.json`，22,231 bytes，SHA256 `eb9fc54cb1a029eeb1e502f6c4965dda2edf856d29e3c5ca8b86db4e196626dd`。
- Git：所属 repository 为 SoMA，branch 为 `deform360-adaptation`。roadmap 为已 tracked 文件；新增 contract 当前未跟踪，等待明确 commit 指令纳入 Git。本轮未 add/commit/push，未同步服务器；后续应通过 Git 同步，不将服务器临时路径作为权威引用。

### T9：只解决目标分辨率对应的内参

Status: PASS

**问题：** 1280×720→640×360 后如何保持像素中心 convention？

**为什么现在解决：** RGB/mask 导出必须使用同一几何规则。

**输入文件：**

- `datasets/deform360/processed/008-pink-cloth/episode_0/metadata.json` 中的 K。
- T7 固定的相机名单。

**预计涉及文件：**

- `SoMA/tools/deform360_adapter/`：calibration 部分，具体文件名尚未确定。
- `SoMA/mmgs/datasets/embodied_dataset.py`：`extract_intrinsics`。
- `SoMA/mmgs/datasets/utils/cameras.py`：相机投影。

**最小修改：** 仅定义目标 K、WH 和 resize convention。

**验证方法：** 比较原图投影经 resize 后的位置与目标相机投影；检查已有主点 `639.5/359.5` 的映射。

**PASS：** fx/fy、主点、WH 与实际 resize 一致；SoMA 投影误差控制在约定的 1 像素内。

**FAIL 后检查：** 半像素偏移、H/W 顺序和 SoMA 对称投影；不先增加图像 warp。

#### Result

2026-09-25：PASS。仅建立目标分辨率的 intrinsics contract；不修改 dataset loader、训练代码、camera selection、RGB/mask pipeline 或 Gaussian initialization，未执行 T10。

- 实际输入是 processed undistorted calibration，等效为零 skew 的 PINHOLE 模型；三个 camera 的 `k1/k2/p1/p2` 均为 0，metadata 的 K 与标定文本逐值一致。发布文件没有显式 camera model 枚举，不能据此反推 raw capture 的镜头模型；此处不重复 undistort。
- 定义全幅 `1280×720 → 640×360`，`sx=sy=0.5`，不 crop、不 padding、不 warp。使用以整数为像素中心的 half-pixel 几何：`u'=(u+0.5)*sx-0.5`，`v'=(v+0.5)*sy-0.5`。对应 `fx'=sx*fx`、`fy'=sy*fy`、`cx'=sx*(cx+0.5)-0.5`、`cy'=sy*(cy+0.5)-0.5`。
- 所有主点从 `(639.5,359.5)` 变为 `(319.5,179.5)`；不能简单将 cx/cy 除以 2，否则每轴偏移 0.25 pixel。source/target WH 与传给 SoMA 的 `img_hw=[360,640]` 分别明确记录。
- 新增 [camera_intrinsics_contract.json](contracts/008-pink-cloth/episode_0/camera_intrinsics_contract.json)。Config A/B 的相机 ID 和顺序与 T8 及人工确认配置逐项一致，提供 `[2,3,3]` / `[3,3,3]` 的目标 K。
- SoMA `extract_intrinsics` 只读取 fx/fy，渲染使用对称投影；`ndc2Pix` 所隐含的主点正是 `((W-1)/2,(H-1)/2)`，与本次目标 K 完全一致，当前三相机无需修改 loader。

| Camera（共同前缀 brics-odroid-） | 原始 fx | 原始 fy | 目标 fx | 目标 fy |
|---|---:|---:|---:|---:|
| 023_cam0 | 883.532092296697 | 885.959648843671 | 441.7660461483485 | 442.9798244218355 |
| 009_cam1 | 994.010595847117 | 988.340865609786 | 497.0052979235585 | 494.170432804893 |
| 014_cam1 | 942.364497116677 | 937.850675702226 | 471.1822485583385 | 468.925337851113 |

后续 RGB/mask 导出必须遵守该完整画幅与像素中心映射；本轮没有生成或验证实际导出的图片，实际导出一致性留给对应后续 TODO。

#### Evidence

- 从服务器真实 `metadata.json` 与 `metric_params_refined_undistorted.txt` 按 camera name 精确读取；对照未修改的 T8 contract、两个 camera config 和 camera manifest，校验身份、顺序及输入 SHA256。读取前后 metadata、标定文本和原始 `splat_113.ply` hash 不变。
- 三个目标 K 均为有限 `[3,3]`、正 focal、零 skew、齐次末行 `[0,0,1]`；主点位于目标图像内且等于 SoMA 对称中心，缩放前后 FoV 差为 0。
- 使用内存坐标 ramp 检查现有 OpenCV `INTER_LINEAR` 和 `INTER_AREA` 的 2 倍降采样采样中心，两者最大误差均为 0 pixel。没有读取或导出 dataset RGB/mask，没有实现导出 pipeline。
- 对真实 frame 113 全部 12,861 个 Gaussian 中心，用 T8 w2c 转到各 camera，比较原图投影经 half-pixel resize 与目标 K 的投影，最大误差 `1.137e-13` pixel；三个 camera 下所有中心都落在 640×360 范围内。
- AST 提取并在服务器 CPU 执行未修改的 `mmgs/datasets/embodied_dataset.py` 中 `focal2fov` / `extract_intrinsics`（197/200 行）以及 `mmgs/datasets/utils/cameras.py:151` 的 `getProjectionMatrix`，按现有 rasterizer 的投影 epsilon 与 NDC-to-pixel 公式做 float32 数值检查；最大误差 `8.949e-5` pixel，小于本 TODO 的 1 pixel 容差。没有 CUDA 初始化或 GPU renderer 执行。
- Rasterizer 证据：服务器 `SoMA/gaussian-splatting/submodules/diff-gaussian-rasterization/cuda_rasterizer/auxiliary.h:41` 的 `ndc2Pix`，以及 `forward.cu:196–197,241` 的透视除法与像素转换。源码 hash、精确函数行号和逐 camera 验证数值保存在 contract。
- Artifact：`camera_intrinsics_contract.json`，16,956 bytes，SHA256 `3f4c3bbec096eb3ffb7dc3ee7928d68b464241bc9c383327dfc090ca9b393b97`。
- Git 范围：SoMA repository / `deform360-adaptation`；本次用户授权在 T9 PASS 后统一提交 T8 contract、T9 contract 和 roadmap，提交标题为 `Add SoMA-D360 camera geometry contracts`，具体提交与 push 状态以 Git 记录及本轮汇报为准。服务器 checkout 尚需后续通过 Git 显式同步。

### T10：定义单帧 30 点 controller representation

Status: PASS

**问题：** 一只夹爪如何得到固定身份的 30 个几何锚点？

**为什么现在解决：** 先确定 representation，再批量生成 trajectory。

**输入文件：**

- `datasets/deform360/processed/008-pink-cloth/episode_0/robot/robot.npy` 中 frame 113 的 pose/opening。
- 夹爪运动学：`deform360/deform360/processing/control_points_stage.py`。
- opening 映射：`deform360/deform360/processing/urdf_render.py`。

**预计涉及文件：**

- `SoMA/tools/deform360_adapter/`：拟新增的 controller 工具，文件名尚未确定。
- `SoMA/mmgs/datasets/embodied_dataset.py`：controller cluster 规则，只读。

**最小修改：** 只生成一个 frame 的 30 点和左右手指标签。

**验证方法：** 复用 `deform360/deform360/processing/urdf_render.py` 中的 `opening_to_umi_joints`（第 39 行附近）与固定几何关系；记录 opening clipping；核验手指归属和世界坐标。

**PASS：** `[30,3]` 有限、点身份固定、左右分组明确；没有读取 tactile 数值来选择点。

**FAIL 后检查：** EEF/root 坐标、opening 单位与 clipping、URDF offset；不改机器人轨迹。

#### Result

2026-09-25：PASS。只为 `source frame 113 = local frame 0` 定义并生成单帧 `float32 [30,3]` controller representation；未生成 trajectory、cache、graph，未执行 T11/T12，未修改 model、dataset loader、training code、Deform360 或 URDF。

- 新增权威 artifact：[controller_single_frame_contract.json](contracts/008-pink-cloth/episode_0/controller_single_frame_contract.json)，包含完整 30 点世界坐标、固定身份、左右标签、root 局部坐标、opening 映射、输入 hash 和验证证据。
- 位姿来源是 `robot/robot.npy:T_worlds[113]`，直接作为 `T_world_from_eef/root`；root 对应 URDF `umi_gripper_base`。单位为米，`P_world=P_root@R.T+t`，保持发布的 annotation world 坐标，不额外叠加 EEF offset、不重定位、不翻轴。`actions[113]` 只用于交叉校验 pose/opening，不重复作为变换应用。
- 原始 opening 为 `0.04499152071070773 m`，位于官方 `[0.04,0.112] m` 范围内，clipping 为 false。原样复用 `opening_to_umi_joints` 得到 `joint_left=+0.03571221967425896 m`、`joint_right=-0.03571221967425896 m`；两个 finger revolute joints 保持官方默认 0，不将 opening 直接当作两指点间距。
- 本次定义 v0 稀疏几何方案：每指固定 15 点，取已有 URDF 表面几何网格的行 `[0,6,11]`、列 `[0,8,16,23,31]`，形成每指 `3×5`。这是 v0 的确定性采样设计，不是声称官方已有 30 点格式；不使用随机采样、contact active 或 tactile 数值选择点。
- 点序固定：索引 `[0,15)` 为 `finger_left`，标签 0；`[15,30)` 为 `finger_right`，标签 1；每组按行优先、列递增。点 ID 为 `left/right_rXX_cXX`，绑定固定 URDF link/local anchor；左右表示同一夹爪的两根手指，不表示双臂或世界坐标 x 的符号。
- URDF 中 `tactile_*` link 名仅作为已有的静态几何位置来源，没有读取 tactile 文件、数值、history 或运行 contact detector。30 点全部由本帧 robot pose/opening 和固定几何决定。

实际几何记录：官方映射在本帧得到 `right_x-left_x=-0.0004995993485 m`，即两侧采样面沿 root-x 有约 0.50 mm 的交叉；独立 URDF FK 复现同样结果。保留原始 opening/offset，不按 x 排序或交换左右标签。T10 PASS 表示固定身份与运动学转换通过，不代表已证明夹爪几何无穿透、物体接触对齐或 graph 有效。

2026-09-25 人工决定：v0 默认/canonical controller geometry 保持 `candidate_7mm`，优先保留官方/原始 URDF 几何假设及 T10 已定义的稀疏表示不变，作为 baseline；`candidate_5mm` 保留为后续 alternative ablation，`candidate_0mm` 保留为 reference control。7 mm 在部分近闭合帧出现负 signed gap，作为已记录的 representation limitation，本阶段不修正；这不是对物理无穿透或真实接触面拟合的认证。未修改几何数值、生成逻辑或源码，未执行 T11/T12。

#### Evidence

- 输入：真实 `robot/robot.npy`，SHA256 `ba12216e8f9924e593bb470eda5ef33411e50262271879c38ecfc55c9f07c561`；只使用 frame 113 的 `T_worlds`、`openings`、`actions` 及 `bimanual=False`。以仓库 T4 `frame_manifest.json` 核对 `113↔0`。原始 robot 文件和引用源码/URDF 的读取前后 hash 均不变。
- 源码证据：Deform360 `processing/urdf_render.py:39` 的 `opening_to_umi_joints`；`processing/control_points_stage.py:79` 的 `_taxel_grid_root_frame` 与 `:117` 的 `gripper_taxel_points`；`robot.py:167` 附近的 `RobotState` 对 world-from-EEF 和米制单位的说明。源码常数、函数行号及文件 hash 保存在 contract。
- CPU 上提取执行未修改的 opening/几何函数，与 XML 中 `umi_tactile.urdf` 的 prismatic、zero-angle revolute 和 fixed anchor joint 链独立计算交叉验证。官方完整网格仅作为同一 frame 的内存参考，最终只序列化选定的 30 点；没有运行完整 processing stage。
- 验证：shape `[30,3]`、dtype `float32`、全部 finite；30 个唯一点 ID、30 个唯一世界坐标；左右各 15 点且顺序固定。与官方几何结果最大差为 0，独立 URDF FK 世界坐标最大差 `8.327e-17 m`，float32 转换最大差 `6.667e-9 m`，世界坐标到 root 往返最大差 `1.111e-16 m`。
- 同一 frame 重复计算逐 bit 一致；JSON 重读并显式转换为 `np.float32` 后逐 bit 一致。JSON 本身不携带二进制 dtype，因此 contract 明确 `controller_dtype=float32` 和读取方式，未额外生成 NPY。点数组按 little-endian float32 / C order 的 SHA256 为 `6d76fa3863e9d6acf96678d98f1936851e25966a005b1d32539ef9bb2636041b`。
- SoMA `mmgs/datasets/embodied_dataset.py:611` 的 `_load_controller_cluster_mask` 仅只读参考：两组 controller 与后续 `dis_split` 连续前/后半分组支持保留 finger-major 点序；本轮未生成 cluster mask 或验证未来任意 cluster 配置。
- Artifact：`controller_single_frame_contract.json`，30,070 bytes，SHA256 `a2aa7684e9612d4b80eeede253fb890a63b6f8dbd87665843222170d341d733a`。
- 2026-09-25：已生成 source 113–140 的 [review visualization](visualizations/t10-review-20260925/README.md)，待人工确认；仅补充观察材料，不修改 T10 既有结论或 representation，不执行 T11/T12。
- 2026-09-25：visual review 后对 signed gap 做了 [follow-up check](visualizations/t10-review-20260925/signed_gap_followup.md)，解释采样面位置顺序反转及实体穿透尚未验证的区别；不修改 T10 PASS，不推进 T11/T12。
- 2026-09-25：controller geometry assumptions are now parameterized through [controller_geometry_config.json](contracts/008-pink-cloth/episode_0/controller_geometry_config.json); no canonical offset changed. 7/5/0 mm 均为固定身份的单帧静态参数检查，当前默认仍为 7 mm；未生成 trajectory，未执行 T11/T12。
- 2026-09-25：已生成 source 113–140 的 [0/5/7 mm offset comparison review](visualizations/t10-offset-comparison-20260925/README.md)，含两路 RGB overlay、代表帧、可拖拽三维图及 gap 统计，待人工确认 canonical offset；当前默认仍为 candidate_7mm，T10 既有结论与 T11/T12 状态不变。
- Git：所属 SoMA repository / `deform360-adaptation`。roadmap 为 tracked 文件；新 contract 当前尚未 tracked。本轮未 add/commit/push，等待单独 checkpoint 指令，后续需通过 Git 同步到服务器；本轮未更新服务器 checkout。
- 2026-09-25 checkpoint 确认：用户已人工确认 7 mm 为 v0 canonical baseline，5 mm 为 ablation candidate，0 mm 为 reference control；决定及原因保存在 `controller_geometry_config.json:human_decision`。review 中的“待人工确认”与 config 输入 hash 保留为生成时的历史记录，本条与 config 中的人工决定为当前状态。用户授权以 `Add SoMA-D360 controller representation contract` 提交 T10 contract、roadmap 及两个 review 目录并 push；具体提交状态以 Git 记录为准，服务器 checkout 仍需后续显式 Git 同步。

### T11：将固定 controller representation 扩展为 trajectory

Status: PASS

**问题：** 固定 30 点能否随 pose/opening 正确运动？

**为什么现在解决：** representation 已由 T10 固定。

**输入文件：**

- `datasets/deform360/processed/008-pink-cloth/episode_0/robot/robot.npy`
- T4 frame manifest。
- T10 固定锚点及其标签。

**预计涉及文件：**

- `SoMA/tools/deform360_adapter/`：controller 工具，文件名尚未确定。
- `SoMA/mmgs/datasets/embodied_dataset.py`：trajectory 读取。

**最小修改：** 输出 `controller_points [194,30,3] float32`。

**验证方法：** 首尾及抽样帧独立计算；逐帧检查点身份和 source frame 对应。

**PASS：** shape、dtype、有限性全部正确，local 0 对应 source 113，没有重排或跨帧重新采样锚点。

**FAIL 后检查：** 位姿乘法方向、索引偏移、广播和点序。

#### Result

2026-09-25：PASS。仅将 canonical `candidate_7mm` 的固定单帧定义扩展为完整 controller trajectory，保留原始 opening mapping 与已知 signed-gap limitation；未修改 loader/training code，未读取 tactile，未生成 `track_process_data.pkl`、cache、graph 或 cluster，未执行 T12。

- 新增 [build_controller_trajectory.py](../../tools/deform360_adapter/build_controller_trajectory.py)，从 geometry config 读取 offset、局部方向、Y/Z 网格、finger base、opening 参数及 world 变换约定；固定网格只构建一次，不跨帧重采样或重排。
- 使用 T4 `frame_manifest.json` 的逐帧映射：source 113–306 ↔ local 0–193，输出 `controller_points [194,30,3] float32`。left indices 0–14 / right indices 15–29，全部 point ID 与 T10 一致。
- 服务器派生数据：`datasets/deform360/derived/008-pink-cloth/episode_0/t11_controller_candidate_7mm/controller_points.npy`，69,968 bytes；不纳入 Git、不复制到 Mac。
- 新增权威小型记录：[controller_trajectory_contract.json](contracts/008-pink-cloth/episode_0/controller_trajectory_contract.json)，包含输入/hash、canonical candidate、frame mapping、点序、服务器输出路径/hash、验证结果与执行工具 hash。
- 完整窗口中 136 帧触发官方 `[0.04,0.112] m` opening clipping，194 帧的 joint 映射与官方函数逐值一致。186 帧 signed gap 为负，min/median/max 为 −5.075160 / −5.075160 / +3.258952 mm；不修改 clipping、offset 或左右标签，不据此声称实体穿透已验证。
- T11 checkpoint 人工决定：继续使用 canonical `candidate_7mm`，不修正 negative signed gap 或官方 opening clipping；两者作为 v0 baseline 的已知 representation limitation 保留，不据此宣称真实接触几何无误。

#### Evidence

- 准备动作：按用户指令删除未跟踪的 `t10-review-20260925.zip` 后，Mac SoMA working tree clean；服务器原 working tree clean，以 fetch + `merge --ff-only` 从 `8e8772a` 同步到 `b615c0b094ec340c29dedd75967aef4f88d03ade`，同步后仍 clean，无覆盖未提交修改。
- 使用服务器现有 Python 3.10.21 / NumPy 1.26.4，CPU 执行。新工具经逐文件复制到服务器临时执行目录，未通过未提交源码覆盖服务器 checkout；长期版本保存在 Mac SoMA tools 目录，等待用户确认后 Git 同步。
- 验证 shape `[194,30,3]`、dtype float32、全部 finite、每帧 30 个唯一点、固定 identity/order、无删除 frame；NPY 保存并重读逐 bit 相同，trajectory[0] 与 T10 source 113 float32 坐标逐 bit 相同。
- 独立验证：AST 提取未修改的 `opening_to_umi_joints`，与配置驱动的 194 帧 joint 值比较，最大误差 0。另从原始 URDF XML 组合 joint-chain FK，对 source 113、114、140、170、200、233、267、268、306 重算；float64 世界坐标最大误差 `5.551115123125783e-17 m`，转为 float32 后九帧全部逐值一致，包括 local 0 与 local 193。
- 所有输入文件/hash 在执行前后不变；T10 单帧 contract 与 geometry config 未修改。pose rotation 最大正交误差 `1.5543122344752192e-15`，determinant 最大误差 `1.7763568394002505e-15`。
- 输出 NPY 文件 SHA256：`d0575b90ebb8aaeb8d4e9154a2ac20985bc3ca1d650432a25110011414d72820`；数组 little-endian float32 / C order bytes SHA256：`5682358def9df9bd52a27715f515acc61c935e645b8f341bd0c6ebde008a0dc6`。
- Git：SoMA / `deform360-adaptation`。roadmap 为 tracked 修改，新工具和 trajectory contract 尚未 tracked；本轮未 add/commit/push。服务器数据保持 server-only，Mac 小型文件等待用户确认后提交并通过 Git 同步。T11 已完成并停止，T12 TODO 不变。

### T12：确认 controller 层级分组不跨手指

Status: PASS

**问题：** SoMA 默认连续分组可能把 15+15 点切成 14+16。

**为什么现在解决：** trajectory 正确不代表图中的 controller cluster 正确。

**输入文件：**

- T10 标签。
- T11 controller trajectory；实际输出路径以 T11 记录为准。
- 现有 controller scheme，参照 `SoMA/configs/SoMA/cloth_lift_stage1.py`。

**预计涉及文件：**

- `SoMA/mmgs/datasets/embodied_dataset.py`：`_load_controller_cluster_mask`。
- `SoMA/configs/SoMA/deform360_v0_stage1.py`（拟新增）：controller scheme。
- `SoMA/configs/SoMA/deform360_v0_stage2.py`（拟新增）：controller scheme。

**最小修改：** 只调整点排列或 controller scheme 中的一项，使之符合已选 representation。

**验证方法：** 打印每层每个 cluster 的原始点 ID 和手指标签。

**PASS：** 两个顶层 cluster 分别属于两个手指，层级映射合法，没有跨指混合。

**FAIL 后检查：** 连续分组边界、第一层 cluster 数和旧 cache；不改 controller 几何。

#### Result

2026-09-25：FAIL。实际现有 `dis_split` 默认层级为 30→15→2→1。第一聚合层 C7 包含 14(left_r11_c31) 与 15(right_r00_c00)；双手指层 C0 为 0–13，C1 为 14–29，后者跨指，未满足 PASS 条件。末尾 2→1 是显式共同 controller root，双指共同祖先不等同于手指层意外混合。

仅在内存比较首层 `num_cluster=10` 的方案，得到 30→10→2→1；双手指层分别为 0–14 / 15–29，无手指层混合。不改点身份与 trajectory，仅建议后续调整 v0 grouping scheme；尚未应用任何配置或源码修改。仅重排点无法把每指奇数 15 点完整拆成纯指二元组。等待人工确认方案；不执行 T13。

2026-09-25 人工确认后的收尾：PASS。正式采用独立 canonical grouping contract：`cluster_type=dis_split`，首层 `num_cluster=10`，第二项保留 `downsample_rate=0.5`；现有 loader 后续层实际按索引二分。层级为 30→10→2→1，每个首层 cluster 连续 3 点，双手指层分别对应原始 0–14 / 15–29，共同 root 允许合并。逐层覆盖全部 30 点恰好一次，无重复/遗漏，索引合法，手指层无混合。上文默认 FAIL 和候选阶段记录保留为历史。

本轮仅新增 grouping contract，未创建训练 config；后续 v0 Stage 1/2 配置必须采用此 scheme，JSON 不会自动改变现有 sample 配置。未修改几何、7 mm、point identity/order、trajectory 或 model architecture。T12 完成后停止，不执行 T13。

#### Evidence

- [逐层全部 point IDs / left-right 标签、源码位置及方案比较](t12-controller-grouping-audit.md)。
- [机器可读分组审计及输入/source hashes](contracts/008-pink-cloth/episode_0/controller_grouping_audit.json)。真实 T11 trajectory SHA256：`d0575b90ebb8aaeb8d4e9154a2ac20985bc3ca1d650432a25110011414d72820`。
- CPU 执行现有分组函数，文件读写/cache 均替换为内存操作；无磁盘 PKL/cache，无完整 graph/rollout，无几何、opening 或 T10/T11 contract 变更。
- T11 checkpoint：`e755d1e3b3060dca7f7cc16b29b63ff8202c42c0` 已推送，开始 T12 前 Mac working tree clean。T12 文件属于 SoMA / `deform360-adaptation`：roadmap tracked 修改，两份审计文件新建未跟踪；未 commit/push，服务器 Git checkout 待后续同步。

- 本次收尾新增 [canonical grouping contract](contracts/008-pink-cloth/episode_0/controller_grouping_contract.json)，含逐层完整成员/标签、p2c、来源 hashes 与验证结果。与此前真实 loader 内存执行输出逐项一致；再次静态验证每层覆盖、标签和索引。
- 服务器 T11 NPY SHA256 再次核验仍为 `d0575b90ebb8aaeb8d4e9154a2ac20985bc3ca1d650432a25110011414d72820`；T10 contract、geometry config 和 loader/config 源文件 hash 均未改变。
- 本次未 commit/push。SoMA / `deform360-adaptation`：roadmap 为 tracked 修改，grouping contract 与此前两份审计文件未跟踪，待确认后提交；另一端 Git 同步尚未进行。

### T13：确认重力方向与世界坐标

Status: PASS

**问题：** 不能直接沿用官方 sample 的 gravity quaternion。

**为什么现在解决：** 错误重力会污染后续动力学验证。

**输入文件：**

- Gaussian 坐标依据：`datasets/deform360/processed/008-pink-cloth/episode_0/splatfacto/splat_113.ply` 及对应格式说明。
- Robot 坐标依据：`datasets/deform360/processed/008-pink-cloth/episode_0/robot/robot.npy` 及对应坐标说明。
- 发布标定：`datasets/deform360/processed/008-pink-cloth/episode_0/metric_params_refined_undistorted.txt`
- T6 时间约定。

**预计涉及文件：**

- `SoMA/mmgs/datasets/embodied_dataset.py`：gravity 处理。
- `SoMA/configs/SoMA/deform360_v0_stage1.py`（拟新增）：`rot_est`。
- `SoMA/configs/SoMA/deform360_v0_stage2.py`（拟新增）：`rot_est`。

**最小修改：** 仅确定并输出本 scene 的重力方向描述。

**验证方法：** 建立坐标方向证据，验证 SciPy xyzw 和 inverse rotation 后的实际重力向量。

**PASS：** 向量方向、模长及时间缩放可明确解释；Gaussian、controller、camera 使用同一世界坐标。

**FAIL 后检查：** 坐标定义、旋转方向和四元数顺序；不独立旋转某一种模态。

#### Result

**当前收口结论：PASS（user-approved engineering assumption）。** 人工固定 v0 annotation-world `up=+Z`、gravity direction `-Z`，`raw_gravity_world=[0.0,0.0,-9.8]`，`rot_est` 为 SciPy xyzw identity `[0,0,0,1]` 或 omitted。沿用 T6，由 dataset 仅一次乘 `(real_dt/comp_dt)^2=4`，得到 nominal `external_gravity=[0.0,0.0,-39.2]`；不预缩放、不额外旋转。

PASS 仅表示“v0 gravity convention 已固定并可执行”，不是官方 gravity definition，也不表示真实物理竖直已独立测量或验证。no-tactile baseline 与未来 tactile variant 必须使用完全相同的上述 gravity convention；gravity uncertainty 是共同 limitation，不作为 tactile/no-tactile 比较变量。本阶段不再要求继续寻找 physical world-up。

以下此前 provenance / support-plane FAIL 记录原样保留为历史 limitation；其中待确认物理方向、null 和下一步调查是当时状态，已由上述人工工程约定完成 v0 收口，并不被追认成物理验证成功。

2026-09-25：FAIL（物理竖直方向证据不足，并非已发现坐标错位）。服务器 SoMA 在 clean 状态下 fast-forward 到 `a4eb19dac824275efb8b33c6f7ef8a658993f98c` 后完成只读检查。现有 Gaussian、robot/controller 与 T8 camera 保持同一发布标定世界系；未施加任何独立旋转。该系的物理 X/Y/Z 方向、世界 up 和重力参考没有在检查的 metadata/robot 字段与源码约定中给出，不能将 camera OpenCV 轴或 gripper root 轴当作 world-up；因此不能确认四者（含 gravity）完全对齐。

已明确 `external = R_xyzw(rot_est)^(-1) · (raw_gravity × 4)`。T6 两阶段 real_dt=1/15、comp_dt=1/30，倍率仅应用一次；基准模长 9.8 对应 external 模长 39.2，不预乘、不按 gap/FPS 再乘。若将来直接输入已证实的 world gravity，应省略 rot_est 或用 identity；若输入参考系负 Z 重力，q 必须表示 world→reference，逆旋转映射回 world。当前 raw vector、q 和 final vector 均留空，不能当作可执行最终配置。

最小下一步仅补充该标定系的官方 gravity/world-up 定义或独立测量的竖直参考，不猜轴、不试翻轴、不照搬 sample quaternion。不执行 T14。

本次 support-plane 直接估计尝试：继续 FAIL。读取 36 cameras × source 113/120/200/267，共 144 组实际 depth/mask，物体 mask 外非零深度总数为 0。RGB 中可见透明承物板，但现有 rendered_depth 仅提供物体深度，不能反投影得到独立板面点。本轮未拟合布料来冒充承物板；未得到可靠 plane normal，角度误差及 raw/final gravity 保持 null/N/A。RGB-only 板角/边同名点三角化尚未执行，需要可确认属于板面的非共线对应点；本结论不声称图像路线不可能。未根据 camera/cloth/robot 朝向猜轴。不执行 T14。

#### Evidence

- [gravity_world_frame_contract.json](contracts/008-pink-cloth/episode_0/gravity_world_frame_contract.json)：FAIL，明确 `usable_as_final_gravity_config=false`，包含原始文件 hash、T8/T10/T11 引用、坐标检查和 T6 时间规则。
- 实际 PLY 12861 点；用 robot frame 113 重算 30 点与 T10 float32 逐值一致。Config A/B 所有相机下 Gaussian 正深度比例 1.0、controller 正深度 30/30，world→camera→world 最大误差 2.22e-16 m。往返只验证代数一致性，不能证明物理重力方向；没有据此宣称世界 Z 向上。
- 原样抽取并 CPU 执行 `embodied_dataset.py:921–934`。合成 identity xyzw `[0,0,0,1]` 与输入 `[0,0,-9.8]` 得 `[0,0,-39.20000076293945]`；合成绕 X 正 90° 的 xyzw `[0.70710678,0,0,0.70710678]` 得约 `[0,-39.2,0]`，验证 inverse 方向与模长。这两例仅为代码语义测试，不是目标 episode 的重力决定。
- `deform360/README.md:491–522` 明确 metric world Gaussian、T_worlds 与 OpenCV camera-local 轴，未定义物理 world-up；实际 metadata 只有 intrinsics/WH/fps/frame_num/start_frame/end_frame/cameras，robot 只有 actions/T_worlds/openings/bimanual。calibration qvec/tvec 是相机姿态，不含独立重力标定。
- 输入数据检查前后 hash 不变；未修改 Gaussian/controller/camera、loader/model、config，未训练。SoMA / deform360-adaptation：roadmap tracked 修改，新 gravity contract 未跟踪；未 commit/push。服务器源码已同步，新增 T13 文档待后续 Git 同步。

- [support-plane review](visualizations/t13-support-plane-review-20260925/README.md)：完整 144 组统计、切片 hash、三相机 RGB/depth 对照及现有几何 3D 诊断图。没有可信 plane，因此未画 normal/gravity 箭头。
- gravity contract 已追加本次直接检查结果及缺失条件，仍 `FAIL` / `usable_as_final_gravity_config=false`；T6 ×4 和 inverse rotation 既有语义不变。未修改原始 HDF5/视频/PLY、controller、calibration、loader/model/config。SoMA / deform360-adaptation：roadmap tracked 修改，gravity contract/review 未跟踪；未 commit/push，待后续 Git 同步。

- 人工收口更新：gravity contract 顶层改为工程约定 PASS，记录 user approval、physical verification=false、official definition=false、raw/identity/×4/final vector 及跨 variant 不变规则；旧 support-plane FAIL 与源数据验证 evidence 未改动。
- 数值核验：`[0,0,-9.8]×4=[0,0,-39.2]`；既有 CPU 执行 dataset identity inverse 的 float32 结果为 `[0,0,-39.20000076293945]`，差异仅舍入。无源码/config/data 改动；未执行 T14，未 commit/push。

### T14：只导出 RGB

Status: PASS

**问题：** SoMA 需要逐帧图像文件。

**为什么现在解决：** 时间与相机几何规则已经固定。

**输入文件：**

- T7 选定两个相机的 `datasets/deform360/processed/008-pink-cloth/episode_0/<camera_id>/undistorted.mp4`。
- T4 时间映射、T7 相机顺序、T9 resize/内参约定。

**预计涉及文件：**

- `SoMA/tools/deform360_adapter/`：拟新增的 RGB 工具，文件名尚未确定。
- `SoMA/mmgs/datasets/utils/io.py`：读取接口。

**最小修改：** 导出两个相机的 194 张 640×360 RGB PNG。

**验证方法：** 顺序解码，抽查首尾、split 边界以及 gap=10 采样帧。

**PASS：** 每相机文件名连续 `0.png…193.png`，尺寸/色序正确，source frame 映射一致。

**FAIL 后检查：** 视频 seek 偏差、BGR/RGB、编号或 resize；不同时处理 mask。

#### Result

2026-09-25：PASS。按本次明确指令扩展原两相机导出范围，为 Config A/B 的三个唯一 camera 各生成一份 canonical RGB；未改变相机身份或顺序。A 引用 023_cam0→009_cam1，B 引用 023_cam0→009_cam1→014_cam1，同一 camera 共用同一路径，不重复保存。

服务器产物：`/data1/userdata/tcweng/projects/tcgs/datasets/deform360/derived/008-pink-cloth/episode_0/t14_rgb/<camera_id>/0.png…193.png`。三个 camera 各 194 张，共 582 张，约 131 MiB，全部 server-only。

严格按 T4 source 113–306 ↔ local 0–193，顺序解码完整 357 帧，无 seek、无 tactile 过滤、无 split 改动。按照 T9 将全幅 1280×720 用 OpenCV INTER_AREA 缩至 640×360（2×2 box 支持中心符合 half-pixel 映射），无 crop/warp/再次 undistort。显式 BGR→RGB，再用 Pillow 写 RGB PNG；全部 uint8、值域位于 [0,255]。

每帧重新读取 PNG，验证 RGB 数组与写入前逐值一致；第二次完整顺序解码使用独立 NumPy 2×2 平均，全部 582 张与对应 source 内容最大差 0.5 灰度级（整数舍入）。每 camera 194 个文件名/像素 hash 均唯一，无遗漏、重复内容或编号跳跃。未处理 mask，未组装 scene、修改 loader/model/training/config 或 T9 contract，未训练，不执行 T15。

#### Evidence

- [独立 RGB export 工具](../../tools/deform360_adapter/export_rgb.py)：拒绝已存在输出，校验 T4 timestamp hash 与逐帧 token，校验 T7/T9 camera order，完整顺序解码两次，PNG RGB roundtrip、独立 box-average 验证、逐文件 SHA256。仅 CPU 执行，无 CUDA。
- [RGB export contract](contracts/008-pink-cloth/episode_0/rgb_export_contract.json)：约 217 KB，包含三个视频 SHA256、工具/输入 contract hashes、582 条逐帧输出 manifest、RGB 像素 hash、min/max、两套配置共享资产引用、验证记录。
- 三个 camera 分别验证 source 113→local 0、123→10（gap=10）、200→87、267→154（train 末帧）、268→155（test 首帧）、306→193；上述抽查及全部输出帧的独立 box-average 最大差均不超过 0.5。
- 服务器执行前 clean，`fetch + merge --ff-only` 同步到 `edd7211eb023c53bf12fca3ca4dacf1040a000ca`。输入视频执行前后 hash 一致；Mac 与服务器工具 hash 相同，输入 T4/T7/T9/T5 contract hashes 保持不变。
- Git：SoMA / deform360-adaptation。Mac roadmap tracked 修改，新增工具和 RGB contract 尚未跟踪；服务器仅新增同版本工具未跟踪，PNG 位于仓库外。未 add/commit/push。Mac 未下载 RGB，后续 checkpoint 后通过 Git 同步小型文件；服务器现有未提交工具须核对同 hash 后安全处理，不覆盖 dirty tree。

### T15：只导出 object mask

Status: PASS

**问题：** 真实 HDF5 是 0/1，而 SoMA 图像 reader 按 255 归一化。

**为什么现在解决：** 防止监督图像被缩暗到原来的 1/255。

**输入文件：**

- T7 选定两个相机的 `datasets/deform360/processed/008-pink-cloth/episode_0/<camera_id>/mask_refined.h5`。
- T4 时间映射、T7 相机顺序、T9 resize/内参约定。

**预计涉及文件：**

- `SoMA/tools/deform360_adapter/`：拟新增的 mask 工具，文件名尚未确定。
- `SoMA/mmgs/datasets/utils/io.py`：mask 读取。

**最小修改：** 最近邻 resize，输出单通道 0/255 PNG。

**验证方法：** 检查全部输出取值、尺寸、对应帧及 RGB/mask 叠图。

**PASS：** 每相机 194 张、仅 0/255、与 RGB 对齐，无意外空 mask。

**FAIL 后检查：** HDF5 key、帧偏移、插值方式、通道数；不重跑 segmentation。

#### Result

2026-09-26：PASS。按本次用户明确要求，为 Config A/B 三个唯一 camera 各导出一次 canonical object mask。每台 194 张 `0.png…193.png`，source 113–306 ↔ local 0–193，合计 582 张；约 2.3 MiB，保持 server-only。

服务器路径：`/data1/userdata/tcweng/projects/tcgs/datasets/deform360/derived/008-pink-cloth/episode_0/t15_mask/<camera_id>/`。A 顺序为 023_cam0→009_cam1，B 为 023_cam0→009_cam1→014_cam1，均引用同一批资产，不重复保存。

源 `mask_refined.h5:data` 验证为 `[357,720,1280] uint8`，窗口内仅 0/1。全幅 Pillow NEAREST 输出 `[360,640] uint8` 单通道 L PNG，将原前景 1 编码为 255、背景保持 0。half-pixel 逆映射 `x_source=2*x_target+0.5`，最近邻平局取高索引，即 `[1::2,1::2]`；独立数组索引核对全部 582 张逐值一致。无双线性、crop、warp、形态学或 segmentation 重跑。

所有帧 foreground 非空、值域仅 0/255，编号完整，每台 194 个像素 hash 均唯一；T4/T7/T9/T14 帧/相机顺序一致。source 113/200/267/306 的三相机叠图未见明显错帧或尺度偏移；保留原始 segmentation 边界，不宣称原分割完美。未处理 016_cam0，未改变 split/tactile 筛选/RGB/loader/model/training code，不执行 T16。

#### Evidence

- [独立 mask exporter](../../tools/deform360_adapter/export_mask.py)：CPU h5py/NumPy/Pillow，拒绝已有输出，逐帧源标签校验、独立最近邻采样验证、PNG 单通道/shape/dtype/值域/foreground/编号/重复内容校验。
- [mask export contract](contracts/008-pink-cloth/episode_0/mask_export_contract.json)：源 HDF5 SHA256、data key、工具及输入 contract hashes、582 项输出 manifest/hash、相机顺序、A/B 共享引用、全部验证结果。
- [12 张代表帧 RGB/mask overlay](visualizations/t15-mask-review-20260926/overlays.jpg)：行依次 023_cam0/009_cam1/014_cam1，列 source 113/200/267/306（local 0/87/154/193）。仅小型 review 拼图到 Mac，无 canonical mask/RGB sequence 复制。
- 输入 HDF5 前后 hash、全部 T14 RGB 文件 hash、已有 T4/T7/T9/T14/T5 contract hashes 均未改变。导出无源码/配置变更。
- 按明确授权将服务器未跟踪 T14 工具移至 `/tmp/tcgs-t14-tool-backup-alxjc4yw/export_rgb.py`，随后 clean fast-forward 到 `14261bcab4281a87e79f13b7292f41aa8be18817`，ahead/behind=0/0；tracked/备份 SHA256 同为 `6cdef1dff087a0acf8a166eb222a1b285fece0e9c765aae855611aa8e46b20da`，备份保留。T15 工具从仓库外临时路径执行，服务器 checkout 仍 clean。
- Git：SoMA / deform360-adaptation。Mac roadmap tracked 修改，新 exporter、mask contract 和 review JPG 未跟踪，未 add/commit/push；后续 checkpoint 后通过 Git 同步小型文件到服务器。

### T16：明确 object mask 与遮挡 loss 的边界

Status: PASS

**问题：** 现有包没有单独的 robot obstacle mask；object mask 不等于 robot mask。

**为什么现在解决：** 避免 loader 能跑，但监督语义被误述。

**输入文件：**

- T15 导出的 object mask；实际输出路径以 T15 记录为准。
- 现有 render loss 实现：`SoMA/mmgs/models/heads/acc_decoder.py`、`SoMA/mmgs/models/losses/ssim_loss.py`。

**预计涉及文件：**

- `SoMA/mmgs/datasets/utils/io.py`：GT/mask 读取。
- `SoMA/mmgs/models/heads/acc_decoder.py`：`forward_train`，第 193 行附近。
- `SoMA/mmgs/models/losses/ssim_loss.py`：`forward`，第 57 行附近。

**最小修改：** 单独固定 v0 的监督约定，默认保留现有 loss；不伪造 obstacle mask。

**验证方法：** 检查 object masked GT、空 obstacle 分支、L2/SSIM 的实际权重处理。

**PASS：** 明确哪些像素参与哪些 loss，记录当前 SSIM 不使用 `mask_weights` 的限制；没有宣称已实现完整遮挡处理。

**FAIL 后检查：** 类别标签、mask 分支或 loss 参数传递；若确需改 loss，另列单项，不混入 adapter。

#### Result

2026-09-26：PASS（监督语义固定，不表示遮挡问题已解决）。v0 保留当前 loss，采用黑背景、关闭 random/white background、bounding_box=None 的全幅监督约定；后续 scene packaging 必须仅将 T15 object mask 分类为 object，不创建或伪造 robot/obstacle mask。

Reader 将 RGB 乘 object mask/255，dataset 再将 RGB /255 得 gt_label；`mask_object` 实际是 masked RGB，不是 binary loss weight。object mask 改写 GT，不直接作为 L2/SSIM 权重。没有 obstacle 时 dataset 原生返回零 controller/robot mask；这表示无排除信息，不是测得全零 robot segmentation。

`controller_img_mask` 是所有 obstacle 的 union（包含 robot），decoder 用其补集作为权重。L2 消费该权重；v0 权重全 1，背景和 object-mask 外 robot/遮挡区域仍参与监督，GT 在这些区域为黑。`pure_robot_img_mask` 是标签含 robot 的 obstacle 子集，当前 Stage1/2 render loss 不读取它。

SSIMLoss.forward 未消费传入的 mask_weights，实际对 bbox 内整幅预测和 masked RGB GT 计算 SSIM；背景及窗口边界均参与。默认 bbox 为全图，因此 v0 L2 与 SSIM 均不是 foreground-only loss。未来即使加入真实 obstacle，当前 SSIM 仍不会自动排除其区域。

v0 接受以上已知 limitation，不修 loss/loader。若以后要求完整遮挡排除，须另立独立任务取得真实 robot mask，并单独验证 SSIM 的 mask/window/reduction 语义。未新增或执行后续 TODO，不进入 T16.5/T17。

#### Evidence

- [supervision_mask_contract.json](contracts/008-pink-cloth/episode_0/supervision_mask_contract.json)：实际调用链、公式、源文件行号/hash、输入 RGB/mask contract hashes、缺失项、后续 packaging 要求及 limitation。
- `io.py::read_video_image_rgba_cv2_mask:367–456`；`embodied_dataset.py:136–153,846–897,944–958`：object-masked GT、全幅 bbox、空 obstacle/robot 分支。
- `acc_decoder.py::forward_train:193–213`；`sim_head.py::loss:52–78`；`l2_loss.py:10–39,68–95`；`ssim_loss.py:11–28,57–81`：L2 实际使用 complement weight；SSIM 不转发该 kwargs。
- Stage1 simulator `705–723`、Stage2 `779–794` 对应目标帧取 GT 和 controller_img_mask；全仓搜索 pure_robot_img_mask 仅发现 dataset 返回及注释掉的 test 引用，没有当前训练消费。
- T15 checkpoint `6ef674571cc239d939a08f4cdd1213f8f6c1c48c` 已 push，T16 开始前 Mac clean、ahead/behind=0/0。本轮只读审计，未构建 scene/model、未运行训练。SoMA / deform360-adaptation：roadmap tracked 修改、新 supervision contract 未跟踪，T16 未 commit/push，服务器待后续 Git 同步。

### T16.5：批量生成 SH0 compatible Gaussian sequence

Status: PASS

**问题：** 如何将 Deform360 的 Gaussian sequence 转换为 SoMA 使用的统一 SH0 representation？

**为什么现在解决：** T2 只验证 initial Gaussian conversion，不代表整个 sequence 已完成转换；在 scene packaging 和 training 前，需要保证所有需要的 frame 使用一致的 Gaussian schema。

**输入文件：**

- `datasets/deform360/processed/008-pink-cloth/episode_0/splatfacto/splat_<frame>.ply`
- T4 frame mapping：`SoMA/docs/deform360/contracts/008-pink-cloth/episode_0/frame_manifest.json`。
- T2 conversion validation result：本 roadmap 中 T2 的 Result / Evidence。

**预计涉及文件：**

- `SoMA/tools/deform360_adapter/`：拟新增 Gaussian sequence conversion 工具，具体文件名尚未确定。
- 不修改 `SoMA/gaussian-splatting/scene/gaussian_model.py`。

**最小修改：** 扩展 T2 已验证的 conversion 方法，生成独立 SH0 PLY sequence，保留原始 SH3 PLY。逐帧确认待移除的 `f_rest_*` 全部为零；若存在非零项，停止并报告，不将 T2 的单帧结论外推为整段结论。

**验证方法：**

- frame 范围：source 113–306，对应 local 0–193。
- 逐帧比较派生文件与对应原始文件：Gaussian 数量是否一致、point order 是否一致、xyz/opacity/scale/rotation 是否保持一致；保留 SH DC 及其他非 `f_rest_*` 属性。
- 在 Slurm GPU allocation 内使用原始 GaussianModel(0) 验证所有 frame 均可读取。

**PASS：** 所有需要的 frame 均存在 SH0 派生 PLY；与对应原始 Gaussian identity 一致；原始数据未修改。

**identity 边界：** 此处指每帧转换前后的点身份及顺序一致，不据此假定不同 source frame 的重建 PLY 之间具有固定 identity；不将逐帧重建 Gaussian 自动作为 SoMA rollout state。

**FAIL 后检查：** 单帧转换失败、非零高阶 SH、PLY schema 差异、property 顺序、文件完整性。

#### Result

2026-09-26：PASS。source 113–306 ↔ local 0–193 的 194 个 PLY 全部通过整段预检、SH0 转换、逐字段保真检查及 Slurm GPU allocation 内原始 GaussianModel(0) 加载验证。

所有输入的 f_rest_0…44 均严格等于 0；仅移除这 45 个属性，保留 xyz、normals、SH DC、opacity、scale、rotation 的名称、property 顺序、dtype、逐行字节和 vertex order。无 NaN/Inf，原始文件和 loader 执行前后 SHA256 不变。

派生文件位于服务器 `datasets/deform360/derived/008-pink-cloth/episode_0/t16_5_sh0/`，共 194 个，文件大小合计 171,368,354 bytes（约 163.43 MiB；du 显示 164M）。没有复制 PLY 到 Mac，也没有纳入 Git。

identity 一致仅指同一 source frame 转换前后；本窗口各帧点数范围 12,475–13,731，不能据此建立跨帧固定 Gaussian identity。不得使用 future reconstructed PLY 作为 rollout reset state。未组装 scene、未训练、未执行 T17；T2/T3 历史结论不变。

#### Evidence

- Git-managed 小型 contract（当前未跟踪、待审阅提交）：[gaussian_sequence_sh0_contract.json](contracts/008-pink-cloth/episode_0/gaussian_sequence_sh0_contract.json)，包含全部 194 帧 source/output 路径、SHA256、vertex count、schema/zero-SH/finite/bitwise equality/loader 检查结果及 T4 manifest hash。
- 独立工具：`tools/deform360_adapter/convert_gaussian_sequence_sh0.py`。先预检完整窗口，再创建独占输出；非零高阶 SH、schema/dtype/order 差异、非有限值、损坏文件或已有输出均报错停止，不静默覆盖/续转。执行副本为服务器 `/tmp/tcgs_t16_5_convert_20260926.py`，hash 与 Mac 工具一致并记入 contract。
- Slurm job `25821`：`srun -p 5090 --gres=gpu:1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:10:00 --job-name=tcgs-t16-5 /data1/userdata/tcweng/miniconda3/envs/soma/bin/python -B /tmp/tcgs_t16_5_convert_20260926.py --workspace /data1/userdata/tcweng/projects/tcgs --output /data1/userdata/tcweng/projects/tcgs/datasets/deform360/derived/008-pink-cloth/episode_0/t16_5_sh0 --contract /tmp/tcgs_t16_5_contract_20260926.json`，退出码 0。nvidia-smi、torch.cuda.is_available()、GaussianModel(0) 均在 allocation 内执行；RTX 5090，PyTorch 2.7.1+cu128。
- 每帧 loader tensor：xyz=[N,3]，DC=[N,1,3]，rest=[N,0,3]，opacity=[N,1]，scale=[N,3]，rotation=[N,4]；全部 float32/CUDA/finite。N 对应该帧的原始 vertex count。
- frame 113 派生 SHA256=`fd79797c7b7da6ba5db02a4de5fe3c5e463cb7db1eabbe6a6a57bc283e51f7de`，与 T2 派生文件完全相同。loader SHA256=`546ba41528a879af4d83bd280a3ef3cbcca9a5e9132204185aaa8f8ee9ffa310`，未修改。
- 服务器日志：`/tmp/tcgs_t16_5_20260926.log`；完整验证与逐帧 hash 已持久记录于上述仓库 contract，日志不是唯一证据。
- T16 checkpoint `a15db49de10fee8b1f50f307ae6fd53a11d5ce1c` 已 push；开始本任务前 Mac ahead/behind=0/0 且 clean，服务器 clean 后 fast-forward 到同一 HEAD。T16.5 修改属于 SoMA / deform360-adaptation，尚未 commit/push；服务器执行工具在 /tmp，checkout 仍 clean。小型文件待确认后通过 Git 同步；PLY 永远 server-only。

### T17：只组装 SoMA scene metadata 与目录契约

Status: PASS

**问题：** 独立组件还需要满足 loader 的文件名、目录和字段要求。

**为什么现在解决：** 所有组件已经各自验证。

**输入文件：**

- T2、T4–T16 的已批准产物；各实际输出路径以相应 TODO 的记录为准。

**预计涉及文件：**

- `SoMA/tools/deform360_adapter/`：scene packager，具体文件名尚未确定。
- `SoMA/mmgs/datasets/embodied_dataset.py`：路径和 metadata 读取。

**最小修改：** 组装 canonical PLY 路径，以及文件名为 `track_process_data.pkl`、`calibrate.pkl` 的产物、metadata、scene_info、mask 标签；这些派生产物的输出目录尚未确定，不预设具体路径。

**验证方法：** 仅执行静态 schema/path 检查，不实例化模型。

**PASS：** 所有必需文件可定位，camera 顺序一致，`WH=[640,360]`，标签恰好匹配一个 object。

**FAIL 后检查：** 硬编码路径、mask_info 命名、序列长度或配置字段；不改已验证的内容转换。

#### Result

2026-09-26：PASS（仅静态 scene packaging / interface contract，不是 dataset/model 执行或训练 PASS）。服务器生成 Config A/B 两个轻量 scene，固定顺序分别为 023_cam0、009_cam1 和 023_cam0、009_cam1、014_cam1。

Package 根目录：`datasets/deform360/derived/t17_scene_packages/pink_cloth_episode_0/`；入口为 `config_a_2cam/`、`config_b_3cam/`。两者的 color/<index> 链接 T14 canonical RGB，mask/<index>/1 链接 T15 canonical mask；各 mask_info_<index>.json 仅含 {"1":"cloth"}，不伪造 robot/obstacle mask。

共享 `shared/track_process_data.pkl` 仅封装 T11 controller_points；两入口链接该文件，float32 [194,30,3] 逐值/字节不变。每入口 calibrate.pkl 为 T8 c2w 的 [N,4,4] 数组，metadata.json 的 intrinsics 来自 T9，WH=[640,360]、frame_num=194。固定 pi3/gs/point_cloud/iteration_10000/point_cloud.ply 链接 T16.5 source 113 SH0，没有 future reconstructed PLY 路径或 rollout reset。

scene_info.json 只保存真实配置读取的 gravity_rot_quat=[0,0,0,1]；没有虚构平面尺寸/法向。scene_interface.json 记录 T12 controller_cfg、T13 raw gravity=[0,0,-9.8]、comp_dt=1/30、real_dt=1/15，dataset 只乘一次 ×4 得 [0,0,-39.2]。此为用户批准的工程约定，不宣称真实物理竖直。

训练 split_list 为 [[0,155]]；连续评估入口为 [[0,194]]，另记录 warm-up [0,155)、评分 [155,194)。该接口文件不是可执行训练配置，评分逻辑需后续集成；不得直接使用 [[155,194]] 作为测试重置段。T6 Stage 1/2 dataset gap=10/1 原样记录。未生成 cluster/cache/graph，未实例化 dataset/Camera/model，未运行 CUDA、render、forward 或训练；未执行 T18。

#### Evidence

- 新工具：`tools/deform360_adapter/package_scene.py`。新 contract：[scene_package_contract.json](contracts/008-pink-cloth/episode_0/scene_package_contract.json)，包含 A/B metadata 与接口定义、13 份输入 contract hash、生成 PKL/JSON hash、全部符号链接映射、controller/initial Gaussian provenance 和验证范围。
- 实际官方 sample：`datasets/soma_sample/soma_data_sample/cloth_lift/left_lift_1/`；只读检查 metadata.json、scene_info.json、mask/mask_info_0.json、calibrate.pkl（[3,4,4] ndarray）、track_process_data.pkl（仅 controller_points）。未复制官方 sample 的 calibration/gravity 或多类 mask 标签。
- 源码依据：`mmgs/datasets/embodied_dataset.py` readEmbodiedCameras 第 211–260 行按数字 camera 索引读取 color/<index>/<start>.png，c2w 由 extract_extrinsics 第 184 行求逆；_parse_dataset 第 384–396 行使用 joblib.load(calibrate.pkl)；_read_camera_transforms_embodied 第 986–1024 行读取 intrinsics/WH/frame_num。resolution 使用 [H,W]=[360,640]，防止 static image loader 第 46–68 行二次错误缩放。
- Initial Gaussian 路径依据 embodied_dataset.py 第 513–518 行；controller PKL 依据第 611–626、918–919 行。T12 grouping scheme 原样写入接口 controller_cfg，未执行聚类。scene_info gravity_rot_quat 由 cloth_lift_stage1.py 配置读取。
- mask 接口依据 `mmgs/datasets/utils/io.py` 第 367–456 行。其 mask_dir.replace("/<camera-index>", ...) 会替换所有匹配片段；因此 package 父路径避免 `/008-pink-cloth/` 中 `/0` 冲突，底层 canonical 资产仍在原目录。已对 A/B 每相机执行同一字符串构造并核验实际 JSON 路径和唯一 object 标签。
- 静态验证：582 RGB + 582 mask 的文件 SHA256 与 T14/T15 manifest 一致；全为 640×360，RGB mode=RGB、mask mode=L；两个入口所有 194 local paths 可解析。controller NPY/file/array hash 与 T11 一致；PKL 重读仅含 controller_points，shape/dtype/bytes 不变。calibrate.pkl 经 joblib 重读与 T8 数组逐值相等；所有 JSON 重读一致；T4/T5 映射边界一致；initial SH0 hash 与 T16.5 一致。
- 共新增 14 个实际文件、14 个符号链接，实际文件合计 78,122 bytes。RGB/mask/Gaussian 不复制，controller 仅增加一份 loader 必需的 PKL 封装。大资产和 PKL 留服务器，定义及 hash 由 Git-managed contract/tool 追踪。
- 执行：服务器 soma Python CPU-only 运行 `/tmp/tcgs_t17_package_scene_20260926.py --workspace /data1/userdata/tcweng/projects/tcgs --output /data1/userdata/tcweng/projects/tcgs/datasets/deform360/derived/t17_scene_packages/pink_cloth_episode_0 --contract /tmp/tcgs_t17_scene_contract_20260926.json`，退出码 0。未初始化 CUDA，无 Slurm GPU 作业。
- workspace skill checkpoint `cd45c7aedcb251c681d177e0a73d1834e542da8a` 已 push，root main clean/0–0；SoMA 本轮开始 HEAD=496973fc5e524b55ee6eee3b2ff8b80d60d40fc1，Mac clean 且与 origin 一致，服务器 clean 后 fast-forward 到同一 HEAD。T17 修改所属 SoMA/deform360-adaptation：roadmap 已 tracked，tool/contract 当前未跟踪，尚未 commit/push；后续小型文件需 Git 同步。

---

## 第三阶段：从单 sample 到极小 rollout

### T18：验证初始静态渲染的几何对齐

Status: PASS

**问题：** PLY、camera、RGB/mask 是否真正处在同一几何系统？

**为什么现在解决：** 应在动力学训练前排除投影错误。

**输入文件：**

- T17 组装的派生 scene 的 local frame 0；实际路径以 T17 记录为准。

**预计涉及文件：**

- `SoMA/gaussian-splatting/scene/gaussian_model.py`：Gaussian 加载。
- `SoMA/mmgs/datasets/utils/cameras.py`：相机投影。
- `SoMA/mmgs/models/utils/render.py`：静态渲染。

**最小修改：** 仅增加静态渲染检查入口，不构建动力学 rollout。

**验证方法：** 两相机初始 render 叠图；独立数值投影对照。

**PASS：** 投影误差符合 T9 约定，初始物体无镜像、轴翻转或明显整体偏移；输出有限。

**FAIL 后检查：** 按 camera convention→单位→frame 选择顺序定位；不通过训练补偿错位。

#### Result

2026-09-26：PASS。仅表示 T17 package 的 initial SH0 Gaussian、camera、RGB/mask 在 source 113/local 0 的几何对齐可进入后续 dataset/model smoke test；不表示 dynamics、forward prediction 或 training 正确。未执行 T19。

Config A/B camera identity/order 均核验；共享相机的 K/c2w 和三种输入文件 hash 完全一致，实际渲染三个唯一相机。全部 12861 centers 为正深度且位于 640×360 图像范围；SoMA 相机投影对独立 float64 K·w2c 的最大误差分别为 0.0000926、0.0000779、0.0001118 px，满足 T9 1 px 容差；render 输出全部 finite。

助手实际查看 RGB/render/mask 叠图：未见镜像、整体轴翻转或明显整体平移。alpha≥0.5 与 object mask IoU：023=0.9767、009=0.8286、014=0.8920。009 的主要差异位于夹爪遮挡侧，render 在 mask 排除区域仍有布面；质心差约 (-19.70,-4.10) px 不可直接解释为整体相机平移。014 有局部遮挡缺口差异。此为局部 visibility/reconstruction/mask 差异的可能解释，未独立证明全部成因；T16 无 robot obstacle mask 的 limitation 保留，不宣称遮挡问题已解决。

#### Evidence

- Review：[README](visualizations/t18-static-render-review-20260926/README.md)、[comparison_grid.jpg](visualizations/t18-static-render-review-20260926/comparison_grid.jpg)、[projection_diagnostics.json](visualizations/t18-static-render-review-20260926/projection_diagnostics.json)。目录还含每相机 comparison.jpg、render.png 和 projection.jpg；仅少量 review 图同步 Mac，无完整 sequence。
- 独立入口 `tools/deform360_adapter/review_static_render.py`，未修改 loader/Camera/renderer。读取 T17 package，逐项核验 PKL/JSON hashes、links、T8/T9 camera geometry、initial PLY。结果输入 RGB/mask hash 再与 T14/T15 local0 manifest 核验；下载图像 hashes 与服务器输出一致。
- 服务器 Slurm job 25822，命令：`srun -p 5090 --gres=gpu:1 --ntasks=1 --cpus-per-task=2 --mem=8G --time=00:05:00 --job-name=tcgs-t18 /data1/userdata/tcweng/miniconda3/envs/soma/bin/python -B /tmp/tcgs_t18_review.py --workspace /data1/userdata/tcweng/projects/tcgs --output /data1/userdata/tcweng/projects/tcgs/datasets/deform360/derived/t18_static_render_review_20260926`，退出码 0。CUDA/nvidia-smi 全部在 allocation 内；RTX 5090，torch.no_grad()，无 backward/训练。
- 原始实现：gaussian-splatting/scene/gaussian_model.py GaussianModel(0).load_ply；mmgs/datasets/utils/cameras.py Camera/getWorld2View2/getProjectionMatrix；mmgs/models/utils/render.py render_gaussian_physdreamer，黑背景、antialiasing=True。第二次 white override 静态 render 用作 alpha footprint 诊断，不改变 Gaussian 属性。
- 独立 NumPy float64 world→camera→K 投影，与真实 Camera float32 full_proj_transform 和 ndc2Pix 像素中心公式对比全部点；无“试翻轴”修正。水平/垂直镜像 IoU 仅为对照，全部低于原位 IoU；alpha 阈值 0.1/0.5/0.9 的完整统计已记录。
- T17 checkpoint df9d0cd20bb3d14be6155e15cd5237becb1df782 已 push；开始前 Mac clean、ahead/behind=0/0，服务器 clean 后 fast-forward。T18 文件所属 SoMA/deform360-adaptation；roadmap tracked，新增工具/review 当前 untracked，尚未 commit/push。服务器工具在 /tmp，checkout 未修改；后续轻量内容待通过 Git 同步。

### T19：让 EmbodiedDataset 读取一个 sample

Status: PASS

**问题：** 完整 scene 是否满足实际 dataset contract？

**为什么现在解决：** 静态文件验证不能代替真实 loader。

**输入文件：**

- T17 组装的派生 scene；实际路径以 T17 记录为准。
- 已确定的时间与 split 参数（T5、T6）。

**预计涉及文件：**

- `SoMA/configs/SoMA/deform360_v0_stage1.py`（拟新增）：dataset 部分。
- `SoMA/mmgs/datasets/embodied_dataset.py`：实际 dataset 读取。

**最小修改：** 仅串联 dataset 配置，读取一个 sample；cache 只能写派生 scene。

**验证方法：** 检查 images、GT、controller、camera、gravity、`gs_aligned_frame`、`seq_num`。

**PASS：** 首状态为 local 0；gap=10 的训练 sample 对应 source 113、123…263；shape 与 camera 顺序正确。

**FAIL 后检查：** 配置继承、`resolution` 的 H/W、split、路径和缓存键。

#### Result

2026-09-26：PASS。以真实 mmgs.datasets.embodied_dataset.EmbodiedDataset(env_cfg=..., phase="train") 分别实例化 A/B 并读取 dataset[0]，没有模拟 loader 或修改源码。两者 dataset length=1、seq_idx=0；scene_name 分别为 config_a_2cam/config_b_3cam。

采样 local=[0,10,…,150]，source=[113,123,133,143,153,163,173,183,193,203,213,223,233,243,253,263]，16 帧；训练 split=[[0,155]]，无 source≥268 监督。gs_aligned_frame=0，初态固定 source113 SH0。

A/B camera 数为 2/3，实际顺序为 023_cam0、009_cam1（B 追加 014_cam1）。每相机 img=[3,360,640]，full_gt_label/gt_label=[16,3,360,640]，controller_img_mask/pure_robot_img_mask=[16,1,360,640]；两类 obstacle mask 均为零，遵循 T16，而非实测无遮挡。每个 sampled RGB 和 masked GT 均与对应 canonical frame 逐值核验。

controller_trajectory=[16,30,3] float32，与 T11 canonical 7mm trajectory 的相同索引逐值相等，点序不变。返回的 controller p2c 每层逐项等于 T12，层级 30→10→2→1，无回退。raw gravity=[0,0,-9.8]，实际 external=[[0,0,-39.20000076293945]]（float32），只执行一次 ×4。volume_scalar=[[512]]。所有 sample tensor/array 与 camera 数值均 finite。

seq_num=[10] 是源码定义的 video_range[1]−video_range[0]（采样帧步长），不是 16 帧的计数。实际训练 loader 会随机化 camera batching；本次在每次 constructor 前固定 NumPy seed=5，使 2/3 camera permutation 均为 identity，并验证实际返回顺序。未修改/禁用随机逻辑，不保证后续未固定种子的训练批次始终同序。

无需 loader 修改。仅串联 dataset smoke 参数并允许 dataset 自身在派生 scene 中生成 object/controller 聚类缓存；未构建 graph、未实例化 dynamics 模型、未 render/forward/backward/训练，未执行 T20。

#### Evidence

- 入口：`tools/deform360_adapter/check_dataset_sample.py`；完整实际 sample summary、env_cfg、shape/dtype/finite、frame/camera IDs、cache hashes 见 [dataset_sample_contract.json](contracts/008-pink-cloth/episode_0/dataset_sample_contract.json)。
- 基于 T17 scene_interface 的 env_cfg，仅补训练 split、frame_gap=10，以及官方 cloth_lift_stage1.py 的 object cluster scheme [downsample_rate=0.02,0.2]、volume_scalar=512；这些是本次 dataset-only 检查参数，不将其冒充新批准的完整 training config。T12 controller_cfg 原样保持，未新建或修改训练配置。
- Slurm job 25823，RTX 5090；命令：`srun -p 5090 --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:10:00 --job-name=tcgs-t19 /data1/userdata/tcweng/miniconda3/envs/soma/bin/python -B /tmp/tcgs_t19_dataset.py --workspace /data1/userdata/tcweng/projects/tcgs --report /tmp/tcgs_t19_dataset_report.json`；退出码 0。CUDA/nvidia-smi 在 allocation 内执行，sample 读取用 torch.no_grad()。
- 原始 loader 文件路径与导入类所属 module.__file__ 一致，SHA256 执行前后不变。embodied_dataset.py 第 351–352 行调用内部聚类/合并映射；第 358–366 行训练相机随机顺序；第 826、918–934 行采样/controller/gravity；第 965 行 seq_num。未 mock/monkeypatch。
- Cache 仅在 A/B 派生 scene 的 cluster_mask/dis_split 和 controller_mask/dis_split 中各新增一个 PKL：object key `dis_splitnl_2_downsample_rate_0d02_downsample_rate_0d2_cluster_mask.pkl`；controller key `controller_dis_splitnl_2_nc_10_downsample_rate_0d5_track_process_data.pkl`。完整路径/hash 在 contract；server-only，未加入 Git。它们是 dataset 层级映射，不是 T20 graph construction。
- T17 所有原生成文件 hash 执行前后不变，initial SH0/trajectory hash 保持；sample controller 与 canonical[0:155:10] 严格一致；每个采样 RGB 和 mask GT 对照真实 PNG 严格相等。对所有嵌套 tensor/array 和 Camera matrices 检查 finite。
- T18 checkpoint 7030f215a2fec0afd3cc576315d1cc07bccb8d77 已 push；执行前 Mac clean、ahead/behind=0/0，服务器 clean 后 fast-forward。T19 修改所属 SoMA/deform360-adaptation：roadmap tracked，新增检查工具及 sample contract untracked；暂未 commit/push。服务器 checkout 仍 clean，小型文件待之后 Git 同步。

### T20：只验证 graph construction

Status: PASS

**问题：** object/controller 层级能否形成合法图？

**为什么现在解决：** 将图问题与模型 forward 问题分开。

**输入文件：**

- T19 实际读取的 sample。

**预计涉及文件：**

- `SoMA/mmgs/models/simulators/gs_simulator_embodied.py`：`_preprocess`。
- `SoMA/mmgs/models/utils/dgl_graph.py`：graph construction。

**最小修改：** 仅串联预处理并记录各层 node/edge/mapping。

**验证方法：** 检查索引范围、controller pin、特征有限性和 cluster 覆盖。

**PASS：** 无非法索引/NaN，全部 Gaussian 都有合法层级映射，controller 分组与 T12 一致。

**FAIL 后检查：** 单位、cluster 阈值、mapping 和 cache；不调整模型层数或显存策略。

#### Result

2026-09-26：PASS。真实 Config A EmbodiedDataset sample → GsSimulatorEmbodied._preprocess → GsHieEmbodiedDGLProcessor。四层 object/controller/edge 数分别为 12861/30/0、640/10/9768、7/2/72、1/1/2；层级连接图另记录在 report.json。leaf spatial edge=0 是官方 forward_last_layer=False 路径，并非整图无边。controller 实际层级严格为 30→10→2→1。

逐层检查 pin mask、node/edge index 范围、p2c shape/全覆盖、所有特征 finite；leaf object/point order 与 initial Gaussian 一致，controller 前30行与目标帧控制点一致。没有非法索引、NaN/Inf。

#### Evidence

- 本次用户明确授权 T20→T21→T22→T23 严格串行 batch；前项 PASS 后才进入本项。Slurm job 25824，RTX5090 32607MiB；本项耗时 7.656s，peak allocated=277390336 bytes（0.258GiB），peak reserved=304087040 bytes。T20耗时含初始化，其余为阶段墙钟时间，非性能benchmark。
- [完整report](validation/t20-t23-smoke-20260926/report.json) 的 stages.T20 独立记录结果；[说明](validation/t20-t23-smoke-20260926/README.md)、[原始日志](validation/t20-t23-smoke-20260926/run.txt)。独立工具 tools/deform360_adapter/smoke_dynamics_batch.py；原始SoMA源码hash执行前后不变。
- 固定Config A（023_cam0、009_cam1），T19训练sample source113…263、gap10，T11 7mm轨迹逐值核验，T12 grouping、T13 external=[0,0,-39.2]、T14/T15资产不变。官方模型结构、loss与dt保留，完整有效配置在report；无未来PLY读取入口。
- T19 checkpoint 2cd8dd6cfafe329499930c4c6bf52763beb796ab 已push，batch开始前Mac clean/0–0，服务器clean后fast-forward。新增工具/小型报告属于SoMA/deform360-adaptation，当前untracked；roadmap tracked修改。batch尚未commit/push；未执行T24、未保存checkpoint或正式训练。

### T21：只运行一次 forward/render/loss

Status: PASS

**问题：** 一个预测步是否贯通？

**为什么现在解决：** 尚未验证目标帧选择与损失计算。

**输入文件：**

- T20 构建的 graph。
- T19 sample 中对应 next-frame GT。

**预计涉及文件：**

- `SoMA/mmgs/models/simulators/gs_simulator_embodied.py`：单步前向。
- `SoMA/mmgs/models/heads/acc_decoder.py`：decode head。
- `SoMA/configs/SoMA/deform360_v0_stage1.py`（拟新增）：单步 smoke 设置。

**最小修改：** 单步前向验证入口，不 backward、不更新权重。

**验证方法：** 记录预测 shape、目标 source frame、各 loss 和 render。

**PASS：** 预测点数固定，所有输出/loss 有限；第一个 target 是 source 123，而不是 initial 113。

**FAIL 后检查：** controller future/current 索引、GT 索引、render 参数和 loss 输入。

#### Result

2026-09-26：PASS。直接在 T20 的 graph 上调用真实 encode_decode 和 _encode_decode_train，无 backward。initial source113/local0，controller prev/current=113/123，GT target=source123/local10，不是113自监督。12861个object Gaussian输出保持[12861,3]，两camera render均[3,360,640]，位置/covariance/render/depth/各loss均finite。

Total loss=59908.2734375；完整分项见 report.json。未更新权重；但原实现训练路径/ register_norm=True会更新normalizer统计，因此与后续loss不构成学习改善对照。

#### Evidence

- 本次用户明确授权 T20→T21→T22→T23 严格串行 batch；前项 PASS 后才进入本项。Slurm job 25824，RTX5090 32607MiB；本项耗时 0.469s，peak allocated=1489330688 bytes（1.387GiB），peak reserved=1593835520 bytes。T20耗时含初始化，其余为阶段墙钟时间，非性能benchmark。
- [完整report](validation/t20-t23-smoke-20260926/report.json) 的 stages.T21 独立记录结果；[说明](validation/t20-t23-smoke-20260926/README.md)、[原始日志](validation/t20-t23-smoke-20260926/run.txt)。独立工具 tools/deform360_adapter/smoke_dynamics_batch.py；原始SoMA源码hash执行前后不变。
- 固定Config A（023_cam0、009_cam1），T19训练sample source113…263、gap10，T11 7mm轨迹逐值核验，T12 grouping、T13 external=[0,0,-39.2]、T14/T15资产不变。官方模型结构、loss与dt保留，完整有效配置在report；无未来PLY读取入口。
- T19 checkpoint 2cd8dd6cfafe329499930c4c6bf52763beb796ab 已push，batch开始前Mac clean/0–0，服务器clean后fast-forward。新增工具/小型报告属于SoMA/deform360-adaptation，当前untracked；roadmap tracked修改。batch尚未commit/push；未执行T24、未保存checkpoint或正式训练。

### T22：只验证一次 backward 与 optimizer step

Status: PASS

**问题：** 损失是否能正确更新动力学参数？

**为什么现在解决：** forward 成功不代表训练成立。

**输入文件：**

- T21 验证过的同一个 sample。

**预计涉及文件：**

- `SoMA/configs/SoMA/deform360_v0_stage1.py`（拟新增）：单步训练入口。
- 现有 optimizer hook；运行环境依赖中的具体文件路径执行时定位，不预设 workspace 内的新文件。

**最小修改：** 只增加一次 backward/step。

**验证方法：** 检查梯度有限性、预期参数的梯度与更新前后差异。

**PASS：** backward/step 成功，目标模型参数得到非零有限更新，数据文件不变。

**FAIL 后检查：** detach、参数注册、loss 梯度路径和 optimizer 参数集合。

#### Result

2026-09-26：PASS。同一sample/config重新构图，执行一次forward、backward、Adam.step。total loss=23067.3828125；梯度global norm=81870.2374，275个参数有非零有限梯度，275个参数发生有限更新，包含backbone。optimizer参数集合与requires_grad模型参数精确一致，scene Gaussian输入不在optimizer中；Gaussian tensors/源数据hash保持不变。

两项encoder.emb_norm.weight/bias没有梯度，源码只定义未在该路径调用，完整missing列表已记录；不宣称所有注册参数都获梯度，也不擅自修改。Adam lr=0.0004沿用官方配置，此处仅证明连通性。

#### Evidence

- 本次用户明确授权 T20→T21→T22→T23 严格串行 batch；前项 PASS 后才进入本项。Slurm job 25824，RTX5090 32607MiB；本项耗时 0.762s，peak allocated=1492045312 bytes（1.390GiB），peak reserved=1610612736 bytes。T20耗时含初始化，其余为阶段墙钟时间，非性能benchmark。
- [完整report](validation/t20-t23-smoke-20260926/report.json) 的 stages.T22 独立记录结果；[说明](validation/t20-t23-smoke-20260926/README.md)、[原始日志](validation/t20-t23-smoke-20260926/run.txt)。独立工具 tools/deform360_adapter/smoke_dynamics_batch.py；原始SoMA源码hash执行前后不变。
- 固定Config A（023_cam0、009_cam1），T19训练sample source113…263、gap10，T11 7mm轨迹逐值核验，T12 grouping、T13 external=[0,0,-39.2]、T14/T15资产不变。官方模型结构、loss与dt保留，完整有效配置在report；无未来PLY读取入口。
- T19 checkpoint 2cd8dd6cfafe329499930c4c6bf52763beb796ab 已push，batch开始前Mac clean/0–0，服务器clean后fast-forward。新增工具/小型报告属于SoMA/deform360-adaptation，当前untracked；roadmap tracked修改。batch尚未commit/push；未执行T24、未保存checkpoint或正式训练。

### T23：只验证 rollout=3

Status: PASS

**问题：** 自回归状态和多步监督是否连续正确？

**为什么现在解决：** 单步检查无法发现跨步索引问题。

**输入文件：**

- 已验证 sample。
- T22 配置：`SoMA/configs/SoMA/deform360_v0_stage1.py`。

**预计涉及文件：**

- `SoMA/configs/SoMA/deform360_v0_stage1.py`（拟新增）：rollout 上限。
- `SoMA/mmgs/models/simulators/gs_simulator_embodied.py`：多步状态传递。

**最小修改：** 只将 rollout 从 1 改为 3。

**验证方法：** 逐步记录 source target 123、133、143，检查前一步预测如何进入下一步。

**PASS：** 三步 forward/backward 有限、Gaussian identity 不变、状态传递与既有 detach 规则一致。

**FAIL 后检查：** 状态更新、目标索引、共享图或实际报错位置；OOM 时停止，不自动优化显存或改其他变量。

#### Result

2026-09-26：PASS。使用T22更新后的同一模型实例，真实forward_train rollout=3；initial113，依次target123、133、143（local10/20/30）。三步forward与汇总loss backward成功，无额外optimizer step。total loss=18705.154296875，梯度global norm=136866.0205；所有逐步输出/loss/gradient有限，无OOM。

观察包装逐值核验：step2/3 cur_state来自上一step的pred_pos[30:]，prev_state来自前一步输入；controller和GT使用对应帧，未从未来重建PLY重置。12861点数及结构行序保持。保留官方accumulate_gradient=False的detach规则（不是完整跨步BPTT），并保留源码当前cur_cov不随pred_cov更新的行为；未改变算法语义。

#### Evidence

- 本次用户明确授权 T20→T21→T22→T23 严格串行 batch；前项 PASS 后才进入本项。Slurm job 25824，RTX5090 32607MiB；本项耗时 0.758s，peak allocated=3892000768 bytes（3.625GiB），peak reserved=4221566976 bytes。T20耗时含初始化，其余为阶段墙钟时间，非性能benchmark。
- [完整report](validation/t20-t23-smoke-20260926/report.json) 的 stages.T23 独立记录结果；[说明](validation/t20-t23-smoke-20260926/README.md)、[原始日志](validation/t20-t23-smoke-20260926/run.txt)。独立工具 tools/deform360_adapter/smoke_dynamics_batch.py；原始SoMA源码hash执行前后不变。
- 固定Config A（023_cam0、009_cam1），T19训练sample source113…263、gap10，T11 7mm轨迹逐值核验，T12 grouping、T13 external=[0,0,-39.2]、T14/T15资产不变。官方模型结构、loss与dt保留，完整有效配置在report；无未来PLY读取入口。
- T19 checkpoint 2cd8dd6cfafe329499930c4c6bf52763beb796ab 已push，batch开始前Mac clean/0–0，服务器clean后fast-forward。新增工具/小型报告属于SoMA/deform360-adaptation，当前untracked；roadmap tracked修改。batch尚未commit/push；未执行T24、未保存checkpoint或正式训练。

### T24：验证 checkpoint 保存与恢复

Status: PASS

**问题：** 训练状态能否可靠恢复？

**为什么现在解决：** 正式训练前需要可复查、可继续的运行基础。

**输入文件：**

- T23 的短运行状态；保存位置以 T23 运行记录为准。

**预计涉及文件：**

- 现有 runner/checkpoint 实现；运行环境依赖中的具体文件路径执行时定位，不预设 workspace 内的新文件。

**最小修改：** 只增加 save/reload 验证，不继续训练。

**验证方法（用户批准的最终范围）：** 通过真实 runner save/resume 比较 model state、optimizer/Adam buffers、12 个 normalizer 统计 tensor、runner epoch/iter，检查保存/恢复 tensor finite 与 Gaussian/source assets 未被 optimizer 修改。固定输入输出比较作为 numerical reproducibility 诊断证据保留，不再作为 checkpoint integrity 的硬性 gate。

**PASS（最终验收）：** PASS — checkpoint state recovery。真实 runner checkpoint save/resume 成功；model state、optimizer state/Adam buffers、normalizer 12 个统计 tensor exact restore；runner epoch/iter 正确恢复；saved/restored tensors finite；Gaussian/source assets 未被 optimizer 修改。

**验收范围修正说明：** 用户明确批准结束数值重复性追查。原判据“需要恢复的状态一致，预测在预先约定的数值容差内一致”保留为历史验收依据；其下所有 FAIL/比较数值仍有效。本次不改算法或 atol/rtol=1e-5，而是将 fixed-output <1e-5 从 checkpoint integrity 的硬性 PASS 条件中移除。此 PASS 不追认非有限值的 rollout=3 checkpoint 为有效，也不表示所有 Python 属性都被序列化。

**FAIL 后检查：** checkpoint 缺项、scene 初始化、normalizer 状态和随机性。

#### Result

2026-09-26：BLOCKED（前置状态未持久化，尚未执行 checkpoint save/reload）。T23 使用的模型更新、optimizer 和 normalizer/runtime 状态仅存在于已退出的 Slurm 进程，未保存 checkpoint。现有 JSON/report 和日志不能恢复该状态，因此不能对原 T23 状态进行逐项恢复验证，也不能以重新随机初始化的模型冒充恢复成功。

遵守本次串行 gate，停止整个 batch；T25/T26/T27 未执行，保持 TODO，未执行 T28。未启动新的 CUDA 作业、训练或修改源码。最小后续方案是另行授权重建一次相同 canonical setup 的短运行状态，在同一进程退出前通过真实 runner 保存，再独立 reload 比较参数、optimizer、计数器、normalizer/runtime 与固定输入输出；必须明确这是新一次短运行，而不是恢复 job 25824 的丢失状态。

2026-09-26 后续执行：FAIL。用户授权重建新短运行（不是恢复 job 25824）。Slurm 25826 / RTX 5090 使用 seed=5、Config A、canonical geometry/grouping/gravity，真实 EpochRunner + OptimizerHook + CheckpointHook 完成 1 epoch / 1 batch / 1 optimizer step，rollout=3，targets=123/133/143。原 runner 写出 epoch_1.pth 后，梯度 finite 断言失败；独立只读检查发现 275 个 model tensor 和 550 个 optimizer buffer 含 NaN/Inf。立即停止，未运行新进程 resume 或 fixed-input comparison，未执行 T25–T28。

该失败文件不能作为有效 checkpoint。此新运行直接从初始化状态做 rollout=3，没有先重复旧 T21/T22 的 normalizer 累积及参数更新；这是否导致非有限值尚未验证，不将其写成已确认原因。未修改核心源码或修复后重跑。旧 BLOCKED 历史保留。

2026-09-26 finite retry：FAIL（本次失败点是输出容差，不是 NaN/Inf）。保留 previous FAIL：random-init rollout=3 optimizer step produced nonfinite gradients/state；current retry deliberately uses previously validated T22-style rollout=1 finite step to isolate checkpoint semantics。先复用一次 T21-style no-update forward，再经真实 EpochRunner/model.train_step 做一次 rollout=1 更新（source113→123）。验证工具在 Adam.step 前确认 gradients finite，step 后确认 model/Adam/normalizer finite，随后原 CheckpointHook 保存。Job 25837 保存成功；新进程 job 25838 通过真实 resume 恢复。model state、optimizer groups/IDs/name order/buffers/step、12 个 normalizer 统计 tensor、runner epoch=1/iter=1 均逐项一致。

固定输入 eval/no_grad 比较确认没有更新 model/normalizer/optimizer。只读量化 job 25839：pred_pos max diff=5.960464477539063e-08；pred_cov=4.3655745685100555e-11；023 render=0.00018256902694702148；009 render=8.32974910736084e-05；momentum loss=0.00011563301086425781。两个 render 和 momentum loss 未通过预设 atol=rtol=1e-5，所有输出 finite。原因尚未确定；未放宽容差、未修改 checkpoint、未继续训练；未分析旧 rollout=3 稳定性，未执行 T25–T28。

Runtime 边界：global_step 在当前配置中不存在；iters_per_epoch=None/is_est_vel=True 一致。model.num_iter/num_epoch 的每次调用镜像未序列化（1→0），真实进度由已恢复 runner counters 管理，forward_train:638 在使用前重写镜像；不声称所有 Python 属性均恢复。工具 job 25835 因 batch_processor 接口限制在训练前失败，25836 在有限更新后因读取不存在的 global_step 失败，均未保存 checkpoint；报告保留，未复用其状态。

2026-09-26 determinism baseline audit：Case 2，T24 保持 FAIL。两个完全独立的新进程 A（Slurm 25840）/B（25841）分别 resume 同一 finite checkpoint，各执行一次 eval/no_grad fixed forward，零 optimizer step。A/B 的 pred_pos、pred_cov、023/009 render 以及所有 loss/metric 全部 bitwise equal，max/mean abs diff 均为 0；seed/config/input/model/optimizer/initial Gaussian/checkpoint hashes 及 backend flags 一致，且各自对保存 reference 的 model/optimizer/normalizer/counters exact 检查通过。

pre-save 与 A/B 仍有相同非零差异（render max 1.8256902694702148e-4 / 8.32974910736084e-5），不能据本次实验将 1e-4 归因为实测的 A/B numerical floor；原 atol=rtol=1e-5 不变。差异在 pred_pos/cov 阶段已存在，范围缩小为训练后进程与 fresh resume 的执行状态差异；尚未找到已证实遗漏的 runtime field，不能声称已证明 checkpoint 丢失参数。只读源码确认 num_iter/num_epoch 镜像不被当前 direct fixed-output 路径读取；已恢复 normalizer 统计在 eval 中不更新。后续定位需比较 graph/feature 中间量及未序列化执行状态，本轮未扩大实验。

2026-09-26 post-save reference audit：Case B，T24 保持 FAIL。新有限短状态 job 25842：一次 warmup forward → rollout=1 train forward → backward → gradient finite guard → Adam.step → model/Adam/normalizer finite guard → 原 CheckpointHook save → eval/no_grad reference S；事件顺序逐项记录。独立 job 25843/25844 resume 后各计算 A/B。结果 S≠A=B；S/A 和 S/B 的 pred_pos max=8.940696716308594e-08、pred_cov=3.637978807091713e-11、023 render=8.868575096130371e-4、009 render=5.335211753845215e-4、momentum loss=1.7082691192626953e-4。A/B 全输出 bitwise equal；原 atol=rtol=1e-5 不变。

S/A/B 的目标输入、参数/Normalizer/scene Gaussian 的值与布局、eval flags、render pipeline、backend flags 一致；每次 fixed forward 前后定向状态快照不变。只观察到 model.num_iter/num_epoch 调用镜像及 Python RNG 不同；镜像不被 direct one_step forward 读取，Python RNG 在 A/B 间也不同且所有 RNG 在 forward 中未推进，不能认定为原因。scene_init_* 指定候选属性在此实际配置中不存在。未找到具有因果证据的遗漏 mutable runtime field；未修改算法、未放宽判据、未盲查全部属性。

旧 R 原样保留。审阅旧工具确认 R 原本也在 checkpoint 保存后、eval/no_grad 中生成，“pre-save reference”此前措辞不精确。新旧短运行的 checkpoint model 并非全 exact，因此 R/S 比较不能单独证明 reference 捕获时机问题。四组完整 max/mean/bitwise/loss/metric 比较均已记录。未执行 T25–T28。

2026-09-26 same-process fresh instance audit：Case 3，T24 保持 FAIL。Job 25845（PID 1148289）沿既有 finite rollout=1 路径建立 M0，一次 warmup forward + 一次 guarded Adam update，真实 runner 保存 C；M0→S0，同进程新建 M1/optimizer/runner 并 resume C→S1，随后 M0→S0_repeat、M1→S1_repeat。两对象同时保留且 object ID 不同。进程退出后 job 25846（PID 1148551）独立 resume C→A。没有 reseed/empty_cache/backend 干预、逐层 instrumentation 或算法改动。

结果：S1 不等于 S0，也不等于 A，均未通过原 render tolerance；S0/S0_repeat 同样不逐位一致，而 S1/S1_repeat 全输出 bitwise equal。S0/S1 的 pred_pos max=2.9802322387695312e-08、023/009 render max=7.564425468444824e-4 / 1.3486146926879883e-3；S1/A 的 pred_pos max=8.940696716308594e-08、023/009 render max=7.56487250328064e-4 / 2.740606665611267e-3。全部输出 finite，atol=rtol=1e-5 不变。

每次推理前 model/optimizer/normalizer exact、checkpoint hash、runner epoch=1/iter=1、输入/camera/controller/Gaussian、eval flags、dtype/device/layout 与 forward 实参核验通过；每次推理前后定向快照不变。S0_repeat 按要求发生在 M1 构造和推理之后，不能将其差异直接解释为永久 object-local defect。当前不是简单 Case 1 或 Case 2；不再猜测 Python field 或 CUDA history。下一步应独立授权 first-divergence tracing，本轮未执行，也未执行 T25–T28。

2026-09-26 用户批准的最终收口：**PASS — checkpoint state recovery**。依据已有 finite rollout=1 证据：真实 runner save/resume 成功，model state exact、optimizer state/Adam buffers exact、normalizer 12 个统计 tensor exact、runner epoch/iter 正确恢复；保存/恢复 tensors finite；Gaussian/source assets 未被 optimizer 修改。本次仅修正文档验收范围，没有新 CUDA 实验。

**Known limitation / follow-up：numerical reproducibility。** 固定输出不保证 bitwise 或 1e-5 级跨运行一致性；同一原 model object 的 repeated forward 也观察到小的数值差异。fixed-output <1e-5 不再是 checkpoint integrity 的硬性 PASS 条件，原比较 tolerance 和所有实验数值不变。不宣称已确定差异来源，也不宣称差异绝对无影响。停止 Case 3 追查及 first-divergence tracing；历史中提出的追查建议不再是当前待执行任务。只有后续 resume training 出现系统性 trajectory / metric discontinuity 时再重新开启此问题。T24 完成后停止，等待 T25 授权。

#### Evidence

- [T20–T23 README](validation/t20-t23-smoke-20260926/README.md) 第 51 行明确记录未保存 checkpoint，optimizer 更新仅存在于已退出的进程内。
- [smoke tool](../../tools/deform360_adapter/smoke_dynamics_batch.py) 第 36 行声明无 checkpoint；第 39 行 `save()` 只写 JSON report；第 189 行执行 optimizer step，但工具没有模型/optimizer checkpoint 保存调用。
- 服务器 `sacct -j 25824 --format=JobID,State,ExitCode --noheader`：25824 及 25824.0 均为 `COMPLETED 0:0`。保留的 `/tmp/tcgs_t20_t23_report.json`（136867 bytes）及 `/tmp/tcgs_t20_t23.log`（60455 bytes）是诊断记录，不包含可恢复训练状态。
- T20–T23 checkpoint 已提交并 push：`92107be55dd5403eb591a3538d9d1d8742df6264`。开始本次检查前 Mac HEAD 与 origin 一致、ahead/behind=0/0、working tree clean；服务器从 `2cd8dd6` clean fast-forward 至同一提交。
- 本次仅修改 SoMA/deform360-adaptation 中本 roadmap（Git tracked），尚未 commit/push；该 BLOCKED 记录尚需后续同步至服务器。

- 新增 [T24 contract](contracts/008-pink-cloth/episode_0/checkpoint_recovery_contract.json)、[运行报告](validation/t24-checkpoint-20260926/save_report.json)、[失败 checkpoint 检查](validation/t24-checkpoint-20260926/checkpoint_inspection.json)、[日志](validation/t24-checkpoint-20260926/run.txt)。Slurm 25826 FAILED / exit 1:0 / elapsed 11 秒。
- Server-only checkpoint：`outputs/deform360/t24-recovery-20260926/epoch_1.pth`，30025021 bytes，SHA256 `57323ae9bceff44fa413e0f2d43c9aa5f1521984b49690558154131f3cea7333`；meta epoch=1 / iter=1，289 model keys、275 optimizer states。保存成功不表示数值有效或恢复成功。
- 76 个输入/源码文件 hash 核验无变化；Gaussian 在 optimizer 之外。checkpoint 未同步 Mac。新工具、contract、小型证据属于 SoMA/deform360-adaptation，未 commit/push；服务器 checkout clean，工具在 `/tmp` 执行；小型版本管理文件待后续 Git 同步。

- 新一轮 [finite recovery evidence](validation/t24-finite-recovery-20260926/README.md)：save/reload/差异 reports、reference manifest、日志及工具检查错误记录。原失败证据未修改。
- 当前 finite checkpoint：`outputs/deform360/t24-finite-recovery-v3-20260926/epoch_1.pth`，30024957 bytes；SHA256 `2df68bbe34e0a5f3f6c23e56d6ecd924eb47aa51466ff347045ee8b3f3ac730f`，仅 server-only。成功保存有限状态，但完整 T24 尚未 PASS。
- 小型工具/contract/roadmap/evidence 属于 SoMA/deform360-adaptation，未 commit/push，待后续 Git 同步；原 `t24-checkpoint-20260926.zip` 未操作。

- [A/B determinism audit](validation/t24-repeatability-20260926/README.md) 与 [完整 max/mean/bitwise 比较](validation/t24-repeatability-20260926/comparison.json)。两份输出 tensor 仅留服务器 `outputs/deform360/t24-repeatability-20260926/`。
- 新增独立 A/B audit 与 CPU comparison 工具；checkpoint/model/runner/旧验证 tolerance 均未修改，旧 BLOCKED/FAIL 历史保留。未 commit/push，未执行 T25–T28。

- [Post-save S/A/B event/state audit](validation/t24-postsave-reference-20260926/README.md)、[全部比较](validation/t24-postsave-reference-20260926/comparison.json)。新增工具 `audit_checkpoint_reference_state.py` / `compare_checkpoint_reference_state.py`，仅验证用途。
- 新 server-only checkpoint：`outputs/deform360/t24-postsave-reference-20260926/epoch_1.pth`；30024957 bytes；SHA256 `95e54bd6f97fd3508fb5e3dd75129a3b5a978a3dc28376c504c01e68b2e39723`。旧 checkpoint/reference 未改。
- SoMA/deform360-adaptation 小型工具/roadmap/contract/evidence 尚未 commit/push；新增内容待 Git 同步。既有 BLOCKED/FAIL 历史完整保留。

- [Same-process isolation report](validation/t24-same-process-20260926/README.md) 包含全部 5 组输出的 max/mean/bitwise/original-gate 数值与精确事件顺序；JSON reports/state snapshots/logs 同目录。
- 新 server-only C：`outputs/deform360/t24-same-process-20260926/epoch_1.pth`，30024957 bytes，SHA256 `421070e708ca8920de7b661a2708ec974ec4008c7469e80f7585e17a571a7979`。所有旧 checkpoint/reference 保留。
- 新增独立验证工具 `audit_checkpoint_same_process.py` / `compare_checkpoint_same_process.py`；SoMA/deform360-adaptation 的小型变更尚未 commit/push，待后续 Git 同步，服务器 checkout clean。历史 BLOCKED/FAIL 不改写。

- 最终 PASS 为用户批准的 **acceptance-scope revision**，不是新实验或旧 FAIL 的改写。依据 [finite save/resume evidence](validation/t24-finite-recovery-20260926/README.md)（jobs 25837/25838）及 [same-process exact-state checks](validation/t24-same-process-20260926/README.md)（jobs 25845/25846）。原 output-gate FAIL 与 BLOCKED / rollout=3 FAIL / repeatability audit 全部保留。
- [Checkpoint recovery contract](contracts/008-pink-cloth/episode_0/checkpoint_recovery_contract.json) 顶层 status=PASS，新增最终 PASS criterion、numerical reproducibility limitation 和条件性 follow-up；各历史 attempt/audit status 保持不变。本次仅修改这两个文档文件，SoMA/deform360-adaptation，未 commit/push；待后续 Git 同步。Current task 仍为 T24，T25 未执行。

---

### T24.5：数值重复性 first-divergence audit

Status: PARTIALLY_ATTRIBUTED

**Non-gating diagnostic。** T24 保持 **PASS — checkpoint state recovery**。T24.5 的 ATTRIBUTED / PARTIALLY_ATTRIBUTED / UNRESOLVED 分类均不影响 T25–T27 推进，不将 T25 标记 BLOCKED；下一 TODO 仍需单独授权。本项是 2026-09-26 用户新授权的独立诊断，不改写 T24 中当时结束追查的历史决定。

**问题 / motivation：** checkpoint state exact restore 后仍观察到数值重复性差异，最早在哪个真实 stage/module/native operation boundary 分叉？只定位，不修复。

**前置依赖与输入：** T24 已 PASS；现有有限 checkpoint `outputs/deform360/t24-same-process-20260926/epoch_1.pth`；T19–T23 canonical Config A；source 113→123 的相同 Gaussian/controller/camera/GT。

**预计涉及文件 / 最小修改：** 新增独立 `trace_numerical_divergence.py`、CPU 比较工具、小型诊断 contract / reports；只做观察性 wrappers，不修改 model/runner/checkpoint、normalizer、renderer、算法、precision/backend flags 或 tolerance。CUDA 全部通过 Slurm，不训练、不 optimizer.step。

**Experiment：** 先记录 `_preprocess`、backbone/message-passing encoder、decoder/raw deformation parameters、pred_pos/cov、render input/output 和 loss 等真实边界；仅在首次不同的 preprocessing 内做一次细追踪，直到 DGL native SpMM 边界。保存 shape/dtype/device/layout、finite、统计与 SHA256，并比较 exact/max/mean diff；记录四类 RNG 前后状态。历史训练对象已退出，本轮 fresh resumed M0/M1/repeats 不冒充旧 S0。

**结果分类：** ATTRIBUTED 需要完整因果链及足以解释变化的 execution semantics；PARTIALLY_ATTRIBUTED 用于 identical input→native op→first nonidentical output，但内部原因未证明；无法稳定定位则 UNRESOLVED。三者均非后续 TODO gate。

#### Result

2026-09-26：**PARTIALLY_ATTRIBUTED**。Slurm 25847/25848（coarse）、25849（唯一一次 preprocessing-only fine trace），RTX 5090；12 次 eval/no_grad fixed forward，0 training / optimizer step。粗追踪最后 exact 为 `_preprocess` 输入，首先不同为第一层 hierarchy graph 输出，早于 encoder/renderer。

Fine 同对象 `M0_first vs M0_repeat`：最后 exact 为 `native_spmm.external.copy_lhs.sum.input`；第一个非 exact 为其 sum 输出 `[650,3] float32`，max abs diff=0.000244140625，mean abs diff=2.50400641025641e-7。同进程 fresh `M0_first vs M1_first`：在 `attr.copy_lhs.sum` 首次分叉，输出 `[650,5] float32`，max=1.862645149230957e-9，mean=2.292486337515024e-12。两组对应的有序边索引、feature 值、layout/COO format 一致，此前全部已记录阶段 exact。

调用点为 `GsHieEmbodiedDGLProcessor._preprocess_hierarchy`，`mmgs/models/utils/dgl_graph.py:493`（attr）/`:509`（external）的 `copy_u + mean`；DGL mean 先执行 sum，`dgl._sparse_ops._gspmm` 调 native `_CAPI_DGLKernelSpMM`，随后才做 degree division。上述 external 差异是 sum 阶段数值，不代表修改 gravity。未证明 kernel 内部是否为 atomic/reduction-order 等原因，因此不标 ATTRIBUTED。

12 次 forward 的 Python / NumPy / torch CPU / torch CUDA RNG 前后均未推进；observed tensors 全 finite；registered state/normalizer 未改变；checkpoint、源码和所检查输入 hash 不变。保留原 atol=rtol=1e-5 的比较，未要求输出通过该 gate。未训练、未修复，未执行 T25–T28。

**Known limitation：** snapshots 会改变同步、时序与分配历史；本结果定位 instrumented fixed-inference 的最早分叉，不独占解释旧 S0/S1/A 全部差异，也不改写旧 fresh A/B bitwise-equal 证据。未确认 native 内部原因，未宣称数值差异绝对无影响。T24 的 PASS 及全部 BLOCKED/FAIL/audit 历史保持原样；T24.5 不阻塞后续协议/训练/cache 工作。

#### Evidence

- [完整审计报告](validation/t24_5-first-divergence-20260926/README.md)、[边界与输出比较](validation/t24_5-first-divergence-20260926/summary.json)、[小型 contract](contracts/008-pink-cloth/episode_0/numerical_first_divergence_contract.json)。逐张量统计/哈希、全部 coarse/fine stages、RNG 和来源见报告目录。
- checkpoint SHA256 `421070e708ca8920de7b661a2708ec974ec4008c7469e80f7585e17a571a7979`；完整 tensor traces 保持 server-only：`outputs/deform360/t24_5-coarse-20260926/`、`outputs/deform360/t24_5-fine-20260926/`，各 trace path/bytes/hash 可追溯。
- [观测工具](../../tools/deform360_adapter/trace_numerical_divergence.py)、[CPU 比较工具](../../tools/deform360_adapter/compare_divergence_trace.py)。属于 SoMA/deform360-adaptation；roadmap tracked，新增文件尚未 tracked，未 commit/push。服务器 checkout clean，以 `/tmp` 工具副本执行；小型文件待未来 Git checkpoint 同步。未执行 T25。

---

## 第四阶段：建立可复查的 no-tactile baseline

### T25：冻结首个 baseline 的运行协议

Status: TODO

**问题：** smoke PASS 后，训练预算与评价方式还未固定。

**为什么现在解决：** 防止边跑边同时改多个变量。

**输入文件：**

- T5 split contract。
- T23/T24 验证结果。
- T17 固定的派生 scene；实际路径以 T17 记录为准。

**预计涉及文件：**

- `SoMA/configs/SoMA/deform360_v0_stage1.py`（拟新增）：运行 schedule。
- 实验记录；实际文件路径由对应运行确定。

**最小修改：** 只固定种子、步数预算、rollout 上限、日志/输出路径、checkpoint 选择规则。

**验证方法：** 展开最终配置，检查训练 target 与测试边界。

**PASS：** 每个参数有确定值；测试段不用于调参或选 checkpoint；所有模型输入均不含 tactile。

**FAIL 后检查：** 默认 rollout 自动增长、重复 dataset 次数、validation/test 路由或未固定参数。

#### Result

Not executed.

#### Evidence

Not executed.

### T26：执行约定预算的 Stage 1 训练

Status: TODO

**问题：** 已贯通的流程能否完成一个有明确预算的训练运行？

**为什么现在解决：** 前置加载、梯度、rollout 和恢复已分别通过。

**输入文件：**

- T25 冻结的 `SoMA/configs/SoMA/deform360_v0_stage1.py` 及对应运行记录。

**预计涉及文件：**

- `SoMA/configs/SoMA/deform360_v0_stage1.py`（拟新增）：冻结配置。
- `SoMA/tools/train.py`：现有训练入口。

**最小修改：** 不改模型或组件，只执行该运行。

**验证方法：** 核对完成步数、loss/梯度有限性、checkpoint、训练帧范围和日志。

**PASS：** 完成预定预算并产出可恢复 checkpoint；如实记录收敛情况。

**FAIL 后检查：** 首个异常 step、输入和运行状态；不把“运行完成”写成预测质量已验证。

#### Result

Not executed.

#### Evidence

Not executed.

### T27：只生成并核验 Stage-1 cache

Status: TODO

**问题：** Stage 2 需要按帧定位的预测初态。

**为什么现在解决：** 不能用 Deform360 的后续重建 Gaussian 替代这个 cache。

**输入文件：**

- T26 checkpoint；实际路径以 T26 记录为准。
- 已批准的训练区间数据（T5、T17）。

**预计涉及文件：**

- `SoMA/mmgs/models/simulators/gs_simulator_embodied.py`：`save_gaussian` 及 cache 生成入口。

**最小修改：** 只导出 Stage-1 预测 cache。

**验证方法：** 检查 local frame 0 和 Stage 2 所需起点、`pred_pos/pred_cov`、N 和来源记录。

**PASS：** 所需 cache 全部存在、有限、identity 固定，源于该 checkpoint；不包含测试真值初始化。

**FAIL 后检查：** frame_gap、文件编号、首帧 cache、导出开关；不开始 Stage 2。

#### Result

Not executed.

#### Evidence

Not executed.

### T28：只串联一个 Stage 2 子窗口

Status: TODO

**问题：** dense frame 与 coarse cache 起点是否正确对应？

**为什么现在解决：** Stage 2 有独立的子窗口和初始化逻辑。

**输入文件：**

- T27 Stage-1 cache；实际路径以 T27 记录为准。
- T5 split contract。
- T17 派生 scene 中按 gap=1 读取的数据；实际路径以 T17 记录为准。

**预计涉及文件：**

- `SoMA/configs/SoMA/deform360_v0_stage2.py`（拟新增）：dataset/cache 配置。
- `SoMA/mmgs/models/simulators/gs_simulator_embodied_stage2.py`：Stage 2 子窗口及 cache 初始化。

**最小修改：** 只读取一个训练子窗口并加载对应 cache。

**验证方法：** 打印窗口起止、source/local 索引、cache key、controller 和 GT 首尾。

**PASS：** cache 与子窗口起点一致；窗口完全位于训练段；至少有一个有效预测 target。

**FAIL 后检查：** coarse/dense gap 混淆、末尾窗口截断、cache key 和多余偏移。

#### Result

Not executed.

#### Evidence

Not executed.

### T29：只验证 Stage 2 一个优化步骤

Status: TODO

**问题：** Stage 2 的 dense-step 梯度路径是否成立？

**为什么现在解决：** Stage 1 的 smoke 不能替代 Stage 2。

**输入文件：**

- T28 已核验的子窗口。
- 对应权重及 T27 cache；实际路径以相应 TODO 记录为准。

**预计涉及文件：**

- `SoMA/configs/SoMA/deform360_v0_stage2.py`（拟新增）：单步 smoke 设置。
- `SoMA/mmgs/models/simulators/gs_simulator_embodied_stage2.py`：Stage 2 优化步骤。

**最小修改：** 只运行一个 forward/backward/step。

**验证方法：** 检查 dense target、loss、梯度、更新参数与 cache 使用。

**PASS：** 目标帧正确，梯度和输出有限，optimizer 正常更新。

**FAIL 后检查：** Stage 2 时间尺度、cache 初态、模板状态和梯度路径。

#### Result

Not executed.

#### Evidence

Not executed.

### T30：执行固定预算的 Stage 2 训练

Status: TODO

**问题：** 能否得到用于连续 rollout 的 dense dynamics checkpoint？

**为什么现在解决：** Stage 2 单窗口已经通过验证。

**输入文件：**

- T29 已验证的 `SoMA/configs/SoMA/deform360_v0_stage2.py` 配置。
- T27 cache；实际路径以 T27 记录为准。

**预计涉及文件：**

- `SoMA/configs/SoMA/deform360_v0_stage2.py`（拟新增）：运行 schedule。

**最小修改：** 固定并执行一个 Stage 2 预算，不改其他组件。

**验证方法：** 检查所有训练窗口、完成步数、loss 和最终 checkpoint。

**PASS：** 完成预算，无测试监督进入训练，checkpoint 可恢复。

**FAIL 后检查：** 第一个失败窗口、cache 边界和状态更新；不同时更换 loss 或数据。

#### Result

Not executed.

#### Evidence

Not executed.

### T31：验证真正的 continuous rollout

Status: TODO

**问题：** 推理是否持续依赖预测状态，而非分段借用未来状态？

**为什么现在解决：** segmented rollout 不能代替目标中的 continuous prediction。

**输入文件：**

- T30 checkpoint；实际路径以 T30 记录为准。
- T2 派生并由 T17 组装的 initial Gaussian。
- T11 固定 controller trajectory。

**预计涉及文件：**

- `SoMA/mmgs/models/simulators/gs_simulator_embodied_stage2.py`：continuous 路径。
- `SoMA/tools/test.py`：现有测试入口。

**最小修改：** 只启用并验证 continuous 推理协议。

**验证方法：** 追踪各模板边界的状态来源；从 local 0 起滚动，检查 155–193 测试段。

**PASS：** 后续模板来自在线预测更新；不读取后续重建 PLY、不注入测试真值状态，完成到 source 306。

**FAIL 后检查：** `test_rollout_mode`、online cache 更新、边界重置和预测起止索引。

#### Result

Not executed.

#### Evidence

Not executed.

### T32：汇总首个 no-tactile baseline 结果

Status: TODO

**问题：** 是否已形成可复查、可供后续 tactile 消融比较的 baseline？

**为什么现在解决：** “能训练”和“有可解释评估结果”需要分别确认。

**输入文件：**

- T26/T30 checkpoint；实际路径以对应 TODO 记录为准。
- T31 predictions；实际路径以 T31 记录为准。
- 固定测试 GT（T5、T14、T15）。
- 对应运行记录。

**预计涉及文件：**

- 独立评估/报告入口，具体文件路径尚未确定；不再改训练组件。

**最小修改：** 只计算固定协议下的指标并整理证据。

**验证方法：** 核对评分帧、相机、分辨率、mask 规则和预测状态来源。

**PASS：** 结果可追溯到唯一数据 manifest、配置和 checkpoint；区分 train/test、Stage 1/2、smoke/完整预算；明确无 tactile conditioning。

**FAIL 后检查：** 评价索引、mask、checkpoint 来源或缺失输出；不通过改模型补齐报告。

#### Result

Not executed.

#### Evidence

Not executed.

---

主线不包含显存优化或 tactile conditioning。T0 使用 tactile 仅用于核验**官方已有 contact-window 起点**；后续 controller 选择、帧序列和模型输入不以 tactile 数值为条件。

**当前全部 TODO 均未执行；第一项是 T0。**
