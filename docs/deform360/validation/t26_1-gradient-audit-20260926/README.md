# T26.1：Stage-1 first nonfinite gradient audit

**PARTIALLY_ATTRIBUTED — blocking diagnostic for T26。T26 保持 FAIL/nonfinite。**

先完成 T26 checkpoint：`2aedd3bc28e70a94f940d0e8bcf863544c3ec473`，push成功，进入诊断前ahead/behind=0/0、工作树clean。没有重启正式Stage-1 training。

## 实验与同一性

通过原始 `tools/train.py` 的 seed→model→dataset→optimizer→runner 初始化路径执行一次forward/backward。只在model实例内修改诊断horizon，或选择已有step/component weighted loss；不改冻结文件。OptimizerHook的诊断替代只执行一次backward并退出，绝不clip/step/save。没有额外warmup或调用model.init_weights；不是T22手动harness。

11个独立Python进程全部核验：289个initial registered state tensor hashes、input tensors、camera geometry、Gaussian字段exact一致。rollout3 total loss=40912.95703125，精确复现正式T26。Slurm25851/25852/25853/25854，RTX5090；不改deterministic/TF32/cuBLAS flags。

## Rollout threshold

| Rollout | Total loss | Finite grad tensors | Nonfinite tensors | NaN | +Inf | -Inf | Peak allocated GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 59908.7890625 | 0 | 275 | 2466339 | 0 | 0 | 1.387 |
| 2 | 45269.37109375 | 0 | 275 | 2466339 | 0 | 0 | 2.481 |
| 3 | 40912.95703125 | 0 | 275 | 2466339 | 0 | 0 | 3.684 |

**最早失败rollout=1；最早失败step=1：source113→123。** 第一层按授权完整执行1/2/3对照；此后全部收窄到rollout1，没有继续更长rollout诊断。所有forward outputs与loss finite。每个失败probe第一个bad parameter（按named_parameters遍历顺序）是`scene_attr.rope`，这不是first bad operation。

277个trainable参数张量中，275个有梯度；2个`backbone.encoder.emb_norm.weight/bias`没有grad，既不能记为finite也不能记为nonfinite。

## 单step / 单component

| Step / target | Objective (真实weight) | Loss | Gradient | finite/bad tensors | NaN / +Inf / -Inf | Global norm (仅finite) |
|---|---|---:|---|---|---|---:|
| 1 / 123 | combined | 59908.7890625 | NONFINITE | 0 / 275 | 2466339 / 0 / 0 | N/A |
| 1 / 123 | momentum | 21445.3359375 | FINITE | 275 / 0 | 0 / 0 / 0 | 7186147.15948657 |
| 1 / 123 | l2 | 33750.86328125 | NONFINITE | 0 / 275 | 2466339 / 0 / 0 | N/A |
| 1 / 123 | ssim | 4712.59228515625 | NONFINITE | 0 / 275 | 2466339 / 0 / 0 | N/A |

momentum=1、L2render=.9、SSIM=.1均由真实loss实现应用；没有重复乘权重。static只是无grad的零占位，其余配置loss在此实际path未调用。各component独立新进程、独立graph，不在同一graph累计backward。单项L2和SSIM已各自失败，因此“单项全部finite、相加才overflow”的分支不成立，未做无关的gradient-sum试验。

## First bad backward boundary

`finite loss gradient → finite grad_out_color/depth → _C.rasterize_gaussians_backward → NaN grad_means3D / grad_covariance → 275个参数梯度NaN`

两次anomaly（L2/SSIM）均报告`_RasterizeGaussiansBackward returned nan values in its 0th output`。再只包装该native调用，以原参数调用原函数、原样返回结果，记录以下事实：

| Component / camera | grad_out_color max abs | color grad shape | returned means3D NaN | returned covariance NaN | ±Inf |
|---|---:|---|---:|---:|---:|
| l2 / brics-odroid-009_cam1 | 0.8999999761581421 | [3,360,640] float32 | 0 | 0 | 0 |
| l2 / brics-odroid-023_cam0 | 0.8999999761581421 | [3,360,640] float32 | 3 | 6 | 0 |
| ssim / brics-odroid-009_cam1 | 0.4844515919685364 | [3,360,640] float32 | 0 | 0 | 0 |
| ssim / brics-odroid-023_cam0 | 2.026884078979492 | [3,360,640] float32 | 3 | 6 | 0 |

023_cam0 的两个probe均只有Gaussian **row3648（zero-based）**在native出口首次非有限：means3D `[12861,3]` 有3 NaN，covariance `[12861,6]` 有6 NaN；这些tensor其余finite元素max abs=0。所有返回项的+Inf/-Inf均为0；means2D/colors/opacity/SH/scales/rotations仍finite。009_cam1所有返回梯度为零且finite。grad_out_depth `[1,360,640]`全零finite。

**最后finite边界**：传入native rasterizer backward的color/depth梯度及所有显式浮点输入。**第一个nonfinite边界**：native函数返回的means3D/covariance梯度。opaque uint8工作缓冲未解释其内部数值；没有证明kernel内具体哪一项运算先失败。不能凭返回NaN就排除kernel内部先出现Inf。

实际source call chain：

- `mmgs/models/simulators/gs_simulator_embodied.py:435` → `AccDecoder.pre_render`。
- `mmgs/models/heads/acc_decoder.py:141` → `render_gaussian`。
- `mmgs/models/utils/render.py:231` → `GaussianRasterizer`。
- `/data1/userdata/tcweng/miniconda3/envs/soma/lib/python3.10/site-packages/diff_gaussian_rasterization/__init__.py:127` → `_C.rasterize_gaussians_backward`（自定义C++/CUDA extension）。
- [安装版本的带行号边界源码](installed_rasterizer_boundary.txt)；Python文件和.so SHA256保存在native reports及contract。

## 观测上下文与限制

首步predicted position max abs≈114.12248、packed covariance max abs≈62.07981，均finite；两个render图像全零。forward仍报告positive radii数量385/975（num_rendered是native计数，不当作Gaussian数量）。这些是现象，不是已证实的kernel因果解释。本轮不修改几何、初始化、loss、renderer、precision、epsilon或schedule，不提出已经验证的修法。

T22的finite backward没有在正式随机初始化path中复现；既有T22/T24事实保留，不改写成相同条件。当前证据把边界定位到renderer，而不是证明DGL、LR、clipping或某个特定算术表达式导致此错误。

## Evidence / Git

- [机器可读contract](../../contracts/008-pink-cloth/episode_0/first_nonfinite_gradient_contract.json)。
- `reports/`：11份报告（每份≤205KB），包含逐参数NaN/Inf/finite范围、hash、loss、memory；`logs/`：对应原始文本日志及anomaly forward trace。
- [rollout probes](rollout_probes.sh)、[component probes](component_probes.sh)、[anomaly probes](anomaly_probes.sh)、[native probes](native_probes.sh)、[Slurm accounting](slurm_accounting.txt)。
- [probe tool v1](probe_tool_v1.py)与[anomaly tool snapshot](probe_tool_anomaly.py)保留已执行工具hash；最终工具在`tools/deform360_adapter/audit_first_nonfinite_gradient.py`。
- [final integrity](final_integrity.json)：17项核心源码hash、794项既有资产hash不变；服务器Git clean；没有*.pth。没有生成tensor dumps。
- T26.1新增文件均属于SoMA/deform360-adaptation，暂未commit/push；服务器执行副本在`outputs/deform360/t26_1-audit-20260926/`，不覆写服务器源码。T26 FAIL及既有历史保留，T27未执行。
