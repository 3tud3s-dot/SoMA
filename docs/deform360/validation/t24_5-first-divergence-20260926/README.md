# T24.5：数值重复性 first-divergence audit

2026-09-26 · **PARTIALLY_ATTRIBUTED** · **Non-gating diagnostic**

T24 仍为 **PASS — checkpoint state recovery**。本诊断不改变 checkpoint 验收，不阻塞 T25–T27，也未执行 T25–T28。所有历史 BLOCKED/FAIL/repeatability audit 均保留。

## 结论与证据边界

首次观测到的数值分叉位于 graph preprocessing 的第一层 hierarchy 聚合，早于 dynamics network、pred_pos 与 renderer。细追踪证明：**相同 EID 顺序的图边、相同输入特征 → DGL `_gspmm(copy_lhs, sum)` → 首次非逐位一致的输出**。其调用内部的 native 边界为 `_CAPI_DGLKernelSpMM`。

仅定位到 native operation 边界，未深入 kernel，也没有证明 atomics、归约顺序、workspace 或其他机制中的哪一个导致变化，因此使用 PARTIALLY_ATTRIBUTED，不使用 ATTRIBUTED。没有进行修复或干预实验，也不宣称该边界是所有历史差异的唯一原因。

## 实验范围

- 历史 training-derived S0 进程已经退出。本轮禁止训练，因此全部 M0/M1 都是从现有有限 checkpoint 重建的对象；这些标签不冒充旧 S0。
- 使用 `outputs/deform360/t24-same-process-20260926/epoch_1.pth`；SHA256 `421070e708ca8920de7b661a2708ec974ec4008c7469e80f7585e17a571a7979`，30024957 bytes。原文件未修改。
- Config A：023_cam0、009_cam1；source 113 → 123；7 mm controller；30→10→2→1；external gravity `[0,0,-39.2]`。真实 dataset/model/preprocessing/renderer/loss。
- 只有 eval/no_grad fixed forward；训练步数与 optimizer step 均为 0。每次 forward 后 registered model/normalizer、optimizer、定向 runtime 状态检查不变。
- 未启用新的 deterministic algorithms，未改 TF32/cuBLAS/CUDA flags、precision、renderer 或 tolerance；继承既有 harness 的 seed=5 和 backend 设置。原 atol=rtol=1e-5 仅作报告，未改为 checkpoint gate。

| 级别 | Slurm job | GPU | 工具内耗时 | 结果 |
|---|---|---|---:|---|
| coarse A | 25847 | RTX 5090 | 10.671 s | 完成，0:0 |
| coarse B | 25848 | RTX 5090 | 10.656 s | 完成，0:0 |
| preprocess-fine A | 25849 | RTX 5090 | 11.122 s | 完成，0:0 |

同一进程的事件顺序：restore M0 → M0_first → M0_repeat → 新建 M1 并 resume 相同 checkpoint → M1_first → M0_after_M1。两个 coarse 进程独立；之后只有一次 preprocessing 范围内的 fine refinement。共 12 次 fixed forward。

## Coarse 阶段

观测真实 `_preprocess` 输入/输出、backbone 输入、其 `encoder`（message-passing encoder）输入/输出、backbone 输出、`pre_predict`、`dynamic_proj`、`pre_render`、`encode_decode` 输出及 image loss。此实现的 raw decoder 输出是 deformation-gradient 参数，不虚构 velocity 模块。trace JSON 保存完整事件顺序。

三组 coarse 比较（同对象重复、同进程 fresh instance、两个独立 fresh process）最后相同边界均为 `001:preprocess.input:0`，首次不同边界均为 `002:preprocess.output:0`。例如同对象重复的第一层 `attr [650,5]` max diff=5.820766091346741e-11。native 前的完整 base graph、hierarchy 输入在细追踪中也被记录。

## 唯一一次 fine refinement

只包装 graph base/hierarchy 方法和 preprocessing 内的 DGL SpMM 调用；委托原函数，原样返回结果。没有给所有 PyTorch layer 加 hook。

| 比较 | 最后 exact 输入 | 第一非 exact 输出 | shape / dtype | max abs diff | mean abs diff |
|---|---|---|---|---:|---:|
| A_M0_first vs A_M0_repeat | `external` SpMM 输入 | `external` SpMM sum 输出 | `[650, 3]` float32 | 0.000244140625 | 2.50400641025641e-07 |
| A_M0_first vs A_M1_first | `attr` SpMM 输入 | `attr` SpMM sum 输出 | `[650, 5]` float32 | 1.862645149230957e-09 | 2.292486337515024e-12 |

以上两组比较：此前全部已记录阶段 exact；有序 src/dst/eid、feature tensor、shape/dtype/device/stride/contiguous 一致；COO sparse format 状态一致，观测读取未创建新的格式。拓扑为 12891 个 source node → 650 个 destination cluster，含 12861 Gaussian + 30 controller source nodes。所有观测 tensor finite。

`external` 的 mean reducer 在 DGL 内先做 sum，再按入度除法，因此表中 `2.44140625e-4` 是**除法前的 sum 差异**，不是改动了 canonical gravity。`attr` 同样在 mean 的 sum 子步骤首次出现差异。不同对象比较的 first affected field 不同，但定位到同一类型的 native 聚合边界。

### 实际调用链

- [`GsSimulatorEmbodied._preprocess`](../../../../mmgs/models/simulators/gs_simulator_embodied.py) line 257：调用 graph generator。
- [`GsHieEmbodiedDGLProcessor._preprocess_hierarchy`](../../../../mmgs/models/utils/dgl_graph.py) line 493（attr）、509（external）：`send_and_recv(copy_u, mean)`。
- 安装的 DGL 2.1.0 `dgl/ops/spmm.py:79–85` 将 mean 传为 sum，`:114` 才除入度。
- `dgl/backend/pytorch/sparse.py:165`：`GSpMM.forward` 调 `_gspmm`。
- `dgl/_sparse_ops.py:239`：`_CAPI_DGLKernelSpMM`，DGL native CUDA operation，不是 Gaussian rasterizer。安装源码绝对路径与 SHA256 见 `server_artifact_manifest.json`。

### 下游结果（未修改原 tolerance）

| 比较 | tensor | max abs diff | mean abs diff | bitwise equal | 原 atol/rtol gate |
|---|---|---:|---:|---|---|
| A_M0_first vs A_M0_repeat | loss_mse_momentum | 5.519390106201172e-05 | 5.519390106201172e-05 | False | False |
| A_M0_first vs A_M0_repeat | pred_cov | 3.637978807091713e-11 | 9.498936177951379e-13 | False | True |
| A_M0_first vs A_M0_repeat | pred_pos | 8.940696716308594e-08 | 6.124138134629859e-09 | False | True |
| A_M0_first vs A_M0_repeat | render/0 | 0.0002972632646560669 | 4.277578993962918e-08 | False | False |
| A_M0_first vs A_M0_repeat | render/1 | 0.002740412950515747 | 5.489726955029592e-08 | False | False |
| A_M0_first vs A_M1_first | loss_mse_momentum | 8.0108642578125e-05 | 8.0108642578125e-05 | False | False |
| A_M0_first vs A_M1_first | pred_cov | 3.637978807091713e-11 | 9.05053243409872e-13 | False | True |
| A_M0_first vs A_M1_first | pred_pos | 5.960464477539062e-08 | 4.935759945388456e-09 | False | True |
| A_M0_first vs A_M1_first | render/0 | 0.0002465248107910156 | 3.53468182631197e-08 | False | False |
| A_M0_first vs A_M1_first | render/1 | 0.002740472555160522 | 4.308873359251366e-08 | False | False |

完整 loss/metric 比较见 fine comparison JSON。pred_pos/cov 包含 30 controller + 12861 Gaussian nodes（12891）；render 顺序为 023、009，shape `[3,360,640]`。这些结果显示从 preprocessing 到预测和图像的差异链，尚未通过因果干预证明单个 kernel 解释所有下游误差。

## RNG / state / integrity

12 次 forward 前后的 Python、NumPy、torch CPU、torch CUDA RNG SHA256 各自不变；所以这些受观测 fixed forward 没有消费这些 RNG stream，不支持“forward 内随机采样”解释本次分叉。不同进程初始化前后的 RNG 差异不由此全局排除。

所有 trace 的 registered state hashes 一致（包括 normalizer）；canonical 配置相同；checkpoint、源码与工具覆盖的 scene/input assets hash 检查通过。相应逐张量信息记录 shape、dtype、device、stride/contiguous、finite、min/max/mean 与 SHA256。此轮不重复完整 checkpoint recovery 的 save/resume 验收。

## 已知限制与非阻塞语义

- CPU tensor snapshots 会同步 GPU、改变时序及分配历史；观测本身可能改变具体数值。这是 instrumented inference 的边界证据，不能改写以前未 instrument 的 fresh A/B bitwise-equal 事实。
- 未观察旧 training-derived S0 活体；没有通过新训练重建它。
- 没有证明 native kernel 内部原因；没有断言“所有数值差异无影响”。
- 保留 T24 的 PASS 与 limitation。T24.5 classification 无论如何不阻塞 T25+；若未来 resume training 出现系统性 trajectory/metric discontinuity，可另行授权调查或修复。

## 文件与复查

- [`summary.json`](summary.json)：比较摘要、first boundary 的两侧统计、完整 final-output metrics。
- [`coarse/`](coarse/)：两进程 trace stats / reports、三组 CPU 比较和当时执行的 coarse 工具快照。
- [`fine/`](fine/)：preprocessing-only trace stats / report 和两组比较。
- [`server_artifact_manifest.json`](server_artifact_manifest.json)：复制 JSON 的服务器 SHA256、native 源码 hash、环境版本；每个 JSON 小于 1 MiB。
- 工具：`tools/deform360_adapter/trace_numerical_divergence.py`、`compare_divergence_trace.py`；独立 validation instrumentation，不改 SoMA 核心源码。
- 小型 contract：`docs/deform360/contracts/008-pink-cloth/episode_0/numerical_first_divergence_contract.json`。
- 完整 tensor `.pt` 仅服务器：`outputs/deform360/t24_5-coarse-20260926/`、`outputs/deform360/t24_5-fine-20260926/`。每份 trace 的 path/bytes/SHA256 在对应 JSON 中；未复制 Mac、未纳入 Git。

### 命令（服务器，全部 CUDA 在 Slurm 内）

```bash
srun -p 5090 --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=24G --time=00:10:00 \
  /data1/userdata/tcweng/miniconda3/envs/soma/bin/python -B /tmp/tcgs_t245_fine.py \
  --workspace /data1/userdata/tcweng/projects/tcgs \
  --checkpoint-dir /data1/userdata/tcweng/projects/tcgs/outputs/deform360/t24-same-process-20260926 \
  --output /data1/userdata/tcweng/projects/tcgs/outputs/deform360/t24_5-fine-20260926 \
  --trace-level preprocess-fine --label A
```

以上为已完成 job 25849 的记录，非继续运行指令。coarse jobs 使用 `/tmp/tcgs_t245_coarse.py` 与独立 coarse 输出目录，label 分别 A/B；对应源码快照已保留。输出工具拒绝覆盖已有报告。CPU 比较使用 `compare_divergence_trace.py --directory ... --left ... --right ...`，不初始化 CUDA。

本轮未 commit/push。所属 SoMA/deform360-adaptation；新增工具/contract/evidence 尚未 tracked。服务器 checkout 保持 clean，工具以 `/tmp` 副本执行；小型文件待未来 checkpoint 后通过 Git 同步。
