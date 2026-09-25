# Negative signed gap impact assessment

**T10.5 controller representation semantics audit · 2026-09-25 · 只读审计**

**结论：T10 的负 gap 主要属于 B（采样几何问题），不是 SoMA 明文禁止的 representation，也尚未构成 C（graph 构建错误）的证据。** SoMA 使用给定运动的 controller 点作为条件节点，不把它们当作必须满足实体不穿透的碰撞面。但位置会影响邻居和边特征，因此也不能断言毫米级偏差对模型无影响。

| 检查 | 实际证据与含义 |
|---|---|
| 原始 sample 点 | 服务器 `datasets/soma_sample/soma_data_sample/cloth_lift/left_lift_1/track_process_data.pkl` 实读为 `float64 [200,30,3]`。所有两两距离在 200 帧内最大变化仅 `1.28e-16 m`，表现为共同刚性运动的点集；没有左右手指、表面法向或厚度字段。不能把其前后半组直接解释为两根物理手指。 |
| sample overlap | 已有 `controller_mask/dis_split/controller_dis_splitnl_2_downsample_rate_0d5_downsample_rate_0d5_track_process_data.pkl` 给出 `30→15→2→1`。复合后的两组含 14/16 个原始点；frame 0 两组均为三维分布，其凸包交集可容纳半径 **10.373 mm** 的球。全序列跨组最近点距离 **3.260 mm**，无跨组完全重合点。存在几何包络重叠，但这不是“官方夹爪实体穿透”的证据，也不是可照搬到 T10 的 signed surface gap。 |
| 是否要求不穿透 | 当前 embodied 路径将 controller 当前/上一帧/初态坐标拼入节点，并设置 controller pin mask；检查过的 loader、processor、backbone、decoder 没有 controller 实体碰撞、surface normal 或 signed-gap 判据。输入的是运动条件，不是接触约束求解器。[1] |
| 距离是否有作用 | 有。默认 `forward_last_layer=False`，原始点层不建空间边；层级点先聚合，再用 QueryAndGroup 候选及 `distance < radius` 构建初态/当前态边并去重。该 sample config 的层级半径为 **0.2/0.5 m**、候选数 16；不能误称 30 个原始点直接按 0.03 m 建边。边特征含相对方向、当前/初态距离比及相对速度；不是只使用 controller ID。[2][3] |

**T10 的归类与影响。** `g=x_right−x_left` 沿 root +X，113–140 中 25 帧为负，范围 `−2.113～+0.750 mm`。点身份保持固定，匹配左右点还相差 1 mm root-Z；negative gap 不等于点重合、NaN 或点序翻转。[4]

- **A：有语义约定，但不是纯符号解释。** SoMA 不消费 g 或面法向，T10 的点可以作为运动锚点输入；更换 gap 符号不能消除采样面位置顺序反转。
- **B：主要归类。** 原始 opening 映射与每侧 7 mm inward offset 产生负间隙，是采样几何偏差；实体穿透与物理可接受度尚未验证。它不自动使轨迹格式无效。
- **C：潜在下游影响，尚非已发现的构图故障。** 几何变化可改变聚合位置、候选邻居、方向和距离比；“偏差小于 radius”不足以证明无影响。ratio 特征的距离除数未在对应公式中 clamp，真正的零长度边需单独关注，不能与负 signed gap 混为一谈。[3]

**另一个独立语义注意点：**默认第一次聚合按连续两点配对，`14/15` 会进入同一个 cluster 7；再按 7/8 个 cluster 分组，原始点对应为 `0–13 / 14–29`。对 T10 的 `0–14 left / 15–29 right` 而言，会混入一枚 left anchor。因此，“点标签明确”不等于“现有层级严格保持两指分离”。这是现有 grouping convention 与 T10 语义的衔接问题，不是负 gap 触发的新 bug；仅记录，未改配置或生成 cluster。[5]

**审计判断：**不应仅为消除负号而翻轴、交换手指或修 opening。T10 数值 PASS 可保留；后续是否接受几何偏差及层级分组语义需要明确取舍，当前没有训练或图运行证据证明影响为零。本次不执行 T11/T12，不修改 T10 contract、生成逻辑、roadmap 或 SoMA 源码，不 commit/push。

证据定位（源码相对 SoMA root）：[1] `mmgs/models/simulators/gs_simulator_embodied.py:231–257,664–703`；[2] `configs/_base_/models/gs_simulator_embodied.py:82`、`configs/SoMA/cloth_lift_stage1.py:81–89,134–144`、`mmgs/models/utils/dgl_graph.py:426–450,519–525,561–584`；[3] `mmgs/models/backbones/meshgraphnet_embodied.py:314–343`；[4] 本目录 `signed_gap_followup.json` 与 T10 contract；[5] `mmgs/datasets/embodied_dataset.py:679–727`。审计基线 SoMA `79ab9be`；sample 点文件 SHA256 `d56438167b93c0661c69bb68292fc95bb803d1b22b04cac9b9545407a54c40ba`，已有 cluster 文件 SHA256 `ca7a02222e37695d37215a49d6a111b84d64fd5d8136244f013464d6815529d0`。sample 统计/凸包检查仅用 CPU，未实例化模型或构建新 graph。
