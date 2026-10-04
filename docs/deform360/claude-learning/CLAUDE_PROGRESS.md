# Claude 学习辅助文档：SoMA-D360 学习进度

> **Claude 学习辅助文档**。只记录学习路线与理解进度，**不是**工程 TODO，也不代表任何训练/实验状态。
> 工程状态以 `../SoMA-D360-v0-implementation-todo.md` 与 `../contracts/` 为准；既有个人路线 `../learning/source-reading-roadmap.md`（R01–R25）保留不动，本文件用 U 编号并标注对应 R 编号。
> 新窗口接续：先读本文件“当前位置”，再读 `CLAUDE_NOTES.md`，然后按单元卡打开源码（行号以当时 HEAD 为准，使用前重新核对）。

## 当前位置

- 源码基线：SoMA `deform360-adaptation` @ `aeeac82`（2026-09-28），工作树有未提交的 `.gitignore`、reports、wandb 文档与 importer（非本学习任务产生）。
- 当前单元：**U2.4 Decoder 与下传** — 已讲解，等待回答检查问题。
- 下一步：U2.4 回答后进入 U2.5（packed6、render 梯度边界、pred_rot）。学习者 U0.1–U2.3 基本全对（U1.5 Q2 cache、U2.1 Q1 边数上限各有一处已纠正），保持正常偏快节奏，重点放在状态/时间/identity 推理而非名词。
- 本阶段约束：只读源码/论文，只维护 `claude-learning/` 两份文档；不改源码/config、不启动训练或 CUDA、不 commit/push。

## 项目状态快照（只作学习背景；2026-10-03 读取 Mac 端文件，未连服务器核验）

| 环节 | 工程记录状态 | 学习时的含义 |
|---|---|---|
| Stage-1（T26） | PASS，46 epoch / 2300 steps，`epoch_46.pth` | 有完整 coarse checkpoint |
| Stage-1 cache（T27） | PASS，gap10 命名 `frame_0…150` | Stage-2 起点可用 |
| Stage-2（T30） | 用户 2026-10-03 确认：**尚未跑完**；服务器上最新的 `epoch_k.pth` 即当前进度（k 未在 Mac 核验；Mac 工作区无任何 .pth，checkpoint 均 server-only） | 学习中讨论 Stage-2 结果时只能说“中间 checkpoint” |
| Continuous（T31/T31.1） | exploratory SUCCESS，仅 epoch22 中间 checkpoint | 不是最终 baseline 数字 |
| 汇总（T32） | TODO；存在未跟踪的 report 草稿目录 | 无正式评估结论 |

## Curriculum（主线：论文 Figure 2 + §3/§4）

每单元 30–60 分钟；一次只推进一个单元。状态：TODO / IN_PROGRESS / PASS / REVISIT。

| ID | 单元 | 论文锚点 | 对应 R | 状态 |
|---|---|---|---|---|
| **M0 全局** |||||
| U0.1 | 训练与推理分别在做什么：输入、监督、状态来源 | Fig.2、Eq.4–5、§5.1 | R01 | PASS |
| U0.2 | 调用链与循环层级：train.py→runner→train_step→forward_train；test.py→single_gpu_rollout→simple_test | — | R01/R06 | PASS |
| **M1 数据与时空对应** |||||
| U1.1 | 一个 sample 的字段卡：inputs / gt_label / meta | §4.1 | R02 | PASS |
| U1.2 | 时间：source↔local、split、gap10/gap1、range 半开、dt/real_dt 与 gravity×4 | §4.2.3 | R03 | PASS |
| U1.3 | 相机与图像：2 camera、640×360、mask 后的 GT、外参/内参约定 | §4.1 | — | PASS |
| U1.4 | Gaussian：PLY→GaussianModel，pos/cov/SH0/opacity，fixed identity | §3 | — | PASS |
| U1.5 | Controller：PKL `[194,30,3]`→sample `[T,30,3]`，30→10→2→1 分组 | §4.2.1 | R04 | PASS |
| **M2 网络** |||||
| U2.1 | 层级与图：拼接顺序、p2c、层内边、pin | Eq.1、App.B.2 | R07 | PASS |
| U2.2 | 物理特征：node/edge feature、gravity、Normalizer | Eq.9–11 | R08–R10 | PASS |
| U2.3 | 一层 message passing 与 16 层 encoder | App.B.2 | R11 | PASS |
| U2.4 | Decoder：latent→11 参数→F→位置，粗到细传播 | Eq.2 | R12–R13 | IN_PROGRESS |
| U2.5 | Covariance 与渲染：FΣFᵀ、packed6、render 梯度边界 | Eq.3 | R14 | TODO |
| **M3 Stage-1** |||||
| U3.1 | Loss：masked L2 + SSIM + momentum，权重与聚合 | Eq.12–13 | R15 | TODO |
| U3.2 | rollout 循环的状态账本、detach、curriculum 3→15 | Eq.4 | R16–R17 | TODO |
| U3.3 | 优化与 checkpoint：OptimizerHook、clip、Adam、Hood LR、resume vs load | — | R19–R20 | TODO |
| **M4 Cache → Stage-2** |||||
| U4.1 | Stage-1 cache：`frame_<10i>` 键、来源、identity | §4.2.3 | R18 | TODO |
| U4.2 | Stage-2：窗口、cache 初始化、model.frame_gap vs dataset gap、schedule 映射 | §4.2.3 | R21–R22 | TODO |
| **M5 Rollout 与评估** |||||
| U5.1 | Continuous rollout：自回归、online template、10 帧边界 prev_state 重设 | §5.1 | R23 | TODO |
| U5.2 | 评估：指标、train/test 边界、数据泄漏检查、当前限制 | §5、App.B.4 | — | TODO |
| **M6 触觉（只讨论，不实施）** |||||
| U6.1 | 触觉数据：时间可用性、identity 空间、shape | — | R24 | TODO |
| U6.2 | 接入位置候选、设计理由、公平消融协议 | — | R25 | TODO |

允许的跳跃：M1 内部 U1.3/U1.4 可互换；U3.2 可提前到 U2.x 之前（只看状态账本，网络当黑盒）。

## 已完成记录

| 日期 | 单元 | 内容摘要 | 学习者反馈 / 检查结果 |
|---|---|---|---|
| 2026-10-03 | 准备 | 读 AGENTS.md、roadmap、Stage-1/2 config、split/camera/sample contracts、Stage-1/2 simulator 主循环、test.py rollout、论文全文摘要 | — |
| 2026-10-03 | U0.1 | 讲解训练 vs 推理：输入/监督/状态来源，两阶段时间尺度，用 Config A 样本串联 | 三题独立作答全对：step3 状态账本正确；能区分 future action 与 future observation；能说出用重建 splat 起步破坏 identity 与训练/推理分布一致。补充：cov 每步用 template Σ；触觉“当前”须精确到预测 t→t+1 时最多用到 t；见待解问题 T-1。→ PASS |
| 2026-10-03 | U0.2 | 讲解三层循环、train/test 两条调用链、rollout 循环位置差异、推理循环的三种用途 | 三题全对：epoch2→9 步、每 it 1 次 backward/step、每 epoch 50 次；tactile history 应是显式 rollout 状态（训练=forward_train 局部、按 sample/窗口重置；推理=与 prev/cur 并列由 single_gpu_rollout 持有；不随 template 边界重置；避免 `self.` 隐藏状态）；能指出 eval 写 `scene_init_*` 会污染后续训练起点、破坏可复现性。→ PASS |
| 2026-10-03 | U1.1 | 讲解 sample 字段按角色分组（条件输入 / 监督 / 只用于诊断 / 路由 meta / 不在 sample 中的状态）、collate 的 batch 维、T16 mask 语义与论文 Eq.12 的差异 | 三题全对：seq_num Stage-1=10、Stage-2 `[0,13)`=1，帧数应取 `gt_label[0].shape[1]`；tactile = observation condition，同一 video_range 切片，预测 t→t+1 最多读 `[:t+1]`；遮挡区 GT 黑、render 有布、L2 权重 1 → occlusion supervision mismatch，恰在 tactile 最有信息的接触区。补充：加 tactile 时断言长度=GT T；gap10 下按 video_range 切片会丢掉中间 9 帧 tactile（引出 U1.2）；opacity/SH 不更新 → 模型只能把 Gaussian 挪离接触区来降 loss。→ PASS |
| 2026-10-03 | U1.2 | 讲解四条时间轴（source/local/sample k/墙钟）+ model dt；半开 split 与两阶段实际覆盖（Stage-2 窗口重叠 3 帧、local 150–154 不作 dense target）；dt 在 backbone 中的用法与两阶段约 10× 速度尺度差；gravity×4 对两阶段都不物理一致（推导） | 三题全对：`[30,43)` frame_idx=5 → controller k5/k6 = local 35/36 = source 148/149，target k6，起点 `frame_30` 是 Stage-1 预测；vel 约 10×，Normalizer 只缩放数值不修复动力学语义；tactile 应按固定墙钟窗口 `[t−H,t]` 而非 rollout step 定义。补充：anchor_acc 随 τ² 变化 → 两阶段约 100× 且相对固定 gravity 的比例也变；“最近一帧”若按最近邻可能取到 t_k 之后的样本，必须取 ≤ t_k 的最新样本才因果。→ PASS |
| 2026-10-03 | U1.3 | 讲解 Config A 相机身份与顺序、c2w→R/T→world_view_transform 链、内参只用 fx/fy（主点必须居中）、RGB/mask 半像素 resize 一致、渲染背景黑与 GT 一致、相机顺序打乱对 loss 无影响 | 三题全对：cx 错 → 系统性水平偏移，模型把相机误差学成几何误差，两视角下所需 3D 校正互相矛盾；per-camera 模块必须以稳定 camera_id 作 key，不能用 list index；`[X,1] @ full_proj_transform` → 除以 w → `ndc2Pix` 得 (u,v)，不再转置。补充：所需世界位移 ≈ 20·z/fx 随深度变化，单视角也不是纯平移；初始帧不作 target，误差从第 1 步开始被惩罚；Camera 对象本身没有 id 字段，要从 `img_path` 的 `color/<n>` 取 numeric id，再经 scene_package `numeric_camera_mapping` 映射；w = z_cam，rasterizer 剔除 `p_view.z ≤ 0.2` 的点（auxiliary.h:166）。→ PASS |
| 2026-10-03 | U1.4 | 讲解 PLY 字段与 SH0 转换（T16.5）、raw 参数 vs 激活、模拟器从 GaussianModel 取的四样东西（xyz/scaling/cov/渲染外观）、随时间变化的只有 pos+cov6、行号即 identity、帧间重建 PLY 无对应 | 三题全对：只能改 pos/cov，外观误差被吸收进几何/动力学；逐行 L2 错在无跨帧 identity，应用集合级（Chamfer/depth/occupancy）几何约束；contact flag 由当前 cur_state + 当前 controller 现算、按当前行序、Stage-2 起点由 cache pred_pos 现算，不作为固定属性继承。补充：被吸收的误差进入共享网络权重，影响其他帧；用 36 相机重建做额外监督会突破 Config A 两相机的信息预算，baseline 与 tactile 必须同等对待；contact flag 用 controller 的 t 还是 t+1 语义不同（引出 U1.5）。→ PASS |
| 2026-10-03 | U1.5 | 讲解 controller 来源（URDF tactile 锚点 3×5/指、7mm 偏置、opening 截断与负间隙限制）、sample 切片、`_preprocess` 拼接时 controller 行比 object 行超前一步、controller 节点特征复制 Gaussian 0、30→10→2→1 按索引分组与 merged p2c 节点数、cluster cache 只按 scheme 命名 | Q1 全对：frame_idx=2 时 Gaussian prev/cur = local 10/20，controller prev/cur = local 20/30；merged_cur 比的是布@20 vs 手@30；因果 contact 应用布 cur@20 + controller prev@20。Q2 核心对（不报错；point_id↔row↔cluster 必须一致），**纠正**：controller `dis_split` label 数组只依赖 n，重生成 cache 得到完全相同的 `[0,0,0,1,…]`，所以“重新生成 cache”修不了问题；改点序改变的是每组的物理含义。列在外后连续 3 个 = 同一列的 3 行，每指 5 组，空间上反而更紧凑；`n//2` 仍按手指切开。Q3 对：四组控制变量消融。补充：先查 T11 的原始 opening 在截断前是否仍有信息，若原始信号本身饱和，“+opening”对照在夹紧段也无信息，需要别的本体信号；按阶段（夹紧帧）分段报指标。→ PASS |
| 2026-10-03 | U2.1 | 讲解 batch_preprocess 构出的 4 张图（12891/650/9/2）与 3 张 connect 图、各层节点顺序（controller 永远在前）、簇状态 = diag_volume 加权均值、pin_mask 用 max 聚合、边 = kNN16 + radius 过滤，static(template) ∪ dynamic(cur−static)、每步重建；G0 无边（forward_last_layer=False）、网络只跑 G2→G1、G3 只提供 anchor | Q1 基本对（先 kNN16 再 radius 过滤；controller→布 边要求 controller 簇进入布 receiver 的 top-16 且 <0.2；反向需单独判断，不保证双向）。**纠正**：每个 receiver 的入边上限约为 30，不是 16：static（template 上，kNN16 去掉自环后 ≤15）∪ dynamic-only（cur 上，≤15，`r_counts==1` 只保留不在 static 中的边，:586–591、:520–523）。补充：布内部 k 是主要约束（640 个直径 <2 cm 的簇，0.2 m 球内远超 16 个），所以 controller 要比布 receiver 的第 16 近邻还近才能连上〔估算〕；static 边按 template 算，template 时刻手和布相邻，整个窗口都会保留这条边，与当前距离无关。Q2 对；补充：新加的 G0 ndata key 不会自动聚合，`node_attr_scheme` 和 `init_node_features` 都是显式列出字段的，两处都要改。Q3 对。（原补充“pin=1 会触发 decoder 覆盖”**有误，已在 U2.3 更正**：`pre_predict` 没有转发 `apply_pin`，覆盖不会执行，见待解问题。）→ PASS |
| 2026-10-03 | U2.2 | 讲解 backbone 入口与 share_weight；node 11 维 = norm(−a_anchor+g) + norm(相对 anchor 的速度) + 常数 attr；G2 的 anchor_next = anchor_cur 导致 “acc” 退化，G1 的 anchor_next 来自 G2 预测；pin_mask 未进入节点特征；edge 9 维 = 方向 + 拉伸比 + 相对速度（normalize）+ Δθ cos/sin；Normalizer 在线 FP64 累积、跨层/节点类型共享 | 三题全对。Q1：g 是全局常数，并且与 −a 一起归一化，均值里也含 g，所以 g 被精确抵消（每次调用先累积再归一化，normalization.py:126–130，从第一次调用起就成立）；当 g 随样本变化、统计冻结后改 g、或 g 进入未归一化分支时才重新起作用。Q2：G2 的 “acc” = (A_cur − A_prev)/dt² + g，本质是上一步父位移/dt²；G1 的父项跨 t−1→t+1。补充：G2 上手和布的两个 root 时间跨度不同，手 root 是 traj t→t+1（即动作），布 root 是 t−1→t；窗口起点布 root 这一项只剩 g。Q3：pin 全 0 + attr 相同，tactile 补 0 就等于隐式的节点类型标志；先给两组都加显式 type/pin 特征，tactile 通道结构一致，baseline 填 0。补充：“B + type” 是一个新 baseline，原 baseline 也要一起报告；tactile 不能和布行共用 Normalizer（G1 有 640/650 是布行的 0，会把 tactile 的尺度放大），应单独归一化。→ PASS |
| 2026-10-03 | U2.3 | 讲解 h⁰/e⁰ 嵌入、一层 MP（边残差更新 → 入边 sum 聚合 → 节点残差更新，pre-LN）、16 层独立权重 G2/G1 共享、无跨层级/跨步隐状态（Markov）、未用的 emb_norm 与缺失的最终 LN、输出 `out_node` → decoder 入口；更正 pin 覆盖不执行 | 三题全对。Q1：孤立节点 16 层后只依赖自身初始特征；tactile 只放 controller 时需经消息边影响布。**精确化**：手和布在 G3 之前是两棵独立子树，anchor 链从不跨子树，“经 G2 手指预测 → G1 anchor_next 间接影响”只作用于手的子树，最终仍要经 G1 的 controller→布 边，**边是唯一通道**。Q2：sum 对入度敏感；新增 contact/tactile 边的消融应保持边集合不变、只改消息内容。Q3：固定墙钟窗口重采样直接拼接，或由 caller 做因果汇总；owner 分别是 forward_train 局部 buffer 和 single_gpu_rollout。补充：Stage-2 窗口起点需要窗口之前的真实 tactile（`video_range` 不覆盖），dataset 要单独读取；用 0 填充会造成训练/推理历史分布不一致，与 prev=cur 同类。→ PASS |
| 2026-10-03 | U2.4 | 讲解 11 参数 → F（四元数 + 默认 [1,0,0,0]、exp(clamp±5)/几何均值 → det F=1）、`next = A_next + F(X − A_tem)`、G2→G1→G0 的覆盖链（G1 cur/template ← c1，anchor ← P_j）与复合式 x = P_j + F1F2(X0 − X2)、Σ=(F1F2)Σ_tem(F1F2)ᵀ、G1 controller 被改写、momentum 只在 G1 | 待回答 |

## 待解问题（学习中发现，标注证据等级）

- [已解决 U2.4，见 NOTES §4.5] 论文 Eq.2 的 X：G2 层是 `template_state`（Stage-1 = 初始 Gaussian；Stage-2 = 窗口起点 cache）；更细层的 X 被 `fix_bug` 分支覆盖为粗层预测（simulator :364–370），所以每层 F 是相对修正，总形变 = F1·F2。
- [✓ 静态] G1 controller 簇的 `cur_state`/`template_state` 在 i=1 下传时被覆盖为 G2 的粗预测（:364 未排除 pin，pin 版本已注释 :368）。真实 traj[t+1] 只经 G3 手 root 均值和 G2 两根手指均值进入模型；G1 的 10 个 controller 簇看到的是“预测的手”。手指 F 只受间接监督（经 G1 边影响布的 render loss，以及 G1 momentum 正则）。**影响 M6**：contact/tactile 相关的 controller 位置要直接从 `controller_trajectory` 取，不能读图里的 ndata。
- [✓ 静态] 每层 det F = +1（单位四元数旋转 × Πs=1，`norm_volumn=True` 默认，acc_decoder.py:84–92、:305–309），所以 det Σ 全程等于 template 值；轴比最大约 e^10（clamp ±5）。
- [✓ 静态] 布 root anchor 固定在 R_cur（时刻 t），F=I 时预测 = template 整体平移到当前 root。t→t+1 的平移和自 template 以来的全部形变都只能由 F 作用在相对 parent 的 offset 上表达。
- [? U2.5] `get_shs_rotations_batched`（transformation_utils.py:40–48）复合成 R2·R1，而位置/cov 用的是 F1·F2，顺序不一致。D360 SH0 + `use_rotation=False`，推测无影响；U2.5 核对 :409–426 的 scene_rot 用法。
- [U3.1] 层级 momentum 正则只在 i>1（G1）计算（simulator :325–329）。
- [已确认 U2.1，见 NOTES §4.2] 论文写的层级是 `[n, n/2, 2]`。D360 用 `dis_split`，`downsample_rate` 实际上是 complete-linkage 的**距离阈值（米）**，取值 0.02 和 0.2（embodied_dataset.py:538–545）。由 T19 p2c 的 shape 可得 Gaussian 侧 12861→640→7→1，controller 侧 30→10→2→1。合并后每层节点数为 12891 / 650 / 9 / 2 / 1，两个 root 只在最顶层合并。图里怎么使用这些层，留到 U2.1。
- [对齐风险，静态确认] `cluster_mask` 和 `controller_mask` 的缓存文件名只由分组 scheme 决定（:495–499、:645–650），文件存在就直接复用。两侧风险不对称：controller 的 `dis_split` label 数组只依赖点数 n（与坐标、点序都无关），缓存只在 n 变化时才会过期；点序变化不会让缓存过期，但会改变每组对应的物理点。Gaussian 的 label 依赖 PLY xyz，换 PLY 后旧缓存会给出错误分组；N 不变时这个错误不会报错。服务器上实际存在哪些缓存文件，在 Mac 上没有核验。
- [观察] controller 第 0 层按**索引连续**每 3 个一组（:690–709）。例如 cluster 1 = {r00_c23, r00_c31, r06_c00}，跨越了手指两端，空间上不紧凑。T12 只验证了不跨手指。对动力学有什么影响未知。
- [表示限制，影响 M6] opening 在 136/194 帧被截到 0.04 m，7 mm 偏置下 186 帧的有符号间隙为负（T11）。夹紧阶段 controller 只表达刚体位姿，不表达继续闭合或夹持力度。如果 tactile 带来增益，可能有一部分只是在弥补这个缺陷。消融时可以用 candidate_5mm/0mm 作对照。
- [事实] 30 个 controller 点取自 URDF `tactile_{left|right}_{row}_{col}` 锚点（12×32 网格中的 3×5 子集，T10）。理论上 taxel 可以按 point_id 和 controller 行对齐，但 AGENTS.md 要求把 tactile 当独立特征处理，不建成 taxel 图节点。M6 再讨论。
- [待验证] 论文 Stage-2 “随机子序列”，官方实现是固定 `split_list` 窗口 + Stage-1 cache 起点；论文未写 cache 机制。U4.2 对照。
- [已记录限制] Continuous 每 10 帧 template 切换时 prev_state 被设为当前边界预测（速度历史清零），见 T31。U5.1 精读。
- [已确认] T30 Stage-2 未完成（用户 2026-10-03）；最新 epoch 编号未在 Mac 核验。
- [T-1 触觉任务定义，M6 必答] 预测 t→t+1 时 tactile 最多用到 t。但 SoMA 推理是 open-loop 模拟：若目标是“给新动作序列做模拟/规划”，未来 tactile 根本不存在，只能用 rollout 起点前的历史；若目标是“边观测边预测下一步”，每步可用到 t 的真实 tactile。两种设定的评估协议不同，消融前必须先定。
- [? U4.2] Stage-2 的 `scene_init_pos/cov` 是普通 dict 中 `requires_grad=True` 的 leaf，不在 state_dict/optimizer 中；梯度是否累积在这些 leaf 上、有无副作用，待 U4.2 核对。
- [设计原则，M6 复用] 学习者 U0.2 结论：tactile history 的 owner = rollout 调用者（训练 forward_train 局部 / 推理 single_gpu_rollout 循环变量），与 prev/cur 同生命周期；不放在模块属性里。
- [已核对 U1.3] `_parse_idx` 在 phase `train`/`all` 下（D360 config 三个 split 都是 `phase='all'`）构造时打乱一次相机顺序。cam/rgba/GT/mask/bbox 都由同一 `cam_list_idx` 生成（:818–819），排列一致；render loss 对相机取均值，对顺序对称；`save_image` 用 `cam.img_path` 取相机 id（test.py:251）。已读路径中未发现按索引假设相机身份的代码；指标汇总脚本未逐一核对。
- [? U4.2] 每个训练窗口起点 prev_state = cur_state（Stage-1 :690；Stage-2 `scene_init_prev_pos` 是同一 cache pos 的拷贝 :236）→ Gaussian 自身速度项为 0，即使布在运动。是否为官方有意设计、对 Stage-2 中段窗口的影响，U4.2 核对。
- [推导，非官方说明] gravity −39.2 在 Stage-1（物理一致约 −245）与 Stage-2（约 −2.45）都不物理一致；T6/T13 也声明非物理。**U2.2 结论〔✓ 静态推导〕**：g 只出现在节点特征 `−a_anchor + g`（meshgraphnet_embodied.py:376），对所有层、所有节点是同一个常数（dgl_graph.py:435–439，簇取均值后不变）；anchor_normalizer 的均值也含 g，所以 `((x+g) − (μ_x+g))/σ_x = (x−μ_x)/σ_x`，g 被精确抵消。`selfsup_loss=False` 时 g 不出现在别处（acc_decoder.py:250 只在 selfsup 中用）。所以当前 D360 config 下 gravity 的取值对模型没有影响；只有在多场景/多 g 混训、按节点施加外力、推理换 g 而统计冻结时才有意义。
- [✓ 静态] 节点特征里没有节点类型：`pin_mask` 读到了（:369），但没有拼进去（:387 TODO）；attr 对所有节点都相同（scene_attr 是 [1,5] 的 expand，controller 复制第 0 行，簇聚合后 density 不变）。encoder 只能靠运动学特征区分手和布；decoder 的 pin 覆盖实际也不执行（见下条），所以在 G2/G1 的整个网络路径里，pin_mask 只影响簇聚合，不影响计算。影响 M6：tactile 只拼在 controller 节点上（布补 0）会顺带引入一个隐式的类型标志，构成混杂变量。
- [观察，意图未知] G2（i=1）有 `anchor_next = anchor_cur`（simulator :302–303），所以 “anchor_acc” = (root_cur − root_prev)/dt² + g，是位移除以 dt²，不是二阶差分。G1 的 `anchor_next` 取自 G2 的预测（:320–324），所以 G1 的 vel = (cur−prev) − (parent_next − parent_prev)，父节点那一项跨 t−1→t+1 两步。未见官方说明。
- [估算 / ? U4.2] Normalizer `max_accumulations=1e6` 按调用次数计，每层每步调用一次。Stage-1 约 33k 步 × 2 层 ≈ 66k 次；Stage-2 若跑满 61 epoch × 600 it × ~12 步 × 2 层 ≈ 0.88M 次，合计约 0.94M，接近但不超过上限。统计按行数加权，主要由 Stage-2 和 G1（650 行）主导，并且混合了 controller 与布的行。
- [? M6] Mac 端文件未核实 tactile 采样率/时间戳；frame_manifest `tactile_filter_applied=false`。
- [✓ 静态 / ? 运行时] Gaussian 外观（opacity/SH/scaling/rotation）不在 optimizer 中（GaussianModel 非 nn.Module，存于普通 dict）。但这些张量 `requires_grad=True` 且参与渲染，`.grad` 可能每步累积而从不被 optimizer.zero_grad 清零——只占显存、不影响更新；未在运行时核验。U3.3 一并看。
- [? 官方注释] `forward_train:671` 的 `original_mov_cov = scene_gaussian.get_covariance() # TODO: here is not correct`，官方没有说明哪里不对。可能和 `get_covariance` 传入未归一化的 `_rotation` 有关，但 `build_rotation` 内部会归一化（general_utils.py:79–81），所以数值上应该没有问题。真实意图未知，U2.5 再看。
- [事实] rasterizer 的 `in_frustum` 会剔除 `p_view.z ≤ 0.2` 的 Gaussian（auxiliary.h:166）。D360 布的相机深度约 0.7 m，目前不受影响；如果以后加近距离相机（比如腕部相机），需要注意。
- [✓ 静态] G3（2 个 root 节点）也按 radius 5.0 建了边，但 `encode_decode` 只处理 lv_idx 2、1（:286–291），G3 的边没人用，它只通过 `_postprocess_hie` 给 G2 提供 anchor。最后一层 p2c（2→1）在 `batch_preprocess` 里只用来凑长度（:615、:640 只循环到倒数第二层），pin 用的 abs hierarchy 已被注释（:649–655）。
- [? MMCV] `group_cfg.max_radius=None` 时 QueryAndGroup 走 kNN（MMCV 1.7 标准实现，本机未读源码）。G2 只有 9 个节点，小于 sample_num=16，k>n 时 kNN kernel 的补位索引行为未核验；推测补位的是 0 号节点，经 radius 过滤和 unique 去重后，G2 实际上是“0.5 m 内的点全连接”。
- [观察，影响 M6] controller–布 之间没有专门的边类型。它们和布–布、手–手的边走同一套 kNN + radius 规则、同一种边特征；节点上能区分两者的只有 `pin_mask`。controller–布 交互最细只发生在 G1，G1 布簇直径 < 2 cm（complete linkage 阈值 0.02），controller 簇是每 3 个锚点的均值。
- [时间语义] dynamic 边在 merged `cur_state` 上建，其中 controller 簇处在 t+1、布簇处在 t（U1.5）。“手是否靠近布”这条边是用下一时刻的手和当前时刻的布判断的。static 边在 template 上建，两侧都处在 template 时刻。
- [✓ 静态 / ? 运行时，**更正 U2.1 的说法**] decoder 的 pin 覆盖不会执行：simulator 调 `decode_head.pre_predict(..., apply_pin=True)`（:309；stage2 :355），但 `AccDecoder.pre_predict(self, base_graph, **kwargs)` 调的是 `self.predict(base_graph)`，没有转发 kwargs（acc_decoder.py:123–124），而 `predict` 默认 `apply_pin=False`（:329）。所以 :324–325 的 `next_pos = … + anchor_next·pin` 永远不会走到。结果是 G2/G1 的 controller 节点也由 F 预测位置；G1 controller 簇的 `anchor_next` 来自 G2 对手指的**预测**，不是 traj[t+1]；G0 的 controller 行预测最后被切掉（:729）。是否为官方有意设计不清楚。U2.4 细看这对下传链的影响。
- [✓ 静态] 没有跨层级、跨步的隐状态：每个层级由 `init_features` 从物理量重新嵌入（meshgraphnet_embodied.py:429），层级之间、rollout 步之间只通过几何量（`pred_pos` → `anchor_next`、下一步的 cur/prev）传递。模型对 (prev, cur, template, controller) 是 Markov 的，**影响 M6**：tactile 历史只能作为显式输入窗口，或作为由 rollout 调用方持有的新增状态。
- [✓ 静态，观察] `pre_norm=True` 时 backbone 的 node_norm/edge_norm 是 Identity（:233、:251）；`MeshGraphNetEncoder.emb_norm`（:148）建了但 forward 没用（:166–168）；:168 注释说最后的 LN 由 head 负责，但 AccDecoder `pre_norm` 默认 False → `node_norm`=Identity（acc_decoder.py:65），D360 也没覆盖。所以送进 `dynamic_proj` 的是 16 层残差累积后未归一化的 `out_node`。backbone.forward 的 `connect_graph` 参数也没用（:425）。
- [? U3.3] Mac 未安装 MMCV，`OptimizerHook` 顺序（zero_grad→backward→clip→step）来自 MMCV 1.7 标准实现与 T26.5 记录，未在本机读源码。
