# T26.3 — random-init dynamics scale / normalization audit

Classification: **SCALE_MISMATCH_ATTRIBUTED**（首次 anchor normalization 的数值尺度失配）。

这不是“adapter 把毫米当米”的结论，也不是所有 renderer 内部 NaN 算术都已归因。已观测的 normalization cancellation → 巨大 encoder input → 饱和 deformation scales → 巨大位移/病态协方差链条成立。T26 仍 FAIL/blocking；T26.1/T26.2 保持原分类；本轮没有修复或重启训练。

## 运行范围与 provenance

- T26.2 checkpoint `afc4def87ee839bc27b9e9c8769a177e50c11ed7` 已提交/push，进入本诊断前两端 clean，local/origin/server 一致、ahead/behind=0/0。
- Slurm 25860（D360 coarse）、25861（official/T22 coarse）、25862（三路径 normalization fine）；全部 COMPLETED，amax / RTX 5090，elapsed 14/23/36 秒。各路径独立 Python 进程，seed=5。
- formal 两路径均通过真实 `tools/train.py` / runner 的首个 `run_iter`，仅在观测器内将 diagnostic rollout 设为1，正常 model train-mode first forward、loss；在任何 optimizer hook/backward 之前退出。没有训练 step、clipping、optimizer.step。配置文件没改。
- T22 路径调用原 smoke harness：保留其 T20→T21→T22 forward 顺序，在第一次 `Tensor.backward` 请求之前终止；本轮**没有重新测 T22 backward**。历史 finite backward 事实仍来自原报告。
- official control 使用服务器已有 `left_lift_1` 与 `environment/soma-stage1-isotropic-20260923/cloth_lift_stage1_scale3.py`，引用此前已验证的 `datasets/soma_sample/derived/left_lift_1/point_cloud_scale3.ply`（等价 isotropic scale3 compatibility 文件）。未下载/转换数据。官方实际 render shape 3×480×640，D360 3×360×640；各自保留原 camera、gravity、scene geometry，不作为同一个物理场景。
- expanded model 对比仅 scene/cluster/controller 配置不同；所有289个 construction state tensors 的 hashes，formal D360、official、T22 **exact 相同**。T22 `init_weights()` 的 changed_keys=[]，不能将差异归咎于重新初始化权重。
- `provenance.json`：794个 D360 asset hashes 全部不变，frozen config hash不变，server repo clean；包含 source/control-config/official-derived-PLY SHA256、全部 model config differences、Slurm accounting。
- `*-v1.json` 是 coarse 观测，`*-fine-v2.json` 仅进一步观测已定位 Normalizer 的方差；未给所有层加 hook。所有大 tensor 留在进程内，不落盘。JSON仅统计、hash、少量 row/9×3粗层输入。

## 官方 control 对比

| Metric | Official cloth_lift | Deform360 |
|---|---:|---:|
| N | 12169 | 12861 |
| initial bbox diagonal (m) | 0.90332156 | 0.52363909 |
| displacement p50 (m) | 0.09674681 | 50.638094 |
| displacement p95 (m) | 0.19537313 | 100.80446 |
| displacement p99 (m) | 0.22035127 | 109.79598 |
| displacement max (m) | 0.26686625 | 117.19542 |
| displacement/bbox p50 / p99 / max | 0.10710119 / 0.24393447 / 0.29542775 | 96.70419 / 209.67872 / 223.80953 |
| coarse anchor-acc input abs p99 / max | 33.851929 / 33.851933 | 39.207401 / 39.207401 |
| coarse normalized anchor feature abs p99 / max | 3.8078442 / 3.8078442 | 574874.88 / 574874.88 |
| all-hierarchy raw anchor-acc abs p99 / max (max across stages) | 59.131317 / 69.022614 | 14087.664 / 14087.664 |
| all normalized feature abs p99 / max (max across stages) | 4.0602218 / 42.747471 | 574874.88 / 574874.88 |
| coarse F singular-value p99 / max | 1.082109 / 1.082109 | 785.77229 / 785.77229 |
| predicted covariance PD fraction | 12169/12169 (100%) | 2644/12861 (20.5583%) |

这里 acceleration 是实际 **anchor feature**，不是 network predicted acceleration。网络预测 deformation parameters；不存在可报告的 output acceleration normalizer。跨阶段 p99/max 是各阶段统计的最大值，不是将不同 shape 混合后重算的全局分位数。

## 全体 Deform360 与 row3648

- initial xyz min `[0.06101363,-0.04881996,-0.06714770]`，max `[0.41370803,0.02706642,0.31238598]`，std `[0.08281216,0.00813558,0.09108298]`。
- predicted xyz min `[-26.37232590,-114.12248230,-28.51290512]`，max `[26.71943665,108.79503632,25.48235321]`，std `[13.21437590,55.50879068,16.08805997]`；bbox diagonal=235.42821258m。
- displacement min=7.54555742，p50=50.63809422，p90=91.15116025，p95=100.80446000，p99=109.79597523，max=117.19542015 m。
- >0.1、>0.5、>1、>5m 的数量均为12861。row3648 位移7.68263768m，仅处于全体 **0.29547 percentile（≤该值比例）**，不是最大位移离群点。
- row3648 coarse广播前 `[0.20144451,-0.00939316,0.18189050]`；coarse广播后 `[0.35693389,-0.56254065,-0.09181349]`；fine广播后 `[0.81589413,-0.44087982,-7.46397066]`。
- 对应 coarse ancestor 从 `[0.23217736,-0.00974389,0.16741940]` 预测为 `[3.70280337,-14.40290642,-5.78169298]`；fine ancestor 输出 `[0.36182427,1.75525093,-7.72865009]`。局部7.68m来自层级 anchors/F/template 的组合，不能把它写成9.8×dt²的积分。

## First scale-explosion boundary

`backbone.anchor_normalizer`，level2，输入9×3（7 object clusters + 2 controller clusters）。输入z接近-39.2但并非完全常量：两个controller为-39.2074013，其余为约-39.20001。

[normalization.py](../../../../mmgs/models/utils/normalization.py) `_std_with_epsilon` 99–103行：实际 float32 `E[x²]−E[x]²` 的 z 方差=**0**。同一已捕获float32输入用float64 centered recomputation，variance=**9.443712590354275e-6**，std=**0.0030730624123753614**。不是重跑/替换模型，只是CPU诊断算术。

实际 std floor=1e-8；mean_z=-39.20165252685547；最大 centered residual 约0.00574875，除以1e-8得到574874.875。错误的近零方差估计是第一处异常尺度放大。两个矩在约1536量级相减，真实微小方差被float32消减；不是输入NaN或神经网络先产生大数。

| Stage (level 2) | abs max | abs p99 |
|---|---:|---:|
| backbone.node_encoder.input | 574874.88 | 574874.88 |
| backbone.node_encoder.output | 51687.176 | 35434.174 |
| backbone.edge_encoder.input | 5.4882731 | 2.4222522 |
| backbone.edge_encoder.output | 0.60183281 | 0.29170011 |
| message_passing.input | 51687.176 | 35434.174 |
| message_passing.output | 51687.492 | 35434.357 |
| decode_head.dynamic_proj.input | 51687.492 | 35434.357 |
| decode_head.dynamic_proj.output | 2344.749 | 2344.7426 |
| decoder.raw_log_scales | 2077.2168 | 2077.2147 |

raw decoder log-scale约±2077触及原有[-5,5] clamp；随后 exp + volume normalization 产生约 `[785.7720,0.0356740,0.0356740]`。体积约束det≈1 **不等于** 变形大小受限。MP output与input同量级；它不是第一个放大边界。

## Normalizer / T22 路径

实际只有3个 Normalizer（共享于各hierarchy）：anchor3、node-velocity3、edge7，合计12个registered统计tensor。初始count=0、mean=0、std=1e-8，全部train/is_training=True、force_nonorm=False。每次forward先累积当前输入，再用累计统计归一化；无output-acceleration normalizer，attr为Identity。完整前后mean/std/count和input/output分布保存在各JSON的normalizers。

- D360粗层：anchor count0→9，std=`[0.60387236,0.82374626,1e-8]`；node velocity std=`[0.00114950,0.00010319,0.00013187]`，normalized max2.12198；edge count0→72，normalized max5.48827。node+anchor最终count659，edge9840。
- 官方粗层31个nodes：anchor std=`[0.19216210,0.35801372,0.29465660]`，normalized max3.80784；未触发Z方差归零。
- T22原harness先直接preprocess，T21 train-mode encode_decode/register_norm=True。该forward已累积异常coarse预测产生的fine features；没有optimizer更新也会改变normalizer。
- T22第二次forward之前anchor count=659，mean≈`[-23.9684,99.1251,1.9620]`，std≈`[2140.85,8901.01,3667.47]`；node std≈`[84.058,349.870,144.183]`，edge normratio mean≈455.298。第二次粗层 normalized anchor max0.0118011，位移p50=0.0433643m/max0.1409805m，PD=12861/12861。
- 这是“已被第一次异常forward改变的统计”掩盖了fresh first-forward问题；不能据此把预热forward作为已验证修复，也不能把历史T22称为formal第一批同状态backward。
- 其他实际区别：formal build model后dataset/RepeatDataset(50)、phase=all、runner batchedsample、完整forward_train；harness先dataset phase=train、manual batching、model.init_weights、手动encode_decode(num_epoch=0,num_iter=0)、额外T20 preprocess。formal `set_random_seed`含Python/NumPy/Torch，harness显式Torch/NumPy并设threads4；本审计三路进程OMP/MKL=4。construction hashes证明权重相同，边界处小浮点差异存在；并非据此证明所有中间tensor/layout bitwise一致。

## dt / 实际运动语义

- `embodied_dataset.py:927–934`：rawgravity `[0,0,-9.8]` × `(real_dt/comp_dt)^2` = ×4，identity rot_est不额外旋转；实际 external `[0,0,-39.20000076293945]`。
- `gs_simulator_embodied.py:85–87`：backbone/head dt=1/15。
- `meshgraphnet_embodied.py:374–386`：anchor feature = `−(anchor_next−2*anchor_cur+anchor_prev)/dt² + external`；relative velocity = `((cur−anchor_next)−(prev−anchor_prev))/dt`。
- `meshgraphnet_embodied.py:326–344`：edge velocity=(edge_delta_current−edge_delta_previous)/dt；ratio edge由unit direction、length/template_length ratio、relative velocity组成。
- `acc_decoder.py:296–320`：11维输出→U/log-scales/V→F，`next_pos=anchor_next+F@(template−anchor_template)`。
- `gs_simulator_embodied.py:350–370`：层级广播同类F位置变换并更新template。position生成**没有**另一个acc×dt²的Euler积分，也没有再次乘frame_gap10。
- model位置初始prev==cur；controller使用source113/123，提供sampled movement。frame_gap只负责dataset取帧；momentum loss中dt参与损失，不是额外推进状态。
- 因此未发现accidental ×10/×15/×30/×frame_gap；固定的×4保留。gravity绝对值较大、z向变化很小的组合触发了方差消减，但不授权改gravity/dt。

## Controller / edge scale

D360 controller首个sampled step位移范围0.0104804–0.0113629m，p50=0.0109165m；到最近object Gaussian距离0.00395046–0.0475880m，p50=0.0248028m。Gaussian最近邻p50=0.00205818m，p99=0.00700912m；object bbox0.523639m。

初始fine graph edge length p50=0.0200431/max0.0510373m；coarse p50=0.165508/max0.250799m。官方对应fine p50=.0266059/max.110699、coarse p50=.176991/max.347910m。这里是实际图边；leaf graph无边，不虚构leaf距离分布。raw边向量/长度及normalized边特征均在JSON；首个coarse normalized edge max5.48827，不是第一次爆炸源。未发现支持1000倍meter/mm错配的证据。

## Covariance 逐层传播

initial covariance全部12861正定，min eigen p0=9.62318e-9，p50=2.50782e-7，condition p99=98.2410/max105.323。仅观察原始 `F@(Σ@Fᵀ)`；float64分支复制同一initial matrix和实际FP32 F，在CPU重算，不回写model。

| hierarchy | arithmetic | PD count /12861 | min eigen p0 / p01 / p50 | condition p99 / max |
|---|---|---:|---|---|
| 2 | actual_float32 | 2454 | -1.327255e-06 / -3.5521661e-07 / -1.7184726e-08 | 1.5887575e+10 / 1.7947764e+13 |
| 2 | float64_recompute | 12861 | 1.2632162e-11 / 3.7447398e-11 / 4.1194439e-10 | 4.0640108e+10 / 4.8218643e+10 |
| 1 | actual_float32 | 2644 | -2.0515511e-06 / -4.392004e-07 / -2.4304726e-08 | 9.5211811e+09 / 8.2727043e+11 |
| 1 | float64_recompute | 12861 | 1.2861556e-11 / 3.7710764e-11 / 4.1926868e-10 | 3.5280405e+10 / 4.2647519e+10 |

第一次失PSD在 **level2/coarse F** 后（graph 7 object+2controller；leaf12861+30，fine640+10）。FP32上三角镜像对应实际packed covariance，strict eigen>0无新增容差；symmetrize平均结果也有非PD，数量另存JSON。float64两层均全部正定，但真实condition约1e10量级，属于高度病态并非“float64就几何合理”。

coarse F singular p99/max=785.772292，min≈0.0356703，det(F)范围1.000284–1.000476；fine singular p99=1.085818/max1.087776，min0.929197，det≈1。每层矩阵幅度、最小特征值/condition完整分位数均有记录。covstat的eigen/statistics用float64分析实际float32快照，不意味着实际训练使用float64。

## Renderer / 归因边界

D360实际positive radius数量023=385、009=975（不是严格等价于图像内可见/贡献像素）；positive radius p50/max分别651/18735与673/2665 pixels。官方三个camera均12169个positive radius，p50/max分别7/91、5/66、6/68。没有用这一前向审计再次backward，也不宣称已经定位native内部第一条NaN指令。

分类选择 **SCALE_MISMATCH_ATTRIBUTED**：normalizer浮点消减造成明确input feature尺度失配，能够解释异常预测量级；并非官方此seed/sample的正常random-init行为。formal与T22有明确normalizer统计历史差异，但相同权重初始化，无需假定未知模型初始化bug。

## 后续最小候选（未实施）

建议将T26.4范围优先限定为 **Normalizer方差估计的数值稳定性**：单独评估稳定的centered/online variance或提高统计精度，并同时做D360/official对照，保持几何、dt、loss、rollout不变。不是先修renderer、clamp covariance或添加warm-up。本轮未改变epsilon、precision、normalization、kernel或任何训练算法；是否修复需另行授权。

## 文件与状态

- 新增 `tools/deform360_adapter/audit_dynamics_scale.py`，本report及6个小型JSON观测报告、provenance。
- 新增 `docs/deform360/contracts/008-pink-cloth/episode_0/dynamics_scale_normalization_contract.json`。
- roadmap新增T26.3；T26FAIL/T26.1/T26.2历史保留，T27正文未动。
- SoMA / deform360-adaptation；本轮T26.3未commit/push，待后续Git同步。无server-only数据入Git；服务器repo保持clean。
