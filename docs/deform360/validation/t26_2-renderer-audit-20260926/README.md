# T26.2 — row3648 renderer degeneracy causal audit

2026-09-26。**PARTIALLY_ATTRIBUTED**。T26 保持 FAIL / blocking。本轮未修复、未重启正式训练、未执行 T27、无 optimizer.step/clipping。

已将 T26.1 以 `1abfde56066f40140ab4f7dcd49b5799fbd13ac3` 提交并 push。Mac clean / ahead-behind 0/0 后开始；服务器 clean fast-forward 到同一提交。诊断范围仅 seed5、formal `tools/train.py` random-init、Config A、rollout1、source113→123、row3648（zero-based，12861 个 Gaussian 中的第3649个）。

## 结论与归因边界

已隔离导致原始 NaN 的 row；已证明预测位置、数值非正定协方差与 023 投影的组合处于退化域。尚未证明 native kernel 内**具体哪条算术指令**首先产生 NaN，故不标 ATTRIBUTED。

原始输入中，finite color/depth adjoints → native backward 返回 row3648 `grad_means3D` 3 NaN、`grad_covariance` 6 NaN，均 0 ±Inf；其余返回梯度 finite。传播到275个 model parameter tensor，共2,466,339 NaN。009单独backward全部finite。

这不是一个只看 `det>0` 就能排除的问题：3D covariance 的两个特征值均为微小负数，det反而为正；023投影放大后，+0.3仍不足以使二维 covariance 正定。不能将它简单归为只由mean或只由cov导致。

## 实验与可比性

- 最终 Slurm **25858**：original / exclude / restore mean / restore covariance / restore both / original009，6个独立Python process，RTX5090 / amax。此前25856/25857同样完成全部probe。最终probe各约14秒；最大allocated约1.4GiB，具体见[probe summary](probe_summary.json)。
- 289个初始model registered tensor与T26.1 exact；6个有效probe的初始化、sample、Gaussian、camera矩阵、**干预前**全部render tensor hashes exact。[核验](identity_verification.json)。
- 调用真实 L2Loss（weight .9，sum），只选择指定camera；不改变损失定义。单camera数值未再除2，而formal两camera aggregate取平均。原始023 L2=42903.05078125、009=24598.67578125，其均值=33750.86328125，与T26.1一致。
- 初次observer job25855在进入renderer前失败：误把scene中的numeric camera目录当成camera ID字符串。仅修正诊断工具的路径读取，用numeric index加T26.1精确view hash校验；保留[失败report](reports/initial_tool_camera_path_error.json)、[log](logs/initial_tool_camera_path_error.txt)、[旧工具](probe_v1_camera_path_error.py)。这不属于模型/renderer失败。

- 收尾复核纠正了CPU重算off-diagonal索引：GLM `cov[0][1]`对应NumPy `[1,0]`；矩阵在float32计算后不保证两侧位级相同。初轮native结果没有依赖该重算。修正后job25858重做6个独立probe，finite/nonfinite和native几何结果相同；[初轮文件](preliminary/)保留，最终表格/contract引用reports中的最终版本。mean-only梯度norm在两轮为333.04024 / 3389.70381，虽都finite但并不稳定；不在本轮扩展重复性追查，不把finite说成数值问题解决。

## 世界空间 row3648

| 项目 | initial113 | predicted step1 |
|---|---|---|
| mean xyz | `[0.201444507,-0.009393156,0.181890503]` | `[0.815894127,-0.440879822,-7.463970661]` |
| covariance eigenvalues | `[3.05919e-7,1.49701e-5,2.51038e-5]` | `[-1.02859535e-7,-4.99623997e-8,12.27248055]` |
| determinant | `1.14966164e-16` | `6.30696176e-14` |
| trace | `4.03797842e-5` | `12.2724803984` |
| 2-norm condition | `82.0602` | `245634329.337` |
| symmetric / PSD / PD | true / true / true | true / false / false |

位移delta=`[0.61444962,-0.431486666,-7.645861149]`，norm=`7.68263769m`。预测packed cov（xx,xy,xz,yy,yz,zz）：

```text
[0.4963315427, -2.4002785683, 0.2890297472,
 11.6078376770, -1.3977587223, 0.1683111787]
```

重建矩阵：

```text
[[ 0.4963315427, -2.4002785683,  0.2890297472],
 [-2.4002785683, 11.6078376770, -1.3977587223],
 [ 0.2890297472, -1.3977587223,  0.1683111787]]
```

opacity=`0.9688374996`；SH0 DC=`[1.29549503,-1.35197484,0.126461312]`；实际precomputed RGB=`[0.865452409,0.118614912,0.535674095]`。完整矩阵、delta、hash见[original report](reports/original.json)。光谱用float64分析**实际float32 packed值**，不是修正后的covariance。

全体12861中的ascending rank / percentile≤：

| 指标 | rank | percentile |
|---|---:|---:|
| world position norm 7.52136 | 34 | 0.2644% |
| min eigenvalue | 2272 | 17.6658% |
| max eigenvalue | 11702 | 90.9883% |
| determinant | 12248 | 95.2337% |
| condition number | 7954 | 61.8459% |

row并非世界空间condition的最极端outlier。initial全部12861 covariance PD；prediction仅2644 PD，其余10217 non-PD。不能把总体几何问题说成只有一颗坏Gaussian。

## 两camera的真实可见性与投影

| 项目 | 023_cam0 | 009_cam1 |
|---|---:|---:|
| camera xyz（float64复算） | `[7.16725994,1.95641031,0.22598191]` | `[-6.41115519,0.20323192,4.64898571]` |
| native depth | 0.2259818912 | 4.6489858627 |
| initial depth | 0.6072165343 | 0.4721001107 |
| x/z, y/z | 31.71608, 8.65738 | -1.379044, 0.0437153 |
| native pixel | 14330.5820,4014.5415 | -365.8921,201.1028 |
| native radius | 18735px | 1134px |
| tiles touched | 920 | 920 |
| total positive-radius Gaussians | 385 | 975 |
| row z rank among positive-radius | 1 / 385 (0.2597%) | 275 / 975 (28.2051%) |
| native conic `(A,B,C)` | `[-.16297205,-.47731861,-1.39798832]` | `[3.01981330,-1.01423931,.34065181]` |
| native opacity after AA | 1.12547779 | .00484418729 |

positive-radius z分布（min / p01 / p05 / median / p95 / p99 / max）：

- 023：`.225982 / .449710 / 1.088693 / 4.865810 / 9.416477 / 11.265868 / 13.868164`。
- 009：`2.291931 / 2.688222 / 3.276300 / 5.582719 / 12.822150 / 13.765851 / 14.224573`。

真正native near判据是`view.z>0.2`（auxiliary.h:166），不是Camera对象的znear=.01。row023离culling边界约0.026m。中心远在图像外，但半径极大，tile矩形覆盖40×23全部920个tiles，因此仍进入preprocess/backward。**tiles/radius>0不等于实际有非零pixel贡献**；原始两个render均全黑。

clip齐次坐标、J、T、view/proj矩阵与hash、near-pass分布均在original JSON中；使用进入native的实际矩阵，不另换相机约定。

## 安装实现与fp32 / fp64投影重算

[安装来源](installed_source_provenance.json)：

- rasterizer子模块commit=`9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0`，clean。
- installed `.so` 与保留build `.so` SHA256均为`139c2a96118f26f5af40d2d12567c4dec37249142ece70cfcb29483485d9fc50`。
- direct_url指向已有本地wheel，build.ninja明确当前source paths，CUDA12.8/sm120。
- 因而用**当前安装的本地构建证据**；未假定互联网上upstream自动等于安装版。

[源码摘录](installed_source_excerpt.txt)：forward.cu:73–113先限制x/z、y/z至±1.3*tanFoV，构造J、W、T，算`T^T Sigma T`；:219–240加0.3，求det/conic，**只检查det==0，不排除det<0**；:246–255算radius/tile bounds。backward.cu:145+重算投影，:211–241为AA项，:249+为conic inverse gradient。

CPU NumPy复算从实际float32输入分别提升/保持float64/float32，遵循源码矩阵方向与乘法关联；**没有保证与nvcc/GLM/FMA位级一致**。真实native只观测到depth、means2D、radii、conic/opacity和tiles。读取geomBuffer前缀的布局有rasterizer_impl.cu:155–165依据；被cull行不读未初始化字段。

| camera / precision | raw `(a,b,c)` | raw det | regularized det | regularized eigenvalues |
|---|---|---:|---:|---|
| 023 / fp32 | `[34925296,-11924632,4071456.25]` | -33554432 | -33554432 | `[-.63907372,38996753.1391]` |
| 023 / fp64 | `[34925278.8317,-11924624.3563,4071453.2887]` | -24680705.7813 | -12981686.0938 | `[-.33289163,38996733.0533]` |
| 009 / fp32 | `[14466.86914,43073.80469,128248.39844]` | 128 | 42880 | `[.30054077,142715.56372]` |
| 009 / fp64 | `[14466.86835,43073.81049,128248.41323]` | -239.6049376 | 42575.0695374 | `[.29832110,142715.58326]` |

023加0.3的fp32首对角实际不变（该量级ULP=4）；第二对角仅加0.25。即使float64完全加0.3，仍有负eigenvalue。009 raw det在两种精度间翻符号，但regularized covariance两者均有效。023 det不是绝对值接近0，而是大数相减、相对于`a*c`约1.42e14很小且为负；不可只用absolute det判断。

| camera / precision | regularized condition | conic `(A,B,C)` | radius-related λmax |
|---|---:|---|---:|
| 023 / fp32 | 6.10207e7 | `[-.12133886,-.35538173,-1.04085493]` | 38996752 |
| 023 / fp64 | 1.17145e8 | `[-.31363057,-.91857285,-2.69035001]` | 38996733.0533 |
| 009 / fp32 | 4.74863e5 | `[2.99087453,-1.00451970,.33738732]` | 142715.5625 |
| 009 / fp64 | 4.78396e5 | `[3.01229604,-1.01171439,.33980375]` | 142715.5833 |

扫描全部positive-radius行的fp32复算：023仅row3648的regularized2D不合法，009没有。实际023 native conic也非正定。具体NaN来源仍不能只凭这些结果确定：AA backward有分母平方/相消敏感表达式，但本次CPU复算**没有证明实际kernel发生0/0**；不把推测写成确定根因。

## Causal ablation

| Probe | mean | covariance | 023 native / model gradients | radius | model grad norm |
|---|---|---|---|---:|---:|
| original | predicted | predicted | NaN / NaN | 18735 | 非有限 |
| exclude row3648 | absent | absent | finite / finite | — | 0 |
| restore mean | initial | predicted | finite / finite | 5289 | 3389.70381 |
| restore covariance | predicted | initial | finite / finite | 0 | 0 |
| restore both | initial | initial | finite / finite | 12 | 0 |
| 009 control | predicted | predicted | finite / finite | 1134 | 0 |

所有finite项NaN/+Inf/-Inf均0。完整返回tensor统计在[各probe报告](probe_summary.json)。

- exclude使NaN消失，证明此输入中row3648是被隔离的offending Gaussian；其他12860行保持相同。
- restore mean仍触及920tiles且model有非零finite梯度，其regularized2D eigenvalues≈`[.26559458,3107028.3162]`，det≈825209.87。原predicted covariance在不同projection下不必出NaN。
- restore cov的2D eigenvalues≈`[70.36613,123.89179]`、det≈8717.786，但足迹落在图像外，radius=0；其finite有culling混杂因素，不能宣称单纯covariance替换已经证明kernel对该投影可微。
- restore both恢复正常投影，radius12、6tiles。mean/cov均作为常数替入，model零梯度不能当成动力学训练通过。
- 两个单独替换都finite，因此不满足用户表中“必须both才成功”的狭义Interaction分类，也不支持独占的Mean-only或Covariance-only；结论为**组合几何/投影退化已隔离，精确native算术部分归因**。

## pred_cov 来源

- `mmgs/models/heads/acc_decoder.py:289–326`：FFN输出11维，U quaternion / 3 log-scale / V quaternion；log-scale clamp[-5,5]→exp，按乘积三次根归一化；quaternion加identity offset再normalization。
- `mmgs/models/utils/deformation_gradient.py:104–109`：`F=U diag(s) V^T`。
- `mmgs/utils/transformation_utils.py:18–20,33–38`：层级依次执行`F Sigma F^T`，再取upper6；此处不是仅旋转，而是完整deformation gradient。
- `mmgs/models/simulators/gs_simulator_embodied.py:393–411`：更新cur_cov，去掉前30 controller nodes，传给renderer。

不是网络直接预测packed cov，也没有cov residual相加。该合同变换在精确算术中保PSD；**最终fp32 packed结果没有PSD/eigenvalue检查或下界**。初始PD、最终大量non-PD已实测，但本轮未逐层追溯第一次PSD丢失，不能声称具体某层roundoff已确证。

## 后续最小候选（未实施）

若单独授权T26.3，可先评估forward/backward共享的**projected covariance正定性检查及显式异常处理**，以免非法conic进入求导；这会改变无效输入处理/监督，必须单独验证，不能仅凭gradient finite接受。另一独立候选是有限精度下保持PSD的covariance传播。极大random-init位移及整体近秩亏几何仍须记录，不能由局部保护掩盖。

未加epsilon/clamp/nan_to_num，未改AA/near-plane/renderer/precision/network init/loss/LR/schedule；未永久删除任何Gaussian。

## 完整性与文件

794个源/派生资产hash全部未变；服务器SoMA clean，config与T25 hash不变。原Gaussian对象所有字段hash未变；所有model state finite；未保存checkpoint。[完整性](final_integrity.json)、[Slurm](slurm_accounting.txt)。

工具：[audit_renderer_degeneracy.py](../../../../tools/deform360_adapter/audit_renderer_degeneracy.py)。
Contract：[renderer_degeneracy_contract.json](../../contracts/008-pink-cloth/episode_0/renderer_degeneracy_contract.json)。
报告只含小型JSON/hash/文本，没有tensor dump或dataset本体。服务器执行副本位于`/data1/userdata/tcweng/projects/tcgs/outputs/deform360/t26_2-audit-20260926/`。本轮T26.2未commit/push，Mac新增Git-managed候选文件等待确认；服务器repo未写入未提交代码，工具在outputs下执行。
