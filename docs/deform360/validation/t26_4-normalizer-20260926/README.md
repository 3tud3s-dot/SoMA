# T26.4 Normalizer numerical-stability regression

Status: **PASS** — 限于此次 numerical-estimator patch 与分层 regression；**不是正式 T26 训练 PASS**。原 T26 FAIL/nonfinite 事实及历史保留。没有恢复46-epoch训练，没有执行T27。

## Production diff

仅 `mmgs/models/utils/normalization.py`：四个nontrainable registered parameters初始化为float64；输入在square/sum/add前转统计dtype；mean/variance/std保持原累计矩公式；forward/inverse内部高精度算术，输出回输入dtype；修正错误注释。

API、四个key、shape、注册顺序、requires_grad=False、epsilon、max_accumulations、population moments语义、更新时机、train/eval/hierarchy sharing不变。没有改成buffer，没有改gravity/dt/controller/data/rollout/LR/loss/decoder/renderer/Welford/warm-up，没有clamp新增。

Production SHA256: `f78712dd0760fcf16a58431d7fdfd240a2e38cba89c400b405df93798f364220`，Mac/server一致。

## 执行与 gate

T26.3先以 `f5212eb76821ac2264e0b3b2d4fb0b4c4c55b26d` commit/push；进入T26.4前 local/origin/server一致，clean，0/0。

1. Slurm **25863**：纯CPU torch numerical/save/reload，两个独立Python进程，COMPLETED，3秒。
2. Slurm **25864**：official fresh forward，再D360 fresh forward，独立进程，COMPLETED，31秒。审阅全体分布/投影及两张最小review图后才允许下一gate。
3. Slurm **25865**：D360另一个fresh进程，formal tools/train.py首sample rollout1 combined backward，无clip/step，COMPLETED，17秒；observer计时10.523秒，peak allocated=1474113024B（约1.373GiB）。

GPU：amax RTX5090。所有GPU操作在Slurm allocation内。正式config文件未改；observer内只把本次诊断rollout设为1，并在runner第一个run_iter后终止，优化器hook未运行；不会启动46epochs。全部输入canonical Config A/seed5/source113→123，零历史normalizer，不使用T21预热。

## Regression 1：numerics

真实输入直接来自已提交T26.3 `d360-fine-v2.json` 的9×3数组（输入hash见unit-save.json），没有重新拟造。

- Z variance实算=9.44371277000755e-6，independent centered FP64 reference=9.443712590354275e-6。
- actual std=0.0030730624416056944，reference=0.0030730624123753614。
- normalized absmax=1.8708287477，finite，output float32。
- synthetic [-39.2,-39.2074] PASS；真正constant输入variance=0且std floor不变，finite。
- whole-batch vs 3 chunks：count/sum/squared_sum/mean/std容差内一致；_num_accumulations按调用次数分别1/3，这是保留的既有语义。
- eval冻结统计；no_grad+train仍更新；inverse输出float32并恢复输入。
- unit tolerances：mean atol/rtol1e-12；variance atol2e-12/rtol1e-7；std atol1e-9/rtol1e-6。完整结果在unit-save.json。这些仅用于纯数值单测，不作为renderer bitwise gate。

## Regression 2：dtype migration / optimizer mapping

旧文件：`outputs/deform360/t24-finite-recovery-v3-20260926/epoch_1.pth`。只读取，SHA256在unit-save.json。

- old→new：三组共12个FP32 statistics keys严格加载，无missing/unexpected；shape不变，FP64值exact等于旧值.double()。**不声称恢复旧统计丢失精度。**
- new→new：保存新FP64 statistics state，再独立Python进程reload，keys/dtype/tensors exact。state文件留server：`outputs/deform360/t26_4-normalizer-20260926/unit/new_statistics.pth`，不入Git。本次按授权验证state migration，不重复完整T24。
- actual模型全部289个registered parameter names/order与旧checkpoint state_dict一致；277个requires_grad参数顺序与旧T24 optimizer名单一致。按照该映射载入旧Adam，275个已有moment states shape匹配，其余两项本来没有Adam状态。
- actual formal runner optimizer各group的参数名列表已记录；本修复不删除或重排nontrainable统计parameters。没有调用任何optimizer.step。
- 修复前后统计dtype变化是预期行为；同key可加载不代表旧算法/新算法output bitwise一致。

## Regression 3：official cloth_lift before / after

使用T26.3既有left_lift_1 / scale3-compatible PLY / existing official config，相同seed5/formal第一forward、无backward。

| Metric | Before | After |
|---|---:|---:|
| anchor Z std | 0.2946566045 | 0.2946533365 |
| normalized anchor abs p99 | 3.807844162 | 3.8078866 |
| normalized anchor abs max | 3.807844162 | 3.8078866 |
| displacement p50 (m) | 0.0967468105 | 0.09674681042 |
| displacement p95 (m) | 0.1953731284 | 0.1953729911 |
| displacement p99 (m) | 0.2203512675 | 0.2203510531 |
| displacement max (m) | 0.2668662528 | 0.2668659188 |
| displacement/bbox p50 | 0.1071011864 | 0.1071011863 |
| displacement/bbox p99 | 0.2439344724 | 0.243934235 |
| displacement/bbox max | 0.2954277473 | 0.2954273776 |
| coarse raw log-scale absmax | 0.08570549637 | 0.08570531011 |
| coarse F singular p99 | 1.082109041 | 1.082109062 |
| coarse F singular max | 1.082109041 | 1.082109062 |
| covariance PD count | 12169 | 12169 |
| covariance min-eigen min | 1.460346551e-09 | 1.460344766e-09 |
| covariance min-eigen p01 | 1.38262474e-07 | 1.382622682e-07 |
| covariance min-eigen p50 | 1.745698597e-05 | 1.745695349e-05 |
| covariance condition p99 | 1.651656798 | 1.651660229 |
| covariance condition max | 1.677790063 | 1.677793235 |

coarse anchor std三通道修复前 `[0.19216210,0.35801372,0.29465660]`，后 `[0.192142435796,0.357945296854,0.294653336504]`，与同batch FP64 centered reference匹配。coarse variance最大绝对reference误差约2.85e-13，优于原float32估计。

位移max变化约3.34e-7m，p99变化约2.14e-7m；PD保持100%。三camera positive radius均12169，radius p50/max仍7/91、5/66、6/68，未见尺度异常漂移。render均finite；统计与baseline的对照保存于JSON（不是bitwise或逐像素exact承诺）。

## Regression 4：D360 fresh first-forward

| Metric | Before | After |
|---|---:|---:|
| anchor Z std | 9.999999939e-09 | 0.003073062442 |
| normalized anchor abs p99 | 574874.875 | 1.870828748 |
| normalized anchor abs max | 574874.875 | 1.870828748 |
| displacement p50 (m) | 50.63809422 | 0.04305410232 |
| displacement p95 (m) | 100.80446 | 0.104867709 |
| displacement p99 (m) | 109.7959752 | 0.1277848699 |
| displacement max (m) | 117.1954201 | 0.1407891122 |
| displacement/bbox p50 | 96.70419047 | 0.082220948 |
| displacement/bbox p99 | 209.6787224 | 0.2440323354 |
| displacement/bbox max | 223.8095333 | 0.2688666968 |
| coarse raw log-scale absmax | 2077.216797 | 0.0843360275 |
| coarse F singular p99 | 785.7722921 | 1.080700488 |
| coarse F singular max | 785.7722921 | 1.080700488 |
| covariance PD count | 2644 | 12861 |
| covariance min-eigen min | -2.051551079e-06 | 8.272876455e-09 |
| covariance min-eigen p01 | -4.392004037e-07 | 2.317261473e-08 |
| covariance min-eigen p50 | -2.430472585e-08 | 2.363799187e-07 |
| covariance condition p99 | 9521181098 | 132.3945607 |
| covariance condition max | 8.272704264e+11 | 160.4210185 |

fine F最大奇异值约1.08795；不是通过强行限制F得到，decoder/原clamp完全没变。

| Camera | Positive radius before→after | radius p50/max before→after | projected centers in image | invalid projected covariance |
|---|---:|---|---:|---:|
| 023_cam0 | 385→12861 | 651/18735→9/28 | 12861/12861 | 0/12861 |
| 009_cam1 | 975→12861 | 673/2665→11/36 | 12763/12861 | 0/12861 |

对所有positive-native-radius Gaussians（修复后即全部12861），按T26.2已核验installed formula做CPU FP32 cov2D重算，检查finite、regularized determinant/eigen/conic；不是只检查row3648，也不假装读取native内部矩阵。023/009最小regularized det分别约0.14805/0.13309。native forward RGB均finite。

[023 review](camera_0_review.png) / [009 review](camera_1_review.png)：每张依次为target123、random-init prediction、overlay。人工查看物体尺度与投影区域合理，无原来的巨大splat/出界爆炸，但仍有随机初始化导致的旋转、形变、位置和局部孔洞误差；009部分边缘略出画面。**这不是预测准确性或训练效果PASS。**

## Regression 5：combined backward

fresh formal路径再运行一次forward；只对真实runner解析后的combined loss backward。

- total loss=23109.26171875；momentum=6.7191066742，SSIM=4094.86572265625，L2=19007.67578125，static=0；所有component/metric finite。
- 275个有梯度parameter tensors全部finite：NaN=0、+Inf=0、−Inf=0。
- global gradient norm=92198.3961772，未clip、未step。梯度较大，不能据此保证长期训练稳定。
- `backbone.encoder.emb_norm.weight/bias` 两个requires_grad参数仍无gradient（当前forward不消费），未伪报277项均有梯度。
- 两camera native backward incoming color/depth及所有返回梯度都finite。grad_means3D / grad_covariance也全部finite，row3648不再出现原NaN。

## 数据完整性 / limitations

794个canonical asset hashes未变，T25 config hash未变。server唯一tracked modification是normalization.py，与Mac同hash；工具在server outputs执行，未复制大数据到Mac。unit state .pth留server。

旧checkpoint加载只保留旧已存数值；旧受异常forward影响的统计不会被自动纠正。本轮fresh regression不代表允许用旧统计直接继续训练。FP64 moments仍有极端尺度下的理论消减风险；没有切Welford。multi-GPU/statistics同步、half()路径未验证；本次未改变语义。

T24/T24.5数值重复性limitation保持；本轮不宣称bitwise renderer。官方对照与D360均仅首步；rollout3/6/9、46epochs、OOM和长期稳定性均未验证/重跑。T26历史FAIL不改写，本项PASS仅满足T26.4授权gate。

## Evidence format

`official.json` / `d360-forward.json` 的 `variance_audit.actual_fp32_raw_variance` 继承旧observer字段名，但本次由FP64 statistics计算，**实际dtype=float64**。随后仅修正observer字段名；`d360-backward.json` 改为 `actual_statistics_raw_variance` 并显式记录dtype。没有因此改生产patch或重跑forward gate。两个observer版本/hash可追溯到server outputs；本轮Git工具为修正标签后的版本。

`unit-save.json` 的std字段是reference；actual std另见d360-forward normalizers.after.std；单测已断言二者满足明确容差。Covariance eigen diagnostics用float64分析真实float32模型输出，不代表模型covariance改为FP64。

## Git

SoMA / deform360-adaptation。T26.3已push；T26.4的production patch、tools、contract、reports和roadmap未commit/push。服务器暂留同一未提交production patch，后续同步前须核验，不能盲目覆盖。未启动正式T26，未执行T27。
