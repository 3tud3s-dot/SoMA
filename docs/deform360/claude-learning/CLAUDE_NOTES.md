# Claude 学习辅助文档：SoMA-D360 长期笔记

> **Claude 学习辅助文档**。只记需要长期记住的结构、公式、入口和易混点；随学习修正。不是工程 contract，数值以 `../contracts/` 和源码为准。
> 标记：〔论文〕论文方法；〔官方〕官方实现；〔D360〕本项目适配决策；〔✓〕已由源码/contract 核对；〔?〕待验证。
> 源码基线 `aeeac82`；行号使用前重新核对。

## 0. 一句话 pipeline

〔论文 Eq.4〕`G_t = φθ(G_{t-1}, G_{t-2}, R_t)`：用前两步 Gaussian 状态 + 当前机器人动作预测下一步状态；渲染后与真实图像比较训练（Eq.5）。推理时只给初始 Gaussian 与动作序列，自回归滚动，不看未来图像。

```text
训练: 初始Gaussian(PLY) + controller轨迹 + 相机 ──► rollout k 步(每步: 建图→GNN→F→pos/cov→渲染)
                                         └──► 每步渲染 vs masked GT 图像 + momentum ──► 聚合 loss ──► 一次 backward/step
推理: 初始Gaussian + controller轨迹 + 相机 ──► 每步预测喂回下一步(no_grad) ──► 渲染只用于算指标/出图
```

## 1. 训练 vs 推理：输入、监督、状态来源〔✓ U0.1〕

| | 模型输入（条件） | 监督 / 只用于评分 | 状态来源 |
|---|---|---|---|
| Stage-1 训练 | 初始 Gaussian（source113 PLY）、controller `[T,30,3]`、2 相机、gravity、p2c 分组 | masked RGB `gt_label`（target = frame_idx+1）+ momentum | 第 1 步 = 初始；之后 = 上一步 prediction（detach） |
| Stage-2 训练 | 同上，但起点 = Stage-1 cache `frame_<gs_aligned_frame>` | 同上，dense gap1 | 第 1 步 = cache；之后 = prediction |
| Continuous 推理 | 初始 Gaussian + 全程 controller + 相机 | GT 只用于指标 | 全程 prediction；每 10 帧把当前预测登记为新 template |

易混：controller 轨迹是“已知动作”，全程可见是合法的；未来 **图像/Gaussian 重建** 不得作为状态输入。

## 2. 时间与样本〔D360 ✓〕

- source = local + 113；train local `[0,155)` = source 113…267；test local `[155,194)` = source 268…306（split_contract）。
- Stage-1：`split_list [[0,155]]`、dataset `frame_gap=10` → 16 帧 local 0,10,…,150；targets source 123…263，共 15 个 transition。
- Stage-2：15 个窗口 `[0,13),[10,23),…,[140,150)`，dataset `frame_gap=1`；每窗口起点 = cache `frame_<起点 local>`。
- 时间尺度〔官方 / D360 T6 ✓〕：两阶段 `model.dt = backbone.dt = decoder.dt = 1/15`（官方 `1/30*10/5`），是**内部数值约定，不是物理秒**：Stage-1 一步实际 ≈0.333s，Stage-2 一步 ≈0.033s，但 dt 都是 1/15。dataset `env_cfg.dt=1/30`、`real_dt=1/15`，只用于 gravity 乘 `(real_dt/dt)^2=4` → external `[0,0,-39.2]`（embodied_dataset.py:921–934）。
- `meta.seq_num = video_range[1]-video_range[0]`，Stage-1 下等于 10（步长），**不是帧数**。

## 2.1 时间轴卡〔✓ U1.2〕

| 轴 | 定义 | 谁用 |
|---|---|---|
| source | `aligned_timestamps.txt` 0-based 行号，113…306 | 原始数据、contract 对照 |
| local | source − 113，0…193 | split、cache key、`gs_aligned_frame` |
| sample k | `video_range` 中的位置（`embodied_dataset.py:826`） | 模型内部唯一可见的时间 |
| 墙钟 | 微秒时间戳（T6）：dense 均值 0.0333s（0.0309–0.0370），coarse 均值 0.3333s | 代码不读；只用于审计 |
| model dt | 常数 1/15 | backbone 速度/加速度特征 |

- 换算：local = `video_range[k]`；Stage-1 local=10k；Stage-2 窗口 i local = 10i + k。
- 覆盖：Stage-1 监督 local 10…150（151–154 未用）；Stage-2 窗口 `[10i,10i+13)` 相邻重叠 3 帧，末窗 `[140,150)`；local 150–154 不作 dense target；cache key 150 不作起点（T30.1）。
- dt 用法：`meshgraphnet_embodied.py:330` 相对速度 Δ/dt、`:374` anchor_acc = (next−2cur+prev)/dt² + external、`:382` vel /dt → Normalizer → 网络。两阶段 dt 相同但物理步长差 10× → 同物理速度下 Stage-1 速度特征约为 Stage-2 的 10×；Stage-2 继承 Stage-1 FP64 Normalizer 并继续累积（T30.1）。
- gravity〔推导〕：与 anchor_acc 同单位应为 `g·(τ/dt)²`：Stage-1 ≈ −245、Stage-2 ≈ −2.45；实际两阶段都用 −39.2 → 固定偏置特征，非物理（T6/T13 同样声明）。
- 外观不可学〔✓ 静态〕：`GaussianModel` 是普通 class（`data_preprocess/.../scene/gaussian_model.py:30`，非 nn.Module），存在普通 dict `gs_scene_dict`（simulator :189）里；其 `_xyz/_features_dc/_opacity/_scaling/_rotation` 虽是 `nn.Parameter`（:328–333），但不注册进 simulator → `build_optimizer(model)`（apis/train.py:76）不包含它们。可学的只有网络权重与 `scene_attr`（:207）。
- 窗口起点 prev = cur（Stage-1 :690；Stage-2 `scene_init_prev_pos` 拷贝 :236）→ Gaussian 自身 cur−prev = 0。〔U4.2 精读〕

## 3. Config A 实际 sample（T19 contract）

Stage-1，collate 前：`img` 2×`[3,360,640]`；`gt_label` 2×`[16,3,360,640]`（object-masked RGB）；`full_gt_label` 同 shape；`controller_trajectory [16,30,3]`；`controller_img_mask_list` 2×`[16,1,360,640]`；`bbox [2,16,4]`；`p2c_mapping` 4 层 `(30,12861)/(10,640)/(2,7)/(1,1)`；`external [0,0,-39.2]`；`gs_aligned_frame=0`。N=12861 Gaussians。collate 后多 batch 维 1，所以代码写 `gt[0, idx]`。

## 3.1 Sample 字段卡：按“角色”分组〔✓ U1.1〕

产生：`embodied_dataset.py::__getitem__:798–974`；`collate:1074`（pop `cam`/`scene_name` → 通用 collate 加 batch 维 1 → 放回）。所有时间维都由同一个 `video_range = range(split[0], split[1], frame_gap)`（:826）切出，所以“第 i 帧”在 GT、mask、bbox、controller 里指同一时刻。

| 角色 | 字段（Stage-1 shape，collate 前） | 被谁消费 |
|---|---|---|
| 条件输入（动作/环境） | `controller_trajectory [16,30,3]`；`external [1,3]`=`[0,0,-39.2]`；`volume_scalar [[512]]`；`p2c_mapping` 4 层 | `forward_train:662–682`：`[t]`/`[t+1]` 作 controller prev/cur；`diag_volume=prod(scaling·512)`；gravity 进图特征 |
| 条件输入（观测几何） | `cam` 2 个相机对象 | 渲染投影；`to_device` :650 |
| 监督 | `gt_label` 2×`[16,3,H,W]` = I·M（object-masked RGB，背景黑）；`controller_img_mask_list` 2×`[16,1,H,W]`（D360 全 0）；`bbox [2,16,4]`（全图） | `forward_train:705–723` 取 `[0, t+1]` → `AccDecoder.forward_train:193`：`mask_weights=1-mask`、bbox 裁剪 |
| 只诊断/出图 | `full_gt_label`（未 mask RGB）；`img`（静态 rest 图，CHW；训练只用来取 `.device`）；`pure_robot_img_mask_list`（全 0，代码中未消费） | `apis/test.py:185,230` `save_image` |
| 路由 / 索引 | `gs_aligned_frame`=`video_range[0]`；meta `scene_name/seq_idx/idx/eval_start_frame/seq_num` | Stage-1 不用 `gs_aligned_frame`；Stage-2 用它选 cache key（:718）；`scene_name` 选 PLY/attr；`eval_start_frame` 决定从第几步开始记指标（test.py:191,208） |
| **不在 sample 里** | 初始 Gaussian pos/cov/SH/opacity、`scene_attr`（可学习 `nn.Parameter`）、prev/cur 状态 | 模型 `__init__` 从 PLY 载入（:196–207）；状态由 rollout 循环持有 |

要点：
- 帧数来源是 `gt_label[0].shape[1]`（训练 :689 `num_frame=min(T-1, rollout)`，test.py:179），**不是** `meta.seq_num`（=步长 10，test.py:180 已注释掉）。
- index 0 的 GT 只是“初始帧”，从不作为 target；target 永远是 `t+1`。
- 命名陷阱：`mask_object`（io.py）装的是 masked RGB 不是二值 mask；`img` 注释写 `[n_cam,H,W,3]` 实为 CHW；`controller_img_mask` 实际语义是“排除区域”（obstacle+robot），D360 为空。
- 〔论文 vs D360〕Eq.12 对 pred 和 GT 都乘 M、只在物体区域算；D360：GT 乘 M，render 只画物体 Gaussian（`render_mov_only=True`），但 L2 权重全图=1（背景黑像素也参与），SSIM 不吃 `mask_weights`，robot 遮挡未处理（T16）。这套约定在 baseline 与 tactile 对比中必须完全一致。

## 3.2 相机与图像卡〔✓ U1.3〕

- 身份/顺序：Config A index0 = `brics-odroid-023_cam0`，index1 = `brics-odroid-009_cam1`（camera_config_2cam）；scene package 目录 `color/0`、`color/1`（scene_package_contract `numeric_camera_mapping`）。光轴夹角约 117°；选择理由里写明“局部夹爪遮挡仍在视野中”（camera_manifest）。
- 外参链（T8）：标定是 COLMAP 风格 w2c（`X_cam = R·X_world + t`，OpenCV 轴 x右 y下 z前），package 存 **c2w** 到 `calibrate.pkl` → `extract_extrinsics(c2w)`（embodied_dataset.py:184）：`w2c = inv(c2w)`，`Camera.R = R_w2c^T`，`Camera.T = t_w2c`（3DGS 约定）→ `getWorld2View2`（cameras.py:137）转回 w2c → `.transpose` 存为 `world_view_transform`（行向量约定）；`camera_center = inv(wvt)[3,:3]`。
- 内参（T9）：1280×720 → 640×360，`K_t = [[.5,0,-.25],[0,.5,-.25],[0,0,1]]·K_s`；cam0 fx/fy ≈ 441.8/443.0，cam1 ≈ 497.0/494.2，cx/cy = 319.5/179.5。loader `extract_intrinsics`（:200）**只用 fx/fy → FoV**；`getProjectionMatrix`（cameras.py:151）对称 → 隐含主点 ((W−1)/2,(H−1)/2)。D360 主点恰好居中才成立（T9 已验证）。znear/zfar = 0.01/100。
- `img_hw` 来自 metadata `WH` 反转（:988）= [360,640]；每个 split（Stage-2 有 15 个）各建一组 Camera，几何相同，`static_img_path = color/<cam>/<split[0]>.png`（:226）。
- 像素一致性：RGB（T14）INTER_AREA、mask（T15）NEAREST 取 `[1::2,1::2]`，与 T9 半像素约定一致；RGB 已 BGR→RGB。
- 渲染：`AccDecoder.pre_render:127` 用预测 pos + 预测 cov3D、PLY 固定 SH 与 opacity；背景 = `render_pipe_cfg.white_bg=False` → 黑（acc_decoder.py:80–81），D360 无 `white_bg` 键 → 与 GT 黑背景一致。loss 对相机取均值（T16）。
- 易混：`__getitem__:843` 的局部变量 `seq_name` 实际是相机目录号，不是序列名。

## 3.3 Gaussian 卡〔✓ U1.4〕

- 来源：`splatfacto/splat_113.ply`（source 113 = local 0）。T16.5 确认 194 帧的 `f_rest_0..44` 全部严格为 0，于是删掉这些字段，得到 `t16_5_sh0/splat_113_sh0.ply`（无损，顶点顺序保持不变）。scene package 的 `pi3/gs/point_cloud/iteration_10000/point_cloud.ply` 指向它；config 设 `sh_degree=0`。N = 12861。
- PLY 字段：`x y z | nx ny nz（loader 不读）| f_dc_0..2 | opacity | scale_0..2 | rot_0..3`。
- `load_ply`（gaussian_model.py:284–334）存的是 **raw 参数**：`_xyz [N,3]`、`_features_dc [N,1,3]`、`_features_rest [N,0,3]`、`_opacity`（logit）、`_scaling`（log）、`_rotation`（四元数 wxyz，未归一化）。
- 激活（:103–147）：`get_scaling = exp`、`get_opacity = sigmoid`、`get_rotation = normalize`。`get_covariance`：L = R(q)·diag(s)，Σ = L·Lᵀ，再用 `strip_symmetric` 打包成 6 维 `[xx, xy, xz, yy, yz, zz]`（general_utils.py:64–76）。
- 模拟器从 GaussianModel 取四样东西（`forward_train:668–678`）：
  - `get_xyz`：初始 cur/prev 状态，同时是 template。
  - `get_scaling`：算 `diag_volume = prod(scaling·512)`，形状 [N,1]（:678）。按轴先乘 512，大概是为了把毫米级的尺度归一到 O(1)〔推测〕。
  - `get_covariance()`：template Σ，形状 [N,6]，每一步都用它算 FΣFᵀ（:714 传入 `original_mov_cov.detach().clone()`）。
  - 渲染外观：`get_opacity` 和 `get_features`（`pre_render:127`、render.py:134）。
- 随时间变化的只有 **pos [N,3] 和 cov6 [N,6]**，两者都是预测值。opacity、颜色、N、行顺序全程固定。SH0 时 `eval_sh` 退化为 C0·dc + 0.5，颜色与视角无关；`use_rotation=False`。
- Identity：行号 i 就是 Gaussian i 的身份。`p2c_mapping`、cache 的 `pred_pos`/`pred_cov`、渲染全部按这个行顺序。T16.5 写明 194 个重建 PLY **之间没有跨帧对应**，所以不能用后续帧的 PLY 重置状态，也不能直接拿它做逐点监督。
- Stage-2：template pos/cov 取 cache 的 `scene_init_*[key]`，opacity 和 SH 仍然来自 frame-113 的 PLY（U4.2 再看）。

## 3.4 Controller 卡〔✓ U1.5〕

- 来源（T10/T11）：夹爪 0 的两根手指，每根取 URDF `tactile_*` 锚点组成的 3×5 网格（行 0/6/11，列 0/8/16/23/31），并沿手指局部 ±X 偏置 7 mm。索引 0–14 是左指，15–29 是右指，每根手指内部行在外、列在内。身份由“URDF link + 局部锚点”决定，与世界坐标无关。世界系，单位米，float32。
- 轨迹：`robot.npy` 的 `T_worlds[f]` 和 `openings[f]` 经 FK 得到 `controller_points.npy [194,30,3]`，打包成 `track_process_data.pkl {'controller_points'}`（package_scene.py:109–115）。dataset 用同一个 `video_range` 切片（embodied_dataset.py:917–919）。
- 限制：opening 有 136 帧被截到下限 0.04 m；7 mm 偏置下 186 帧的两指平面有符号间隙为负（中位数 −5.08 mm）。所以夹紧阶段只有刚体位姿在变，闭合程度和力度没有表达。另有 5 mm、0 mm 两个候选 offset 作为消融或参照。
- 拼接时间对齐（`forward_train:693–703` → `_preprocess:239–241`）：

  | merged 槽位 | controller 行 [0,30) | object 行 [30,12891) |
  |---|---|---|
  | prev | `traj[t]` | 状态 t−1 |
  | cur | `traj[t+1]` | 状态 t |
  | template | `traj[0]`（Stage-1 :669 = local 0；Stage-2 :734 = 窗口起点 local 10i） | 初始 PLY / cache `frame_<10i>` pos |

  controller 行比 object 行**超前一步**，这一步就是论文 Eq.4 里的 R_t（下一时刻已知动作）。`simple_test` 用同样的索引（`frame_idx+1`）。两侧 template 指向同一时刻；continuous 模式下改用 `traj[template_frame − gs_aligned_frame]`（stage2 :960），细节放到 U5.1。
- controller 节点的 attr、diag_volume、cov 直接复制自 **Gaussian 第 0 行**（`_preprocess:245–247`），只是占位。`pin_mask` 的前 30 行为 1（:686–687）。预测结果切掉前 30 行（`pred_pos[num_controller_points:]`）。
- 分组（`dis_split`，`[{'num_cluster':10},{'downsample_rate':0.5}]`，`_load_controller_cluster_mask:611–744`）：第 0 层按索引连续每 3 个一组，得到 10 组（:690–709）；下一层在 `n//2` 处切开，得到 2 根手指（:714–727）；再追加 `zeros` 归到 1 个 root（:736）。根节点数在代码里写死为 2（:621、断言 :738）。T12 记录的 p2c：`[[0,0,0,1,1,1,…,9,9,9],[0,0,0,0,0,1,1,1,1,1],[0,0]]`。
- Gaussian 侧（`_load_cluster_mask:467–609`）：在初始 PLY 的 xyz 上做 complete-linkage 聚类，距离阈值 0.02 m 和 0.2 m，结果是 12861→640→7→1。
- 合并（`_merge_cluster_mask:454–465`）：每一层把 object 的 label 加上 controller 在该层的簇数，因此两侧在根以下从不共簇；最后追加 `[zeros(1), zeros(1)]`，把两个 root 合成全局 root。sample 里的 `p2c_mapping` 是按层的 (controller, object) 两段；`_preprocess:255` 把两段拼成合并后的 mapping。
- 缓存：`<scene>/cluster_mask/dis_split/...pkl` 和 `<scene>/controller_mask/dis_split/controller_...pkl` 都只按 scheme 命名，文件存在就复用（:495–499、:645–650），换数据不会自动失效。controller 侧的 label 数组只取决于点数 n，不读坐标（中心 `new_pcs` 不保存），所以重新生成会得到同一个数组。点序决定的是“哪几个物理点成为一组”。Gaussian 侧的 label 取决于 xyz。

## 4. 代码入口〔✓〕

| 环节 | 位置 |
|---|---|
| 训练入口 | `tools/train.py::main` → `mmgs/apis/train.py::train_model`（runner、OptimizerHook、EvalHook、load/resume :141–153） |
| 一个 iteration | `core/runner/epoch_runner.py::run_iter` → `models/simulators/base.py::train_step:90`（forward→`_parse_losses`） |
| Stage-1 rollout | `gs_simulator_embodied.py::forward_train:625–786`；状态更新 :727–734 |
| 建图 | `_preprocess:231`（controller 拼在前面）→ `dgl_graph.py::GsHieEmbodiedDGLProcessor` |
| 网络+解码+渲染 | `encode_decode:276–444`（粗→细层循环、cov :394、render :435） |
| rollout 长度 | `_rollout_steps:584`：`3 + 3*epoch`，被可用帧数截断 |
| Stage-1 cache | `save_gaussian:446`（命名 `frame_idx*model.frame_gap`） |
| Stage-2 | `gs_simulator_embodied_stage2.py`：cache 读入 `_init_scene_init_pos_cov:218`；`forward_train:683`；continuous `simple_test:898`、`update_gaussian:502` |
| 推理循环 | `tools/test.py` → `apis/test.py::single_gpu_rollout:163`（:235–236 把预测喂回） |

## 4.1 循环层级与调用链〔✓ U0.2〕

```text
训练  tools/train.py::main → apis/train.py::train_model → runner.run
  L1 epoch 循环（MMCV EpochBasedRunner）  Stage-1 46 / Stage-2 61
   L2 iteration 循环 = 一个 batch（EpochRunner.run_iter → BaseSimulator.train_step）
        Stage-1 50 it/epoch（1 sample×Repeat50）；Stage-2 600 it/epoch（15 窗口×Repeat40）
     L3 rollout 循环（在 forward_train 内部）  长度 = min(可用 transition, 3+3·epoch)
        每步: _preprocess → encode_decode → 渲染 loss；状态 detach 后前进
     ← 聚合各步 loss（mean）→ _parse_losses 求和成一个标量
   ← OptimizerHook: backward → clip(max_norm=1) → Adam.step   〔MMCV，本机未读源码〕
  epoch 末: LR hook / checkpoint hook / EvalHook(single_gpu_test，train-only 诊断)

推理  tools/test.py → apis/test.py::single_gpu_test → single_gpu_rollout
  外层: 遍历 dataset sample；内层 frame_idx 循环，每帧调一次 model(return_loss=False)
       → simple_test（rollout_size=1，no_grad）；prev/cur 状态由调用者 :235–236 传回
```

- 关键差异：训练时多步状态活在 `forward_train` 内部的局部变量；推理时活在 `single_gpu_rollout` 的循环变量 + Stage-2 的 `scene_init_*` 字典。未来 tactile history 也要选“谁拥有它”。
- 同一条推理循环被用了三次：epoch 末 EvalHook、T27 cache 生成（Stage-1 模型 + `flag_save_gaussian=True`）、T31 continuous rollout（Stage-2 模型 + `test_rollout_mode='continuous'`）。
- rollout 长度用的 `num_epoch` 是 runner 的 0-based epoch：Stage-1 epoch0→3 步，epoch4 起 15 步（被 16 帧截断）。Stage-2 窗口 13 帧→最多 12 步，末窗 `[140,150)`→9 步。

## 4.2 层级图卡〔✓ U2.1 静态〕

入口：`_preprocess:257` → `dgl_graph.py::GsHieEmbodiedDGLProcessor.batch_preprocess:603–672`。每个 rollout step 都重建一次（simulator :699 在循环内）。

| 图 | 节点数 | 节点顺序 | 层内边 | 网络是否处理 |
|---|---|---|---|---|
| G0 点层 | 12891 | 0–29 controller，30–12890 Gaussian | 无（`dynamic_base=forward_last_layer=False`，:425–431） | 否：位置由上层 F 广播得到（U2.4） |
| G1 | 650 | 0–9 controller 簇，10–649 布簇 | radius 0.2 | 是（第二个处理） |
| G2 | 9 | 0–1 手指，2–8 布簇 | radius 0.5 | 是（最先处理） |
| G3 | 2 | 0 controller root，1 布 root | radius 5.0（`radius[-1]*10`，:408），建了但没人用 | 否：只给 G2 提供 anchor |

- connect 图（`_preprocess_hierarchy:455–527`）：G_l 与 G_{l+1} 合在一张图里，前面是点、后面是簇（簇 id + n_points）；p2c 边和 c2p 边都是每个子节点对应 1 条。G_{l+1} 的节点 j 就是 merged label j，所以每一层都是 controller 在前。
- 簇节点属性（`node_attr_scheme:626–637`）：
  - `diag_volume` 求和。
  - `pin_mask` 取 max（“or”），controller 簇为 1，布簇为 0；因为直到 root 之前两侧都不共簇，所以不会出现混合。
  - `attr` 取均值，density 按体积换算（:488–497）。
  - `external` 取均值。
  - prev/cur/template/orig_template 都按 `diag_volume` 加权平均。controller 的体积是复制 Gaussian 0 的，所以 controller 簇就是锚点的算术均值。
- 边（`_dynamic_edges:561–592`）：对每个 receiver 取 16 个最近 sender（kNN；`max_radius=None`〔MMCV，未读源码〕），只保留距离 < radius 的，去重、去自环。
  - **static** 边在 `template_state` 上建，**dynamic** 边在 `cur_state` 上建并去掉已在 static 中的，两者合并（:520–523）。
  - DGL 方向：src = 邻居（sender），dst = receiver（:967）。kNN 不对称，所以边也不一定双向。
  - 所有边共用一套规则，没有 controller–布 专用的边类型。
- anchor（`_postprocess_hie:594–601`）：把父簇的 prev/cur/template/orig_template 经 c2p 广播给子节点，存为 `anchor_*`。G0 的 anchor 来自 G1，G1 的来自 G2，G2 的来自 G3。
- 处理顺序（`encode_decode:286–291`）：i=1 时处理 G2，i=2 时处理 G1，到 lv_idx=0 就 break。由粗到细，细节放到 U2.3/U2.4。

## 4.3 Backbone 输入特征卡〔✓ U2.2 静态〕

入口：`encode_decode:307 extract_feat` → `MeshGraphNetHieEmbodied.forward:425` → `init_features:403`（node :362，edge `init_edge_features_fixbug:314`）→ encoder（U2.3）。`share_weight=True`（base config :83 / stage2 :107）：G2 和 G1 共用同一个 backbone、decoder 和 Normalizer。D360：`fix_bug=True`、`edge_mode='ratio'`、`edge_theta=True`、`attr_mode` 默认 `'cat'`、`num_fcs=3`、`pre_norm=True`、dt=1/15。

节点特征，11 维（:366–400）。记号 A = 该节点的父簇 anchor（经 c2p 广播得到）：

| 分量 | 公式 | 维 | 归一化 |
|---|---|---|---|
| 父加速度 + 重力 | `−(A_next − 2A_cur + A_prev)/dt² + g` | 3 | anchor_normalizer |
| 相对父的速度 | `((x_cur − A_next) − (x_prev − A_prev))/dt` | 3 | node_normalizer |
| attr | `scene_attr` [1,5] 可学习，所有节点相同 | 5 | Identity |

- 没有绝对位置，也没有绝对速度，全部相对父 anchor 表达。`−a_anchor + g` 可以理解为父参考系中的“惯性力 + 重力”〔推断〕。
- `A_next` 的来源：G2 取 `= A_cur`（:302），G1 取 G2 的预测 `pred_pos`（:320–324）。粗层的预测就是通过这一项流入细层特征的。
- `pin_mask` 不在特征里（:387 TODO），节点类型对 encoder 不可见。

边特征，9 维（:314–360）。r 是 receiver（dst），s 是 sender（src）：

| 分量 | 公式 | 维 | 归一化 |
|---|---|---|---|
| 方向 | `Δcur/‖Δcur‖`，Δ = x_r − x_s | 3 | edge_normalizer |
| 拉伸比 | `‖Δcur‖/‖Δorig_template‖` | 1 | 同上 |
| 相对速度 | `(Δcur − Δprev)/dt` | 3 | 同上 |
| 角度变化 | 从 r 的 anchor 看 r 与 s 的夹角，当前减 template，取 cos/sin（:345–355） | 2 | 不归一化 |

- 没有绝对边长，只有比值，所以与尺度无关。比值的分母没有 clamp，两个 template 点重合时会出问题（T10.5 已记录）。夹角用叉积的模算 sin，因此不带符号。

Normalizer（normalization.py）：
- 按通道在线累积 sum 和 sum²（FP64），均值和标准差由此算出，std 下限为 1e-8（:61–131）。
- 只在 `train()` 模式下累积（:126、:133），每次调用先累积再归一化。上限是 1e6 次调用。
- 是 `nn.Parameter(requires_grad=False)`：进 state_dict、随 checkpoint 保存，但不参与优化。Stage-2 加载后会继续累积（T30.1）。

## 4.4 Message passing 卡〔✓ U2.3 静态〕

位置：`meshgraphnet_embodied.py`，`MeshGraphNetEncoderLayer:21–121`、`MeshGraphNetEncoder:124–169`、`FFN`（utils/feedforward_networks.py:9–86）。

- 初始嵌入：h⁰ = node_encoder(11→128→128→128，SiLU，final_act)；e⁰ = edge_encoder(9→128→128→128)。`pre_norm=True`，所以嵌入后不做 LN（:233、:251 为 Identity）。
- 一层（:98–121），所有边和节点并行：
  1. 边更新：`e' = e + MLP_e(LN([e, h_dst, h_src]))`（:69–81；FFN 384→128→128→128，残差 = e）。dst = receiver，src = sender。
  2. 聚合：`agg_i = Σ_{j→i} e'_ji`（`send_and_recv(copy_e, sum)`，:114）。只汇总入边；图里没有边时 agg = 0（:115–117）。
  3. 节点更新：`h' = h + MLP_v(LN([h, agg]))`（:84–96；FFN 256→128→128→128，残差 = h）。
  - 边的 latent 也逐层演化（MeshGraphNets 风格），不是只用初始边特征。
- 16 层各有独立权重（ModuleList :149–151）；G2 和 G1 共用这 16 层（share_weight）。每个 rollout step：G2 跑 16 层 → decoder → G1 用 G2 的预测重新嵌入，再跑 16 层 → decoder。
- 没有隐状态：latent 不跨层级、不跨步传递，只传几何量。模型是 Markov 的。
- 输出：`g.ndata['out_node']`（16 层残差流，未做最终 LN）→ `AccDecoder.node_pred_pos_dg`（acc_decoder.py:290；`dynamic_proj` 128→64→32→11）→ U2.4。
- pin 更正：`pre_predict` 没有转发 `apply_pin`（acc_decoder.py:123–124），所以 decoder 的 pin 覆盖（:324–325）不会执行。

## 4.5 Decoder 与下传卡〔✓ U2.4 静态〕

- 11 参数 → F（acc_decoder.py:290–320；deformation_gradient.py:104–109）：
  - U、V：输出 + `[1,0,0,0]` 后归一化（单位四元数）。
  - s：`exp(clamp(raw, ±5))` 再除以几何均值（`norm_volumn=True` 默认，:84–92），所以 Πs = 1。
  - F = U·diag(s)·Vᵀ·ratio(1.0)。det F = +1，输出≈0 时 F≈I。
- 位置：`next = A_next + F·(X − A_tem)`（:320）；pin 覆盖不执行（U2.3）。
- 下传链（simulator `encode_decode:302–380`；X = template，R = 所属 root）：

```text
G2:   P_j  = R_cur + F2_j (X2_j − R_tem)              # 布 root 在 t；手 root = mean traj[t+1]
i=1 下传: G1 cur/template ← c1_k = R_cur + F2_j (X1_k − R_tem)   (:364–370)
          G1 anchor_next/anchor_tem ← P_j                         (:320–324)
          G0 cur/template ← c0_i = R_cur + F2_j (X0_i − R_tem)
G1:   Q_k  = P_j + F1_k F2_j (X1_k − X2_j)
i=2 下传: G0 cur = P_j + F1_k F2_j (X0_i − X2_j)   = pred_pos (:404)
cov:  Σ_i  = (F1_k F2_j) Σ_tem,i (F1_k F2_j)ᵀ      # outF_lv_list=[F1,F2]，由后往前作用（transformation_utils.py:33–38）
```

- cur_cov 每步都是 template 的 Σ（:681，:730 注释掉了更新），不累乘。
- G1 的 controller 簇也被覆盖为预测值；真实 traj[t+1] 只在 G3 手 root 和 G2 手指上。
- 手、布两棵子树只在 G3 之上相连，anchor 链从不跨子树；手→布 的信息只能走消息边。

## 5. 易混点清单

- 一次 iteration 内有多步 rollout，loss 先聚合（`avg_loss=True` 取均值），**只做一次** backward/step。
- 每步 `prev/cur` 被 detach：状态向前传，梯度不跨步传（每步 loss 只回传到本步网络计算）。
- `template_state` = 形变参考（Stage-1 恒为初始 Gaussian），F 是“相对 template 的形变”，因此 covariance 每步用 template 的 Σ 计算 FΣFᵀ，而不是累乘上一步 Σ。〔✓ 代码；与论文 Eq.3 的 Σ_k 对应〕
- Stage-1 config 只设 dataset `frame_gap=10`；model.frame_gap 默认 1，只影响 cache 文件命名（T27 用 helper 显式设 10）。Stage-2 config model.frame_gap=10 决定 template 块长度，dataset frame_gap=1 决定 dense 采样。
- 训练时的 val/test 路由都是 train-only 诊断，不是 holdout。
