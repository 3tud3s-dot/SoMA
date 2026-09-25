# T10 controller offset comparison

日期：2026-09-25。**待人工选择 canonical offset；当前默认仍为 candidate_7mm。** 本目录仅为 review artifact，不执行 T11/T12，不生成正式 trajectory、cache 或 graph，不修改已有 T10 contract、源码、loader 或训练代码。

## 查看入口

- [总 comparison 视频](offset_comparison.mp4)：列为 0 / 5 / 7 mm；前两行是 023_cam0 / 009_cam1 RGB overlay，第三行为同尺度的 root-space 三维几何。
- [可拖拽三维对比图](interactive_geometry_comparison.html)：自包含 HTML，离线可打开。拖动一个面板可同步旋转三组，滑块切换 source frame，悬停查看 anchor index；使用宽窗口或水平滚动。3D 使用等比例毫米坐标，仅缩小显示取景，不改变数据。
- 四张代表帧：[113](comparison_source_113.jpg)、[120](comparison_source_120.jpg)、[130](comparison_source_130.jpg)、[140](comparison_source_140.jpg)。每张保持同一列顺序、相机 crop、缩放与标签布局。
- [Gap 曲线](gap_comparison.png)、[统计 JSON](gap_statistics.json)、[逐帧 CSV](gap_per_frame.csv)、[来源与验证记录](review_metadata.json)。
- [相机全景与固定 crop](camera_context_source_113.jpg)：辅助理解放大区域在原图中的位置。

| Candidate | 023_cam0 overlay | 009_cam1 overlay |
|---|---|---|
| 0 mm | [视频](candidate_0mm_023_cam0_overlay.mp4) | [视频](candidate_0mm_009_cam1_overlay.mp4) |
| 5 mm | [视频](candidate_5mm_023_cam0_overlay.mp4) | [视频](candidate_5mm_009_cam1_overlay.mp4) |
| 7 mm | [视频](candidate_7mm_023_cam0_overlay.mp4) | [视频](candidate_7mm_009_cam1_overlay.mp4) |

另提供 `geometry_source_113/120/130/140.png` 四张独立三维对照图。

## 比较条件

- 使用 [controller_geometry_config.json](../../contracts/008-pink-cloth/episode_0/controller_geometry_config.json)，所有 candidate 均保留 30 点、left 0–14 / right 15–29、相同 Y/Z 网格、robot pose/opening 与 opening mapping。只改变两指的 local-X offset，left 为 +d、right 为 −d。
- 同一 source frame 113–140（共 28 帧）同时读取两路 RGB 与 robot。代表帧为 113、120、130、140。视频为 6 fps 慢放，约 4.67 秒，原视频为 30 fps；没有插帧或导出完整序列。
- 橙色圆点为 left，青色菱形为 right。30 个 index 标签使用固定引线位置；被遮挡的锚点仍然绘制，因此重叠标记不等于实际可见接触点。
- RGB 使用 T8 w2c 与 T9 source K 进行世界坐标投影，再做仅用于展示的固定 crop/resize。复用旧 review 的 crop：023_cam0 `[308,85,181,181]`，009_cam1 `[278,349,342,342]`；每个 crop 显示为 368×368，采用 half-pixel 映射。三个 candidate 全窗口的 30 点均在 crop 内。
- 三维图使用 **gripper root 坐标**，消除刚体 pose 运动，专门观察开合与 offset；RGB 保留真实世界位姿变化。星标（交互图为十字）是 finger-link origin，虚线/引导线连向网格，短粗线表示从 local-X=0 网格中心到采样面中心的 offset。0 mm 短线长度为零，不把 15 个点收缩为同一个 origin。

## Gap 的定义与统计

`g = right sampling-plane root-X − left sampling-plane root-X`，单位为 mm。同一侧 15 点的 root-X 相同。正数表示两面按预期顺序分开；负数表示两面沿该轴的位置顺序反转。它不是点对欧氏距离，也不是实体 mesh 穿透深度。左右身份始终由 finger link 决定，不按 gap 符号交换标签。

| Candidate | min (mm) | median (mm) | max (mm) | 负 gap 帧数 |
|---|---:|---:|---:|---:|
| candidate_0mm | 11.887244 | 13.191359 | 14.750391 | 0/28 |
| candidate_5mm | 1.887244 | 3.191359 | 4.750391 | 0/28 |
| candidate_7mm | −2.112756 | −0.808641 | 0.750391 | 25/28 |

三组最小值都在 frame 136，最大值都在 126。7 mm 只有 117、126、127 为正，没有恰好为零的帧。0 mm 曲线等于 7 mm 曲线加 14 mm，5 mm 曲线等于 7 mm 曲线加 4 mm；平行曲线来自左右各改变同一个 offset，没有改 opening 或 pose。

## 视觉观察与建议

已逐张检查四个代表帧、两路相机的 comparison；交互图验证了旋转和滑块切换，所有 7 个视频经解码计数均为 28 帧、6 fps。

- **0 mm：** 两组锚点分离最明显，表示 finger-link 中心平面而非已有 URDF 接触锚点位置。在两路 RGB 中，相对 5/7 mm 的网格更向两侧分开；作为几何语义对照有用。不能因为 gap 全正就认定它更接近真实 contact surface。
- **5 mm：** 相对 0 mm 更靠近内侧抓持区域，仍有约 1.9–4.8 mm 的正间隙，两侧点通常比 7 mm 更容易区分。初步视觉上是值得优先人工复核的折中候选，尚未测量实体表面拟合误差。
- **7 mm：** 保留 T10 原有 contact-surface hypothesis，更接近闭合的内侧位置；多帧左右标记明显重叠，与小/负 signed gap 一致。不能把重叠或负 gap 直接解释成 mesh 碰撞、错误点序或视觉漂移。
- **接触面判断的边界：** 从代表图看，5/7 mm 均比 0 mm 更集中在内侧抓持区域，但 5 与 7 的投影差仅约 023_cam0 的 2.10–2.26 原图像素、009_cam1 的 3.84–4.34 原图像素。遮挡、RGB 清晰度与现有标定限制了毫米级判断，无法可靠宣布哪一个更贴合真实 tactile 接触面。本轮没有读取 tactile 数值，也没有确立 sensor-to-finger 的物理对应。
- **漂移、翻转、身份：** 代表图未见明显整体漂离或翻转。全 28 帧计算中，每指局部点保持固定，恢复误差最大 `1.39e-17 m`；world↔root 误差最大 `2.22e-16 m`。三组最大相邻 anchor 位移均约 1.86 mm，没有数值跳变、动态重排或左右标签交换。这些检查不证明亚毫米级物理附着精度。

**建议（非决定）：** 人工重点比较 5 mm 与 7 mm 的内侧接触边界，保留 0 mm 作为中心平面对照。5 mm 可作为优先检查的折中候选，但不单凭“无负 gap”替换默认值。最终 canonical offset 由用户确认；本轮 7 mm 默认与 T10 PASS 保持不变。

## 来源与版本管理

原始数据只读于服务器 `datasets/deform360/processed/008-pink-cloth/episode_0/`。服务器 review 产物位于 `/data1/userdata/tcweng/projects/tcgs/outputs/deform360/t10-offset-comparison-20260925/`；本目录为本轮明确授权复制到 Mac 的少量 review 媒体，不包含原始数据视频。

输入 contract/hash、解码帧 hash、逐 candidate 数值检查、视频 frame count 与输出 hash 均记录于 `review_metadata.json`。7 mm 在 frame 113 与 T10 的 float32 坐标逐 bit 相同，全部 28 帧 gap 与先前 review 一致。robot/calibration 的读取前后 hash 不变；原视频的 size/mtime 不变。仅使用服务器 CPU NumPy/OpenCV/Matplotlib/Plotly/ffmpeg，不初始化 CUDA。

本目录属于 SoMA repository / `deform360-adaptation`，当前为 untracked review artifact；roadmap 为 tracked 文件，仅补充链接。未 add/commit/push，未同步服务器 Git checkout。T10 contract 与 geometry config 内容均不变；T10 PASS、T11/T12 TODO 均保持不变。
