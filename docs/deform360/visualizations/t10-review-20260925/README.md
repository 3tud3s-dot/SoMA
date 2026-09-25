# T10 controller 人工可视化 review

日期：2026-09-25。状态：**待人工确认**。这是 T10 representation 的短窗口观察材料，不是 T11/T12 执行结果，也不是正式 trajectory contract。未修改 T10 定义或既有结论，未 commit/push。

## 如何查看

先播放 [RGB + tactile 同步 review](rgb_tactile_review.mp4)，再打开两个独立 camera 视频查看放大区域及索引：

| 文件 | 内容 |
|---|---|
| [023_cam0_anchor_overlay.mp4](023_cam0_anchor_overlay.mp4) | 023_cam0 原图全景、固定区域放大、30 点及索引 |
| [009_cam1_anchor_overlay.mp4](009_cam1_anchor_overlay.mp4) | 009_cam1 同样形式，可辅助观察夹爪另一面 |
| [rgb_tactile_review.mp4](rgb_tactile_review.mp4) | 两路 overlay、四路 tactile 热图、active-count 历史及当前帧游标 |
| [023_cam0_snapshots.jpg](023_cam0_snapshots.jpg) | source 113、120、130、140 的 2×2 快照 |
| [009_cam1_snapshots.jpg](009_cam1_snapshots.jpg) | 同四帧的第二视角快照 |
| [tactile_snapshots.jpg](tactile_snapshots.jpg) | 同四帧、每帧四路 tactile |
| [controller_3d_view.png](controller_3d_view.png) | world-space anchors、root 路径，以及 opening / signed gap 曲线 |
| [review_metadata.json](review_metadata.json) | 逐帧 opening/tactile 统计、输入与输出 hash、投影 crop 和时间对应记录 |

## 范围和图例

- source frames **113–140（含两端，28 帧）**，对应 local 0–27。三个视频均为 28 帧，按 **6 fps** 慢放，时长约 4.67 秒；原始 RGB 为 30 fps。此窗口用于观察持续抓持，不展示首次接触发生过程。
- 橙色圆点：`finger_left`，indices **0–14**；青色菱形：`finger_right`，indices **15–29**。两组是同一夹爪的两根手指，不是两只手臂，也不是图像左右。
- 固定 T10 行 `[0,6,11]`、列 `[0,8,16,23,31]`；各帧只代入官方 opening 映射和当帧 `T_worlds`，不重新选点。frame 113 世界坐标与 T10 的 float32 数值逐 bit 相同。
- 索引引线的版面按首帧投影高度排布，便于阅读；这只影响标签位置，不改变 controller 数组点序。放大 crop 在整个窗口内固定，未对夹爪位置做额外拟合。
- 绿色十字为 **URDF gripper root**，不是经过额外校准的 TCP。格线表示采样面网格。被夹爪或布遮挡的点仍然画出；overlay 不执行遮挡剔除，不能将所有可见标记解释为物理可见表面。
- RGB 使用 T8 w2c 和 T9 记录的 source K 投影，再按显示尺寸缩放；没有改写原始 RGB、mask、calibration 或 controller contract。没有输出正式 controller trajectory、cache 或 graph。

## tactile 的显示与同步边界

- 四路按原始名称排列：`brics-odroid_tactilel_left`、`brics-odroid_tactilel_right`、`brics-odroid_tactiler_left`、`brics-odroid_tactiler_right`。
- 显示完整 `16×32` 热图，不转置、不翻转；色标全窗口、全 sensor 固定 **[0,1]**，没有逐帧归一化。白线以下为最后 4 行。
- active count 为 `values > 0`、仅统计官方几何/contact rule 使用的 rows 0–11，共 384 个位置；完整 16 行的额外计数也保存在 metadata。该数值不是接触力或已校准压力。
- 同一个 source frame 同时取 RGB、robot 和 `synced_tactile.npy`。两路 RGB 的 28 个 `aligned_timestamps.txt` token 逐项相同。tactile 依赖官方发布的同步 frame index，本轮未独立验证 sensor-clock 延迟。
- `tactilel/tactiler` 仅标识文件命名分组；当前 robot 是 `bimanual=False`，本轮不推断两组传感器与唯一 robot gripper 的物理对应。需要人工确认后才能解释“图像上的哪根手指对应哪路信号”。

## 本轮观察：不是人工验收结论

1. 两路四帧快照中，点大体位于夹爪/指尖附近并随夹爪移动，未见明显整体翻转或大范围漂离。root 首尾位移约 **27.89 mm**，最大相邻 root 位移 **1.39 mm**，最大 anchor 位移 **1.86 mm**；这些数值不证明亚毫米级附着精度。
2. opening 范围 **43.23–46.36 mm**，是近闭合状态的小幅变化，不是大幅开合演示。点身份固定，左右点在图像中高度重叠；颜色覆盖和遮挡会影响肉眼判断。
3. **明确需要复核的几何现象：** `right_x-left_x` 采样面 signed gap 在 **−2.113–+0.750 mm** 间变化，28 帧中 **25 帧为负**。这是原始 T10/URDF/opening 映射的结果，未修正。不能将 T10 数值 PASS 当作无穿透验证；也不能仅凭该 signed gap 宣称实体网格必然穿透。
4. 四路 active count 范围依次为 **17–18、11–13、0–1、3–8**。热图以持续的稀疏响应为主，`tactiler_right` 在窗口后段有增强；未据此宣称 contact onset、压力大小或 causality 已验证。
5. 图像显示持续抓持布角，与存在持续 tactile 响应并不矛盾，但未完成 sensor-to-finger 物理对应和独立时间延迟验证，不能据此最终确认接触区域一一匹配。

## 人工确认清单

- [ ] anchors 是否附着于预期手指表面，而非随错误 offset 漂移。
- [ ] URDF 左右分组与实际设备、传感器物理分组是否相符。
- [ ] opening 变化时采样面开合是否合理；负 signed gap 是否可接受或需独立检查。
- [ ] index / point ordering 是否稳定，是否出现跳变、翻转或可见穿模。
- [ ] tactile 响应变化是否符合持续抓持直觉。
- [ ] RGB 接触区域与对应 tactile 信号是否大致一致。

建议先完成上述人工 review，再决定是否提交 T10 checkpoint。若发现 representation 问题，先单独定位，不自动执行 T11/T12，也不在本 review 中改动表示。

## 版本管理

所属仓库：SoMA；分支：`deform360-adaptation`。本目录是本轮明确授权生成和复制到 Mac 的少量 review 媒体例外，不授权把 dataset 视频纳入 Git。当前未暂存、commit 或 push；视频是否纳入后续 checkpoint 应在届时单独确定。原始数据保留服务器，T10/T8/T9 contract hash 保持不变。
