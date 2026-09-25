# T18 初始静态渲染 review

结论：PASS，仅表示 source 113 / local 0 的初始几何对齐可进入后续 smoke test，不表示 dynamics、forward prediction 或训练正确。未执行 T19。

- Config A：023_cam0、009_cam1；Config B：023_cam0、009_cam1、014_cam1。完整 camera ID 和顺序在 projection_diagnostics.json 中。
- A/B 共用的两相机 K、c2w、PLY/RGB/mask SHA256 逐项一致，所以渲染三个唯一相机，不重复渲染相同输入。
- comparison_grid.jpg 为总览；每个 *_comparison.jpg 为原 RGB / 黑背景 Gaussian render / object mask / 叠图。
- 叠图绿色=object mask 边界，紫色=白色 override render 的 alpha≥0.5 边界；*_projection.jpg 的青色点为独立 pinhole 投影抽样（每 40 个 Gaussian center），未修改数据或坐标。
- 使用 SoMA 原始 GaussianModel(0)、Camera 和 render_gaussian_physdreamer；torch.no_grad()，黑背景，原函数 antialiasing=True。未构建 dynamics 模型、未调用 backward。Slurm job 25822，RTX 5090。

| camera | 最大独立投影误差 px | alpha≥0.5 / mask IoU | 质心差 dx,dy px |
|---|---:|---:|---:|
| 023_cam0 | 0.0000926 | 0.9767 | +0.13, −0.46 |
| 009_cam1 | 0.0000779 | 0.8286 | −19.70, −4.10 |
| 014_cam1 | 0.0001118 | 0.8920 | −2.01, −0.13 |

独立 float64 K·w2c 投影对照实际 SoMA float32 full_proj_transform 与 rasterizer 像素映射，最大误差远小于 T9 的 1 px 容差；三个视角各 12861 个 Gaussian center 均在正深度且落入图像范围。原位 IoU 均优于水平/垂直镜像诊断；这些翻转只用于数值对照，未修改任何相机或资产。

图像检查：023 轮廓紧密对齐；009 的主要差异位于夹爪遮挡侧，render 布面延伸到 mask 排除的区域，质心差不应直接当作相机平移；014 存在局部遮挡缺口差异。可见主体未见镜像、轴翻转或明显整体偏移。遮挡/重建边缘语义是可能解释，未独立证明其全部来源，也没有解决 T16 缺失 robot mask 的 limitation。IoU 阈值仅作诊断，不是新训练目标或性能评价。

完整参数、输入/源码/图像 hash 和 metrics 见 projection_diagnostics.json。视觉结论来自助手查看实际输出，不冒充用户人工确认。
