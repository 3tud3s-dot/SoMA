# T13 support-plane feasibility check

结论：FAIL，当前未获得可可靠拟合的 support-plane 3D 点；不是已拟合 normal 后发现不稳定。

## 实际输入检查

服务器目标：`datasets/deform360/processed/008-pink-cloth/episode_0`。source frames 113、120、200、267，全部在训练窗口内。检查 metadata 列出的 36 cameras，共 144 组 depth/mask 切片。`rendered_depth.h5:data` 为 uint16 `[357,720,1280]`，`mask_refined.h5:data` 为 uint8 同 shape（已核对三个选定相机）。

所有 144 组 `count((depth > 0) & (mask == 0)) = 0`。这说明采样帧中的正深度完全位于物体 mask 内；不是“透明板深度较稀疏”，而是没有独立的板面背景深度。选定相机的 RGB / depth / 非零深度覆盖图确认正深度覆盖布料。不能给 RGB 板面像素借用布料、透过板看到的背景或反射物的深度。

- [全部相机统计与切片 SHA256](all_camera_depth_audit.json)
- [三个候选相机详细统计](depth_audit.json)
- [source 113 RGB / depth 对照](support_depth_evidence.jpg)
- [现有几何 3D 诊断图](available_geometry_3d.png)

## 为何没有给出 normal / gravity

透明板在 RGB 中可见，但本轮没有建立可靠的板面同名点集合。透明板透射/反射区域内的背景特征不等于板面特征；可见板边与固定支架可作为后续人工确认的对应点候选，但未做人工标注或 RGB-only stereo 拟合，不能声称该替代路线已被排除。

直接对现有 Gaussian 或正深度拟合只能得到物体/布料几何，不是独立 support plane。即使多视角结果一致，也可能只是同一 Gaussian reconstruction 的重复渲染，并非独立物理验证。没有用 cloth 悬垂、camera 朝向或 robot pose 反推一个 normal；没有主平面时，这些辅助信息不能决定主估计。

plane normal、raw gravity、external gravity 和跨视角/帧 normal 角度误差均为 null / N/A，而不是 0。3D 图仅展示已有 Gaussian、T10 controller 和 T8 Config B cameras；未画虚构的 support plane 或 gravity arrow，图中竖直显示轴不代表 physical up。

## 最小所缺证据

1. 板面至少三个非共线、实际更宜六个以上分散点的可靠多视角同名点；可以来自人工确认的板边/角点或板上固定标记。需要同一物理平面、正深度、足够三角化角度与小重投影残差，并在留出视角/帧复核；静止板重复帧不提供新的空间基线。
2. 或包含板面的真实/完整场景深度；当前 object-only rendered_depth 不满足。
3. 得到平面后，仍须显式采用“承物板为物理水平面”的假设或测量依据；利用 object/controller 在其上侧确定 normal 符号。之后才使用 raw=-9.8*up，external=4*raw，rot_est omitted/identity。

本轮无源码/数据改动，无 CUDA/训练，无 T14，无 commit/push。仅产生小型 review artifacts 与 T13 文档更新。
