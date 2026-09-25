# T7 camera review — 待人类确认

候选组合：brics-odroid-023_cam0、brics-odroid-009_cam1。T7 为 IN_PROGRESS；未最终锁定。

- interactive-layout.html：可拖拽旋转、缩放、悬停/点选相机；含 XY/XZ 投影视图。
- soma_layout.png、deform360_layout.png：静态 3D/XY/XZ 布局。
- soma_representative.jpg：3 cameras × frames 0/60/140。
- deform360_candidates.jpg：2 candidates × source 113/200/267，均为训练帧。
- layout.json：相机位置、光轴、ID、SoMA serial 对应与绘图约定。

两数据集使用独立原生世界坐标；XY/XZ 不是已验证的重力对齐方向。星标为光轴最小二乘汇聚点，不是实测物体中心。SoMA calibrate.pkl 按 loader 的 c2w 读取；Deform360 metric qvec 按 wxyz w2c 解释，位置=-R.T@t、方向=R.T[:,2]。未执行 T8 外参转换或生成训练用标定。

SoMA 输入为 datasets/soma_sample/soma_data_sample/cloth_lift/left_lift_1 的 calibrate.pkl、metadata.json、color/<camera>/<frame>.jpg；Deform360 输入为 datasets/deform360/processed/008-pink-cloth/episode_0 的 metadata.json、metric_params_refined_undistorted.txt 和候选 camera 的 undistorted.mp4。路径均相对 tcgs root。

这些是少量 review 资产，不是 RGB/mask 批量导出。当前未 commit/push；需后续纳入 SoMA Git 同步。
