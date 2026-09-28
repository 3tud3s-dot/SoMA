# T31.1：Continuous rollout artifact export — PASS

本项只增加展示/复查产物。T31原exploratory SUCCESS及其limitations不变；不是最终baseline。T30当前24h续跑保持原样，未新启动训练，未执行T32。

## 执行与计数

- Slurm25886，COMPLETED/0:0，91秒，独立GPU allocation。
- epoch22/iter13200 checkpoint；SHA256 `31223b8ca972db83a78412040ddee9dcf7e5836f63fb11cae54b5f91aec9380c`。
- 原T31进程已退出且未保存图像，无法事后找回张量。本次实际执行了一次同协议推理并加保存钩子；不能表述为“未执行新推理”或声称与上次bitwise相同。
- 193 transitions产生193预测状态（local1–193/source114–306）；额外保存initial local0/source113。故194NPZ、每相机194PNG、两相机388PNG。不是194次预测。
- frame_000图像来自原simple_test首步已有的encode_decode_render_only；frame_001…193来自原encode_decode的pred_img_list。没有新增renderer/forward调用。

## 保存位置与验证

服务器目录：`/data1/userdata/tcweng/projects/tcgs/outputs/deform360/t31-exploratory-epoch22-artifacts-20260928/`。

```
manifest.json + manifest.sha256
provenance.csv                  # 194行，含initial
metrics.csv                     # 193行，无initial transition metric
renders/{023_cam0,009_cam1}/frame_000.png … frame_193.png
states/frame_000.npz … frame_193.npz
rgb_gt_prediction_error.mp4 + .json + .sha256
report.json / validation.json / README.md
```

全部PNG解码RGB/uint8/[360,640,3]；全部NPZ只有pred_pos[12861,3]、pred_cov[12861,6]、float32/finite。保存后逐值读回并验证hash。RGB与mask未复制，manifest逐帧引用T14/T15 canonical路径和原hash。manifest本身hash放在外部manifest.sha256，避免自引用。视频有单独hash sidecar。

PNG只作显示编码：原CHW float RGB→HWC，clip[0,1]→round(255*x)→uint8；manifest保留每帧原张量hash及超范围值数量。不会将量化数据送回模型。CSV指标来自本次原有metrics/diagnostics，不从PNG重新计算。

Provenance的previous_state_hash表示前一帧**位置**，不是声称速度history严格连续。consumer_previous_state_reset另记现有模板边界prev_state重设；boundary_update表示当前预测帧触发在线模板更新。后续cache_key指在线prediction模板，不表示读取磁盘未来cache。19次边界重设保持不变；无future PLY/GT Gaussian/future cache注入。

## CPU视频

`make_t31_video.py`只依赖NumPy/Pillow/ffmpeg，不导入torch/SoMA。两行四列：RGB、RGB×object mask、现有render、uint8绝对误差。误差是展示，不替代浮点域指标。10fps仅播放节奏，不代表原数据采样率。194帧，1280×490，19.4秒；ffprobe核验通过。抽查source213画面：相机、标签和四列布局可辨，无行覆盖。

```bash
python tools/deform360_adapter/make_t31_video.py <artifact_directory>
```

重新导出另一个checkpoint时，通过原推理工具的`--checkpoint`、`--checkpoint-sha256`、`--expected-epoch`、`--expected-iter`及全新`--output`目录；加`--export-artifacts`。仍须Slurm。不得覆盖旧artifact或用未来checkpoint改写本次结论。

## 工程边界

只修改独立观察工具，新增序列化/CPU视频工具；模型、renderer、rollout、state update、evaluation协议及配置均未改。schema见manifest.schema.json，跨记录/实际数组验证见validate_artifacts.py。只读schema说明不能代替读回验证。

全部大产物留服务器；Mac只保存小型contract/README/schema/validation及工具。39KB临时视频抽帧仅在Mac /tmp做视觉检查，不纳入Git。未commit/push。原T31/T32和T30未提交工作保留。
