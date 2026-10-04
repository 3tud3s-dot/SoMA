# T32 交付报告草稿资产（尚未完成）

迁移快照：2026-10-04。本目录只保存已整理的图表输入、表格、T18静态参考图和绘图脚本；中文科研报告正文尚未完成，不表示T32 PASS。

- `assets/report_data.json`：既有验证报告的统计摘录；历史Stage-2数据截至epoch22，不代表服务器当前进度。
- `assets/source_manifest.json`：原始输入路径/hash。
- `assets/build_figures.py`：CPU绘图和数值回放生成脚本，本次迁移不运行。
- `assets/t18_*_static_comparison.jpg`：初始静态对照，不是T31动态预测视频。
- `tables/`：既有训练/rollout数值和来源表。

此前图表生成在服务器临时目录`/tmp/t32-report-20260927/`执行，本地没有取回最终图/视频；临时文件是否仍存在需另行核验。T31原始检查未保存预测渲染帧，不能把数值动画称为RGB/GT/预测对比视频。本次不重跑T31、不训练、不补造结果。
