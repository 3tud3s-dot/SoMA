# T30：24小时基础设施续跑

2026-09-28，用户明确授权 T30 再续24小时。保留job25875 TIMEOUT/22完整epochs历史。不是从头训练；不重启T31，不执行T27或改算法。

- Job：25884；partition5090，amax，1×RTX5090；sbatch时限24h。
- Git HEAD：9cd6fe0706f1b1ad4f4f1b0430bbaafb5dccd16e，服务器源码clean；本地此前T31/T32未提交内容保留。
- Frozen config SHA256：c4042fa409bdb4bc2746629a390f9763aa7f277a8ec54d2e43e314b30566e16a，配置未改。
- 从`outputs/deform360/stage2/config_a_seed5/epoch_22.pth`恢复，SHA256 `31223b8ca972db83a78412040ddee9dcf7e5836f63fb11cae54b5f91aec9380c`；size30039037B。
- CPU preflight确认model/Adam/12FP64stats finite，epoch22/iter13200/Adamstep13200。未初始化CUDA。
- GPU内使用真实runner.resume：model/statistics与optimizer全部递归exact比较通过，counter恢复22/13200。不是仅load weights或fresh Adam。checkpoint未保存RNG全状态，因此不宣称与未中断执行bitwise等价。
- 首个恢复batch：human epoch23/iter13201，requested rollout69，窗口限制effective12，LR4.1953125e-6；cache70、source183→195。loss2874.18359375，275梯度finite，preclip18366.31097/postclip0.9999999935，Adamstep13201。model/Adam/Normalizer全部finite。
- 完整目标仍61epochs/36600steps。上次超时前未保存的188updates不计作已恢复状态；本次从最后完整epoch边界开始。
- 仅修改独立observer的resume断言、累计计数、日志路径与sbatch时限；生产源码、config、数据、cache未改。每epoch保存和原train-side evaluation保持不变。
- workdir仍`outputs/deform360/stage2/config_a_seed5/`，新日志在`outputs/deform360/t30-resume24h-control-20260928/`。原epoch1–22及旧日志保留。
- 观察器失败即停止：非有限/OOM/schema不自动修复，不自动重跑。24h到时未完成也不自动再提交。
- 每5个新完整epoch报告并更新ETA，下次epoch27。按旧最近5epoch中位2043.546秒，剩余39epoch约22.14h；仅估算。

本次状态IN_PROGRESS，不是T30 PASS。61epochs全部完成后仍须CPU核验最终checkpoint。未commit/push。
