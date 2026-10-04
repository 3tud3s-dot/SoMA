# SoMA-D360 跨电脑交接

更新时间：2026-10-04。此次只保存代码、文档、学习记录与小型报告资产，不迁移服务器数据或CUDA环境。

## 仓库与路径

- workspace：`https://github.com/3tud3s-dot/tcgs.git`，`main`。它忽略三个独立子仓库，clone workspace不会自动下载子仓库。
- SoMA：`https://github.com/3tud3s-dot/SoMA.git`，`deform360-adaptation`。
- Deform360：`https://github.com/lhy0807/deform360.git`，本项目源码基线`d8522a4403b766aeb387510c04e89032a56fdf35`。
- 服务器workspace：`/data1/userdata/tcweng/projects/tcgs`；原Mac SSH alias为`amax`，新电脑需使用其实际可用连接方式。
- PhysTwin非当前SoMA任务必需。原电脑PhysTwin有官方远端没有的两个提交，本次未迁移；不要声称clone官方仓库可恢复它们。

## 开始工作前

先读workspace `AGENTS.md`、本文件及`docs/deform360/SoMA-D360-v0-implementation-todo.md`，具体结论查contracts/validation。AGENTS中“root不是Git”的描述属于早期快照；用户已授权建立并使用tcgs root Git。各子仓库边界不变。

本机用于编辑/文档/Git；GPU、数据和训练继续在服务器。不要在新电脑安装服务器CUDA环境，也不要下载数据/checkpoint。SSH凭据、Codex聊天和本机environment记录不随Git自动迁移。

当前本机不含SoMA的`gaussian-splatting/`运行依赖目录；服务器有既有运行环境。普通源码/文档工作可以进行，若需阅读该目录从服务器只读获取必要源码；不要凭此重新安装训练环境。

## 状态与边界

- Stage-1 T26已完成46epochs/2300steps，T27 cache及T28/T29已验证。
- Stage-2曾完成epoch22/iter13200；job25884从该checkpoint恢复。用户现确认25884被服务器故障中断，未如期跑完；当前机器无法访问服务器，最新完整checkpoint/作业状态未知，必须在能连接的机器核验，不照抄旧IN_PROGRESS记录。
- 此次迁移不授权自动启动训练、停止作业或覆盖服务器源码。续跑按独立用户授权进行，使用最新健康checkpoint和真实runner.resume，保留61epochs/36600steps总预算与Adam/Normalizer/counters。
- T31现有结果是epoch22中间checkpoint的探索性结果，不是最终baseline；每10帧prev_state重设限制保留。
- T32报告只有草稿资产，正文未完成。学习状态另见`claude-learning/`及`learning/source-reading-roadmap.md`，不是工程验收状态。
- W&B历史导入工具和文档已保留；不要重复上传已导入run。迁移不读取或拷贝token。

两端同步先看git status/branch/HEAD/diff；dirty tree不覆盖、不stash/clean，不自动push官方upstream。新电脑开发后提交到SoMA开发分支，原电脑回来时在clean状态fetch并fast-forward即可。Git不携带旧Codex会话，本文件和roadmap用于接续。
