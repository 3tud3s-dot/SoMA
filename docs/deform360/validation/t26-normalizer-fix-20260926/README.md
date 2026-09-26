# T26 Normalizer-fix rerun — BLOCKED / observer logging

2026-09-26。**本轮不是 NaN/Inf 或 OOM，而是观测日志写入失败。** 已停止，未重试、未resume、未执行T27。

## 前置与启动

T26.5以 `0a628fa4dec9d8d267b131187f06b9582bc18c7b` 提交并push（Validate Stage-1 formal first optimizer step），Mac/origin/server同HEAD、0/0、clean。server Normalizer已入Git，无未提交patch。

- sbatch **25867**，amax / NVIDIA GeForce RTX 5090，Slurm FAILED / exit1:0，elapsed **83秒**。
- fresh seed5 / model / optimizer / zero-history FP64 normalizer在首batch断言通过。`load_from=None`、`resume_from=None`，无warm-up。
- 独立输出：`/data1/userdata/tcweng/projects/tcgs/outputs/deform360/stage1/config_a_seed5_normalizer_fix`；旧FAIL run、T26.5 artifact均保留。
- config SHA256：`e65d1c7848bd222f55c0e18050c464eec5922aef76c1bc62a06813359cca4930`。
- Normalizer SHA256：`f78712dd0760fcf16a58431d7fdfd240a2e38cba89c400b405df93798f364220`。
- 与初次T26的production差异仅为已批准 Normalizer FP64 statistics 修复；frozen schedule/LR/loss/detach/clip均不变。

## 完成到哪里

**1个训练epoch / 50次optimizer step**，未到epoch2。

| Epoch | Training rollout | Steps | LR | Total loss first / mean / last | Preclip norm min–max | Peak allocated / reserved |
|---|---|---|---|---|---|---|
| 1 | 3 | 50 | 0.000404 | 23798.4258 / 8077.1483 / 5499.4512 | 77383.4917–430046.2908 | 3.7030 / 3.9785 GiB |

平均loss components：momentum3.325205、L2 5842.164014、SSIM2231.659058，static0。50次梯度检查均finite；postclip global norm最大1.000000134（浮点舍入，clip阈值仍1）。50次更新后model/Adam/normalizer检查均finite。

训练仅source113→123/133/143。DataLoader读取source113…263；原train-only evaluation完成source123…263的15个预测step。**evaluation rollout15不能当作training rollout15通过**；训练rollout6/9/12/15均未到达。

每步position等于前一步predicted position的断言通过；原initial covariance复用语义保持，无future PLY reset。794项source assets/hash在结束后再次通过核验。

## 首个失败

epoch1训练完成、checkpoint保存后，原train-only evaluation已经返回。观察器 `evaluate:243` → `event:70` 调用JSON序列化，遇到NumPy `float32` metric：

```text
TypeError: Object of type float32 is not JSON serializable
```

这是本轮复用/扩展观察器时漏掉的类型兼容问题，不是模型训练错误。`clean()`只处理Python float/dict/list，没有转换NumPy scalar。随后finally的`save()`同样失败，导致原 `training_report.json` 仍停在checkpoint后的 **IN_PROGRESS** 快照。

**原始report不改写；最终状态以本README、summary.json、postrun_verification.json、stderr和Slurm记录为准。** 没有把这个logging failure伪装成OOM/nonfinite，也没有借机自动修改或恢复训练。

## Last-good checkpoint

唯一epoch checkpoint：

```text
/data1/userdata/tcweng/projects/tcgs/outputs/deform360/stage1/config_a_seed5_normalizer_fix/epoch_1.pth
```

- size：**30038461 bytes**。
- SHA256：`ac600808287c27277edf39dba59d0bf21a7d90204345cf479a06c8d00bf173ae`。
- metadata：epoch=1，iter=50；model/optimizer均存在。
- 独立CPU读取复核：model/Adam全部finite，275组Adam step均为50；12个normalizer统计均FP64、finite。
- 没有实例化model或resume，不是再次T24恢复实验。未将checkpoint下载Mac/加入Git。
- `latest.pth`只是指向epoch_1.pth的symlink；没有final epoch_46.pth。

## 后续最小建议（未实施）

仅修观察器的JSON清理：支持NumPy scalar（必要时array）及错误路径，并先做CPU序列化回归。之后由用户决定新的训练恢复方式。本轮未修改该observer执行副本、未自动重跑或resume。

## 文件与Git

- [最终summary](summary.json)
- [原始checkpoint时刻report（stale IN_PROGRESS）](training_report.json)
- [CPU核验](postrun_verification.json)
- [完整stderr](stderr.txt)、[stdout](stdout.txt)、[事件记录](events.jsonl)
- [原样运行观察器](run_stage1_observed.py)、[sbatch](stage1.sbatch)、[preflight](preflight.json)、[展开配置](expanded_config.py)

当前所属SoMA/deform360-adaptation；roadmap和本目录未commit/push。server repo clean，local/origin/server提交仍一致0/0；服务器数据/训练产物不入Git。保留T26 initial nonfinite FAIL和T26.1–T26.5全部历史，不执行T27。
