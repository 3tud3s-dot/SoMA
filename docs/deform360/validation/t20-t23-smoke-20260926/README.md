# T20–T23 Config A smoke batch

结果：各项独立 PASS；无正式训练、无 T24。一次用户授权串行执行，前项 PASS 后才运行下一项。

Slurm job 25824；NVIDIA GeForce RTX 5090, 32607 MiB；SoMA HEAD `2cd8dd6cfafe329499930c4c6bf52763beb796ab`。

| TODO | 耗时 s | peak allocated GiB | total loss |
|---|---:|---:|---:|
| T20 | 7.656 | 0.258 | — |
| T21 | 0.469 | 1.387 | 59908.2734375 |
| T22 | 0.762 | 1.390 | 23067.3828125 |
| T23 | 0.758 | 3.625 | 18705.154296875 |

T20 耗时含 imports、dataset/model 初始化；其他为对应步骤的墙钟时间，检查中的 CUDA 数值读取进行了同步，不是独立性能 benchmark。

## 协议与执行路径

Config A 顺序为 brics-odroid-023_cam0、brics-odroid-009_cam1。source113=local0；训练 gap10；T19 sample 监督范围113…263。T21/T22预测113→123；T23从同一初始态预测123、133、143。controller=7mm，p2c=30→10→2→1，gravity external=[0,0,-39.2]，initial SH0=12861点，640×360。

模型参数沿用官方 cloth_lift Stage1（16 encoder layers、share_weight、loss、dt=1/15等），只替换 scene/cluster/controller 入口与 smoke rollout。来源为历史路径重定位配置，完整配置及 hash 记录在 report.json。optimizer=Adam(lr=0.0004, betas=[0.9,0.999], weight_decay=0, amsgrad=False)，没有做超参选择。

T20 调用真实 model._preprocess；T21 直接使用 T20 graph 调用真实 encode_decode/_encode_decode_train，仅 forward/render/loss，无 backward；T22 同一 sample/config 重新构图做一次 forward/backward/optimizer.step；T23 使用同一模型实例在 T22 更新后调用真实 forward_train(rollout=3)，再一次 backward，无额外 optimizer.step。T23 方法观察包装只调用原方法并核验输入/返回，没有更改算法或源码。

## Graph

| layer | object nodes | controller nodes | edges |
|---|---:|---:|---:|
| 0 | 12861 | 30 | 0 |
| 1 | 640 | 10 | 9768 |
| 2 | 7 | 2 | 72 |
| 3 | 1 | 1 | 2 |

leaf graph 的 spatial edge=0 符合原 forward_last_layer=False 路径；层级连边另记于 report.json.connections。每层 pin mask、p2c覆盖/索引范围、node/edge特征 shape/finite 均验证。不是把0条leaf spatial edge误报成整图无边。

## 梯度与自回归

T22：275个参数有非零有限梯度，275个参数发生有限变化，梯度global norm=81870.2374。T23：梯度global norm=136866.0205，有限。Gaussian输入参数不在 model.parameters()/optimizer集合中，原Gaussian tensors和源数据文件hash保持不变。

两项的 backbone.encoder.emb_norm.weight/bias 没有梯度；meshgraphnet_embodied.py:148定义该层，但该文件没有使用它的调用。这不等于所有已注册参数均获梯度，证据完整保留 missing 列表，未顺手修模型。

T23 step2/3 的 cur_state 与前一步 predicted object slice 逐值相等，prev_state 与前一步输入相等，controller及GT索引分别严格匹配。保留 accumulate_gradient=False 的 detach；不是跨三步完整BPTT。原实现 cur_cov 更新被注释，本轮保持原行为，不宣称验证了新的协方差自回归方案。

T21/T22 loss差异不能当作学习改善：T21无optimizer step，但原实现 register_norm=True及训练路径会更新normalizer统计；T23还使用T22更新后的模型权重。这里只验证数值/梯度连通性，非质量指标。

## 证据与边界

完整 loss、逐步state来源、预测/render shapes、参数梯度/变化、memory、source/data hash：report.json。完整原始输出：run.txt。工具：tools/deform360_adapter/smoke_dynamics_batch.py。

服务器命令：`srun -p 5090 --gres=gpu:1 --ntasks=1 --cpus-per-task=4 --mem=24G --time=00:20:00 --job-name=tcgs-t20-23 <soma-env>/bin/python -B /tmp/tcgs_t20_t23_smoke.py --workspace /data1/userdata/tcweng/projects/tcgs --report /tmp/tcgs_t20_t23_report.json`。退出码0。CUDA全部在allocation内。

未保存模型checkpoint，没有训练权重文件同步Mac；本次optimizer更新仅存在已退出的进程内。没有修改loader/simulator/geometry/loss/data，未执行T24。小型工具/报告待确认后提交。
