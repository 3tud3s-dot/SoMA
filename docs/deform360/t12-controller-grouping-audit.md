# T12 controller hierarchy audit

日期：2026-09-25。结论：**FAIL：当前默认分组跨手指；修复方案仅做内存比较，尚未应用。**

T11 checkpoint：`e755d1e3b3060dca7f7cc16b29b63ff8202c42c0`，已推送到 `origin/deform360-adaptation`，进入 T12 前 Mac working tree clean。

## 方法与范围

使用服务器真实 T11 `candidate_7mm` trajectory 的首帧，以及现有 `_load_controller_cluster_mask` 函数。仅把文件读写、目录创建与 cache 存在检查替换为内存操作，执行真实 `dis_split` 分支。未生成磁盘 PKL/cache，未构建模型 graph 或 rollout，未修改任何配置、源码、几何或 trajectory。该函数仅用首帧确定分组；连续分组不依赖坐标值，固定索引关系适用于完整时序。源文件 SHA256 已与 Mac 核对一致；输入/hash 和完整映射见 [JSON 证据](contracts/008-pink-cloth/episode_0/controller_grouping_audit.json)。

## 结果与最小方案

| 方案 | 层级数量（含原始点和共同根） | 第一聚合层混合 | 双手指层混合 |
|---|---|---|---|
| 默认 `.5, .5` | 30 → 15 → 2 → 1 | C7：14(left)+15(right) | C1：14–29 |
| 仅建议：首层 `num_cluster=10` | 30 → 10 → 2 → 1 | 无 | 无：0–14 / 15–29 |

最后的 `2→1` 是 loader 显式附加的共同 controller root，包含双指是预期共同祖先；它与手指层的意外混合不同。表格仍如实把共同根标为跨指。T12 当前失败是因为双手指层未能得到纯 left / right，而非共同根存在。

- **建议**：未来 v0 Stage 1/2 controller scheme 的第一项使用 `dict(num_cluster=10)`，保留第二项。此处仅比较，未创建或修改配置。每个首层 cluster 含连续 3 点，两指各 5 个 cluster；保持深度、30 点身份和 T10/T11 数值不变。中间节点数由 15 减至 10，centroid/拓扑会改变，不能声称图完全等价；完整图验证仍未执行。
- 仅重排点不可消除默认全部成对分组的混合：每指 15 是奇数，无法各自完全划分为纯指二元组；还会改变已固定的数组身份约定。不建议重排。
- 修改 loader 为 label-aware grouping 需要代码改动并处理下一层边界，范围大于配置一项。不建议作为首选。
- 仅改变第二项 downsample rate 无效：`dis_split` 后续层直接使用 `n_points // 2`。首层改为 30 个 cluster 也可保留双指，但不做首层聚合、节点更多。
- 新 scheme 的 cache key 不同；今后执行修复时应核对 key 和实际映射，不覆盖或删除旧 cache。本轮所有 cache 操作均为内存模拟。
- 保留 7 mm negative signed gap 和官方 opening clipping，不据此认证真实接触几何，不执行 T13。等待人工确认 grouping 方案。

## 源码位置

- `mmgs/datasets/embodied_dataset.py::_load_controller_cluster_mask`，611–747 附近；617–621 读取首帧和两个 controller；630–650 cache key；679–711 首层连续分组；713–729 后续二分；737–741 附加共同根。
- `configs/SoMA/cloth_lift_stage1.py` 85–87、`cloth_lift_stage2.py` 86–88：默认两层 `.5` scheme。
- `mmgs/datasets/embodied_dataset.py::_merge_cluster_mask` 454–464：合并层级并附加占位映射。
- `mmgs/models/utils/dgl_graph.py::GsHieEmbodiedDGLProcessor.batch_preprocess` 613–648：处理层级，最后额外占位映射不参与循环；不能把 loader 共同根与该占位混同。

## 每层完整成员

以下每项格式为 `原始 index:point_id (left/right)`。L0 为原始点，L1 为首次聚合，L2 为双手指层，L3 为共同根。

### 当前默认

#### L0：30 clusters

| Cluster | 原始点及标签 | 跨手指 |
|---|---|---|
| 0 | 0:left_r00_c00 (left) | 否 |
| 1 | 1:left_r00_c08 (left) | 否 |
| 2 | 2:left_r00_c16 (left) | 否 |
| 3 | 3:left_r00_c23 (left) | 否 |
| 4 | 4:left_r00_c31 (left) | 否 |
| 5 | 5:left_r06_c00 (left) | 否 |
| 6 | 6:left_r06_c08 (left) | 否 |
| 7 | 7:left_r06_c16 (left) | 否 |
| 8 | 8:left_r06_c23 (left) | 否 |
| 9 | 9:left_r06_c31 (left) | 否 |
| 10 | 10:left_r11_c00 (left) | 否 |
| 11 | 11:left_r11_c08 (left) | 否 |
| 12 | 12:left_r11_c16 (left) | 否 |
| 13 | 13:left_r11_c23 (left) | 否 |
| 14 | 14:left_r11_c31 (left) | 否 |
| 15 | 15:right_r00_c00 (right) | 否 |
| 16 | 16:right_r00_c08 (right) | 否 |
| 17 | 17:right_r00_c16 (right) | 否 |
| 18 | 18:right_r00_c23 (right) | 否 |
| 19 | 19:right_r00_c31 (right) | 否 |
| 20 | 20:right_r06_c00 (right) | 否 |
| 21 | 21:right_r06_c08 (right) | 否 |
| 22 | 22:right_r06_c16 (right) | 否 |
| 23 | 23:right_r06_c23 (right) | 否 |
| 24 | 24:right_r06_c31 (right) | 否 |
| 25 | 25:right_r11_c00 (right) | 否 |
| 26 | 26:right_r11_c08 (right) | 否 |
| 27 | 27:right_r11_c16 (right) | 否 |
| 28 | 28:right_r11_c23 (right) | 否 |
| 29 | 29:right_r11_c31 (right) | 否 |

#### L1：15 clusters

| Cluster | 原始点及标签 | 跨手指 |
|---|---|---|
| 0 | 0:left_r00_c00 (left); 1:left_r00_c08 (left) | 否 |
| 1 | 2:left_r00_c16 (left); 3:left_r00_c23 (left) | 否 |
| 2 | 4:left_r00_c31 (left); 5:left_r06_c00 (left) | 否 |
| 3 | 6:left_r06_c08 (left); 7:left_r06_c16 (left) | 否 |
| 4 | 8:left_r06_c23 (left); 9:left_r06_c31 (left) | 否 |
| 5 | 10:left_r11_c00 (left); 11:left_r11_c08 (left) | 否 |
| 6 | 12:left_r11_c16 (left); 13:left_r11_c23 (left) | 否 |
| 7 | 14:left_r11_c31 (left); 15:right_r00_c00 (right) | 是 |
| 8 | 16:right_r00_c08 (right); 17:right_r00_c16 (right) | 否 |
| 9 | 18:right_r00_c23 (right); 19:right_r00_c31 (right) | 否 |
| 10 | 20:right_r06_c00 (right); 21:right_r06_c08 (right) | 否 |
| 11 | 22:right_r06_c16 (right); 23:right_r06_c23 (right) | 否 |
| 12 | 24:right_r06_c31 (right); 25:right_r11_c00 (right) | 否 |
| 13 | 26:right_r11_c08 (right); 27:right_r11_c16 (right) | 否 |
| 14 | 28:right_r11_c23 (right); 29:right_r11_c31 (right) | 否 |

#### L2：2 clusters

| Cluster | 原始点及标签 | 跨手指 |
|---|---|---|
| 0 | 0:left_r00_c00 (left); 1:left_r00_c08 (left); 2:left_r00_c16 (left); 3:left_r00_c23 (left); 4:left_r00_c31 (left); 5:left_r06_c00 (left); 6:left_r06_c08 (left); 7:left_r06_c16 (left); 8:left_r06_c23 (left); 9:left_r06_c31 (left); 10:left_r11_c00 (left); 11:left_r11_c08 (left); 12:left_r11_c16 (left); 13:left_r11_c23 (left) | 否 |
| 1 | 14:left_r11_c31 (left); 15:right_r00_c00 (right); 16:right_r00_c08 (right); 17:right_r00_c16 (right); 18:right_r00_c23 (right); 19:right_r00_c31 (right); 20:right_r06_c00 (right); 21:right_r06_c08 (right); 22:right_r06_c16 (right); 23:right_r06_c23 (right); 24:right_r06_c31 (right); 25:right_r11_c00 (right); 26:right_r11_c08 (right); 27:right_r11_c16 (right); 28:right_r11_c23 (right); 29:right_r11_c31 (right) | 是 |

#### L3：1 clusters

| Cluster | 原始点及标签 | 跨手指 |
|---|---|---|
| 0 | 0:left_r00_c00 (left); 1:left_r00_c08 (left); 2:left_r00_c16 (left); 3:left_r00_c23 (left); 4:left_r00_c31 (left); 5:left_r06_c00 (left); 6:left_r06_c08 (left); 7:left_r06_c16 (left); 8:left_r06_c23 (left); 9:left_r06_c31 (left); 10:left_r11_c00 (left); 11:left_r11_c08 (left); 12:left_r11_c16 (left); 13:left_r11_c23 (left); 14:left_r11_c31 (left); 15:right_r00_c00 (right); 16:right_r00_c08 (right); 17:right_r00_c16 (right); 18:right_r00_c23 (right); 19:right_r00_c31 (right); 20:right_r06_c00 (right); 21:right_r06_c08 (right); 22:right_r06_c16 (right); 23:right_r06_c23 (right); 24:right_r06_c31 (right); 25:right_r11_c00 (right); 26:right_r11_c08 (right); 27:right_r11_c16 (right); 28:right_r11_c23 (right); 29:right_r11_c31 (right) | 是 |

### 建议方案（未应用）

#### L0：30 clusters

| Cluster | 原始点及标签 | 跨手指 |
|---|---|---|
| 0 | 0:left_r00_c00 (left) | 否 |
| 1 | 1:left_r00_c08 (left) | 否 |
| 2 | 2:left_r00_c16 (left) | 否 |
| 3 | 3:left_r00_c23 (left) | 否 |
| 4 | 4:left_r00_c31 (left) | 否 |
| 5 | 5:left_r06_c00 (left) | 否 |
| 6 | 6:left_r06_c08 (left) | 否 |
| 7 | 7:left_r06_c16 (left) | 否 |
| 8 | 8:left_r06_c23 (left) | 否 |
| 9 | 9:left_r06_c31 (left) | 否 |
| 10 | 10:left_r11_c00 (left) | 否 |
| 11 | 11:left_r11_c08 (left) | 否 |
| 12 | 12:left_r11_c16 (left) | 否 |
| 13 | 13:left_r11_c23 (left) | 否 |
| 14 | 14:left_r11_c31 (left) | 否 |
| 15 | 15:right_r00_c00 (right) | 否 |
| 16 | 16:right_r00_c08 (right) | 否 |
| 17 | 17:right_r00_c16 (right) | 否 |
| 18 | 18:right_r00_c23 (right) | 否 |
| 19 | 19:right_r00_c31 (right) | 否 |
| 20 | 20:right_r06_c00 (right) | 否 |
| 21 | 21:right_r06_c08 (right) | 否 |
| 22 | 22:right_r06_c16 (right) | 否 |
| 23 | 23:right_r06_c23 (right) | 否 |
| 24 | 24:right_r06_c31 (right) | 否 |
| 25 | 25:right_r11_c00 (right) | 否 |
| 26 | 26:right_r11_c08 (right) | 否 |
| 27 | 27:right_r11_c16 (right) | 否 |
| 28 | 28:right_r11_c23 (right) | 否 |
| 29 | 29:right_r11_c31 (right) | 否 |

#### L1：10 clusters

| Cluster | 原始点及标签 | 跨手指 |
|---|---|---|
| 0 | 0:left_r00_c00 (left); 1:left_r00_c08 (left); 2:left_r00_c16 (left) | 否 |
| 1 | 3:left_r00_c23 (left); 4:left_r00_c31 (left); 5:left_r06_c00 (left) | 否 |
| 2 | 6:left_r06_c08 (left); 7:left_r06_c16 (left); 8:left_r06_c23 (left) | 否 |
| 3 | 9:left_r06_c31 (left); 10:left_r11_c00 (left); 11:left_r11_c08 (left) | 否 |
| 4 | 12:left_r11_c16 (left); 13:left_r11_c23 (left); 14:left_r11_c31 (left) | 否 |
| 5 | 15:right_r00_c00 (right); 16:right_r00_c08 (right); 17:right_r00_c16 (right) | 否 |
| 6 | 18:right_r00_c23 (right); 19:right_r00_c31 (right); 20:right_r06_c00 (right) | 否 |
| 7 | 21:right_r06_c08 (right); 22:right_r06_c16 (right); 23:right_r06_c23 (right) | 否 |
| 8 | 24:right_r06_c31 (right); 25:right_r11_c00 (right); 26:right_r11_c08 (right) | 否 |
| 9 | 27:right_r11_c16 (right); 28:right_r11_c23 (right); 29:right_r11_c31 (right) | 否 |

#### L2：2 clusters

| Cluster | 原始点及标签 | 跨手指 |
|---|---|---|
| 0 | 0:left_r00_c00 (left); 1:left_r00_c08 (left); 2:left_r00_c16 (left); 3:left_r00_c23 (left); 4:left_r00_c31 (left); 5:left_r06_c00 (left); 6:left_r06_c08 (left); 7:left_r06_c16 (left); 8:left_r06_c23 (left); 9:left_r06_c31 (left); 10:left_r11_c00 (left); 11:left_r11_c08 (left); 12:left_r11_c16 (left); 13:left_r11_c23 (left); 14:left_r11_c31 (left) | 否 |
| 1 | 15:right_r00_c00 (right); 16:right_r00_c08 (right); 17:right_r00_c16 (right); 18:right_r00_c23 (right); 19:right_r00_c31 (right); 20:right_r06_c00 (right); 21:right_r06_c08 (right); 22:right_r06_c16 (right); 23:right_r06_c23 (right); 24:right_r06_c31 (right); 25:right_r11_c00 (right); 26:right_r11_c08 (right); 27:right_r11_c16 (right); 28:right_r11_c23 (right); 29:right_r11_c31 (right) | 否 |

#### L3：1 clusters

| Cluster | 原始点及标签 | 跨手指 |
|---|---|---|
| 0 | 0:left_r00_c00 (left); 1:left_r00_c08 (left); 2:left_r00_c16 (left); 3:left_r00_c23 (left); 4:left_r00_c31 (left); 5:left_r06_c00 (left); 6:left_r06_c08 (left); 7:left_r06_c16 (left); 8:left_r06_c23 (left); 9:left_r06_c31 (left); 10:left_r11_c00 (left); 11:left_r11_c08 (left); 12:left_r11_c16 (left); 13:left_r11_c23 (left); 14:left_r11_c31 (left); 15:right_r00_c00 (right); 16:right_r00_c08 (right); 17:right_r00_c16 (right); 18:right_r00_c23 (right); 19:right_r00_c31 (right); 20:right_r06_c00 (right); 21:right_r06_c08 (right); 22:right_r06_c16 (right); 23:right_r06_c23 (right); 24:right_r06_c31 (right); 25:right_r11_c00 (right); 26:right_r11_c08 (right); 27:right_r11_c16 (right); 28:right_r11_c23 (right); 29:right_r11_c31 (right) | 是 |
