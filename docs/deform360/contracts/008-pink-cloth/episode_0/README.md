# SoMA-D360-v0 contract 权威版本

本目录的 `frame_manifest.json` 和 `split_contract.json` 是长期 contract 的权威版本，应随 SoMA 仓库进入 Git 管理并同步到 Mac 与服务器。归档时保持服务器原始 JSON 字节不变；本次尚未暂存、commit 或 push。

以下路径均相对于 tcgs root。

| Artifact | 权威仓库路径 | 原始生成位置（provenance） |
|---|---|---|
| Frame manifest | `SoMA/docs/deform360/contracts/008-pink-cloth/episode_0/frame_manifest.json` | `datasets/soma_d360_v0/008-pink-cloth/episode_0/t4_frame_mapping/frame_manifest.json` |
| Split contract | `SoMA/docs/deform360/contracts/008-pink-cloth/episode_0/split_contract.json` | `datasets/soma_d360_v0/008-pink-cloth/episode_0/t5_split_contract/split_contract.json` |

## 引用与 provenance

- 后续读取 split contract 的 frame mapping 依赖时，使用本目录的 `frame_manifest.json`。JSON 中原有 `frame_mapping_path` 保留原始生成位置，不应作为唯一查找路径；上表定义其仓库内权威引用。本轮未修改或实现 loader 路径解析。
- `input_sha256` 与 `timestamp_file_sha256` 保留原始输入路径及 hash，不因归档而改写。split contract 中原 frame manifest 路径对应的 hash 同样适用于本目录逐字节一致的副本。
- 数据集、timestamps 与 `evaluation.initial_gaussian` 所引用的 PLY 仍为 server-only 数据依赖；路径相对于服务器 tcgs root 解析。它们不因 JSON 归档而被复制或纳入 Git。
- 原始生成文件保留，供历史追溯；未来 contract 变更应更新仓库内权威版本，不从旧生成位置无条件覆盖。

## 归档校验

| 文件 | Bytes | SHA256 |
|---|---:|---|
| `frame_manifest.json` | 528101 | `d48338cf7a606b1ee4bc2e5a560b5a53c017efdc9880a1db282a8494b1b013db` |
| `split_contract.json` | 10753 | `fd1370b2202836c2c8be2589955bca33bbe88e38a067fb00bb937e48500debd1` |

本目录不包含 PLY、视频、HDF5、NumPy 数据、cache 或 checkpoint。
