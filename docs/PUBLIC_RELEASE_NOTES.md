# 公开版说明（脱敏与复现）

本仓库是内部研究仓库的**净化发布版**：不含内部提交历史；云端账号、网关主机名、
K8s 命名空间/集群 ID、节点名与内网 IP 均已替换为占位符（脚本内替换为确定性映射，
替换后扫描命中数为 0）。

| 占位符 | 含义 | 回填位置 |
|---|---|---|
| `<HPC_HOST>` / `<HPC_DOMAIN>` / `<HPC_GATEWAY>` | HPC 网关主机/域名 | ssh 目标、投递脚本 |
| `<HPC_ACCOUNT>`（及 `_B`/`_C`） | HPC 账号名（`/work/home/<账号>`） | `configs/*.yaml` 的 `ns3.binary`、`ns3.cwd` |
| `<HPC_VENDOR>` / `<HPC_SSH_ALIAS>` | 平台名与本机 ssh 别名 | `~/.ssh/config` |
| `<K8S_NAMESPACE>` / `<K8S_CLUSTER_ID>` | 容器组命名空间与 kubeconfig 目录 | 运维脚本（本仓库未含） |
| `<LOGIN_NODE>`、`<POD_GROUP*>` | 登录节点名、GPU 容器组前缀 | 运维脚本（本仓库未含） |

**本地即可复现**：`pytest tests/ -q` → 148 passed（走 `scripts/ns3_mock.py` 协议桩，
无需 NS-3）。要复现真实 NS-3 数值，需自行编译 `ns3/hsr-rl.cc`（NS-3.43 + LENA），
并把 `configs/*.yaml` 的 `ns3.binary` 指向你的二进制。云端投递/监督脚本与逐集原始证据
（约 150 MB）属于内部运维，未随公开版发布；方法论见
`docs/整段路径测评体系与阶段化方案.md`，验收判据与各轮终判见 `reports/`。

**指标口径**：README 中 H13 的吞吐/时延数字为「同 seed 配对 + legacy 动作掩码 +
弱目标硬盾 + 切换后测量新鲜性门控」尺下实测（2026-10-14/16 在新 HSR4 二进制上复测，
与 10-07 验收逐族一致到 ±0.02 Mbps）。换推理尺结论会变，见
`reports/H16_acceptance_criteria_v2_prereg.md` 的 v2.1 修订。
