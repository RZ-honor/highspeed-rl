# configs 说明

本目录存放训练、评估与阶段实验配置。

## 常用入口

- `default.yaml`：本地/快速冒烟基线
- `bc_ns3.yaml`：Stage 1 行为克隆（BC）常用配置
- `ns3.yaml`：Stage 2 NS-3 在线训练常用配置
- `eval_*.yaml`：分场景评估配置（normal/tunnel/loadimbal/congested）

## 历史阶段配置

`ns3_stage*.yaml`、`bc_stage*.yaml` 为阶段性实验留档（H4/H8/H13/H14/H15/H16 等）。
使用前请在实验记录中标注来源与用途，避免与当前默认流程混淆。

## 使用建议

1. 先复制一份配置再改，不直接覆盖历史阶段文件。
2. 评估时优先使用 `best.pt` 对应配置，避免 `latest.pt` 漂移。
3. 云端路径占位符说明见 `docs/PUBLIC_RELEASE_NOTES.md`。

