# highspeed-rl（高铁场景智能切换 RL）

面向 300–350 km/h 高铁场景的强化学习切换决策项目。核心为 Dueling-DDQN（含 MLP / ODE / Hybrid / Latent ODE 变体），与 3GPP A3 规则基线在统一评测口径下对比。

> 设计记录优先级：
> 1) **`/home/runner/work/highspeed-rl/highspeed-rl/docs/建模与训练完整方案.md`**（当前建模与训练主记录）
> 2) `CLAUDE.md`（开发/运维硬规则）
> 3) 其他演进与历史文档（用于背景与决策过程）

---

## 1. 项目简介

本仓库实现并维护一个双后端、同契约的切换决策训练系统：

- **Python 后端**：低成本训练/行为克隆/快速消融
- **NS-3 LENA 后端**：包级仿真、在线微调、验收评测

两后端对齐动作空间、奖励定义与核心 `info` 字段，支持“先便宜学、再真实域微调”的两阶段流程。

---

## 2. 核心能力与当前记录（非营销口径）

以下是仓库内“当前记录”口径，均可在文档中追溯，不代表所有配置都能复现同样数字：

| 项目 | 当前记录来源 |
|---|---|
| 当前生产候选基线（H13） | `CLAUDE.md` 的 H13 章节（2026-10-07 定稿记录） |
| 公开版脱敏与复现边界 | `docs/PUBLIC_RELEASE_NOTES.md` |
| 阶段性验收/判据 | `reports/` 下各阶段报告 |
| NS-3 后续差距与优化路线 | `docs/ns3_next_steps.md` |

> 建议：在汇报指标前，先标注“配置文件 + checkpoint + 场景 + seed + 判据版本”。

---

## 3. 整体架构（训练与仿真）

```text
Stage 1 (Python BC) -> Stage 2 (NS-3 online fine-tune) -> Paired evaluation (A3 vs RL)

src/envs/handover_env.py   (Python backend)
src/envs/ns3_env.py        (NS-3 backend wrapper)
ns3/hsr-rl.cc              (NS-3 LENA binary)
src/agents/*               (Dueling-DDQN and variants)
scripts/*                  (train/eval/baseline/sanity tools)
```

---

## 4. 仓库目录（按职责）

| 路径 | 说明 |
|---|---|
| `src/` | 环境、智能体、安全层、训练组件 |
| `scripts/` | 训练/评估/基线/探针脚本（见 `scripts/README.md`） |
| `configs/` | 训练与评估配置（见 `configs/README.md`） |
| `ns3/` | NS-3 C++ 场景与协议头文件 |
| `tests/` | Pytest 测试（含 mock/配置校验/核心逻辑） |
| `docs/` | 设计记录、评测方法、演进文档（见 `docs/README.md`） |
| `reports/` | 阶段验收与判据报告 |
| `CLAUDE.md` | 开发与云端运行硬规则（必须遵守） |

---

## 5. Python 与 NS-3 后端契约（读者先知道什么）

- 共享离散动作空间（7 动作语义保持一致）
- 共享奖励主逻辑（覆盖/吞吐/切换代价等）
- 关键 `info` 字段（如 `action_source`、`action_executed`）用于审计与训练一致性
- NS-3 侧对不支持能力（如 LENA 中 EN-DC 空缺）做显式屏蔽/近似，细节见：
  - `src/envs/ns3_env.py`
  - `docs/ns3_next_steps.md`
  - `docs/建模与训练完整方案.md`

---

## 6. 两阶段训练流程（最常用）

### Stage 1：行为克隆（Python，低成本）

```bash
cd /home/runner/work/highspeed-rl/highspeed-rl
python -m scripts.pretrain_bc --config configs/bc_ns3.yaml --net hybrid
```

### Stage 2：NS-3 在线微调（真实仿真）

```bash
cd /home/runner/work/highspeed-rl/highspeed-rl
python -m scripts.train_ns3 --config configs/ns3.yaml --device cuda --net hybrid --init checkpoints_ns3/bc_init.pt
```

> 建议先跑 `scripts/reward_sanity.py` 做门禁，再开长训练。

---

## 7. 从哪里开始（按你的目标）

| 你的目标 | 建议先读 | 建议先跑 |
|---|---|---|
| 只想理解项目全貌 | 本 README + `docs/README.md` + `docs/建模与训练完整方案.md` | 无 |
| 只想确认仓库可用 | `tests/README.md` | `python -m pytest tests/ -q` |
| 做 Python 预训练/克隆 | `scripts/README.md`（`pretrain_bc.py`）+ `configs/README.md` | `python -m scripts.pretrain_bc --config configs/bc_ns3.yaml --net hybrid` |
| 做 NS-3 训练 | `CLAUDE.md`（DCU/容器硬规则）+ `ns3/README_NS3.md` | `python -m scripts.train_ns3 --config configs/ns3.yaml --device cuda --net hybrid` |
| 做 A3 vs RL 评估 | `scripts/README.md`（`eval_ns3.py` / `baselines_ns3.py`） | `python -m scripts.eval_ns3 --config configs/eval_normal.yaml --baseline a3 --n-eval 4` |

---

## 8. 快速开始（本地开发）

```bash
cd /home/runner/work/highspeed-rl/highspeed-rl
python -m pip install -e .[dev]

# 本地 mock 冒烟（无需 NS-3）
python -m scripts.train_ns3 --config configs/default.yaml --mock --net hybrid --n-episodes 3
```

---

## 9. 测试命令

```bash
cd /home/runner/work/highspeed-rl/highspeed-rl
python -m pytest tests/ -q
```

可按需先跑子集：

```bash
python -m pytest tests/test_config_validation.py -q
python -m pytest tests/test_ns3_env.py -q
```

---

## 10. 云端 DCU / 容器运行规范（重要）

请以 `CLAUDE.md` 为准；这里仅列最关键约束：

1. **训练/仿真必须在 DCU 容器内执行**，不要在登录节点直接跑 NS-3。
2. 容器重建后先执行：`scripts/cloud/bootstrap_container.sh`。
3. 长任务使用 `setsid` + 持久日志，避免会话中断导致任务丢失。
4. 关注 DCU 利用率；低利用率通常是采集瓶颈，不是网络规模不够。

公开版占位符说明见：`docs/PUBLIC_RELEASE_NOTES.md`。

---

## 11. 评估方式（建议固定流程）

1. 同 seed 配对评估 A3 与 RL
2. 统一配置、统一场景族（normal / tunnel / load / congested 等）
3. 输出至少包含：吞吐、时延、outage/RLF、ho、ping-pong
4. 结果归档到 `reports/`，并标注对应 checkpoint 与配置

常用命令：

```bash
# A3 基线
python -m scripts.eval_ns3 --config configs/eval_tunnel.yaml --baseline a3 --n-eval 4 --seed 900000

# RL（使用 best.pt，避免 latest.pt 漂移）
python -m scripts.eval_ns3 --config configs/eval_tunnel.yaml --net hybrid --resume checkpoints_ns3/best.pt --n-eval 4 --seed 900000

# 成组对比
python -m scripts.baselines_ns3 --config configs/ns3.yaml --net hybrid --checkpoint checkpoints_ns3/best.pt
```

---

## 12. 常见问题 / 故障排查

| 问题 | 处理建议 |
|---|---|
| NS-3 启动失败或握手失败 | 检查 `configs/*.yaml` 中 `ns3.binary`、`ns3.cwd` 是否有效；先用 `scripts/smoke_ns3.py`/`--mock` 分离问题 |
| 训练几乎不占用 DCU | 先看采集并行度、batch、`learn_steps_per_cycle`，不要先盲目加大网络 |
| 结果与文档指标不一致 | 对齐 seed、场景、配置、checkpoint、验收判据版本 |
| 容器重建后 import 失败 | 先跑 `scripts/cloud/bootstrap_container.sh` 再执行训练 |

---

## 13. 文档索引

- 文档总导航：`docs/README.md`
- 当前建模/训练主记录：`docs/建模与训练完整方案.md`
- 公开版脱敏说明：`docs/PUBLIC_RELEASE_NOTES.md`
- NS-3 差距与下一步：`docs/ns3_next_steps.md`
- 阶段验收与判据：`reports/`

---

## 14. 贡献说明（文档与代码）

- 提交前至少执行与改动相关的最小测试（文档改动至少做路径/命令自检）
- 不要在无必要时修改训练契约、动作语义或 NS-3 协议字段
- 涉及云端流程的改动，需同时更新 `README.md` 与 `CLAUDE.md` 的对应说明
