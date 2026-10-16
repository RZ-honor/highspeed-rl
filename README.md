# High-Speed Rail RL Handover

基于强化学习的高铁网络智能切换决策系统。Dueling-DDQN（MLP / Neural-ODE /
Hybrid ODE+MLP）在 3GPP TR 38.901 RMa 信道下学习切换/双连接决策，对比基线为
3GPP A3 事件切换，目标场景 300–350 km/h。

两套仿真后端共享同一冻结契约（14/28 维观测、7 动作、同奖励）：

- **Python**（`src/envs/handover_env.py`）—— 便宜，用于 BC 与消融
- **NS-3 LENA**（`src/envs/ns3_env.py` + `ns3/hsr-rl.cc`）—— 包级真仿，用于在线微调与验收

训练为两段式：Python 行为克隆（A3 教师）→ NS-3 在线微调（并行采集器）。

## 当前状态

- **生产产物 = `checkpoints_stageH13/best.pt`**（2026-10-07 定稿）：对 A3 全族
  增益 +15.6%~+28.8%、中断黑洞时间 −74%、p95 时延 −8.6%~−25%、RLF=0、
  切换次数与 A3 持平。数据与边界见 [CLAUDE.md](CLAUDE.md)。
- 部署定位：**网络侧**（gNB-CU 的 RRM 模块 / O-RAN Near-RT RIC xApp），
  车载 CPE 仅做标准测量上报。

## 快速开始

```bash
# 环境（本地 Windows）
conda activate <你的环境>
pip install -e .

# 本地 mock 冒烟（无需 NS-3）
python -m scripts.train_ns3 --config configs/default.yaml --mock --net hybrid --n-episodes 3

# 测试（本地全量 ~2 min；云端容器内 101 项 ~190 s）
pytest tests/ -q

# 云端（海光 DCU 容器）训练 / 评估 / 验收：见 CLAUDE.md「Commands」
```

## 目录

| 路径 | 内容 |
|---|---|
| `src/` | 环境（RMa 信道、移动性、双后端 env）、智能体、安全层、训练 |
| `ns3/` | LENA 场景 C++（`hsr-rl.cc` / `hsr-common.h`，56→72→84 B 协议三版） |
| `scripts/` | 训练 / BC / 评估 / 基线 / 配对矩阵 / 路径探针 / 容量探针 |
| `configs/` | 各阶段训练与评估场景（H13 为当前最优配置） |
| `session_logs/` | 云端运维工具（投递、守卫、验收报告、阶段配置生成器） |
| `archive/` | 已退役的调试脚本与历史派发记录（cmd_0001–0195） |
| `docs/` | 设计文档与测评体系（`建模与训练完整方案.md` 为设计基线） |

> 公开版脱敏与复现说明见 [docs/PUBLIC_RELEASE_NOTES.md](docs/PUBLIC_RELEASE_NOTES.md)；各阶段预注册判据与验收终判在 [reports/](reports/)。
