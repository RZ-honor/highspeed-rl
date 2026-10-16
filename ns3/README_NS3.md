# NS-3 交互环境（Stage A/B/C）

用真实 NS-3 LTE LENA 仿真替换 Python 信道模型，RL 智能体不变。
设计文档：[docs/superpowers/specs/2026-08-16-ns3-env-design.md](../docs/superpowers/specs/2026-08-16-ns3-env-design.md)

| 阶段 | 内容 | 状态 |
|---|---|---|
| Stage A | 独立 LENA 场景自检（无 IPC） | 已并入 `hsr-rl.cc --test` |
| Stage B | 锁步 IPC（stdin/stdout 二进制协议）+ `NS3HandoverEnv` | ✅ |
| Stage C | 云端训练 + 评估 | 交付待云端运行 |

## 架构

```
Python (RL, T4)                       NS-3 (LENA, C++)
   │  spawn: hsr-rl 二进制               │
   │  ◄── magic "HSR4" ──────────────   │
   │  每 100ms:                          │
   │  ◄── HsrState (99B packed) ─────   │  测量上报+FlowMonitor 聚合
   │  ── HsrAction (8B packed) ─────►   │  HandoverRequest 应用
   │  ... (锁步，episode 结束关 stdin)   │
```

- **传输**：子进程 stdin/stdout（非命名管道——规避 mkfifo 竞态，天然跨平台，
  mock 可在 Windows 本机跑）。二进制协议见 `hsr-common.h` / `src/envs/ns3_env.py`。
- **切换控制**：`NoOpHandoverAlgorithm` 关闭自动切换，动作 1/2/5 触发真实
  `LteHelper::HandoverRequest`；200ms 保护间隔防乒乓轰炸。A3 兜底在 Python
  侧 `NS3HandoverEnv.step` 内部（与 Python 环境完全一致）。
- **测量**：RSRP 来自 eNB 侧 `LteEnbRrc::RecvMeasurementReport`（周期 120ms
  上报，缓存最近值）；吞吐来自 FlowMonitor 每间隔 rxBytes 增量——**切换中断
  是真实仿真出来的**，不再是人工 g(τ) 折算。

## 云端构建与运行（T4, /workspace/ns3, ns-3.43）

### 1. 上传代码

```bash
# 仓库上传到云端（git clone 或 scp），假设位于 /workspace/highspeed-rl
cp /workspace/highspeed-rl/ns3/hsr-common.h /workspace/highspeed-rl/ns3/hsr-rl.cc /workspace/ns3/scratch/
```

### 2. 构建（release 快数倍，推荐）

```bash
cd /workspace/ns3
./ns3 configure -d release     # 若已是 release 配置可跳过
./ns3 build hsr-rl
# 产物: build/scratch/ns3-dev-hsr-rl/release/hsr-rl
```

### 3. 场景自检（无 IPC，验证仿真与切换机制）

```bash
./ns3 run "hsr-rl --test --nEnb=8 --spacing=1000 --offset=50 --dur=20"
```

预期输出（每 100ms 一行）：
```
0.3s serv=1 rsrp=-75dBm nbr[2:-88dBm ...] dl=0.00Mbps ho=0 rlf=0
...  # 经过 eNB 时 A3 自检策略切换，ho 递增
=== DONE: steps=197 totalHO=7 ===
```

验收要点：能编译运行不崩溃；`rsrp` 在 -70~-120 dBm 量级；随 UE 前进
`serv` 变化、`ho` 递增；`dl` 非零。

### 4. RL 训练（Stage C）

```bash
cd /workspace/highspeed-rl
python -m scripts.train_ns3 --config configs/ns3.yaml --device cuda --net hybrid
# 网络可选 mlp / ode / hybrid；续训加 --resume checkpoints_ns3/latest.pt
```

`configs/ns3.yaml` 中 `ns3.binary` 指向 hsr-rl 产物路径（默认 release 构建，
若用 debug 构建请改路径）。每 10 集存 checkpoint，训练结束自动跑 5 集 eps=0
评估并打印吞吐/切换/乒乓/outage。

### 4b. 基线对照（衡量真实增益的参照系）

```bash
# 先跑基线（A3 / 贪心 / 固定间隔，同场景同种子同指标）
python -m scripts.baselines_ns3 --config configs/ns3.yaml

# 训练完成后带 RL checkpoint 一起对比，输出增益表
python -m scripts.baselines_ns3 --config configs/ns3.yaml \
    --checkpoint checkpoints_ns3/latest.pt --net hybrid
```

输出 RL vs A3(3GPP) 的吞吐/切换/乒乓/outage/回报增益百分比。
另有第二参照：`./ns3 run "hsr-rl --test"` 是 LENA 侧内置 A3 策略的原生
结果（非同口径，仅作 sanity 参照）。

### 5. 常见问题

| 症状 | 原因/处理 |
|---|---|
| 握手失败 `magic=...` | binary 路径不对（检查 `ns3.binary`），或非 hsr-rl 程序 |
| `hsr-rl 提前退出` | 看 env 抛出的 stderr 尾部；先用 `--test` 模式复现 |
| 训练每集很慢 | 用 release 构建；降低 `--cbrRate`（默认 100Mb/s 饱和流量） |
| 超时被 kill | `ns3.timeout_s`（默认 600s/集）调大 |

## 本地无 NS-3 的协议测试（Windows 可跑）

```bash
python -m pytest tests/test_ns3_env.py -v          # 协议/环境逻辑（用 mock）
python -m scripts.train_ns3 --mock --net hybrid    # 训练全流程冒烟（mock 后端）
```

## 已知近似（与 Python 环境的偏离）

1. **不支持双/三连接**：LENA 无 EN-DC，真实后端合法动作仅为 0/1/2/5；
   动作 3/4/6 被掩码，不增加估算吞吐。
   HSR4 的 `ho_interrupt_ms` 表示 RRC Start→EndOk 时长，不是业务 blackout；
   `no_rx_interval_ms` / `no_rx_run_ms` 是零收包区间统计，不是包级恢复间隔。
2. **SINR 为估计值**：由上报 RSRP 按与 Python 信道相同公式计算，非物理层实测。
3. **RSRP 趋势为后向差分**（最近 1s），Python 环境为前向预测——语义近似。
4. **拓扑固定间距**：每 episode 通过 `--rngRun` 变化阴影衰落，间距/偏移固定。

## 文件清单

| 文件 | 说明 |
|---|---|
| `ns3/hsr-common.h` | 二进制协议（C++ 侧，99B 状态 + 8B 动作，static_assert 校验） |
| `ns3/hsr-rl.cc` | NS-3 锁步仿真（含 `--test` 自检模式），替代旧 hsr-simulation.cc |
| `src/envs/ns3_env.py` | `NS3HandoverEnv`（obs/reward/info 契约对齐 Python 环境） |
| `scripts/train_ns3.py` | 训练入口（--net mlp/ode/hybrid，--resume 续训，--mock 冒烟） |
| `scripts/baselines_ns3.py` | 基线对照（A3/贪心/固定间隔 + 可选 RL checkpoint 增益表） |
| `scripts/ns3_mock.py` | hsr-rl 协议级 mock（本地测试用） |
| `tests/test_ns3_env.py` | 协议与环境逻辑测试 |
| `configs/ns3.yaml` | 云端训练配置 |

> 注：旧 `hsr-simulation.cc` 的 UE 侧 `RecvMeasurementReport` trace 路径在
> ns-3.43 不存在（该 trace 仅在 eNB 侧），且 `MeasResults` 字段自 3.30 起
> 改为 `measResultPCell`/`measResultListEutra`——旧文件无法在 3.43 编译，
> 已由 `hsr-rl.cc` 取代。
