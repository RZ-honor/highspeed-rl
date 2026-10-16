# NS-3 真机仿真结果分析与下一步优化路线

> 数据来源：Stage B/C，ns-3.43 LENA，`hsr-rl.cc`（NoOp 自动切换，RL 经 FIFO 锁步驱动）。
> RL = DuelingDDQN(Hybrid)，A3 = 内置 `A3RsrpHandoverAlgorithm` 基线。

## 1. 首轮结果（修复前的射频默认值）

| 指标 | RL (Hybrid) | A3 基线 | 增益 |
|---|---|---|---|
| reward | 374.08 (±7.01) | 337.65 (±0.77) | **+10.8%** |
| throughput | 8.34 Mbps (±0.22) | 7.86 Mbps (±0.01) | **+6.1%** |
| HO 次数 | 2.4 (±0.49) | 3.0 (±0.0) | **-20%** |
| pingpong | 0 | 0 | 持平 |
| outage | 0 | 0 | 持平 |

**相对增益可信**（RL vs A3 同一仿真器），方向与 Python 仿真一致：RL 用更少切换换更高吞吐。

## 2. 两个必须先修的问题

### 2.1 绝对吞吐错误（已修）

LENA 默认 `DlBandwidth=25RB(5MHz)`、`TxPower=30dBm`、`NF=9dB`、`EARFCN=100(2.1GHz)`，
与 spec 的 20MHz/43dBm/7dB/3.5GHz 不符 → 吞吐被压到 ~8 Mbps。
`hsr-rl.cc` 已加 `--dlRb/--txPowerDbm/--ueNoiseFigure/--earfcn`，默认对齐 20MHz/43dBm/7dB。
重跑后单用户 PF 调度预期 ~40-70 Mbps（CBR 100Mbps 不限）。
3.5GHz(n78) 需 LENA 频段表补丁（LENA 最高 band40/2.3GHz，无 band42）；
当前 `--earfcn 100` 用 band1 2.1GHz，相对对比仍成立。

### 2.2 场景无区分度（pingpong/outage 全 0）→ 隧道 v2 已修（2026-09-16）

8 eNB、spacing=1000m、50s、83.3m/s → 全程仅穿越 ~4 站，覆盖裕量大、
无隧道/无覆盖边缘，**A3 本身就不犯错**。Python 仿真里 ODE/Hybrid 的核心优势
（乒乓 -66%、过训练稳定）在这种"太简单"的场景中无法显现。
→ 必须引入压力场景才能评估决策增益。

**隧道 v2（Task 1.1–1.3，2026-09-16 云端 `--test` 自检）**：
v1 对所有链路叠 25 dB 且洞内无基站 → 进洞必 RLF。v2 改为链路感知
`TunnelLossModel`（宏站穿洞 25 dB，洞内微站仅耦合 3 dB）+ 洞内微 eNB
（spacing=250 m / Tx=30 dBm）。云端二进制 `ns3.43-hsr-rl` 238240 B，
2026-09-16 18:12，g++-10 全量重编。

`--test --dur=52 --tunnels=2000:2800,5000:5600 --tunnelLossDb=25`：

| | spacing=250（覆盖） | spacing=0（对照） |
|---|---|---|
| steps / rlf | 516 / **0** | 516 / **0** |
| 洞内 RSRP min/mean | **-87 / -75.5** dBm | -87 / -72.4 dBm |
| 洞内 serving cell | {3, **9, 10, 11**} | {3} 仅宏站 |
| HO 次数 | 8（t=26.0 切入 cell 9） | 4 |
| 洞内 dl mean / zero_frac | 31 Mbps / 0.479 | （宏站维持） |

验收：洞内 RSRP > -100 dBm、rlf=0、A3 切入微站（cell≥9）——**满足**。
对照 `--tunnelEnbSpacing=0` **未**复现 v1 的洞内 RSRP 骤降/RLF（链路感知
TunnelLossModel 仍给 serving-macro 可用 RSRP）；真正对照是「有无微站 HO」。
正式评估走 Python `A3OperatorBaseline`（TTT 320 ms）+ 16 背景 UE，与
`--test` 内置 A3 不是同一策略，见 Task 1.6。

**Task 1.6 正式评估（2026-09-16）未过关；RF 三档已耗尽：**

| 拍 | 配置 | A3 outage | A3 steps | 备注 |
|---|---|---|---|---|
| cmd_0064 | Tx=30 / spacing=250 / ml=3 | 1.0（ep1-2） | 258 | ep3 `NS_FATAL` IRA |
| cmd_0067 | Tx=**33** + HO 时序修复 | **1.0（4/4）** | 256 | 无 abort；RL 同样 4/4 RLF |
| cmd_0068 | spacing **250→200** | **1.0（4/4）** | 254 | RL steps=274 ho=4，仍 RLF |
| cmd_0069 | microLoss **3→0** | **1.0（4/4）** | 254 | A3 与 0068 **逐集数字相同**；旋钮对 A3 零效果 |
| cmd_0070 | `--test` nbr（nBgUe=0 vs 16） | —（非正式评估） | — | 口部 **含** 微站 cell 9/10/11；`--test` rlf=0，25.5s HO→cell 9 |

cmd_0067 修了 bg UE `HandoverRequest` 乐观改 `servingCell`（`NS_FATAL` 消失），
但正式 Python A3/RL 仍在 ~25.4 s（第一隧道口 x=2000 m）RLF。microLoss 3→0
未改变 A3 四集 reward/steps。**禁止**调低 `tunnel_loss_db` / 缩短隧道。
计划 RF 三档 + 测量上报诊断 **已全部走完**，1.6 **仍未验收**。

**cmd_0070（2026-09-16 21:37 CST，`NBRTEST_EXIT=0`）**：当前旋钮
Tx=33 / spacing=200 / microLoss=0 / loss=25，`--test --dur=36` 对比
nBgUe=0 vs 16。两组解析几乎相同：

| | nBgUe=0 | nBgUe=16 |
|---|---|---|
| parsed_lines / t_range | 356 / 0.3–35.8 s | 同左 |
| first_rlf | **NONE** | **NONE** |
| first_serv_micro | t=25.5 s serv=**9** rsrp=-58 | 同左 |
| last_ho / last_serv | 8 / 4 | 同左 |
| mouth_has_micro_nbr | **1**（hits=59；cell 9/10/11） | 同左 |
| mouth_serv_hist | {3:31, 9:16, 10:4} | 同左 |

口部时序（nBgUe=16，nBgUe=0 同切换时刻）：

```
23.0s serv=3 rsrp=-69 nbr[9:-83] tunnel=0 ho=2
24.1s serv=3 rsrp=-65 nbr[9:-80] tunnel=1 ho=2   # 进第一隧道
24.3s serv=3 rsrp=-62 nbr[9:-77] dl=0.00
25.3s serv=3 rsrp=-66 nbr[9:-62]                 # 微站首次 +4 dB，A3 进入
25.4s serv=3 rsrp=-66 nbr[9:-62] ho=3            # testMode 立即 HO（TTT=0）
25.5s serv=9 rsrp=-58 nbr[3:-67]                 # 已在微站
```

**测量上报假设已否定**：16 背景 UE **没有**把微站 cellId 挤出 nbr 列表，
也 **没有**导致 `--test` RLF。`--test` 内置 A3（offset 3 dB、**无 TTT / 无 L3**）
在 25.4 s 切到 cell 9 并活过 35.8 s。正式评估走 Python
`A3OperatorBaseline`（L3 α=0.4 + TTT **320 ms** + 16 bg），cmd_0069 四集
均 steps=254 / ho=2 / outage=1.0——ho=2 全是进洞前宏站切换，**从未发出微站 HO**。
口部微站从 t=23.0 就可听但弱于 serving，直到 25.3 才过 +3 dB；testMode 立刻切，
正式 A3 还要再等 320 ms TTT（加上 L3 滞后），窗口不够，在 25.4 s 死掉。

根因是 **策略层 TTT/L3 vs testMode TTT=0**，不是 RF、不是测量漏报。
用户已授权改策略层（2026-09-16）：**仅** `eval_tunnel.yaml` 设 `ttt_ms=40` /
`l3_alpha=1.0` / `raw_enter=true`，并加吞吐塌陷兜底 `fallback_tp_collapse_mbps=0.1`。
其它 eval_*.yaml 仍 320/0.4。cmd_0071 正式 A3+RL 评估在飞
（`logs/tunnel_v2_policy.log`）。RF 旋钮不再拧，不得启 Stage2，等 0071 EXIT。

Task 2.2 / 2.4 见 §4.3 / §4.4（截断样本；无完整 480 步 episode）。

## 3. 环境建模优化（按优先级）

1. **压力场景参数化**（高优先；**隧道项已修，2026-09-16**）：
   - `--spacing 300-800m`（密集站，切换频次 ×2-3，乒乓窗口增多）
   - `--speed` 提到 350-400 km/h（Doppler/测量滞后更严重）
   - ~~加隧道段（额外穿透 25dB）制造覆盖势阱~~ → **v2 已落地**：
     链路感知 TunnelLossModel + 洞内微站（`--tunnelEnbSpacing=250
     --tunnelEnbTxDbm=30 --tunnelMicroLossDb=3`）。`--test` 洞内 RSRP
     min -87 dBm、rlf=0、A3 HO 切入 cell 9/10/11。详见 §2.2。
   - 拉大穿透损耗/降低 eNB 密度制造覆盖边缘
   目标：让 A3 基线出现非零 pingpong/outage，RL 才有优化空间。

2. **双连通动作的真实性**（结构性，中优先）：
   a3/a4/a6（dual/triple）在 LTE LENA 里是**空操作**——LENA 无 EN-DC，
   多小区容量求和只是奖励近似。3/7 动作在 NS-3 里无真实数据面增益。
   两条路：(a) NS-3 后端掩码这 3 个动作，只评 a0/a1/a2/a5；
   (b) 迁移 5G-LENA(nr) 获得真实 MC-DC/双连接。建议先 (a) 保证评估诚实，
   把 (b) 列为后续阶段。

3. **车体穿透损耗**（中优先）：LENA 测量不含车体损耗。当前若未在 reward/
   测量侧注入 28dB/cosθ，覆盖裕量被高估。建议在 RSRP 上报侧统一扣穿透损耗，
   与 Python `channel_3gpp.py` 一致。

4. **切换中断 τ**：LENA 原生建模 HO 中断（RLC 缓存/数据面停顿），
   比 Python 的 g(τ) 近似更真实——Python 侧的 τ 模型应以 NS-3 实测中断时长标定。

## 4. Agent 决策优化（按优先级）

1. **Python 预训练 → NS-3 微调（sim2sim 迁移，最高价值）**：
   NS-3 单 episode 墙钟时间远高于 Python 信道。obs 契约（14 维）与 info 键完全一致，
   可直接加载 Python 训练的 checkpoint 作为 NS-3 训练的初始权重，
   在 NS-3 上只做少量 episode 微调。这把昂贵的 NS-3 交互从"探索"降级为"校准"。
   落地：`scripts/train.py --backend ns3 --init-checkpoint checkpoints/hybrid_latest.pt`。

2. **Hybrid 路由 α 在难场景的验证**：当前 α 统计来自简单场景。
   在隧道/密集站场景下采集 α 分布，验证"平稳区间 α 低（ODE 主导）、
   切换/隧道口 α 高（MLP 主导）"的路由假设是否在真实物理层成立。

3. **安全层与 NS-3 内置 A3 的去重**（Task 2.2，见 §4.3）：NS-3 已是 NoOp
   （全由 RL 驱动），Python 侧 SafetyLayer 的 A3 兜底经 FIFO 下发。截断
   样本上 src 分布已核对；完整 480 步 episode 尚未出现。

4. **奖励权重用 NS-3 实测重标定**（Task 2.4，见 §4.4）：吞吐/中断/切换的
   相对尺度在真实数据面下与 Python Shannon 近似不同。当前隧道评估全部
   在第一洞口 RLF 截断，量级表数据不足，**不改 w_***。

### 4.3 安全层去重（Task 2.2，截断样本已核对）

数据：cmd_0069 正式评估（Tx=33 / spacing=200 / microLoss=0，4 ep，
均在 steps≈254–274 第一隧道口 RLF；**非**完整 480 步）。`eval_ns3.py`
已打 `src_count`。NS-3 侧 `hsr-rl.cc` testMode A3 仅 `--test` 生效，
正式评估走 Python 动作，不存在 C++/Python 双触发。

| 来源 | A3-as-policy | RL (hybrid) | 判据 |
|---|---|---|---|
| `rl` | 100% | 23.6% | — |
| `safety_a3` | 0% | **0.0%** | < 5% → 达标（但见下） |
| `masked_dual` | 0% | **76.0%** | > 0 → dual 掩码生效 |
| `masked_weak` | 0% | 0.4% | 无激增 |

α（同批 RL）：mean=0.9813，HO 步=0.8861，非 HO 步=0.9828。

**已验证（截断口径）**：`masked_dual` 生效；`masked_weak` 正常；
NoOp 无双触发。`safety_a3=0%` 满足「< 5% / RL 主导」字面判据，但结合
cmd_0070 口部时序，这同时说明兜底 **没在洞口开火**：serving 宏站 RSRP
仍约 -62..-66 dBm，`_sinr_all` 估计 SINR 未连续 3 步低于 -5 dB，且
`_a3_action` 要邻区 > serving+3 dB——微站直到 25.3 s 才满足，正式评估
25.4 s 已死。safety_a3=0 是「没帮上忙」而非「策略很稳」。完整 episode
出现前不把 2.2 标成最终验收。

### 4.4 奖励尺度（Task 2.4，数据不足）

cmd_0069 四集全部在 ~25.4 s / steps=254（A3）或 274（RL）被 RLF 终止，
剩余步数惩罚主导 return，**没有**完整 480 步量级可拆：

- 吞吐项 `log1p(cap)`：截断均值 tp≈20.9（A3）/ 22.7（RL）Mbps，不能代表洞内微站段
- 切换项：A3 ho=2（仅进洞前宏站）；`--test` 同场景 ho=8（含微站）——正式评估没切到微站，HO 项失真
- 中断项：未活过洞口 HO，`hoInterruptMs` 分布缺微站切入样本
- 稳定项：dual 被掩码，恒 links=1，该项恒 0（已知诚实近似）

判据「量级差 >3× 才重归一化」**无法判定**。结论：**不改 w_***，等 1.6
有存活全集再复核。

## 5. 建议的下一轮实验矩阵

| 实验 | 后端 | 目的 |
|---|---|---|
| RF 修复后重跑 baseline vs RL | NS-3, 20MHz/43dBm | 拿到真实吞吐绝对值 |
| 密集站+隧道压力场景 | NS-3, spacing 300-800 + tunnel | 让 A3 出乒乓/中断，显现 RL 增益 |
| Python 预训练 → NS-3 微调 | NS-3, init ckpt | 验证 sim2sim 迁移与样本效率 |
| 掩码 dual 动作 vs 不掩码 | NS-3 | 量化空操作动作对策略的污染 |
| α 分布采集（难场景） | NS-3 | 验证混合路由假设 |
