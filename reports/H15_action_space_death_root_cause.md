# H15 实际切换失效聚焦因果评估（2026-10-14）

## 结论

H15 load16 候选的两次新增死亡（503000@196、505000@190）与**配置化 mask 放开 action5**存在最强、最直接的近因链：同一 H15 checkpoint 的记录中，死亡前 action5 被 Q 值选中并实际 dispatch；目标约 -89/-90 dBm，较当前服务小区差约 10/9 dB，而候选记录的 `masked_actions` 只有静态非法动作 `[3,4,6]`。按 `src/agents/safety.py:40-56,84-105` 的 legacy 规则，`max_drop_vs_serving_db=6` 会屏蔽这些 action5。故“移除动态弱目标 mask 释放劣质 action5”是**确认的配置差异**、**较强的死亡近因**，但因远端 paired legacy run 不可达，不能把“legacy 必然存活”写成已完成的反事实事实。

环境层另有实质性测量语义风险：native `RecvMeasurementReport`（`ns3/hsr-rl.cc:350-378`）写入服务小区/RSRP/邻区缓存；`HoEndOk`（`420-462`）只记切换事件、计数和时间，不刷新该缓存。Python 侧服务估计以缓存值计算，并以 RSRP>-190 判定有效（`src/envs/ns3_env.py:422-445,494-562,711-723,915-936`）。因此切换完成后的旧测量可能继续参与三次低 SINR 终止。native `rlf=0` 或 `HoEndOk` 不能证明切换后的实时无线质量。此项是**较强环境混杂/放大器**，不是仅凭当前工件可单独证明的唯一死亡原因。

## 实际离线结果

候选 load16 16 集中 14 集 survived，死亡为：

| seed | 终止步 | 终止 | 死亡前关键动作 | 目标/服务 | 记录 Q 证据 |
|---|---:|---|---|---|---|
| 503000 | 196 | low_sinr | action5@191–194；action1@195–196 | action5 时 -89 vs -79 dBm，差10 dB | step191 Q=[-54.04,-55.60,-53.11,...]，action5 为合法动作中最大；step192同样 action5 最大 |
| 505000 | 190 | low_sinr | action5@187–188；action1@189–190 | action5 时 -90 vs -81 dBm，差9 dB | step187 Q=[-45.03,-46.76,-45.52,...]，action5 为合法动作中最大 |

两条轨迹均记录 `masked_actions=[3,4,6]`，因此 action5 未被配置 mask 拦截。动作5 是真实 NS-3 handover（`ns3/hsr-rl.cc:513-555`），请求结果随后出现 accepted/HO completion 或保护间隔拒绝；死亡随后以 Python `low_sinr` 结束，而非 native RLF。该事实证明“危险 action5 已被释放并进入环境”，不证明每次 HO completion 的物理结果本身可靠。

历史 paired matrix 也显示 checkpoint/配置差异必须分开：H15A 的 RL 在 tunnel 与 congested 各有 2/3 死亡，而 H13B 同配置对应族均存活；normal/load H15A 存活。但该矩阵不是同一 H15 checkpoint 的 legacy-vs-configured 对照，不能替代指定的两 seed 反事实。

## 因果证据矩阵

| 环节/假设 | 结论 | 证据与边界 |
|---|---|---|
| configured mask 移除动态弱目标保护 | **确认** | `safety.py:40-56`：configured 且 guard disabled 时不启用 SafetyLayer；`ns3_env.py:627-637,689-702`：环境弱目标保护同样依赖 `_rules_active()` |
| 放开的 action5 是两次新增死亡近因 | **较强** | 两条实际轨迹的 Q 选中、目标差9–10 dB、action5 连续出现，随后 low_sinr；缺 paired legacy 生存结果 |
| 仅 removal mask 已证明必死 | **待证** | 原 tunnel 历史显示 mask removal alone 不救 tunnel；load 两 seed 的 legacy run 因 SSH 不可达未完成 |
| native HO completion 可作为新服务无线测量 | **排除** | `HoEndOk:420-462` 不更新 `g_servingCell/g_servingRsrpRaw/g_nbr`；更新只在 `RecvMeasurementReport:350-378` |
| 旧缓存可能误触发低 SINR | **较强** | 缓存无 measurement epoch/freshness；低 SINR 计数 `915-936` 使用有效缓存估计；但当前工件未能单独量化旧值贡献 |
| 奖励权重直接造成该动作 | **待证** | 候选保存了 episode 总 reward，但 probe 没有逐步 `reward_components`；不能由权重推断动作偏好 |
| Double-Q/n-step/PER 算术错误 | **排除（当前证据）** | `agent.py` 与 `replay.py` 审计未见当前 arithmetic/terminal-discount 错误；另有 legacy 动态 next-mask 未进入 bootstrap 的语义不一致，但不能定位本次两死 |
| BC 标签代表成功 HO completion | **排除** | `pretrain_bc.py:64-91` 存 `info.action_executed`/dispatch action，不是 completion-qualified label；教师仍受 env 兜底和请求拒绝影响 |
| A3 是独立反事实干预 | **排除** | death probe 的 A3 是轨迹上动作建议；collector/pretrain 使用 A3Operator，不能把 A3 last30 解释为重新执行后的结果 |
| normal-only best/训练覆盖导致 tail 脆弱 | **待证** | 训练历史显示 normal curriculum/best 选择与多族验收后置；未取得 replay 家族/危险 action5 覆盖计数，不能升级为确认 |

## 环境、奖励、训练与网络判断

### 环境/native
真实 action 映射为 1/2/5，分别选择排序邻区 0/1/2（`hsr-rl.cc:509-555`）。邻区排序按 RSRP 降序；同值 comparator 没有显式 cell-ID tie-break（`350-376`），这是可复现性风险，但当前死亡轨迹没有证据表明排序身份错位。主要已证实风险是 HO completion 不刷新缓存、Python 低 SINR 计数缺乏 epoch/serving-cell 新鲜度约束。

### 奖励
现有 probe 保存 `tp` 和 episode reward，但不保存逐步 reward_components、EMA/baseline、HO interruption 完整字段，因此无法诚实地给出死亡前每一步 throughput/coverage/gap/quality/outage 的精确分解。不能用 `.3`、`.15` 等权重代替实测归因。RLF 与 low_sinr 还应区分：两例终止均为 low_sinr，native rlf=0。

### DDQN/数据分布
Double-Q 选择/评估分离、n-step discount 与 PER 索引逻辑当前审计未显示基础 bug。已知风险是训练 bootstrap 只使用静态合法动作，无法使用需要 next-state 的动态 legacy mask；这能造成部署语义不一致，但目前没有足够数据将其定为 503000/505000 的直接根因。训练和 BC 采用 episode-level mixed policy；A3 标签是实际 dispatch action，仍不能保证成功完成。normal-only best 与 teacher-heavy 分布可能造成弱 action5 tail coverage 不足，当前应标待证。

### 网络 Q/观测/mask
死亡前记录足以证明网络不是“随机偶发”：action5 在候选合法集合中通常是 Q 最大动作；但由于未保存完整 28 维 obs、网络版本的逐层前向输入和逐步奖励，不能做不存在的数据重建。记录的 q 与 mask 足以完成本次**动作释放**判断，不能证明 Q 的训练偏好来自哪一奖励项。

## 首要根因排序

1. **首要：配置化 mask 与 guard disabled 的安全语义差异**，释放 action5 弱目标切换；这是最应优先隔离的新增死亡近因。
2. **次要但重要：native 测量缓存的 completion/freshness 语义**，可能使切换后旧服务测量继续触发 low_sinr，削弱“HO 已完成/rlf=0”的解释力。
3. **待证：训练分布/normal-only checkpoint 选择对弱目标 action5 的尾部覆盖不足**。目前不应先改 DDQN、PER、n-step 或 BC 主流程。

## 最小必要验证（未执行修复）

仅需同一 H15 checkpoint、同 seed 503000 和 505000、同 backend/config，运行 legacy mask 与 configured 各一组，并比较：首次 action5 是否被 mask、HO 请求/完成、low_sinr 步、最终 survival。该对照即可回答“mask removal 是否造成新增死亡”，无需全矩阵。随后如仍需环境归因，只记录 completion 后首个新测量 epoch/serving cell 与缓存更新时间，比较 low counter 是否在新测量前继续累加。

本续接期间按约束重试远端连接仍在 SSH 握手阶段被 reset，未执行 kubectl、仿真或两 seed 对照；因此以上 remote control 结果明确为 unavailable，不冒充通过或失败结论。

## 首选修复位置与无需改动项

研究结论对应的首选修复位置是**策略/环境安全门与测量 freshness 语义的隔离层**：先恢复可验证的弱目标保护语义，再单独校正 completion 后测量缓存资格；不要先改 checkpoint、奖励权重、Double-Q、PER、n-step 或 BC 标签流程。本文不执行任何实现修复、不重训、不覆盖生产 H13 工件。

机器可读证据见 `focused_root_cause_20261014.json`。
