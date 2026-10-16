# H16 动作空间死亡修复 —— 远端反事实验收报告（2026-10-14）

## 结论

**硬盾（safety.hard_shield，推理/环境/bootstrap 三侧同开关）是两条 H15 死亡的
操作性修复：同一 H15 checkpoint、同一 seed、同一 HSR4 二进制下，shield ON 把
seed 503000/505000 从 196/190 步 low_sinr 死亡救到 480 步全程存活，且控制 seed
（500000/501000）不受损。测量冻结（fix 3）单独开启时也能「救活」这两条集——
但它是靠压制终止探测器救的（a5 仍派发 111/56 次、ho=16/12、episode 回报为负），
不是安全修复；在 shield ON 的轨迹上它零触发（逐位相同）。结论：fix 1+2 是解，
fix 3 只是测量语义补丁，验收价值以「shield 之上无副作用」为限。**
生产产物不变，H13 继续。**本实验是推理侧反事实，不含重训**。

## 实验设置（溯源）

- 隔离副本：远端 `/work/home/<HPC_ACCOUNT>/h16_shield_cf_20261014`（源 =
  `h15_survival_fix_20261012` + `h16_shield_patch_20261014.tgz`）。
- 补丁 SHA256（远端与本地逐位一致，门禁 `GATES_PASS`）：
  - safety.py `c1da89b9…d121ec2`、agent.py `dbee62d7…408578df`、
    ns3_env.py `e9d25da3…e90c8c7c`、diag_death_probe.py `a7903ee9…b62a8d`、
    configs/ns3_stageH16_shieldfix.yaml `d75f6f46…b946693d`
  - HSR4 二进制 `8d42cc7ebadce872a3a0a8b1617ff05770e8e4a1b6c90dae01d0e3862d34b9d7`
    （与生产 HSR4 记录一致）
  - checkpoint `checkpoints_stageH15/best.pt` sha256 `426c8802…c35f7155`
- 运行口径 = paired_matrix/diag_death_probe 同 env+agent+seed 构造（eps=0，CPU，
  seed_base 500000，load 族，`--seed-list`）。

## 五臂矩阵（同 checkpoint、同 seed、同二进制，只换安全语义）

| 臂 | 语义 | seed 503000 | seed 505000 |
|---|---|---|---|
| cf_repro | survival_fix（无 shield、无 grace） | 196 步 low_sinr，ho_to_weaker，R=-80.73 | 190 步 low_sinr，ho_to_weaker，R=-77.00 |
| cf_shield | H16（shield+grace） | **480 存活** R=16.11 ho=6 a5_exec=0 | **480 存活** R=22.21 ho=4 a5_exec=0 |
| cf_shield_only | shield、grace=0 | **480 存活** R=16.11（与 cf_shield 逐位同） | **480 存活** R=22.21（同） |
| cf_grace_only | 无 shield、grace=3 | 480「存活」 R=-19.00 ho=16 a5_exec=111 | 480「存活」 R=-12.26 ho=12 a5_exec=56 |
| cf_shield_ctrl | H16 控制 seed 0/1（原本存活） | 480 R=48.66 ho=4 | 480 R=44.28 ho=4 |

（seed 列按 500000+idx*1000：3→503000、5→505000。）

## 关键分歧点证据（同状态反事实）

seed 503000，step 71 两侧观测逐位相同（obs_identical=True，同权重）：

- Q[0,1,2,5] = [-61.03, -57.52, -60.24, **-49.91**]；候选 [-79,-81,-89,-200]，
  SINR [1.2,-2.5,**-12.2**,-120]。
- repro：无屏蔽 → 选 action5（切 -89/SINR-12 的 nbr2）→ 一路连切 4 步 →
  191-196 步低 SINR 连续终止。
- shield：masked_actions 含 {2,5} → 退选 action1 → 全程未再派发 a5 → 480 步存活。

这直接证实 focused_root_cause 的判断：「action5 在候选合法集合中通常是 Q 最大动作」
——不是随机偶发，是 Q 的真实偏好；盾必须在选择点拦截，而不能指望重标定奖励。

## 归因与边界

1. **操作性修复 = 硬盾**。cf_shield 与 cf_shield_only 逐位相同 → grace 在该轨迹
   上从未触发；cf_grace_only 虽不终止，但保留了致命派发（111/56 次 a5），
   且低 SINR 步 16/14 全部落在 HO 事件 3 步邻域内（=冻结吞掉了全部 strike）——
   这正是「用测量补丁伪装存活」的形态，不可当性能结论引用。
2. grace 保留在 H16 配置里的理由仅是：native HoEndOk 不刷新 serving 缓存是真实
   环境缺陷（focused_root_cause 判「较强」混杂项），在 shield 之上它无副作用；
   终止标准本身不豁免（run cap=2×grace、HO 失败立即恢复计数，单测覆盖）。
3. **未验证项**：load16 全 16 seed × 五族配对矩阵、H16 重训（bootstrap 一致性
   的效果只能在训练里体现）、native 侧 cache epoch（需改 hsr-rl.cc 重编译）、
   tunnel/congested 两族「少切晚切」型死亡（H15 原判负主因，机制不同，本次
   五臂实验未涉及）。生产继续 H13。
4. 本实验全程只读评估，不训练、不覆盖产物、不改 H13/H15 配置与二进制。

## 文件

- 远端证据包：`h16_cf_batch1.tgz`/`h16_cf_batch2.tgz`（本地已解包：`logs/`、
  `configs/cfg_shield_only.yaml`、`cfg_grace_only.yaml`）。
- 派发脚本：`session_logs/h16_setup_20261014.sh`（门禁）、
  `h16_arm_20261014.sh`（三臂）、`h16_attr_20261014.sh`（归因两臂）。
- 代码修复：`src/agents/safety.py`、`src/agents/agent.py`、`src/envs/ns3_env.py`；
  回归测试 `tests/test_safety.py`、`tests/test_agent.py`、`tests/test_ns3_env.py`
  （全量 pytest 148 passed，本地）。


## 晚间追加（2026-10-14 21:58）：六项预写死判据在「固定新语义」下的重测

同 checkpoint（H15 best.pt，sha 426c8802…）、同 HSR4 二进制（sha 8d42cc7e…）、
同 seed 构造；评估语义统一为 `cfg_eval_shield`（硬盾 + 测量冻结），H13 作同语义参照。
详细归因见同目录 `report_families_20261014.md`。

| 判据 | 结果 | 数据 |
|---|---|---|
| ① load16 主判据 | **PASS** | 16/16 存活，全部 480 步，action5 实际派发 **0** 次，mean R=43.84，mean tp=9.92 |
| ② A3 一致性 | 沿用 | 二进制 sha 与 H15 验收时逐位相同（当年 runA/runB max_abs_diff=0.0）；本轮未重跑双臂一致性 |
| ③ 五族 survival=1.00 | **PASS** | H15 五族 ×3 seed ×2 集 = 30/30 存活（H13 同语义参照亦 30/30） |
| ④ Δtput ≥ H13−TOL | **PASS（反超）** | 五族 H15−A3 全部高于 H13−A3（差 +0.26~+0.83 Mbps，tol≈0.08~0.09）；H15−A3 逐 seed 配对 95%CI 全正且显著 |
| ⑤ ho 不反弹 | **PASS** | 五族 H15 ho 与 H13 完全相等（normal/load/hotcold 4.0；tunnel/congested 8.17） |
| 总判定 | **修复语义下 H15 具备替代候选资格** | 见下方边界 |

关键更正（相对 2026-10-13 判负报告的推断）：tunnel/congested 的「少切晚切」表象是
**陈旧测量把成功切换当步判死**造成的——冻结后同一策略 ho 从 4 回到 8.17、末 30 步
min SINR 从 −6.28 回到 +7.24；硬盾只负责 load 族的 ho_to_weaker（该族 a5sel>0，
tunnel/congested 中 a5sel=0 故盾与复现臂逐位相同）。两条死亡机制各由一项修复关闭。

## 仍未成立的边界（不写进结论就不诚实）

1. 这是**既有 H15 checkpoint 在修复语义下的推理侧重测**。fix2（bootstrap 同
   可行动作集）在评估路径上不可见，其收益只能由 H16 重训验证；H15 的 Q 值
   本身仍是在旧 bootstrap 下学的。
2. 判据口径本身换了尺：陈旧测量误杀修复后，H13/H15 在 30 集矩阵里都是 1.00
   存活——本轮的公平性来自「两臂同尺重测」，而非沿用旧尺的历史数字。
3. 归因限于所测代表 seed（每族 2 个）；load16 已覆盖 16 集，五族矩阵 30 集，
   但 fade 族仍未纳入；②的一致性未重跑。
4. 生产仍为 H13；是否替换需走既定流程（H16 重训 + 全矩阵 + A3 一致性复跑），
   本轮只读评估未改任何生产产物。
