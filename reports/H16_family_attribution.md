# H15 原失败家族（tunnel/congested）× 修复归因 —— 2026-10-14 晚间追加

## 一句话结论

H15 判负的主证据（tunnel/congested 各死 2/3 seed、268 步 low_sinr、ho=4）
在**同一 checkpoint、同一 seed、同一 HSR4 二进制**下被逐位复现，并被逐项归因：
**该失败模式完全由「HO 完成当步用源小区陈旧测量计入连续低 SINR 终止」造成，
修复3（测量 freshness）单独即可救活；硬盾（fix1）在这两族从未触发（与复现臂
逐位相同）**。修复后五族存活 1.00、Δtput +24%~+34% vs A3。生产替换仍需
H13 同语义参照矩阵 + load16 全 16 seed 收口（见文末状态）。

## 六臂证据表（diag_death_probe，同 env/agent/seed 构造，eps=0，CPU，
seed_base=500000，load→[3,5]，tunnel/congested→[0,2]，n-eps 1）

| 臂 | 语义 | seed | steps | 终止 | R | ho | a5选中 | a5执行 | min SINR(末30) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
| tun_repro | legacy 评估 | 500000 | 268 | low_sinr | −52.87 | 4 | 0 | 0 | −6.28 |
| tun_repro | legacy 评估 | 502000 | 268 | low_sinr | −52.94 | 4 | 0 | 0 | −6.28 |
| tun_shield_only | +盾，无冻结 | 500000 | **268** | low_sinr | −52.87 | 4 | 0 | 0 | −6.28 |
| tun_shield_only | +盾，无冻结 | 502000 | **268** | low_sinr | −52.94 | 4 | 0 | 0 | −6.28 |
| tun_grace_only | 无盾，+冻结 | 500000 | **480** | 存活 | +25.81 | 9 | 0 | 0 | +7.24 |
| tun_grace_only | 无盾，+冻结 | 502000 | **480** | 存活 | +27.42 | 9 | 0 | 0 | +7.24 |
| tun_shield+grace | H16 全量 | 500000 | 480 | 存活 | +25.81 | 9 | 0 | 0 | +7.24 |
| tun_shield+grace | H16 全量 | 502000 | 480 | 存活 | +27.42 | 9 | 0 | 0 | +7.24 |
| cong_repro | legacy 评估 | 500000 | 268 | low_sinr | −52.26 | 4 | 0 | 0 | −6.28 |
| cong_repro | legacy 评估 | 502000 | 268 | low_sinr | −57.68 | 4 | 0 | 0 | −6.28 |
| cong_shield+grace | H16 全量 | 500000 | 480 | 存活 | +24.62 | 9 | 0 | 0 | +7.24 |
| cong_shield+grace | H16 全量 | 502000 | 480 | 存活 | +29.64 | 9 | 0 | 0 | +7.24 |

（R 与 steps 的逐位相同 → 盾在这些轨迹上与复现**完全等价**：a5sel=0，即策略
在这两族从未想切第三邻区，硬盾没有可拦之物；grace_only 与 shield+grace 的
R/steps/ho/minSINR 完全相同 → 救活者就是冻结。）

## 死亡过程的物理还原（seed 500000，tunnel，逐位复现于 2026-10-13 cmd_0203）

```
266-267 服务 SINR=-6.28（连续第 1、2 次低 SINR）
268   RL 选 action1，request_result=1（接受），ho_event=1（完成），
      但当步 serving 上报仍是源小区的 -6.28 → 第 3 次计数 → 终止
269   (冻结后) 仍 -6.28（陈旧）→ 继续冻结
270-271 新小区上报 -3.80 / 14.6Mbps → 计数归零
272   -7.38（第 1 次新低 SINR）
273   再次 HO 完成
274+  SINR +12.95，tp≈21.9Mbps —— 隧道内微站接管成功，一路活到 480 步
```

即：策略的切换决策是**对的**，旧评估器用切换前的测量把这次成功切换判成了死亡。
原报告把这类死亡描述为「该进洞前切微站的时机没抓住 / 少切晚切」——晚间数据
显示在测量语义修正后，同一策略的 ho 从 4 变 9、末 30 步 min SINR 从 −6.28
变 +7.24，「少切」表象随之消失（此前 268 步死在洞里，自然切不满）。

## 与 load 族归因的合并（下午的 cmd_0204）

| 死亡形态 | 出现家族 | 操作性修复 | 单独验证 |
|---|---|---|---|
| ho_to_weaker（切向 −9~−11dB、SINR≈−13 的目标） | load 16 集（503000/505000） | **fix1 硬盾**（a5_exec 8/2 → 0，480 存活） | grace_only 也「存活」但 a5 仍派 111/56 次、ho=16/12、R 为负 → 属伪装，不算修复 |
| 陈旧测量误杀（HO 完成当步计旧值） | tunnel/congested（500000/502000） | **fix3 测量冻结**（268→480，R 转正，min SINR 回正） | shield_only 与复现臂逐位相同 → 与本形态无关 |

两项修复各管一条独立死亡机制，无重叠；**fix2（bootstrap 同可行动作集）在评估
路径上不体现，只能在重训中生效，仍未验证**。

## 五族矩阵（固定新语义 cfg_eval_shield：legacy policy_mask + 硬盾 + 冻结）

H15（checkpoints_stageH15/best.pt，3 seed × 2 集，逐 seed 与 A3 配对）：

| 族 | A3 tp | RL tp | Δ vs A3 | RL 存活 | RL ho |
|---|---:|---:|---:|---:|---:|
| normal | 9.92 | 12.30 | +24.1% | 1.000 | 4.0 |
| tunnel | 9.54 | 11.94 | +25.1% | 1.000 | 8.2 |
| load | 7.98 | 10.41 | +30.4% | 1.000 | 4.0 |
| congested | 6.80 | 9.11 | +33.9% | 1.000 | 8.2 |
| hotcold | 6.40 | 8.43 | +31.7% | 1.000 | 4.0 |

判据④（H13 同语义参照矩阵，2026-10-14 21:11 收齐，tol=0.02·|ΔH13|+0.05）：

| 族 | H13 tp | H15 tp | Δ13 | Δ15 | Δ15−Δ13 | tol | ④判定 |
|---|---:|---:|---:|---:|---:|---:|---|
| normal | 11.47 | 12.30 | +1.55 | +2.38 | +0.83 | 0.08 | PASS |
| tunnel | 11.38 | 11.94 | +1.85 | +2.40 | +0.55 | 0.09 | PASS |
| load | 9.60 | 10.41 | +1.62 | +2.42 | +0.80 | 0.08 | PASS |
| congested | 8.77 | 9.11 | +1.97 | +2.31 | +0.34 | 0.09 | PASS |
| hotcold | 8.17 | 8.43 | +1.77 | +2.03 | +0.26 | 0.09 | PASS |

判据④整体 **PASS**，且五族 H15−A3 全部高于 H13−A3（不只是不劣，是反超）。
新语义下 H13 与 H15 的 30 集矩阵存活都是 1.000。判据①（load16 全 16 seed，
cfg_eval_shield，两片 × 4 seed × 2 集）21:14 起在跑（cmd_0210）。

## 状态与边界

- 本实验全部只读评估：不训练、不改奖励与终止阈值（−6 dB/连续 3 步不变，
  冻结只是不把陈旧读数计入 strike，且封顶 2×grace、HO 失败立即恢复——
  真实劣化照死，单测 `test_stale_measurement_does_not_advance_rlf_but_truthful_one_terminates` 锁定）。
- 归因结论限于所测 seed（每族 2 个代表 seed）；H16 配置（`policy_mask:
  configured` + 硬盾）与 `cfg_eval_shield`（legacy policy_mask + 硬盾）语义
  不同，前者面向重训、后者面向在 legacy 语义上的最小修复评估。
- 证据：`logs/tun_*.json`、`logs/fam_*.json`、`logs/matrix_h15.json`；
  派发：`session_logs/h16_fam_20261014.sh`（cmd_0205）、
  `h16_tunattr_20261014.sh`（cmd_0206）、`h16_matrix_20261014.sh`（cmd_0207）、
  `h16_matrix_h13_20261014.sh`（cmd_0209）。
- 远端隔离目录：`/work/home/<HPC_ACCOUNT>/h16_shield_cf_20261014`；
  pod 换代（旧 `<POD_GROUP_A>-57f5cf8565-9zdfn` 卷回收导致 cmd_0207 的
  h13 臂 worker `os.getcwd()` FileNotFoundError 整臂崩溃，h15 臂已完整落盘；
  h13 臂以 cmd_0209 重投）。
