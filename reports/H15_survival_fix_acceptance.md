# H15 survival fix 实际验收 20261014

## 总结
五族配对矩阵：**PASS_MATRIX**，90/90集，候选H15存活30/30。**尚不批准生产替换，继续H13**；完整验收FAIL：load16为14/16完整存活（87.5%），两集候选死亡；native源码构建溯源PASS。
原PID312在2400秒预算内完成74集后exit124。仅补缺失16个hotcold identity，独立timeout3600/workers6，没有重算74集；补集退出码0。

## 阶段与证据
- 本地回归PASS：132 passed/1 warning；独立验收入口3 passed；py_compile、config、480步严格选优PASS。
- 同组替代pod身份PASS：原<POD_GROUP_A>-57f5cf8565-f2h7l NotFound；唯一替代<POD_GROUP_A>-57f5cf8565-76k8d，containerGroupName=<POD_GROUP_A>，pod-template-hash=57f5cf8565。
- 旧三seed rescue虽然480步，但旧版本不算最新PASS。新矩阵三个原死亡seed（tunnel500000/502000、congested500000）救援单项PASS。
- 首次依赖启动FAIL：0集前gymnasium缺失exit1；完整旧日志保留。隔离wheel gymnasium0.29.1/farama_notifications安装.acceptance_deps后修复PASS，不改全局环境。
- A3旧pm_h15A/pm_h13B数字最大差0，但配置/版本证据不足，禁止复用；当前30个共享A3reference，三策略同spawn核验。未另做当前A3重复试验。
- load16实际完成16集，四分片均exit0、汇总exit0；验收FAIL，不将进程成功视为存活PASS。

## 存活、完整性与终止原因
|策略|样本|存活|最差族存活率|终止原因|
|---|---:|---:|---:|---|
|A3|30|30|100.0000%|{"truncated": 30}|
|H13|30|29|83.3333%|{"truncated": 29, "low_sinr": 1}|
|H15|30|30|100.0000%|{"truncated": 30}|
死亡：H13/load/seed501001，444步，low_sinr，final_x=3723.510009765625m。参考死亡不构成候选硬失败，但其短轨迹吞吐不支持优越结论。

480步路线约4km（spawn约25m，final约4023m），仅为该配置完整路线，不声称完成8km全地理线路。

## 吞吐、公平、HO与rescue
均为逐集均值再求族均值。Mbps；HO为完成事件。load参考H13含死亡短轨迹，保留原数值但禁止吞吐优越结论。
|族/策略|throughput|aggregate|cell aggregate|p05|p50|Jain|delay ms|HO均值|HO尝试总数|rescue|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|normal/A3|9.915568|11.390012|9.917337|1.866846|3.447612|0.505942|13.724594|4.000000|48|0|
|normal/H13|9.917350|11.391861|9.919119|1.866861|3.447612|0.505921|13.447042|4.000000|55|0|
|normal/H15|11.922248|11.720062|10.249219|1.913187|3.543930|0.503915|8.758497|4.000000|28|0|
|tunnel/A3|9.539215|10.958888|9.540928|1.774331|3.491073|0.503199|17.169707|8.166667|98|0|
|tunnel/H13|9.671708|11.091351|9.673421|1.785761|3.517383|0.502732|14.824725|8.166667|102|0|
|tunnel/H15|10.734937|11.192280|9.776506|1.812104|3.565620|0.501428|13.378756|8.166667|84|4|
|load/A3|7.981758|34.733427|9.356649|0.055675|0.649285|0.379330|15.859056|4.000000|48|0|
|load/H13|7.993313|34.766413|9.372717|0.055845|0.649677|0.379342|15.261225|4.166667|52|0|
|load/H15|10.132764|35.110444|9.851145|0.055652|0.649483|0.374013|10.976271|4.000000|28|0|
|congested/A3|6.804716|43.594191|8.772835|0.044181|0.491779|0.368061|20.011217|8.166667|98|0|
|congested/H13|7.003398|43.795523|8.971671|0.044162|0.491967|0.366297|17.055596|8.166667|100|0|
|congested/H15|8.323605|43.721600|8.925044|0.044166|0.491456|0.366155|16.760334|8.166667|78|4|
|hotcold/A3|6.402910|41.941792|8.777665|0.048984|0.475671|0.367161|16.788080|4.000000|48|0|
|hotcold/H13|6.524416|42.071764|8.890482|0.048997|0.474633|0.365584|14.059058|4.000000|52|0|
|hotcold/H15|8.429947|42.157104|9.007354|0.049001|0.474948|0.364489|13.533447|4.000000|24|0|

## JSON硬判据
候选每集alive/full480/truncated/nonterminated；exact unique90；30个同spawn配对；各族原delta不劣容差0.02*abs(deltaH13)+0.05；HO_H15<=1.2*HO_H13+0.01。参考死亡另报，不用全策略存活门槛停止候选验收。
- {"name": "exact_unique90", "passed": true}
- {"name": "missing16_command_exit0", "passed": true}
- {"name": "no_worker_errors", "passed": true}
- {"name": "candidate30_alive_full480", "passed": true}
- {"name": "raw_history_consistent", "passed": true}
- {"name": "normal_original_delta_noninferiority", "passed": true, "H15_delta": 2.0066801595145947, "H13_delta": 0.0017825569543585829, "tolerance": 0.05003565113908717, "reference_has_dead_trajectory": false, "superiority_claim_permitted": true}
- {"name": "normal_ho_nonrebound", "passed": true, "H15_HO": 4, "H13_HO": 4}
- {"name": "tunnel_original_delta_noninferiority", "passed": true, "H15_delta": 1.1957228464412317, "H13_delta": 0.1324935519446928, "tolerance": 0.05264987103889386, "reference_has_dead_trajectory": false, "superiority_claim_permitted": true}
- {"name": "tunnel_ho_nonrebound", "passed": true, "H15_HO": 8.166666666666666, "H13_HO": 8.166666666666666}
- {"name": "load_original_delta_noninferiority", "passed": true, "H15_delta": 2.151005774901293, "H13_delta": 0.011555608948309981, "tolerance": 0.050231112178966204, "reference_has_dead_trajectory": true, "superiority_claim_permitted": false}
- {"name": "load_ho_nonrebound", "passed": true, "H15_HO": 4, "H13_HO": 4.166666666666667}
- {"name": "congested_original_delta_noninferiority", "passed": true, "H15_delta": 1.518889225190506, "H13_delta": 0.19868189387230473, "tolerance": 0.0539736378774461, "reference_has_dead_trajectory": false, "superiority_claim_permitted": true}
- {"name": "congested_ho_nonrebound", "passed": true, "H15_HO": 8.166666666666666, "H13_HO": 8.166666666666666}
- {"name": "hotcold_original_delta_noninferiority", "passed": true, "H15_delta": 2.0270364786941633, "H13_delta": 0.12150599908911541, "tolerance": 0.05243011998178231, "reference_has_dead_trajectory": false, "superiority_claim_permitted": true}
- {"name": "hotcold_ho_nonrebound", "passed": true, "H15_HO": 4, "H13_HO": 4}
- {"name": "all30_triplets_same_spawn", "passed": true}

## 救援归因
总override 8 次；完整逐步归因在机器JSON rescue_events与原始history。共享env safety_a3触发，精确deadline/其他fallback并发触发未单独标记，不包装为H15裸网络收益。HO尝试不等于native接受请求。

## 版本与运行边界
所有读取/派发使用<HPC_ACCOUNT>@<HPC_HOST>:65032、原KC/NS、同组POD以及container-0：`kubectl --kubeconfig=$KC -n $NS exec $POD -c container-0 -- bash -lc`。连接有限重试（连续3次失败停止），没任取pod/换账户。仅隔离目录更新修复/依赖，无训练、阈值或reward修改、生产覆盖、git提交或H13替换。
manifest保存配置/seed/二进制与checkpoint SHA256。最新env/safety/agent/train/probe/config匹配本地；旧manifest在错误隔离源码路径记录null；追加溯源找到真实validation与scratch构建副本，源码/header与原记录逐位相同，build.log有编译和链接记录，实际binary hash也完全匹配；未重编译。
- configs/ns3_stageH15_survival_fix.yaml: bb318742b16acbbf6cf897eaef21ecdbcccea444da5e39962eab9fe2f9e603b8
- checkpoints_stageH15/best.pt: 426c88029b8faa2c17cd0dceef7d7269d9f40f46fe6834202b2f0950c35f7155
- checkpoints_stageH13/best.pt: a4b8f44d2762b80cc20700218e90e0cdf5e4a7f4d6f889cdb9ff418a4afb24b2
- /work/home/<HPC_ACCOUNT>/ns3root/ns-3.43/build/scratch/hsr4_validation_20261012/ns3.43-hsr4-smoke: 8d42cc7ebadce872a3a0a8b1617ff05770e8e4a1b6c90dae01d0e3862d34b9d7
- /work/home/<HPC_ACCOUNT>/h15_survival_fix_20261012/session_logs/h15_survival_fix/acceptance_runner.py: b4c38c633052c1af5b1d17f4c58edb35aa6db117766fec5263bd6da9baa13f39
- src/agents/safety.py: 6e5c72bfb607be1a856ca2b6189b2d799f6ac4d9917e8fdaec91d121b8ee9d18
- src/agents/agent.py: 7845e6762418e89ca345fab4fd7ffbe10a3b92110e4cdd79bc0732a626c8ecc1
- src/envs/ns3_env.py: 2ec25a9fea3b4b2bc71c4d0ac674197385c9991488ba9221a9a2eede3b5043d1
- scripts/train_ns3.py: 22b4e711cad4bb073c5e3c53910ead7cd1598ef2b3be097e262411cf10d56f17
- session_logs/diag_death_probe.py: a7903ee933c6fe32b6c164b477759cd0218738398eb838610a88e875aab62a8d
- ns3/hsr-rl.cc: None
- ns3/hsr-common.h: None

## 文件
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\acceptance_hard_criteria_20261014.json
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\acceptance_remote_evidence_20261014.tgz
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\missing16_remote_evidence_20261014.tgz
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\missing16_launch_20261014.json
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\pod_identity_acceptance_20261014.json
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\version_comparison_20261014.json
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\original_a3_consistency_20261014.json
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\pytest_acceptance_20261014.log
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\local_checks_20261014.json
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\raw\acceptance_20261014\logs\acceptance_20261014.log
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\raw\acceptance_20261014\logs\acceptance_20261014.exit
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\raw\acceptance_20261014\logs\acceptance_20261014_retry1.log
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\raw\acceptance_20261014\logs\acceptance_20261014_retry1.exit
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\raw\acceptance_20261014\logs\acceptance_20261014_missing16.log
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\raw\acceptance_20261014\logs\acceptance_20261014_missing16.exit
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\raw\acceptance_20261014\logs\acceptance_20261014_retry1
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\raw\acceptance_20261014\logs\acceptance_20261014_missing16
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\raw\acceptance_20261014\logs\acceptance_20261014_retry1\manifest.json
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\raw\acceptance_20261014\logs\acceptance_20261014_retry1\verdict.json
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\raw\acceptance_20261014\logs\acceptance_20261014_missing16\results.json

## 剩余门禁实测及最终结论
完整验收 **FAIL**，生产继续H13。五族矩阵PASS_MATRIX保持，不重跑；load16全部16个唯一seed完成，14集full480/truncated，2集low_sinr死亡，原分类survived14/ho_to_weaker2，no_ho_count=0；即使no_ho为0，也因survival<1而FAIL。

|seed|steps|termination|class|death_x|completed HO|attempts|
|---|---:|---|---|---:|---:|---:|
|500000|480|truncated|survived|None|4|4|
|500001|480|truncated|survived|None|4|5|
|501000|480|truncated|survived|None|4|5|
|501001|480|truncated|survived|None|4|6|
|502000|480|truncated|survived|None|4|4|
|502001|480|truncated|survived|None|4|4|
|503000|196|low_sinr|ho_to_weaker|1657.6700439453125|4|12|
|503001|480|truncated|survived|None|4|4|
|504000|480|truncated|survived|None|4|4|
|504001|480|truncated|survived|None|4|4|
|505000|190|low_sinr|ho_to_weaker|1607.68994140625|4|6|
|505001|480|truncated|survived|None|4|4|
|506000|480|truncated|survived|None|4|4|
|506001|480|truncated|survived|None|4|5|
|507000|480|truncated|survived|None|4|5|
|507001|480|truncated|survived|None|4|4|

load16实际请求/执行不同次数：0。共享救援环境收益不归因于裸网络。死亡分类是末30步启发式标签，不等于单一因果证明。
死亡last_ho：{"seed": 503000, "last_ho": {"step": 196, "x": 1657.6700439453125, "srv_rsrp": -90.0, "srv_sinr": -13.768776337045374, "tgt_rsrp": -78.0, "tgt_sinr": 2.473261292602005, "executed_action": 1, "executed_target": {"rsrp": -78.0, "sinr": 2.473261292602005, "cell": 3}, "policy_requested_target": {"rsrp": -78.0, "sinr": 2.473261292602005, "cell": 3}, "sinr_after_mean": null, "a3_action": 1, "a3_tgt_rsrp": -78.0, "steps_to_end": 0, "n_ho_total": 4}}
死亡last_ho：{"seed": 505000, "last_ho": {"step": 190, "x": 1607.68994140625, "srv_rsrp": -90.0, "srv_sinr": -13.461673685872626, "tgt_rsrp": -79.0, "tgt_sinr": 0.7464250502370782, "executed_action": 1, "executed_target": {"rsrp": -79.0, "sinr": 0.7464250502370782, "cell": 3}, "policy_requested_target": {"rsrp": -79.0, "sinr": 0.7464250502370782, "cell": 3}, "sinr_after_mean": null, "a3_action": 1, "a3_tgt_rsrp": -79.0, "steps_to_end": 0, "n_ho_total": 4}}

实际二进制SHA256：`8d42cc7ebadce872a3a0a8b1617ff05770e8e4a1b6c90dae01d0e3862d34b9d7`，与既有HSR4记录完全相同。真实构建source/header hashes见native_provenance JSON，历史selftest不替代load16。
- {"name": "load16_exact_unique", "passed": true}
- {"name": "load16_commands_exit0", "passed": true}
- {"name": "load16_original_primary", "passed": false, "detail": {"name": "J_main_load16", "n": 16, "survival": 0.875, "death_classes": {"survived": 14, "ho_to_weaker": 2}, "no_ho_count": 0, "pass": false}}
- {"name": "load16_full480_survival", "passed": false}
- {"name": "native_source_binary_provenance", "passed": true}
- {"name": "load16_same_config_checkpoint_binary", "passed": true}
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\load16_remote_evidence_20261014.tgz
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\load16_launch_20261014.json
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\load16_download_exit_20261014.json
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\load16_runtime_hashes_20261014.json
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\native_provenance_20261014.json
- D:\PROJECT\highspeed-rl\session_logs\h15_survival_fix\raw\acceptance_20261014\logs\acceptance_load16_20261014

## 失败轨迹只读事实与推断边界
两集全部action_source=rl，load16总override=0。RL主动请求第三邻区(action5)且执行一致，A3当时stay；503000在192步request_result=1、193步HO完成，505000在187步request_result=1且HO完成。目标cell1 RSRP=-90，分别比当时服务弱11/10dB，候选SINR约-13.46/-12.87；随后服务cell1 RSRP=-90，SINR约-13.77/-13.46，方向及量级一致但非完全相等。
两集终止前三步服务SINR持续低于-6，native rlf=0；最后一步回切强目标action1、request_result=1、HO完成且吞吐恢复，但该步服务测量仍弱，故Python三步low_sinr检测终止。不能把HO完成事件等同当步测量已更新，也不能称最后HO失败。
源码fallback_before_rlf仅覆盖action不属于HO_ACTIONS/DUAL_ACTIONS时，不改写主动HO；两集弱HO之后继续HO5/HO1，因此未有safety_a3接管。configured mask在此仅屏蔽3/4/6，没有屏蔽5；env弱目标保护另需_rules_active()。这是当前配置执行边界，不是经干预验证的唯一根因。
本阶段同矩阵config/checkpoint/binary；新增seed索引3和5暴露死亡，并非换配置。class ho_to_weaker只是末30步启发式；FAIL依据候选真实终止且非full480，无需依赖根因推断。无修复、重训、重编译或生产替换。
