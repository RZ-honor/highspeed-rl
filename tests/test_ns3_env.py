"""NS3HandoverEnv 协议级测试（用 scripts/ns3_mock.py 模拟 NS-3，无需真实 NS-3）。"""

import sys

import numpy as np

from src.utils.config import load_config
from src.envs.ns3_env import (HSR_STATE_DTYPE, HSR_ACTION_DTYPE, NS3HandoverEnv,
                              load_gap_penalty)

CFG = load_config("configs/default.yaml")


def make_env(episode_steps=30, dur=6.0, load_obs=False):
    cfg = load_config("configs/default.yaml")
    cfg["seed"] = 7
    cfg["env"] = dict(cfg["env"])
    cfg["env"]["episode_steps"] = episode_steps
    if load_obs:
        cfg["env"]["extended_obs"] = True
        cfg["env"]["load_obs"] = True
        cfg["env"]["obs_dim"] = 28
    cfg["ns3"] = {
        "binary": sys.executable,
        "args": f"-m scripts.ns3_mock --dur={dur} --nEnb=8 --spacing=600 --offset=50",
        "cwd": ".",
        "timeout_s": 60,
    }
    return NS3HandoverEnv(cfg)


def test_deadline_fallback_is_opt_in_and_dispatches_before_rlf():
    for enabled in (False, True):
        env = make_env()
        try:
            env.reset(seed=7)
            env.env_cfg.update(fallback_before_rlf=enabled,
                               fallback_persist_steps=8, rlf_consecutive_steps=3)
            env.cfg["safety"]["mode"] = "guard"
            env._rlf_count = 1
            env._prev_sinr = np.array([-6.3, 1., -10., -20.])
            env._prev_rsrp = np.array([-77., -73., -90., -100.])
            env._fb_count = 0
            _, _, _, _, info = env.step(0)
            assert info["action_requested"] == 0
            assert info["action_executed"] == (1 if enabled else 0)
            assert info["action_source"] == ("safety_a3" if enabled else "rl")
            assert env.env_cfg["rlf_consecutive_steps"] == 3
        finally:
            env.close()


def test_protocol_sizes():
    # HSR4：HSR3 的容量字段上新增切换请求/进行中/失败及区间遥测。
    assert HSR_STATE_DTYPE.itemsize == 99
    assert HSR_ACTION_DTYPE.itemsize == 8


def test_capacity_fields_present():
    """容量指标必须在 dtype 中且为数值型（探针依赖这些键）。"""
    for k in ("n_ue_cell", "tp_agg_mbps", "tp_agg_cell_mbps",
              "tp_ue_p05_mbps", "tp_ue_p50_mbps", "jain"):
        assert k in HSR_STATE_DTYPE.names, f"缺少容量字段 {k}"
        assert np.issubdtype(HSR_STATE_DTYPE[k].base, np.number)


def test_reset_obs_shape_and_range():
    env = make_env()
    obs, info = env.reset(seed=7)
    assert obs.shape == (CFG["env"]["obs_dim"],)
    assert np.all((obs >= 0) & (obs <= 1.0 + 1e-6))
    assert "candidate_rsrp" in info and "serving_sinr" in info
    assert len(info["candidate_rsrp"]) == 4
    env.close()


def test_step_returns_finite():
    env = make_env()
    obs, info = env.reset(seed=7)
    for _ in range(5):
        obs, r, term, trunc, info = env.step(0)
        assert obs.shape == (CFG["env"]["obs_dim"],)
        assert np.isfinite(r)
        assert np.all(np.isfinite(obs))
    env.close()


def test_episode_truncates():
    env = make_env(episode_steps=20, dur=6.0)
    env.reset(seed=7)
    steps = 0
    for _ in range(30):
        _, r, term, trunc, info = env.step(0)
        steps += 1
        if term or trunc:
            break
    assert steps <= 21
    assert trunc or term
    env.close()


def test_ho_action_changes_serving():
    env = make_env()
    obs, info = env.reset(seed=7)
    s0 = info["serving_idx"]
    changed = False
    for _ in range(40):
        # 一直切向最强邻区（mock 立即生效）
        obs, r, term, trunc, info = env.step(1)
        if info["serving_idx"] != s0:
            changed = True
            break
        if term or trunc:
            break
    assert changed, "动作 1 应导致服务小区变化"
    env.close()


def test_done_state_ends_episode():
    """dur 耗尽时 mock 发 done=1，env 应以 truncated 结束。"""
    env = make_env(episode_steps=200, dur=2.5)  # dur < episode_steps*dt
    env.reset(seed=7)
    done = False
    for _ in range(250):
        _, r, term, trunc, info = env.step(0)
        if term or trunc:
            done = True
            break
    assert done
    env.close()


def test_load_obs_adds_five_dims():
    """负载可观测量（P1-6）：开启后 obs_dim = 23 + 5，且 info 暴露各小区负载。

    动机：旧观测无任何负载维度，「切向空闲次强小区做分流」在观测上不可实现。
    本用例保证新增的 5 维（4 个负载 + 隧道指示）确实进入观测并在 info 可见。
    """
    env = make_env(load_obs=True)
    obs, info = env.reset(seed=7)
    assert obs.shape == (28,), obs.shape
    assert np.all((obs >= 0) & (obs <= 1.0 + 1e-6))
    for k in ("load_serving", "load_nbr", "in_tunnel_ns3"):
        assert k in info, k
    assert len(info["load_nbr"]) == 3
    obs2, r, term, trunc, info2 = env.step(0)
    assert obs2.shape == (28,)
    env.close()


def test_load_obs_off_keeps_23():
    env = make_env()
    obs, _ = env.reset(seed=7)
    assert obs.shape == (int(CFG["env"]["obs_dim"]),)
    env.close()


def test_load_gap_penalty_basic():
    """显式负载项：服务小区比最轻的有效邻区更拥挤才计罚。"""
    # 服务 6 / 邻区 2 → gap 4，deadband 1，scale 4 → (4-1)/4 = 0.75
    assert abs(load_gap_penalty(6, [2, 9, 9], [True, True, True], 4.0, 1.0) - 0.75) < 1e-6
    # 服务更轻 → 不罚
    assert load_gap_penalty(1, [5, 5, 5], [True, True, True], 4.0, 1.0) == 0.0
    # 差距在 deadband 内 → 不罚（避免负载噪声引发乒乓）
    assert load_gap_penalty(3, [2, 2, 2], [True, True, True], 4.0, 1.0) == 0.0
    # 无测量有效邻区 → 不罚（不可为躲负载指向未知小区）
    assert load_gap_penalty(9, [0, 0, 0], [False, False, False], 4.0, 1.0) == 0.0
    # 只比较有效邻区：无效邻区的 0 不应被当成「很空闲」
    assert load_gap_penalty(5, [0, 4, 0], [False, True, False], 4.0, 1.0) == 0.0
    # 封顶 1.0
    assert load_gap_penalty(50, [0, 0, 0], [True, True, True], 4.0, 1.0) == 1.0


def test_close_idempotent():
    env = make_env()
    env.reset(seed=7)
    env.close()
    env.close()


def test_full_final_state_and_interval_telemetry():
    env = make_env(episode_steps=200, dur=1.1)
    env.env_cfg["rlf_consecutive_steps"] = 1000
    try:
        env.reset(seed=7)
        while True:
            obs, reward, term, trunc, info = env.step(0)
            if term or trunc:
                break
        assert trunc and not term
        assert info["time_ms"] == 1100
        assert info["serving_rsrp"] > -190
        assert info["ue_x"] > 80
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
        assert info["telemetry_available"]
        assert info["interval_ms"] > 0
    finally:
        env.close()


def test_hsr3_state_compatibility():
    env = make_env()
    s = np.zeros(1, dtype=HSR_STATE_DTYPE)
    s["jain"] = .75
    raw = s.tobytes()[:84]
    env._state_size = 84
    env._read_exact = lambda n: raw if n == 84 else None
    parsed = env._read_state()
    assert parsed["jain"] == .75
    assert parsed["ho_failed"] == 0


def test_rlf_overlap_penalized_once():
    results = []
    for native, consecutive in ((False, False), (True, False), (False, True), (True, True)):
        env = make_env()
        try:
            env.reset(seed=7)
            state = env._read_state
            def read_with_rlf():
                s = state().copy()
                s["rlf"] = int(native)
                return s
            env._read_state = read_with_rlf
            env.env_cfg["w_outage"] = 0.0
            env.env_cfg["w_coverage"] = 0.0
            env.env_cfg["rlf_sinr_db"] = 100.0 if consecutive else -100.0
            env.env_cfg["rlf_consecutive_steps"] = 1
            _, reward, term, _, info = env.step(0)
            results.append((reward, term, info["reward_components"]))
        finally:
            env.close()
    baseline = results[0][0]
    penalty = (float(env.env_cfg.get("rlf_drop_penalty", 1.0)) * 29
               + float(env.env_cfg.get("rlf_termination_penalty", 10.0)))
    cap = int(env.env_cfg.get("rlf_remaining_cap_steps", 0))
    if cap > 0:
        penalty = (float(env.env_cfg.get("rlf_drop_penalty", 1.0)) * min(29, cap)
                   + float(env.env_cfg.get("rlf_termination_penalty", 10.0)))
    assert not results[0][1]
    for reward, term, components in results[1:]:
        assert term
        assert np.isclose(baseline - reward, penalty)
        assert np.isclose(components["rlf"], -penalty)
    assert results[1][2]["rlf"] == results[2][2]["rlf"] == results[3][2]["rlf"]


def test_tunnel_observation_uses_deployed_state():
    env = make_env(load_obs=True)
    # Tunnel deployment is a scenario fact, not a stale base command override.
    env.cfg["ns3"]["scenario"] = {"randomize": False, "tunnels": [[0, 1000]]}
    try:
        obs, info = env.reset(seed=7)
        assert obs[18] == obs[27] == 1
        assert info["in_tunnel_ns3"]
        # 协议位是 HSR4 新版本唯一运行时真值；本地缓存变化不应影响观测。
        env._tun_segs = []
        obs, _, _, _, info = env.step(0)
        assert obs[18] == obs[27] == 1
    finally:
        env.close()


def test_empty_family_tunnels_override_base_deployment():
    env = make_env(load_obs=True)
    env.args_tpl += " --tunnels=0:1000"
    env.cfg["ns3"]["scenario"] = {"tunnels": []}
    try:
        obs, info = env.reset(seed=7)
        assert obs[18] == obs[27] == 0
        assert not info["in_tunnel_ns3"]
        assert env._tun_segs == []
    finally:
        env.close()


def test_spawn_cmd_never_sends_empty_tunnels():
    # HSR4 真实验收发现：NS-3 CommandLine 拒绝空值参数（`--tunnels=` 直接
    # 打印 usage 退出）。空隧道必须靠「剔除 base 旧部署 + 不发标记」实现。
    env = make_env(load_obs=True)
    env.args_tpl += " --tunnels=0:1000 --tunnelEnbSpacing=250"
    env.cfg["ns3"]["scenario"] = {"tunnels": []}
    try:
        env.reset(seed=7)
        assert not any(a.startswith(("--tunnels=", "--tunnelEnb", "--tunnelMicro"))
                       for a in env._spawn_cmd)
    finally:
        env.close()


def test_spawn_cmd_carries_deployed_tunnels():
    env = make_env(load_obs=True)
    env.cfg["ns3"]["scenario"] = {"randomize": False, "tunnels": [[0, 1000]]}
    try:
        env.reset(seed=7)
        assert "--tunnels=0:1000" in env._spawn_cmd
    finally:
        env.close()


def test_spawn_empty_after_randomize_sampling():
    # cmd_0199 首跑的 collector 失败路径：候选非空但 randomize 过滤成空。
    # 判定必须跟着采样结果走：不发空值标记，且剔除 base 残留部署。
    env = make_env(load_obs=True)
    env.args_tpl += " --tunnels=0:1000"
    env.cfg["ns3"]["scenario"] = {"randomize": True, "tunnels": [[2000, 2800]]}

    class _Rng:
        def random(self):
            return 0.99      # 每段保留概率 <0.6 → 候选全部被过滤

        def uniform(self, lo, hi):
            return lo

    env._rng = _Rng()
    try:
        env.reset(seed=7)
        assert env._tun_segs == []
        assert not any(a.startswith(("--tunnels=", "--tunnelEnb", "--tunnelMicro"))
                       for a in env._spawn_cmd)
    finally:
        env.close()


# ---------------- 动作空间死亡修复（H15 根因 → H16） ----------------

def _hs_env(grace=0, hard=True):
    """28 维隔离环境：configured/guard 关闭语义（legacy 安全层不生效）+ 可选硬盾。

    hard_shield/measurement_grace 在 __init__ 里被解析成缓存字段，测试里改
    cfg 之后必须同步重算，否则改的是「配置」而不是「生效值」。
    """
    from src.agents.safety import hard_shield_params
    env = make_env(episode_steps=60, dur=8.0, load_obs=True)
    env.cfg["safety"] = dict(env.cfg["safety"])
    env.cfg["safety"].update(mode="guard", policy_mask="configured")
    if hard:
        env.cfg["safety"].update(hard_shield=True,
                                 hard_max_drop_vs_serving_db=10,
                                 hard_min_target_rsrp_dbm=-110,
                                 hard_min_target_sinr_db=-6)
    env.env_cfg["measurement_grace_steps"] = grace
    env._hard = hard_shield_params(env.cfg)
    env._meas_grace = int(env.env_cfg["measurement_grace_steps"])
    env.guard = None
    return env


def test_hard_shield_rewrites_lethal_ho_at_env_gate():
    """死亡复现：action5 → -89dBm（弱10dB 且 SINR-13.5）被改写 stay。"""
    env = _hs_env()
    try:
        env.reset(seed=7)
        env._prev_rsrp = np.array([-79.0, -84.0, -82.0, -89.0])
        env._prev_sinr = np.array([-5.0, 1.0, 0.5, -13.5])
        _, _, _, _, info = env.step(5)
        assert info["action_executed"] == 0
        assert info["action_source"] == "masked_hard"
    finally:
        env.close()


def test_hard_shield_permits_healthy_slightly_weaker_ho():
    """宽松性：略弱(≤10dB)但 SINR 健康的目标照常放行（负载均衡空间）。"""
    env = _hs_env()
    try:
        env.reset(seed=7)
        env._prev_rsrp = np.array([-80.0, -85.0, -84.0, -83.0])
        env._prev_sinr = np.array([0.0, 2.0, 1.5, 1.0])
        _, _, _, _, info = env.step(1)
        assert info["action_source"] == "rl"
        assert info["action_executed"] == 1
    finally:
        env.close()


def test_measurement_staleness_state_machine():
    """纯函数冻结语义：目标上报到/HO失败/超窗 → 立即恢复计数；绝不豁免。"""
    from src.envs.ns3_env import measurement_staleness as ms
    # grace=0 → 恒不冻结（生产默认=旧行为）
    assert ms(5, 42, 8, 0, 10, False, False, True) == (False, 42, 8, 0)
    # pending 无 → 不冻结
    assert ms(5, None, 8, 3, 10, False, False, True) == (False, None, 8, 0)
    # HO 后首个状态：serving 仍为源小区 → 冻结
    assert ms(5, 42, 8, 3, 10, False, False, True) == (True, 42, 8, 1)
    # 目标小区上报到达 → 解除
    assert ms(6, 42, 8, 3, 42, False, False, True) == (False, None, 8, 0)
    # 测量未上报（哨兵）不得误判为「目标已刷新」
    assert ms(6, 42, 8, 3, 42, False, False, False) == (True, 42, 8, 1)
    # HO 失败 → 解除（源小区就是真实服务，低 SINR 必须照常计数）
    assert ms(6, 42, 8, 3, 10, True, False, True) == (False, None, 8, 0)
    # 进行中 → 续窗
    assert ms(7, 42, 8, 3, 10, False, True, True) == (True, 42, 10, 1)
    # 超窗仍未见目标 → 解除（不豁免真实劣化）
    assert ms(9, 42, 8, 3, 10, False, False, True) == (False, None, 8, 0)
    # 连续冻结总长封顶（run_cap）：反复派发 HO 不能无限推迟终止
    assert ms(8, 42, 99, 3, 10, False, True, True, run=6, run_cap=6)         == (False, None, 99, 0)


def test_stale_measurement_does_not_advance_rlf_but_truthful_one_terminates():
    """集成：陈旧测量冻结计数；宽限耗尽后真实低 SINR 仍然终止（不伪造存活）。"""
    env = _hs_env(grace=2, hard=False)
    try:
        env.reset(seed=7)
        env.env_cfg["rlf_consecutive_steps"] = 3
        env.env_cfg["measurement_grace_steps"] = 2
        # 关闭一切救援改写（persist 拉满、preempt 门限低于注入的 -100），
        # 让冻结→恢复→终止的路径保持确定，不被 A3 兜底续窗干扰。
        env.env_cfg["fallback_persist_steps"] = 100
        env.env_cfg["preempt_rsrp_dbm"] = -150.0
        env.env_cfg["fallback_tp_collapse_mbps"] = 0.0
        state0 = env._read_state

        steps = {"n": 0}
        def read_stale():
            s = state0().copy()
            steps["n"] += 1
            s["serving_cell"] = 1          # 恒为源小区（目标永不上报）
            s["serving_rsrp_dbm"] = -100   # 服务测量持续低
            s["ho_event"] = 0
            s["rlf"] = 0
            return s
        env._read_state = read_stale
        # 派发的 HO 目标（nbr1）从上一状态候选取
        env._prev_cells = [1, 42, 43, 44]
        _, _, term, _, info = env.step(1)
        assert info["action_executed"] == 1
        assert info["meas_stale"] is True and env._rlf_count == 0
        # 再一步（step=2 < grace_until=3）：仍冻结，计数保持 0
        _, _, term, _, info = env.step(0)
        assert not term and info["meas_stale"] is True and env._rlf_count == 0
        # 宽限耗尽（step≥3）→ 恢复真实计数；低 SINR 持续 → 数步内必须终止，
        # 且冻结期间的步数不计入（终止点必然晚于「若无冻结」的终止点）
        terms = 0
        counts = []
        for _ in range(5):
            _, _, term, _, info = env.step(0)
            counts.append(env._rlf_count)
            if term:
                terms += 1
                break
        assert terms == 1, "陈旧宽限绝不能豁免真实劣化：计数恢复后必须终止"
        assert counts and counts[0] >= 1, "宽限耗尽后低 SINR 必须重新累计"
    finally:
        env.close()
