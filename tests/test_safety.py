import numpy as np
import torch
from src.utils.config import load_config
from src.agents.safety import RuleGuard, SafetyLayer, legal_actions, mask_actions

CFG = load_config("configs/default.yaml")


def make_layer():
    return SafetyLayer(CFG["safety"])


def test_backend_action_capability_masks_ns3_but_not_default_or_mock():
    import copy
    default = copy.deepcopy(CFG)
    assert legal_actions(default, 7) == tuple(range(7))
    ns3 = copy.deepcopy(CFG)
    ns3["ns3"] = {"binary": "unused"}
    assert legal_actions(ns3, 7) == (0, 1, 2, 5)
    mock = copy.deepcopy(ns3)
    mock["mock_backend"] = True
    assert legal_actions(mock, 7) == tuple(range(7))


def test_mask_actions_preserves_shape_and_masks_only_unsupported():
    q = torch.tensor([[1., 2., 3., 4., 5., 6., 7.]])
    masked = mask_actions(q, (0, 1, 2, 5))
    assert masked.shape == q.shape
    assert torch.equal(masked[0, [0, 1, 2, 5]], q[0, [0, 1, 2, 5]])
    assert torch.isneginf(masked[0, 3])
    assert torch.isneginf(masked[0, 4])
    assert torch.isneginf(masked[0, 6])


def test_mask_low_rsrp():
    sl = make_layer()
    q = torch.tensor([[1.0, 5.0, 2.0, 0.0, 0.0, 3.0, 0.0]])
    # candidate_rsrp = [serving=-90, n1=-120(低), n2=-80, n3=-75]
    info = {"candidate_rsrp": np.array([-90.0, -120.0, -80.0, -75.0]),
            "serving_idx": 0, "last_ho_step": -100, "step_idx": 5}
    masked = sl.mask(q, info)
    # 切到 n1(动作1) 因 RSRP -120 < -110 被掩码
    assert masked[0, 1].item() == -float("inf")
    assert masked[0, 0].item() == 1.0  # stay 不受影响


def test_pingpong_block():
    sl = make_layer()
    q = torch.tensor([[1.0, 9.0, 2.0, 0.0, 0.0, 3.0, 0.0]])
    info = {"candidate_rsrp": np.array([-90.0, -80.0, -80.0, -80.0]),
            "serving_idx": 0, "last_ho_step": 4, "step_idx": 5}
    masked = sl.mask(q, info)
    # 近2步内切换 → 掩码所有切换动作(1,2,5)；双连(3,4,6)允许
    assert masked[0, 1].item() == -float("inf")
    assert masked[0, 3].item() == 0.0


def test_should_fallback():
    sl = make_layer()
    # 服务 SINR 低且 RL 选了 stay → 触发回落
    assert sl.should_fallback(serving_sinr=-7.0, action=0)
    assert not sl.should_fallback(serving_sinr=5.0, action=0)


# ---------------- 规则门禁（给 RL 探索空间） ----------------

def test_guard_stays_open_by_default():
    """默认放行：样本不足/未退化时不得拉起规则，否则探索空间被压死。"""
    g = RuleGuard({"window_episodes": 4, "min_episodes": 2})
    assert g.active is False
    for r in (100.0, 90.0):
        g.observe("rl", r)
    assert g.active is False          # 只有 rl 一侧样本，不构成比较
    assert g.comparisons == 0


def test_guard_engages_only_when_rl_below_rule():
    g = RuleGuard({"window_episodes": 4, "min_episodes": 2, "release_comparisons": 2})
    # 第一轮：RL(200) 优于 A3(100) → 维持放行
    g.observe("a3", 100.0); g.observe("a3", 100.0)
    g.observe("rl", 200.0); g.observe("rl", 200.0)
    assert g.active is False
    # 第二轮：RL 退化到 50 < A3 100 → 规则接管
    g.observe("a3", 100.0); g.observe("a3", 100.0)
    g.observe("rl", 50.0); g.observe("rl", 50.0)
    assert g.active is True


def test_guard_release_has_hysteresis():
    """追平规则后必须连续 release_comparisons 次才释放，避免抖动。"""
    g = RuleGuard({"window_episodes": 4, "min_episodes": 2, "release_comparisons": 3})
    g.observe("a3", 100.0); g.observe("a3", 100.0)
    g.observe("rl", 10.0); g.observe("rl", 10.0)
    assert g.active is True
    # RL 追平（略好于规则）：前 2 次仍保持接管
    for _ in range(2):
        g.observe("a3", 100.0); g.observe("a3", 100.0)
        g.observe("rl", 110.0); g.observe("rl", 110.0)
        assert g.active is True
    # 第 3 次释放
    g.observe("a3", 100.0); g.observe("a3", 100.0)
    g.observe("rl", 110.0); g.observe("rl", 110.0)
    assert g.active is False


def test_guard_can_be_disabled():
    g = RuleGuard({"enable": False})
    g.observe("a3", 100.0); g.observe("a3", 100.0)
    g.observe("rl", -9999.0); g.observe("rl", -9999.0)
    assert g.active is False


# ---------------- hard shield（常开宽松安全盾，修复1+2） ----------------

def _hs_cfg(**over):
    import copy
    cfg = copy.deepcopy(CFG)
    cfg["ns3"] = {"binary": "unused"}
    cfg["env"] = dict(cfg["env"])
    cfg["env"]["extended_obs"] = True
    cfg["env"]["obs_dim"] = 28
    cfg["safety"] = dict(cfg["safety"])
    cfg["safety"].update(hard_shield=True, mode="guard", policy_mask="configured")
    cfg["safety"].update(over)
    cfg["guard"] = {"enable": False}
    return cfg


def test_hard_shield_blocks_the_h15_death_actions():
    """复现 503000/505000：action5 目标弱 10 dB、自身 SINR≈-13 dB → 必须被拦。

    configured + guard 关闭时 legacy SafetyLayer 不生效（这正是死亡近因），
    hard shield 独立于门禁常开。
    """
    from src.agents.safety import hard_mask, select_action
    cfg = _hs_cfg()
    info = {"candidate_rsrp": np.array([-79.0, -84.0, -82.0, -89.0]),
            "candidate_sinr": np.array([-5.0, 1.0, 0.5, -13.5]),
            "last_ho_step": -100, "step_idx": 191}
    q = torch.tensor([[0.0, -55.60, -53.11, -999., -998., -54.04, -997.]])
    assert select_action(q, cfg, info) != 5
    # 被拦的必须是 5（-89 弱10dB 且 SINR -13.5），1/2（-84/-82 健康）仍可放行
    masked = hard_mask(q.clone(), info, cfg)
    assert torch.isneginf(masked[0, 5])
    assert torch.isfinite(masked[0, 1]) and torch.isfinite(masked[0, 2])


def test_hard_shield_permits_healthy_slightly_weaker_target():
    """宽松性：切向略弱(≤10dB)但 SINR 健康的目标不受影响（负载均衡空间保留）。"""
    from src.agents.safety import hard_mask
    cfg = _hs_cfg()
    q = torch.tensor([[0.0, 5.0, 4.0, 0.0, 0.0, 3.0, 0.0]])
    info = {"candidate_rsrp": np.array([-80.0, -85.0, -84.0, -83.0]),
            "candidate_sinr": np.array([0.0, 2.0, 1.5, 1.0])}
    masked = hard_mask(q, info, cfg)
    assert torch.equal(masked[0, [1, 2, 5]], q[0, [1, 2, 5]])


def test_hard_shield_rsrp_floor_and_absent_neighbor_blocked():
    from src.agents.safety import hard_mask
    cfg = _hs_cfg()
    q = torch.zeros(1, 7)
    info = {"candidate_rsrp": np.array([-90.0, -115.0, -95.0, -200.0]),
            "candidate_sinr": np.array([5.0, 5.0, 5.0, 5.0])}
    masked = hard_mask(q, info, cfg)
    assert torch.isneginf(masked[0, 1])   # -115 < -110 地板
    assert torch.isneginf(masked[0, 5])   # -200 哨兵=无测量邻区
    assert torch.isfinite(masked[0, 2])


def test_hard_shield_no_block_before_first_measurement():
    """测量未上报（服务 -200 哨兵）时不拦——与 legacy 同口径，防开局误杀。"""
    from src.agents.safety import hard_mask
    cfg = _hs_cfg()
    q = torch.ones(1, 7)
    info = {"candidate_rsrp": np.array([-200.0, -200.0, -200.0, -200.0])}
    assert torch.equal(hard_mask(q, info, cfg), q)


def test_hard_shield_off_by_default_legacy_unchanged():
    """未启用时行为逐位不变（默认关闭，保护既有 checkpoint/配置）。"""
    from src.agents.safety import hard_mask, policy_mask
    import copy
    cfg = copy.deepcopy(CFG)
    cfg["ns3"] = {"binary": "unused"}
    cfg["safety"] = dict(cfg["safety"])
    cfg["safety"].update(mode="guard", policy_mask="configured")
    cfg["guard"] = {"enable": False}
    q = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    info = {"candidate_rsrp": np.array([-79.0, -89.0]),
            "candidate_sinr": np.array([-5.0, -13.5])}
    assert torch.equal(hard_mask(q, info, cfg), q)
    # configured+guard 关闭且 shield 关：只剩静态合法集 = 旧 H15 死亡语义（回归锚点）
    m = policy_mask(q, cfg, info, actions=(0, 1, 2, 5))
    assert torch.equal(m, mask_actions(q, (0, 1, 2, 5)))


def _norm_r(x):
    return np.clip((x + 140.0) / 96.0, 0.0, 1.0)


def _norm_s(x):
    return np.clip((x + 30.0) / 60.0, 0.0, 1.0)


def test_hard_bootstrap_admissible_matches_inference_mask():
    """修复2 核心：bootstrap 从 next_state 观测重建的可行动作集与推理端一致。

    推理侧 = hard_mask ∩ 静态合法；bootstrap 侧 = hard_admissible_from_obs
    ∩ 静态合法。同一候选测量必须得到同一 7 维布尔集。
    """
    from src.agents.safety import (hard_admissible_from_obs, hard_mask,
                                   mask_actions)
    cfg = _hs_cfg()
    cases = [
        # (rsrp, sinr) —— 含死亡形态、健康弱目标、地板、哨兵、无测量
        ([-79.0, -84.0, -82.0, -89.0], [-5.0, 1.0, 0.5, -13.5]),
        ([-80.0, -85.0, -84.0, -83.0], [0.0, 2.0, 1.5, 1.0]),
        ([-90.0, -115.0, -95.0, -200.0], [5.0, 5.0, 5.0, 5.0]),
        ([-90.0, -98.0, -99.0, -100.0], [-10.0, -7.0, 0.0, 3.0]),
        ([-200.0, -200.0, -200.0, -200.0], [15.0, 15.0, 15.0, 15.0]),
    ]
    for rsrp, sinr in cases:
        obs = np.zeros(28, dtype=np.float32)
        obs[0], obs[1] = _norm_r(rsrp[0]), _norm_s(sinr[0])
        obs[2], obs[3], obs[14] = _norm_r(rsrp[1]), _norm_r(rsrp[2]), _norm_r(rsrp[3])
        obs[15], obs[16], obs[17] = _norm_s(sinr[1]), _norm_s(sinr[2]), _norm_s(sinr[3])
        obs[22] = 1.0 if rsrp[0] > -190.0 else 0.0
        info = {"candidate_rsrp": np.array(rsrp), "candidate_sinr": np.array(sinr)}
        infer = torch.isfinite(
            mask_actions(hard_mask(torch.ones(1, 7), info, cfg), (0, 1, 2, 5))[0])
        boot = hard_admissible_from_obs(torch.from_numpy(obs[None]), cfg, 7)[0]
        boot = boot & torch.isfinite(mask_actions(torch.ones(7), (0, 1, 2, 5)))
        assert torch.equal(boot, infer), f"rsrp={rsrp} sinr={sinr}: boot={boot} infer={infer}"


def test_hard_bootstrap_boundary_no_extended_obs():
    """14 维旧观测缺 nbr3/SINR 维 → 显式返回 None（不冒充等价）。"""
    from src.agents.safety import hard_admissible_from_obs
    import copy
    cfg = copy.deepcopy(_hs_cfg())
    cfg["env"]["extended_obs"] = False
    assert hard_admissible_from_obs(torch.zeros(1, 14), cfg, 7) is None
