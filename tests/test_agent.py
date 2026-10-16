import numpy as np
import torch
from src.utils.config import load_config
from src.agents.agent import DuelingDDQNAgent

CFG = load_config("configs/default.yaml")


def make_agent(device="cpu"):
    cfg = dict(CFG)
    cfg["seed"] = 0
    return DuelingDDQNAgent(cfg, torch.device(device))


def test_ns3_action_mask_act_and_target():
    import copy
    cfg = copy.deepcopy(CFG)
    cfg["ns3"] = {"binary": "unused"}
    ag = DuelingDDQNAgent(cfg, torch.device("cpu"))
    class Fixed(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([[1., 2., 3., 999., 998., 4., 997.]]).expand(len(x), -1)
    ag.online_net = ag.target_net = Fixed()
    obs = np.zeros(14, dtype=np.float32)
    info = {"candidate_rsrp": np.array([-90., -80., -80., -80.]),
            "last_ho_step": -100, "step_idx": 0}
    assert ag.act(obs, info, 0.) == 5
    assert {ag.act(obs, info, 1.) for _ in range(200)} == {0, 1, 2, 5}
    batch = {"state": torch.zeros(1, 14), "next_state": torch.zeros(1, 14),
             "action": torch.tensor([0]), "reward": torch.tensor([0.]),
             "discount": torch.tensor([0.5])}
    assert ag._q_and_target(batch)[1].item() == 2.
    assert make_agent().legal_actions == tuple(range(7))


def test_policy_mask_configured_guard_and_exploration():
    import copy
    from src.agents.safety import select_action, RuleGuard
    cfg = copy.deepcopy(CFG)
    cfg["ns3"] = {"binary": "unused"}
    cfg["safety"].update(mode="guard", policy_mask="configured")
    cfg["guard"] = {"enable": False}
    info = {"candidate_rsrp": [-80., -120., -130., -140.],
            "last_ho_step": -100, "step_idx": 0}
    q = torch.tensor([[0., 9., 8., 999., 998., 7., 997.]])
    assert select_action(q, cfg, info) == 1
    cfg["safety"]["mode"] = "strict"
    assert select_action(q, cfg, info) == 0
    assert {select_action(q, cfg, info, eps=1.) for _ in range(40)} == {0}
    cfg["safety"]["mode"] = "guard"
    guard = RuleGuard({"enable": True})
    guard._active = True
    assert select_action(q, cfg, info, guard=guard) == 0
    cfg["safety"]["policy_mask"] = "legacy"
    assert select_action(q, cfg, info) == 0


def test_act_shape():
    ag = make_agent()
    obs = np.zeros(14, dtype=np.float32)
    info = {"candidate_rsrp": np.array([-90.0, -80.0, -80.0, -80.0]),
            "serving_idx": 0, "last_ho_step": -100, "step_idx": 0}
    a = ag.act(obs, info, eps=0.0)
    assert 0 <= a < 7


def test_learn_returns_finite():
    ag = make_agent()
    for _ in range(80):
        ag.store(np.zeros(14, dtype=np.float32), 0, 1.0, np.zeros(14, dtype=np.float32), False)
    loss = ag.learn()
    assert np.isfinite(loss)


def test_target_sync():
    ag = make_agent()
    w0 = ag.target_net.shared[0].weight.data.clone()
    ag.sync_target()
    w1 = ag.target_net.shared[0].weight.data.clone()
    assert torch.equal(w0, w1)


def test_bc_anchor_keeps_q_near_reference():
    """BC 锚定必须把 Q 约束在起点策略附近（治「后期坍缩为不切换」的漂移）。

    实测：在线微调约 c100 达峰后 Q(HO) 下漂 → 快照不切 → buffer 里 HO 更少 →
    Q(HO) 更低，自强化坍缩。锚定项 ‖Q(s,·)−Q_ref(s,·)‖² 的作用就是给这条
    正反馈上锁：**允许**吸收干净奖励带来的改进，**禁止**整体偏好漂移。
    """
    cfg = dict(CFG)
    cfg["agent"] = dict(cfg["agent"])
    cfg["agent"]["bc_anchor_w"] = 5.0          # 强锚定，便于观察约束效果
    ag = DuelingDDQNAgent(cfg, torch.device("cpu"))
    assert ag.set_bc_anchor() is True
    assert ag.anchor_net is not None

    obs = np.zeros(14, dtype=np.float32)
    for _ in range(80):                        # 构造一个「奖励全打给 stay」的批次
        ag.store(obs, 1, 5.0, obs, False)
    ref_before = ag.anchor_net(torch.as_tensor(obs).unsqueeze(0)).detach().clone()

    for _ in range(60):
        ag.learn()

    with torch.no_grad():
        q_now = ag.online_net(torch.as_tensor(obs).unsqueeze(0))
    drift = float((q_now - ref_before).abs().max().item())
    # 强锚定下漂移必须显著小于「无锚定」对照
    cfg2 = dict(CFG)
    cfg2["agent"] = dict(CFG["agent"])
    cfg2["agent"]["bc_anchor_w"] = 0.0
    ag2 = DuelingDDQNAgent(cfg2, torch.device("cpu"))
    ag2.online_net.load_state_dict(ag.online_net.state_dict())   # 同起点
    ag2.sync_target()
    for _ in range(80):
        ag2.store(obs, 1, 5.0, obs, False)
    for _ in range(60):
        ag2.learn()
    with torch.no_grad():
        q_free = ag2.online_net(torch.as_tensor(obs).unsqueeze(0))
    drift_free = float((q_free - ref_before).abs().max().item())
    assert drift < drift_free, (drift, drift_free)


def test_bc_anchor_disabled_by_default():
    ag = make_agent()
    assert ag.anchor_net is None
    assert ag.set_bc_anchor() is False          # w=0 时不启用


def test_save_load(tmp_path):
    ag = make_agent()
    p = tmp_path / "m.pt"
    ag.save(str(p))
    ag2 = make_agent()
    ag2.load(str(p))
    info = {"candidate_rsrp": np.array([-90., -80., -80., -80.]),
            "serving_idx": 0, "last_ho_step": -100, "step_idx": 0}
    a1 = ag.act(np.zeros(14, dtype=np.float32), info, 0.0)
    a2 = ag2.act(np.zeros(14, dtype=np.float32), info, 0.0)
    assert a1 == a2


def _boot_obs28(rsrp, sinr, reported=True):
    import numpy as np
    def nr(x):
        return min(max((x + 140.0) / 96.0, 0.0), 1.0)
    def ns(x):
        return min(max((x + 30.0) / 60.0, 0.0), 1.0)
    obs = np.zeros(28, dtype=np.float32)
    obs[0], obs[1] = nr(rsrp[0]), ns(sinr[0])
    obs[2], obs[3], obs[14] = nr(rsrp[1]), nr(rsrp[2]), nr(rsrp[3])
    obs[15], obs[16], obs[17] = ns(sinr[1]), ns(sinr[2]), ns(sinr[3])
    obs[22] = 1.0 if reported else 0.0
    return obs


def _h16_cfg():
    import copy
    cfg = copy.deepcopy(CFG)
    cfg["ns3"] = {"binary": "unused"}
    cfg["env"] = dict(cfg["env"])
    cfg["env"].update(extended_obs=True, obs_dim=28)
    cfg["safety"] = dict(cfg["safety"])
    cfg["safety"].update(hard_shield=True, mode="guard",
                         policy_mask="configured")
    cfg["guard"] = {"enable": False}
    return cfg


def test_bootstrap_excludes_hard_masked_action_from_target():
    """修复2：target 的 argmax 必须落在与部署一致的 admissible 集上。

    next_state 里 nbr3 比服务弱 15 dB → action5 被 hard shield 屏蔽；
    尽管 Q(5)=100 全局最大，bootstrap 只能取合法集内最大值 Q(1)=9。
    """
    import numpy as np
    import torch
    cfg = _h16_cfg()
    ag = DuelingDDQNAgent(cfg, torch.device("cpu"))
    q_row = torch.tensor([0.0, 9.0, 8.0, 0.0, 0.0, 100.0, 0.0])

    class Fixed(torch.nn.Module):
        def forward(self, x):
            return q_row.expand(len(x), -1)

    ag.online_net = ag.target_net = Fixed()
    s2 = _boot_obs28([-80.0, -70.0, -75.0, -95.0],
                     [5.0, 5.0, 5.0, 5.0])
    batch = {"state": torch.zeros(1, 28), "next_state": torch.from_numpy(s2)[None],
             "action": torch.tensor([0]), "reward": torch.tensor([0.0]),
             "discount": torch.tensor([1.0])}
    _, y = ag._q_and_target(batch)
    assert y.item() == 9.0          # 而非 100（被屏蔽）或 8

    # 同一 obs 在 shield 关闭时保持旧行为（target=100）——回归锚点
    import copy
    cfg_off = copy.deepcopy(cfg)
    cfg_off["safety"]["hard_shield"] = False
    ag2 = DuelingDDQNAgent(cfg_off, torch.device("cpu"))
    ag2.online_net = ag2.target_net = Fixed()
    _, y_off = ag2._q_and_target(batch)
    assert y_off.item() == 100.0


def test_bootstrap_leaves_terminal_and_static_paths_untouched():
    """terminal（discount=0）与未启用 shield 的批次逐位等于旧公式。"""
    import torch
    cfg = _h16_cfg()
    ag = DuelingDDQNAgent(cfg, torch.device("cpu"))
    q_row = torch.tensor([0.0, 9.0, 8.0, 0.0, 0.0, 100.0, 0.0])

    class Fixed(torch.nn.Module):
        def forward(self, x):
            return q_row.expand(len(x), -1)

    ag.online_net = ag.target_net = Fixed()
    s2 = _boot_obs28([-80.0, -70.0, -75.0, -95.0], [5.0, 5.0, 5.0, 5.0])
    batch = {"state": torch.zeros(1, 28), "next_state": torch.from_numpy(s2)[None],
             "action": torch.tensor([2]), "reward": torch.tensor([3.0]),
             "discount": torch.tensor([0.0])}
    q_sa, y = ag._q_and_target(batch)
    assert y.item() == 3.0          # terminal：bootstrap 折扣为零
    assert q_sa.item() == 8.0
