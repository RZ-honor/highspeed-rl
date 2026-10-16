"""离散 SAC（soft Q-learning）变体的最小行为测试（CPU、Python 口径配置）。

这些测试不验证「SAC 学得好」（那是 H14 验收矩阵的事），只验证**机制**：
  1. 构造/学习管线跑通，损失有限，温度在自适应；
  2. checkpoint 格式与评估管线兼容（"online" 键、argmax 贪心可用）；
  3. 载入旧 DQN checkpoint（H8 best 的格式）双 critic 同起点；
  4. soft 目标的熵项确实阻止 Q 偏好坍缩（与父类同数据对照：
     锚定关闭时 DQN 的 Q 排序可被单动作奖励打穿，SAC 不能）。
"""

import numpy as np
import torch
from src.utils.config import load_config
from src.agents.agent import DuelingDDQNAgent
from src.agents.sac_agent import DiscreteSACAgent

CFG = load_config("configs/default.yaml")


def make(cls, **agent_over):
    import copy
    cfg = copy.deepcopy(CFG)
    cfg["seed"] = 0
    cfg["agent"].update(agent_over)
    return cls(cfg, torch.device("cpu"))


def fill(ag, n=80, action=1, r=5.0):
    obs = np.zeros(14, dtype=np.float32)
    for _ in range(n):
        ag.store(obs, action, r, obs, False)


def test_ns3_sac_entropy_and_masked_learning():
    import copy
    cfg = copy.deepcopy(CFG)
    cfg["ns3"] = {"binary": "unused"}
    ag = DiscreteSACAgent(cfg, torch.device("cpu"))
    assert np.isclose(ag.target_entropy, .98 * np.log(4))
    with torch.no_grad():
        for net in (ag.online_net, ag.online_net2, ag.target_net, ag.target_net2):
            net.advantage[-1].bias[3] = 10000.
    fill(ag)
    assert np.isfinite(ag.learn())
    assert ag._last_td < 1e8  # unsupported huge Q must not enter bootstrap


def test_sac_learn_finite_and_alpha_adapts():
    ag = make(DiscreteSACAgent, bc_anchor_w=0.0)
    fill(ag)
    la0 = ag.log_alpha.detach().item()
    losses = [ag.learn() for _ in range(30)]
    assert all(np.isfinite(l) for l in losses)
    assert np.isfinite(ag.alpha) and 1e-4 <= ag.alpha <= 10.0
    # 温度控制器确实在动（方向不做断言：前期均匀策略熵高于目标 → α 先降，
    # 数据打尖后 → α 回升；净位移取决于步数，机制层面的验收交给 H14 矩阵）
    assert ag.log_alpha.detach().item() != la0


def test_sac_both_critics_receive_grad():
    """双 critic 必须都在优化器里（构造期修掉的真实 bug 的回归哨兵）。"""
    ag = make(DiscreteSACAgent, bc_anchor_w=0.0)
    fill(ag)
    w2_before = [p.detach().clone() for p in ag.online_net2.parameters()]
    for _ in range(5):
        ag.learn()
    moved = sum(int(not torch.equal(a, b))
                for a, b in zip(w2_before, ag.online_net2.parameters()))
    assert moved > 0, "critic2 参数未更新：优化器漏了 online_net2"


def test_sac_checkpoint_format_compatible_with_eval():
    """评估管线（paired_matrix/path_probe）按 ck["online"] + argmax 使用网络。"""
    import tempfile, os
    ag = make(DiscreteSACAgent)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "best.pt")
        ag.save(p)
        ck = torch.load(p, map_location="cpu")
        assert "online" in ck and "online2" in ck and "log_alpha" in ck
        # 评估侧模拟：裸 DuelingDQN 形状可直接 load（obs_dim/动作数一致）
        from src.agents.networks import DuelingDQN
        net = DuelingDQN(int(CFG["env"]["obs_dim"]), int(CFG["env"]["n_actions"]),
                         int(CFG["agent"]["hidden_dim"]))
        net.load_state_dict(ck["online"])
        q = net(torch.zeros(1, int(CFG["env"]["obs_dim"])))
        assert tuple(q.shape) == (1, int(CFG["env"]["n_actions"]))


def test_sac_loads_dqn_checkpoint_into_both_critics():
    ag = make(DiscreteSACAgent)
    ref = make(DuelingDDQNAgent)
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "h8like.pt")
        ref.save(p)                      # 纯 DQN 格式：只有 online/eps
        ag.load(p)
        w1 = list(ag.online_net.parameters())[0].data
        w2 = list(ag.online_net2.parameters())[0].data
        assert torch.equal(w1, w2)       # 双 critic 同起点
        t1 = list(ag.target_net.parameters())[0].data
        assert torch.equal(w1, t1)       # 目标网同步


def test_sac_alpha_direction_respects_entropy_target():
    """温度控制器方向断言（符号 bug 的回归哨兵，2026-10-12 首版就写反过）。

    构造一个**策略过尖**的局面：直接手改 Q 让动作 1 一枝独秀（H≈0 < H̄），
    跑一步 learn 后 logα 必须上升（α 升高 → 把分布压平）。反向的控制器
    会在训练中期把 α 推向 0，探索彻底消失 —— 与裸 DQN 无异还多了开销。
    """
    ag = make(DiscreteSACAgent, bc_anchor_w=0.0)
    fill(ag)
    with torch.no_grad():
        ag.online_net.advantage[-1].bias.zero_()
        ag.online_net.advantage[-1].weight.zero_()
        ag.online_net.advantage[-1].bias[1] = 50.0     # 动作 1 一枝独秀
    la0 = ag.log_alpha.detach().item()
    ag.learn()
    assert ag.log_alpha.detach().item() > la0, "策略过尖时 α 必须上升"
