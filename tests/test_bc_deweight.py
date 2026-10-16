"""BC → RL 初始化的去加权校准测试。

背景：BC 用加权交叉熵（切换类权重最高 20，本例 17.74）召回少数类，学到的
logit 比真实后验高 log(w)。直接当 RL 初始 Q 会把 Q(HO) 系统性抬高（实测
RL 在服务 SINR=+15.5dB 时切向 -16dB 小区、3-4 步 RLF）。存盘前必须减回去。
"""

import math

import torch

from src.utils.config import load_config
from src.agents.hybrid_network import HybridDuelingDQN
from src.agents.safety import HO_ACTIONS
from scripts.pretrain_bc import deweight_action_logits

CFG = load_config("configs/default.yaml")


def make_net():
    torch.manual_seed(0)
    c = CFG
    return HybridDuelingDQN(int(c["env"]["obs_dim"]), int(c["env"]["n_actions"]),
                            int(c["agent"]["hidden_dim"]),
                            **c.get("net_kwargs", {}))


def test_deweight_lowers_ho_q_by_log_w():
    net = make_net()
    n_act = int(CFG["env"]["n_actions"])
    w = 17.74
    class_w = torch.ones(n_act)
    for a in HO_ACTIONS:
        class_w[a] = w

    x = torch.randn(4, int(CFG["env"]["obs_dim"]))
    with torch.no_grad():
        q_before = net(x)
    assert deweight_action_logits(net, class_w)
    with torch.no_grad():
        q_after = net(x)

    k = len(HO_ACTIONS)
    # q 偏移 = t_a − mean(t)：切换动作相对**非切换**动作下降 log(w)，
    # 全体共享一个常数偏移 mean(t)（不影响 argmax，也不影响相对 Q 差）。
    for a in HO_ACTIONS:
        d = float((q_after[:, a] - q_before[:, a]).mean())
        assert abs(d + math.log(w) * (1 - k / n_act)) < 1e-4, (a, d)
    stay = [a for a in range(n_act) if a not in HO_ACTIONS][0]
    d_stay = float((q_after[:, stay] - q_before[:, stay]).mean())
    assert abs(d_stay - math.log(w) * k / n_act) < 1e-4
    # 真正重要的：**相对差**恰好收敛 log(w)
    rel_before = float((q_before[:, HO_ACTIONS[0]] - q_before[:, stay]).mean())
    rel_after = float((q_after[:, HO_ACTIONS[0]] - q_after[:, stay]).mean())
    assert abs((rel_after - rel_before) + math.log(w)) < 1e-4


def test_deweight_is_noop_without_weight():
    net = make_net()
    class_w = torch.ones(int(CFG["env"]["n_actions"]))
    x = torch.randn(2, int(CFG["env"]["obs_dim"]))
    with torch.no_grad():
        q_before = net(x)
    deweight_action_logits(net, class_w)
    with torch.no_grad():
        q_after = net(x)
    assert torch.allclose(q_before, q_after)


def test_deweight_returns_false_on_unknown_head():
    """拿不到 advantage 末层时必须明确返回 False（不静默改错地方）。"""

    class _Bare(torch.nn.Module):
        def forward(self, x):
            return x

    assert deweight_action_logits(_Bare(), torch.ones(7)) is False
