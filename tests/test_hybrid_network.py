"""HybridDuelingDQN 混合网络测试。"""

import torch
from src.agents.hybrid_network import HybridDuelingDQN


def test_hybrid_forward_shape():
    net = HybridDuelingDQN(obs_dim=10, n_actions=7, hidden=128)
    x = torch.randn(4, 10)
    q = net(x)
    assert q.shape == (4, 7)


def test_hybrid_is_differentiable():
    net = HybridDuelingDQN(obs_dim=10, n_actions=7, hidden=128)
    x = torch.randn(2, 10, requires_grad=True)
    q = net(x)
    loss = q.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_route_weight_range():
    net = HybridDuelingDQN(obs_dim=10, n_actions=7, hidden=128)
    x = torch.randn(8, 10)
    alpha = net.route_weight(x)
    assert alpha.shape == (8,)
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0


def test_hybrid_vs_ode_mlp_endpoints():
    """当 route 偏向 0 → 近似纯 ODE；偏向 1 → 近似纯 MLP。"""
    net = HybridDuelingDQN(obs_dim=10, n_actions=7, hidden=128)
    x = torch.randn(2, 10)
    # 正常前向：α 在 (0,1) 之间
    q_normal = net(x)
    assert q_normal.shape == (2, 7)
    # 极端情况：强制 route 输出 0 或 1
    with torch.no_grad():
        # route = Linear+Sigmoid，设 bias=-100 → sigmoid≈0 (ODE 主导)
        net.route[0].bias.fill_(-100.0)
    q_ode_like = net(x)
    with torch.no_grad():
        # route = Linear+Sigmoid，设 bias=+100 → sigmoid≈1 (MLP 主导)
        net.route[0].bias.fill_(100.0)
    q_mlp_like = net(x)
    # 两者应产生不同输出（不同分支权重）
    assert not torch.allclose(q_ode_like, q_mlp_like, atol=1e-4)
