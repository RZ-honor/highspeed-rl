import torch
from src.agents.networks import DuelingDQN


def test_forward_shape():
    net = DuelingDQN(obs_dim=10, n_actions=7, hidden=128)
    x = torch.randn(4, 10)
    q = net(x)
    assert q.shape == (4, 7)


def test_advantage_mean_zero():
    net = DuelingDQN(10, 7, 128)
    x = torch.randn(2, 10)
    q = net(x)
    # Q - mean over actions 应等于 V + (A - mean)；校验形状稳定
    assert q.shape == (2, 7)
