import torch
import torch.nn as nn

from src.agents.neural_ode import (
    ODEFunc,
    rk4_step,
    integrate,
    NeuralODEDuelingDQN,
)


class _ConstField(nn.Module):
    """dh/dt = c（常量场），RK4 应精确复现 h0 + c·dt。"""

    def __init__(self, c):
        super().__init__()
        self.c = torch.tensor(c, dtype=torch.float32)

    def forward(self, h):
        return self.c.expand_as(h)


def test_rk4_const_field_exact():
    """常量场 dh/dt = c，解析解 h0 + c·dt，RK4 应精确（误差 0）。"""
    f = _ConstField([1.0, 2.0, 0.5, 0.25])
    h0 = torch.tensor([0.0, 1.0, 2.0, 3.0])
    dt = 0.5
    h_rk = rk4_step(f, h0, dt)
    true = h0 + f.c * dt
    assert torch.allclose(h_rk, true, atol=1e-5)


def test_integrate_matches_analytic():
    """多步积分常量场：h(t) = h0 + c·t，与解析解一致。"""
    f = _ConstField([1.0, -1.0])
    h0 = torch.tensor([0.0, 0.0])
    h_int = integrate(f, h0, n_steps=8, t_span=1.0)
    true = h0 + f.c * 1.0
    assert torch.allclose(h_int, true, atol=1e-4)


def test_ode_dqn_forward_shape():
    net = NeuralODEDuelingDQN(obs_dim=10, n_actions=7, hidden=128, ode_steps=8)
    x = torch.randn(4, 10)
    q = net(x)
    assert q.shape == (4, 7)


def test_ode_dqn_is_differentiable():
    net = NeuralODEDuelingDQN(10, 7, 128, ode_steps=8)
    x = torch.randn(3, 10)
    q = net(x).mean()
    q.backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_ode_dqn_vs_plain_matches_with_zero_ode():
    """ODE 层退化为恒等（f=0）时，输出应与无 ODE 演化的 DuelingDQN 一致。"""
    from src.agents.networks import DuelingDQN

    torch.manual_seed(0)
    plain = DuelingDQN(10, 7, 128)
    torch.manual_seed(0)
    ode_net = NeuralODEDuelingDQN(10, 7, 128, ode_steps=8)
    with torch.no_grad():
        for p in ode_net.odefunc.parameters():
            p.zero_()
    ode_net.shared.load_state_dict(plain.shared.state_dict())
    ode_net.value.load_state_dict(plain.value.state_dict())
    ode_net.advantage.load_state_dict(plain.advantage.state_dict())

    x = torch.randn(5, 10)
    q_plain = plain(x)
    q_ode = ode_net(x)
    assert torch.allclose(q_ode, q_plain, atol=1e-4), (q_ode - q_plain).abs().max()