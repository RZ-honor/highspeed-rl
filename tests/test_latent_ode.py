"""潜变量 ODE-RNN 的结构与动力学学习测试。

重点验证两件事：
  1. 潜状态确实**跨 step 传播**（不再是每步从 h(0) 重新积分）；
  2. ODE 确实学到了**随真实时间的演化规律**——表现为无观测外推的
     预测误差显著低于「持久性基线」（x̂_{t+k} = x_t）。
"""

import numpy as np
import torch
import torch.optim as optim

from src.agents.latent_ode import LatentODEQNetwork
from src.agents.seq_replay import EpisodeReplay


def make_net(obs_dim=2, n_actions=7, hidden=32, **kw):
    torch.manual_seed(0)
    return LatentODEQNetwork(obs_dim, n_actions, hidden, ode_hidden=32, **kw)


# ---------------- 结构 ----------------

def test_forward_shape_matches_dueling_contract():
    net = make_net(obs_dim=14, n_actions=7, hidden=64)
    q = net(torch.randn(5, 14))
    assert q.shape == (5, 7)


def test_flow_is_identity_when_field_is_zero():
    """f_θ ≡ 0 时，沿真实时间演化不改变状态（ODE 退化为恒等）。"""
    net = make_net(obs_dim=4, n_actions=3, hidden=16)
    with torch.no_grad():
        for p in net.odefunc.parameters():
            p.zero_()
    h0 = torch.randn(3, 16)
    assert torch.allclose(net.flow(h0, 0.1), h0, atol=1e-6)


def test_flow_depends_on_real_dt():
    """积分步长必须是真实时间：dt 翻倍，常量场下的位移也翻倍。"""
    net = make_net(obs_dim=2, n_actions=3, hidden=8)
    with torch.no_grad():
        net.odefunc.net[2].bias.fill_(1.0)   # 近似常量场
        net.odefunc.net[0].weight.zero_()
        net.odefunc.net[2].weight.zero_()
    h0 = torch.zeros(1, 8)
    d1 = (net.flow(h0, 0.1) - h0).abs().mean()
    d2 = (net.flow(h0, 0.2) - h0).abs().mean()
    assert d2 > d1 * 1.5, (d1.item(), d2.item())


def test_step_advances_latent_and_is_differentiable():
    """Q 路径与预测路径必须都可微，且梯度要能穿过 RK4 积分回到 f_θ。"""
    net = make_net(obs_dim=6, n_actions=5, hidden=24)
    h0 = net.init_hidden(4)
    h1 = net.step(h0, torch.randn(4, 6), 0.1)
    assert h1.shape == h0.shape
    assert not torch.allclose(h0, h1)

    # Q 路径：encoder / odefunc / gru / value / advantage 参与，decoder 不参与
    net.zero_grad()
    net.q(h1).mean().backward()
    q_path = [net.encoder, net.odefunc, net.gru, net.value, net.advantage]
    for mod in q_path:
        assert all(p.grad is not None for p in mod.parameters() if p.requires_grad), mod
    # 关键：梯度必须真正穿过 ODE 积分到达 f_θ（否则 ODE 分支是死的）
    assert net.odefunc.net[0].weight.grad.abs().sum() > 0

    # 预测路径：包括 decoder，且同样要回传到 f_θ
    # （重新前向：h1 的计算图已在上面 backward 时释放）
    net.zero_grad()
    x = torch.randn(4, 6)
    net.predict(net.step(net.init_hidden(4), x, 0.1), 5, 0.1).mean().backward()
    assert all(p.grad is not None for p in net.decoder.parameters() if p.requires_grad)
    assert net.odefunc.net[0].weight.grad.abs().sum() > 0


def test_unroll_and_predict_shapes():
    net = make_net(obs_dim=14, n_actions=7, hidden=32)
    xs = torch.randn(12, 4, 14)          # [T, B, obs_dim]
    hs = net.unroll(xs)
    assert hs.shape == (12, 4, 32)
    pred = net.predict(hs[-1], 10, 0.1)
    assert pred.shape == (10, 4, 14)


def test_latent_carries_history_same_obs_different_q():
    """结构性核心：同一观测在不同历史下应给出不同 Q。

    原实现每步从 h(0) 重新积分，Q 只依赖当前观测；潜变量版本必须打破这一点，
    否则「ODE 时间演化」就是空谈。
    """
    net = make_net(obs_dim=8, n_actions=4, hidden=32)
    x = torch.randn(1, 8)
    hist_a = torch.randn(20, 1, 8)
    hist_b = torch.randn(20, 1, 8) * 3.0
    with torch.no_grad():
        ha = net.unroll(hist_a)[-1]
        hb = net.unroll(hist_b)[-1]
        qa = net.q(net.step(ha, x, 0.1))
        qb = net.q(net.step(hb, x, 0.1))
    assert (qa - qb).abs().max() > 1e-3


# ---------------- 序列回放 ----------------

def test_episode_replay_windows_are_contiguous():
    rep = EpisodeReplay(capacity=4)
    for ep in range(3):
        for t in range(30):
            rep.push(np.array([ep * 1.0, t * 0.1], dtype=np.float32))
        rep.end_episode()
    assert len(rep) == 3
    w = rep.sample_windows(8, 18)
    assert w.shape == (8, 18, 2)
    # 窗口内时间维必须严格递增 0.1（连续性）
    d = np.diff(w[0][:, 1])
    assert np.allclose(d, 0.1, atol=1e-6)


def test_episode_replay_discards_stubs():
    rep = EpisodeReplay(capacity=4)
    rep.push(np.zeros(3, dtype=np.float32))
    rep.end_episode()
    assert len(rep) == 0


# ---------------- 动力学学习 ----------------

def _sine_dataset(n=3000, dt=0.1):
    """二维确定性周期序列：正弦/余弦，可外推。"""
    t = np.arange(n) * dt
    return np.stack([0.5 + 0.4 * np.sin(0.8 * t), 0.5 + 0.4 * np.cos(0.8 * t)],
                    axis=1).astype(np.float32)


def test_dynamics_learns_to_extrapolate_without_observations():
    """用多步外推损失训练后，预测误差须显著低于持久性基线。

    这是「预测式切换」成立的前提：外推期间不给任何观测，网络只能靠
    f_θ 积分出未来。若此测试失败，说明 ODE 没有学到动力学。
    """
    torch.manual_seed(0)
    np.random.seed(0)
    data = _sine_dataset()
    rep = EpisodeReplay(capacity=4)
    for row in data:
        rep.push(row)
    rep.end_episode()

    net = LatentODEQNetwork(obs_dim=2, n_actions=3, hidden=48,
                            ode_steps=2, ode_hidden=48, dt_s=0.1)
    opt = optim.Adam(net.parameters(), lr=3e-3)
    burn_in, horizon, n_epochs = 8, 10, 400

    def mse_and_baseline():
        with torch.no_grad():
            win = torch.as_tensor(rep.sample_windows(64, burn_in + horizon),
                                  dtype=torch.float32).transpose(0, 1)
            h = net.unroll(win[:burn_in])[-1]
            pred = net.predict(h, horizon, 0.1)
            tgt = win[burn_in:burn_in + horizon]
            m = float(((pred - tgt) ** 2).mean())
            b = float(((win[burn_in - 1].unsqueeze(0).expand_as(tgt) - tgt) ** 2).mean())
        return m, b

    m0, b0 = mse_and_baseline()
    for _ in range(n_epochs):
        win = torch.as_tensor(rep.sample_windows(32, burn_in + horizon),
                              dtype=torch.float32).transpose(0, 1)
        h = net.unroll(win[:burn_in])[-1]
        loss = ((net.predict(h, horizon, 0.1) - win[burn_in:burn_in + horizon]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    m1, b1 = mse_and_baseline()

    assert m1 < m0 * 0.5, f"动力学未收敛: {m0:.4f} -> {m1:.4f}"
    assert m1 < b1 * 0.5, f"未超越持久性基线: model={m1:.4f} baseline={b1:.4f}"
