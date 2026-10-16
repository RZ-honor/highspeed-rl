"""Neural ODE 特征演化网络（演进方案 §5.1/§9 的最小可验证落地）。

将 DuelingDQN 的共享特征层扩展为连续动力学：状态特征沿伪时间 τ 演化
    dh/dτ = f_θ(h) ,  h(0) = shared(x)
用 RK4 积分到 h(1)，再走价值/优势流得到 Q(s,a)。

意义：用「连续深度网络」替代固定深度的 MLP 特征提取。在同等训练预算下，
检验「ODE 状态演化是否比固定层数 MLP 带来增益」——即演进方案主张的
「ODE 做状态动力学、离散 RL 做决策」的公平对比体。
"""

import torch
import torch.nn as nn


class ODEFunc(nn.Module):
    """ODE 右端函数：dh/dτ = f_θ(h)。小型 MLP，参数即动力学。"""

    def __init__(self, dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


def rk4_step(f: nn.Module, h: torch.Tensor, dt: float) -> torch.Tensor:
    """单步 4 阶 Runge-Kutta 积分（纯 torch，可微，误差 O(dt^4)）。"""
    dt = float(dt)
    k1 = f(h)
    k2 = f(h + 0.5 * dt * k1)
    k3 = f(h + 0.5 * dt * k2)
    k4 = f(h + dt * k3)
    return h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(f: nn.Module, h0: torch.Tensor, n_steps: int, t_span: float = 1.0) -> torch.Tensor:
    """把 h0 从 τ=0 积分到 τ=t_span，共 n_steps 步 RK4。"""
    dt = t_span / n_steps
    h = h0
    for _ in range(n_steps):
        h = rk4_step(f, h, dt)
    return h


class NeuralODEDuelingDQN(nn.Module):
    """Dueling DQN + ODE 特征演化层。

    结构与 DuelingDQN 相同，唯一差异是 shared 后插入一个 ODE 积分层：
    h0 = shared(x)；h1 = RK4(odefunc, h0)；然后 h1 进入 V/A 流。
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128,
                 ode_steps: int = 8, ode_hidden: int = 32, **kwargs):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.odefunc = ODEFunc(hidden, ode_hidden)
        self.ode_steps = ode_steps
        self.value = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, n_actions))

    def forward(self, x):
        h0 = self.shared(x)
        h1 = integrate(self.odefunc, h0, self.ode_steps)
        v = self.value(h1)
        a = self.advantage(h1)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q