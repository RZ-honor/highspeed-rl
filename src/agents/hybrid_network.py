"""混合系统 Dueling DQN：ODE 分支（连续区间）+ MLP 分支（跳变事件），软路由融合。

演进方案 §5.2 的工程化落地：
  - 平稳区间（无切换/隧道）→ ODE 分支主导（α→0），连续动力学更自然
  - 跳变区间（切换中/进出隧道）→ MLP 分支主导（α→1），离散决策更直接
  - 路由权重 α 由网络自学习（不是固定规则）

架构：
  shared(x) → h0
  h_ode = RK4(odefunc, h0)    (连续动力学)
  h_mlp = mlp_refine(h0)      (跳变/离散)
  α = σ(route(h0))             (软切换, α∈[0,1])
  h = (1-α)·h_ode + α·h_mlp   (混合特征)
  Q = V(h) + A(h) - mean(A)   (Dueling 输出)
"""

import torch
import torch.nn as nn

from src.agents.neural_ode import ODEFunc, integrate


class HybridDuelingDQN(nn.Module):
    """ODE+MLP 混合 Dueling DQN，软路由自学习分支权重。

    与 DuelingDQN / NeuralODEDuelingDQN 接口完全一致：
    __init__(obs_dim, n_actions, hidden) → nn.Module,
    forward(x: [B, obs_dim]) → [B, n_actions]。
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128,
                 ode_steps: int = 8, ode_hidden: int = 32, **kwargs):
        super().__init__()
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # ODE 分支：连续动力学特征演化
        self.odefunc = ODEFunc(hidden, ode_hidden)
        self.ode_steps = ode_steps
        # MLP 分支：跳变/离散事件处理
        self.mlp_refine = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # 路由网络：输出 α∈[0,1]，α→0 ODE 主导，α→1 MLP 主导
        self.route = nn.Sequential(
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )
        # Dueling 分解
        self.value = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, n_actions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = self.shared(x)
        # ODE 分支
        h_ode = integrate(self.odefunc, h0, self.ode_steps)
        # MLP 分支
        h_mlp = self.mlp_refine(h0)
        # 路由权重
        alpha = self.route(h0)  # [B, 1]
        # 混合特征
        h = (1.0 - alpha) * h_ode + alpha * h_mlp
        # Dueling 分解
        v = self.value(h)
        a = self.advantage(h)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q

    def route_weight(self, x: torch.Tensor) -> torch.Tensor:
        """返回路由权重 α（用于统计，不参与梯度）。"""
        with torch.no_grad():
            h0 = self.shared(x)
            return self.route(h0).squeeze(-1)  # [B]
