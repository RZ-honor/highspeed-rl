"""潜变量 ODE-RNN：把 Neural ODE 从「伪时间特征提取器」改为「真实时间动力学模型」。

背景：原实现的根本缺陷
----------------------
原 :class:`src.agents.neural_ode.NeuralODEDuelingDQN` 每次前向都执行::

    h0 = shared(x)                      # 观测编码
    h1 = RK4(odefunc, h0, τ: 0 → 1)     # 伪时间积分
    Q  = V/A(h1)

积分轴是**伪时间** τ，且每次前向都从 h(0) 重新出发——它与 episode 的
真实时间轴没有任何对应关系。其单步形式

    h + dt/6 · (k1 + 2·k2 + 2·k3 + k4)

本质上就是一个固定步长的残差块，所以整个「ODE 层」等价于一个 8 层残差 MLP。
它无法表达「未来 1 秒 RSRP 会跌破门限」这类**预测性**决策所需的能力，
而这正是高铁切换场景（350km/h、TTT 320ms、隧道突变）的核心难点。
这是所有「RL-ODE 优势」主张落空的架构根因。

重构：ODE-RNN（Rubanova et al. 2019, Latent ODEs for Irregularly-Sampled TS）
---------------------------------------------------------------------------
潜状态 h_t 沿 episode 的**真实时间轴**连续传播，跨 step 保持（不再每步重置）::

    h_t^-  = ODESolve(h_{t-1}, f_θ, Δt)      # 观测之间：沿真实时间连续演化
    h_t    = GRUCell(h_t^-,  e(x_t))         # 到达新观测：离散跳变更新
    Q(s,a) = V(h_t) + A(h_t, a) - mean(A)

并额外训练一个预测头，强迫 f_θ 学到真实的信道时间演化::

    h_{t+k} = ODESolve(h_t, f_θ, k·Δt)       # 无观测，纯动力学外推
    x̂_{t+k} = decoder(h_{t+k})
    L_dyn   = mean_k || x̂_{t+k} - x_{t+k} ||²

``L_dyn`` 是「预测式切换」得以成立的唯一途径：只有当 f_θ 真的学到了
RSRP/SINR 随时间的演化规律，外推 k 步后的隐状态才携带未来信息，
决策才能「提前于 A3 事件成立之前」发起切换。

接口契约
--------
* ``__init__(obs_dim, n_actions, hidden, ode_steps, ode_hidden, dt_s, **kwargs)``
  与 :class:`src.agents.networks.DuelingDQN` 兼容（额外参数走 config 的
  ``net_kwargs``），可直接注入 :class:`src.agents.agent.DuelingDDQNAgent`。
* ``forward(x)`` 无状态单步（h 从 0 出发），保留 DuelingDQN 的调用契约。
* 有状态路径用 :meth:`init_hidden` / :meth:`step` / :meth:`flow` / :meth:`unroll`。
"""

from typing import List, Optional

import torch
import torch.nn as nn

from src.agents.neural_ode import ODEFunc, integrate


class LatentODEQNetwork(nn.Module):
    """潜变量 ODE-RNN Dueling DQN：真实时间动力学 + 多步外推预测头。

    Parameters
    ----------
    obs_dim, n_actions, hidden :
        与 DuelingDQN 同义。
    ode_steps :
        **每个 env 步**内部的 RK4 子步数。注意：这是把一个 Δt=dt_s 的真实
        时间区间切成 n 段积分，与原实现的「伪时间 τ∈[0,1] 切 8 段」语义不同。
        一个 env 步只有 100ms，动力学变化平缓，2 个子步的 RK4 局部截断误差
        已远小于网络本身的拟合误差；取更大值只增加算力不改变精度。
    ode_hidden :
        ODE 右端函数 f_θ 的隐层宽度。
    dt_s :
        一个 env 步对应的**真实时间**（秒）。必须与环境一致，否则外推步长错误。
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128,
                 ode_steps: int = 2, ode_hidden: int = 64, dt_s: float = 0.1,
                 **kwargs):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden = int(hidden)
        self.ode_steps = int(ode_steps)
        self.dt_s = float(dt_s)

        # 观测编码（跳变分支的输入）
        self.encoder = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden), nn.ReLU(),
        )
        # 连续动力学：dh/dt = f_θ(h)，自治系统，不含显式时间项
        # （时间信息已由 dt_s 的积分步长 encode）
        self.odefunc = ODEFunc(self.hidden, int(ode_hidden))
        # 到达新观测时的跳变更新（ODE-RNN 的 RNN 部分）
        self.gru = nn.GRUCell(self.hidden, self.hidden)

        self.value = nn.Sequential(nn.Linear(self.hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(self.hidden, 64), nn.ReLU(),
                                       nn.Linear(64, self.n_actions))
        # 预测头：隐状态 → 观测空间，用于多步外推的辅助监督
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.obs_dim),
        )

    # ---------------- 基础构件 ----------------

    def init_hidden(self, batch_size: int = 1, device=None) -> torch.Tensor:
        """初始潜状态（全零）。"""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden, device=device)

    def flow(self, h: torch.Tensor, dt: Optional[float] = None) -> torch.Tensor:
        """沿真实时间演化 dt 秒：h(t+dt) = h(t) + ∫ f_θ(h) dt（RK4）。"""
        dt = self.dt_s if dt is None else float(dt)
        return integrate(self.odefunc, h, self.ode_steps, dt)

    def observe(self, h_minus: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """到达新观测时的跳变更新：h = GRUCell(e(x), h^-)。"""
        return self.gru(self.encoder(x), h_minus)

    def q(self, h: torch.Tensor) -> torch.Tensor:
        """Dueling 分解：Q = V + A - mean(A)。"""
        v = self.value(h)
        a = self.advantage(h)
        return v + a - a.mean(dim=1, keepdim=True)

    # ---------------- 对外接口 ----------------

    def step(self, h: torch.Tensor, x: torch.Tensor,
             dt: Optional[float] = None) -> torch.Tensor:
        """推进一个 env 步：先沿真实时间演化，再用新观测跳变更新。"""
        return self.observe(self.flow(h, dt), x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """无状态单步前向（h 从 0 出发），保持与 DuelingDQN 的调用契约。

        用于行为克隆等不需要跨步传播的场景；有状态路径请用 :meth:`step`。
        """
        h = self.init_hidden(x.shape[0], x.device)
        return self.q(self.step(h, x))

    def unroll(self, xs: torch.Tensor, h0: Optional[torch.Tensor] = None,
               dt: Optional[float] = None) -> torch.Tensor:
        """对序列做 burn-in，返回每一步的潜状态。

        Parameters
        ----------
        xs : Tensor [T, B, obs_dim]，时间主序。
        h0 : Tensor [B, hidden]，可选初始状态。

        Returns
        -------
        Tensor [T, B, hidden]：xs[t] 被观测之后的潜状态。
        """
        T, B = xs.shape[0], xs.shape[1]
        h = self.init_hidden(B, xs.device) if h0 is None else h0
        out = []
        for t in range(T):
            h = self.step(h, xs[t], dt)
            out.append(h)
        return torch.stack(out, dim=0)

    def rollout(self, h: torch.Tensor, k: int,
                dt: Optional[float] = None) -> List[torch.Tensor]:
        """无观测外推 k 步（纯动力学），返回每步的潜状态列表。"""
        hs = []
        cur = h
        for _ in range(int(k)):
            cur = self.flow(cur, dt)
            hs.append(cur)
        return hs

    def predict(self, h: torch.Tensor, k: int,
                dt: Optional[float] = None) -> torch.Tensor:
        """无观测外推 k 步并解码为观测预测。

        Returns
        -------
        Tensor [k, B, obs_dim]
        """
        return torch.stack([self.decoder(z) for z in self.rollout(h, k, dt)], dim=0)
