import torch.nn as nn


class DuelingDQN(nn.Module):
    """共享特征层 + 价值流 V(s) + 优势流 A(s,a)；Q = V + A - mean(A)。"""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128, **kwargs):
        """``**kwargs`` 吸收 config 的 ``net_kwargs``（本网络不使用），
        使所有网络工厂拥有统一签名、可互换注入 agent。"""
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, n_actions))

    def forward(self, x):
        h = self.shared(x)
        v = self.value(h)
        a = self.advantage(h)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q
