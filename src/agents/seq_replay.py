"""按 episode 存储观测序列，为潜变量动力学损失提供多步预测窗口。

动机
----
潜变量 ODE 的动力学损失需要「未来 k 步的真实观测」作为监督目标：

    L_dyn = Σ_k || decoder(ODESolve(h_t, f_θ, k·Δt)) - x_{t+k} ||²

而 PER 回放缓冲是**逐转移随机采样**的，不保留时序邻接关系，无法提供
x_{t+k}。本模块用一个独立的、按 episode 组织的环形缓冲补齐这一能力。

与 PER 的职责划分（刻意解耦，互不干扰）：
  * PrioritizedReplayBuffer —— RL 的 TD 损失，保留优先级机制
  * EpisodeReplay          —— 动力学的预测损失，均匀采样窗口

内存开销：600 步 × 16 维 × float32 ≈ 38 KB/episode；capacity=200 → ~7.6 MB。
"""

from collections import deque
from typing import Optional

import numpy as np


class EpisodeReplay:
    """episode 级观测序列环形缓冲。

    Parameters
    ----------
    capacity :
        最多保留的 episode 数（超出后丢弃最旧的）。
    """

    def __init__(self, capacity: int = 200):
        self.capacity = int(capacity)
        self._episodes = deque(maxlen=self.capacity)
        self._cur: list = []

    # ---------------- 写入 ----------------

    def push(self, obs) -> None:
        """追加一步观测到当前 episode。"""
        self._cur.append(np.asarray(obs, dtype=np.float32))

    def end_episode(self) -> None:
        """结束当前 episode；长度过短（<2）的片段直接丢弃。"""
        if len(self._cur) >= 2:
            self._episodes.append(np.stack(self._cur, axis=0))
        self._cur = []

    def clear(self) -> None:
        self._episodes.clear()
        self._cur = []

    # ---------------- 读取 ----------------

    def __len__(self) -> int:
        return len(self._episodes)

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    def sample_windows(self, n: int, window_len: int) -> Optional[np.ndarray]:
        """采样 n 个长度为 window_len 的连续观测窗口。

        Returns
        -------
        np.ndarray [n, window_len, obs_dim]；若数据不足返回 None。
        """
        if not self._episodes or n <= 0 or window_len < 2:
            return None
        obs_dim = self._episodes[0].shape[1]
        out = np.zeros((n, window_len, obs_dim), dtype=np.float32)
        for i in range(n):
            ep_idx = int(np.random.randint(len(self._episodes)))
            ep = self._episodes[ep_idx]
            # 窗口不足时从开头对齐（短 episode 也能贡献样本，只是一次性偏向前段）
            start_max = max(0, ep.shape[0] - window_len)
            start = int(np.random.randint(start_max + 1)) if start_max > 0 else 0
            seg = ep[start:start + window_len]
            out[i, :seg.shape[0]] = seg
            if seg.shape[0] < window_len:
                # 右填充：复制末帧，避免引入 0 值伪样本
                out[i, seg.shape[0]:] = seg[-1]
        return out
