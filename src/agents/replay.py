from collections import deque

import numpy as np
import torch


def _as_flat_float(x) -> torch.Tensor:
    """把潜状态（torch.Tensor，可能在 GPU；或 numpy 数组）压成 1 维 float32。"""
    if isinstance(x, torch.Tensor):
        return x.detach().to(torch.float32).reshape(-1)
    return torch.as_tensor(np.asarray(x, dtype=np.float32)).reshape(-1)


class SumTree:
    """Sum-tree 优先级存储：叶子存优先级，内部存子树和。"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, delta: float):
        while idx > 0:
            idx = (idx - 1) // 2
            self.tree[idx] += delta

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def get(self, s: float):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """PER 回放缓冲。

    Parameters
    ----------
    cfg :
        ``cfg["agent"]`` 字典。
    store_latents :
        为 True 时每次转移额外保存一对潜状态 ``(h_t, h_{t+1})``，
        供潜变量网络（:class:`src.agents.latent_ode.LatentODEQNetwork`）
        在随机采样下仍能取到对应的隐状态。非潜变量网络保持 False，
        元组结构与采样返回字段均不变。
    """

    def __init__(self, cfg: dict, store_latents: bool = False):
        self.capacity = int(cfg["buffer_size"])
        self.alpha = float(cfg["per_alpha"])
        self._beta0 = float(cfg["per_beta_start"])
        self._beta1 = float(cfg["per_beta_end"])
        self.beta = self._beta0
        self.eps = float(cfg["per_epsilon"])
        self.tree = SumTree(self.capacity)
        self.store_latents = bool(store_latents)
        self.gamma = float(cfg.get("gamma", 1.0))
        # Side arrays preserve the legacy five-/seven-field replay tuples.
        self.discounts = np.zeros(self.capacity, dtype=np.float32)
        self.lengths = np.ones(self.capacity, dtype=np.int64)

    def __len__(self):
        return self.tree.n_entries

    def add(self, state, action, reward, next_state, done,
            latent=None, next_latent=None, discount=None, n_steps=1):
        # 新样本给当前最大优先级，保证至少被采样一次
        max_p = float(np.max(self.tree.tree[-self.capacity:])) if self.tree.n_entries > 0 else 1.0
        if max_p == 0.0:
            max_p = 1.0
        if self.store_latents:
            data = (state, action, reward, next_state, done, latent, next_latent)
        else:
            data = (state, action, reward, next_state, done)
        slot = self.tree.write
        self.discounts[slot] = (0.0 if done else
                                self.gamma ** n_steps if discount is None else discount)
        self.lengths[slot] = n_steps
        self.tree.add(max_p, data)

    def sample(self, batch_size: int):
        total = self.tree.total()
        if total <= 0:
            total = 1e-6
        seg = total / batch_size
        slots, priorities, states, actions, rewards, nexts, dones = [], [], [], [], [], [], []
        latents, next_latents = [], []
        for i in range(batch_size):
            s = np.random.uniform(seg * i, seg * (i + 1))
            leaf_idx, p, data = self.tree.get(s)
            # 缓冲未填满时可能采到零优先级叶 → 回退到已写入槽
            if data is None and self.tree.n_entries > 0:
                slot = int(np.random.randint(self.tree.n_entries))
                leaf_idx = slot + self.capacity - 1
                p = self.tree.tree[leaf_idx]
                data = self.tree.data[slot]
            slots.append(leaf_idx - self.capacity + 1)  # 返回数据槽索引
            priorities.append(p)
            states.append(data[0]); actions.append(data[1]); rewards.append(data[2])
            nexts.append(data[3]); dones.append(data[4])
            if self.store_latents:
                latents.append(data[5]); next_latents.append(data[6])
        slots = np.array(slots, dtype=np.int64)
        probs = np.array(priorities, dtype=np.float64) / max(total, 1e-6)
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        weights = weights / (weights.max() + 1e-8)
        batch = {
            "state": torch.as_tensor(np.array(states), dtype=torch.float32),
            "action": torch.as_tensor(np.array(actions), dtype=torch.int64),
            "reward": torch.as_tensor(np.array(rewards), dtype=torch.float32),
            "next_state": torch.as_tensor(np.array(nexts), dtype=torch.float32),
            "done": torch.as_tensor(np.array(dones), dtype=torch.float32),
            "discount": torch.as_tensor(self.discounts[slots].copy()),
            "n_steps": torch.as_tensor(self.lengths[slots].copy()),
        }
        if self.store_latents:
            # 潜状态入池时形如 [1, hidden]（act 为单步批），这里压成 [B, hidden]。
            # 注意：潜状态可能驻留在 GPU 上，不能走 numpy 转换。
            batch["latent"] = torch.stack([_as_flat_float(l) for l in latents])
            batch["next_latent"] = torch.stack([_as_flat_float(l) for l in next_latents])
        return batch, slots, torch.as_tensor(weights, dtype=torch.float32)

    def update_priorities(self, idxs, td_errors):
        """idxs 为 sample 返回的数据槽索引（0-based）。"""
        td = np.abs(np.asarray(td_errors, dtype=np.float64))
        for i, slot in enumerate(idxs):
            leaf_idx = int(slot) + self.capacity - 1
            p = (td[i] + self.eps) ** self.alpha
            self.tree.update(leaf_idx, float(p))

    def anneal_beta(self, frac: float):
        self.beta = float(self._beta0 + (self._beta1 - self._beta0) * frac)


class NStepAccumulator:
    """n 步回报累积器（设计方案 §4.3 P2）。

    动机
    ----
    切换/中断的后果要延迟数步才在吞吐上显现：HO 中断本身 12-40ms，切换后
    新链路的信道质量还需数步才能稳定评估。单步 TD 的信用分配链路是
    ``Q(s_t,a_t) ← r_t + γ·Q(s_{t+1},a_{t+1})``，每多走一步就多一次
    bootstrap 误差；把 n 步折扣真实回报一次性写进 target，可把 bootstrap
    次数从 1 降到 1/n，从而实质加速收敛（文档原文：「改动仅限回放缓冲与
    目标计算」）。

    实现
    ----
    维护长度 n 的滑动窗口，每 push 一条转移就吐出**一条** n 步折扣转移：

        R_t^(n) = Σ_{k=0}^{n-1} γ^k · r_{t+k}
        y       = R_t^(n) + γ^n · Q_target(s_{t+n}, a*) · (1 - done)

    * 窗口内出现 ``done`` 即截断累积（超出 episode 边界的奖励不得混入），
      此时 ``next_state`` 取该 done 步的 ``s'``，「有效 n」自然缩短。
    * 滑动窗口逐条 pop，每条转移仍被完整消费一次，缓冲样本量与 n=1 时
      相同（PER 优先级按 n 步 TD 误差更新，语义自洽）。
    * ``n=1`` 退化为原单步直通行为，可作回退开关。

    潜变量网络额外携带 ``(h_t, h_{t+1})``：窗口末端记录的 ``next_latent``
    即 ``h_{t+n}``，与 ``next_state = s_{t+n}`` 严格对应。

    **必须在 episode 边界调用** :meth:`flush`（见 ``DuelingDDQNAgent
    .flush_nstep``）：截断 episode（``truncated`` 而 ``done=False``）的窗口
    若不清空，会把上一集的奖励带进下一集的轨迹。
    """

    def __init__(self, n: int, gamma: float, buffer: PrioritizedReplayBuffer,
                 store_latents: bool = False):
        self.n = max(1, int(n))
        self.gamma = float(gamma)
        self.buffer = buffer
        self.store_latents = bool(store_latents)
        self._q = deque()

    def __len__(self) -> int:
        return len(self._q)

    @property
    def enabled(self) -> bool:
        return self.n > 1

    def push(self, s, a, r, s2, done, latent=None, next_latent=None) -> None:
        done = bool(done)
        self._q.append((s, a, r, s2, done, latent, next_latent))
        if len(self._q) >= self.n:
            # 满窗：正常吐出，窗口内若无 done 则保留 bootstrap（done_last=False）。
            self._emit()
        if done:
            # episode 正常结束：剩余短窗口用真实 done 收尾。
            self.flush()

    def flush(self) -> None:
        """Emit short windows with their actual length and terminal flag.

        Time limits retain bootstrap at the last valid observation with gamma**K.
        Only a real terminal disables bootstrap.
        """
        while self._q:
            self._emit()

    def discard(self) -> None:
        """Drop pending windows after an invalid/failed trajectory boundary."""
        self._q.clear()

    def _emit(self) -> None:
        s0, a0 = self._q[0][0], self._q[0][1]
        latent0 = self._q[0][5]
        ret = 0.0
        s2_last = self._q[0][3]
        next_latent_last = self._q[0][6]
        done_last = False
        for k, item in enumerate(self._q):
            ret += (self.gamma ** k) * float(item[2])
            s2_last = item[3]
            if self.store_latents:
                next_latent_last = item[6]
            if item[4]:
                done_last = True
                break
        length = k + 1
        self.buffer.add(s0, a0, ret, s2_last, done_last,
                        latent=latent0, next_latent=next_latent_last,
                        discount=0.0 if done_last else self.gamma ** length,
                        n_steps=length)
        self._q.popleft()
