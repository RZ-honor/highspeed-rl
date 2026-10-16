import numpy as np
import torch
from src.utils.config import load_config
from src.agents.replay import PrioritizedReplayBuffer

CFG = load_config("configs/default.yaml")


def make_buf(size=100):
    cfg = dict(CFG)
    cfg["agent"]["buffer_size"] = size
    return PrioritizedReplayBuffer(cfg["agent"])


def test_add_and_sample_shapes():
    b = make_buf()
    for _ in range(120):
        b.add(np.zeros(10, dtype=np.float32), 0, 1.0, np.zeros(10, dtype=np.float32), False)
    batch, idx, w = b.sample(16)
    assert batch["state"].shape == (16, 10)
    assert batch["action"].shape == (16,)
    assert w.shape == (16,)
    w_np = w.numpy()
    assert np.all((w_np > 0) & (w_np <= 1.0 + 1e-5))


def test_priority_affects_sampling():
    b = make_buf(size=50)
    b.add(np.zeros(10, dtype=np.float32), 0, 1.0, np.zeros(10, dtype=np.float32), False)
    b.add(np.ones(10, dtype=np.float32), 1, 1.0, np.ones(10, dtype=np.float32), False)
    b.update_priorities(np.array([0, 1]), np.array([0.01, 100.0]))
    counts = {0: 0, 1: 0}
    for _ in range(200):
        _, idx, _ = b.sample(1)
        counts[int(idx[0])] += 1
    assert counts[1] > counts[0]  # 高优先级采样更多
