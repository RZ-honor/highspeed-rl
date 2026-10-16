import os
import random
from typing import Optional

import numpy as np
import torch

# 众核主机（海光 DCU 容器可见 32 核）上，torch 默认把 intra-op 线程开到
# 核数；本项目是小网络 + 小 batch，OpenMP 反复同步反而造成数量级劣化。
# 实测同一段潜变量 ODE 训练循环（40 epoch）：
#     32 线程 306.9s / 8 线程 3.6s / 4 线程 2.9s / 1 线程 2.6s
# 因此这里默认给线程数封顶，并允许用环境变量覆盖：
#     export HSR_TORCH_THREADS=4
_THREAD_ENV = "HSR_TORCH_THREADS"
_DEFAULT_CAP = 8          # 普通多核机器
_MANYCORE_CAP = 4         # 众核机器 (>= _MANYCORE_AT 核)
_MANYCORE_AT = 16


def torch_thread_cap() -> int:
    n = int(os.cpu_count() or 1)
    return _MANYCORE_CAP if n >= _MANYCORE_AT else min(_DEFAULT_CAP, n)


def configure_torch_threads(explicit: Optional[int] = None) -> int:
    """把 torch intra-op 线程数限制在合理范围，返回生效值。"""
    if explicit is None:
        env = os.environ.get(_THREAD_ENV, "").strip()
        explicit = int(env) if env.isdigit() else None
    target = int(explicit) if explicit else torch_thread_cap()
    target = max(1, min(int(target), int(os.cpu_count() or 1)))
    try:
        torch.set_num_threads(target)
    except Exception:      # 某些构建不支持动态调整
        return torch.get_num_threads()
    return torch.get_num_threads()


configure_torch_threads()


def get_device(pref: str = "auto") -> torch.device:
    configure_torch_threads()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


def set_seed(seed: int) -> None:
    configure_torch_threads()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)