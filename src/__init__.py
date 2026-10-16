"""highspeed-rl 源码包。

导入本包会顺带把 torch 的 intra-op 线程数限制到合理范围，
避免众核主机（如云端海光容器）上小网络训练出现数量级劣化，
详见 `src/training/device.py::configure_torch_threads`。
"""

try:  # 纯规则类脚本不一定装了 torch
    from src.training.device import configure_torch_threads

    configure_torch_threads()
except Exception:  # pragma: no cover - 环境缺失时静默降级
    pass
