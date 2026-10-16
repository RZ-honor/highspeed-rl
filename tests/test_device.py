import os

import torch

from src.training.device import (configure_torch_threads, get_device,
                                 set_seed, torch_thread_cap)


def test_device_auto():
    dev = get_device("auto")
    assert dev.type in ("cuda", "cpu")


def test_seed_repro():
    set_seed(0)
    a = torch.randn(3)
    set_seed(0)
    b = torch.randn(3)
    assert torch.equal(a, b)


def test_thread_cap_bounded(monkeypatch):
    """众核主机上必须封顶：32 线程会让小网络训练劣化两个数量级。"""
    monkeypatch.delenv("HSR_TORCH_THREADS", raising=False)
    n = configure_torch_threads()
    assert 1 <= n <= max(1, os.cpu_count() or 1)
    assert n <= torch_thread_cap()
    assert n <= 8


def test_thread_cap_env_override(monkeypatch):
    monkeypatch.setenv("HSR_TORCH_THREADS", "2")
    assert configure_torch_threads() == 2
    assert configure_torch_threads(1) == 1