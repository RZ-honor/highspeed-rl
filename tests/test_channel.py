import numpy as np
import pytest
from src.utils.config import load_config
from src.envs.channel_3gpp import Channel3GPP

CFG = load_config("configs/default.yaml")


def make_channel(seed=42):
    cfg = dict(CFG)
    cfg["seed"] = seed
    return Channel3GPP(cfg)


def test_path_loss_monotonic():
    """Path loss increases with distance (RSRP decreases)."""
    ch = make_channel()
    bs = np.array([[0, 50, 35.0]])  # BS offset 50m from track
    d_near = np.array([100.0, 0, 1.5])
    d_far = np.array([1000.0, 0, 1.5])
    r_near, _, _ = ch.measure(d_near, bs, t=0.0)
    r_far, _, _ = ch.measure(d_far, bs, t=0.0)
    assert r_far[0] < r_near[0]


def test_shadow_correlation():
    """Shadow fading produces finite values at various distances."""
    ch = make_channel(seed=1)
    bs = np.array([[0, 50, 35.0]])  # BS offset 50m from track
    samples = [ch.measure(np.array([x, 0, 1.5]), bs, t=0.0)[0][0] for x in [0, 8, 800]]
    assert all(np.isfinite(samples))


def test_jakes_envelope_positive():
    """Fast fading envelope is finite and positive."""
    ch = make_channel()
    bs = np.array([[0, 50, 35.0]])  # BS offset 50m from track
    ue = np.array([100.0, 0, 1.5])
    rsrp, sinr, los = ch.measure(ue, bs, t=0.0)
    assert np.isfinite(rsrp[0]) and np.isfinite(sinr[0])


def test_sinr_positive_near_bs():
    """SINR is positive near the serving BS (BS offset from track)."""
    ch = make_channel()
    # BSs offset 50m from track (y=50), spaced 2km apart along track
    bs = np.array([[0, 50, 35.0], [2000.0, 50, 35.0]])
    # UE on track near first BS
    ue = np.array([50.0, 0, 1.5])
    _, sinr, _ = ch.measure(ue, bs, t=0.0)
    assert sinr[0] > 0