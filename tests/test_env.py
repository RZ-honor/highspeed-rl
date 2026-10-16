import numpy as np
from src.utils.config import load_config
from src.envs.channel_3gpp import Channel3GPP
from src.envs.mobility import Mobility
from src.envs.handover_env import HandoverEnv

CFG = load_config("configs/default.yaml")


def make_env(seed=0):
    cfg = dict(CFG)
    cfg["seed"] = seed
    ch = Channel3GPP(cfg)
    mo = Mobility(cfg, seed=seed)
    return HandoverEnv(cfg, ch, mo)


def test_reset_shapes():
    env = make_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (CFG["env"]["obs_dim"],)
    assert np.all((obs >= 0) & (obs <= 1.0 + 1e-6))
    assert "candidate_rsrp" in info and "serving_idx" in info
    assert "serving_sinr" in info


def test_step_stay():
    env = make_env()
    env.reset(seed=0)
    obs, r, term, trunc, info = env.step(0)  # a0 = 保持
    assert obs.shape == (14,)
    assert np.isfinite(r)
    assert "pingpong" in info


def test_action_space():
    env = make_env()
    assert env.action_space.n == CFG["env"]["n_actions"]


def test_episode_terminates():
    env = make_env()
    env.reset(seed=0)
    steps = 0
    for _ in range(CFG["env"]["episode_steps"] + 5):
        s, r, term, trunc, info = env.step(0)
        steps += 1
        if term or trunc:
            break
    assert steps <= CFG["env"]["episode_steps"] + 1


def test_ho_interruption_window_decays_throughput():
    """切换中断窗 g(τ) 必须在真实中断参数下起作用（设计方案 §1.4）。

    反证原实现：中断 12-40ms << 步长 100ms ⇒ K=ceil(t_int/Δt)≡1，τ 从 1/K=1
    起步 ⇒ sin(π·1)=0 ⇒ g≡1，中断窗完全失效（切换零代价）。修正后 τ 取步中点
    0.5、并按 duty-cycle frac=t_int/Δt 折算，40ms 中断 → 本步损失 A_HO×0.4。
    """
    cfg = load_config("configs/default.yaml")
    cfg["env"]["ho_interruption_ms_min"] = 40
    cfg["env"]["ho_interruption_ms_max"] = 40
    cfg["seed"] = 0
    env = HandoverEnv(cfg, Channel3GPP(cfg), Mobility(cfg, seed=0))
    env.reset(seed=0)
    _, _, _, _, info = env.step(1)          # a1 = HO→最强邻
    assert info["ho_event"] is True
    # K = ceil(40/100) = 1 → τ = (1-0.5)/1 = 0.5，frac = 0.4
    assert np.isclose(info["ho_tau"], 0.5)
    assert np.isclose(info["ho_decay"], 1.0 - 0.8 * 0.4)

    # 窗口只有 1 步：下一步应恢复无衰减
    _, _, _, _, info2 = env.step(0)
    assert info2["ho_decay"] == 1.0
    assert info2["ho_tau"] == 0.0
