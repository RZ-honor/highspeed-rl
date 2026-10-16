import numpy as np
import torch
from src.utils.config import load_config
from src.training.device import set_seed
from src.envs.channel_3gpp import Channel3GPP
from src.envs.mobility import Mobility
from src.envs.handover_env import HandoverEnv
from src.agents.agent import DuelingDDQNAgent
from src.agents.baselines import A3Baseline


def test_full_episode_rl_and_a3():
    cfg = load_config("configs/default.yaml")
    cfg["env"]["episode_steps"] = 30
    cfg["agent"]["buffer_size"] = 300
    set_seed(0)
    ch = Channel3GPP(cfg)
    mo = Mobility(cfg, seed=0)
    env = HandoverEnv(cfg, ch, mo)
    agent = DuelingDDQNAgent(cfg, torch.device("cpu"))

    obs, info = env.reset(seed=0)
    for t in range(30):
        a = agent.act(obs, info, eps=1.0)
        nobs, r, term, trunc, ninfo = env.step(a)
        agent.store(obs, a, r, nobs, term)
        agent.learn()
        obs, info = nobs, ninfo
        if term or trunc:
            break

    # A3
    obs, info = env.reset(seed=0)
    a3 = A3Baseline(cfg["safety"])
    for t in range(30):
        a = a3.act(obs, info)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    # 只要不抛异常即通过
    assert True
