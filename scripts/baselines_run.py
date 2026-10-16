import numpy as np

from src.utils.config import load_config
from src.training.device import set_seed
from src.envs.channel_3gpp import Channel3GPP
from src.envs.mobility import Mobility
from src.envs.handover_env import HandoverEnv
from src.agents.baselines import (
    A3Baseline, GreedyBaseline, FixedIntervalBaseline, TabularQAgent
)


def _eval_one(cfg, policy, n_eps):
    ch = Channel3GPP(cfg)
    mo = Mobility(cfg, seed=cfg.get("seed", 42))
    env = HandoverEnv(cfg, ch, mo)
    tps = []
    for ep in range(n_eps):
        obs, info = env.reset(seed=cfg.get("seed", 42) + 2000 + ep)
        ep_tp, steps = 0.0, 0
        for t in range(int(cfg["env"]["episode_steps"])):
            action = policy.act(obs, info)
            obs, r, term, trunc, ninfo = env.step(action)
            ep_tp += ninfo.get("throughput_mbps", 0.0)
            steps += 1
            if term or trunc:
                break
        tps.append(ep_tp / max(steps, 1))
    return float(np.mean(tps))


def run(cfg: dict) -> dict:
    set_seed(cfg.get("seed", 42))
    n = int(cfg["eval"]["n_episodes"])
    results = {}
    results["A3"] = _eval_one(cfg, A3Baseline(cfg["safety"]), n)
    results["greedy"] = _eval_one(cfg, GreedyBaseline(), n)
    results["fixed"] = _eval_one(cfg, FixedIntervalBaseline(10), n)
    # 表格 Q：在线交互训练（短）
    tq = TabularQAgent()
    ch = Channel3GPP(cfg)
    mo = Mobility(cfg, seed=cfg.get("seed", 42))
    env = HandoverEnv(cfg, ch, mo)
    for ep in range(n):
        obs, info = env.reset(seed=cfg.get("seed", 42) + 3000 + ep)
        for t in range(int(cfg["env"]["episode_steps"])):
            a = tq.act(obs, info)
            nobs, r, term, trunc, ninfo = env.step(a)
            tq.update(obs, a, r, nobs, term)
            obs, info = nobs, ninfo
            if term or trunc:
                break
    results["tabular_q"] = _eval_one(cfg, tq, n)
    return results


if __name__ == "__main__":
    print(run(load_config("configs/default.yaml")))
