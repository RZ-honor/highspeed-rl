import numpy as np

from src.utils.config import load_config
from src.training.device import get_device, set_seed
from src.envs.channel_3gpp import Channel3GPP
from src.envs.mobility import Mobility
from src.envs.handover_env import HandoverEnv
from src.agents.agent import DuelingDDQNAgent


def evaluate(cfg: dict, ckpt_path: str) -> dict:
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    ch = Channel3GPP(cfg)
    mo = Mobility(cfg, seed=cfg.get("seed", 42))
    env = HandoverEnv(cfg, ch, mo)
    agent = DuelingDDQNAgent(cfg, device)
    agent.load(ckpt_path)
    safety = agent.safety

    n = int(cfg["eval"]["n_episodes"])
    rec = {"throughput": [], "ho_count": [], "pingpong": [], "outage": [],
           "dual_usage": [], "fb_rate": []}
    for ep in range(n):
        obs, info = env.reset(seed=cfg.get("seed", 42) + 1000 + ep)
        ep_tp, ho, pp, out, dual_steps, fb = 0.0, 0, 0, 0, 0, 0
        steps = 0
        for t in range(int(cfg["env"]["episode_steps"])):
            action = agent.act(obs, info, eps=0.0)
            # 3GPP A3 兜底
            if safety.should_fallback(info["serving_sinr"], action):
                action = safety.a3_action(info)
                fb += 1
            obs, r, term, trunc, ninfo = env.step(action)
            ep_tp += ninfo.get("throughput_mbps", 0.0)
            if ninfo.get("ho_event"):
                ho += 1
            if ninfo.get("pingpong"):
                pp += 1
            if len(ninfo.get("active_links", [])) > 1:
                dual_steps += 1
            steps += 1
            if term:
                out += 1
            if term or trunc:
                break
        rec["throughput"].append(ep_tp / max(steps, 1))
        rec["ho_count"].append(ho)
        rec["pingpong"].append(pp)
        rec["outage"].append(out)
        rec["dual_usage"].append(dual_steps / max(steps, 1))
        rec["fb_rate"].append(fb / max(steps, 1))
    return {k: float(np.mean(v)) for k, v in rec.items()}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default="checkpoints/latest.pt")
    a = p.parse_args()
    print(evaluate(load_config(a.config), a.checkpoint))
