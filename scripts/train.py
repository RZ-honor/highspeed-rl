import os

from src.utils.config import load_config
from src.training.device import get_device, set_seed
from src.envs.channel_3gpp import Channel3GPP
from src.envs.mobility import Mobility
from src.envs.handover_env import HandoverEnv
from src.agents.agent import DuelingDDQNAgent


def run(cfg: dict, ckpt_path: str):
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "auto"))
    n_eps = int(cfg["training"]["n_episodes"])
    warmup = int(cfg["training"]["warmup_steps"])

    ch = Channel3GPP(cfg)
    mo = Mobility(cfg, seed=cfg.get("seed", 42))
    env = HandoverEnv(cfg, ch, mo)
    agent = DuelingDDQNAgent(cfg, device)
    safety = agent.safety

    for ep in range(n_eps):
        obs, info = env.reset(seed=cfg.get("seed", 42) + ep)
        ep_ret = 0.0
        for t in range(int(cfg["env"]["episode_steps"])):
            action = agent.act(obs, info, agent.eps)
            # 3GPP A3 兜底：服务 SINR 跌破临界且 RL 未发起合法切换 → 回落到 A3 决策
            if safety.should_fallback(info["serving_sinr"], action):
                action = safety.a3_action(info)
            next_obs, reward, term, trunc, next_info = env.step(action)
            agent.store(obs, action, reward, next_obs, term)
            obs, info = next_obs, next_info
            if t >= warmup:
                agent.learn()
            ep_ret += reward
            if term or trunc:
                break
        # episode 边界：刷新 n 步回报窗口（截断 episode 的短窗口不得跨集累积）
        agent.flush_nstep()
        agent.decay_eps()
        agent.anneal_beta((ep + 1) / n_eps)
        print(f"ep {ep:03d} ret={ep_ret:.2f} eps={agent.eps:.3f}", flush=True)

    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
    agent.save(ckpt_path)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--checkpoint", default="checkpoints/latest.pt")
    a = p.parse_args()
    run(load_config(a.config), a.checkpoint)
