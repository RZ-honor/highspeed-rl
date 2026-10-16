"""NS-3 后端基线对照（Stage C）。

在同一 NS3HandoverEnv 上运行 3GPP A3 / 贪心 / 固定间隔基线，
可选 --checkpoint 同时评估已训练 RL agent，输出同口径对比表。
同种子、同场景、同指标——这是衡量 RL 在真实仿真中增益的参照系。

云端：
  python -m scripts.baselines_ns3 --config configs/ns3.yaml
  python -m scripts.baselines_ns3 --config configs/ns3.yaml \
      --checkpoint checkpoints_ns3/latest.pt --net hybrid

本地 mock 冒烟（无 NS-3）：
  python -m scripts.baselines_ns3 --mock
"""

import argparse
import sys

import numpy as np

from src.utils.config import load_config
from src.envs.ns3_env import NS3HandoverEnv
from src.agents.baselines import A3Baseline, GreedyBaseline, FixedIntervalBaseline


def run_policy(env, act_fn, n_eps, seed_base, family=None):
    rec = {"throughput": [], "ho": [], "pingpong": [], "outage": [], "reward": []}
    for ep in range(n_eps):
        opts = {"family": family} if family else None
        obs, info = env.reset(seed=seed_base + ep, options=opts)
        ep_tp, ho, pp, out, steps, ep_r = 0.0, 0, 0, 0, 0, 0.0
        for _ in range(env.episode_steps):
            a = act_fn(obs, info)
            obs, r, term, trunc, ninfo = env.step(a)
            ep_r += r
            ep_tp += ninfo.get("throughput_mbps", 0.0)
            if ninfo.get("ho_event"):
                ho += 1
            if ninfo.get("pingpong"):
                pp += 1
            if term:
                out += 1
            info = ninfo
            steps += 1
            if term or trunc:
                break
        rec["throughput"].append(ep_tp / max(steps, 1))
        rec["ho"].append(ho)
        rec["pingpong"].append(pp)
        rec["outage"].append(out)
        rec["reward"].append(ep_r)
    out = {k: float(np.mean(v)) for k, v in rec.items()}
    # 存活率（跑满 480 步的比例）是主指标：全灭分布下 R 会被 RLF 罚没淹掉
    out["survival"] = 1.0 - out["outage"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ns3.yaml")
    p.add_argument("--n-episodes", type=int, default=None,
                   help="评估集数（默认取 eval.n_episodes）")
    p.add_argument("--checkpoint", default=None,
                   help="已训练 RL checkpoint，加入对比")
    p.add_argument("--net", choices=["mlp", "ode", "hybrid"], default="hybrid",
                   help="checkpoint 对应的网络结构")
    p.add_argument("--device", default="cuda")
    p.add_argument("--mock", action="store_true", help="本地协议 mock 冒烟")
    p.add_argument("--families", default=None,
                   help="按场景族逐族配对评估（逗号分隔，如 "
                        "normal,tunnel,load,congested）。同一族下各策略共用同一组 "
                        "seed，消除场景间方差——这是「RL 相对规则是否有增益」的"
                        "唯一可比口径。")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.mock:
        cfg["ns3"] = {
            "binary": sys.executable,
            "args": "-m scripts.ns3_mock --dur=6 --nEnb=8 --spacing=600 --offset=50",
            "cwd": ".",
            "timeout_s": 60,
        }
        cfg["env"]["episode_steps"] = 50
        cfg["eval"]["n_episodes"] = 3
        args.device = "cpu"
    n_eps = args.n_episodes or int(cfg["eval"]["n_episodes"])
    seed_base = int(cfg.get("seed", 42)) + 500000  # 与训练/评估种子空间隔离

    env = NS3HandoverEnv(cfg)

    policies = {}
    a3 = A3Baseline(cfg["safety"])
    greedy = GreedyBaseline()
    fixed = FixedIntervalBaseline(interval_steps=10)
    policies["A3(3GPP)"] = lambda obs, info: a3.act(obs, info)
    policies["Greedy"] = lambda obs, info: greedy.act(obs, info)
    policies["Fixed(1s)"] = lambda obs, info: fixed.act(obs, info)

    if args.checkpoint:
        import torch
        from src.training.device import get_device
        from src.agents.agent import DuelingDDQNAgent
        from src.agents.networks import DuelingDQN
        from src.agents.neural_ode import NeuralODEDuelingDQN
        from src.agents.hybrid_network import HybridDuelingDQN
        nets = {"mlp": DuelingDQN, "ode": NeuralODEDuelingDQN, "hybrid": HybridDuelingDQN}
        agent = DuelingDDQNAgent(cfg, get_device(args.device), net_factory=nets[args.net])
        agent.load(args.checkpoint)
        agent.eps = 0.0
        tag = f"RL({args.net})"
        policies[tag] = lambda obs, info: agent.act(obs, info, eps=0.0)

    print(f"NS-3 基线对照: {n_eps} episodes, binary={cfg['ns3']['binary']}")
    keys = ["throughput", "ho", "pingpong", "outage", "survival", "reward"]
    fams = [s.strip() for s in args.families.split(",")] if args.families else [None]

    for fam in fams:
        print(f"\n########## 族 = {fam} ##########")
        results = {}
        for name, fn in policies.items():
            print(f"--- {name} ---", flush=True)
            results[name] = run_policy(env, fn, n_eps, seed_base, family=fam)

        header = f"{'指标':12s}" + "".join(f"{n:>14s}" for n in results)
        print(header)
        for k in keys:
            row = f"{k:12s}"
            for name in results:
                row += f"{results[name][k]:14.4f}"
            print(row)

        if args.checkpoint:
            rl_key = [n for n in results if n.startswith("RL(")][0]
            a3m, rlm = results["A3(3GPP)"], results[rl_key]
            print(f"--- {rl_key} vs A3(3GPP) ---")
            for k in keys:
                better = "high" if k in ("throughput", "reward", "survival") else "low"
                sign = 1 if better == "high" else -1
                gain = sign * (rlm[k] - a3m[k]) / (abs(a3m[k]) + 1e-9) * 100
                print(f"  {k:12s} {a3m[k]:10.4f} → {rlm[k]:10.4f}  ({gain:+.1f}%)")


if __name__ == "__main__":
    main()
