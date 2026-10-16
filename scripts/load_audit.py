"""负载分流机制审计：RL 是否真的「切向更空闲的小区」？

为什么需要这个审计
------------------
配对评估显示 RL 的吞吐增益随负载严重度单调递增（normal +5.4% → congested
+11.8%），这是「负载维度被用上」的**间接**证据；但间接证据排除不掉混淆
（例如 RL 只是切换时机更好）。本脚本给出**直接**证据：在同一组 seed 上跑
A3 规则与 RL，逐步记录服务小区的背景 UE 数（负载代理 load_serving），比较：

  * load_time_mean  服务小区负载的**时间均值**（越小 = 越少待在拥塞小区）
  * load_after_ho   **切换后**新服务小区的负载均值（RL 的目标是否更空闲）
  * d_load_ho       切换前后负载变化（Δ<0 = 主动从拥塞小区卸载）

判据：若 RL 的时间均值/切换后负载显著低于 A3 而吞吐更高，则确认增益来自
负载分流，而非其他因素。

用法：
  PYTHONPATH=. python -m scripts.load_audit --config configs/ns3_stageG4.yaml \
      --checkpoint checkpoints_stageG4/best.pt --families load,congested --n-eps 3
"""

import argparse
import sys

import numpy as np

from src.utils.config import load_config
from src.envs.ns3_env import NS3HandoverEnv
from src.agents.baselines import A3Baseline


def audit_policy(env, act_fn, n_eps, seed_base, family):
    loads, after_ho, d_load, tputs, surv, ho_total = [], [], [], [], 0, 0
    for ep in range(n_eps):
        opts = {"family": family} if family else None
        obs, info = env.reset(seed=seed_base + ep, options=opts)
        prev_load, ep_tp, n = None, 0.0, 0
        term = False
        for _ in range(env.episode_steps):
            a = int(act_fn(obs, info))
            obs, r, term, trunc, info = env.step(a)
            load_now = float(info.get("load_serving", 0))
            loads.append(load_now)
            ep_tp += float(info.get("throughput_mbps", 0.0))
            if info.get("ho_event"):
                ho_total += 1
                after_ho.append(load_now)
                if prev_load is not None:
                    d_load.append(load_now - prev_load)
            prev_load = load_now
            n += 1
            if term or trunc:
                break
        tputs.append(ep_tp / max(n, 1))
        if not term:
            surv += 1

    def _m(x):
        return float(np.mean(x)) if len(x) else float("nan")

    return {
        "load_time_mean": _m(loads),
        "load_after_ho": _m(after_ho),
        "d_load_ho": _m(d_load),
        "tput": _m(tputs),
        "survival": surv / max(n_eps, 1),
        "ho_per_ep": ho_total / max(n_eps, 1),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ns3_stageG4.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--net", choices=["mlp", "ode", "hybrid"], default="hybrid")
    p.add_argument("--device", default="cuda")
    p.add_argument("--families", default="load,congested")
    p.add_argument("--n-eps", type=int, default=3)
    p.add_argument("--seed", type=int, default=770000)
    args = p.parse_args()

    cfg = load_config(args.config)
    if not cfg["env"].get("load_obs"):
        sys.exit("[FATAL] 需要 env.load_obs=true 的配置（负载审计依赖负载观测）。")
    env = NS3HandoverEnv(cfg)

    a3 = A3Baseline(cfg["safety"])
    policies = {"A3(3GPP)": lambda o, i: a3.act(o, i)}

    if args.checkpoint:
        from src.training.device import get_device
        from src.agents.agent import DuelingDDQNAgent
        from src.agents.networks import DuelingDQN
        from src.agents.neural_ode import NeuralODEDuelingDQN
        from src.agents.hybrid_network import HybridDuelingDQN
        nets = {"mlp": DuelingDQN, "ode": NeuralODEDuelingDQN, "hybrid": HybridDuelingDQN}
        agent = DuelingDDQNAgent(cfg, get_device(args.device), net_factory=nets[args.net])
        agent.load(args.checkpoint)
        agent.eps = 0.0
        policies[f"RL({args.net})"] = lambda o, i: agent.act(o, i, eps=0.0)

    keys = ["load_time_mean", "load_after_ho", "d_load_ho", "tput", "survival", "ho_per_ep"]
    for fam in [s.strip() for s in args.families.split(",") if s.strip()]:
        print(f"\n########## 负载分流审计：族 = {fam}（{args.n_eps} 集/策略）##########", flush=True)
        res = {}
        for name, fn in policies.items():
            res[name] = audit_policy(env, fn, args.n_eps, args.seed, fam)
            print(f"  [done] {name}", flush=True)

        print(f"  {'指标':16s}" + "".join(f"{n:>14s}" for n in res))
        for k in keys:
            print(f"  {k:16s}" + "".join(f"{res[n][k]:14.3f}" for n in res))

        if len(res) == 2:
            a, b = list(res.values())
            print(f"  ⇒ 时间均值负载 : RL {b['load_time_mean']:.2f} vs A3 {a['load_time_mean']:.2f}"
                  f"  (Δ {b['load_time_mean'] - a['load_time_mean']:+.2f})")
            print(f"  ⇒ 切换后负载   : RL {b['load_after_ho']:.2f} vs A3 {a['load_after_ho']:.2f}"
                  f"  (Δ {b['load_after_ho'] - a['load_after_ho']:+.2f})")
            print(f"  ⇒ 吞吐         : RL {b['tput']:.2f} vs A3 {a['tput']:.2f}"
                  f"  ({(b['tput'] - a['tput']) / max(a['tput'], 1e-9) * 100:+.1f}%)")
            verdict = ("已确认：RL 主动切向更空闲的小区"
                       if b["load_time_mean"] < a["load_time_mean"] else
                       "未观察到负载分流（增益需另找原因）")
            print(f"  ⇒ 判定：{verdict}")

    env.close()


if __name__ == "__main__":
    main()
