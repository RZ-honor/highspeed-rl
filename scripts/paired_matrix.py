"""多种子配对评估矩阵（S1）：给「RL 是否优于规则」一个**有置信区间**的答案。

为什么需要它
------------
此前所有对比（含头条结论 +5.4%~+11.8%）都是**单 seed、4 集**。而 B4 的重启
实验已经证明同配置的 run-to-run 方差很大 —— 没有多种子，增益数字只是观测值，
不是结论（不知道它是否被种子运气放大）。

设计要点
--------
1. **配对**：同一 seed 上跑 A3 与 RL，逐 seed 取差 Δ = RL − A3。配对差分消除
   场景间方差，比独立均值比较灵敏得多。
2. **多种子 + bootstrap CI**：对 seed 做有放回重采样（10000 次）得到 Δ 的
   95% 置信区间；CI 不含 0 才说「显著」。这样不需要任何分布假设，也不依赖 scipy。
3. **并行**：每集是一个独立 NS-3 进程，天然可并行；串行跑 96 集要 1 小时以上，
   并行 8 路可压到十几分钟。
4. **主指标是 survival**：Fixed(1s) 这类策略吞吐最高而 outage=1.0，只看吞吐
   会得出荒谬结论。

用法：
  PYTHONPATH=. python -m scripts.paired_matrix --config configs/ns3_stageG4.yaml \
      --checkpoint checkpoints_stageG4/best.pt --seeds 4 --n-eps 3 \
      --families normal,tunnel,load,congested --workers 8
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np

from src.utils.config import load_config
from src.envs.ns3_env import NS3HandoverEnv
from src.agents.baselines import A3Baseline

# episode 记录字段
KEYS = ("throughput", "survival", "reward", "ho", "pingpong")


def _run_one(job: Tuple) -> Dict:
    """单进程内跑完 (seed, family, policy) 的 n_eps 集，返回逐集记录。"""
    (cfg_path, seed, family, policy, n_eps, ckpt, net, seed_base0) = job
    cfg = load_config(cfg_path)
    env = NS3HandoverEnv(cfg)
    a3 = A3Baseline(cfg["safety"])
    agent = None
    if policy.startswith("RL("):
        from src.training.device import get_device
        from src.agents.agent import DuelingDDQNAgent
        from src.agents.networks import DuelingDQN
        from src.agents.neural_ode import NeuralODEDuelingDQN
        from src.agents.hybrid_network import HybridDuelingDQN
        nets = {"mlp": DuelingDQN, "ode": NeuralODEDuelingDQN, "hybrid": HybridDuelingDQN}
        # 工人进程用 CPU：RL 前向很小，避免多进程抢 GPU
        agent = DuelingDDQNAgent(cfg, get_device("cpu"), net_factory=nets[net])
        agent.load(ckpt)
        agent.eps = 0.0

    rec = {k: [] for k in KEYS}
    for ep in range(n_eps):
        obs, info = env.reset(seed=seed_base0 + seed * 1000 + ep,
                              options={"family": family} if family else None)
        tp, steps, ho, pp, r_sum, term = 0.0, 0, 0, 0, 0.0, False
        for _ in range(env.episode_steps):
            a = agent.act(obs, info, eps=0.0) if agent else a3.act(obs, info)
            obs, r, t, tr, info = env.step(int(a))
            tp += float(info.get("throughput_mbps", 0.0))
            ho += int(bool(info.get("ho_event")))
            pp += int(bool(info.get("pingpong")))
            r_sum += float(r)
            steps += 1
            if t or tr:
                term = bool(t)
                break
        rec["throughput"].append(tp / max(steps, 1))
        rec["survival"].append(0.0 if term else 1.0)
        rec["reward"].append(r_sum)
        rec["ho"].append(float(ho))
        rec["pingpong"].append(float(pp))
    env.close()
    return {"seed": seed, "family": family, "policy": policy, "rec": rec}


def _boot_ci(deltas: np.ndarray, n_boot: int = 10000, rng=None) -> Tuple[float, float]:
    """配对差的 bootstrap 95% 置信区间（不依赖 scipy，无需分布假设）。"""
    rng = rng or np.random.default_rng(0)
    if len(deltas) == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(deltas), size=(n_boot, len(deltas)))
    means = deltas[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ns3_stageG4.yaml")
    p.add_argument("--checkpoint", default=None, help="RL checkpoint；不给则只跑 A3")
    p.add_argument("--net", choices=["mlp", "ode", "hybrid"], default="hybrid")
    p.add_argument("--families", default="normal,tunnel,load,congested")
    p.add_argument("--seeds", type=int, default=4, help="种子数（配对差分的样本量）")
    p.add_argument("--n-eps", type=int, default=3, help="每个种子每族跑几集")
    p.add_argument("--workers", type=int, default=8, help="并行 NS-3 进程数")
    p.add_argument("--seed-base", type=int, default=500000)
    p.add_argument("--policy-filter", default=None,
                   help="只跑指定策略（逗号分隔），如 a3,rl；默认全部")
    p.add_argument("--json-out", default=None,
                   help="逐 (族,策略,seed) 均值结果写 JSON（验收判据从数值读，"
                        "不 grep 文本 —— 脚本漂移教训）")
    args = p.parse_args()

    fams = [s.strip() for s in args.families.split(",") if s.strip()]
    seeds = list(range(args.seeds))

    policies = ["A3(3GPP)"]
    if args.checkpoint:
        policies.append(f"RL({args.net})")
    if args.policy_filter:
        want = [s.strip().lower() for s in args.policy_filter.split(",")]
        policies = [p_ for p_ in policies
                    if ("a3" in want and p_.startswith("A3"))
                    or ("rl" in want and p_.startswith("RL"))]

    jobs = [(args.config, sd, fam, pol, args.n_eps, args.checkpoint, args.net,
             args.seed_base)
            for fam in fams for sd in seeds for pol in policies]
    print(f"配对矩阵：{len(fams)} 族 × {len(seeds)} seed × {len(policies)} 策略 "
          f"× {args.n_eps} 集 = {len(jobs)} 任务，并行 {args.workers}", flush=True)

    results = {fam: {pol: {} for pol in policies} for fam in fams}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_run_one, j) for j in jobs]
        done = 0
        for fu in as_completed(futs):
            r = fu.result()
            results[r["family"]][r["policy"]][r["seed"]] = {
                k: float(np.mean(v)) for k, v in r["rec"].items()}
            done += 1
            print(f"  [{done}/{len(jobs)}] {r['family']} seed={r['seed']} {r['policy']}",
                  flush=True)

    if args.json_out:
        import json
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump({"families": fams, "seeds": seeds, "policies": policies,
                       "results": results}, fh, ensure_ascii=False, indent=1)
        print(f"  JSON → {args.json_out}", flush=True)

    print("\n========== 多种子配对结果（每 seed 取均值，跨 seed 报 mean±std） ==========")
    for fam in fams:
        print(f"\n########## 族 = {fam}（{len(seeds)} seed × {args.n_eps} 集）##########")
        per_seed = {}
        for pol in policies:
            arr = np.array([results[fam][pol][sd]["throughput"] for sd in seeds])
            arr_r = np.array([results[fam][pol][sd]["reward"] for sd in seeds])
            arr_s = np.array([results[fam][pol][sd]["survival"] for sd in seeds])
            arr_h = np.array([results[fam][pol][sd]["ho"] for sd in seeds])
            per_seed[pol] = dict(tput=arr, reward=arr_r, survival=arr_s, ho=arr_h)
            print(f"  {pol:10s} 吞吐 {arr.mean():7.2f} ±{arr.std(ddof=1):5.2f} | "
                  f"存活 {arr_s.mean():.2f} | 奖励 {arr_r.mean():7.2f} ±{arr_r.std(ddof=1):5.2f} | "
                  f"ho {arr_h.mean():5.2f}")
        if len(policies) == 2:
            a, b = policies[0], policies[1]
            for k, better_high in (("tput", True), ("reward", True), ("survival", True)):
                d = per_seed[b][k] - per_seed[a][k]
                lo, hi = _boot_ci(d)
                sig = "显著" if (lo > 0 and hi > 0) or (lo < 0 and hi < 0) else "不显著"
                print(f"    Δ{k:8s}= {d.mean():+7.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]  "
                      f"({sig})")


if __name__ == "__main__":
    sys.exit(main())
