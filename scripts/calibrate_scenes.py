"""Stage0 场景族标定：用 A3 教师跑每个 scenario_family，判定能否入训练池。

动机（本轮教训）：训练分布一旦对「规则策略」都不可存活，RL 就没有任何
正样本 —— 实测 congested 族 A3 教师 4/4 全灭（258/480 步、outage=1.0、
R=14.5±13.8），而同一策略在评估分布上是 480 步、outage=0、R=539.9。
因此**任何族进入训练池前必须先过教师可存活性标定**。

产出：
  1. 每个族的 存活率 / 平均步数 / A3 平均回报 / A3 平均吞吐
  2. 入池建议（survival ≥ --min-survival 且 steps ≥ min-steps-ratio）
  3. --write 时把结果落到 configs/scene_baseline.json ——
     这是「课程训练池」与「奖励基线折算」的**单一数据源**，
     ns3.yaml 的 reward_baselines 从该 JSON 同步（人工或 --sync）。

用法（云端容器内）：
  PYTHONPATH=. /opt/conda/bin/python -m scripts.calibrate_scenes \
      --config configs/ns3.yaml --n-eps 4 --write
"""

import argparse
import json
import os

import numpy as np

from src.utils.config import load_config
from src.envs.ns3_env import NS3HandoverEnv
from src.agents.baselines import A3OperatorBaseline


def run_family(cfg, family, n_eps, seed_base):
    """在指定族上跑 A3 教师 n_eps 集，返回统计量。"""
    dt_s = float(cfg["mobility"]["dt_s"])
    a3_cfg = cfg.get("a3_operator", {})
    a3_cfg.setdefault("fc_hz", float(cfg["channel"]["fc_hz"]))
    env = NS3HandoverEnv(cfg, guard=None)
    pol = A3OperatorBaseline(cfg["safety"], a3_cfg, dt_s)
    rets, steps, tps, surv, log_tps = [], [], [], [], []
    try:
        for k in range(n_eps):
            obs, info = env.reset(seed=seed_base + k, options={"family": family})
            if hasattr(pol, "reset"):
                pol.reset()
            ep_r = ep_tp = n = 0
            term_early = False
            log_tp_buf = []
            while True:
                a = int(pol.act(obs, info))
                obs, r, term, trunc, info = env.step(a)
                n += 1
                ep_r += float(r)
                tp_now = float(info.get("throughput_mbps", 0.0))
                ep_tp += tp_now
                log_tp_buf.append(float(np.log1p(max(tp_now, 0.0))))
                if term or trunc:
                    term_early = bool(term)
                    break
            rets.append(ep_r); steps.append(n)
            tps.append(ep_tp / max(n, 1))
            surv.append(0.0 if term_early else 1.0)
            # 对数域均值：与奖励项 r_tp = log1p(tp) **同一统计量**。
            # 用线性均值做基线会每步扣掉一个巨大常数（实测让 A3 存活集
            # R 从 +500 变 -331），使「早死（RLF 封顶 -70）」反而比
            # 「活到最后」更划算 —— 必须同域才不会倒挂激励。
            log_tps.append(float(np.mean(log_tp_buf)) if log_tp_buf else 0.0)
    finally:
        env.close()
    return {
        "n_eps": n_eps,
        "survival": float(np.mean(surv)),
        "steps_mean": float(np.mean(steps)),
        "reward_mean": float(np.mean(rets)),
        "reward_std": float(np.std(rets)),
        "throughput_mean": float(np.mean(tps)),
        "log_tp_mean": float(np.mean(log_tps)),   # ← reward_baselines 应取此项
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ns3.yaml")
    p.add_argument("--n-eps", type=int, default=4)
    p.add_argument("--seed", type=int, default=700000)
    p.add_argument("--min-survival", type=float, default=0.7,
                   help="入池门槛：教师存活率")
    p.add_argument("--min-steps-ratio", type=float, default=0.8,
                   help="入池门槛：教师平均步数 / episode_steps")
    p.add_argument("--out", default="configs/scene_baseline.json")
    p.add_argument("--write", action="store_true", help="落盘基线 JSON")
    args = p.parse_args()

    cfg = load_config(args.config)
    fams = (cfg.get("ns3") or {}).get("scenario_families") or {}
    if not fams:
        print("配置里没有 ns3.scenario_families，无可标定族。")
        return
    ep_steps = int(cfg["env"]["episode_steps"])
    print(f"=== Stage0 场景族标定（A3 教师，{args.n_eps} 集/族）===")
    print(f"入池门槛: survival ≥ {args.min_survival}, "
          f"steps ≥ {args.min_steps_ratio}×{ep_steps}")

    out = {}
    rows = []
    for name in fams:
        s = run_family(cfg, name, args.n_eps, args.seed)
        ok = (s["survival"] >= args.min_survival
              and s["steps_mean"] >= args.min_steps_ratio * ep_steps)
        s["in_pool"] = bool(ok)
        out[name] = s
        rows.append((name, s, ok))
        print(f"  {name:10s}  survival={s['survival']:.2f}  "
              f"steps={s['steps_mean']:.0f}/{ep_steps}  "
              f"R={s['reward_mean']:.1f}±{s['reward_std']:.1f}  "
              f"tp={s['throughput_mean']:.2f}Mbps  -> "
              f"{'入池' if ok else '出池(教师不可存活)'}")

    # 建议：把 enabled 与入池结论对齐；reward_baselines 用教师吞吐均值
    print("\n建议 ns3.yaml 修改：")
    print("  ns3.scenario_families.<族>.enabled =",
          {n: bool(s["in_pool"]) for n, s in out.items()})
    print("  reward_baselines =  # 对数域均值，与 r_tp=log1p(tp) 同域；"
          "用线性均值会让存活集奖励变负、激励倒挂")
    print("   ", {n: round(s["log_tp_mean"], 3) for n, s in out.items()
                  if s["in_pool"]})

    if args.write:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
