"""load 族死亡**复现**与分类 —— 口径与 paired_matrix 逐字一致。

为什么必须重写（不用 scripts/diagnose_rl.py）
--------------------------------------------
cmd_0185 用 diagnose_rl 跑 load 族 8 集，结果 survival=1.00、8/8 survived，
**完全没有复现** cmd_0184 观测到的存活 0.75。逐位相同的 last_ho 暴露了原因：
diagnose_rl 与 paired_matrix 在三个地方口径不同 ——

  1. env：diagnose_rl 用 NS3HandoverEnv(cfg, guard=None)（**绕开安全层**）；
     paired_matrix 用 NS3HandoverEnv(cfg)（默认 guard）。
  2. 策略：diagnose_rl 用裸网络 argmax；paired_matrix 用
     DuelingDDQNAgent.act(obs, info, eps=0.0)（含动作掩码/安全改写）。
  3. seed：diagnose_rl 用 seed+k；paired_matrix 用 seed_base + seed*1000 + ep。
     两者数值**完全不重合**（500000..500007 vs 500000/500001/501000/501001…），
     而掉话是由背景负载决定的 ⇒ 换一批 seed 就等于换了一个问题。

本脚本因此逐字复刻 paired_matrix 的 env/agent/seed 构造，只额外做两件事：
  * 逐步记录 RL 决策与 A3 反事实决策（用于死因分类）；
  * 记录 action_source 分布（看安全层到底接管了多少）。

用法（seed 分片并行，每片一个进程）：
  PYTHONPATH=. python -m session_logs.diag_death_probe \
      --config configs/ns3_stageH11_loadonly.yaml \
      --checkpoint checkpoints_stageH11/best.pt \
      --seed-list 0,1 --seeds-base 500000 --n-eps 2 --out logs/dh_01.json
"""

import argparse
import json
import os
import sys

import numpy as np

from src.utils.config import load_config
from src.envs.ns3_env import NS3HandoverEnv
from src.agents.baselines import A3Baseline

_HO_TO_CAND = {1: 1, 2: 2, 5: 3}
WINDOW = 30


def action_targets(policy_action, executed_action, cand, cand_sinr, neighbors):
    """Pre-dispatch targets; legacy tgt fields now mean dispatched target."""
    def target(action):
        idx = _HO_TO_CAND.get(action)
        if idx is None:
            return {"rsrp": None, "sinr": None, "cell": None}
        return {"rsrp": float(cand[idx]), "sinr": float(cand_sinr[idx]),
                "cell": int(neighbors[idx - 1]) if len(neighbors) >= idx else None}
    requested, executed = target(policy_action), target(executed_action)
    return {"policy_requested_target": requested, "executed_target": executed,
            "tgt_rsrp": executed["rsrp"], "tgt_sinr": executed["sinr"]}


def classify(hist, cause, min_target_rsrp):
    """与 scripts/diagnose_rl.classify 同逻辑（死亡前 30 步窗口比对 RL 与 A3）。"""
    if cause == "truncated":
        return "survived"
    w = hist[-WINDOW:]
    rl_ho = sum(1 for h in w if h.get("ho_completed", False))
    a3_ho = sum(1 for h in w if h["a_a3"] != 0)
    s = [h["serving_sinr"] for h in w]
    slope = float(np.polyfit(np.arange(len(s)), s, 1)[0]) if len(s) >= 3 else 0.0
    if rl_ho == 0 and a3_ho >= 3:
        return "no_ho_while_a3_would"          # 该切没切
    if rl_ho == 0 and slope < -0.05:
        return "no_ho_while_link_degrading"    # 链路在掉而 RL 不动
    weak = [h for h in w if h.get("ho_completed", False)
            and h["executed"] in _HO_TO_CAND
            and h["tgt_rsrp"] is not None
            and (h["tgt_rsrp"] < h.get("pre_serving_rsrp", h["serving_rsrp"])
                 or h["tgt_rsrp"] < min_target_rsrp)]
    if weak and a3_ho == 0:
        return "ho_to_weaker_against_a3"       # A3 不动而 RL 切向更弱
    if weak:
        return "ho_to_weaker"                  # 切向弱小区
    return "unknown"


def last_ho_detail(hist, after=10):
    idx = [i for i, h in enumerate(hist) if h.get("ho_completed", False)]
    if not idx:
        return None
    i = idx[-1]
    h = hist[i]
    nxt = hist[i + 1:i + 1 + after]
    return {
        "step": h["step"], "x": h["x"],
        "srv_rsrp": h["serving_rsrp"], "srv_sinr": h["serving_sinr"],
        "tgt_rsrp": h["tgt_rsrp"], "tgt_sinr": h["tgt_sinr"],
        "executed_action": h["executed"],
        "executed_target": h.get("executed_target"),
        "policy_requested_target": h.get("policy_requested_target"),
        "sinr_after_mean": float(np.mean([n["serving_sinr"] for n in nxt])) if nxt else None,
        "a3_action": h["a_a3"], "a3_tgt_rsrp": h["a3_tgt_rsrp"],
        "steps_to_end": len(hist) - 1 - i, "n_ho_total": len(idx),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--net", default="hybrid", choices=["mlp", "ode", "hybrid"])
    p.add_argument("--family", default="load")
    p.add_argument("--seed-list", default="0,1", help="本进程负责的 seed 下标")
    p.add_argument("--seed-base", type=int, default=500000)
    p.add_argument("--n-eps", type=int, default=2)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    env = NS3HandoverEnv(cfg)                       # ← 与 paired_matrix 一致（默认 guard）
    a3 = A3Baseline(cfg["safety"])

    from src.training.device import get_device
    from src.agents.agent import DuelingDDQNAgent
    from src.agents.networks import DuelingDQN
    from src.agents.neural_ode import NeuralODEDuelingDQN
    from src.agents.hybrid_network import HybridDuelingDQN
    nets = {"mlp": DuelingDQN, "ode": NeuralODEDuelingDQN, "hybrid": HybridDuelingDQN}
    agent = DuelingDDQNAgent(cfg, get_device("cpu"), net_factory=nets[args.net])
    agent.load(args.checkpoint)
    agent.eps = 0.0                                  # ← 与 paired_matrix 一致

    min_tgt = float(cfg["safety"]["min_target_rsrp_dbm"])
    ep_steps = int(cfg["env"]["episode_steps"])
    seeds = [int(x) for x in args.seed_list.split(",") if x.strip() != ""]

    episodes = []
    try:
        for sd in seeds:
            for ep in range(args.n_eps):
                # seed 公式与 paired_matrix._run_one 逐字一致
                seed = args.seed_base + sd * 1000 + ep
                obs, info = env.reset(seed=seed, options={"family": args.family})
                agent.reset_latent()
                spawn_x = float(info.get("ue_x", -1.0))
                a3.reset() if hasattr(a3, "reset") else None
                hist, r_sum, cause = [], 0.0, None
                src_count = {}
                for _ in range(ep_steps):
                    a_rl = int(agent.act(obs, info, eps=0.0))
                    a_a3 = int(a3.act(obs, info))
                    cand = np.asarray(info["candidate_rsrp"], dtype=np.float64)
                    cand_sinr = np.asarray(info.get("candidate_sinr", cand * 0.0),
                                           dtype=np.float64)
                    nobs, r, term, trunc, ninfo = env.step(a_rl)
                    r_sum += float(r)
                    src = str(ninfo.get("action_source", "?"))
                    src_count[src] = src_count.get(src, 0) + 1
                    a_exec = int(ninfo.get("action_executed", a_rl))
                    hist.append({
                        "step": int(ninfo.get("step_idx", len(hist))),
                        "x": float(ninfo.get("ue_x", -1.0)),
                        "serving_rsrp": float(ninfo.get("serving_rsrp", -200.0)),
                        "serving_sinr": float(ninfo.get("serving_sinr", 0.0)),
                        "a_rl": a_rl, "a_a3": a_a3, "executed": a_exec,
                        "pre_x": float(info.get("ue_x", -1.0)),
                        "pre_serving_rsrp": float(info["serving_rsrp"]),
                        "pre_serving_sinr": float(info["serving_sinr"]),
                        "candidate_rsrp": cand.tolist(),
                        "candidate_sinr": cand_sinr.tolist(),
                        "neighbors": info.get("neighbors", []),
                        "serving_cell": ninfo.get("serving_idx"),
                        "q": agent.last_decision["q"],
                        "masked_actions": agent.last_decision["masked_actions"],
                        "ho_completed": bool(ninfo.get("ho_event", False)),
                        "request_result": int(ninfo.get("ho_request_result", 0)),
                        "ho_in_progress": bool(ninfo.get("ho_in_progress", False)),
                        "ho_failed": bool(ninfo.get("ho_failed", False)),
                        **action_targets(a_rl, a_exec, cand, cand_sinr,
                                         info.get("neighbors", [])),
                        "a3_tgt_rsrp": (float(cand[_HO_TO_CAND[a_a3]])
                                        if a_a3 in _HO_TO_CAND else None),
                        "tp": float(ninfo.get("throughput_mbps", 0.0)),
                        "rlf": int(ninfo.get("rlf", 0)),
                    })
                    obs, info = nobs, ninfo
                    if term or trunc:
                        cause = ("rlf" if (term and int(ninfo.get("rlf", 0)) == 1)
                                 else ("low_sinr" if term else "truncated"))
                        break
                w = hist[-WINDOW:]
                episodes.append({
                    "seed_idx": sd, "seed": seed, "family": args.family,
                    "steps": len(hist), "reward": r_sum,
                    "termination": cause or "truncated",
                    "class": classify(hist, cause or "truncated", min_tgt),
                    "death_x": (None if (cause or "truncated") == "truncated"
                                else float(hist[-1]["x"])),
                    "rl_ho_last30": sum(h["ho_completed"] for h in w),
                    "a3_ho_last30": sum(1 for h in w if h["a_a3"] != 0),
                    "srv_sinr_min_last30": (float(np.min([h["serving_sinr"] for h in w]))
                                            if w else None),
                    "ho_total": sum(h["ho_completed"] for h in hist),
                    "ho_requests": sum(h["executed"] in _HO_TO_CAND for h in hist),
                    "spawn_x": spawn_x, "history": hist,
                    "src_count": src_count,
                    "last_ho": last_ho_detail(hist),
                })
                print("  [seed=%d ep=%d] steps=%d term=%s class=%s R=%.2f ho=%d"
                      % (sd, ep, len(hist), cause or "truncated",
                         episodes[-1]["class"], r_sum, episodes[-1]["ho_total"]),
                      flush=True)
    finally:
        env.close()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.checkpoint, "family": args.family,
                   "episodes": episodes}, f, ensure_ascii=False, indent=2)
    surv = float(np.mean([e["termination"] == "truncated" for e in episodes]))
    print("  => 本片 survival=%.3f  n=%d  已写入 %s" % (surv, len(episodes), args.out))


if __name__ == "__main__":
    main()
