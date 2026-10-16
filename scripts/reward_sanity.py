"""G3：奖励判别性门禁 —— 训练前先确认「奖励能区分好坏策略」。

为什么需要这道门禁
------------------
本轮所有"训练后才发现学坏了"的返工，本质都是**奖励本身没被验证过**：
  * 双重 log1p 基线：教师回报被抬高 ~240，动作间差异被淹没；
  * 吞吐主项：强噪声（CV≈1.12、45% 步接近 0）+ 随未观测负载漂移（同 SINR
    下族间差 2.5 倍）⇒ 观测→回报是「一对多」，TD 目标不可学。
这两类问题都能在**训练前**用三个策略跑几集就发现：坏策略的回报必须明显
低于好策略，且好策略的期望回报应为正。

判据（任一不满足即拒绝进入 Stage2）：
  1. A3 教师平均回报 **> 0**
  2. A3 教师 **明显优于** always-stay（差距 > 判据阈值）
  3. A3 教师 **优于** 随机策略
  4. 退化策略（always-stay）在含切换需求的场景上 **不应** 拿到最高回报

用法：
  PYTHONPATH=. python -m scripts.reward_sanity --config configs/ns3.yaml --n-eps 3
"""

import argparse
import sys

import numpy as np

from src.utils.config import load_config
from src.agents.baselines import A3OperatorBaseline
from src.envs.ns3_env import NS3HandoverEnv, family_pool


class _ConstPolicy:
    """恒定动作（如 always stay），用于构造退化对照。"""

    def __init__(self, action: int):
        self.a = int(action)

    def reset(self):
        pass

    def act(self, obs, info):
        return self.a


class _RandomPolicy:
    """随机动作（均匀），下界对照。"""

    def __init__(self, n_actions: int, rng):
        self.n = int(n_actions)
        self.rng = rng

    def reset(self):
        pass

    def act(self, obs, info):
        return int(self.rng.integers(0, self.n))


def run(env, pol, n_eps, seed0, family, steps_max):
    rets, surv, steps = [], [], []
    for k in range(n_eps):
        if hasattr(pol, "reset"):
            pol.reset()
        obs, info = env.reset(seed=seed0 + k,
                              options={"family": family} if family else None)
        ep_r, n = 0.0, 0
        term = False
        while n < steps_max:
            a = int(pol.act(obs, info))
            obs, r, t, tr, info = env.step(a)
            ep_r += float(r)
            n += 1
            term = bool(t)
            if t or tr:
                break
        rets.append(ep_r)
        steps.append(n)
        surv.append(0.0 if term else 1.0)
    return {
        "reward_mean": float(np.mean(rets)),
        "reward_std": float(np.std(rets)),
        "survival": float(np.mean(surv)),
        "steps_mean": float(np.mean(steps)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ns3.yaml")
    p.add_argument("--n-eps", type=int, default=3)
    p.add_argument("--seed", type=int, default=910000)
    p.add_argument("--families", default=None,
                   help="只测这些族（逗号分隔）；默认测训练池全部")
    p.add_argument("--min-margin", type=float, default=20.0,
                   help="教师相对 always-stay 的最小回报差距")
    args = p.parse_args()

    cfg = load_config(args.config)
    dt = float(cfg["mobility"]["dt_s"])
    a3_cfg = cfg.get("a3_operator", {})
    a3_cfg.setdefault("fc_hz", float(cfg["channel"]["fc_hz"]))
    teacher = A3OperatorBaseline(cfg["safety"], a3_cfg, dt)
    env = NS3HandoverEnv(cfg, guard=None)
    stay = _ConstPolicy(0)
    rand = _RandomPolicy(int(cfg["env"]["n_actions"]), np.random.default_rng(args.seed))
    fams = family_pool(cfg) or [None]
    if args.families:
        want = [s.strip() for s in args.families.split(",") if s.strip()]
        fams = [f for f in fams if (f in want)]
    steps_max = int(cfg["env"]["episode_steps"])

    print("=== G3 奖励判别性门禁 ===")
    print(f"  判据：教师回报>0、教师−stay ≥ {args.min_margin:.0f}、教师>随机")
    ok = True
    try:
        for fam in fams:
            r_tea = run(env, teacher, args.n_eps, args.seed, fam, steps_max)
            r_stay = run(env, stay, args.n_eps, args.seed + 100, fam, steps_max)
            r_rand = run(env, rand, args.n_eps, args.seed + 200, fam, steps_max)
            print(f"\n  族={fam}")
            for name, m in (("教师A3", r_tea), ("常温stay", r_stay), ("随机", r_rand)):
                print(f"    {name:8s} R={m['reward_mean']:8.1f} ±{m['reward_std']:6.1f} "
                      f"存活={m['survival']:.2f} 步数={m['steps_mean']:.0f}")
            margin = r_tea["reward_mean"] - r_stay["reward_mean"]
            checks = [
                (r_tea["reward_mean"] > 0, "教师回报 > 0"),
                (margin >= args.min_margin, f"教师−stay ≥ {args.min_margin:.0f}（实测 {margin:.1f}）"),
                (r_tea["reward_mean"] > r_rand["reward_mean"], "教师 > 随机"),
            ]
            for passed, desc in checks:
                print(f"    [{'PASS' if passed else 'FAIL'}] {desc}")
                ok = ok and passed
    finally:
        env.close()

    print()
    if ok:
        print("[G3 PASS] 奖励能区分好坏策略，可进入 Stage2。")
        return 0
    print("[G3 FAIL] 奖励判别性不足 —— 禁止用当前奖励配置启动 Stage2。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
