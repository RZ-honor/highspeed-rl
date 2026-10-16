"""配置生效值校验门禁（S0）：把「我想要的」与「实际生效的」对齐后再开跑。

为什么必须有这道门禁
--------------------
本项目已发生 **三次** 同类事故，且三次都不是逻辑错误，而是配置意图与生效值漂移：
  1. ns3.yaml 缺 w_outage 等键 → env_cfg.get(...,0) 静默取 0 → 弱覆盖预警整段失效
  2. n-step 目标用 γ 而非 γ^n → 等效折扣被抬高，远期价值系统性高估
  3. 阶段配置生成器把 guard.enable 从 true 静默改成 false → 规则接管层全程装死
   （监控里 guard={'comparisons':0,...} 却无人察觉）

共同根因：没有机制保证「声明的意图」等于「运行时的值」。本脚本把可自动判定的
部分变成硬校验（ERROR 直接退出 1），把需要人确认的部分变成显眼告警（WARN），
并打印**相对基线配置的生效值 diff**（让任何改动都无处可藏）。

用法（建议在 train_ns3 之前跑）：
  PYTHONPATH=. python -m scripts.check_config --config configs/ns3_stageG4.yaml
  PYTHONPATH=. python -m scripts.check_config --config configs/ns3_stageG4.yaml \
      --baseline configs/ns3.yaml --strict
"""

import argparse
import sys

from src.utils.config import load_config

# 安全/奖励关键项：缺键或为 0 会静默关闭一整类行为（事故 1 的教训）
CRITICAL_ENV = {
    "w_coverage": "覆盖质量主奖励项（路径 A 的核心）",
    "w_outage": "弱覆盖预警（曾因缺键静默为 0）",
    "rlf_drop_penalty": "RLF 期间的流量惩罚",
    "rlf_termination_penalty": "RLF 终止惩罚（按剩余步数计）",
    "fallback_persist_steps": "RLF 兜底持续性（防单步深衰落误触发）",
    "weak_coverage_sinr_db": "弱覆盖预警门限",
}


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ns3.yaml")
    p.add_argument("--baseline", default="configs/ns3.yaml",
                   help="基线配置（默认 ns3.yaml），用于打印生效值 diff")
    p.add_argument("--strict", action="store_true",
                   help="WARN 也视为失败（CI 用）")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    base = load_config(args.baseline)
    errs, warns = [], []

    env = cfg["env"]
    # ---- 1) obs_dim 与观测开关一致性 ----
    ext = bool(env.get("extended_obs", False))
    load_obs = bool(env.get("load_obs", False))
    want = 14 if not ext else (28 if load_obs else 23)
    if int(env["obs_dim"]) != want:
        errs.append(f"obs_dim={env['obs_dim']} 与 (extended_obs={ext}, load_obs={load_obs}) "
                    f"不一致，应为 {want}（不一致会让 _build_obs 断言失败或静默错位）")

    # ---- 2) 安全/奖励关键项 ----
    for k, why in CRITICAL_ENV.items():
        if k not in env:
            errs.append(f"env 缺键 {k}（{why}）——会静默回退默认值")
        elif float(env[k]) == 0.0:
            errs.append(f"env.{k}=0（{why}）——该项实际未生效")

    # ---- 3) 规则接管层的意图 vs 生效值（事故 3）----
    mode = str((cfg.get("safety") or {}).get("mode", "strict"))
    g_en = bool((cfg.get("guard") or {}).get("enable", False))
    if mode == "guard" and not g_en:
        warns.append("safety.mode=guard 但 guard.enable=false == 规则接管层**实际关闭**"
                     "（全程放行给 RL）。若这是刻意的（当前各 stage 默认如此），"
                     "请确认评估口径同样无 guard；若不是刻意的，应立即开启。")

    # ---- 4) n-step 与折扣（事故 2 相关，仅提示量纲）----
    n_step = int((cfg.get("agent") or {}).get("n_step", 1))
    gamma = float((cfg.get("agent") or {}).get("gamma", 0.99))
    if not args.quiet:
        print(f"  n_step={n_step}, γ^n={gamma ** n_step:.4f}（目标侧 bootstrap 必须按此折扣）")

    # ---- 5) 生效值 diff（让漂移无处可藏）----
    fb, fc = _flat(base), _flat(cfg)
    diffs = [k for k in sorted(set(fb) | set(fc)) if fb.get(k) != fc.get(k)]
    if diffs and not args.quiet:
        print(f"\n  相对基线 {args.baseline} 的生效值差异（{len(diffs)} 项）：")
        for k in diffs[:40]:
            print(f"    {k}: {fb.get(k)!r} -> {fc.get(k)!r}")

    # ---- 判定 ----
    print()
    for e in errs:
        print(f"  [ERROR] {e}")
    for w in warns:
        print(f"  [WARN ] {w}")
    if not errs and not warns:
        print(f"  [OK] {args.config} 配置自洽（无 ERROR / WARN）")

    if errs or (args.strict and warns):
        print(f"\n[CHECK_CONFIG FAIL] {len(errs)} error, {len(warns)} warn —— 禁止开跑")
        return 1
    print(f"[CHECK_CONFIG PASS] {args.config}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
