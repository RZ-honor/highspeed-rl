"""整段路径（journey-level）测评探针：把一次行程切成等长**里程段**，逐段统计
时延 / 速率 / 覆盖 / 移动性，输出「里程 → 指标」曲线与路径级聚合结论。

为什么需要它（现有指标的盲区）
------------------------------
capacity_probe 只看「稳态均值」：它告诉我们 N=128 时全网人均 0.58 Mbps，却回答
不了运营真正会追问的问题：
  * 这一段 20 km 里，**哪些里程**时延或速率不达标？是洞口、小区边界还是进站段？
  * 切一次断多久？全程断了几次？中断总时长折算到每公里是多少？
  * 最差 1% 的里程有多差（尾部风险），而不是平均值有多好？
  * 隧道口、服务小区边界这些**离散事件点**附近的指标凹陷有多深、恢复要多久？
均值会把这些局部凹陷全部抹平，而高铁的可用性问题恰恰是局部的。

设计
----
1. **里程分箱**：以起点里程为 0，按 `--seg-len`（默认 500 m）分段；每步按当前
   ue_x 落到对应段。跨 N 段时同时输出「段内最差」与「段内均值」——只报均值会
   掩盖切换中断（几十毫秒的塌陷在 500 m 均值里不可见）。
2. **路径级聚合**（可直接写进验收门禁）：
     avail_km            可用里程（未 RLF 且未掉话）占比
     delay_ok_km         区间平均时延的跨区间 p95 ≤ 门限的里程占比（非包级）
     rate_ok_km          最差用户速率 ≥ 门限的里程占比
     weak_cov_km         SINR < RLF+3dB 的里程占比
     ho_per_km           切换次数/公里（移动性代价）
     interrupt_ms_per_km RRC Start→EndOk 总时长/公里（非业务中断）
     worst10_tp          最差 10% 段的速率中位（尾部风险，比均值有意义）
3. **A3 vs RL 逐段配对**：同 seed 跑两种策略，按里程段逐段取差 Δ，然后**按里程
   打印差异曲线**——「RL 在哪一段赢、哪一段输」比单一总增益更能指导下一步改进。

用法：
  PYTHONPATH=. python -m scripts.path_probe --config configs/ns3_capacity.yaml \
      --checkpoint checkpoints_stageH8/best.pt --both --n-eps 2 --seg-len 500 \
      --out configs/path_a3_vs_rl.json
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from src.utils.config import load_config
from src.envs.ns3_env import NS3HandoverEnv
from src.agents.baselines import A3Baseline

STEP_KEYS = ("x", "t", "rsrp", "sinr", "tp_ctrl", "tp_cell", "tp_p05", "delay",
             "ho", "ho_int", "rlf", "tunnel", "n_ue", "load")


def _run_one(job):
    (cfg_path, ep, ckpt, net, policy, seg_len, seed_base, dur, steps,
     device, family, n_bg, cbr, voip, video, dlrb) = job
    cfg = load_config(cfg_path)
    # 背景 UE 规模（压测的自变量）。与 capacity_probe 一致：直接覆盖 scenario，
    # 并固定场景其余维度，避免把"规模效应"与"抽样噪声"混在一起。
    cfg["ns3"]["scenario"] = dict(cfg["ns3"].get("scenario") or {})
    if n_bg > 0:
        cfg["ns3"]["scenario"]["n_bg_ue"] = int(n_bg)
    # 业务量：控车速率 + 背景业务配比。要看到 RL 与 A3 的差异必须制造**压力**
    # （cmd_0155 的教训：轻载时两者决策完全重合，Δ 恒为 0）。
    if cbr > 0:
        toks = [t for t in str(cfg["ns3"].get("args", "")).split()
                if not t.startswith("--cbrRate=")]
        toks.append(f"--cbrRate={float(cbr):g}")
        cfg["ns3"]["args"] = " ".join(toks)
    # 带宽维度（--dlRb，100 RB = 18 MHz）：用于给出"每 MHz 可承载用户数"，
    # 也是区分"带宽瓶颈 vs 连接数瓶颈"的关键自变量。
    if dlrb > 0:
        toks = [t for t in str(cfg["ns3"].get("args", "")).split()
                if not t.startswith("--dlRb=")]
        toks.append(f"--dlRb={int(dlrb)}")
        cfg["ns3"]["args"] = " ".join(toks)
    if voip >= 0.0 or video >= 0.0:
        cfg["ns3"]["scenario"]["randomize"] = False
        if voip >= 0.0:
            cfg["ns3"]["scenario"]["voip_frac"] = float(voip)
        if video >= 0.0:
            cfg["ns3"]["scenario"]["video_frac"] = float(video)
    if dur > 0:
        toks = [t for t in str(cfg["ns3"].get("args", "")).split()
                if not t.startswith("--dur=")]
        toks.append(f"--dur={float(dur):g}")
        cfg["ns3"]["args"] = " ".join(toks)
    if steps > 0:
        cfg["env"]["episode_steps"] = int(steps)

    env = NS3HandoverEnv(cfg)
    a3 = A3Baseline(cfg["safety"])
    agent = None
    if ckpt and policy == "rl":
        from src.training.device import get_device
        from src.agents.agent import DuelingDDQNAgent
        from src.agents.networks import DuelingDQN
        from src.agents.neural_ode import NeuralODEDuelingDQN
        from src.agents.hybrid_network import HybridDuelingDQN
        nets = {"mlp": DuelingDQN, "ode": NeuralODEDuelingDQN, "hybrid": HybridDuelingDQN}
        agent = DuelingDDQNAgent(cfg, get_device(device), net_factory=nets[net])
        agent.load(ckpt)
        agent.eps = 0.0

    obs, info = env.reset(seed=seed_base + ep,
                          options={"family": family} if family else None)
    rec = {k: [] for k in STEP_KEYS}
    x0 = float(info.get("ue_x", 0.0))
    term = 0
    for _ in range(env.episode_steps):
        a = agent.act(obs, info, eps=0.0) if agent else a3.act(obs, info)
        obs, r, t, tr, info = env.step(int(a))
        rec["x"].append(float(info.get("ue_x", 0.0)) - x0)
        rec["t"].append(float(info.get("step_idx", len(rec["t"]))) * 0.1)
        rec["rsrp"].append(float(info.get("serving_rsrp", -200.0)))
        rec["sinr"].append(float(info.get("serving_sinr", -30.0)))
        rec["tp_ctrl"].append(float(info.get("throughput_mbps", 0.0)))
        rec["tp_cell"].append(float(info.get("tp_agg_cell_mbps", 0.0)))
        rec["tp_p05"].append(float(info.get("tp_ue_p05_mbps", 0.0)))
        rec["delay"].append(float(info.get("ctrl_delay_ms", 0.0)))
        rec["ho"].append(int(bool(info.get("ho_event"))))
        rec["ho_int"].append(float(info.get("ho_interrupt_ms", 0.0)))
        rec["rlf"].append(int(bool(info.get("rlf"))))
        rec["tunnel"].append(int(info.get("in_tunnel_ns3", 0)))
        rec["n_ue"].append(float(info.get("n_ue_cell", 0.0)))
        rec["load"].append(float(info.get("load_serving", 0.0)))
        if t or tr:
            term = 1 if t else 0
            break
    env.close()
    rec = {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}
    return {"ep": ep, "policy": policy, "seg_len": seg_len, "term": term,
            "steps": int(rec["x"].size), "rec": rec}


def _segments(rec, seg_len):
    """把逐步记录按里程分箱，返回 {seg_idx: {指标}} 与段数。"""
    xs = rec["x"]
    if xs.size == 0:
        return {}, 0
    idx = np.floor(xs / seg_len).astype(int)
    n_seg = int(idx.max()) + 1
    out = {}
    for s in range(n_seg):
        m = idx == s
        if not np.any(m):
            continue
        dly = rec["delay"][m]
        tp = rec["tp_ctrl"][m]
        sinr = rec["sinr"][m]
        # tp_min 用段内 p5：但切换黑障（几百 ms 吞吐≈0）会主导 p5，使所有段都
        # 判为"不达标"（实测 rate_ok_km 恒 0）。故同时给出**中位数**用于达标
        # 判定，并用 zero_frac 单独刻画"吞吐黑洞"（业务连续性），两者不可混用。
        out[s] = dict(
            n=int(m.sum()),
            km=float(s * seg_len / 1000.0),
            tp_p50=float(np.percentile(tp, 50)) if tp.size else 0.0,
            zero_frac=float(np.mean(tp <= 0.01)) if tp.size else 0.0,
            sinr_mean=float(sinr.mean()),
            sinr_min=float(sinr.min()),
            rsrp_min=float(rec["rsrp"][m].min()),
            tp_mean=float(tp.mean()),
            tp_min=float(np.percentile(tp, 5)) if tp.size else 0.0,
            tp_cell_mean=float(rec["tp_cell"][m].mean()),
            tp_p05_mean=float(rec["tp_p05"][m].mean()),
            delay_p50=float(np.percentile(dly, 50)) if dly.size else 0.0,
            delay_p95=float(np.percentile(dly, 95)) if dly.size else 0.0,
            delay_max=float(dly.max()) if dly.size else 0.0,
            ho=int(rec["ho"][m].sum()),
            ho_int_ms=float(rec["ho_int"][m].sum()),
            rlf=int(rec["rlf"][m].sum()),
            tunnel_frac=float(rec["tunnel"][m].mean()),
            n_ue_mean=float(rec["n_ue"][m].mean()),
            load_mean=float(rec["load"][m].mean()),
        )
    return out, n_seg


def _path_summary(segs, n_seg, seg_len, qos_ms, qos_mbps):
    if not segs:
        return {}
    tot_m = n_seg * seg_len
    km = tot_m / 1000.0
    d95 = np.array([s["delay_p95"] for s in segs.values()])
    tmin = np.array([s["tp_min"] for s in segs.values()])
    tmed = np.array([s.get("tp_p50", s["tp_mean"]) for s in segs.values()])
    zero = np.array([s.get("zero_frac", 0.0) for s in segs.values()])
    speed = 83.3   # m/s（300 km/h）；用于把"零速率占比"折成每公里毫秒
    smin = np.array([s["sinr_min"] for s in segs.values()])
    tp = np.array([s["tp_mean"] for s in segs.values()])
    rlf_seg = sum(1 for s in segs.values() if s["rlf"] > 0)
    ho = sum(s["ho"] for s in segs.values())
    hint = sum(s["ho_int_ms"] for s in segs.values())
    k = max(1, int(np.ceil(0.1 * len(tp))))
    return dict(
        total_km=km,
        n_seg=len(segs),
        delay_ok_km=float(np.mean(d95 <= qos_ms)),          # 里程占比
        # 达标判定改用**段内中位速率**（p5 会被切换黑障主导而恒为 0）
        rate_ok_km=float(np.mean(tmed >= qos_mbps)),
        # 业务连续性：吞吐黑洞（≈0）折算成每公里毫秒数
        blackout_ms_per_km=float(np.mean(zero) * seg_len / speed * 1000.0),
        weak_cov_km=float(np.mean(smin < -3.0)),            # SINR 最低点 < -3 dB
        rlf_km_frac=float(rlf_seg / len(segs)),
        ho_per_km=float(ho / km) if km > 0 else 0.0,
        interrupt_ms_per_km=float(hint / km) if km > 0 else 0.0,
        worst10_tp=float(np.median(np.sort(tp)[:k])),       # 尾部风险
        tp_km_mean=float(tp.mean()),
        delay_p95_km=float(np.median(d95)),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ns3_capacity.yaml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--net", choices=["mlp", "ode", "hybrid"], default="hybrid")
    p.add_argument("--both", action="store_true", help="同 seed 跑 A3 与 RL")
    p.add_argument("--n-eps", type=int, default=2)
    p.add_argument("--seg-len", type=float, default=500.0, help="里程段长度 m")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed-base", type=int, default=880000)
    p.add_argument("--dur", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--family", default=None)
    p.add_argument("--n-bg", type=int, default=0,
                   help="覆盖背景 UE 规模（0 = 用配置值）；压测的自变量")
    p.add_argument("--cbr", type=float, default=0.0,
                   help="覆盖控车 CBR Mb/s（0 = 配置值 100 饱和源）")
    p.add_argument("--voip", type=float, default=-1.0, help="VoIP 占比（<0 = 不改）")
    p.add_argument("--video", type=float, default=-1.0, help="视频占比（<0 = 不改）")
    p.add_argument("--dlrb", type=int, default=0,
                   help="覆盖下行 RB 数（100 = 18 MHz；0 = 用配置值）")
    p.add_argument("--qos-ms", type=float, default=150.0)
    p.add_argument("--qos-mbps", type=float, default=2.0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    policies = ["a3", "rl"] if (args.checkpoint and args.both) else \
               (["rl"] if args.checkpoint else ["a3"])
    jobs = [(args.config, ep, args.checkpoint, args.net, pol, args.seg_len,
             args.seed_base, args.dur, args.steps, args.device, args.family,
             args.n_bg, args.cbr, args.voip, args.video, args.dlrb)
            for ep in range(args.n_eps) for pol in policies]
    print(f"整段路径探针：N={args.n_bg if args.n_bg > 0 else '配置值'} × "
          f"{args.n_eps} 集 × {len(policies)} 策略 = {len(jobs)} 任务，"
          f"段长 {args.seg_len:.0f} m，并行 {args.workers}", flush=True)

    got = {(pol, ep): None for pol in policies for ep in range(args.n_eps)}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fu in as_completed([ex.submit(_run_one, j) for j in jobs]):
            r = fu.result()
            got[(r["policy"], r["ep"])] = r
            if args.out:
                try:
                    with open(args.out + ".partial.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"policy": r["policy"], "ep": r["ep"],
                                            "steps": r["steps"], "term": r["term"]},
                                           ensure_ascii=False) + "\n")
                except Exception:
                    pass
            print(f"  [{r['policy']} ep={r['ep']}] steps={r['steps']} "
                  f"term={r['term']}", flush=True)

    # ---- 逐策略：分段 + 路径级汇总 ----
    per_pol = {}
    for pol in policies:
        all_segs, km, n_seg = {}, 0.0, 0
        for ep in range(args.n_eps):
            r = got.get((pol, ep))
            if not r:
                continue
            segs, ns = _segments(r["rec"], args.seg_len)
            km += ns * args.seg_len / 1000.0
            n_seg += ns
            for s, v in segs.items():
                all_segs.setdefault(s, []).append(v)
        merged = {s: {k: float(np.mean([v[k] for v in vs])) for k in vs[0]}
                  for s, vs in all_segs.items()}
        per_pol[pol] = dict(segs=merged,
                            summary=_path_summary(merged, n_seg, args.seg_len,
                                                  args.qos_ms, args.qos_mbps))

    print("\n========== 整段路径聚合指标（里程口径）==========")
    keys = ("total_km", "delay_ok_km", "rate_ok_km", "weak_cov_km", "rlf_km_frac",
            "ho_per_km", "interrupt_ms_per_km", "blackout_ms_per_km", "worst10_tp",
            "tp_km_mean", "delay_p95_km")
    print(f"{'指标':>20s} " + " ".join(f"{p:>10s}" for p in policies))
    for k in keys:
        row = " ".join(f"{per_pol[p]['summary'].get(k, float('nan')):10.3f}"
                       for p in policies)
        print(f"{k:>20s} {row}")

    print("\n========== 里程-指标曲线（每段）==========")
    hdr = (f"{'km':>6s} " + " ".join(
        f"{p+'|SINRmin':>13s} {p+'|tp_min':>11s} {p+'|p95ms':>10s} {p+'|ho':>5s}"
        for p in policies))
    print(hdr)
    seg_ids = sorted(set().union(*[set(per_pol[p]["segs"]) for p in policies]))
    for s in seg_ids:
        cells = []
        for p in policies:
            v = per_pol[p]["segs"].get(s)
            if not v:
                cells.append(f"{'-':>13s} {'-':>11s} {'-':>10s} {'-':>5s}")
            else:
                flag = "*" if v["sinr_min"] < -3.0 or v["delay_p95"] > args.qos_ms \
                    or v["tp_min"] < args.qos_mbps else " "
                cells.append(f"{v['sinr_min']:12.1f}{flag} {v['tp_min']:11.2f} "
                             f"{v['delay_p95']:10.1f} {v['ho']:5.0f}")
        print(f"{per_pol[policies[0]]['segs'].get(s, {}).get('km', s * args.seg_len / 1000.0):6.2f} "
              + " ".join(cells))
    print("（段尾 * = 该段存在不达标项：SINR_min < -3dB / p95 时延超门限 / 最差速率低于门限）")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({p: per_pol[p] for p in policies}, f,
                      ensure_ascii=False, indent=2)
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    sys.exit(main())
