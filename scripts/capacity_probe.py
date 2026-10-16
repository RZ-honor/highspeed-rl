"""高铁多用户容量/速率/时延/带宽验证探针（100~1000 用户规模）。

为什么需要它
------------
此前的全部指标都是**控车单用户**口径（吞吐/存活/切换次数），回答不了运营真正
关心的问题：
  * 一节车厢/一个小区能带多少用户？（容量极限）
  * 每人能分到多少带宽？最差用户会不会饿死？（速率与公平性）
  * 时延（尤其是切换中断、拥塞排队）到多少毫秒就不可用了？（时延门限）
  * 业务量（小区聚合吞吐、频谱效率）在什么点饱和？（带宽极限）

设计（指标定义）
----------------
对每个用户规模 N 跑固定策略（默认 A3 规则，可用 --checkpoint 换成 RL），
采集逐步的 info，产出：

  【速率】
    ctrl_tp_p05/p50/p95   控车吞吐分位（Mbps）
    tp_agg_mean           全网聚合吞吐（Mbps，全部 UE）
    tp_cell_mean          服务小区聚合吞吐（Mbps）
    per_user_mean         人均吞吐 = **服务小区聚合** / 小区附着用户数
                          ⚠ 必须用小区口径：全网聚合 / 单小区用户数会高估约
                          n_cells 倍（HSR2 的缺陷，HSR3 已修）
    tp_ue_p05             边缘用户速率 = per-UE 平均吞吐的 5 分位（Mbps）
    tp_ue_p50             中位用户速率（Mbps）
    jain                  公平性指数 (Σx)²/(n·Σx²)
    spec_eff              频谱效率 = 小区聚合 Mbps / 带宽 MHz（bps/Hz）
  【时延】
    ctrl_delay_p50/p95    区间平均单向时延的跨区间分位（ms，非包级 p95）
    ho_interrupt_mean     RRC Start→EndOk 时长均值（ms，非业务包间隙）
  【带宽】
    bw_mhz                配置带宽（由 dlRb 换算：100 RB ≈ 20 MHz）
    rb_total              可用 RB 数
  【可靠性 / 移动性】
    survival              控车存活率（跑满 episode 的比例）
    rlf_rate              RLF 步骤占比
    ho_per_km             切换次数 / 行驶公里（移动性代价）
    pingpong_rate         乒乓切换占比
  【容量极限（核心产出）】
    在 QoS 门限下求解最大可承载用户数 N*：
      门限：per_user ≥ --qos-mbps（默认 2 Mbps）
            且 ctrl_delay_p95 ≤ --qos-ms（默认 150 ms）
            且 survival ≥ 0.95
    输出每个 N 是否达标，并给出**最后一个达标的 N**（饱和点）与首次违反的门限。

墙钟/内存（P0 外推依据）
------------------------
每个任务回报 wall_s（单集墙钟）与 rss_mb（进程峰值内存）。100~1000 用户的
可行性必须由这两个数外推决定：NS-3 是单线程仿真，墙钟随 UE 数超线性增长，
ns3.timeout_s（默认 600s）与容器内存都可能先于「容量极限」成为拦路石。

用法：
  # 标称压测（固定场景 + 固定业务配比，只扫 N）
  PYTHONPATH=. python -m scripts.capacity_probe --config configs/ns3_stageH8.yaml \
      --users 100,200,400 --n-eps 3 --fixed-scene --voip 0.2 --video 0.4
  # 快速可行性标定（先看墙钟/内存，再看容量）
  PYTHONPATH=. python -m scripts.capacity_probe --config configs/ns3_stageH8.yaml \
      --users 32,100,200 --n-eps 1 --fixed-scene --attach-stagger 5 --skip-lost-pkts
  # RL vs A3
  PYTHONPATH=. python -m scripts.capacity_probe --config configs/ns3_stageH8.yaml \
      --checkpoint checkpoints_stageH8/best.pt --users 100,200 --n-eps 2
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from src.utils.config import load_config
from src.envs.ns3_env import NS3HandoverEnv
from src.agents.baselines import A3Baseline


def _run_one(job):
    (cfg_path, n_users, ep, ckpt, net, family, scene, voip, video,
     stagger, skip_lost, dur, steps, device, policy, cbr) = job
    cfg = load_config(cfg_path)
    # 直接指定背景 UE 规模（绕过族的随机区间）
    cfg["ns3"]["scenario"] = dict(cfg["ns3"].get("scenario") or {})
    cfg["ns3"]["scenario"]["n_bg_ue"] = int(n_users)
    # 场景模式三态（此前是二态，会在没给 --fixed-scene 时**强行打开** randomize，
    # 把 ns3_capacity.yaml 里刻意设的 randomize=false 覆盖掉）：
    #   fixed  = 固定场景 + 固定业务配比（压测推荐：只扫 N，消除抽样噪声）
    #   random = 显式随机化（domain randomization 口径）
    #   config = 沿用配置文件，不改
    if scene == "fixed":
        cfg["ns3"]["scenario"]["randomize"] = False
        cfg["ns3"]["scenario"]["voip_frac"] = float(voip)
        cfg["ns3"]["scenario"]["video_frac"] = float(video)
        cfg["ns3"]["scenario"]["bg_cluster_frac"] = 0.0
    elif scene == "random":
        cfg["ns3"]["scenario"]["randomize"] = True
    # 大规模压测的性能/成本开关（透传给 NS-3）
    extra = ""
    if stagger > 0.0:
        extra += f" --attachStaggerS={float(stagger):.1f}"
    if skip_lost:
        extra += " --skipLostPkts=1"
    # 仿真时长：容量是**稳态**量，不一定要跑满 52s。缩短 dur 是压墙钟最直接的
    # 杠杆（墙钟 ≈ 与仿真时长成正比），用于把 1000 UE 拉进可跑范围。
    args_str = str(cfg["ns3"].get("args", ""))
    if dur > 0:
        toks = [t for t in args_str.split() if not t.startswith("--dur=")]
        toks.append(f"--dur={float(dur):g}")
        args_str = " ".join(toks)
    # 控车业务量（--cbrRate，Mb/s）：这是「速率场景」与「容量场景」的分水岭。
    # 默认 100 Mb/s 是**饱和源**，控车一人吃满小区 —— 那样测到的是单用户峰值
    # 而不是容量。要测 N* 必须把控车降到典型业务速率（如 5 Mb/s）。
    if cbr > 0:
        toks = [t for t in args_str.split() if not t.startswith("--cbrRate=")]
        toks.append(f"--cbrRate={float(cbr):g}")
        args_str = " ".join(toks)
    if extra:
        args_str += extra
    cfg["ns3"]["args"] = args_str
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
        # 默认 CPU：RL 前向很小，避免多进程抢同一张卡；--device cuda 可显式把推理
        # 放到 DCU 上（worker 少时用得上，也让卡不至于全程闲着）。
        agent = DuelingDDQNAgent(cfg, get_device(device), net_factory=nets[net])
        agent.load(ckpt)
        agent.eps = 0.0

    obs, info = env.reset(seed=770000 + ep, options={"family": family} if family else None)
    rec = dict(ctrl_tp=[], tp_agg=[], tp_cell=[], tp_p05=[], tp_p50=[], jain=[],
               n_ue=[], n_bg=[], delay=[], ho_int=[], load_srv=[],
               ho=0, pp=0, steps=0, rlf=0, term=0, dist_km=0.0)
    x0 = float(info.get("ue_x", 0.0))
    # 墙钟/内存：100~1000 用户可行性的外推依据（P0）
    try:
        import resource
        rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:                      # pragma: no cover - Windows
        resource = None
        rss0 = 0
    t_wall = time.time()
    for _ in range(env.episode_steps):
        a = agent.act(obs, info, eps=0.0) if agent else a3.act(obs, info)
        obs, r, t, tr, info = env.step(int(a))
        rec["ctrl_tp"].append(float(info.get("throughput_mbps", 0.0)))
        rec["tp_agg"].append(float(info.get("tp_agg_mbps", 0.0)))
        rec["tp_cell"].append(float(info.get("tp_agg_cell_mbps", 0.0)))
        rec["tp_p05"].append(float(info.get("tp_ue_p05_mbps", 0.0)))
        rec["tp_p50"].append(float(info.get("tp_ue_p50_mbps", 0.0)))
        rec["jain"].append(float(info.get("jain", 0.0)))
        rec["n_ue"].append(float(info.get("n_ue_cell", 0.0)))
        rec["n_bg"].append(float(info.get("n_bg_total", 0.0)))
        rec["load_srv"].append(float(info.get("load_serving", 0.0)))
        rec["delay"].append(float(info.get("ctrl_delay_ms", 0.0)))
        rec["ho_int"].append(float(info.get("ho_interrupt_ms", 0.0)))
        rec["ho"] += int(bool(info.get("ho_event")))
        rec["pp"] += int(bool(info.get("pingpong")))
        rec["rlf"] += int(bool(info.get("rlf")))
        rec["steps"] += 1
        if t or tr:
            rec["term"] = 1 if t else 0
            break
    rec["dist_km"] = abs(float(info.get("ue_x", 0.0)) - x0) / 1000.0
    rec["wall_s"] = time.time() - t_wall
    # RUSAGE_CHILDREN：内存大头在 NS-3 **子进程**里，只量 SELF 会恒定看到
    # Python 父进程的 354 MB（实测三个规模都是同一个数，即漏掉了 NS-3）。
    try:
        kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss if resource else 0
        me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else 0
        rec["rss_mb"] = max(kids, me) / 1024.0
    except Exception:
        rec["rss_mb"] = 0.0
    env.close()
    return {"n_users": n_users, "ep": ep, "rec": rec, "policy": policy}


def _agg(reps, load_denom=32.0):
    def cat(k):
        out = []
        for r in reps:
            out.extend(r["rec"][k])
        return np.asarray(out, dtype=np.float64)

    ctrl = cat("ctrl_tp")
    agg = cat("tp_agg")
    cell = cat("tp_cell")
    p05 = cat("tp_p05")
    p50 = cat("tp_p50")
    jn = cat("jain")
    nue = cat("n_ue")
    nbg = cat("n_bg")
    lsrv = cat("load_srv")
    dly = cat("delay")
    hint = cat("ho_int")
    ho = float(sum(r["rec"]["ho"] for r in reps))
    pp = float(sum(r["rec"]["pp"] for r in reps))
    rlf = float(sum(r["rec"]["rlf"] for r in reps))
    steps = float(sum(r["rec"]["steps"] for r in reps))
    km = float(sum(r["rec"]["dist_km"] for r in reps))
    surv = float(np.mean([1.0 - r["rec"]["term"] for r in reps]))
    # 人均速率：**服务小区聚合** / 小区附着用户数（HSR3 口径）
    m = nue > 0
    per_user = float(np.mean(cell[m] / np.maximum(nue[m], 1.0))) if np.any(m) else 0.0
    # 口径 B：全网人均 = 全网聚合 / (背景 UE 总数 + 控车)。
    # 用途：与「小区人均」互为交叉验证 —— 若两者差很多，说明服务小区口径被
    # 隧道微站（背景 UE 极少）或负载不均污染，此时应以口径 B 报告容量结论。
    net_m = nbg >= 0
    per_user_net = (float(np.mean(agg[net_m] / np.maximum(nbg[net_m] + 1.0, 1.0)))
                    if np.any(net_m) else 0.0)
    return dict(
        ctrl_tp_p05=float(np.percentile(ctrl, 5)) if len(ctrl) else 0.0,
        ctrl_tp_p50=float(np.percentile(ctrl, 50)) if len(ctrl) else 0.0,
        ctrl_tp_p95=float(np.percentile(ctrl, 95)) if len(ctrl) else 0.0,
        tp_agg_mean=float(np.mean(agg)) if len(agg) else 0.0,
        tp_cell_mean=float(np.mean(cell)) if len(cell) else 0.0,
        tp_ue_p05=float(np.mean(p05[p05 > 0])) if np.any(p05 > 0) else 0.0,
        tp_ue_p50=float(np.mean(p50[p50 > 0])) if np.any(p50 > 0) else 0.0,
        jain=float(np.mean(jn[jn > 0])) if np.any(jn > 0) else 0.0,
        n_ue_mean=float(np.mean(nue)) if len(nue) else 0.0,
        per_user_mean=per_user,          # 口径 A：小区人均
        per_user_net_mean=per_user_net,  # 口径 B：全网人均（交叉验证）
        delay_p50=float(np.percentile(dly, 50)) if len(dly) else 0.0,
        delay_p95=float(np.percentile(dly, 95)) if len(dly) else 0.0,
        ho_int_mean=float(np.mean(hint[hint > 0])) if np.any(hint > 0) else 0.0,
        ho=ho, ho_per_km=(ho / km) if km > 0 else 0.0,
        pingpong_rate=(pp / steps) if steps else 0.0,
        rlf_rate=(rlf / steps) if steps else 0.0,
        survival=surv,
        # 可行性（P0 外推用）
        wall_s_mean=float(np.mean([r["rec"].get("wall_s", 0.0) for r in reps])),
        wall_s_max=float(np.max([r["rec"].get("wall_s", 0.0) for r in reps])),
        rss_mb_max=float(np.max([r["rec"].get("rss_mb", 0.0) for r in reps])),
        # 负载观测饱和比例：load_serving 归一化分母默认 32，达到即观测饱和
        # （RL 的负载维度失去分辨力 → 大规模评估结论不可直接用 H8 权重解释）
        load_sat_ratio=(float(np.mean(lsrv >= load_denom)) if len(lsrv) else 0.0),
    )


def _boot_ci(deltas, n_boot=5000, seed=0):
    """配对差的 bootstrap 95% 置信区间（不依赖 scipy，无需分布假设）。"""
    d = np.asarray(deltas, dtype=np.float64)
    if d.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ns3_stageH8.yaml")
    p.add_argument("--checkpoint", default=None, help="不给则用 A3 规则")
    p.add_argument("--net", choices=["mlp", "ode", "hybrid"], default="hybrid")
    p.add_argument("--users", default="50,100,200,400")
    p.add_argument("--n-eps", type=int, default=2)
    p.add_argument("--family", default=None)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--qos-mbps", type=float, default=2.0)
    p.add_argument("--qos-ms", type=float, default=150.0)
    p.add_argument("--qos-jain", type=float, default=0.0,
                   help="公平性下限（0 = 不设门限）")
    p.add_argument("--fixed-scene", action="store_true",
                   help="固定场景与业务配比（压测推荐）：只扫 N，消除抽样噪声")
    p.add_argument("--random-scene", action="store_true",
                   help="显式打开场景随机化（不给此/--fixed-scene 时沿用配置文件）")
    p.add_argument("--voip", type=float, default=0.2, help="--fixed-scene 下 VoIP 占比")
    p.add_argument("--video", type=float, default=0.4, help="--fixed-scene 下视频占比")
    p.add_argument("--attach-stagger", type=float, default=0.0,
                   help="背景 UE 接入错峰窗口 s（0 = 全部 t=0 同时接入）")
    p.add_argument("--skip-lost-pkts", action="store_true",
                   help="跳过 FlowMonitor 丢包统计（大规模省 CPU）")
    p.add_argument("--dur", type=float, default=0.0,
                   help="覆盖 NS-3 仿真时长 s（0 = 用配置值）；压墙钟的直接杠杆")
    p.add_argument("--steps", type=int, default=0,
                   help="覆盖 env.episode_steps（与 --dur 配套）")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                   help="RL 推理设备；cuda = 用 DCU（容器可见 1 张，worker 多时慎用）")
    p.add_argument("--load-denom", type=float, default=32.0,
                   help="负载观测归一化分母（用于饱和告警）")
    p.add_argument("--cbr", type=float, default=0.0,
                   help="覆盖控车 CBR 速率 Mb/s（0 = 用配置值 100，即饱和源）；"
                        "测容量应降到典型业务速率（如 5）")
    p.add_argument("--both", action="store_true",
                   help="同 seed 同时跑 A3 与 RL 并给出配对差 Δ 与 bootstrap CI")
    p.add_argument("--out", default=None, help="结果 JSON 落盘路径")
    args = p.parse_args()

    # 策略集：默认只跑一种（给了 ckpt 就跑 RL）；--both 则同 seed 配对跑两种
    policies = ["a3"]
    if args.checkpoint:
        policies = ["a3", "rl"] if args.both else ["rl"]

    users = [int(s) for s in args.users.split(",") if s.strip()]
    cfg = load_config(args.config)
    # RB 数在 ns3 命令行里（--dlRb=100），不在 channel 段
    rb = 100.0
    for tok in str(cfg["ns3"].get("args", "")).split():
        if tok.startswith("--dlRb="):
            try:
                rb = float(tok.split("=", 1)[1])
            except ValueError:
                pass
    bw_mhz = rb * 0.18  # 1 RB ≈ 180 kHz

    scene = ("fixed" if args.fixed_scene
             else "random" if args.random_scene else "config")
    jobs = [(args.config, n, ep, args.checkpoint, args.net, args.family,
             scene, args.voip, args.video,
             args.attach_stagger, args.skip_lost_pkts, args.dur, args.steps,
             args.device, pol, args.cbr)
            for n in users for ep in range(args.n_eps) for pol in policies]
    print(f"容量探针：{len(users)} 个规模 × {args.n_eps} 集 × {len(policies)} 策略"
          f" = {len(jobs)} 任务，并行 {args.workers}；带宽 {bw_mhz:.1f} MHz"
          f"（{rb:.0f} RB）；场景={scene}；"
          f"控车 CBR={'配置值' if args.cbr <= 0 else f'{args.cbr:g} Mb/s'}", flush=True)

    reps = {(n, pol): [] for n in users for pol in policies}
    # 增量落盘：单集墙钟在 100~1000 规模下可达数十分钟，若整轮被 timeout 杀掉
    # 而结果只在最后写盘，就等于白跑。每个任务一完成立刻追加一行 JSONL。
    part = (args.out + ".partial.jsonl") if args.out else None
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_run_one, j) for j in jobs]
        done = 0
        for fu in as_completed(futs):
            r = fu.result()
            reps[(r["n_users"], r["policy"])].append(r)
            done += 1
            rec = r["rec"]
            print(f"  [{done}/{len(jobs)}] N={r['n_users']} {r['policy']:>3s} "
                  f"ep={r['ep']} "
                  f"wall={rec.get('wall_s', 0.0):.0f}s "
                  f"rss={rec.get('rss_mb', 0.0):.0f}MB", flush=True)
            if part:
                try:
                    with open(part, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "n_users": r["n_users"], "ep": r["ep"],
                            "policy": r["policy"],
                            "steps": rec["steps"], "term": rec["term"],
                            "wall_s": rec.get("wall_s", 0.0),
                            "rss_mb": rec.get("rss_mb", 0.0),
                            "tp_cell_mean": (float(np.mean(rec["tp_cell"]))
                                             if rec["tp_cell"] else 0.0),
                            "tp_ue_p05_mean": (float(np.mean(rec["tp_p05"]))
                                               if rec["tp_p05"] else 0.0),
                            "n_ue_mean": (float(np.mean(rec["n_ue"]))
                                          if rec["n_ue"] else 0.0),
                            "delay_p95": (float(np.percentile(rec["delay"], 95))
                                          if rec["delay"] else 0.0),
                        }, ensure_ascii=False) + "\n")
                except Exception as e:      # 落盘失败不应中断测量
                    print(f"  [warn] 增量落盘失败: {e}", flush=True)

    rows = []
    print("\n========== 容量 / 速率 / 时延 / 带宽 画像 ==========")
    header = (f"{'N':>5s} {'pol':>3s} {'人均Mbps':>9s} {'网人均':>8s} {'边缘p05':>8s} {'中位p50':>8s} "
              f"{'公平性':>7s} {'小区Mbps':>9s} {'频谱效率':>8s} {'时延p50':>8s} "
              f"{'时延p95':>8s} {'切中断ms':>9s} {'切换/km':>8s} {'RLF率':>7s} "
              f"{'存活':>6s} {'墙钟s':>7s} {'RSS_MB':>8s} {'达标':>5s}")
    print(header)
    for n in users:
        for pol in policies:
            m = _agg(reps[(n, pol)], load_denom=args.load_denom)
            m["n_users"] = n
            m["policy"] = pol
            # 频谱效率用**小区**聚合（与人均同一口径）
            m["spec_eff"] = (m["tp_cell_mean"] * 1e6) / (bw_mhz * 1e6) if bw_mhz > 0 else 0.0
            ok_tp = m["per_user_mean"] >= args.qos_mbps
            ok_dl = m["delay_p95"] <= args.qos_ms
            ok_sv = m["survival"] >= 0.95
            ok_jn = (args.qos_jain <= 0.0) or (m["jain"] >= args.qos_jain)
            m["qos_ok"] = bool(ok_tp and ok_dl and ok_sv and ok_jn)
            m["violation"] = ([] if ok_tp else ["人均速率"]) + ([] if ok_dl else ["时延"]) \
                + ([] if ok_sv else ["存活"]) + ([] if ok_jn else ["公平性"])
            rows.append(m)
            print(f"{n:5d} {pol:>3s} {m['per_user_mean']:9.2f} {m['per_user_net_mean']:8.2f} "
                  f"{m['tp_ue_p05']:8.2f} "
                  f"{m['tp_ue_p50']:8.2f} {m['jain']:7.3f} {m['tp_cell_mean']:9.2f} "
                  f"{m['spec_eff']:8.2f} {m['delay_p50']:8.1f} "
                  f"{m['delay_p95']:8.1f} {m['ho_int_mean']:9.1f} {m['ho_per_km']:8.2f} "
                  f"{m['rlf_rate']:7.3f} {m['survival']:6.2f} {m['wall_s_mean']:7.0f} "
                  f"{m['rss_mb_max']:8.0f} {'PASS' if m['qos_ok'] else 'FAIL':>5s}")

    # ---- A3 vs RL 配对对比（同 seed / 同规模，逐 seed 取差 + bootstrap CI）----
    if len(policies) == 2:
        print("\n========== A3 vs RL 配对对比（同 seed，Δ = RL − A3）==========")
        print(f"{'N':>5s} {'Δ网人均':>9s} {'Δ%':>7s} {'Δ边缘p05':>9s} {'Δ公平性':>8s} "
              f"{'Δ延时p95':>9s} {'Δ存活':>7s} {'显著':>6s}")
        for n in users:
            a3 = [r for r in rows if r["n_users"] == n and r["policy"] == "a3"]
            rl = [r for r in rows if r["n_users"] == n and r["policy"] == "rl"]
            if not a3 or not rl:
                continue
            # 按 ep 配对：两侧 seed 相同（770000+ep），逐 ep 取差
            def per_ep(pol_side):
                out = {}
                for r in reps[(n, pol_side)]:
                    rec = r["rec"]
                    out[r["ep"]] = (
                        float(np.mean(rec["tp_agg"])) / max(float(np.mean(rec["n_bg"])) + 1.0, 1.0)
                        if rec["tp_agg"] else 0.0,
                        float(np.mean(rec["tp_p05"])) if rec["tp_p05"] else 0.0,
                        float(np.mean(rec["jain"])) if rec["jain"] else 0.0,
                        float(np.percentile(rec["delay"], 95)) if rec["delay"] else 0.0,
                        1.0 - float(rec["term"]),
                    )
                return out
            ea, er = per_ep("a3"), per_ep("rl")
            keys = sorted(set(ea) & set(er))
            if not keys:
                continue
            d_net = np.array([er[k][0] - ea[k][0] for k in keys])
            d_p05 = np.array([er[k][1] - ea[k][1] for k in keys])
            d_jn = np.array([er[k][2] - ea[k][2] for k in keys])
            d_dl = np.array([er[k][3] - ea[k][3] for k in keys])
            d_sv = np.array([er[k][4] - ea[k][4] for k in keys])
            base = float(np.mean([ea[k][0] for k in keys])) or 1e-9
            lo, hi = _boot_ci(d_net)
            sig = "是" if (lo > 0 and hi > 0) or (lo < 0 and hi < 0) else "否"
            print(f"{n:5d} {d_net.mean():+9.3f} {100.0 * d_net.mean() / base:+7.1f} "
                  f"{d_p05.mean():+9.3f} {d_jn.mean():+8.3f} {d_dl.mean():+9.1f} "
                  f"{d_sv.mean():+7.2f} {sig:>6s}")
            print(f"      网人均 95%CI [{lo:+.3f}, {hi:+.3f}]  "
                  f"（口径：全网聚合/(背景UE+控车)，配对样本 n={len(keys)}）")

    print("\n========== 可行性（外推到 1000 用户的依据）==========")
    for r in rows:
        print(f"  N={r['n_users']:5d}  单集墙钟 {r['wall_s_mean']:7.0f}s"
              f"（峰值 {r['wall_s_max']:.0f}s）  峰值内存 {r['rss_mb_max']:.0f} MB"
              f"  负载观测饱和 {r['load_sat_ratio'] * 100:.0f}%")
    sat = [r for r in rows if r["load_sat_ratio"] > 0.5]
    if sat:
        print("  [WARN] 负载观测在部分/全部规模上饱和（load_serving ≥ "
              f"{args.load_denom:.0f}）：用 H8 权重解释这些规模的策略行为不可靠，"
              "下一阶段需提高 load_norm_denom 或改对数归一化。")

    ok = [r for r in rows if r["qos_ok"]]
    print(f"\nQoS 门限：人均 ≥ {args.qos_mbps} Mbps、时延 p95 ≤ {args.qos_ms} ms、"
          f"存活 ≥ 0.95" + (f"、公平性 ≥ {args.qos_jain}" if args.qos_jain > 0 else ""))
    if ok:
        best = max(ok, key=lambda r: r["n_users"])
        print(f"=> 容量极限（最大达标规模）N* = {best['n_users']}  "
              f"（人均 {best['per_user_mean']:.2f} Mbps、时延 p95 {best['delay_p95']:.0f} ms）")
    else:
        print("=> 全部规模均不达标：需降低门限、扩带宽（更多 RB）或减少单用户业务量")
    # N* 按策略分别报（A3 与 RL 的达标规模可能不同，这正是我们要测的东西）
    for pol in policies:
        sub = [r for r in rows if r.get("policy") == pol]
        ok_p = [r for r in sub if r["qos_ok"]]
        if ok_p:
            b = max(ok_p, key=lambda r: r["n_users"])
            print(f"=> [{pol}] 容量极限 N* = {b['n_users']}  "
                  f"（人均 {b['per_user_mean']:.2f} Mbps、时延 p95 {b['delay_p95']:.0f} ms）")
        else:
            print(f"=> [{pol}] 全部规模均不达标")
    nxt = next((r for r in rows if not r["qos_ok"]), None)
    if nxt:
        print(f"=> 首次违反于 N={nxt['n_users']}（{nxt.get('policy', '')}，"
              f"{', '.join(nxt['violation'])}）")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    sys.exit(main())
