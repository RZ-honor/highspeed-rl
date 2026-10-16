"""独立评估：加载已训练 checkpoint，用 greedy 策略 (eps=0) 跑若干 episode 验证效果。
也可跑运营商级 A3 基准（--baseline a3），在同样的 NS-3 仿真环境中对比。

用法:
  # RL 模型评估
  python -m scripts.eval_ns3 --config configs/ns3.yaml --device cuda --net hybrid \\
      --resume checkpoints_ns3/latest.pt --n-eval 5
  # A3 基准（同一 NS-3 环境，A3OperatorBaseline 决策，与采集教师同策略）
  python -m scripts.eval_ns3 --config configs/ns3.yaml --baseline a3 --n-eval 5

场景由 config 的 ns3.scenario 控制（randomize:false + 固定 n_bg_ue/tunnels
即可复现 normal/tunnel/congested 场景；见 configs/eval_*.yaml）。
"""

import argparse
import time

import numpy as np
import torch
from src.agents.safety import legal_actions, mask_actions, select_action

from src.utils.config import load_config
from src.training.device import get_device, set_seed
from src.agents.networks import DuelingDQN
from src.agents.neural_ode import NeuralODEDuelingDQN
from src.agents.hybrid_network import HybridDuelingDQN

NETS = {
    "mlp": DuelingDQN,
    "ode": NeuralODEDuelingDQN,
    "hybrid": HybridDuelingDQN,
}


def run_one(env, net, pol, ep_seed, device, legal, wall_budget_s=480.0):
    """跑单个 episode。net/pol 二选一：net=RL 贪心网；pol=A3 基准（None 互补）。

    wall_budget_s：episode 墙钟上限。env 的 I/O watchdog 只在"无应答"时杀
    NS-3；高背景 UE 负载下可能出现"每步都应答但仿真病态慢"（实测 48 背景 UE
    单集 >50min），必须有墙钟兜底——超时终止本集并计入 outage（env.close
    会 kill 子进程，由 finally 统一处理）。
    """
    obs, info = env.reset(seed=ep_seed)
    if pol is not None:
        pol.reset()
    t0 = time.time()
    ep_r = ep_tp = ho = pp = out = steps = 0
    timed_out = False
    # 动作分布（实际执行动作 0..6）与决策来源占比（rl/safety_a3/masked_dual）。
    # 这是区分「RL 真的学会了不同策略」还是「RL 坍缩成 A3 克隆」的直接证据：
    # 若 RL 的动作直方图与 A3 完全一致，吞吐差异只能来自仿真抖动。
    act_hist = np.zeros(7, dtype=np.int64)
    src_count = {"rl": 0, "safety_a3": 0, "masked_dual": 0, "masked_weak": 0}
    # Hybrid 软路由 α 采集：全序列 + HO 时刻对齐（验证"跳变点 α 升高"假设，
    # 见计划 Task 2.3 / ns3_next_steps §4.2）。非 hybrid 网保持 None。
    has_route = net is not None and hasattr(net, "route_weight")
    alphas, ho_alphas = ([], []) if has_route else (None, None)
    last_alpha = None
    for _ in range(env.episode_steps):
        if net is not None:
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32,
                                    device=device).unsqueeze(0)
                a = select_action(net(x), env.cfg, info, actions=legal)
            if has_route:
                last_alpha = float(net.route_weight(x).item())
                alphas.append(last_alpha)
        else:
            a = int(pol.act(obs, info))
        obs, r, term, trunc, info = env.step(a)
        ep_r += r
        ep_tp += info.get("throughput_mbps", 0.0)
        steps += 1
        act_hist[int(info.get("action_executed", a))] += 1
        src_count[str(info.get("action_source", "rl"))] += 1
        if info.get("ho_event"):
            ho += 1
            if has_route and last_alpha is not None:
                ho_alphas.append(last_alpha)
        if info.get("pingpong"):
            pp += 1
        if term:
            out += 1
        if term or trunc:
            break
        if time.time() - t0 > wall_budget_s:
            # 病态慢：剩余步数按失服计，避免单集拖垮整批评估
            timed_out = True
            out += 1
            break
    alpha_stats = {"alpha_mean": None, "alpha_std": None,
                   "alpha_ho_mean": None, "alpha_nonho_mean": None}
    if has_route and alphas:
        # ho_alphas 存的是 HO 步当时的 α（与 alphas 中对应步同值），
        # 非 HO 步均值由总和代数相减得到。
        alpha_stats = {
            "alpha_mean": float(np.mean(alphas)),
            "alpha_std": float(np.std(alphas)),
            "alpha_ho_mean": float(np.mean(ho_alphas)) if ho_alphas else None,
            "alpha_nonho_mean": (
                float((np.sum(alphas) - np.sum(ho_alphas))
                      / max(len(alphas) - len(ho_alphas), 1))
                if ho_alphas else float(np.mean(alphas))),
        }
    return {"reward": ep_r, "throughput": ep_tp / max(steps, 1),
            "ho": ho, "pingpong": pp, "outage": out, "steps": steps,
            "timed_out": timed_out, "act_hist": act_hist,
            "src_count": src_count, **alpha_stats}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ns3.yaml")
    p.add_argument("--device", default="cuda")
    p.add_argument("--net", choices=list(NETS), default="hybrid")
    p.add_argument("--resume", default=None, help="checkpoint 路径（RL 模式）")
    p.add_argument("--baseline", choices=["none", "a3"], default="none",
                   help="a3 = 跑 3GPP 运营商级 A3 基准（同 NS-3 环境）")
    p.add_argument("--n-eval", type=int, default=5)
    p.add_argument("--seed", type=int, default=900000)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(123)

    from src.envs.ns3_env import NS3HandoverEnv
    env = NS3HandoverEnv(cfg)

    net, pol, device, legal = None, None, None, None
    if args.baseline == "a3":
        from src.agents.baselines import A3OperatorBaseline
        a3_cfg = cfg.get("a3_operator", {})
        a3_cfg.setdefault("fc_hz", float(cfg["channel"]["fc_hz"]))
        pol = A3OperatorBaseline(cfg["safety"], a3_cfg,
                                 float(cfg["mobility"]["dt_s"]))
        print(f"Running A3 baseline in ns-3 env "
              f"(offset={pol.a3_offset_db}dB ttt={pol.ttt_ms}ms "
              f"l3_alpha={pol._l3_alpha} raw_enter={pol.raw_enter})")
    else:
        device = get_device(args.device)
        obs_dim = int(cfg["env"]["obs_dim"]); n_act = int(cfg["env"]["n_actions"])
        net = NETS[args.net](obs_dim, n_act, int(cfg["agent"]["hidden_dim"]),
                             **cfg.get("net_kwargs", {})).to(device)
        ck = torch.load(args.resume, map_location=device)
        net.load_state_dict(ck["online"] if "online" in ck else ck)
        net.eval()
        legal = legal_actions(cfg, n_act)
        print(f"Loaded checkpoint: {args.resume}")

    rec = {"reward": [], "throughput": [], "ho": [], "pingpong": [], "outage": []}
    alpha_rec = []  # 仅 Hybrid RL 模式填充；A3/mlp/ode 为 None
    act_names = ["stay", "HO->1st", "HO->2nd", "dual1st", "dual2nd", "HO->3rd", "triple"]
    total_hist = np.zeros(7, dtype=np.int64)
    total_src = {"rl": 0, "safety_a3": 0, "masked_dual": 0, "masked_weak": 0}
    total_steps = 0
    try:
        n_timeout = 0
        n_crash = 0
        for i in range(args.n_eval):
            try:
                m = run_one(env, net, pol, args.seed + i, device, legal)
            except RuntimeError as e:
                # NS-3 abort（如 INITIAL_RANDOM_ACCESS）不应吞掉整批评估：
                # 记 outage=1 并继续，finally 仍打印 src_count / 均值。
                n_crash += 1
                print(f"  eval {i+1}/{args.n_eval}  CRASH: {e}", flush=True)
                rec["reward"].append(0.0)
                rec["throughput"].append(0.0)
                rec["ho"].append(0)
                rec["pingpong"].append(0)
                rec["outage"].append(1)
                alpha_rec.append((None, None, None, None))
                continue
            for k in rec:
                rec[k].append(m[k])
            alpha_rec.append((m["alpha_mean"], m["alpha_std"],
                              m["alpha_ho_mean"], m["alpha_nonho_mean"]))
            total_hist += m["act_hist"]
            for k in total_src:
                total_src[k] += m["src_count"][k]
            total_steps += int(m["act_hist"].sum())
            n_timeout += int(m["timed_out"])
            src_txt = "  src=" + ",".join(
                f"{k}:{v}" for k, v in m["src_count"].items() if v)
            alpha_txt = (f"  alpha={m['alpha_mean']:.3f}±{m['alpha_std']:.3f}"
                         if m["alpha_mean"] is not None else "")
            print(f"  eval {i+1}/{args.n_eval}  steps={m['steps']}  "
                  f"reward={m['reward']:.2f}  tp={m['throughput']:.2f}Mbps  "
                  f"ho={m['ho']}  pp={m['pingpong']}  outage={m['outage']}"
                  f"{src_txt}{alpha_txt}"
                  f"{'  [WALL TIMEOUT]' if m['timed_out'] else ''}", flush=True)
        if n_timeout:
            print(f"  注意：{n_timeout}/{args.n_eval} 集触发墙钟上限"
                  f"（NS-3 仿真病态慢），该集按 outage 计")
        if n_crash:
            print(f"  注意：{n_crash}/{args.n_eval} 集 NS-3 abort，该集按 outage 计")
    finally:
        env.close()

    print("\n=== Eval 平均 ===")
    for k, v in rec.items():
        print(f"  {k:12s} = {np.mean(v):.4f}  (std={np.std(v):.4f})")

    # 动作分布（实际执行口径）：3/4/6 在 LENA 无 EN-DC 被 mask 成 stay，
    # 故直方图中恒为 0；策略想发双连的步数见 masked_dual 来源占比。
    if total_steps > 0:
        print("\n=== 动作分布（实际执行，占全部决策步比例）===")
        for a in range(7):
            if total_hist[a] > 0 or a in (0, 1, 2, 5):
                print(f"  a{a} {act_names[a]:9s}: {total_hist[a]:6d}  "
                      f"({100.0*total_hist[a]/total_steps:5.1f}%)")
        print("  决策来源: " + "  ".join(
            f"{k}={100.0*v/total_steps:.1f}%" for k, v in total_src.items()))

    # Hybrid 软路由 α 汇总（验收：alpha_std > 0.02 非退化常数；
    # HO 步 α 高于非 HO 步 → 支持"跳变点 MLP 主导"假设）
    if any(a[0] is not None for a in alpha_rec):
        am = np.array([a[0] for a in alpha_rec if a[0] is not None])
        asd = np.array([a[1] for a in alpha_rec if a[1] is not None])
        ho_vals = [a[2] for a in alpha_rec if a[2] is not None]
        nonho_vals = [a[3] for a in alpha_rec if a[3] is not None]
        print("\n=== Hybrid 路由 α ===")
        print(f"  alpha_mean = {am.mean():.4f}  alpha_std(ep内均值) = {asd.mean():.4f}")
        if ho_vals:
            print(f"  HO 步 α = {np.mean(ho_vals):.4f}  "
                  f"非 HO 步 α = {np.mean(nonho_vals):.4f}  "
                  f"(n_ho={len(ho_vals)})")


if __name__ == "__main__":
    main()
