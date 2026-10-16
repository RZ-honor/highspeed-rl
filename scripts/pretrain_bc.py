"""行为克隆预训练：用运营商级 A3 的示范数据初始化 RL 网络。

动机
----
从零 RL 成本高且易退化：ε 退火未完成前 agent 靠随机探索，很容易学到
「永不切换」的局部最优（规避 ho_penalty，却整段掉线）。而标准实践 A3
本身就具备保活能力，把「学会保活」与「学会更优」解耦可大幅降低样本复杂度。

历史事故（必须防止复发）
------------------------
上一版本脚本采集到的示范是 **100% stay**：

    action distribution = [14481, 0, 0, 0, 0, 0, 0]
    bc epoch 2/15  acc=100.0%

原因是采集示范时 A3 基线的 L3 滤波尚未修复（A3 一次都不触发）。
用全 0 标签训出的网络是一个纯粹的 "always stay" 分类器——100% 准确率
是平凡解，注入的先验**比不训练更糟**。本脚本因此强制做退化检测，
示范中没有任何切换动作时直接报错退出。

另一处修正：示范按 **episode 序列**组织，而非打散的 (obs, action) 对。
潜变量 ODE 需要沿时间轴展开（BPTT）；打散的样本对无法提供时序结构。
展开采用 **截断 BPTT**（每 ``--seg-len`` 步截断一次梯度，潜状态继续传递），
以把显存占用限制在常数级。

用法:
  python -m scripts.pretrain_bc --config configs/hsr_load.yaml --net ode \
      --n-demo 30 --epochs 15 --out checkpoints_hsr_load/bc_init.pt
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.config import load_config
from src.training.device import get_device, set_seed
from src.envs.channel_3gpp import Channel3GPP
from src.envs.mobility import Mobility
from src.envs.handover_env import HandoverEnv
from src.agents.agent import DuelingDDQNAgent
from src.agents.safety import mask_actions
from src.agents.networks import DuelingDQN
from src.agents.neural_ode import NeuralODEDuelingDQN
from src.agents.hybrid_network import HybridDuelingDQN
from src.agents.latent_ode import LatentODEQNetwork
from src.agents.baselines import A3OperatorBaseline
from src.agents.safety import HO_ACTIONS

# ode      : 潜变量 ODE-RNN——隐状态沿真实时间传播 + 多步外推预测（当前主力）
# ode_feat : 旧的「伪时间特征提取」式 Neural ODE，仅作为消融对照保留
NETS = {
    "mlp": DuelingDQN,
    "ode": LatentODEQNetwork,
    "ode_feat": NeuralODEDuelingDQN,
    "hybrid": HybridDuelingDQN,
}


def collect_demos(cfg, env, policy, n_episodes, seed0, options=None):
    """用给定策略跑 episode，返回 [(obs_seq, act_seq, rew_seq), ...]。

    ``options`` 透传给 ``env.reset``（NS-3 后端用它指定场景族）。
    记录的动作为**实际执行**动作（NS-3 env 会做安全兜底/掩码改写），
    否则 BC 会学到「教师想做 A、环境实际做 B」的错误标签。
    """
    episodes = []
    steps_max = int(cfg["env"]["episode_steps"])
    for i in range(n_episodes):
        if hasattr(policy, "reset"):
            policy.reset()
        obs, info = env.reset(seed=seed0 + i, options=options)
        obs_seq, act_seq, rew_seq = [], [], []
        for _ in range(steps_max):
            a = int(policy.act(obs, info))
            obs_seq.append(np.asarray(obs, dtype=np.float32).copy())
            act_seq.append(a)
            obs, _r, term, trunc, info = env.step(a)
            act_seq[-1] = int(info.get("action_executed", a))
            rew_seq.append(float(_r))
            if term or trunc:
                break
        if len(obs_seq) < 2:
            continue
        episodes.append((np.stack(obs_seq, axis=0),
                         np.asarray(act_seq, dtype=np.int64),
                         np.asarray(rew_seq, dtype=np.float32)))
    return episodes


def check_not_degenerate(episodes, n_actions):
    """退化检测：示范中必须包含切换动作，否则 BC 只会学到 'always stay'。"""
    acts = np.concatenate([a for _, a, _r in episodes])
    counts = np.bincount(acts, minlength=n_actions)
    n_ho = int(sum(counts[a] for a in HO_ACTIONS))
    frac = n_ho / max(len(acts), 1)
    print(f"  collected {len(episodes)} episodes / {len(acts)} steps; "
          f"action distribution = {counts.tolist()}")
    print(f"  切换动作占比 = {frac*100:.2f}%  （{n_ho} 次 / {len(acts)} 步）")
    if n_ho == 0:
        sys.exit(
            "\n[FATAL] 示范中不含任何切换动作 —— BC 只会学到 'always stay'，"
            "注入的先验比不训练更糟。\n"
            "        请检查 A3 教师：进入门限(a3_offset_db)、L3 滤波(l3_alpha)、"
            "TTT(ttt_ms)、\n        以及 env 的兜底是否抢先做了所有切换"
            "（看 compare_hsr 输出的 fallback 指标）。")
    if frac < 0.002:
        print(f"  [WARN] 切换动作仅占 {frac*100:.3f}%，BC 后策略可能过度保守，"
              f"建议提高 --ho-weight 或缩短 TTT。")
    return counts, frac


def discounted_return_to_go(rew_seq, gamma):
    """G_t = Σ_{k≥t} γ^(k-t)·r_k（与 Q 的语义一致，用于尺度对齐）。"""
    g = np.zeros(len(rew_seq), dtype=np.float64)
    acc = 0.0
    for t in range(len(rew_seq) - 1, -1, -1):
        acc = float(rew_seq[t]) + gamma * acc
        g[t] = acc
    return g


def recalibrate_q_scale(net, episodes, gamma, device, latent, seg_len=64):
    """把 BC 的 Q（logit 量级）仿射重标定到**回报量级**。

    BC 用交叉熵训练，Q 输出是 logit 量级（|Q|≈几），而 RL 的 TD 目标是回报
    量级（本环境一集 200~400）。尺度失配会让 RL 训练初期的 TD 误差≈回报量级，
    把 BC 学到的动作排序冲掉（G7.1 实测：BC 直跑 3/3 存活 R=+235，训练后
    survival=0）。这里用示范的折扣 return-to-go 做线性对齐：
        Q' = a·Q + b，  a = std(G)/std(Q)， b = mean(G) − a·mean(Q)

    Dueling 下 q = V + (A − mean A)：对 value 末层取 (a·v + b)、对 advantage
    末层取 (a·A)，则 q' = a·q + b（mean(aA) = a·meanA）。
    """
    qs, gs = [], []
    net.eval()
    with torch.no_grad():
        for obs_seq, _act, rew_seq in episodes:
            x = torch.as_tensor(obs_seq, dtype=torch.float32, device=device)
            if latent:
                q = net.q(net.unroll(x.unsqueeze(1))[:, 0])
            else:
                q = net(x)
            qs += q.max(1).values.cpu().tolist()
            gs += discounted_return_to_go(rew_seq, gamma).tolist()
    if not qs:
        return None
    q_a = np.asarray(qs); g_a = np.asarray(gs)
    a = float(g_a.std() / max(q_a.std(), 1e-6))
    b = float(g_a.mean() - a * q_a.mean())

    def _affine(seq, scale, shift):
        last = seq[-1]
        if not hasattr(last, "weight"):
            return False
        with torch.no_grad():
            last.weight.mul_(scale)
            if last.bias is not None:
                last.bias.mul_(scale)
                if shift:
                    last.bias.add_(shift)
        return True

    ok_v = _affine(getattr(net, "value", []), a, b)
    ok_a = _affine(getattr(net, "advantage", []), a, 0.0)
    if not (ok_v and ok_a):
        return None
    return a, b, float(q_a.mean()), float(g_a.mean())


def deweight_action_logits(net, class_w) -> bool:
    """抵消加权交叉熵对动作 logit 的抬升（BC → RL 初始化的必需校准）。

    Dueling 输出 q_a = V + A_a − mean(A)。令希望的对数偏移 t_a = −log(w_a)，
    直接把 advantage 末层 bias 改 δ_a = t_a 即可：此时 q 的实际偏移为
    t_a − mean(t)，与期望只差一个**所有动作共同的常数** mean(t)，
    不影响 argmax，也不影响 Q 学习中真正重要的相对差。
    （不要按单动作推导去乘 n/(n−1) —— 同时对 k 个切换动作生效时是错的。）
    """
    adv = getattr(net, "advantage", None)
    if adv is None or len(adv) == 0:
        return False
    layer = adv[-1]
    if not hasattr(layer, "bias") or layer.bias is None:
        return False
    with torch.no_grad():
        for a, w in enumerate(class_w.tolist()):
            if w > 1.0:
                layer.bias[a] += float(-math.log(w))
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/hsr_load.yaml")
    p.add_argument("--net", choices=list(NETS), default="ode")
    p.add_argument("--n-demo", type=int, default=30, help="示范 episode 数")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seg-len", type=int, default=64,
                   help="截断 BPTT 的段长（潜变量网络专用）")
    p.add_argument("--ho-weight", type=float, default=0.0,
                   help="切换动作的类别权重上限；0=按逆频率自动设定（上限 20）")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="checkpoints_hsr_load/bc_init.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backend", choices=["python", "ns3"], default="python",
                   help="示范采集环境。**Stage2 在 NS-3 上训练时这里必须选 ns3**："
                        "Python 与 NS-3 存在域差（路损/干扰/切换频率/RSRP 分布不同），"
                        "用 Python 示范训出的 BC 策略迁到 NS-3 会失效"
                        "（实测：BC 后 RL 全集只切 1 次，而行程需 4-5 次切换）。")
    p.add_argument("--no-calibrate-q", action="store_true",
                   help="关闭 Q 尺度校准（BC 的 Q 是 logit 量级，默认会对齐到回报量级）")
    p.add_argument("--min-ho-recall", type=float, default=0.9,
                   help="G4 门禁：切换动作召回率下限；低于此值判定观测不足，"
                        "拒绝产出 ckpt（0 = 关闭门禁）")
    p.add_argument("--family", default=None,
                   help="NS-3 后端下按场景族采集（课程化；默认单 scenario）")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)
    set_seed(args.seed)

    cfg["env"]["legal_actions"] = [0, 1, 2, 5] if args.backend == "ns3" else list(range(int(cfg["env"]["n_actions"])))
    if args.backend == "ns3":
        from src.envs.ns3_env import NS3HandoverEnv
        env = NS3HandoverEnv(cfg, guard=None)
    else:
        env = HandoverEnv(cfg, Channel3GPP(cfg), Mobility(cfg, seed=cfg.get("seed", 42)))

    a3_cfg = cfg.get("a3_operator", {})
    a3_cfg.setdefault("fc_hz", float(cfg["channel"]["fc_hz"]))
    teacher = A3OperatorBaseline(cfg["safety"], a3_cfg, float(cfg["mobility"]["dt_s"]))

    print(f"Collecting {args.n_demo} demo episodes from A3Operator baseline...")
    opts = {"family": args.family} if args.family else None
    episodes = collect_demos(cfg, env, teacher, args.n_demo, args.seed, options=opts)
    if not episodes:
        sys.exit("[FATAL] 未采集到任何有效 episode。")
    counts, ho_frac = check_not_degenerate(episodes, int(cfg["env"]["n_actions"]))

    # ---- 类别权重：示范里 stay 占绝大多数，需对切换动作加权 ----
    n_actions = int(cfg["env"]["n_actions"])
    class_w = torch.ones(n_actions, device=device)
    n_ho = int(sum(counts[a] for a in HO_ACTIONS))
    if n_ho > 0:
        if args.ho_weight > 0:
            w = float(args.ho_weight)
        else:
            # 逆频率加权：使切换类与 stay 类的总梯度量级相近，避免 98%/2% 的
            # 类别失衡把策略压成 "always stay"（上限 20，防止噪声被过度放大）
            w = min(20.0, float(counts.sum()) / (len(HO_ACTIONS) * n_ho))
        for a in HO_ACTIONS:
            class_w[a] = float(w)
    print(f"  class weight: stay=1.0, HO={class_w[1].item():.2f}")

    # 用 agent 容器承载网络，保证 save/load 格式与 RL 训练一致
    agent = DuelingDDQNAgent(cfg, device, net_factory=NETS[args.net])
    net = agent.online_net
    latent = bool(agent.latent)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    print(f"  net={args.net}  latent={latent}  seg_len={args.seg_len if latent else 'n/a'}")

    rng = np.random.default_rng(args.seed)
    net.train()
    for ep in range(args.epochs):
        order = rng.permutation(len(episodes))
        total, correct, n_steps = 0.0, 0, 0
        for ei in order:
            obs_seq, act_seq, _rew = episodes[ei]
            x = torch.as_tensor(obs_seq, dtype=torch.float32, device=device)   # [T, obs]
            y = torch.as_tensor(act_seq, dtype=torch.int64, device=device)
            T = x.shape[0]

            if latent:
                # 截断 BPTT：梯度每 seg_len 步截断一次，潜状态继续传递
                h = net.init_hidden(1, device)
                for s in range(0, T, args.seg_len):
                    xs = x[s:s + args.seg_len].unsqueeze(1)          # [S,1,obs_dim]
                    ys = y[s:s + args.seg_len]
                    hs = net.unroll(xs, h0=h)                        # [S,1,hidden]
                    q = net.q(hs[:, 0])                              # [S, n_actions]
                    q = mask_actions(q, agent.legal_actions)
                    loss = F.cross_entropy(q, ys, weight=class_w)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    h = hs[-1].detach()                              # 截断梯度，保留状态
                    total += float(loss.item()) * len(ys)
                    correct += int((q.argmax(1) == ys).sum().item())
                    n_steps += len(ys)
            else:
                q = net(x)
                q = mask_actions(q, agent.legal_actions)
                loss = F.cross_entropy(q, y, weight=class_w)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item()) * T
                correct += int((q.argmax(1) == y).sum().item())
                n_steps += T
        acc = correct / max(n_steps, 1)
        # 关键：同时报告「切换动作的召回率」，准确率会被 98% 的 stay 淹没
        print(f"  bc epoch {ep+1}/{args.epochs}  loss={total/max(n_steps,1):.4f}  "
              f"acc={acc*100:.1f}%", flush=True)

    # 训练后统计动作分布 + **切换召回率**（G4 观测充分性门禁）
    pred_counts = np.zeros(n_actions, dtype=np.int64)
    ho_hit = ho_true = 0
    net.eval()
    with torch.no_grad():
        for obs_seq, act_seq, _rew in episodes:
            x = torch.as_tensor(obs_seq, dtype=torch.float32, device=device)
            if latent:
                q = net.q(net.unroll(x.unsqueeze(1))[:, 0])
            else:
                q = net(x)
            q = mask_actions(q, agent.legal_actions)
            pa = q.argmax(1).cpu().numpy()
            pred_counts += np.bincount(pa, minlength=n_actions)
            ho_true += int(np.isin(act_seq, HO_ACTIONS).sum())
            ho_hit += int((np.isin(act_seq, HO_ACTIONS) & np.isin(pa, HO_ACTIONS)).sum())
    ho_recall = ho_hit / max(ho_true, 1)
    print(f"  **切换召回率 = {ho_recall*100:.1f}%  （{ho_hit}/{ho_true} 个教师切换步被预测为切换）**")
    n_pred_ho = int(sum(pred_counts[a] for a in HO_ACTIONS))
    print(f"  BC 后预测动作分布 = {pred_counts.tolist()}（切换 {n_pred_ho} 次）")
    if n_pred_ho == 0:
        sys.exit("[FATAL] BC 后策略从不预测切换 —— 初始化失败，"
                 "请提高 --ho-weight 或增加 --epochs。")

    # ---- 去加权校准（关键）----
    # 加权交叉熵学到的 p̃ ∝ w·p，即切换动作的 logit 比真实后验高 log(w)
    # （本例 w=17.74 → +2.87）。BC 分类要的是「召回少数类」，但 RL 初始化
    # 要的是 **Q 值**：直接沿用会把 Q(HO) 系统性抬高，实测表现为 RL 在服务
    # SINR=+15.5dB 时切向 SINR=-16dB 的小区、3-4 步后 RLF（scripts/
    # diagnose_rl.py 的致命切换细节）。故存盘前把 log(w) 减回去。
    if deweight_action_logits(net, class_w):
        print(f"  已做去加权校准：切换动作 logit -{math.log(class_w[1].item()):.2f}"
              f"（抵消 class weight 带来的偏置）")

    # ---- Q 尺度校准（G7.1 实测驱动）----
    # BC 输出是 logit 量级，RL 需要回报量级；不对齐会在训练初期被大 TD 误差冲掉
    if not args.no_calibrate_q:
        res = recalibrate_q_scale(net, episodes, float(cfg["agent"]["gamma"]),
                                  device, latent, args.seg_len)
        if res:
            a, b, q_m, g_m = res
            print(f"  已做 Q 尺度校准：Q' = {a:.3f}·Q + {b:.2f}"
                  f"（Q 均值 {q_m:.2f} → 回报均值 {g_m:.1f}）")
        else:
            print("  [WARN] Q 尺度校准跳过（未取到 value/advantage 末层）")

    # ---- G4 门禁：观测充分性 ----
    # BC 是**监督学习**：如果连拟合教师的切换决策都做不到，说明观测里根本
    # 没有「何时切、切向谁」的信息，此时进 RL 必然失败（v16 实测：NS-3 域
    # 14 维下 acc 98.3% 却全预测 stay）。把它前置为硬门禁，禁止带病进入 Stage2。
    if ho_recall < args.min_ho_recall:
        sys.exit(f"[G4 FAIL] 切换召回 {ho_recall*100:.1f}% < 门槛 {args.min_ho_recall*100:.0f}%："
                 f"观测不足以判别教师的切换决策。\n"
                 f"  排查顺序：扩充观测（候选 SINR/负载/隧道指示）→ 增加示范集数 "
                 f"→ 提高 --ho-weight / --epochs。禁止用该 ckpt 启动 Stage2。")
    print(f"  [G4 PASS] 切换召回 {ho_recall*100:.1f}% ≥ {args.min_ho_recall*100:.0f}%")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # 与 DuelingDDQNAgent.save 格式一致，便于 train_stress --resume
    torch.save({"online": net.state_dict(), "eps": 0.2}, args.out)
    print(f"Saved BC init -> {args.out}")


if __name__ == "__main__":
    main()
