"""NS-3 在线微调（Stage 2）：ParallelCollector 并行采集 + 主线程 GPU 学习解耦。

Stage1（pretrain_bc）产出 bc_init.pt 后：
  python -m scripts.train_ns3 --config configs/ns3.yaml --device cuda \
      --net hybrid --init checkpoints_ns3/bc_init.pt
本地无 NS-3 冒烟：
  python -m scripts.train_ns3 --config configs/default.yaml --mock --net mlp
"""

import argparse
import os
import queue
import threading
import time

import numpy as np
import torch
from src.agents.safety import legal_actions, mask_actions, select_action

from src.utils.config import load_config
from src.training.device import get_device, set_seed
from src.agents.agent import DuelingDDQNAgent
from src.agents.sac_agent import DiscreteSACAgent
from src.agents.networks import DuelingDQN
from src.agents.neural_ode import NeuralODEDuelingDQN
from src.agents.hybrid_network import HybridDuelingDQN
from src.envs.parallel_collector import CollectorReplayWriter, ParallelCollector

NETS = {"mlp": DuelingDQN, "ode": NeuralODEDuelingDQN, "hybrid": HybridDuelingDQN}
# agent.algorithm → 学习器类。缺省 dqn 保持全部历史配置的行为不变；
# sac = 离散 soft Q-learning（src/agents/sac_agent.py，H14 实验）。
AGENTS = {"dqn": DuelingDDQNAgent, "sac": DiscreteSACAgent}


def start_dcu_sampler(interval_s=30.0):
    """后台采样 DCU 利用率（hy-smi），返回 (stop_event, samples)。

    为什么需要它：本项目实测「训练中 DCU 利用率 ≈0%」——瓶颈在 NS-3 仿真与锁步
    同步，卡一直在等数据。没有这个采样，我们无法判断「加大 batch / 提高采集并行
    度」这类再平衡手段到底有没有把 DCU 用起来。采不到（登录节点/无 hy-smi）时
    静默降级，不影响训练。
    """
    import subprocess
    import threading

    stop = threading.Event()
    samples = []
    if interval_s <= 0:
        return stop, samples

    def _read_once():
        try:
            out = subprocess.run(["hy-smi"], capture_output=True, text=True,
                                 timeout=10).stdout
        except Exception:
            return None
        for line in out.splitlines():
            tok = line.split()
            # 行形如: 0  47.0C  19.0W  auto  450.0W  0%  0%  Normal
            if len(tok) >= 7 and tok[0].isdigit():
                try:
                    return float(tok[6].rstrip("%"))
                except ValueError:
                    return None
        return None

    def _loop():
        while not stop.is_set():
            v = _read_once()
            if v is not None:
                samples.append(v)
            stop.wait(interval_s)

    threading.Thread(target=_loop, daemon=True).start()
    return stop, samples


def evaluate(agent, cfg, eval_device, n_eps, family=None):
    """在评估设备上用 online 权重快照跑 eps=0 评估（不干扰主线程学习）。

    用与 agent 相同的网络类和 net_kwargs 重建一个评估网（统一签名
    (obs_dim, n_actions, hidden, **net_kwargs)），再载入 online 权重。

    ``family``：固定到某个场景族评估（课程监控用），避免与训练随机化纠缠；
    为 None 时保持旧行为（单 scenario 模式）。
    """
    from src.envs.ns3_env import NS3HandoverEnv
    obs_dim = int(cfg["env"]["obs_dim"]); n_act = int(cfg["env"]["n_actions"])
    net = type(agent.online_net)(obs_dim, n_act, agent.hidden,
                                 **agent.net_kwargs).to(eval_device)
    net.load_state_dict(agent.online_net.state_dict())
    net.eval()
    # 评估步数上限（0 = 整集）。存在的理由（2026-09-24 H10 实测）：NS-3 锁步的
    # 步速约 **1 步/秒**（与并发采集器同量级，不是采集器的锅），故整集 480 步
    # ≈8.3 min/集，2 集必超 15 min 的评估预算 ⇒ 每次评估都被放弃 ⇒ **永远没有
    # best.pt**，而 best.pt 是本阶段唯一可用产物（H9 教训）。
    # 截到 max_steps 只缩短评估覆盖的路径长度，不改变"能识别 ho=0 退化"的能力。
    # ⚠ 必须从 cfg 取步数，**不能**用 env.episode_steps —— env 在下面才构造，
    # 第一次提交就是在这里写成 env.episode_steps 而报 UnboundLocalError，
    # 训练在首次评估时崩溃（TRAIN_EXIT=1）。
    max_steps = int((cfg.get("eval") or {}).get("max_steps", 0) or 0)
    ep_steps = int(cfg["env"]["episode_steps"])
    budget_steps = ep_steps if max_steps <= 0 else min(ep_steps, max_steps)
    rec = {"reward": [], "throughput": [], "ho": [], "pingpong": [], "outage": []}
    # 单个评估 env 复用全程（reset 会自动 respawn 子进程）；try/finally 保证
    # 评估途中 NS-3 崩溃抛异常时子进程也被 close，不泄漏。
    families = (cfg.get("eval") or {}).get("families") or [family]
    if isinstance(families, str):
        families = [s.strip() for s in families.split(",") if s.strip()]
    per_family = {}
    episodes = []
    if n_eps <= 0 or not families or ep_steps <= 0:
        raise ValueError("Evaluation requires episodes, families and positive route length")
    env = NS3HandoverEnv(cfg)
    try:
        for k in range(n_eps * len(families)):
            fam = families[k // n_eps]
            obs, info = env.reset(seed=900000 + k % n_eps,
                                  options={"family": fam} if fam else None)
            ep_r = ep_tp = ho = pp = out = steps = 0
            term = trunc = False
            for _ in range(budget_steps):
                with torch.no_grad():
                    x = torch.as_tensor(obs, dtype=torch.float32, device=eval_device).unsqueeze(0)
                    q = net(x)
                    a = select_action(q, cfg, info, actions=agent.legal_actions)
                obs, r, term, trunc, info = env.step(a)
                ep_r += r; ep_tp += info.get("throughput_mbps", 0.0); steps += 1
                if info.get("ho_event"): ho += 1
                if info.get("pingpong"): pp += 1
                if term: out += 1
                if term or trunc: break
            rec["reward"].append(ep_r); rec["throughput"].append(ep_tp / max(steps, 1))
            rec["ho"].append(ho); rec["pingpong"].append(pp); rec["outage"].append(out)
            per_family.setdefault(str(fam), []).append(out)
            full_route = steps == ep_steps
            episodes.append({
                "family": str(fam), "seed": 900000 + k % n_eps,
                "steps": steps, "terminated": bool(term), "truncated": bool(trunc),
                "termination_reason": info.get("termination_reason"),
                "full_route": full_route, "survived": bool(full_route and not term),
            })
    finally:
        env.close()
    result = {k: float(np.mean(v)) for k, v in rec.items()}
    result["worst_family_outage"] = max(float(np.mean(v)) for v in per_family.values())
    result["coverage_complete"] = all(e["full_route"] for e in episodes)
    result["full_survival"] = all(e["survived"] for e in episodes)
    result["episodes"] = episodes
    result["families"] = list(per_family)
    return result


def best_checkpoint_eligible(metrics, cfg):
    """Strict opt-in gate also applies to the first evaluated checkpoint."""
    if not (cfg.get("eval") or {}).get("require_full_survival", False):
        return True
    episodes = metrics.get("episodes", [])
    return bool(episodes and metrics.get("coverage_complete")
                and metrics.get("full_survival")
                and metrics.get("outage", 1) == 0
                and metrics.get("worst_family_outage", 1) == 0
                and all(e["full_route"] and e["survived"]
                        and not e["terminated"] for e in episodes))


def format_eval_metrics(metrics):
    """Keep numerical precision without applying float formatting to metadata."""
    import json
    from numbers import Real
    return "\n".join(
        f"  {key:12s} = " + (f"{value:.4f}" if isinstance(value, Real)
                                 and not isinstance(value, bool)
                                 else json.dumps(value, ensure_ascii=False))
        for key, value in metrics.items())


def evaluate_guarded(agent, cfg, eval_device, n_eps, family=None, timeout_s=0.0):
    """``evaluate`` 的带守护版本：超时则**放弃本次评估**并返回 None。

    背景（2026-09-24 H10 实测）：``evaluate`` 在**主线程串行**跑真实 NS-3，且没有
    任何时间上限。eval_family 误取 congested 后单集 ~22 min、4 集 ~88 min，c50 那次
    评估独占 85 min，训练全程停滞，直到外层 3 h 的 ``timeout`` 才收口 —— 整轮白跑。

    权衡：超时后子线程作为 daemon 继续跑（会残留一个 NS-3 进程），这不如"干净取消"
    理想，但代价远小于把训练主循环无限期堵死；且该路径只在异常慢/挂死时触发。
    调用方必须处理 None（本次跳过评估、不更新 best.pt、不计入 no_improve）。
    """
    if timeout_s <= 0:
        return evaluate(agent, cfg, eval_device, n_eps, family=family)
    box = {}

    def _run():
        try:
            box["m"] = evaluate(agent, cfg, eval_device, n_eps, family=family)
        except Exception as exc:          # noqa: BLE001 - 评估异常不应杀死训练
            box["err"] = exc

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        return None
    if "err" in box:
        raise box["err"]
    return box.get("m")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/ns3.yaml")
    p.add_argument("--n-episodes", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval-device", default=None)
    p.add_argument("--net", choices=list(NETS), default="hybrid")
    p.add_argument("--init", default=None, help="Stage1 bc_init.pt 初始化")
    p.add_argument("--anchor-init", default=None,
                   help="信任域锚点权重（默认与 --init 同）。锚点必须是**在目标分布上"
                        "本身就成立**的策略（如 BC 教师克隆），否则会把策略锁死在一个"
                        "对新分布不适用的点上（F4 教训：锚到上一阶段快照反而全灭）")
    p.add_argument("--resume", default=None)
    p.add_argument("--dcu-sample", type=float, default=30.0,
                   help="DCU 利用率采样间隔 s（0 = 关闭；容器内才有 hy-smi）")
    p.add_argument("--mock", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.mock:
        import sys
        # The subprocess mock speaks the Python seven-action protocol; do not
        # infer NS-3's four-action capability merely because a subprocess
        # configuration is present.
        cfg["mock_backend"] = True
        cfg["ns3"] = {"binary": sys.executable,
                      "args": "-m scripts.ns3_mock --dur=6 --nEnb=8 --spacing=600 --offset=50",
                      "cwd": ".", "timeout_s": 60}
        cfg["env"]["episode_steps"] = 50
        cfg["training"]["n_episodes"] = args.n_episodes or 2
        cfg["eval"]["n_episodes"] = 1
        cfg.setdefault("collector", {})["n_collectors"] = 2
    if args.n_episodes:
        cfg["training"]["n_episodes"] = args.n_episodes

    # Stage2 在线：PER 走普通 obs 重放（并行采集无需 latent 隐状态）
    cfg["agent"]["replay_latent"] = False
    cfg["env"]["legal_actions"] = list(legal_actions(cfg, int(cfg["env"]["n_actions"])))

    device = get_device(args.device)
    set_seed(cfg.get("seed", 42))

    # 评估设备：优先 cuda:1，不可用回退学习设备
    if args.eval_device:
        eval_device = torch.device(args.eval_device) if torch.cuda.is_available() else device
    else:
        ed = cfg["training"].get("eval_device", "cuda:1")
        eval_device = torch.device(ed) if (torch.cuda.is_available()
                                           and torch.cuda.device_count() >= 2
                                           and ed.startswith("cuda:1")) else device
    print(f"learn device={device}  eval device={eval_device}  "
          f"cuda_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    # DCU 利用率采样：目标是让卡"有活干"（实测此前 ≈0%，全在等 NS-3 数据）
    dcu_stop, dcu_samples = start_dcu_sampler(args.dcu_sample)

    agent_cls = AGENTS[str(cfg["agent"].get("algorithm", "dqn")).lower()]
    agent = agent_cls(cfg, device, net_factory=NETS[args.net])
    print(f"algorithm={cfg['agent'].get('algorithm', 'dqn')} -> {agent_cls.__name__}")
    # 锚点先冻结、再载入训练起点：两者可不同（见 --anchor-init 说明）
    if args.anchor_init:
        agent.load(args.anchor_init)
        if agent.set_bc_anchor():
            print(f"BC anchor frozen from {args.anchor_init} (w={agent.bc_anchor_w})")
    if args.init:
        agent.load(args.init)
        agent.target_net.load_state_dict(agent.online_net.state_dict())
        print(f"Initialized from BC checkpoint {args.init}")
    if args.resume:
        agent.load(args.resume)
        print(f"Resumed from {args.resume}")
    # 未显式指定锚点时，退化为本阶段的起点策略（保持旧行为）
    if agent.anchor_net is None and agent.set_bc_anchor():
        print(f"BC anchor enabled (w={agent.bc_anchor_w}) —— Q 被约束在起点策略附近")

    ck_dir = cfg["training"].get("checkpoint_dir", "checkpoints_ns3")
    os.makedirs(ck_dir, exist_ok=True)
    n_eps = int(cfg["training"]["n_episodes"])
    # 预算双口径：cycle 只是节拍器；真正决定训练量的是 env-steps
    # （教训：n_episodes=100 cycles 实际只攒到 11 集经验，严重欠训练）
    budget_steps = int(cfg["training"].get("budget_env_steps", 0))
    min_buf = int(cfg["training"].get("min_buffer_before_learn", 256))
    learn_k = int(cfg["training"].get("learn_steps_per_cycle", 16))
    sync_every = int(cfg["training"].get("target_sync_every", 100))
    eval_every = int(cfg["training"].get("eval_every_cycles", 25))
    eval_n = int(cfg["eval"]["n_episodes"])
    # 内联评估固定在「代表族」上（不再与训练随机化纠缠；正式对比走 eval_ns3）
    eval_fam = (cfg.get("curriculum") or {}).get("eval_family")
    # 单次内联评估的硬上限（秒，0 = 不限制）。存在的理由：2026-09-24 H10 实测
    # evaluate() 串行跑在主线程且**无任何守护** —— eval_family 误取 congested 后
    # 单集 ~22 min、4 集 ~88 min，c50 那次评估独占 85 min 训练全程停滞，直到
    # 3 h 的 timeout 才收口，整轮白跑。评估是监控手段，不能反过来把训练卡死。
    eval_timeout_s = float(cfg["training"].get("eval_timeout_s", 0) or 0)

    # ---- 并行采集 ----
    col_cfg = cfg.get("collector", {})
    n_col = int(col_cfg.get("n_collectors", 6))
    # mixed 策略需要 CPU 网络快照：用与 agent 同类同参构造（统一签名 **net_kwargs）
    snapshot_net = None
    if col_cfg.get("collect_policy", "a3") == "mixed":
        snapshot_net = type(agent.online_net)(
            int(cfg["env"]["obs_dim"]), int(cfg["env"]["n_actions"]),
            agent.hidden, **agent.net_kwargs).to("cpu")
        # 关键：快照必须从 BC 初始化权重起步。否则训练前期 mixed 分支用的是
        # 随机网（甚至 ε-随机），buffer 里灌入大量垃圾动作轨迹，Q 学习被污染。
        snapshot_net.load_state_dict(agent.online_net.state_dict())
        snapshot_net.eval()
    # stop_event 既供 collector 内部连续失败置位，也用于 Ctrl-C 优雅停止。
    stop = threading.Event()
    collector = ParallelCollector(
        cfg, n_envs=n_col, policy=col_cfg.get("collect_policy", "a3"),
        stop_event=stop,
        mixed_frac=float(col_cfg.get("mixed_bootstrap_frac", 0.3)),
        mixed_eps=float(col_cfg.get("mixed_eps", 0.2)),
        snapshot_net=snapshot_net)
    replay_writer = CollectorReplayWriter(agent)
    collector.start()
    snap_every = int(col_cfg.get("snapshot_every_steps", 200))

    print(f"Stage2 online finetune: target cycles≈{n_eps}"
          f"{f', budget={budget_steps} env-steps' if budget_steps else ''}, "
          f"net={args.net}, collectors={len(collector.workers)}")
    t0 = time.time()
    learn_steps = 0
    cycles = 0
    total_trans = 0
    loss_ema = None
    td_ema = None            # learn() 返回的**总和**里的 TD 分量（含 PER 权重）
    an_ema = None            # 锚定分量 ‖Q−Q_ref‖²（未乘 bc_anchor_w）
    best_key = None          # 内联评估择优：(outage, -reward) 字典序最小者
    no_improve = 0
    patience = int(cfg["training"].get("early_stop_patience", 0))
    # 周期落盘间隔（cycle）。此前该键是死键（见下方 cycles 打印块里的说明）。
    ck_freq = int(cfg["training"].get("checkpoint_freq", 0) or 0)
    # ---- 数据耦合的 learn 门禁（治「池子冻结还在学」）----
    # 2026-09-24 H9 实测：buffer 在 c80 就撞满 buffer_size=20000，此后 20 路采集器
    # 因重族（congested 48-96 UE）长 episode 而**脉冲式**到货 —— c165→c200 只新增
    # 302 条，但主循环仍按 learn_steps_per_cycle=8 每 cycle 更新 ⇒ 对同一批 20000
    # 样本反复迭代，loss_ema 从 0.38 涨到 9.3（价值发散），DCU 利用率 0.2%。
    # 修法：learn 必须由**新增经验**驱动，低谷期不做无用更新、把 CPU 让给采集
    # （数据供给才是唯一节流阀，见 docs P7）。0 = 关闭该门禁（旧行为）。
    min_new = int(cfg["training"].get("learn_min_new_transitions", 0))
    new_since_learn = 0      # 距上次 learn 累计的新增 transition 数（跨 cycle 累加）
    try:
        # 以「学习周期」为进度单位；每个周期 drain outbox → learn K 次
        # 结束条件：学到 n_eps 个周期 / 攒够 budget_env_steps 条经验，
        # 或 collector 已置位 stop（连续失败），或外部 Ctrl-C。
        while cycles < n_eps and not stop.is_set():
            if budget_steps > 0 and total_trans >= budget_steps:
                print(f"[budget] {total_trans} env-steps ≥ {budget_steps}，训练收口")
                break
            # drain outbox → 主线程串行 store
            drained = 0
            while True:
                try:
                    item = collector.outbox.get_nowait()
                except queue.Empty:
                    break
                drained += int(replay_writer.consume(item))
            total_trans += drained
            new_since_learn += drained

            if len(agent.buffer) < min_buf:
                time.sleep(0.5)
                continue

            # 新增经验不足（采集脉冲的低谷期）⇒ 跳过本 cycle 的 learn。
            # new_since_learn 跨 cycle 累加，故不会因单个 cycle 到货少而饿死学习；
            # cycles 也**不推进**（与上面 min_buf 分支同语义：这是「等数据」而非
            # 「学了一轮」），否则 pulse 低谷期会用假进度把 n_episodes 预算烧完。
            if min_new > 0 and new_since_learn < min_new:
                time.sleep(0.2)
                continue

            for _ in range(learn_k):
                l = agent.learn()
                if l:
                    loss_ema = l if loss_ema is None else 0.98 * loss_ema + 0.02 * l
                    # loss 分量（见 agent.py 的说明）：总和上涨时，只有拆开才能区分
                    # 「价值发散（TD 项涨）」与「锚定惩罚随 Q 漂移增长（锚定项涨）」。
                    _td = getattr(agent, "_last_td", None)
                    _an = getattr(agent, "_last_anchor", None)
                    if _td is not None:
                        td_ema = _td if td_ema is None else 0.98 * td_ema + 0.02 * _td
                    if _an is not None:
                        an_ema = _an if an_ema is None else 0.98 * an_ema + 0.02 * _an
                learn_steps += 1
                if learn_steps % sync_every == 0:
                    agent.sync_target()
                # mixed 快照按 **learn step** 节奏更新（cycle 数远小于
                # snapshot_every_steps，按 cycle 触发会永不更新）。
                if snapshot_net is not None and learn_steps % snap_every == 0:
                    collector.update_snapshot(
                        {k: v.detach().to("cpu").clone() for k, v in
                         agent.online_net.state_dict().items()})
            new_since_learn = 0      # 本 cycle 已消费掉这些新经验
            agent.anneal_beta(min(1.0, cycles / max(n_eps, 1)))
            agent.eps = max(0.05, agent.eps * 0.995)
            cycles += 1

            if cycles % 5 == 0 or cycles == 1:
                el = time.time() - t0
                print(f"  cycle {cycles:4d}/{n_eps}  trans={total_trans:7d} "
                      f"buf={len(agent.buffer):6d}  eps={agent.eps:.3f}  "
                      f"eps_collected={collector.episodes}  fails={collector.failures}  "
                      f"{el:.0f}s", flush=True)
                # 周期性落盘（2026-09-24 补）：`checkpoint_freq` 此前是**死配置键**
                # —— 代码只读 checkpoint_dir，latest.pt 仅在「评估时」和「训练结束」
                # 两处保存。于是 eval_every_cycles=200 时，两次评估之间崩溃（例如
                # cmd_0173 那次 UnboundLocalError）会**丢失整轮全部产物**（checkpoints
                # 目录为空）。现在按 checkpoint_freq 周期存盘，崩了也能拿到最近快照。
                if ck_freq > 0 and cycles % ck_freq == 0:
                    agent.save(os.path.join(ck_dir, "latest.pt"))
                # 收敛监控（改革前只能靠 eval 猜）：loss 曲线 + 规则门禁状态
                if cycles % 25 == 0:
                    dcu = (f"  dcu%={sum(dcu_samples[-20:]) / max(len(dcu_samples[-20:]), 1):.1f}"
                           if dcu_samples else "")
                    # loss 分量：总和上涨时靠它区分「价值发散」与「锚定惩罚增长」。
                    # 判读：td 涨 ⇒ Q 值在飘/发散（降 lr 或降 learn 步数）；
                    #       td 稳而 anchor 涨 ⇒ 锚定正在正常工作（策略在动但被约束住）；
                    #       两者都涨 ⇒ 更新量远超数据量，先降重放比。
                    comp = ""
                    if td_ema is not None:
                        comp += f"  td={td_ema:.4f}"
                    if an_ema is not None:
                        comp += f"  anchor={an_ema:.4f}"
                    print(f"  [monitor c{cycles}] loss_ema="
                          f"{loss_ema if loss_ema is None else round(float(loss_ema), 4)}"
                          f"{comp}  guard={collector.guard.status()}{dcu}", flush=True)

            if eval_every > 0 and cycles % eval_every == 0:
                m = evaluate_guarded(agent, cfg, eval_device, eval_n,
                                     family=eval_fam, timeout_s=eval_timeout_s)
                if m is None:
                    # 超时放弃本次评估：不更新 best.pt、不计 no_improve（否则会把
                    # "评估跑不完"误判成"策略没进步"而提前早停），训练继续。
                    print(f"  [eval c{cycles}] 超时（>{eval_timeout_s:.0f}s）已放弃本次评估，"
                          f"训练继续（不更新 best.pt）", flush=True)
                    continue
                print(f"  [eval c{cycles}] tp={m['throughput']:.2f}Mbps ho={m['ho']} "
                      f"pp={m['pingpong']} out={m['outage']} R={m['reward']:.1f}", flush=True)
                agent.save(os.path.join(ck_dir, "latest.pt"))
                # ---- 内联评估择优保存 best.pt ----
                # 教训（Stage B 实测）：在线微调**并不单调变好**。c350 时评估是
                # ho=4.0/out=0.0，跑到 c600 反而退化成「几乎不切」（诊断 6/6 为
                # no_ho_while_a3_would，x≈700m 就死）。此后所有评估/诊断若取
                # latest（末快照），等于用一个退化后的权重下结论。故按
                # (outage 低, reward 高) 的字典序保留最佳快照。
                if not best_checkpoint_eligible(m, cfg):
                    print("  [eval] death or incomplete route: best.pt selection skipped", flush=True)
                    continue
                key = (float(m["worst_family_outage"]), -float(m["reward"]))
                if best_key is None or key < best_key:
                    best_key = key
                    agent.save(os.path.join(ck_dir, "best.pt"))
                    no_improve = 0
                    print(f"  [best] 更新 best.pt: out={m['outage']:.1f} "
                          f"R={m['reward']:.1f} (c{cycles})", flush=True)
                else:
                    no_improve += 1
                    # 早停：在线微调**不单调变好**（C4 实测：c50~c200 全 out=0.0，
                    # c250 起退化成 ho=0）。跑过峰值只会浪费墙钟并让最终产物更差，
                    # 故连续 patience 次评估无改进即收口（best.pt 已留住峰值）。
                    if patience > 0 and no_improve >= patience:
                        print(f"  [early-stop] 连续 {no_improve} 次评估未改进 "
                              f"（best: out={best_key[0]:.1f}, R={-best_key[1]:.1f}），收口",
                              flush=True)
                        break
    except KeyboardInterrupt:
        # Ctrl-C：置位 stop 让所有 collector 线程优雅退出，finally 里 join。
        stop.set()
        print("\n[interrupted] stopping collectors ...")
    finally:
        collector.stop()
        replay_writer.discard()
        dcu_stop.set()
        if dcu_samples:
            print(f"\n[DCU] 利用率采样 n={len(dcu_samples)} "
                  f"mean={sum(dcu_samples) / len(dcu_samples):.1f}% "
                  f"max={max(dcu_samples):.0f}%  "
                  f"（若长期 <5%，说明卡在等数据：应提高采集并行度或 batch/"
                  f"learn_steps_per_cycle，而不是加大网络）")

    agent.save(os.path.join(ck_dir, "latest.pt"))
    print(f"\nStage2 done in {time.time()-t0:.0f}s, trans={total_trans}, "
          f"checkpoints in {ck_dir}/")
    if best_key is not None:
        print(f"  best.pt 来自内联评估最优快照：outage={best_key[0]:.1f}, "
              f"R={-best_key[1]:.1f}（后续评估/诊断应优先用 best.pt）")

    m = evaluate(agent, cfg, eval_device, eval_n)
    print("=== Final eval (eps=0) ===")
    print(format_eval_metrics(m))


if __name__ == "__main__":
    main()
