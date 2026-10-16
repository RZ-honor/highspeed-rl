"""并行经验收集器（解耦 NS-3 仿真与 GPU 训练）。

N 个 NS3HandoverEnv 子进程各跑在独立线程，使用**纯 CPU** 策略（A3 示范，或
A3 + 网络的 CPU 快照混合），把带 worker 标识的原始转移推入线程安全 outbox。
GPU 学习主线程按 worker 独立累积 n 步回报，再串行灌入 agent.PER，
避免跨 worker/episode 混合及 GPU 并发竞争。
"""

import queue
import threading
import time

import numpy as np
import torch

from src.envs.ns3_env import NS3HandoverEnv, family_pool
from src.agents.baselines import A3OperatorBaseline
from src.agents.replay import NStepAccumulator
from src.agents.safety import RuleGuard, legal_actions, mask_actions, select_action

# 单个 collector 连续失败超过该阈值 → 判定 NS-3 二进制/环境不可用，停止整个采集
_MAX_CONSECUTIVE_FAILS = 10

# Each outbox message carries its worker identity, including boundaries.
EPISODE_END = object()
EPISODE_ABORT = object()


class CollectorReplayWriter:
    """Main-thread replay writer with one n-step window per worker.

    Parallel training uses observation replay, not shared recurrent state.
    Completed windows remain valid after failures; pending windows are discarded.
    """

    def __init__(self, agent):
        if agent.latent:
            raise ValueError("Parallel collection requires agent.replay_latent=False")
        self.agent = agent
        self.accumulators = {}

    def consume(self, message):
        wid, item = message
        if wid not in self.accumulators:
            self.accumulators[wid] = NStepAccumulator(
                self.agent.n_step, self.agent.gamma, self.agent.buffer)
        acc = self.accumulators[wid]
        if item is EPISODE_END or item is EPISODE_ABORT:
            if item is EPISODE_END:
                acc.flush()
            else:
                acc.discard()
            del self.accumulators[wid]
            return False
        acc.push(*item)
        return True

    def discard(self):
        """Discard unfinished windows at shutdown without draining stale data."""
        for acc in self.accumulators.values():
            acc.discard()
        self.accumulators.clear()



class ParallelCollector:
    def __init__(self, cfg, n_envs=6, policy="a3", stop_event=None,
                 mixed_frac=0.3, snapshot_net=None, mixed_eps=0.2):
        self.cfg = cfg
        self.n_envs = int(n_envs)
        self.policy = policy
        # 规则门禁（共享）：N 路 worker 的回合战绩都汇总到这里做 RL vs 规则对比。
        # guard 模式放行探索，只有 RL 退化到比规则差时才重新启用规则掩码。
        self.guard = RuleGuard(cfg.get("guard", {}))
        # 课程训练池：只从 Stage0 标定达标的（enabled）场景族里均匀采样。
        # 族只改难度维度，拓扑/射频不变；无 families 配置时回退单 scenario。
        self.family_pool = family_pool(cfg)
        # 课程采样权重：族可带 weight（默认 1.0）。
        # 用途：更难/更特殊的族（如 hotcold）应以**低权重**放在最后一级，而不是
        # 与常规族等权混合——实测 hotcold 与 congested 等权混合会把 congested
        # 从存活 1.00 拖到 0.17（更难族的 RLF 主导样本把策略带偏）。
        fams_cfg = (cfg.get("ns3") or {}).get("scenario_families") or {}
        w = np.array([float((fams_cfg.get(f) or {}).get("weight", 1.0))
                      for f in self.family_pool], dtype=np.float64)
        self.family_weights = (w / w.sum()) if w.sum() > 0 else None
        self.stop_event = stop_event or threading.Event()
        self.mixed_frac = float(mixed_frac)
        # mixed 模式下快照动作的 ε-随机探索率：纯贪心快照只会复刻当前策略，
        # buffer 里没有教师(A3)之外的新动作 → RL 无法发现优于教师的决策。
        # 探索动作落到 3/4/6(双连) 会被 env mask 成 stay，无害。
        self.legal_actions = legal_actions(cfg, int(cfg["env"]["n_actions"]))
        self.mixed_eps = float(mixed_eps)
        self.outbox = queue.Queue()          # 线程安全；主线程 drain
        # env 与其 policy 成对构造、成对存放（存 (env, policy) tuple），
        # 避免任一构造失败导致 start() 里 policies[wid] 越界。
        self.workers = []                    # list[(env, policy)]
        self.threads = []
        self.collected = 0
        self.episodes = 0
        self.failures = 0
        self._stat_lock = threading.Lock()   # 保护计数器自增
        # CPU 网络快照（mixed 策略用）；由 train_ns3 用与 agent 相同的类/参数构造后传入，
        # 主线程通过 update_snapshot 更新权重。网络统一签名 (obs,act,hidden,**net_kwargs)。
        self._snapshot = snapshot_net
        self._snap_lock = threading.Lock()
        if self._snapshot is not None:
            self._snapshot.eval()

        a3_cfg = cfg.get("a3_operator", {})
        a3_cfg.setdefault("fc_hz", float(cfg["channel"]["fc_hz"]))
        dt_s = float(cfg["mobility"]["dt_s"])
        for _ in range(self.n_envs):
            env = None
            try:
                # env 与 policy 在同一 try 内成对构造；任一失败则二者都不入队。
                env = NS3HandoverEnv(cfg, guard=self.guard)
                pol = A3OperatorBaseline(cfg["safety"], a3_cfg, dt_s)
                self.workers.append((env, pol))
            except Exception as e:
                print(f"[collector] worker init failed: {e}")
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass

    def update_snapshot(self, state_dict):
        if self._snapshot is not None:
            with self._snap_lock:
                self._snapshot.load_state_dict(state_dict)

    def _act(self, pol, obs, info, rng, use_rl: bool):
        """返回 (动作, 来源)；来源 ∈ {"rl", "a3"}，供规则门禁做成绩对比。

        注意：**按 episode 抽样** 行为策略（_run_one 里每集抽一次 use_rl），
        而不是按步抽样。逐步混合时 mixed_frac=0.3 只会让单集里约三成步来自
        快照，回合既不是纯 RL 也不是纯规则，无法用于「RL 是否退化到比规则
        差」的对比 —— 门禁会永远凑不齐 RL 侧样本而从不生效。
        """
        if use_rl:
            # RL 自探索分支：快照贪心 + ε 随机。这是 buffer 中唯一来自
            # agent 自身策略的动作来源（A3 分支只产生教师动作）。
            with torch.no_grad(), self._snap_lock:
                x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                q = self._snapshot(x)
            return select_action(q, self.cfg, info, self.mixed_eps, rng,
                                 self.legal_actions, self.guard), "rl"
        return int(pol.act(obs, info)), "a3"   # A3（env.step 内还有安全兜底）

    def _run_one(self, env, pol, wid):
        seed_base = int(self.cfg.get("seed", 42)) + 100000 + wid * 10000
        rng = np.random.default_rng(seed_base)
        ep = 0
        consecutive_fails = 0
        while not self.stop_event.is_set():
            ep += 1
            try:
                if hasattr(pol, "reset"):
                    pol.reset()
                # 每集抽一次行为策略标签：整集要么走 RL 快照、要么走 A3 教师。
                use_rl = (self.policy == "mixed" and self._snapshot is not None
                          and rng.random() < self.mixed_frac)
                # 课程：本集场景族按权重从训练池抽（教师可存活的族）
                fam = (rng.choice(self.family_pool, p=self.family_weights)
                       if self.family_pool else None)
                opts = {"family": fam} if fam else None
                obs, info = env.reset(seed=seed_base + ep, options=opts)
                done = trunc = False
                ep_ret = 0.0
                while not (done or trunc) and not self.stop_event.is_set():
                    a, src = self._act(pol, obs, info, rng, use_rl)
                    nobs, r, done, trunc, ninfo = env.step(a)
                    ep_ret += float(r)
                    # 入 buffer 的动作必须是**实际执行**的动作：env 会把
                    # dual(3/4/6) mask 成 stay、弱覆盖时 A3 兜底替换动作。
                    # 按策略输出动作入 buffer 会让 Q 目标与真实后果错位。
                    a_stored = int(ninfo.get("action_executed", a))
                    # Preserve terminated separately from the worker boundary.
                    self.outbox.put((wid, (np.asarray(obs, dtype=np.float32).copy(),
                                          a_stored, float(r),
                                          np.asarray(nobs, dtype=np.float32).copy(),
                                          bool(done))))
                    with self._stat_lock:
                        self.collected += 1
                    obs, info = nobs, ninfo
                with self._stat_lock:
                    self.episodes += 1
                # 规则门禁：本集成绩按本集的行为策略标签归类（episode 级抽样，
                # 同分布可比）。只有 RL 整做成绩低于规则时才拉起规则接管。
                self.guard.observe("rl" if use_rl else "a3", ep_ret)
                # 通知主线程：本 episode 结束 → 刷新 n 步回报窗口。
                self.outbox.put((wid, EPISODE_END if done or trunc else EPISODE_ABORT))
                # 成功跑完一个 episode：重置连续失败计数。
                consecutive_fails = 0
            except Exception as e:
                # FIFO per worker: discard its pending window before any reset.
                self.outbox.put((wid, EPISODE_ABORT))
                with self._stat_lock:
                    self.failures += 1
                consecutive_fails += 1
                print(f"[collector {wid}] ep {ep} error "
                      f"({consecutive_fails} consecutive): {e}")
                if self.stop_event.is_set():
                    break
                # 连续失败指数退避：sleep 2^n（封顶 30s），避免 NS-3 起不来时热空转。
                time.sleep(min(2 ** consecutive_fails, 30))
                try:
                    env.close()
                except Exception:
                    pass
                if consecutive_fails >= _MAX_CONSECUTIVE_FAILS:
                    print(f"[collector {wid}] 连续 {consecutive_fails} 次失败，"
                          f"判定 NS-3 二进制/握手不可用，停止全部采集。"
                          f"请检查 cfg['ns3']['binary'] 路径与场景参数。")
                    self.stop_event.set()
                    return

    def start(self):
        if len(self.workers) == 0:
            raise RuntimeError(
                "所有 NS-3 env/policy 初始化失败（0 个可用 worker）。"
                "通常是 NS-3 二进制路径错误或握手失败：检查 cfg['ns3']['binary']、"
                "cwd 与 ns3_mock/ns3 场景参数，确认二进制可执行。")
        for wid, (env, pol) in enumerate(self.workers):
            t = threading.Thread(target=self._run_one,
                                 args=(env, pol, wid), daemon=True)
            t.start()
            self.threads.append(t)
        print(f"[collector] started {len(self.workers)} parallel ns-3 envs "
              f"(policy={self.policy})")
        if self.family_pool:
            print(f"[collector] 课程训练池: {self.family_pool}"
                  + (f" 权重={[round(float(x), 3) for x in self.family_weights]}"
                     if self.family_weights is not None else ""))

    def stop(self):
        self.stop_event.set()
        for t in self.threads:
            t.join(timeout=10)
        for env, _ in self.workers:
            try:
                env.close()
            except Exception:
                pass
