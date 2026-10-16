"""NS-3 后端交互环境（Stage B）。

通过子进程 stdin/stdout 与 ns-3 的 hsr-rl 二进制锁步交互：
  - NS-3 启动时写 4 字节 magic ``HSR4``
  - 每 dt(100ms)：NS-3 写 HsrState(99B) → 阻塞读 HsrAction(8B) → 应用
  - HSR1/HSR2/HSR3 为旧版协议；HSR4 增加切换请求结果、进行中/失败及
    区间级无包遥测，同时保留旧字段兼容读取

观测/奖励/info 契约与 :class:`src.envs.handover_env.HandoverEnv` 对齐：
14 维 obs、7 动作、相同奖励公式与安全兜底，使 DuelingDDQNAgent /
SafetyLayer / 评估脚本无需改动即可切换后端。

与 Python 环境的已知偏离（详见 docs/superpowers/specs/2026-08-16-ns3-env-design.md §8）：
  1. 双连通在 NS-3 后端被掩码：LENA 无 EN-DC，动作 3/4/6 在物理层是空操作。
     step() 中已强制将 3/4/6 回退为 stay(0)，不进入策略也不计入评估（active_links
     不再含虚拟邻区，吞吐/稳定性奖励不再包含欺骗性双连增益）。真实双连需迁移 5G-LENA(nr)。
  2. SINR 为估计值：由上报 RSRP（服务+top-3 邻区）按与 Python 信道相同的
     公式计算（SINR = P_rx / (noise + Σ interference)），非物理层实测。
  3. RSRP 趋势为后向差分（最近 10 步），Python env 为前向预测——语义近似。
  4. ho_interrupt_ms 是 RRC Start→EndOk 时长，非业务包间隙；
     no_rx_* 为 FlowMonitor 零收包区间统计，不提供包级 p95。
"""

import subprocess
import threading
import time
from collections import deque

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.agents.safety import HO_TARGET_IDX, hard_block_reason, hard_shield_params

# HSR4：协议 84B → 99B，增加切换请求/进行中/失败及区间级无包遥测；保留 HSR3 读取兼容。
MAGIC = b"HSR4"

# 与 ns3/hsr-common.h 的 HsrState 逐字节对齐（#pragma pack(1), 84 字节）
HSR_STATE_DTYPE = np.dtype([
    ("step", "<u4"),
    ("time_ms", "<u4"),
    ("serving_cell", "<u2"),
    ("serving_rsrp_dbm", "<i2"),
    ("nbr_cell", "<i2", (3,)),
    ("nbr_rsrp_dbm", "<i2", (3,)),
    ("n_nbr", "<u2"),
    ("speed_mps", "<f4"),
    ("heading_cos", "<f4"),
    ("dist_to_serving_m", "<f4"),
    ("ue_x", "<f4"),
    ("throughput_mbps", "<f4"),
    ("ho_interrupt_ms", "<u2"),
    ("ho_event", "u1"),
    ("rlf", "u1"),
    ("done", "u1"),
    # 负载/隧道可观测（占用原 reserved[5]，协议尺寸仍为 56B）。
    # load_* = 该小区附着的背景 UE 数（负载代理）；in_tunnel = UE 是否在隧道段。
    ("load_serving", "u1"),
    ("load_nbr", "u1", (3,)),
    ("in_tunnel", "u1"),
    # 容量/业务量指标（100~1000 用户规模验证；HSR3 = 84B）
    #   口径要点：tp_agg_mbps 是**全网**聚合，tp_agg_cell_mbps 才是**服务小区**
    #   聚合 —— 「人均速率」必须用后者除 n_ue_cell，用前者会高估约 n_cells 倍。
    #   tp_ue_p05/p50 是**截至当前**的 per-UE 平均吞吐分位数（累计字节/时长），
    #   不是逐间隔快照；jain 为公平性指数。
    ("n_ue_cell", "<u2"),
    ("n_bg_total", "<u2"),
    ("tp_agg_mbps", "<f4"),
    ("ctrl_delay_ms", "<f4"),
    ("tp_agg_cell_mbps", "<f4"),
    ("tp_ue_p05_mbps", "<f4"),
    ("tp_ue_p50_mbps", "<f4"),
    ("jain", "<f4"),
    ("ho_request_result", "u1"),
    ("ho_in_progress", "u1"),
    ("ho_failed", "u1"),
    ("interval_ms", "<f4"),
    ("no_rx_interval_ms", "<f4"),
    ("no_rx_run_ms", "<f4"),
])

# 与 HsrAction 对齐（8 字节）
HSR_ACTION_DTYPE = np.dtype([("action", "u1"), ("reserved", "u1", (7,))])

assert HSR_STATE_DTYPE.itemsize == 99, "HsrState 布局与 hsr-common.h 不一致"
assert HSR_ACTION_DTYPE.itemsize == 8, "HsrAction 布局与 hsr-common.h 不一致"


def _norm(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def load_gap_penalty(load_serving, loads, valid, scale_ue=4.0, deadband_ue=1.0):
    """待在比候选邻区更拥挤的小区上的惩罚量（0~1，纯函数便于单测）。

    动机（机制审计的关键发现）：负载可观测化之后，RL 的吞吐增益随负载严重度
    递增（+5.4%→+11.8%），但审计显示**时间均值负载几乎不变**（congested 上
    1.478 → 1.473）——增益其实来自切换时机/目标 SINR，**不是负载分流**。
    根因：奖励里没有显式负载项，吞吐只在很间接的层面奖励低负载。

    本项直接给梯度：服务小区比「测量有效的最轻候选邻区」多 deadband_ue 个
    背景 UE 之后，每多 scale_ue 个计满 1.0。只比较**测量有效**的邻区，
    避免为了躲负载而指向远/弱的空闲小区（那切过去就是 RLF）。

    负载是慢变量（背景 UE 逐秒重附着），与路径 A「奖励主项必须用可预测量」
    的原则一致：它比吞吐可预测得多。
    """
    cand = [float(l) for l, v in zip(loads, valid) if v]
    if not cand:
        return 0.0
    gap = float(load_serving) - min(cand)
    if gap <= float(deadband_ue):
        return 0.0
    return float(min((gap - float(deadband_ue)) / max(float(scale_ue), 1e-6), 1.0))


def validate_tunnels(tunnels):
    """校验隧道区间：x2>x1 且互不重叠。"""
    segs = sorted((float(a), float(b)) for a, b in tunnels)
    for a, b in segs:
        if b <= a:
            raise ValueError(f"非法隧道区间 [{a},{b}]：x2 必须大于 x1")
    for (a1, b1), (a2, b2) in zip(segs, segs[1:]):
        if a2 < b1:
            raise ValueError(f"隧道区间重叠：[{a1},{b1}] 与 [{a2},{b2}]")
    return [(a, b) for a, b in segs]


def measurement_staleness(step, pending, until, grace, serving_cell,
                          ho_failed, ho_in_progress, meas_valid,
                          run=0, run_cap=None):
    """HO 后「测量仍属源小区」的判定（修复3，纯函数便于单测）。

    native 只在 RecvMeasurementReport 刷新 serving 缓存（hsr-rl.cc:350-378），
    HoEndOk 不刷新（420-462）。故派发 HO 后、目标小区首个上报到达前，状态携带
    的是**源小区**测量——用它累计连续低 SINR 会把「刚切向强目标」误判成死亡
    （H15 两条 ho_to_weaker 死亡的实测形态）。

    返回 (stale, pending, until, run)：
      * grace<=0 或无 pending → 不冻结（旧行为，生产配置默认如此）；
      * serving_cell 已上报为 pending 目标 → 新测量到达，解除；
      * ho_failed → 切换失败，目标永不上报，解除（此时源小区就是真实服务，
        其低 SINR 必须照常计数——不豁免终止标准）；
      * HO 进行中 → 续窗（进行中上报必然仍是源小区），但连续冻结总长封顶
        `run_cap`（默认 2×grace）：防止「反复派发 HO」把终止无限推迟——
        封顶后低 SINR 照常计数，真实劣化不会被豁免；
      * 步数超过 until 仍未见目标上报 → 解除，恢复正常计数。
    """
    if run_cap is None:
        run_cap = 2 * grace
    if grace <= 0 or pending is None:
        return False, pending, until, 0
    if meas_valid and serving_cell == pending:
        return False, None, until, 0
    if ho_failed:
        return False, None, until, 0
    if ho_in_progress:
        until = max(until, step + grace)
    if step < until and run < run_cap:
        return True, pending, until, run + 1
    return False, None, until, 0


def _sample(val, rng):
    """区间 [lo,hi] → 均匀采样；标量 → 原样返回。"""
    if isinstance(val, (list, tuple)) and len(val) == 2:
        lo, hi = float(val[0]), float(val[1])
        return float(rng.uniform(lo, hi))
    return val


def family_pool(cfg: dict) -> list:
    """从 ns3.scenario_families 里取**启用**的族名列表（课程训练池）。

    `enabled: false` 的族不入池（Stage0 标定不达标的族，例如本轮实测
    congested 族 A3 教师 4/4 全灭 → 存活率 0）。无 families 配置时返回
    空列表 → 行为与旧版一致（单 scenario 模式）。
    """
    fams = (cfg.get("ns3") or {}).get("scenario_families") or {}
    return [name for name, f in fams.items()
            if isinstance(f, dict) and f.get("enabled", True)]


def build_scenario_args(cfg, rng_run, rng, family=None):
    """把 ns3.scenario 配置 + rng_run 拼成 hsr-rl 命令行参数列表。

    randomize=True 时区间参数按 episode 采样（domain randomization）；
    False 时用固定标量。隧道列表原样（随机化时在候选段中随机保留若干段）。

    ``family``（场景族）：在基础 scenario 之上做键级覆盖 —— 族只改
    难度维度（背景 UE 数 / 聚集比例 / 隧道有无），拓扑（站距/微站部署）
    与射频不变。这是课程化训练的基础：教师可存活的族先入池。
    """
    ns3 = cfg.get("ns3", {})
    sc = dict(ns3.get("scenario") or {})
    if family is not None:
        fams = ns3.get("scenario_families") or {}
        if family not in fams:
            raise ValueError(
                f"未知场景族 {family!r}，可选: {sorted(fams)}（或检查 enabled）")
        sc.update(fams[family])
    args = [f"--rngRun={int(rng_run)}"]
    if not sc:
        return args

    randomize = bool(sc.get("randomize", False))
    if randomize:
        n_bg = int(_sample(sc.get("n_bg_ue", [0, 0]), rng))
        vmin, vmax = sc.get("bg_speed_mps", [83, 138])
        voip = _sample(sc.get("voip_frac", 0.2), rng)
        video = _sample(sc.get("video_frac", 0.4), rng)
        cluster = _sample(sc.get("bg_cluster_frac", 0.0), rng)
        tloss = _sample(sc.get("tunnel_loss_db", 25.0), rng)
        cand = sc.get("tunnels", [])
        # 每 episode 随机包含 0~全部候选隧道段
        tunnels = [t for t in cand if rng.random() < 0.6] if cand else []
        spawn = float(sc.get("bg_spawn_s", 6000.0))
    else:
        # 固定模式：标量原样；若误配成区间（randomize 关闭但值仍是 [lo,hi]），
        # 确定性取区间下限，避免 int(list) 崩溃。
        def _fixed(val, default):
            v = sc.get(val, default)
            return v[0] if isinstance(v, (list, tuple)) else v
        n_bg = int(_fixed("n_bg_ue", 0))
        vmin, vmax = sc.get("bg_speed_mps", [83, 138])
        voip = float(_fixed("voip_frac", 0.2))
        video = float(_fixed("video_frac", 0.4))
        cluster = float(_fixed("bg_cluster_frac", 0.0))
        tloss = float(_fixed("tunnel_loss_db", 25.0))
        tunnels = sc.get("tunnels", [])
        spawn = float(sc.get("bg_spawn_s", 6000.0))

    if tunnels:
        validate_tunnels(tunnels)
    args += [
        f"--nBgUe={n_bg}",
        f"--bgSpeedMin={float(vmin)}",
        f"--bgSpeedMax={float(vmax)}",
        f"--bgSpawnS={spawn}",
        f"--voipFrac={float(voip):.3f}",
        f"--videoFrac={float(video):.3f}",
        f"--bgClusterFrac={float(cluster):.3f}",
        f"--tunnelLossDb={float(tloss):.1f}",
    ]
    # S2 冷热对拓扑：给冷（奇号）宏小区加 Tx 功率，使「切向空闲邻区」同时满足
    # RSRP 门限。1000m 站距下空闲邻区天然更弱，不提升功率则负载分流必然以掉话
    # 为代价（实测 congested 存活 1.00 → 0.33）。
    boost = float(sc.get("cold_neighbor_boost_db", 0.0))
    if boost > 0.0:
        args.append(f"--coldNeighborBoostDb={boost:.1f}")
    # 高速快衰落（Jakes + 多普勒）：设计文档要求「每链路独立快衰落」是基线可信
    # 的物理前提，而 NS-3/LENA 默认不带快衰落。开启后 RSRP 会随多普勒抖动，
    # 「最强邻区频繁翻转」才会在观测里出现（默认关闭以保持与既有标定可比）。
    if bool(sc.get("fast_fading", False)):
        args.append("--fastFading=1")
        if sc.get("fc_hz"):
            args.append(f"--fcHz={float(sc['fc_hz']):.3e}".replace("e+0", "e"))
        if sc.get("jakes_n"):
            args.append(f"--jakesN={int(sc['jakes_n'])}")
    # Always emit an explicit tunnel argument, including the empty set.  This
    # prevents a base command's stale deployment from leaking into a clear
    # scenario and lets the NS-3 side distinguish "configured empty" from an
    # omitted legacy option.
    seg_str = ",".join(f"{float(a):.0f}:{float(b):.0f}" for a, b in tunnels)
    args.append(f"--tunnels={seg_str}")
    if tunnels:
        # v2 洞内微站（漏缆近似）：spacing=0 时 C++ 退回 v1 全衰减行为。
        # 部署参数不参与 domain randomization（随机化的是负载/损耗，不是拓扑）。
        args.append(
            f"--tunnelEnbSpacing={float(sc.get('tunnel_enb_spacing_m', 250.0)):.0f}")
        args.append(
            f"--tunnelEnbTxDbm={float(sc.get('tunnel_enb_tx_dbm', 30.0)):.1f}")
        args.append(
            f"--tunnelMicroLossDb={float(sc.get('tunnel_micro_loss_db', 3.0)):.1f}")
    return args


class NS3HandoverEnv(gym.Env):
    """NS-3 LENA 后端的宏切换环境（契约对齐 HandoverEnv）。"""

    metadata = {"render_modes": []}

    def __init__(self, cfg: dict, guard=None):
        self.cfg = cfg
        self.env_cfg = cfg["env"]
        ns3cfg = cfg["ns3"]
        # 规则门禁（可为 None：单 env 评估场景自行构造，详情见 RuleGuard）。
        # 由 ParallelCollector 传入**共享实例**，使 N 路采集的战绩汇总到同一
        # 个 RL-vs-规则对比里；单 env 独立构造时没有 A3 来源，门禁保持开放。
        self.guard = guard
        self.binary = str(ns3cfg["binary"])
        self.args_tpl = str(ns3cfg.get("args", ""))
        self.cwd = ns3cfg.get("cwd") or None
        self.timeout_s = float(ns3cfg.get("timeout_s", 600.0))

        self.episode_steps = int(self.env_cfg["episode_steps"])
        self.obs_dim = int(self.env_cfg["obs_dim"])
        self.n_actions = int(self.env_cfg["n_actions"])
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(cfg.get("seed", 42))
        self._rng_run = int(cfg.get("seed", 42))
        self._dt = float(cfg["mobility"]["dt_s"])
        self._bw = float(cfg["channel"]["bw_hz"])
        self._nf = float(cfg["channel"]["noise_figure_db"])

        # episode 内跟踪状态
        self._proc = None
        self._stderr_tail = deque(maxlen=50)
        self._last_io = 0.0
        self._watchdog = None
        self._step = 0
        self._serving_cell = 0
        self._last_ho_step = -10_000
        self._ho_history = []
        self._rlf_count = 0
        self._buf_kb = 0.0
        self._active = set()          # 活跃链路（cellId 集合）
        self._srv_hist = []           # 服务 RSRP 历史（后向趋势）
        self._n1_hist = []
        self._prev_rsrp = None        # 上一状态候选 RSRP（A3 兜底用）
        self._prev_sinr = None
        self._prev_tp = 0.0           # 上一状态实测吞吐（数据面塌陷兜底）
        # 应急兜底条件的连续满足步数（L3 滤波）。
        # 与 handover_env 保持一致：瞬时触发会让单次深衰落就发起切换，
        # 兜底约 40 次/episode，把策略自身的决策完全架空。
        self._fb_count = 0
        self._tp_collapse_count = 0
        # 当前场景族（课程化；None = 单 scenario 模式）与吞吐基线折算值
        self._family = None
        self._tp_baseline = 0.0
        # 切换质量惩罚的待处理窗口 {step, base_sinr}
        self._hoq = None
        # episode 自适应吞吐基线 + 吞吐平滑值
        self._tp_base = None
        self._tp_buf = []
        self._tp_smooth = None
        # 当前场景族的隧道段（观测 #18 用）
        self._tun_segs = []
        # 观测扩充开关（14 → 23 维）。做成开关便于 14 vs 23 的 A/B，
        # 也让 Python 侧/旧行为不受影响。
        self._ext_obs = bool(self.env_cfg.get("extended_obs", False))
        # 负载/隧道观测（需 NS-3 侧新版 HsrState：占用原 reserved[5]）
        self._load_obs = bool(self.env_cfg.get("load_obs", False))
        # 常开硬盾参数（与 src/agents/safety.py 推理侧同一判据，None = 关闭）。
        self._hard = hard_shield_params(cfg)
        # ---- 测量 freshness（修复3）----
        # native 只在 RecvMeasurementReport 刷新 serving 缓存（hsr-rl.cc:350-378），
        # HoEndOk 不刷新（420-462）。于是 HO 被接受后、目标小区的首个上报到达前，
        # 状态携带的仍是**源小区**的弱测量；连续低 SINR 计数若照用旧测量，
        # 就会把「已切向强目标」判成「该目标也已死亡」（H15 两条死亡的实测形态）。
        # 处理：HO 请求被接受后冻结计数，直到 serving_cell 上报为切换目标
        # （新测量到达，恢复真实计数）或冻结超过 measurement_grace_steps
        # （目标迟迟不被上报 → 不再冻结，绝不豁免）。0 = 关闭（保持旧行为）。
        self._meas_grace = int(self.env_cfg.get("measurement_grace_steps", 0))
        self._pending_target = None
        self._grace_until = 0
        self._stale_run = 0
        self._prev_cells = []
        self._meas_stale = False

    # ---------- 进程与 I/O ----------

    def _spawn(self, rng_run: int, family=None):
        self._close_proc()
        base = [a.format(rng_run=rng_run) for a in self.args_tpl.split()]
        # 去掉基础 args 里可能重复的 rngRun（场景参数统一带 rngRun）
        base = [a for a in base if not a.startswith("--rngRun=")]
        scen = build_scenario_args(self.cfg, rng_run, self._rng, family=family)
        # HSR4 真实验收（2026-10-12/13）：NS-3 CommandLine 拒绝空值参数
        # （`--tunnels=` 打印 usage、握手前退出码 1）。判定必须基于**本集采样
        # 结果**而非配置候选列表 —— randomize 可能把非空候选过滤成空（cmd_0199
        # 首跑的 collector ep 失败即此路径）。build_scenario_args 恒发隧道
        # 标记：非空值 → 部署（同时剔除 base 旧部署，避免同名参数重复）；
        # 空值标记 → 不部署，剔除 base 残留参数且不发标记给 C++。
        has_empty_marker = "--tunnels=" in scen
        has_deploy = any(a.startswith("--tunnels=") and a != "--tunnels=" for a in scen)
        if has_empty_marker or has_deploy:
            base = [a for a in base if not a.startswith(("--tunnels=", "--tunnelEnb", "--tunnelMicro"))]
        if has_empty_marker:
            scen = [a for a in scen if a != "--tunnels="]
        # 观测隧道位必须对应本次实际部署命令；不能从未采样的候选列表重算。
        self._tun_segs = []
        for arg in reversed(base + scen):
            if arg.startswith("--tunnels="):
                spec = arg.split("=", 1)[1]
                self._tun_segs = [tuple(float(v) for v in pair.split(":"))
                                  for pair in spec.split(",") if pair]
                break
        cmd = [self.binary] + base + scen
        self._spawn_cmd = cmd          # 最终命令（回归测试核对部署参数用）
        self._proc = subprocess.Popen(
            cmd, cwd=self.cwd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._stderr_tail.clear()
        threading.Thread(target=self._drain_stderr, daemon=True).start()
        self._last_io = time.time()
        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog.start()
        magic = self._read_exact(4)
        self._state_size = 84 if magic == b"HSR3" else HSR_STATE_DTYPE.itemsize
        if magic not in (MAGIC, b"HSR3"):
            raise RuntimeError(
                f"hsr-rl 握手失败: magic={magic!r}, 期望 {MAGIC!r}\n"
                f"stderr: {''.join(self._stderr_tail)}")

    def _drain_stderr(self):
        try:
            for line in self._proc.stderr:
                self._stderr_tail.append(line.decode("utf-8", "replace"))
        except Exception:
            pass

    def _watchdog_loop(self):
        while self._proc is not None and self._proc.poll() is None:
            time.sleep(1.0)
            if time.time() - self._last_io > self.timeout_s:
                self._proc.kill()
                return

    def _read_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                rc = self._proc.poll()
                raise RuntimeError(
                    f"hsr-rl 提前退出 (returncode={rc})，已读 {len(buf)}/{n} 字节\n"
                    f"stderr: {''.join(list(self._stderr_tail)[-15:])}")
            buf += chunk
            self._last_io = time.time()
        return buf

    def _read_state(self):
        # 用 dtype 的实际尺寸（HSR3 = 84B），避免硬编码与协议扩展脱节
        size = getattr(self, "_state_size", HSR_STATE_DTYPE.itemsize)
        raw = self._read_exact(size)
        return np.frombuffer(raw + bytes(HSR_STATE_DTYPE.itemsize - size),
                             dtype=HSR_STATE_DTYPE)[0]

    def _write_action(self, action: int):
        data = np.zeros(1, dtype=HSR_ACTION_DTYPE)
        data["action"] = action
        try:
            self._proc.stdin.write(data.tobytes())
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            rc = self._proc.poll()
            raise RuntimeError(
                f"hsr-rl 写动作失败 (returncode={rc})\n"
                f"stderr: {''.join(list(self._stderr_tail)[-15:])}")
        self._last_io = time.time()

    def _close_proc(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
            for s in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
                try:
                    s.close()
                except Exception:
                    pass
            self._proc = None

    # ---------- 状态 → 候选 / SINR / obs ----------

    def _candidates(self, s) -> tuple:
        """状态 → (候选 RSRP 数组 [serving, n1, n2, n3], cellId 列表)。"""
        rsrp = [float(s["serving_rsrp_dbm"])]
        cells = [int(s["serving_cell"])]
        for i in range(3):
            if i < int(s["n_nbr"]):
                rsrp.append(float(s["nbr_rsrp_dbm"][i]))
                cells.append(int(s["nbr_cell"][i]))
            else:
                rsrp.append(-200.0)
                cells.append(-1)
        return np.array(rsrp, dtype=np.float64), cells

    def _sinr_all(self, rsrp: np.ndarray) -> np.ndarray:
        """估计各候选链路 SINR（与 Python 信道同公式）。"""
        noise_dbm = -174.0 + 10.0 * np.log10(self._bw) + self._nf
        rx = 10.0 ** (rsrp / 10.0)
        noise = 10.0 ** (noise_dbm / 10.0)
        sinr = np.zeros_like(rsrp)
        total = float(np.sum(rx))
        for i in range(len(rsrp)):
            interf = max(total - rx[i], 0.0)
            sinr[i] = 10.0 * np.log10(max(rx[i] / (noise + interf), 1e-12))
        return sinr

    def _shannon(self, sinr_db: float) -> float:
        s = 10.0 ** (sinr_db / 10.0)
        return float(self._bw * np.log2(1.0 + s)) / 1e6

    def _tunnel_segments(self, family=None):
        """当前（基础 scenario + 族覆盖）的隧道段列表。"""
        ns3 = self.cfg.get("ns3", {})
        sc = dict(ns3.get("scenario") or {})
        if family:
            sc.update((ns3.get("scenario_families") or {}).get(family, {}) or {})
        return [(float(seg[0]), float(seg[1])) for seg in (sc.get("tunnels") or [])]

    def _in_tunnel(self, s) -> bool:
        """返回 NS-3 实际部署的隧道位；旧 HSR3 才回退到本地几何。"""
        if getattr(self, "_state_size", HSR_STATE_DTYPE.itemsize) >= HSR_STATE_DTYPE.itemsize:
            return bool(int(s["in_tunnel"]))
        if not self._tun_segs:
            return False
        x = float(s["ue_x"])
        return any(a <= x <= b for a, b in self._tun_segs)

    def _effective_tp_baseline(self, cap: float, meas_valid: bool = True):
        """本步使用的吞吐基线（对数域）。

        mode=family（旧）：用 Stage0 标定的**族平均**，族内仍有 0~32 个背景 UE
            的差异 → 观测→回报仍是「一对多」（G-Dyn3 实测：同 SINR 下 normal
            47Mbps vs congested 18.6Mbps），TD 目标方差大。
        mode=episode（新）：用**本集开局 warmup 步**估计该集的吞吐水平后冻结。
            同一集内是**常数偏移** ⇒ 不改变该集最优策略，只消除跨集的巨大偏移，
            从而在负载不可观测时也能把 TD 目标方差压下来。

        预热期（尚未冻结局部基线）沿用族基线，保证奖励时间上连续。
        """
        mode = str(self.env_cfg.get("tp_baseline_mode")
                   or self.cfg.get("tp_baseline_mode") or "family")
        if mode != "episode":
            return self._tp_baseline
        if self._tp_base is None:
            if meas_valid:
                self._tp_buf.append(float(cap))
                warm = int(self.env_cfg.get("tp_baseline_warmup_steps")
                           or self.cfg.get("tp_baseline_warmup_steps") or 40)
                if len(self._tp_buf) >= warm:
                    self._tp_base = float(np.mean(np.log1p(np.asarray(self._tp_buf))))
            return self._tp_baseline
        return self._tp_base

    def _build_obs(self, s, rsrp, sinr):
        n1 = rsrp[1]
        n2 = rsrp[2]
        speed = float(s["speed_mps"])
        cos_dir = float(s["heading_cos"])
        dist = float(s["dist_to_serving_m"])
        ho_freq = len([h for h in self._ho_history if self._step - h < 60]) / 10.0

        # RSRP 趋势：后向差分（最近 10 步 ≈ 1s）
        srv_trend = (rsrp[0] - self._srv_hist[-10]) if len(self._srv_hist) >= 10 else 0.0
        n1_trend = (n1 - self._n1_hist[-10]) if len(self._n1_hist) >= 10 else 0.0

        # A3 核心判据：最强邻区与服务区 RSRP 差值
        rsrp_diff = (n1 - rsrp[0]) if n1 > -150.0 else -50.0

        # 多链路增益潜力
        cap_serving = self._shannon(sinr[0])
        cap_n1 = self._shannon(sinr[1]) if n1 > -150.0 else 0.0
        dual_gain = float(np.log1p(cap_serving + cap_n1) - np.log1p(max(cap_serving, 1e-6)))

        obs = np.array([
            _norm(rsrp[0], -140, -44),          # 0: 服务 RSRP
            _norm(sinr[0], -30, 30),            # 1: 服务 SINR（估计）
            _norm(n1, -140, -44),               # 2: 最强邻 RSRP
            _norm(n2, -140, -44),               # 3: 次强邻 RSRP
            _norm(speed, 0, 100),               # 4: 速度
            (cos_dir + 1) / 2,                  # 5: 方向余弦
            _norm(dist, 0, 2000),               # 6: 到服务 BS 距离
            _norm(self._buf_kb, 0, 1000),       # 7: 发送缓冲
            1.0 if n1 > -150.0 else 0.0,        # 8: 邻区可用
            _norm(ho_freq, 0, 1),               # 9: 切换频率
            _norm(srv_trend, -30, 30),          # 10: 服务 RSRP 趋势
            _norm(n1_trend, -30, 30),           # 11: 最强邻 RSRP 趋势
            _norm(rsrp_diff, -50, 50),          # 12: RSRP 差值
            _norm(dual_gain, 0, 5),             # 13: 双连通增益潜力
        ], dtype=np.float32)
        if self._ext_obs:
            obs = np.concatenate([obs, np.array([
            # ---- 以下 9 维为 P4 观测扩充（2026-09-18，env.extended_obs 控制）----
            # 动机：NS-3 域 BC（监督学习）用旧 14 维根本拟合不出教师的切换
            # 决策（acc 98.3% 却从不预测切换），说明「何时切、切向谁」在
            # 观测上不可判别 —— RL 更不可能学会。扩充围绕两点：
            #   ① 目标质量：n3 RSRP + 三个候选的 **SINR**（RSRP 强 ≠ 干扰小）
            #   ② 上下文：隧道内外、离 RLF 多近、在本小区待了多久、
            #      规则是否接管、测量是否有效
            _norm(rsrp[3], -140, -44),          # 14: 第 3 邻区 RSRP
            _norm(sinr[1], -30, 30),            # 15: 最强邻 SINR
            _norm(sinr[2], -30, 30),            # 16: 次强邻 SINR
            _norm(sinr[3], -30, 30),            # 17: 第 3 邻 SINR
            1.0 if self._in_tunnel(s) else 0.0,  # 18: NS-3 实际部署隧道位
            min(self._rlf_count / max(int(self.env_cfg["rlf_consecutive_steps"]), 1), 1.0),
                                                # 19: 离 RLF 判定有多近
            _norm(self._step - self._last_ho_step, 0, 120),
                                                # 20: 在本小区已停留步数
            1.0 if self._rules_active() else 0.0,  # 21: 规则是否接管
            1.0 if float(rsrp[0]) > -190.0 else 0.0,  # 22: 测量是否已上报
            ], dtype=np.float32)])

        # ---- 负载观测（设计文档 P1-6，env.load_obs 控制，需 NS-3 侧新版协议）----
        # 旧观测里没有任何负载维度（只有发送缓冲作代理），于是「切向空闲次强
        # 小区做负载分流」在观测上**不可实现**：RL 看不到「空闲」。这里补上
        # 各候选小区的背景 UE 数（负载代理）与隧道指示，使该目标可学。
        # 归一化分母取 max_bg_ue（默认 32，即场景背景 UE 上限）。
        if getattr(self, "_load_obs", False):
            denom = max(1.0, float(self.env_cfg.get("load_norm_denom", 32.0)))
            obs = np.concatenate([obs, np.array([
                min(float(s["load_serving"]) / denom, 1.0),      # 负载：服务小区
                min(float(s["load_nbr"][0]) / denom, 1.0),        # 负载：最强邻
                min(float(s["load_nbr"][1]) / denom, 1.0),        # 负载：次强邻
                min(float(s["load_nbr"][2]) / denom, 1.0),        # 负载：第 3 邻
                float(s["in_tunnel"]),                            # 隧道内指示
            ], dtype=np.float32)])
        if self._ext_obs:
            assert obs.shape[0] == self.obs_dim, (
                f"obs_dim 不符（extended_obs+load_obs 应为 "
                f"{23 + (5 if getattr(self, '_load_obs', False) else 0)}，"
                f"配置为 {self.obs_dim}）")
        else:
            assert obs.shape[0] == self.obs_dim, (
                f"extended_obs 关闭时 obs_dim 应为 14，配置为 {self.obs_dim}")
        return obs

    # ---------- Gym API ----------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._rng_run = int(seed)
        # 场景族（课程化）：options={"family": name}；缺省回退单 scenario 模式。
        fam = (options or {}).get("family")
        self._family = fam
        # HSR4 的 in_tunnel 位来自 NS-3 实际部署，是运行时唯一事实。
        # _tun_segs 仅保留给旧协议回退，不从候选配置推断，避免随机化后错配。
        self._tun_segs = []
        # 吞吐基线折算（方案 §5.4）：相对同族 A3 教师均值。>0 时
        # r_tp = log1p(tp) - log1p(baseline) —— 「打平教师即零、超越为正」，
        # 是规则门禁在奖励层的镜像，让 RL 从第一步就有差异化信号。
        self._tp_baseline = float(
            (self.cfg.get("reward_baselines") or {}).get(fam, 0.0)
            if fam else 0.0)
        # episode 自适应基线（G-Dyn3 驱动）：用本集开局若干步估计该集的
        # 「场景吞吐水平」，作为本集奖励的常数基线。
        self._tp_base = None
        self._tp_buf = []
        self._tp_smooth = None
        self._spawn(self._rng_run, family=fam)
        s = self._read_state()
        rsrp, cells = self._candidates(s)
        sinr = self._sinr_all(rsrp)
        # 首次状态在 t=0.3s，测量尚未上报（rsrp=-200 哨兵）→ 中性占位
        if float(rsrp[0]) <= -190.0:
            sinr = np.full_like(sinr, 15.0)

        self._step = 0
        self._serving_cell = cells[0]
        self._last_ho_step = -10_000
        self._ho_history = []
        self._rlf_count = 0
        self._buf_kb = float(self._rng.uniform(100, 800))
        self._active = {cells[0]}
        self._srv_hist = [rsrp[0]]
        self._n1_hist = [rsrp[1]]
        self._prev_rsrp = rsrp
        self._prev_sinr = sinr
        self._prev_tp = float(s["throughput_mbps"])
        self._fb_count = 0
        self._tp_collapse_count = 0
        self._hoq = None
        self._pending_target = None      # 测量 freshness（修复3）
        self._grace_until = 0
        self._stale_run = 0
        self._meas_stale = False
        self._prev_cells = list(cells)

        obs = self._build_obs(s, rsrp, sinr)
        info = self._info(s, rsrp, sinr, cells)
        return obs, info

    def _rules_active(self) -> bool:
        """规则是否在本步生效。

        strict 模式：每步都掩码（历史行为）。
        guard  模式：默认**放行**给 RL 探索 —— 只有在 RuleGuard 判定 RL 已
        退化到比规则差时才让规则接管。这样 RL 才有空间发现「切向次强/空闲
        小区」这类规则学不到、但可能存在增益的决策。
        """
        if self.guard is not None:
            return bool(self.guard.active)
        return str(self.cfg.get("safety", {}).get("mode", "strict")) == "strict"

    def step(self, action: int):
        from src.agents.safety import HO_ACTIONS, DUAL_ACTIONS
        self._step += 1
        action_requested = int(action)

        # A3 兜底（与 HandoverEnv 同口径：持续满足 fallback_persist_steps 步）
        # 锁步协议下只能拿到上一状态测量值（写动作 → 读状态），故用 _prev_sinr；
        # 但**必须**加持续性判据，否则单次深衰落就触发一次切换。
        action_source = "rl"
        fallback_sinr = float(self.env_cfg.get("rlf_fallback_sinr_db", -5.0))
        fb_persist = int(self.env_cfg.get("fallback_persist_steps", 3))
        if self._prev_sinr[0] < fallback_sinr:
            self._fb_count += 1
        else:
            self._fb_count = 0
        # 数据面塌陷兜底：隧道口部 serving RSRP 仍约 -62dBm，估计 SINR 不破
        # -5dB，但 dl≈0。仅靠 SINR 路径永远不开火。吞吐连续低于门限且邻区
        # 已过 A3 偏置时强制切（persist 同 fallback_persist_steps，避开 HO 中断窗）。
        collapse_thr = float(self.env_cfg.get("fallback_tp_collapse_mbps", 0.0))
        if collapse_thr > 0.0 and self._prev_tp < collapse_thr:
            self._tp_collapse_count += 1
        else:
            self._tp_collapse_count = 0
        sinr_fb = self._fb_count >= fb_persist
        tp_fb = collapse_thr > 0.0 and self._tp_collapse_count >= fb_persist
        # Opt-in deadline rescue: the persistent fallback must not wait past
        # the unchanged local RLF deadline. Require a legal A3 improvement;
        # never exempt an ongoing/finished HO from the termination detector.
        deadline_fb = (bool(self.env_cfg.get("fallback_before_rlf", False))
                       and self._rlf_count >= max(1, int(self.env_cfg["rlf_consecutive_steps"]) - 2))
        if (sinr_fb or tp_fb or deadline_fb) and action not in HO_ACTIONS \
                and action not in DUAL_ACTIONS:
            a3 = self._a3_action(self._prev_rsrp)
            if a3 != 0:
                action = a3
                action_source = "safety_a3"
                self._fb_count = 0
                self._tp_collapse_count = 0

        # ---- 常开硬盾（修复1，环境侧）----
        # configured 模式 + guard 关闭时，下面的规则接管（_rules_active）为
        # False，弱目标保护整层消失；实测两条死亡正是被释放的 action5 切向
        # 弱 9~11 dB、目标自身 SINR≈-13 dB 的小区（focused_root_cause_20261014）。
        # 硬盾与性能门禁**解耦**：只拦「切过去就低于终止门限」的动作，
        # 不拦「略弱但健康」的负载均衡目标，因此不受 _rules_active 门控。
        # 判据与 src/agents/safety.py:hard_mask 同源（同一 cfg），避免推理/环境分叉。
        if self._hard is not None and action in HO_ACTIONS \
                and float(self._prev_rsrp[0]) > -190.0 \
                and hard_block_reason(self._prev_rsrp, self._prev_sinr,
                                      action, self._hard):
            action = 0
            action_source = "masked_hard"

        # NS-3 后端弱目标动作掩码：LENA 对 HandoverRequest 照单执行（无 RRC
        # 重配失败回退），RL 若选择切向 RSRP 过弱的次强/第三邻区，切过去即
        # RLF（实测 mixed 训练后 5/10 集因此失服，而 A3/bc_init 0 次）。
        # 与 src/agents/safety.py 的 SafetyLayer 同口径补网（Python env 有
        # 这层，NS-3 env 此前只有 A3 兜底）：弱目标或刚切换完的 HO 回退 stay。
        # 锁步协议用 _prev_rsrp（上一状态测量）；测量未上报（-200 哨兵）时不拦。
        #
        # ⚠ 但这层「每步都管」会把 RL 的探索空间压死：评估实测 79.3% 的 RL
        # 决策被改写成 stay（切次强小区做负载分流被 RSRP 门限拦下），退化成
        # ho=0 + 早早 RLF。故改为**门禁化**：默认放行（给 RL 探索空间），仅当
        # RuleGuard 判定 RL 已退化到比规则差时才拉起规则（见 safety.RuleGuard）。
        if action in HO_ACTIONS and float(self._prev_rsrp[0]) > -190.0 \
                and self._rules_active():
            safety_cfg = self.cfg.get("safety", {})
            tgt_idx = {1: 1, 2: 2, 5: 3}[action]
            tgt_rsrp = float(self._prev_rsrp[tgt_idx])
            min_rsrp = float(safety_cfg.get("min_target_rsrp_dbm", -110))
            max_drop = float(safety_cfg.get("max_drop_vs_serving_db", 6))
            block_steps = int(safety_cfg.get("pingpong_block_steps", 2))
            recent_ho = (self._step - self._last_ho_step) <= block_steps
            weak = (tgt_rsrp < min_rsrp
                    or tgt_rsrp < float(self._prev_rsrp[0]) - max_drop)
            if weak or recent_ho:
                action = 0
                action_source = "masked_weak"

        # NS-3 后端动作掩码：LENA 无 EN-DC，动作 3/4/6（双连/三连）在物理层
        # 是空操作，不应作为真实增益进入策略或评估。此处强制回退到 stay(0)，
        # 保证 RL 在 NS-3 上的决策空间诚实（真实双连需迁移 5G-LENA/nr）。
        if action in DUAL_ACTIONS:
            action = 0
            action_source = "masked_dual"

        # 登记实际派发的 HO 目标小区（锁步协议：动作相对**上一状态**的候选
        # 列表），供下面的测量 freshness 判定使用。
        if self._meas_grace > 0 and action in HO_ACTIONS:
            i = HO_TARGET_IDX.get(action)
            if i is not None and i < len(self._prev_cells) \
                    and int(self._prev_cells[i]) > 0:
                self._pending_target = int(self._prev_cells[i])
                self._grace_until = self._step + self._meas_grace

        self._write_action(int(action))
        s = self._read_state()
        if int(s["done"]) == 1:
            self._close_proc()
            # NS-3 到时收尾：用末状态构造截断步
            truncated = True
            terminated = False
        else:
            truncated = self._step >= self.episode_steps
            terminated = False

        rsrp, cells = self._candidates(s)
        sinr = self._sinr_all(rsrp)

        # 测量预热宽限：NS-3 中 UE 附着 + 首次 A3 测量上报要到 t≈1.9s
        # （实测 --test 模式 0.3-1.8s serving_rsrp 恒为 -200 哨兵值）。
        # 此前测量不可用，不能据此累计 RLF（rlf_consecutive_steps=3 会在 0.6s
        # 误杀 episode，导致评估恒为 outage=1/throughput=0），也不应触发 A3 兜底。
        # 用中性 SINR 占位，首次有效上报后恢复真实口径。
        meas_valid = float(rsrp[0]) > -190.0
        if not meas_valid:
            sinr = np.full_like(sinr, 15.0)
            self._fb_count = 0
            self._tp_collapse_count = 0
            self._rlf_count = 0

        # 活跃链路更新（与 _apply_action 语义对齐；真实切换以 s.serving_cell 为准）
        ho_event = bool(int(s["ho_event"]))
        if ho_event:
            self._serving_cell = int(s["serving_cell"])
            self._last_ho_step = self._step
            self._ho_history.append(self._step)

        # ---- 测量 freshness 判定（修复3，只冻结终止计数）----
        # 派发 HO 后、目标小区的测量上报到达前，serving 上报仍是源小区
        # （native 只在 RecvMeasurementReport 刷新缓存，HoEndOk 不刷新）。
        # 规则见 measurement_staleness；绝不豁免终止标准本身。
        (self._meas_stale, self._pending_target, self._grace_until,
         self._stale_run) = measurement_staleness(
                self._step, self._pending_target, self._grace_until,
                self._meas_grace, int(s["serving_cell"]),
                bool(int(s["ho_failed"])), bool(int(s["ho_in_progress"])),
                meas_valid, self._stale_run)
        if action in (1, 2, 5) or ho_event:
            self._active = {int(s["serving_cell"])}
        elif action in (3, 4):
            add = cells[1] if action == 3 else cells[2]
            self._active = {int(s["serving_cell"]), add}
        elif action == 6:
            self._active = {int(s["serving_cell"]), cells[1], cells[2]}

        # 吞吐：服务链路实测 + 附加链路（dual 近似）Shannon 估计
        cap = float(s["throughput_mbps"])
        serving = int(s["serving_cell"])
        for i in range(1, 4):
            if cells[i] in self._active and cells[i] != serving and cells[i] > 0:
                cap += self._shannon(sinr[i])

        # 动态缓冲
        buf_arrival_kbps = float(self.env_cfg.get("buf_arrival_kbps", 5000.0))
        self._buf_kb = max(0.0, self._buf_kb + buf_arrival_kbps * self._dt / 1000.0
                           - cap * self._dt)

        # 吞吐：服务链路实测 + 附加链路（dual 近似）Shannon 估计。
        # 基线折算（方案 §5.4）：配置了 reward_baselines[family] 时减去
        # 同族 A3 教师的对数吞吐 —— 打平教师为零、超越为正、落后为负，
        # RL 从第一步起就有「相对规则」的差异化信号（与 RuleGuard 互为
        # 奖励层/决策层的镜像）。未标定时（=0）保持原口径。
        # ---- 吞吐平滑（奖励噪声探针实测驱动）----
        # 实测：NS-3 实测吞吐 CV≈1.1、**45% 的步接近 0**，奖励项 std 达 1.90，
        # 比切换罚(0.9/次)、弱覆盖(0.6/步)高一个数量级 → 直接主导 TD 目标方差，
        # 是 Q 学习跨场景抖动（ho=0 / ho=28）的直接推手。
        # EMA(K=10) 可把奖励项 std 从 1.90 降到 0.53（−72%），延迟仅约 0.5s，
        # 与 G-Form 给出的 3~6s 决策视野兼容。
        k_smooth = int(self.env_cfg.get("tp_smooth_k")
                       or self.cfg.get("tp_smooth_k") or 0)
        if k_smooth > 0:
            a = 1.0 / float(k_smooth)
            self._tp_smooth = float(cap) if self._tp_smooth is None else \
                a * float(cap) + (1.0 - a) * float(self._tp_smooth)
            tp_used = self._tp_smooth
        else:
            tp_used = float(cap)

        # 基线（对数域均值，见 calibrate_scenes.py 的 log_tp_mean / 本集开局估计）
        base = self._effective_tp_baseline(tp_used, meas_valid)
        if base > 0.0:
            # 注意：基线本身就是**对数域均值**，直接相减即可。
            # 早期实现误写成 log1p(base) → 每步凭空多出约 log1p(2.117)-2.117
            # ≈ -0.99 的常数偏移，把 Q 整体抬高、动作间的小差异被淹没。
            r_throughput = float(np.log1p(tp_used) - base)
        else:
            r_throughput = float(np.log1p(tp_used))

        # ---- 切换质量即时惩罚（诊断驱动的新增项）----
        # 死亡诊断（scripts/diagnose_rl.py）：12 集里 8 集死于 ho_to_weaker
        # （切向更弱小区），且死前 30 步 SINR 崩到 -24dB。根因之一是
        # **信用分配链太长**：切错一次的即时代价只有 ho_penalty×w_handover
        # = 0.9，真实后果（RLF）却在几十步之后，n 步回报覆盖不到 → Q 学不到
        # 「这次切错了」。
        # 对策：切换后 K 步内，新链路 SINR 每低于切换前 1dB 就按权重扣一次，
        # 并**分摊到 K 步**（切换当步即带负奖励，不必等 RLF）。
        r_hoq = 0.0
        hoq_w = float(self.env_cfg.get("ho_quality_w", 0.0))
        if hoq_w > 0.0:
            k_win = max(1, int(self.env_cfg.get("ho_quality_window_steps", 10)))
            cap_db = float(self.env_cfg.get("ho_quality_max_drop_db", 20.0))
            if ho_event:
                # 切换当步：基准 = 切换前（上一状态）的服务链路 SINR
                self._hoq = {"step": int(self._step),
                             "base": float(self._prev_sinr[0])}
            if self._hoq is not None:
                d = int(self._step) - int(self._hoq["step"])
                if 0 <= d < k_win:
                    deficit = max(0.0, self._hoq["base"] - float(sinr[0]))
                    r_hoq = -hoq_w * min(deficit, cap_db) / float(k_win)
                else:
                    self._hoq = None
        r_ho = 0.0
        if ho_event:
            r_ho -= float(self.env_cfg["ho_penalty"])
        if self._pingpong():
            r_ho -= float(self.env_cfg["pingpong_penalty"])
        r_delay = -float(s["ho_interrupt_ms"]) / 100.0
        r_stab = float(self.env_cfg["stability_bonus"]) if len(self._active) > 1 else 0.0

        # 弱覆盖预警（与 HandoverEnv 同口径）：纠正 RL 为省切换而赖在劣化小区
        r_outage = 0.0
        out_w = float(self.env_cfg.get("w_outage", 0.0))
        if out_w > 0.0:
            rlf_thr = float(self.env_cfg["rlf_sinr_db"])
            warn = float(self.env_cfg.get("weak_coverage_sinr_db", -2.0))
            if float(sinr[0]) < rlf_thr:
                r_outage = -out_w * 2.0
            elif float(sinr[0]) < warn:
                span = max(warn - rlf_thr, 1e-6)
                r_outage = -out_w * (warn - float(sinr[0])) / span

        # ---- 覆盖质量项（路径 A：奖励主项改用**可预测的 SINR**）----
        # 依据（本轮三个探针的量化结果）：
        #   G-Form ：RSRP/SINR 动力学外推 MAE 1.31dB vs 持久性 4.73dB，跨族
        #            仅 ×1.00~1.46 ⇒ **可预测、可泛化**
        #   G-Dyn3 ：吞吐由 SINR 预测的 r 仅 0.11~0.32，同 SINR 下族间吞吐差
        #            2.5 倍 ⇒ 存在未观测负载，**不可预测**
        #   噪声探针：实测吞吐 CV≈1.12、45% 的步接近 0，奖励项 std 1.90
        #            （比切换罚 0.9/次高一个数量级）
        # 因此把「吞吐」降为次要项，改以服务链路 SINR 的覆盖质量为主项：
        # 既可直接预测（G-Form 已验证），又与「是否存活」严格同向 —— 赖在
        # 劣化小区会立刻在覆盖项上扣分，无需等几十步后的 RLF。
        r_cov = 0.0
        w_cov = float(self.env_cfg.get("w_coverage", 0.0))
        if w_cov > 0.0 and meas_valid:
            rlf_thr = float(self.env_cfg["rlf_sinr_db"])
            good = float(self.env_cfg.get("coverage_good_sinr_db", 10.0))
            q = float(np.clip((float(sinr[0]) - rlf_thr)
                              / max(good - rlf_thr, 1e-6), 0.0, 1.0))
            r_cov = w_cov * (2.0 * q - 1.0)     # −w_cov … +w_cov（SINR≥good 取满）

        # 目标差距项：邻区明显更好却赖着不切 → 小罚。教的是「切向更好的
        # 小区」这一核心技能，判据用 SINR（可预测）而非吞吐（不可预测）。
        r_gap = 0.0
        w_gap = float(self.env_cfg.get("w_gap", 0.0))
        if w_gap > 0.0 and meas_valid:
            scale = float(self.env_cfg.get("coverage_gap_scale_db", 15.0))
            gap = max(0.0, float(np.max(sinr[1:4])) - float(sinr[0]))
            r_gap = -w_gap * float(np.clip(gap / scale, 0.0, 1.0))

        # ---- 显式负载项（机制审计驱动）----
        # 覆盖/吞吐/邻区差距三项都没直接给「避开拥塞小区」的梯度：吞吐只在很
        # 间接的层面奖励低负载，于是 RL 学会了更好的切换时机，却**没有**学会
        # 负载分流（审计：congested 上时间均值负载 1.478 → 1.473，几乎无差）。
        # 这里补上显式梯度：待在比最轻的有效邻区更拥挤的小区 → 按超出量计罚。
        # 只比较测量有效的邻区（见 load_gap_penalty 说明）。
        r_load = 0.0
        w_load = float(self.env_cfg.get("w_load", 0.0))
        if w_load > 0.0 and meas_valid:
            valid_n = [bool(float(rsrp[i]) > -190.0) for i in (1, 2, 3)]
            r_load = -w_load * load_gap_penalty(
                float(s["load_serving"]), [float(x) for x in s["load_nbr"]], valid_n,
                scale_ue=float(self.env_cfg.get("load_gap_scale_ue", 4.0)),
                deadband_ue=float(self.env_cfg.get("load_gap_deadband_ue", 1.0)))

        reward = (float(self.env_cfg["w_throughput"]) * r_throughput
                  + float(self.env_cfg["w_handover"]) * r_ho
                  + float(self.env_cfg["w_delay"]) * r_delay
                  + float(self.env_cfg["w_stability"]) * r_stab
                  + r_outage
                  + r_hoq       # 切换质量（切后 SINR 恶化）
                  + r_cov       # 覆盖质量（主项，可预测）
                  + r_gap       # 目标差距（切向更好的小区）
                  + r_load)     # 负载差距（避开拥塞小区，显式负载项）

        # Keep the decomposition observable for diagnostics/tests; the scalar
        # reward remains exactly the configured weighted sum.
        reward_components = {
            "throughput": float(self.env_cfg["w_throughput"]) * r_throughput,
            "handover": float(self.env_cfg["w_handover"]) * r_ho,
            "delay": float(self.env_cfg["w_delay"]) * r_delay,
            "stability": float(self.env_cfg["w_stability"]) * r_stab,
            "outage": r_outage,
            "handover_quality": r_hoq,
            "coverage": r_cov,
            "neighbor_gap": r_gap,
            "load": r_load,
            "rlf": 0.0,
        }

        # RLF：NS-3 真实 RLF 或连续低 SINR（与 Python env 一致）。
        # 惩罚必须按**剩余步数**计，而非固定 -10：否则 agent 会发现
        # 「永不切换」可规避 ho_penalty 却只需付一次 -10 → 退化策略。
        # 这里与评估口径（早停后剩余步计 0 吞吐 + 中断）一致。
        #
        # 剩余步数封顶（方案 §5.4，rlf_remaining_cap_steps）：全灭分布下
        # 每集都是数百量级的同款负数，Q 没有区分度且 PER 会放大灾难样本；
        # 封顶后早死仍明显劣于活到最后，但梯度不被淹没。0 = 不封顶（旧行为）。
        remaining = max(0, int(self.episode_steps) - int(self._step))
        cap_steps = int(self.env_cfg.get("rlf_remaining_cap_steps", 0))
        if cap_steps > 0:
            remaining = min(remaining, cap_steps)
        rlf_pen = (float(self.env_cfg.get("rlf_drop_penalty", 1.0)) * remaining
                   + float(self.env_cfg.get("rlf_termination_penalty", 10.0)))
        native_rlf = meas_valid and int(s["rlf"]) == 1
        if native_rlf:
            terminated = True
        if self._meas_stale:
            pass  # 冻结：陈旧测量既不累计也不清零（见上方 freshness 判定）
        elif meas_valid and sinr[0] < float(self.env_cfg["rlf_sinr_db"]):
            self._rlf_count += 1
        elif meas_valid:
            self._rlf_count = 0
        consecutive_rlf = self._rlf_count >= int(self.env_cfg["rlf_consecutive_steps"])
        if consecutive_rlf:
            terminated = True
        r_rlf = -float(rlf_pen) if (native_rlf or consecutive_rlf) else 0.0
        if native_rlf or consecutive_rlf:
            # Native NS-3 RLF and the local consecutive-low-SINR detector may
            # agree on the same step; this is one termination event, not two.
            reward += r_rlf
            reward_components["rlf"] = r_rlf

        # 历史更新（趋势特征）
        self._srv_hist.append(rsrp[0])
        self._n1_hist.append(rsrp[1])
        self._prev_rsrp = rsrp
        self._prev_sinr = sinr
        self._prev_tp = cap
        self._prev_cells = cells

        obs = self._build_obs(s, rsrp, sinr)
        info = self._info(s, rsrp, sinr, cells)
        info["throughput_mbps"] = cap
        info["ho_event"] = ho_event
        # 兼容别名，仅为 RRC Start→EndOk 时长，不可解释为包级业务中断。
        info["ho_interrupt_ms"] = float(s["ho_interrupt_ms"])
        info["pingpong"] = bool(self._pingpong())
        info["action_source"] = action_source
        # 审计用：本步规则是否接管（False = 放行给 RL 探索）
        info["guard_active"] = self._rules_active()
        # 当前场景族（课程化审计：buffer/评估按族统计）
        info["family"] = self._family
        # 负载/隧道（P1-6 审计：用于验证「切向空闲小区」是否真的发生）
        info["load_serving"] = int(s["load_serving"])
        info["load_nbr"] = [int(x) for x in s["load_nbr"]]
        info["in_tunnel_ns3"] = int(s["in_tunnel"])
        # 容量/业务量（100~1000 用户规模验证指标；HSR3 口径）
        info["n_ue_cell"] = int(s["n_ue_cell"])
        info["n_bg_total"] = int(s["n_bg_total"])
        info["tp_agg_mbps"] = float(s["tp_agg_mbps"])          # 全网聚合
        info["ctrl_delay_ms"] = float(s["ctrl_delay_ms"])
        info["tp_agg_cell_mbps"] = float(s["tp_agg_cell_mbps"])  # 服务小区聚合
        info["tp_ue_p05_mbps"] = float(s["tp_ue_p05_mbps"])      # 边缘用户速率
        info["tp_ue_p50_mbps"] = float(s["tp_ue_p50_mbps"])
        info["jain"] = float(s["jain"])
        # 实际执行的动作（经 safety_a3 兜底 / dual mask 之后）。buffer 与
        # 评估统计必须用它：策略输出 a=3/4/6 会被 mask 成 stay(0) 执行，
        # 若按输出动作入 buffer，等于教网络「dual 动作带来 stay 的后果」。
        info["action_requested"] = action_requested
        info["action_executed"] = int(action)  # dispatched, not necessarily completed
        info["termination_reason"] = ("native_rlf" if native_rlf else
                                      "low_sinr" if consecutive_rlf else
                                      "truncated" if truncated else None)
        info["rlf_count"] = self._rlf_count
        info["fallback_count"] = self._fb_count
        # 审计：本步 serving 测量是否仍属切换前的小区（修复3 冻结中）
        info["meas_stale"] = bool(self._meas_stale)
        info["reward_components"] = reward_components

        if terminated or truncated:
            self._close_proc()
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _info(self, s, rsrp, sinr, cells):
        return {
            "telemetry_available": getattr(self, "_state_size", 99) == 99,
            "ho_request_result": int(s["ho_request_result"]),
            "ho_in_progress": bool(s["ho_in_progress"]),
            "ho_failed": bool(s["ho_failed"]),
            "rrc_ho_duration_ms": float(s["ho_interrupt_ms"]),
            "interval_ms": float(s["interval_ms"]),
            "no_rx_interval_ms": float(s["no_rx_interval_ms"]),
            "no_rx_run_ms": float(s["no_rx_run_ms"]),
            "time_ms": int(s["time_ms"]),
            "candidate_rsrp": rsrp.copy(),
            # 负载/隧道（P1-6）：诊断「切向空闲小区」是否真的发生
            "load_serving": int(s["load_serving"]),
            "load_nbr": [int(x) for x in s["load_nbr"]],
            "in_tunnel_ns3": int(s["in_tunnel"]),
            # 候选 SINR（估计）：诊断/观测需要知道「切过去是强还是只有 RSRP 强」
            "candidate_sinr": sinr.copy(),
            "serving_idx": int(s["serving_cell"]),
            "serving_rsrp": float(rsrp[0]),
            "serving_sinr": float(sinr[0]),
            # 绝对位置：诊断死亡地点（是否死在洞口）必需
            "ue_x": float(s["ue_x"]),
            # NS-3 上报的真实 RLF 标志（与 env 判定的 terminated 区分）
            "rlf": int(s["rlf"]),
            # 必须与 candidate_rsrp 长度对齐（无效邻区用 -1 占位），
            # 否则 A3 基线的按小区 L3 滤波会退化成按位置滤波。
            "neighbors": [int(c) for c in cells[1:]],
            "step_idx": int(self._step),
            "last_ho_step": int(self._last_ho_step),
            "active_links": list(self._active),
            "pingpong": False,
            "ho_tau": 0.0,
        }

    def _pingpong(self):
        """仅在乒乓事件发生的那一步为 True（与 handover_env 口径一致）。

        原实现一次乒乓会在随后连续 3 步都为 True，使指标变成「乒乓状态步数」
        而非「乒乓次数」，并被逐步重复计入奖励惩罚。
        """
        if len(self._ho_history) < 2:
            return False
        last = self._ho_history[-1]
        prev = self._ho_history[-2]
        return self._step == last and (last - prev) <= 2

    def _a3_action(self, rsrp) -> int:
        """A3 回落：切到最强合法邻区；无可行则 stay。"""
        min_rsrp = float(self.cfg["safety"]["min_target_rsrp_dbm"])
        serving_rsrp = rsrp[0]
        best, best_i = -200.0, -1
        for i in range(1, 4):
            if rsrp[i] <= -150.0:
                continue
            if rsrp[i] > serving_rsrp + 3.0 and rsrp[i] >= min_rsrp:
                if rsrp[i] > best:
                    best, best_i = rsrp[i], i
        if best_i < 0:
            return 0
        return {1: 1, 2: 2, 3: 5}[best_i]

    def close(self):
        self._close_proc()
        return super().close()
