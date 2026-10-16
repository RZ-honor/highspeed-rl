import numpy as np
import torch
import torch.nn as nn

from src.agents.safety import SafetyLayer
import torch.optim as optim


class A3Baseline:
    """3GPP A3：邻区 RSRP > 服务+3dB 持续 TTT → 切最强邻区。"""

    def __init__(self, safety_cfg):
        from src.agents.safety import SafetyLayer
        self.sl = SafetyLayer(safety_cfg)
        self._trig_count = 0
        self._last_tgt = None

    def act(self, obs, info):
        cand = info["candidate_rsrp"]
        serving = cand[0]
        nbr = cand[1:4] if len(cand) > 3 else cand[1:]
        best, best_i = -200.0, -1
        for i, r in enumerate(nbr):
            if r > serving + 3.0 and r >= self.sl.min_rsrp:
                if r > best:
                    best, best_i = r, i
        if best_i >= 0:
            if self._last_tgt == best_i:
                self._trig_count += 1
            else:
                self._trig_count = 1
                self._last_tgt = best_i
            # 100ms 粒度下 TTT=40ms 已满足
            if self._trig_count >= 1:
                self._trig_count = 0
                return {0: 1, 1: 2, 2: 5}[best_i]
        else:
            self._trig_count = 0
        return 0


class A3OperatorBaseline:
    """运营商级 A3：磁滞 + Time-to-Trigger(TTT) 定时器 + 多普勒自适应。

    对标现网高铁专网实际配置（非理想化 A3）：
      - 磁滞 hysteresis：邻区需持续优于服务小区 hyst_db 才进入触发评估
      - A3 偏置 offset：3GPP A3 事件门限偏置
      - TTT 定时器：条件满足后需持续 ttt_ms 才真正执行切换（防快衰落误判）
      - 多普勒自适应：速度越高 TTT 越短（现网提速补偿），
        但高速下缩短 TTT 又会放大乒乓——这正是 A3 在 350km/h 的两难。

    现网典型参数：hyst=2dB, a3_offset=3dB, TTT=160~320ms。
    350km/h 下 TTT=320ms 意味着列车已走过 ~31m，切换点严重滞后。
    """

    def __init__(self, safety_cfg, a3_cfg=None, dt_s=0.1):
        self.sl = SafetyLayer(safety_cfg)
        a3_cfg = a3_cfg or {}
        self.hyst_db = float(a3_cfg.get("hyst_db", 2.0))
        self.a3_offset_db = float(a3_cfg.get("a3_offset_db", 3.0))
        self.ttt_ms = float(a3_cfg.get("ttt_ms", 320.0))
        self.doppler_adaptive = bool(a3_cfg.get("doppler_adaptive", True))
        self.dt_s = float(dt_s)
        # TTT 定时器状态：记录每个候选邻区条件已持续满足的时长
        self._ttt_elapsed_ms = 0.0
        self._ttt_target = None  # 当前 TTT 计时的目标邻区索引
        # 多普勒：3.5GHz 下 fc_hz 由 cfg 提供，速度从 obs[4] 反归一化
        self._fc_hz = float(a3_cfg.get("fc_hz", 2.1e9))
        # 3GPP L3 测量滤波：Fn = (1-a)·F(n-1) + a·Mn，**按小区 ID 滤波**
        self._l3_alpha = float(a3_cfg.get("l3_alpha", 0.4))
        # 原始测量辅助进入：滤波值未过门限但瞬时 Mn > Ms+Off 时也启动 TTT。
        # 默认关，避免把现网 L3 语义掏空；隧道口部微站骤升时由 eval_tunnel 打开。
        self.raw_enter = bool(a3_cfg.get("raw_enter", False))
        self._filt = {}       # cell_id -> 滤波后的 RSRP
        self._ttt_cell = None  # TTT 计时的目标小区（按 cell_id，非秩位）

    def _effective_ttt_ms(self, speed_mps):
        """多普勒自适应 TTT：速度越高，TTT 越短（现网提速补偿）。

        物理依据：多普勒频偏 fd = v·fc/c。350km/h@2.1GHz → fd≈680Hz，
        信道相干时间 Tc ≈ 1/(2·fd) ≈ 0.74ms，远小于 TTT。
        高速下若 TTT 不变，切换点严重滞后 → 服务小区已跌破 RLF 才切。
        故现网按速度缩短 TTT，但这会削弱抗快衰落能力 → 乒乓回升。
        """
        if not self.doppler_adaptive:
            return self.ttt_ms
        # 基准 300km/h 对应标称 TTT；速度每 +50km/h，TTT 缩减 15%
        v_kmh = speed_mps * 3.6
        scale = max(0.4, 1.0 - 0.15 * max(0.0, (v_kmh - 300.0) / 50.0))
        return self.ttt_ms * scale

    def _l3_filter(self, cells, rsrp_raw):
        """3GPP L3 测量滤波，返回滤波后的候选 RSRP。

        **必须按物理小区 ID 滤波，不能按 RSRP 秩位。** 秩位每步由原始瞬时
        RSRP 重排，快衰落下相邻两步的「第 1 邻区」常常是两个不同的小区；
        按秩位滤波等于把不同小区的测量值混在一个槽位里做平滑——既不符合
        3GPP 语义，也让 TTT 定时器被虚假的「目标变更」反复复位，
        这正是当初 A3 在 1528 步里只触发 1 次的真实原因。
        """
        if len(cells) == len(rsrp_raw):
            keys = list(cells)
        else:
            # 防御：info 里小区列表与候选 RSRP 长度不一致时退化为按位置滤波
            keys = [("pos", j) for j in range(len(rsrp_raw))]
        a = self._l3_alpha
        out = np.empty(len(rsrp_raw), dtype=np.float64)
        for j, k in enumerate(keys):
            prev = self._filt.get(k)
            v = rsrp_raw[j] if prev is None else (1.0 - a) * prev + a * rsrp_raw[j]
            self._filt[k] = v
            out[j] = v
        return out, keys

    def act(self, obs, info):
        rsrp_raw = np.asarray(info["candidate_rsrp"], dtype=np.float64)
        cells = [int(info["serving_idx"])] + [int(c) for c in info.get("neighbors", [])]
        cand, keys = self._l3_filter(cells, rsrp_raw)

        serving = cand[0]
        nbr = cand[1:4] if len(cand) > 3 else cand[1:]
        nbr_keys = keys[1:4] if len(keys) > 3 else keys[1:]
        raw_serving = float(rsrp_raw[0])
        raw_nbr = rsrp_raw[1:4] if len(rsrp_raw) > 3 else rsrp_raw[1:]
        min_rsrp = self.sl.min_rsrp

        # 3GPP A3 进入条件：Mn + Ofn + Ocn - Hys > Ms + Ofs + Ocs + Off
        # 即门限为 a3_offset（典型 3dB）。磁滞(hysteresis)的正确作用是
        # 「去抖动/防乒乓」，而不是抬高进入门限。
        # 曾用 thr = offset + hyst = 5dB，导致门限过严：
        # 邻区永远达不到 5dB 优势，A3 一次都不触发（实测动作分布全 0），
        # 全部切换都由 env 的应急兜底(SINR<-5dB)完成 → 基线失真。
        # 这里：进入用 offset；磁滞用于「目标变更去抖」。
        # raw_enter：L3 滞后会把「瞬时已过 +3dB」的微站藏住（口部实测：
        # raw +4dB 时滤波差仍约 -7dB）。打开后瞬时 Mn>Ms+Off 也可进入 TTT。
        thr = self.a3_offset_db
        best, best_i = -200.0, -1
        for i, r in enumerate(nbr):
            raw_r = float(raw_nbr[i]) if i < len(raw_nbr) else float(r)
            enter = (r > serving + thr and r >= min_rsrp)
            if not enter and self.raw_enter:
                enter = (raw_r > raw_serving + thr and raw_r >= min_rsrp)
            if not enter:
                continue
            # 磁滞去抖：新目标需比当前计时目标好 hyst_db 才允许改换
            # （按小区比较，秩位变化不算目标变更）
            if self._ttt_cell is not None and nbr_keys[i] != self._ttt_cell:
                cur = self._filt.get(self._ttt_cell, -200.0)
                if r < cur + self.hyst_db:
                    continue
            if r > best:
                best, best_i = r, i

        # 无满足门限的邻区 → 复位定时器，stay
        if best_i < 0:
            self._ttt_elapsed_ms = 0.0
            self._ttt_cell = None
            return 0

        # 速度反归一化（obs[4] 是归一化速度，范围见 state 定义）
        speed_mps = float(obs[4]) * 100.0  # 归一化基准 100m/s
        ttt = self._effective_ttt_ms(speed_mps)

        # TTT 定时器：目标**小区**变更才复位重计时（秩位互换不复位）
        tgt_cell = nbr_keys[best_i]
        if self._ttt_cell != tgt_cell:
            self._ttt_cell = tgt_cell
            self._ttt_elapsed_ms = 0.0
        self._ttt_elapsed_ms += self.dt_s * 1000.0

        # TTT 未满足 → 不切（hold）
        if self._ttt_elapsed_ms < ttt:
            return 0

        # TTT 满足 → 执行切换并复位
        self._ttt_elapsed_ms = 0.0
        self._ttt_cell = None
        # best_i 是候选数组中的秩位，直接映射到对应的 HO 动作，
        # 保证「滤波选出的目标小区」与「实际执行切换的小区」一致。
        return {0: 1, 1: 2, 2: 5}[best_i]

    def reset(self):
        self._ttt_elapsed_ms = 0.0
        self._ttt_cell = None
        self._filt = {}


class GreedyBaseline:
    """每步连最强 RSRP 基站。"""

    def act(self, obs, info):
        cand = info["candidate_rsrp"]
        nbr = cand[1:4] if len(cand) > 3 else cand[1:]
        best_i = int(np.argmax(nbr)) if len(nbr) else 0
        if nbr[best_i] > cand[0]:
            return {0: 1, 1: 2, 2: 5}.get(best_i, 0)
        return 0


class FixedIntervalBaseline:
    """每 interval_steps 步切到最强邻区。"""

    def __init__(self, interval_steps: int = 10):
        self.interval = interval_steps

    def act(self, obs, info):
        if info["step_idx"] > 0 and info["step_idx"] % self.interval == 0:
            cand = info["candidate_rsrp"]
            nbr = cand[1:4] if len(cand) > 3 else cand[1:]
            if len(nbr):
                bi = int(np.argmax(nbr))
                return {0: 1, 1: 2, 2: 5}.get(bi, 0)
        return 0


class TabularQAgent:
    """粗离散 3-bin 表格 Q-learning（6 关键维）。"""

    def __init__(self, n_actions: int = 7, lr: float = 0.1, gamma: float = 0.99, eps: float = 0.1):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.Q = {}
        self._rng = np.random.default_rng(0)

    def _discretize(self, obs):
        # 6 关键维：0服务RSRP,1服务SINR,2邻1RSRP,3邻2RSRP,4速度,9切换频率
        keys = [0, 1, 2, 3, 4, 9]
        binned = []
        for k in keys:
            v = float(obs[k])
            b = 0 if v < 0.33 else (1 if v < 0.66 else 2)
            binned.append(b)
        return tuple(binned)

    def _ensure(self, s):
        if s not in self.Q:
            self.Q[s] = np.zeros(self.n_actions)
        return self.Q[s]

    def act(self, obs, info):
        s = self._discretize(obs)
        q = self._ensure(s)
        if self._rng.random() < self.eps:
            return int(self._rng.integers(self.n_actions))
        return int(np.argmax(q))

    def update(self, obs, action, reward, next_obs, done):
        s = self._discretize(obs)
        ns = self._discretize(next_obs)
        q = self._ensure(s)
        nq = self._ensure(ns)
        target = reward + (0.0 if done else self.gamma * np.max(nq))
        q[action] += self.lr * (target - q[action])


class VanillaDQN:
    """原始 DQN：单流 Q（无 V/A 分离）、均匀回放、Double-Q。Dueling 消融对照。"""

    class _SingleStream(nn.Module):
        def __init__(self, obs_dim, n_actions, hidden):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, n_actions),
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, cfg):
        a = cfg["agent"]
        self.device = torch.device("cpu")
        self.online = self._SingleStream(cfg["env"]["obs_dim"], cfg["env"]["n_actions"], a["hidden_dim"])
        self.target = self._SingleStream(cfg["env"]["obs_dim"], cfg["env"]["n_actions"], a["hidden_dim"])
        self.target.load_state_dict(self.online.state_dict())
        self.gamma = float(a["gamma"])
        self.opt = optim.Adam(self.online.parameters(), lr=float(a["lr"]))
        self.buffer = []  # 均匀回放
        self.buf_size = int(a["buffer_size"])
        self.batch = int(a["batch_size"])

    def act(self, obs, info):
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            q = self.online(x)
            return int(q.argmax(1).item())

    def store(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))
        if len(self.buffer) > self.buf_size:
            self.buffer.pop(0)

    def learn(self):
        if len(self.buffer) < self.batch:
            return 0.0
        idx = np.random.choice(len(self.buffer), self.batch, replace=False)
        batch = [self.buffer[i] for i in idx]
        s = torch.as_tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
        a = torch.as_tensor([b[1] for b in batch], dtype=torch.int64)
        r = torch.as_tensor([b[2] for b in batch], dtype=torch.float32)
        s2 = torch.as_tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
        done = torch.as_tensor([b[4] for b in batch], dtype=torch.float32)
        q_sa = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a_star = self.online(s2).argmax(1)
            q_next = self.target(s2).gather(1, a_star.unsqueeze(1)).squeeze(1)
            y = r + self.gamma * q_next * (1 - done)
        loss = ((q_sa - y) ** 2).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return float(loss.item())

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())
