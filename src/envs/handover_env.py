import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _norm(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


class HandoverEnv(gym.Env):
    """高铁宏切换环境：10 维观测、7 动作、多链路奖励，RLF 终止。

    动作含义：0 stay; 1 HO->nbr1; 2 HO->nbr2; 3 dual(+nbr1);
    4 dual(+nbr2); 5 HO->nbr3; 6 triple(+nbr1+nbr2)
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg, channel, mobility):
        self.cfg = cfg
        self.env_cfg = cfg["env"]
        self.ch = channel
        self.mo = mobility
        self.episode_steps = int(self.env_cfg["episode_steps"])
        self.obs_dim = int(self.env_cfg["obs_dim"])
        self.n_actions = int(self.env_cfg["n_actions"])
        self.action_space = spaces.Discrete(self.n_actions)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(0)
        self._step = 0
        self._t = 0.0
        self._dt = float(cfg["mobility"]["dt_s"])
        self._bs = None
        self._serving = 0
        self._active = None  # set of bs idx (active links)
        self._last_ho_step = -10_000
        self._ho_history = []
        self._rlf_count = 0
        self._buf_kb = 0.0
        # 切换进度 τ 建模：切换后进入中断窗口持续 _ho_tau_steps 步
        self._ho_tau = 0.0          # 当前切换进度 [0,1]（窗口内为步中点）
        self._ho_tau_steps = 0      # 中断窗口总步数 K
        self._ho_tau_k = 0          # 窗口内已走过的步数 j
        self._ho_frac = 0.0         # 本步被中断覆盖的时间占比 ∈(0,1]
        self._ho_amplitude = float(self.env_cfg.get("ho_amplitude", 0.8))  # A_HO
        self._last_congestion = 0.0  # 服务小区拥塞度（多用户负载场景）
        self._last_n_ue = 0          # 服务小区并发 UE 数
        self._last_n1_congestion = 0.0  # 最强邻区拥塞度（负载感知切换的判据）
        self._fb_count = 0           # 应急兜底条件的连续满足步数（L3 滤波）
        # 观测是否携带 2 维负载特征（obs[14]=服务小区拥塞度、
        # obs[15]=最强邻区与服务的拥塞度差）。NS-3 后端无负载建模，置 false。
        self.obs_load = bool(self.env_cfg.get("obs_load_features", False))
        if self.obs_load:
            assert self.obs_dim == 16, \
                f"obs_load_features=true 要求 obs_dim=16，当前 {self.obs_dim}"
        else:
            assert self.obs_dim == 14, \
                f"obs_load_features=false 要求 obs_dim=14，当前 {self.obs_dim}"

    # ---- 拓扑内邻居排序 ----
    def _neighbors(self, rsrp):
        order = np.argsort(-rsrp)
        nbrs = [int(i) for i in order if i != self._serving]
        return nbrs[:3]

    def _build_obs(self, rsrp, sinr, nbrs=None):
        if nbrs is None:
            nbrs = self._neighbors(rsrp)
        n1 = nbrs[0] if len(nbrs) > 0 else self._serving
        n2 = nbrs[1] if len(nbrs) > 1 else n1
        speed = float(self.cfg["mobility"]["speed_mps"])
        # 方向余弦：运动沿 x，到服务站向量
        ue_x = float(self.mo.position(self._t)[0])
        s = self._bs[self._serving]
        d_vec = np.array([s[0] - ue_x, s[1]])
        norm = float(np.hypot(*d_vec)) + 1e-6
        cos_dir = float(d_vec[0] / norm)
        dist2d = float(np.hypot(*d_vec))
        ho_freq = len([h for h in self._ho_history if self._step - h < 60]) / 10.0

        # 新增特征：RSRP 趋势（未来 ~1s 即 10 步的变化量近似）
        # 用当前位置和前进 Δx=10·v·dt 处的 RSRP 差值近似斜率
        lookahead_t = self._t + 10.0 * self._dt
        ue_future = self.mo.position(lookahead_t)
        # 趋势测量必须只读：measure() 会推进阴影 AR(1)/LOS 状态并消耗 RNG，
        # 不还原会污染下一步真实位置测量的空间相关性（位置在未来↔当下间来回跳）。
        _snap = self.ch.snapshot_state() if hasattr(self.ch, "snapshot_state") else None
        rsrp_future, _, _ = self.ch.measure(ue_future, self._bs, lookahead_t)
        if _snap is not None:
            self.ch.restore_state(_snap)
        srv_trend = rsrp_future[self._serving] - rsrp[self._serving]  # dB/1s 近似
        n1_trend = rsrp_future[n1] - rsrp[n1] if len(nbrs) > 0 else 0.0

        # RSRP 差值（A3 核心判据）
        rsrp_diff = rsrp[n1] - rsrp[self._serving] if len(nbrs) > 0 else -50.0

        # 多链路增益潜力：双连通额外吞吐的对数差
        cap_serving = self._shannon(sinr[self._serving])
        cap_n1 = self._shannon(sinr[n1]) if len(nbrs) > 0 else 0.0
        dual_gain = float(np.log1p(cap_serving + cap_n1) - np.log1p(max(cap_serving, 1e-6)))

        base = [
            _norm(rsrp[self._serving], -140, -44),  # 0: 服务 RSRP
            _norm(sinr[self._serving], -30, 30),    # 1: 服务 SINR
            _norm(rsrp[n1], -140, -44),              # 2: 最强邻 RSRP
            _norm(rsrp[n2], -140, -44),              # 3: 次强邻 RSRP
            _norm(speed, 0, 100),                    # 4: 速度
            (cos_dir + 1) / 2,                       # 5: 方向余弦
            _norm(dist2d, 0, 2000),                  # 6: 到服务 BS 距离
            _norm(self._buf_kb, 0, 1000),            # 7: 发送缓冲
            1.0 if len(nbrs) > 0 else 0.0,          # 8: 邻区可用
            _norm(ho_freq, 0, 1),                    # 9: 切换频率
            _norm(srv_trend, -30, 30),               # 10: 服务 RSRP 趋势
            _norm(n1_trend, -30, 30),                # 11: 最强邻 RSRP 趋势
            _norm(rsrp_diff, -50, 50),               # 12: RSRP 差值
            _norm(dual_gain, 0, 5),                  # 13: 双连通增益潜力
        ]
        if self.obs_load:
            # 负载感知切换的两个判据。
            # obs[15] 用「邻区拥塞度 − 服务拥塞度」而非绝对值：决策需要的是
            # 比较量，让网络不必自己学减法（与 obs[12] RSRP 差值同理）。
            base += [
                _norm(self._last_congestion, 0.0, 1.0),   # 14: 服务小区拥塞度
                (self._last_n1_congestion - self._last_congestion + 1.0) / 2.0,  # 15: 拥塞度差
            ]
        obs = np.array(base, dtype=np.float32)
        return obs, nbrs

    def _info(self, rsrp, sinr, nbrs):
        cand = np.array([rsrp[self._serving]] + [rsrp[n] for n in nbrs], dtype=np.float64)
        return {
            "candidate_rsrp": cand,
            "serving_idx": int(self._serving),
            "serving_rsrp": float(rsrp[self._serving]),
            "serving_sinr": float(sinr[self._serving]),
            "neighbors": list(nbrs),
            "step_idx": int(self._step),
            "last_ho_step": int(self._last_ho_step),
            "active_links": list(self._active),
            "pingpong": False,  # step 中覆写
            "congestion": float(self._last_congestion),
            "n1_congestion": float(self._last_n1_congestion),
            "n_ue_serving": int(self._last_n_ue),
        }

    # ---- 多用户负载 ----
    def _cell_ue_count(self, idx, load_cfg) -> int:
        """小区 idx 的并发活跃 UE 数（确定性伪随机基线 + 潮汐慢变抖动）。

        现网中靠近车站/隧道口的小区更拥塞；用小区索引确定性生成以保证可复现，
        叠加随时间的慢变抖动模拟整列车通过造成的潮汐负载。
        """
        n_ue_base = int(load_cfg.get("n_ue_per_cell", 0))
        vary = float(load_cfg.get("load_variation", 0.0))
        r = float(((idx + 1) * 2654435761) % 1000) / 1000.0  # [0,1)
        base = n_ue_base * (1.0 - vary + 2.0 * vary * r)
        tide = 1.0 + 0.15 * float(np.sin(self._t * 0.3 + idx))
        return max(1, int(base * tide))

    def _cell_congestion(self, idx, load_cfg) -> float:
        """小区拥塞度：把 UE 数线性归一化到 [0,1]。

        两个被否决的定义（都缺乏判别力，作为观测特征等价于常数）：
          * ``1 − 公平份额/容量`` = ``1 − 1/n_ue``：n_ue=30~90 时取值
            0.967~0.989，几乎不动；
          * ``n_ue × 单UE需求 / 小区容量``：在 n_ue ≥ 75 时饱和到 1.0，
            丢掉负载最重的那一段分辨率。

        按负载的取值区间线性归一化，既与可获得的公平份额（150/n_ue）
        单调对应，又在整个区间保持满分辨率。
        """
        n_base = int(load_cfg.get("n_ue_per_cell", 0))
        vary = float(load_cfg.get("load_variation", 0.0))
        n_min = max(1.0, n_base * (1.0 - vary))
        n_max = max(n_min + 1.0, n_base * (1.0 + vary))
        n_ue = self._cell_ue_count(idx, load_cfg)
        return float(np.clip((n_ue - n_min) / (n_max - n_min), 0.0, 1.0))

    def _update_load_info(self, nbrs):
        """在动作执行**之前**更新负载特征，供本步观测使用。

        必须在 _apply_action 之前调用：观测要反映智能体决策时所看到的状态，
        而不是动作之后（服务小区可能已改变）的状态。
        """
        self._last_congestion = 0.0
        self._last_n1_congestion = 0.0
        self._last_n_ue = 0
        load_cfg = self.cfg.get("load", {}) or {}
        if not load_cfg.get("enabled", False):
            return
        if int(load_cfg.get("n_ue_per_cell", 0)) <= 0:
            return
        self._last_n_ue = self._cell_ue_count(self._serving, load_cfg)
        self._last_congestion = self._cell_congestion(self._serving, load_cfg)
        if len(nbrs) > 0:
            self._last_n1_congestion = self._cell_congestion(nbrs[0], load_cfg)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self.ch._rng = np.random.default_rng(seed + 1)
        self.ch.reset()
        self._bs = self.mo.build_topology()
        self._step = 0
        self._t = 0.0
        self._serving = 0
        self._active = {0}
        self._last_ho_step = -10_000
        self._ho_history = []
        self._rlf_count = 0
        self._fb_count = 0
        self._buf_kb = float(self._rng.uniform(100, 800))
        self._ho_tau = 0.0
        self._ho_tau_steps = 0
        self._ho_tau_k = 0
        self._ho_frac = 0.0
        rsrp, sinr, _ = self.ch.measure(self.mo.position(self._t), self._bs, self._t)
        nbrs = self._neighbors(rsrp)
        self._update_load_info(nbrs)
        obs, nbrs = self._build_obs(rsrp, sinr, nbrs)
        return obs, self._info(rsrp, sinr, nbrs)

    def _shannon(self, sinr_db):
        s = 10.0 ** (sinr_db / 10.0)
        bw = float(self.cfg["channel"]["bw_hz"])
        return float(bw * np.log2(1.0 + s)) / 1e6  # Mbps

    def _apply_action(self, action, rsrp, nbrs):
        """返回 (是否发生切换, 中断ms)。"""
        n1 = nbrs[0] if len(nbrs) > 0 else self._serving
        n2 = nbrs[1] if len(nbrs) > 1 else n1
        n3 = nbrs[2] if len(nbrs) > 2 else n1
        ho = False
        interrupt = 0.0
        if action == 0:
            pass
        elif action in (1, 2, 5):
            tgt = {1: n1, 2: n2, 5: n3}[action]
            if tgt != self._serving:
                self._serving = tgt
                self._active = {tgt}
                ho = True
                interrupt = float(self._rng.uniform(
                    self.env_cfg["ho_interruption_ms_min"], self.env_cfg["ho_interruption_ms_max"]))
        elif action in (3, 4):
            add = n1 if action == 3 else n2
            self._active = {self._serving, add}
            interrupt = float(self.env_cfg["dual_setup_ms"])
        elif action == 6:
            self._active = {self._serving, n1, n2}
            interrupt = float(self.env_cfg["dual_setup_ms"])
        if ho:
            self._last_ho_step = self._step
            self._ho_history.append(self._step)
        return ho, interrupt

    def _pingpong(self):
        """仅在乒乓**事件发生的那一步**返回 True。

        原实现用 ``(self._step - last) <= 2``，一次乒乓会在随后连续 3 步都
        返回 True，造成两个后果：
          1. pingpong 指标统计的是「处于乒乓状态的步数」而非「乒乓次数」，
             被放大约 3 倍（实测 A3 的 ho=7.3 而 pingpong=7.0，看似每次切换
             都是乒乓，实为统计口径错误）；
          2. 每步重复扣 pingpong_penalty(5.0)，惩罚量取决于采样时刻而非事件数。
        改为事件语义：仅在构成乒乓的第二次切换发生那一步为真。
        """
        if len(self._ho_history) < 2:
            return False
        last = self._ho_history[-1]
        prev = self._ho_history[-2]
        return self._step == last and (last - prev) <= 2

    def step(self, action):
        self._step += 1
        self._t += self._dt
        ue = self.mo.position(self._t)
        rsrp, sinr, _ = self.ch.measure(ue, self._bs, self._t)

        # 隧道势阱：UE 在隧道内时，各 BS 的 RSRP 额外衰减
        tunnel_loss = self.mo.in_tunnel(float(ue[0]))
        if tunnel_loss > 0.0:
            rsrp = rsrp - tunnel_loss  # dBm 减去额外穿透损耗
            # 重算 SINR（RSRP 变化影响干扰项）
            noise_dbm = -174.0 + 10.0 * np.log10(float(self.cfg["channel"]["bw_hz"])) + float(self.cfg["channel"]["noise_figure_db"])
            noise_lin = 10.0 ** (noise_dbm / 10.0) / 1000.0
            rx_lin = 10.0 ** (rsrp / 10.0) / 1000.0
            k = int(self.cfg["channel"].get("n_interferers", 0) or 0)
            n_bs = len(rsrp)
            for i in range(n_bs):
                if k > 0 and n_bs - 1 > k:
                    others = np.concatenate([rx_lin[:i], rx_lin[i + 1:]])
                    interf = float(np.sum(np.sort(others)[-k:]))
                else:
                    interf = float(np.sum(rx_lin) - rx_lin[i])
                sinr[i] = 10.0 * np.log10(max(rx_lin[i] / (noise_lin + max(interf, 0.0)), 1e-12))
        nbrs = self._neighbors(rsrp)
        self._update_load_info(nbrs)     # 必须在动作前：观测反映决策时所见状态
        obs, nbrs = self._build_obs(rsrp, sinr, nbrs)

        # A3 兜底下沉到 env.step 内部：measure 后、apply_action 前
        # 消除时序差（原来在脚本层有一步延迟，300km/h 下致命）
        action_source = "rl"
        from src.agents.safety import HO_ACTIONS, DUAL_ACTIONS

        # 双连通动作掩码（P1-2）：LENA(LTE) 后端无 EN-DC，动作 3/4/6 在物理层
        # 是空操作；Python 侧若不一起掩码，就会形成两个漏洞：
        #   1. 吞吐被两条链路的 Shannon 容量直接相加，凭空翻倍；
        #   2. _apply_action 不更新 _last_ho_step，_active 一旦变成双元素就
        #      只有 HO 动作才会重置 → 每步白拿 stability_bonus（600 步累积 +30），
        #      而一次 HO 才 −0.9。这是明确的奖励漏洞，且与 ns3_env 的处理冲突。
        if self.cfg.get("safety", {}).get("mask_dual", False) and action in DUAL_ACTIONS:
            action = 0
            action_source = "masked_dual"

        # 预防性回落：服务小区 RSRP 跌破预警门限（尚未 RLF）且邻区明显更优
        # 对标现网「切换准备门限早于切换执行门限」的分级门限设计，
        # 避免 RL 因过度保守（少切换）而在隧道/遮挡段拖到 RLF。
        pre_rsrp = float(self.env_cfg.get("preempt_rsrp_dbm", -105.0))
        pre_margin = float(self.env_cfg.get("preempt_margin_db", 4.0))
        serving_rsrp_now = float(rsrp[self._serving])
        need_ho = action not in HO_ACTIONS and action not in DUAL_ACTIONS
        if need_ho and serving_rsrp_now < pre_rsrp:
            best_n = None
            for i, nidx in enumerate(nbrs):
                if rsrp[nidx] > serving_rsrp_now + pre_margin:
                    best_n = i
                    break
            if best_n is not None:
                action = {0: 1, 1: 2, 2: 5}[best_n]
                action_source = "safety_preempt"

        # RLF 兜底：SINR 跌破临界且未发起合法切换 → A3 回落。
        # 关键修正：兜底需「持续」满足若干步（对标 L3 测量滤波），
        # 否则 Jakes 快衰落的单次深衰落（实测 SINR 可到 -52dB）就会触发一次切换，
        # 导致兜底每 ~1.1s 触发一次（约 40 次/episode），把策略自身的决策完全架空，
        # 也让基于 RSRP 的 A3 事件根本来不及成立。
        fb_persist = int(self.env_cfg.get("fallback_persist_steps", 3))
        if sinr[self._serving] < float(self.env_cfg.get("rlf_fallback_sinr_db", -5.0)):
            self._fb_count += 1
        else:
            self._fb_count = 0
        if self._fb_count >= fb_persist:
            if action not in HO_ACTIONS and action not in DUAL_ACTIONS:
                # A3 回落：切到最强合法邻区
                a3_action = self._a3_action(rsrp, nbrs)
                if a3_action != 0:
                    action = a3_action
                    action_source = "safety_a3"
                    self._fb_count = 0

        ho, interrupt_ms = self._apply_action(action, rsrp, nbrs)

        # 切换进度 τ 建模：若刚发生切换，进入中断窗口
        if ho and interrupt_ms > 0:
            self._ho_tau_steps = max(1, int(np.ceil(interrupt_ms / 1000.0 / self._dt)))
            self._ho_tau_k = 0
            # 本步被中断覆盖的时间占比。中断 12-40ms << 步长 100ms ⇒ K 恒为 1，
            # 若再假定「整步全损」会把 30ms 的中断记成整步吞吐归零（过罚）。
            self._ho_frac = min(1.0, (interrupt_ms / 1000.0) / self._dt)
            self._ho_tau = 0.0

        # 推进 τ 并计算衰减因子 g(τ) = 1 - A_HO·frac·sin(πτ)
        #
        # τ 取**步中点** (j+0.5)/K ∈ (0,1)，而非 j/K：
        # 原实现 τ 从 1/K 起步，在本场景下 K 恒为 1（12-40ms 中断 vs 100ms
        # 步长），τ 一步就走到 1，sin(π·1)=0 ⇒ g≡1，中断窗在真实参数下完全
        # 失效（等效于切换零代价）。取中点后 K=1 ⇒ τ=0.5 ⇒ sin=1，衰减幅度
        # 由 duty-cycle frac 决定，物理含义明确：40ms 中断 → 本步吞吐损失
        # A_HO×0.4 = 32%。
        ho_decay = 1.0
        if self._ho_tau_steps > 0:
            self._ho_tau_k += 1
            tau = (self._ho_tau_k - 0.5) / float(self._ho_tau_steps)
            self._ho_tau = tau
            ho_decay = max(0.0, 1.0 - self._ho_amplitude * self._ho_frac * np.sin(np.pi * tau))
            if self._ho_tau_k >= self._ho_tau_steps:
                # 窗口结束：清空计数，但保留本步用过的 τ 供 info 观测
                # （下一步若无新窗口再归零）。
                self._ho_tau_steps = 0
                self._ho_tau_k = 0
                self._ho_frac = 0.0
        else:
            self._ho_tau = 0.0

        # 吞吐：各活跃链路 Shannon 求和，切换衰减 g(τ) 折算
        cap = 0.0
        for idx in self._active:
            cap += self._shannon(sinr[idx])
        cap *= ho_decay

        # ---- 多用户负载建模（整列车数百 UE 共享基站资源）----
        # 物理：单小区吞吐上限 = 频谱效率 × RB 总数，被同时在线的 UE 均分。
        # 高铁典型：复兴号 16 编组定员 ~1200 人，单小区常驻 200-600 活跃 UE。
        # 结果：单用户吞吐 = min(Shannon链路容量, 小区剩余资源/N_active)
        # 这使切换决策影响「驻留哪个小区」→ 该小区越拥塞，单用户分得越少，
        # 因此高负载下「提前切到轻载小区」的价值远高于空载场景。
        load_cfg = self.cfg.get("load", {}) or {}
        if load_cfg.get("enabled", False) and int(load_cfg.get("n_ue_per_cell", 0)) > 0:
            # 小区总容量（Mbps）：100RB @20MHz 下行峰值约 150Mbps(2x2 MIMO)
            cell_capacity = float(load_cfg.get("cell_capacity_mbps", 150.0))
            demand = float(load_cfg.get("per_ue_demand_mbps", 0.5))
            burst = float(load_cfg.get("burst_factor", 2.0))

            # 各活跃链路所在小区的公平份额（本 UE 主要从最好的那条获得服务）
            shares = [cell_capacity / float(self._cell_ue_count(idx, load_cfg))
                      for idx in self._active]
            fair_share = max(shares) if shares else cell_capacity
            # 实际吞吐受限于：链路容量、小区公平份额、业务需求上限
            cap = min(cap, fair_share, demand * burst)

        # 动态缓冲更新：业务到达 - 已发吞吐
        buf_arrival_kbps = float(self.env_cfg.get("buf_arrival_kbps", 5000.0))
        self._buf_kb = max(0.0, self._buf_kb + buf_arrival_kbps * self._dt / 1000.0 - cap * self._dt)
        r_throughput = float(np.log1p(cap))

        r_ho = 0.0
        if ho:
            r_ho -= float(self.env_cfg["ho_penalty"])
        if self._pingpong():
            r_ho -= float(self.env_cfg["pingpong_penalty"])

        r_delay = -interrupt_ms / 100.0
        r_stab = float(self.env_cfg["stability_bonus"]) if len(self._active) > 1 else 0.0

        # 势能 shaping：接近隧道入口时给负奖励，引导提前决策（演进方案 §4.3）
        r_potential = 0.0
        pot_weight = float(self.env_cfg.get("potential_shaping_weight", 0.0))
        if pot_weight > 0.0 and len(self.mo.tunnels) > 0:
            ux = float(ue[0])
            for x_start, x_end, _ in self.mo.tunnels:
                dist_to_entry = x_start - ux  # 正=还没到入口
                if 0 < dist_to_entry < 500:  # 入口前 500m 内
                    r_potential = -pot_weight * (1.0 - dist_to_entry / 500.0)

        # 弱覆盖惩罚：服务 SINR 跌破 RLF 门限前的「预警区」就给持续负奖励。
        # 目的：纠正 RL 过度保守（为省切换而赖在劣化小区 → outage 回升）。
        # 对标现网 KPI：弱覆盖占比是高铁专网核心考核指标。
        r_outage = 0.0
        warn_sinr = float(self.env_cfg.get("weak_coverage_sinr_db", -2.0))
        out_w = float(self.env_cfg.get("w_outage", 0.0))
        if out_w > 0.0:
            s_sinr = float(sinr[self._serving])
            if s_sinr < float(self.env_cfg["rlf_sinr_db"]):
                r_outage = -out_w * 2.0   # 已进入 RLF，重罚
            elif s_sinr < warn_sinr:
                # 线性预警：越接近 RLF 惩罚越大
                span = max(warn_sinr - float(self.env_cfg["rlf_sinr_db"]), 1e-6)
                r_outage = -out_w * (warn_sinr - s_sinr) / span

        reward = (float(self.env_cfg["w_throughput"]) * r_throughput
                  + float(self.env_cfg["w_handover"]) * r_ho
                  + float(self.env_cfg["w_delay"]) * r_delay
                  + float(self.env_cfg["w_stability"]) * r_stab
                  + r_potential
                  + r_outage)

        terminated = False
        if sinr[self._serving] < float(self.env_cfg["rlf_sinr_db"]):
            self._rlf_count += 1
        else:
            self._rlf_count = 0
        if self._rlf_count >= int(self.env_cfg["rlf_consecutive_steps"]):
            terminated = True
            # RLF 后用户实际断链到 episode 结束（重建需数秒）。
            # 惩罚必须按「剩余步数」计，而非固定 -10：
            # 否则 agent 会发现「永不切换」可规避 ho_penalty(3.0×40=120)，
            # 而只需付一次 -10 → 学到退化策略（ho≈1，survival 16%）。
            # 这里让断链代价与评估口径一致：剩余步每步罚 rlf_drop_penalty。
            remaining = max(0, int(self.episode_steps) - int(self._step))
            reward -= float(self.env_cfg.get("rlf_drop_penalty", 1.0)) * remaining
            reward -= float(self.env_cfg.get("rlf_termination_penalty", 10.0))
        truncated = self._step >= self.episode_steps

        info = self._info(rsrp, sinr, nbrs)
        info["throughput_mbps"] = cap
        info["ho_event"] = bool(ho)
        info["pingpong"] = bool(self._pingpong())
        info["action_source"] = action_source
        info["ho_tau"] = float(self._ho_tau)
        # 切换中断窗衰减因子（g(τ)）与 duty-cycle：用于诊断「切换代价是否真的
        # 进入了奖励」——原实现在 12-40ms 中断（K≡1）下恒为 1.0，等于零代价。
        info["ho_decay"] = float(ho_decay)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def _a3_action(self, rsrp, nbrs) -> int:
        """A3 回落动作：切到最强合法邻区；无可行则 stay。"""
        min_rsrp = float(self.cfg["safety"]["min_target_rsrp_dbm"])
        serving_rsrp = rsrp[self._serving]
        best, best_i = -200.0, -1
        for i, nidx in enumerate(nbrs):
            r = rsrp[nidx]
            if r > serving_rsrp + 3.0 and r >= min_rsrp:
                if r > best:
                    best, best_i = r, i
        if best_i < 0:
            return 0
        return {0: 1, 1: 2, 2: 5}[best_i]
