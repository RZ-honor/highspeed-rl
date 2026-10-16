"""3GPP TR 38.901 RMa channel model for high-speed rail simulations.

Provides path loss, shadow fading, fast fading, penetration loss, and SINR
computation for Rural Macro (RMa) scenarios at 350 km/h.
"""

import numpy as np

C_LIGHT = 3.0e8


class Channel3GPP:
    """3GPP TR 38.901 RMa channel model.

    Computes per-BS RSRP (dBm), SINR (dB), and LOS indicator for a UE at a
    given position and time. Supports spatial-correlated shadow fading via
    a first-order autoregressive process along the track and Jakes fast fading
    with N-ray Doppler spread.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary. Must contain a ``channel`` key with all
        sub-fields defined in ``configs/default.yaml``. Optionally contains
        a top-level ``seed`` key for reproducibility.
    """

    def __init__(self, cfg: dict):
        c = cfg["channel"]
        self.fc = float(c["fc_hz"])                     # Carrier frequency [Hz]
        self.bw = float(c["bw_hz"])                     # Bandwidth [Hz]
        self.ptx_dbm = float(c["bs_tx_power_dbm"])      # BS TX power [dBm]
        self.h_bs = float(c["bs_height_m"])              # BS antenna height [m]
        self.h_ut = float(c["ue_height_m"])              # UE antenna height [m]
        self.nf = float(c["noise_figure_db"])            # Noise figure [dB]
        self.W = float(c["street_width_w"])              # Street width [m]
        self.h_bld = float(c["building_height_h"])       # Building height [m]
        self.sf_los = float(c["shadow_sigma_los"])       # LOS shadow sigma [dB]
        self.sf_nlos = float(c["shadow_sigma_nlos"])     # NLOS shadow sigma [dB]
        self.dc_los = float(c["d_corr_los"])             # LOS decorrelation dist [m]
        self.dc_nlos = float(c["d_corr_nlos"])           # NLOS decorrelation dist [m]
        self.N_rays = int(c["jakes_n_rays"])             # Jakes N rays
        self.L_pen = float(c["penetration_db"])          # Penetration loss [dB]
        # 天线增益（dBi）。3GPP TR 38.901 的 RMa 路损模型给出的是**路径损耗**，
        # 天线增益是链路预算里独立的一项。此前完全未建模 → 系统性地低估接收
        # 电平约 18dB：500m 处 RSRP ≈ -92dBm（实测服务 SINR 均值仅 3.8dB、
        # 46% 的步处于弱覆盖），这样的网络在现实中处于瘫痪状态，没有任何运营商
        # 会这样部署。高铁专网沿轨使用高增益定向面板（15~21dBi），
        # 车顶天线 + 车厢内分布式系统再贡献数 dB。
        self.G_bs = float(c.get("bs_antenna_gain_dbi", 0.0))
        self.G_ue = float(c.get("ue_antenna_gain_dbi", 0.0))
        # 定向天线可见干扰源数：高铁沿轨定向覆盖，列车实际只见邻近 2~4 个小区。
        # 0 表示退化为传统 reuse-1（全部其余小区均计入干扰）。
        self.n_interferers = int(c.get("n_interferers", 0) or 0)
        self._seed = int(cfg.get("seed", 42))

        # ------------------------------------------------------------------
        # Derived quantities
        # ------------------------------------------------------------------
        # Effective UE height (3GPP RMa: h = h_UT - 1.0 m)
        self.h_eff = self.h_ut - 1.0

        # Break-point distance (3GPP TR 38.901 Table 7.4.1-1) — uses fc in Hz
        self.d_bp = 2.0 * np.pi * self.h_bs * self.h_ut * self.fc / C_LIGHT

        # Carrier frequency in GHz (used in path loss formulas per 3GPP standard)
        self.fc_ghz = self.fc / 1e9

        # ------------------------------------------------------------------
        # Jakes fading —— 到达角共享，但**每条链路（每个 BS）必须有独立的
        # 随机相位**。历史 bug：只生成一组相位并被所有 BS 共用，导致所有
        # 链路乘上同一个瑞利包络 → 快衰落成为共模分量，在 RSRP 差值
        # （A3 的核心判据）中完全抵消，"快衰落导致最强邻区频繁翻转"这一
        # 现象根本不会发生。修正后每条链路独立衰落，邻区排序才会真正抖动。
        # ------------------------------------------------------------------
        rng = np.random.default_rng(self._seed)
        n = np.arange(1, self.N_rays + 1)
        self._alpha = 2.0 * np.pi * n / self.N_rays  # arrival angles
        self._ff_rng = np.random.default_rng(self._seed + 2)
        self._ff_phase = {}   # bs_idx -> 该链路的随机相位向量
        # 多普勒必须用配置的列车速度：历史 bug 里硬编码 83.3(300km/h)，
        # 在 350km/h 场景下多普勒频偏低估约 17%（583Hz vs 680Hz）。
        self._speed_mps = float((cfg.get("mobility") or {}).get("speed_mps", 83.3))
        self._fd = self._speed_mps * self.fc / C_LIGHT  # Doppler shift [Hz]

        # ------------------------------------------------------------------
        # Shadow fading state (per-BS, updated on each measure call)
        # ------------------------------------------------------------------
        self._sf_state = {}   # bs_idx -> current shadow value [dB]
        self._sf_pos = {}     # bs_idx -> last x-position [m]
        # LOS 状态（空间相关，避免每步独立随机翻转造成虚假信号塌陷）
        self._los_state = {}  # bs_idx -> current LOS bool
        self._los_pos = {}    # bs_idx -> last x-position where LOS sampled [m]
        self._los_corr = float(c.get("los_corr_m", 100.0))  # LOS 空间相关距离 [m]
        self._rng = np.random.default_rng(self._seed + 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset shadow fading state (e.g., at the start of a new episode)."""
        self._sf_state = {}
        self._sf_pos = {}
        self._los_state = {}
        self._los_pos = {}

    def snapshot_state(self):
        """捕获阴影衰落/LOS 空间相关状态与 RNG 状态（见 restore_state）。"""
        import copy
        return (copy.deepcopy(self._sf_state), copy.deepcopy(self._sf_pos),
                copy.deepcopy(self._los_state), copy.deepcopy(self._los_pos),
                self._rng.bit_generator.state)

    def restore_state(self, snap) -> None:
        """恢复 snapshot_state 保存的状态。

        观测构造（趋势特征）会在「未来位置」额外调用一次 measure()，
        该调用会把阴影 AR(1)/LOS 状态推进到未来位置并消耗 RNG 抽样子，
        导致下一步真实位置的测量基于「回退位置」的状态（ρ≈0.13，
        空间相关性被破坏）。趋势测量应只读，故用快照/还原包裹。
        """
        import copy
        self._sf_state = copy.deepcopy(snap[0])
        self._sf_pos = copy.deepcopy(snap[1])
        self._los_state = copy.deepcopy(snap[2])
        self._los_pos = copy.deepcopy(snap[3])
        self._rng.bit_generator.state = copy.deepcopy(snap[4])

    def measure(
        self, ue_xyz: np.ndarray, bs_xyz: np.ndarray, t: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute RSRP, SINR, and LOS indicator for each BS.

        Parameters
        ----------
        ue_xyz : np.ndarray, shape (3,)
            UE position (x, y, z) in metres.
        bs_xyz : np.ndarray, shape (N, 3)
            BS positions (x, y, z) in metres.
        t : float
            Simulation time in seconds (used for Doppler phase in Jakes).

        Returns
        -------
        rsrp : np.ndarray, shape (N,)
            RSRP per BS in dBm.
        sinr : np.ndarray, shape (N,)
            SINR per BS in dB.
        los : np.ndarray, shape (N,), bool
            True if the BS-UE link is LOS.
        """
        ue_xy = ue_xyz[:2]
        n_bs = bs_xyz.shape[0]

        rsrp = np.zeros(n_bs, dtype=np.float64)
        sinr = np.zeros(n_bs, dtype=np.float64)
        los = np.zeros(n_bs, dtype=bool)
        rx_lin = np.zeros(n_bs, dtype=np.float64)  # linear Rx power [W]

        # Noise power in linear scale [W]
        noise_dbm = -174.0 + 10.0 * np.log10(self.bw) + self.nf
        noise_lin = 10.0 ** (noise_dbm / 10.0) / 1000.0

        for i in range(n_bs):
            # 2-D and 3-D distances
            d2d = float(np.hypot(bs_xyz[i, 0] - ue_xyz[0],
                                  bs_xyz[i, 1] - ue_xyz[1]))
            d3d = float(np.linalg.norm(bs_xyz[i] - ue_xyz))

            # LOS / NLOS determination（空间相关：位移超过相关距离才重新采样，
            # 避免每步独立随机翻转造成物理上不真实的瞬时信号塌陷）
            ue_x = float(ue_xyz[0])
            p_los = self._los_probability(d2d)
            prev_los_pos = self._los_pos.get(i)
            if prev_los_pos is None or abs(ue_x - prev_los_pos) > self._los_corr:
                is_los = self._rng.random() < p_los
                self._los_state[i] = is_los
                self._los_pos[i] = ue_x
            else:
                is_los = self._los_state[i]
            los[i] = is_los

            # Combine losses
            pl = self._path_loss(d2d, d3d, is_los)
            sf = self._shadow(i, ue_xy, is_los)
            ff = self._fast_fading_gain(i, t)
            pen = self._penetration(ue_xy, bs_xyz[i])

            loss_db = pl + sf + pen - 20.0 * np.log10(max(ff, 1e-6))

            rx_dbm = self.ptx_dbm + self.G_bs + self.G_ue - loss_db
            rx_lin[i] = 10.0 ** (rx_dbm / 10.0) / 1000.0
            rsrp[i] = rx_dbm

        # SINR：干扰项。
        # 原实现按 reuse-1 把「全部其余小区」都计入干扰（12 站 → 11 个干扰源），
        # 导致本场景变成干扰受限：SINR 常年很低，而邻区 RSRP 平均比服务低 14dB，
        # 于是 env 的应急兜底(SINR<-5dB)频繁触发，而基于 RSRP 的 A3 事件
        # （需邻区优于服务 3dB）几乎从不成立 → 基线失真、RL 也学不出东西。
        #
        # 现实高铁专网沿轨敷设定向天线（65° 波瓣，双向覆盖），
        # 列车实际只「看见」邻近 2~4 个小区，远处小区被天线方向图强烈抑制。
        # 故这里只取最强的 n_interferers 个小区作为有效干扰源。
        k = self.n_interferers
        for i in range(n_bs):
            if k > 0 and n_bs - 1 > k:
                others = np.concatenate([rx_lin[:i], rx_lin[i + 1:]])
                # 取最强 k 个干扰源（定向天线可见邻区）
                interf = float(np.sum(np.sort(others)[-k:]))
            else:
                interf = float(np.sum(rx_lin) - rx_lin[i])
            sinr_lin = rx_lin[i] / (noise_lin + max(interf, 0.0))
            sinr[i] = 10.0 * np.log10(max(sinr_lin, 1e-12))

        return rsrp, sinr, los

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _los_probability(self, d2d: float) -> float:
        """RMa LOS probability (3GPP TR 38.901 Table 7.4.2-1)."""
        if d2d <= 10.0:
            return 1.0
        if d2d > 5000.0:
            return 0.0
        return float(np.exp(-(d2d - 10.0) / 1000.0))

    def _pl_los(self, d3d: float) -> float:
        """RMa LOS path loss (3GPP TR 38.901 Table 7.4.1-1).

        Note: fc is in GHz in the path loss formula per the standard.
        """
        h = self.h_eff
        # PL1 — valid for all d3D
        pl1 = (20.0 * np.log10(40.0 * np.pi * self.fc_ghz * d3d / 3.0)
               + min(0.03 * h ** 1.72, 10.0) * np.log10(d3d)
               - min(0.044 * h ** 1.72, 14.77)
               + 0.002 * np.log10(h) * d3d)
        if d3d <= self.d_bp:
            return pl1
        # PL2 — beyond break-point
        pl2 = pl1 + 40.0 * np.log10(d3d / self.d_bp)
        return pl2

    def _pl_nlos(self, d3d: float) -> float:
        """RMa NLOS path loss (3GPP TR 38.901 Table 7.4.1-1).

        Note: fc is in GHz in the path loss formula per the standard.
        """
        hbs = self.h_bs
        hut = self.h_ut
        h = self.h_bld
        pl = (161.04
              - 7.1 * np.log10(self.W)
              + 7.5 * np.log10(h)
              - (24.37 - 3.7 * (h / hbs) ** 2) * np.log10(hbs)
              + (43.42 - 3.1 * np.log10(hbs)) * (np.log10(d3d) - 3.0)
              + 20.0 * np.log10(self.fc_ghz)
              - (3.2 * (np.log10(11.75 * hut)) ** 2 - 4.97))
        return pl

    def _path_loss(self, d2d: float, d3d: float, los: bool) -> float:
        """Total path loss: LOS path, or max(LOS, NLOS)."""
        pl_los = self._pl_los(d3d)
        if los:
            return pl_los
        return max(pl_los, self._pl_nlos(d3d))

    def _shadow(self, bs_idx: int, ue_xy: np.ndarray, los: bool) -> float:
        """Spatially-correlated shadow fading via first-order AR(1) process."""
        x = float(ue_xy[0])  # track position (assumed along x-axis)
        sigma = self.sf_los if los else self.sf_nlos
        dc = self.dc_los if los else self.dc_nlos

        prev_pos = self._sf_pos.get(bs_idx)
        if prev_pos is None:
            val = float(self._rng.normal(0.0, sigma))
        else:
            dx = abs(x - prev_pos)
            rho = float(np.exp(-dx / dc))
            # AR(1) with correlation coefficient rho
            val = (rho * self._sf_state[bs_idx]
                   + np.sqrt(max(1.0 - rho ** 2, 0.0))
                   * float(self._rng.normal(0.0, sigma)))

        self._sf_state[bs_idx] = val
        self._sf_pos[bs_idx] = x
        return val

    def _fast_fading_gain(self, bs_idx: int, t: float) -> float:
        """第 bs_idx 条链路在时刻 t 的 Jakes 瑞利衰落包络。

        每条链路使用各自独立的随机相位（散射环境不同）；到达角共享。
        """
        phi = self._ff_phase.get(bs_idx)
        if phi is None:
            phi = self._ff_rng.uniform(0.0, 2.0 * np.pi, self.N_rays)
            self._ff_phase[bs_idx] = phi
        # Complex envelope: sum of N rays
        h = (1.0 / np.sqrt(self.N_rays)) * np.sum(
            np.exp(1j * (2.0 * np.pi * self._fd * np.cos(self._alpha) * t + phi)))
        return float(np.abs(h))

    def _penetration(self, ue_xy: np.ndarray, bs_xyz: np.ndarray) -> float:
        """Penetration loss based on incidence angle.

        Uses the 3GPP TR 38.901 angular penetration model with a cap
        to prevent singularities at grazing incidence (cos_theta -> 0).
        """
        dx = bs_xyz[0] - ue_xy[0]
        dy = bs_xyz[1] - ue_xy[1]
        horiz = np.hypot(dx, dy)
        if horiz < 1e-6:
            return self.L_pen
        # cos(theta) = |dy| / horiz  (assuming train moves along x-axis,
        # windows face the track perpendicular — y direction)
        cos_theta = abs(dy) / horiz
        # Cap penetration loss at 40 dB to avoid singularities at grazing
        # incidence (e.g., when the BS is directly ahead along the track).
        # cos_theta_min = L_pen / 40 = 0.7, so cap at 0.7.
        cos_theta = max(cos_theta, self.L_pen / 40.0)
        return self.L_pen / cos_theta