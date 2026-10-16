import numpy as np


class Mobility:
    """线性匀速轨迹 + 沿轨基站拓扑 + 可选隧道段。"""

    def __init__(self, cfg: dict, seed: int = 0):
        m = cfg["mobility"]
        self.v = float(m["speed_mps"])
        self.dt = float(m["dt_s"])
        self.spacing_min = float(m["bs_spacing_min_m"])
        self.spacing_max = float(m["bs_spacing_max_m"])
        self.offset_min = float(m["bs_offset_min_m"])
        self.offset_max = float(m["bs_offset_max_m"])
        self.n_bs = int(m["n_bs"])
        self.h_bs = float(cfg["channel"]["bs_height_m"])
        self._rng = np.random.default_rng(seed)
        self._bs = None
        # 隧道段：[(x_start, x_end, extra_loss_db), ...]
        t_cfg = m.get("tunnels", [])
        self.tunnels = [(float(t["x_start"]), float(t["x_end"]), float(t.get("extra_loss_db", 25.0)))
                        for t in t_cfg]

    def build_topology(self) -> np.ndarray:
        x = 0.0
        bs = []
        # 首站靠近起点：否则轨迹起点前是覆盖盲区，episode 一开始就 RLF
        x += float(self._rng.uniform(100.0, 300.0))
        for _ in range(self.n_bs):
            y = float(self._rng.uniform(self.offset_min, self.offset_max))
            bs.append([x, y, self.h_bs])
            x += float(self._rng.uniform(self.spacing_min, self.spacing_max))
        self._bs = np.array(bs, dtype=np.float64)
        return self._bs

    def position(self, t: float) -> np.ndarray:
        return np.array([self.v * t, 0.0, 1.5], dtype=np.float64)

    def in_tunnel(self, x: float) -> float:
        """返回 UE 在位置 x 处的隧道额外损耗 (dB)；不在隧道返回 0。"""
        for x_start, x_end, loss_db in self.tunnels:
            if x_start <= x <= x_end:
                return loss_db
        return 0.0
