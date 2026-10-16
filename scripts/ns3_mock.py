"""hsr-rl 二进制的协议级 mock（Stage B 本地测试用）。

在无 NS-3 的环境（如 Windows 本机）下模拟 hsr-rl 的 stdin/stdout 二进制协议，
用于验证 NS3HandoverEnv 的锁步逻辑、obs 构造与奖励计算。

运行方式（由 NS3HandoverEnv spawn）：
  python -m scripts.ns3_mock --dur=5 --nEnb=8 --spacing=600 --offset=50 --rngRun=1

物理近似（足够驱动协议测试，非精确信道）：
  - 路损 PL = 128.1 + 37.6·log10(d/1000)（城市宏站经典公式）
  - 阴影：每 eNB 随机偏移（由 rngRun 决定，episode 内恒定）
  - 切换：动作 1/2/5 立即生效，中断 20ms 折算为下一步吞吐 ×0.5
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from src.envs.ns3_env import MAGIC, HSR_STATE_DTYPE, HSR_ACTION_DTYPE  # noqa: E402


def read_exact(n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sys.stdin.buffer.read(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return buf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nEnb", type=int, default=8)
    p.add_argument("--spacing", type=float, default=1000.0)
    p.add_argument("--offset", type=float, default=50.0)
    p.add_argument("--speed", type=float, default=83.3)
    p.add_argument("--dur", type=float, default=52.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--rngRun", type=int, default=1)
    p.add_argument("--tunnels", default="")
    # 其余真实后端参数保持可透传；mock 只模拟协议与几何，不模拟隧道 RF 物理。
    args, _unknown = p.parse_known_args()
    tunnel_segments = []
    if args.tunnels:
        tunnel_segments = [tuple(float(v) for v in part.split(":"))
                           for part in args.tunnels.split(",") if part]

    rng = np.random.default_rng(args.rngRun)
    # 每 eNB 阴影偏移 dB（episode 内恒定，跨 episode 随 rngRun 变化）
    shadow = rng.normal(0.0, 4.0, size=args.nEnb)

    enb_x = np.arange(args.nEnb) * args.spacing
    bw = 20e6
    noise_dbm = -174.0 + 10.0 * np.log10(bw) + 7.0

    def rsrp_all(ue_x):
        d = np.maximum(np.hypot(enb_x - ue_x, args.offset), 1.0)
        pl = 128.1 + 37.6 * np.log10(d / 1000.0)
        return 43.0 - pl + shadow

    def sinr_est(rsrp):
        rx = 10.0 ** (rsrp / 10.0)
        noise = 10.0 ** (noise_dbm / 10.0)
        total = rx.sum()
        return 10.0 * np.log10(np.maximum(rx / (noise + np.maximum(total - rx, 0.0)), 1e-12))

    def shannon(sinr_db):
        return bw * np.log2(1.0 + 10.0 ** (sinr_db / 10.0)) / 1e6

    out = sys.stdout.buffer
    out.write(MAGIC)
    out.flush()

    serving = 0  # eNB 索引（cellId = idx + 1）
    ho_this_interval = False
    request_result = 0
    step = 0
    t = 0.3
    last_sample_time = 0.1  # 与业务开始时刻对齐
    while True:
        final = t >= args.dur - args.dt
        if final:
            t = args.dur
        ue_x = args.speed * t
        rsrp = rsrp_all(ue_x)
        order = np.argsort(-rsrp)
        nbrs = [int(i) for i in order if i != serving][:3]

        # 吞吐：服务链路 Shannon（若上一步刚切换，×0.5 近似中断）
        cap = shannon(sinr_est(rsrp)[serving])
        if ho_this_interval:
            cap *= 0.5

        s = np.zeros(1, dtype=HSR_STATE_DTYPE)[0]
        s["step"] = step
        s["time_ms"] = int(t * 1000)
        s["serving_cell"] = serving + 1
        s["serving_rsrp_dbm"] = int(rsrp[serving])
        for i, n in enumerate(nbrs):
            s["nbr_cell"][i] = n + 1
            s["nbr_rsrp_dbm"][i] = int(rsrp[n])
        s["n_nbr"] = len(nbrs)
        s["speed_mps"] = args.speed
        dx = enb_x[serving] - ue_x
        dist = float(np.hypot(dx, args.offset))
        s["heading_cos"] = dx / max(dist, 1e-6)
        s["dist_to_serving_m"] = dist
        s["ue_x"] = ue_x
        s["in_tunnel"] = int(any(a <= ue_x <= b for a, b in tunnel_segments))
        s["throughput_mbps"] = cap
        # 容量字段：mock 只有控车一个 UE，故聚合=单用户、公平性=1.0
        s["n_ue_cell"] = 1
        s["n_bg_total"] = 0
        s["tp_agg_mbps"] = cap
        s["ctrl_delay_ms"] = 12.0
        s["tp_agg_cell_mbps"] = cap
        s["tp_ue_p05_mbps"] = cap
        s["tp_ue_p50_mbps"] = cap
        s["jain"] = 1.0
        s["ho_interrupt_ms"] = 20 if ho_this_interval else 0
        s["ho_event"] = 1 if ho_this_interval else 0
        s["ho_request_result"] = request_result
        s["interval_ms"] = (t - last_sample_time) * 1000
        last_sample_time = t
        s["done"] = int(final)
        out.write(s.tobytes())
        out.flush()

        if final:
            return

        # 读动作（EOF → Python 结束 episode）
        try:
            act = np.frombuffer(read_exact(8), dtype=HSR_ACTION_DTYPE)[0]
        except EOFError:
            return
        action = int(act["action"])

        ho_this_interval = False
        request_result = 0 if action == 0 else 7
        if action in (1, 2, 5):
            idx = {1: 0, 2: 1, 5: 2}[action]
            request_result = 2
            if idx < len(nbrs):
                request_result = 1
                serving = nbrs[idx]
                ho_this_interval = True

        step += 1
        t += args.dt



if __name__ == "__main__":
    main()
