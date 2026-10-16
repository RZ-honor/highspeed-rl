"""Strict HSR4 protocol smoke; mock results are not real NS-3 acceptance."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from src.agents.baselines import A3OperatorBaseline
from src.envs.ns3_env import NS3HandoverEnv
from src.utils.config import load_config


def check_sample(obs, reward, info, tunnels):
    assert np.isfinite(obs).all() and np.isfinite(reward), "nonfinite observation/reward"
    assert info["telemetry_available"], "HSR4 telemetry required"
    assert info["ho_request_result"] in range(7), "invalid request result"
    for key in ("interval_ms", "no_rx_interval_ms", "no_rx_run_ms", "rrc_ho_duration_ms"):
        assert np.isfinite(info[key]) and info[key] >= 0, key
    assert info["no_rx_interval_ms"] <= info["interval_ms"] + 0.01
    expected = any(a <= info["ue_x"] <= b for a, b in tunnels)
    assert bool(info["in_tunnel_ns3"]) == expected, "tunnel deployment mismatch"


def run_case(cfg, name, scenario, seed, dur_s=8.0, steps=400, early=False):
    cfg = copy.deepcopy(cfg)
    cfg["ns3"].pop("scenario_families", None)
    cfg["ns3"]["scenario"] = scenario
    cfg["env"]["episode_steps"] = 5 if early else steps
    cfg["ns3"]["args"] = " ".join(
        a for a in cfg["ns3"]["args"].split()
        if not a.startswith(("--dur=", "--dt="))
    ) + f" --dur={dur_s:g} --dt=0.1"
    cfg["mobility"]["dt_s"] = 0.1
    tunnels = [tuple(t) for t in scenario.get("tunnels", [])]
    env = NS3HandoverEnv(cfg)
    proc = None
    record = {"case": name, "early": early, "seed": seed, "passed": False}
    try:
        obs, info = env.reset(seed=seed)
        proc = env._proc
        assert env._state_size == 99, "HSR4 state size required"
        check_sample(obs, 0.0, info, tunnels)
        policy = A3OperatorBaseline(cfg["safety"], cfg.get("a3_operator", {}), 0.1)
        policy.reset()
        events = 0
        for step in range(cfg["env"]["episode_steps"]):
            obs, reward, term, trunc, info = env.step(int(policy.act(obs, info)))
            check_sample(obs, reward, info, tunnels)
            events += int(info.get("ho_event", False))
            assert not term, "unexpected RLF termination"
            if trunc:
                break
        if not early:
            assert trunc, "missing truncation at natural end"
        assert info["time_ms"] < 2000 if early else info["time_ms"] >= 7000
        record.update(passed=True, steps=step + 1, final_time_ms=info["time_ms"], ho_events=events)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
        if proc is not None and proc.poll() is None:
            record.update(passed=False, error="child process still running after close")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ns3.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--seed", type=int, default=900000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.mock:
        cfg["ns3"].update(binary=sys.executable, cwd=".", args="-m scripts.ns3_mock --nEnb=8 --spacing=1000 --offset=50")
    # tunnel 族参数直接机械派生自已验收的生产标定（A3 survival 1.00），
    # 不在本脚本里手抄隧道/微站数值。
    tunnel_sc = load_config("configs/eval_tunnel.yaml")["ns3"]["scenario"]
    cases = [
        ("normal", {"randomize": False, "n_bg_ue": 0}, 8.0, 400),
        ("tunnel", tunnel_sc, 52.0, 480),
        ("hotcold", {"randomize": False, "n_bg_ue": 0,
                     "cold_neighbor_boost_db": 3.0}, 8.0, 400),
    ]
    results = [run_case(cfg, name, sc, args.seed + i, dur, steps)
               for i, (name, sc, dur, steps) in enumerate(cases)]
    results.append(run_case(cfg, "early", {"randomize": False, "n_bg_ue": 0},
                            args.seed, early=True))
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
              for p in ("ns3/hsr-common.h", "ns3/hsr-rl.cc", "src/envs/ns3_env.py")}
    passed = all(r["passed"] for r in results)
    report = {"backend": "mock" if args.mock else "real", "passed": passed,
              "source_sha256": hashes, "cases": results}
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
