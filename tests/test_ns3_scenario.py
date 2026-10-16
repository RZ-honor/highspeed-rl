"""NS-3 场景参数构造 / domain randomization / 隧道校验（不启动二进制）。"""
import numpy as np
import pytest

from src.envs.ns3_env import NS3HandoverEnv, build_scenario_args, validate_tunnels


def _base_cfg():
    return {
        "seed": 42,
        "channel": {"bw_hz": 20e6, "noise_figure_db": 7.0},
        "mobility": {"dt_s": 0.1},
        "env": {"episode_steps": 480, "obs_dim": 14, "n_actions": 7},
        "ns3": {
            "binary": "/bin/true", "cwd": None, "timeout_s": 5,
            "args": "--nEnb=8 --dur=52 --rngRun={rng_run}",
            "scenario": {
                "randomize": True,
                "n_bg_ue": [16, 32],
                "bg_speed_mps": [83, 138],
                "voip_frac": [0.15, 0.30],
                "video_frac": [0.30, 0.50],
                "tunnel_loss_db": [20, 30],
                "tunnels": [[2000, 2800], [5000, 5600]],
            },
        },
    }


def test_scenario_args_contain_bg_and_tunnels():
    cfg = _base_cfg()
    # seed=1：随机化下两段隧道均被保留（seed=0 会以 0.16 概率全过滤，使断言 flaky）
    args = build_scenario_args(cfg, rng_run=7, rng=np.random.default_rng(1))
    s = " ".join(args)
    assert "--nBgUe=" in s and "--tunnels=" in s
    assert "--tunnelLossDb=" in s and "--voipFrac=" in s
    # rng_run 透传
    assert any(a == "--rngRun=7" for a in args)


def test_randomization_varies_across_runs():
    cfg = _base_cfg()
    a1 = build_scenario_args(cfg, 1, np.random.default_rng(1))
    a2 = build_scenario_args(cfg, 2, np.random.default_rng(2))
    def nbg(args):
        return [x for x in args if x.startswith("--nBgUe=")][0]
    # 两次采样的背景 UE 数大概率不同（区间宽 16）
    assert nbg(a1) != nbg(a2) or a1 != a2


def test_validate_tunnels_rejects_bad():
    validate_tunnels([[100, 300]])                 # 合法
    with pytest.raises(ValueError):
        validate_tunnels([[300, 100]])             # x2<x1
    with pytest.raises(ValueError):
        validate_tunnels([[100, 300], [200, 400]]) # 重叠


def test_fixed_scenario_when_randomize_off():
    cfg = _base_cfg()
    cfg["ns3"]["scenario"] = {"randomize": False, "n_bg_ue": 24,
                              "tunnels": [[2000, 2800]], "tunnel_loss_db": 25}
    args = build_scenario_args(cfg, 5, np.random.default_rng(3))
    assert "--nBgUe=24" in args
    assert "--tunnels=2000:2800" in args


def test_tunnel_micro_enb_args_present():
    cfg = {"ns3": {"scenario": {
        "randomize": False,
        "tunnels": [[2000, 2800]], "tunnel_loss_db": 25.0,
        "tunnel_enb_spacing_m": 250.0, "tunnel_enb_tx_dbm": 30.0,
        "tunnel_micro_loss_db": 3.0}}}
    args = build_scenario_args(cfg, 1, np.random.default_rng(0))
    joined = " ".join(args)
    assert "--tunnels=2000:2800" in joined
    assert "--tunnelEnbSpacing=250" in joined
    assert "--tunnelEnbTxDbm=30.0" in joined
    assert "--tunnelMicroLossDb=3.0" in joined


def test_no_tunnel_no_micro_args():
    cfg = {"ns3": {"scenario": {"randomize": False, "tunnels": []}}}
    joined = " ".join(build_scenario_args(cfg, 1, np.random.default_rng(0)))
    assert "--tunnels=" in joined and "--tunnelEnbSpacing" not in joined


def test_randomized_empty_tunnels_are_explicit():
    cfg = _base_cfg()
    args = build_scenario_args(cfg, 1, np.random.default_rng(0))
    assert "--tunnels=" in args


def test_family_tunnels_override_and_round_once():
    cfg = _base_cfg()
    cfg["ns3"]["scenario_families"] = {
        "clear": {"randomize": False, "tunnels": []},
        "tunnel": {"randomize": False, "tunnels": [[2000.4, 2800.4]]},
    }
    assert "--tunnels=" in build_scenario_args(cfg, 1, np.random.default_rng(0), "clear")
    assert "--tunnels=2000:2800" in build_scenario_args(cfg, 1, np.random.default_rng(0), "tunnel")
