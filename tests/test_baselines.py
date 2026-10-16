import numpy as np
from src.utils.config import load_config
from src.agents.baselines import (
    A3Baseline, A3OperatorBaseline, GreedyBaseline,
    FixedIntervalBaseline, TabularQAgent, VanillaDQN,
)

CFG = load_config("configs/default.yaml")
INFO = {"candidate_rsrp": np.array([-95.0, -80.0, -85.0, -90.0]),
        "serving_idx": 0, "last_ho_step": -100, "step_idx": 0}


def test_a3_picks_strong():
    b = A3Baseline(CFG["safety"])
    a = b.act(np.zeros(14, dtype=np.float32), INFO)
    assert a == 1  # n1(-80) > serving(-95)+3


def test_greedy_picks_strong():
    b = GreedyBaseline()
    assert b.act(None, INFO) == 1


def test_fixed_interval():
    b = FixedIntervalBaseline(interval_steps=10)
    assert b.act(None, {**INFO, "step_idx": 0}) == 0
    assert b.act(None, {**INFO, "step_idx": 10}) in (1, 2, 5)


def test_tabular_q_shape():
    b = TabularQAgent()
    a = b.act(np.zeros(14, dtype=np.float32), INFO)
    assert 0 <= a < 7


def test_vanilla_dqn_act():
    b = VanillaDQN(CFG)
    a = b.act(np.zeros(14, dtype=np.float32), INFO)
    assert 0 <= a < 7


# cmd_0070 口部序列（serv=cell 3，微站 9）：23.0 可听但弱，25.3 才 +4 dB。
# obs[4] = v/100；83.3 m/s → 0.833，多普勒自适应在 300 km/h 标尺=1。
_MOUTH_OBS = np.zeros(14, dtype=np.float32)
_MOUTH_OBS[4] = 0.833
_MOUTH_PRE = [(-69.0, -83.0), (-65.0, -80.0), (-62.0, -77.0)]
_MOUTH_ENTER = (-66.0, -62.0)  # raw nbr = serving + 4 dB


def _mouth_info(serv_rsrp, nbr9_rsrp):
    return {
        "candidate_rsrp": np.array(
            [serv_rsrp, nbr9_rsrp, -90.0, -95.0], dtype=np.float64),
        "serving_idx": 3,
        "neighbors": [9, 10, 11],
        "last_ho_step": -100,
        "step_idx": 0,
    }


def _feed(pol, samples):
    return [pol.act(_MOUTH_OBS, _mouth_info(*s)) for s in samples]


def test_a3_operator_defaults_keep_operator_ttt():
    pol = A3OperatorBaseline(CFG["safety"], dt_s=0.1)
    assert pol.ttt_ms == 320.0
    assert pol._l3_alpha == 0.4
    assert pol.raw_enter is False


def test_a3_operator_mouth_tunnel_policy_handovers():
    """eval_tunnel：α=1 + TTT=40ms + raw_enter，口部 +4 dB 当步应切到第 1 邻区。"""
    pol = A3OperatorBaseline(
        CFG["safety"],
        {"hyst_db": 2.0, "ttt_ms": 40.0, "a3_offset_db": 3.0,
         "doppler_adaptive": True, "l3_alpha": 1.0, "raw_enter": True},
        dt_s=0.1,
    )
    assert _feed(pol, _MOUTH_PRE) == [0, 0, 0]
    a = pol.act(_MOUTH_OBS, _mouth_info(*_MOUTH_ENTER))
    assert a == 1  # HO → strongest (cell 9)


def test_a3_operator_mouth_default_stays():
    """默认 320ms / α=0.4 / 无 raw_enter：L3 把 +4 dB 滤成负差，TTT 从未启动。"""
    pol = A3OperatorBaseline(CFG["safety"], dt_s=0.1)
    assert _feed(pol, _MOUTH_PRE) == [0, 0, 0]
    stay = _feed(pol, [_MOUTH_ENTER] * 4)
    assert stay == [0, 0, 0, 0]


def test_a3_operator_raw_enter_bypasses_l3_lag():
    """α=0.4 仍滞后，但 raw_enter + TTT=40 时瞬时 +4 dB 即可进 TTT 并当步切换。"""
    pol = A3OperatorBaseline(
        CFG["safety"],
        {"ttt_ms": 40.0, "a3_offset_db": 3.0, "l3_alpha": 0.4,
         "raw_enter": True, "doppler_adaptive": True},
        dt_s=0.1,
    )
    _feed(pol, _MOUTH_PRE)
    assert pol.act(_MOUTH_OBS, _mouth_info(*_MOUTH_ENTER)) == 1
