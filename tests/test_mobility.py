import numpy as np
from src.utils.config import load_config
from src.envs.mobility import Mobility

CFG = load_config("configs/default.yaml")


def test_topology_count():
    m = Mobility(dict(CFG), seed=0)
    bs = m.build_topology()
    assert bs.shape == (CFG["mobility"]["n_bs"], 3)
    assert np.all(bs[:, 2] == CFG["channel"]["bs_height_m"])


def test_position_linear():
    m = Mobility(dict(CFG), seed=0)
    p0 = m.position(0.0)
    p1 = m.position(1.0)
    dx = p1[0] - p0[0]
    assert abs(dx - CFG["mobility"]["speed_mps"]) < 1e-6