"""配置启动校验 + 场景族（课程化）相关测试。

动机：ns3.yaml 曾缺 env.w_outage，env 用 cfg.get(...,0) 静默取默认值 →
弱覆盖预警整段失效且无任何报错。这里把「半配置漂移」变成启动即失败，
并覆盖场景族合并/入池过滤的行为。
"""

import pytest
import yaml

from src.utils.config import ConfigError, load_config, validate_config
from src.envs.ns3_env import build_scenario_args, family_pool

CFG = load_config("configs/ns3.yaml")


def test_all_shipped_configs_pass_validation(tmp_path):
    import glob
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = sorted(glob.glob(os.path.join(root, "configs", "*.yaml")))
    assert files, "未找到 configs/*.yaml"
    for f in files:
        validate_config(yaml.safe_load(open(f, encoding="utf-8")))


def test_minimal_config_allowed():
    """非 RL 场景的最小配置（无 env/agent 段）不得被拒。"""
    cfg = validate_config({"a": 1, "b": {"c": 2}})
    assert cfg["a"] == 1


def test_missing_env_key_raises():
    """有 env 段却缺奖励键 → 必须抛错（正是 w_outage 那类缺陷）。"""
    cfg = yaml.safe_load(open("configs/ns3.yaml", encoding="utf-8"))
    cfg["env"].pop("w_outage")
    with pytest.raises(ConfigError) as e:
        validate_config(cfg)
    assert "env.w_outage" in str(e.value)


def test_non_numeric_value_raises():
    cfg = yaml.safe_load(open("configs/ns3.yaml", encoding="utf-8"))
    cfg["env"]["ho_penalty"] = "3.0"
    with pytest.raises(ConfigError) as e:
        validate_config(cfg)
    assert "env.ho_penalty" in str(e.value)


# ---------------- 场景族 ----------------

def test_family_pool_skips_disabled():
    """只有 Stage0 标定达标的族才入池（2026-09-17 标定：tunnel/congested
    教师存活率 0/3 → 出池；tunnel 待教师口径修复后重标定）。"""
    # 教师口径修复后四族全部标定达标（survival 1.00 / 480 步）→ 全部入池。
    # 若某族后续标定掉到 survival<0.7，应在 yaml 里 enabled: false 并同步本断言。
    pool = family_pool(CFG)
    # hotcold = S2 冷热对族（冷小区加功率，使「切向空闲邻区」安全可用）；
    # fading = 开启 Jakes 快衰落的对照族（检验既有结论在真实快衰落下是否成立）。
    # 两者入池前均须先通过各自的 A3 教师标定（存活率 ≥0.7）。
    assert sorted(pool) == ["congested", "fading", "hotcold", "load", "normal", "tunnel"]


def test_family_pool_empty_when_absent():
    assert family_pool({"ns3": {}}) == []


def test_build_scenario_args_family_overrides():
    import numpy as np
    rng = np.random.default_rng(0)
    base = build_scenario_args(CFG, 1, rng)
    fam = build_scenario_args(CFG, 1, rng, family="normal")
    joined = " ".join(fam)
    assert "--nBgUe=" in joined
    # 空隧道参数覆盖基础命令中的旧部署，但不部署洞内微站。
    assert "--tunnels=" in fam
    assert "--tunnelMicroLossDb=" not in joined
    assert "--tunnelEnbSpacing=" not in joined
    tun = " ".join(build_scenario_args(CFG, 1, rng, family="tunnel"))
    assert "--tunnels=" in tun and "--tunnelMicroLossDb=" in tun
    assert isinstance(base, list)


def test_build_scenario_args_unknown_family():
    import numpy as np
    with pytest.raises(ValueError):
        build_scenario_args(CFG, 1, np.random.default_rng(0), family="nope")
