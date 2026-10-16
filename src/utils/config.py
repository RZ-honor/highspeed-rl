"""配置加载与启动时校验。

背景（教训）：`ns3.yaml` 曾缺失 `env.w_outage` 等奖励键，而 `ns3_env.py` 用
`cfg.get(..., 0)` 静默取默认值 → 弱覆盖预警整段失效、训练退化为
「永不切换」，且**没有任何报错**。这类"配置漂移"是系统性风险，必须
fail-fast。

校验策略（宽进严出）：
  * section 若**存在**，则其中所有必需键必须存在且为数值——专治
    「有 env 段但缺了 w_outage」这类半配置漂移；
  * section 整体缺失不报错——兼容测试用的最小配置与规则类脚本
    （它们根本不构造 env/agent）。
"""

import yaml


class ConfigError(ValueError):
    """配置缺失/类型错误（启动即失败，带全部问题清单）。"""


# 各 section 的必需数值键（奖励/安全/学习的关键项；缺任何一项都会让
# env/agent 落到与设计文档不符的退化默认值上）
_REQUIRED = {
    "env": ("obs_dim", "n_actions", "episode_steps",
            "w_throughput", "w_handover", "w_delay", "w_stability",
            "ho_penalty", "pingpong_penalty", "stability_bonus",
            "rlf_sinr_db", "rlf_consecutive_steps",
            "rlf_fallback_sinr_db", "buf_arrival_kbps",
            # ---- 奖励/安全口径键（此前只在 default.yaml 有，6/9 份配置
            # 缺失 → 训练与评估奖励口径不一致且静默。故列为必需）----
            "w_outage", "weak_coverage_sinr_db",
            "rlf_drop_penalty", "rlf_termination_penalty",
            "fallback_persist_steps",
            "preempt_rsrp_dbm", "preempt_margin_db"),
    "safety": ("min_target_rsrp_dbm", "max_drop_vs_serving_db",
               "pingpong_block_steps"),
    "agent": ("gamma", "lr", "hidden_dim", "batch_size", "buffer_size",
              "eps_start", "eps_end", "eps_decay", "target_sync_steps",
              "per_alpha", "per_beta_start", "per_beta_end", "per_epsilon"),
    "training": ("n_episodes", "warmup_steps", "checkpoint_dir"),
}

# 上述必需键中属于「字符串型」的例外（其余一律要求数值）
_STRING_KEYS = {"training": ("checkpoint_dir",)}


def validate_config(cfg: dict) -> dict:
    """校验配置；有问题一次性抛出全部（避免挤牙膏式修配置）。"""
    problems = []
    for sec, keys in _REQUIRED.items():
        sub = cfg.get(sec)
        if sub is None:
            continue                      # 整段缺失：非 RL 场景，放过
        if not isinstance(sub, dict):
            problems.append(f"{sec}: 存在但不是映射 (got {type(sub).__name__})")
            continue
        str_keys = _STRING_KEYS.get(sec, ())
        for k in keys:
            if k not in sub:
                problems.append(f"{sec}.{k}: 缺失（env/agent 会取到与设计"
                                f"文档不符的退化默认值）")
            elif k in str_keys:
                if not isinstance(sub[k], str) or not sub[k]:
                    problems.append(f"{sec}.{k}={sub[k]!r}: 必须是非空字符串")
            elif not isinstance(sub[k], (int, float)) or isinstance(sub[k], bool):
                problems.append(f"{sec}.{k}={sub[k]!r}: 不是数值")
    if problems:
        raise ConfigError("配置校验失败:\n  " + "\n  ".join(problems))
    return cfg


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return validate_config(yaml.safe_load(f))
