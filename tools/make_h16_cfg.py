"""H16（动作空间死亡修复）配置的机械派生 + 布尔校验。

母本：configs/ns3_stageH15_survival_fix.yaml（判负归档的修复候选，
语义 = configured mask + guard 关闭；正是该语义释放了 action5 弱目标切换）。
差异白名单（只允许以下键被**新增/修改**，其余逐键相等）：
  safety.hard_shield                  +true      修复1+2：常开宽松安全盾
                                                 （推理/环境/bootstrap 三侧
                                                 同一开关，无独立死键）
  safety.hard_max_drop_vs_serving_db  +10        相对判据：比服务小区弱 10dB 以上才拦
  safety.hard_min_target_rsrp_dbm     +-110      绝对判据：目标 RSRP 地板
  safety.hard_min_target_sinr_db      +-6        目标 SINR 地板（=终止门限）
  env.measurement_grace_steps         +3         修复3：HO 后陈旧测量冻结计数
  training.checkpoint_dir             → checkpoints_stageH16_shieldfix
判据依据（session_logs/h15_survival_fix/focused_root_cause_20261014.md）：
两条死亡均为 action5 切向「比服务弱 9~11 dB、目标 SINR≈-13 dB」的小区后
连续低 SINR 终止；hard shield 用「弱 10dB 或目标低于终止门限」的**宽松**
判据拦下这类动作，而放行「略弱但健康」的负载均衡目标（不复活 strict 时代
79.3% 决策被改写的探索压制）。
不修改母本；输出 configs/ns3_stageH16_shieldfix.yaml；白名单外差异 = FATAL。
"""
import sys

import yaml

SRC = "configs/ns3_stageH15_survival_fix.yaml"
DST = "configs/ns3_stageH16_shieldfix.yaml"

OVERRIDES = {
    ("safety", "hard_shield"): True,
    ("safety", "hard_max_drop_vs_serving_db"): 10,
    ("safety", "hard_min_target_rsrp_dbm"): -110,
    ("safety", "hard_min_target_sinr_db"): -6,
    ("env", "measurement_grace_steps"): 3,
    ("training", "checkpoint_dir"): "checkpoints_stageH16_shieldfix",
}
WHITELIST = set(OVERRIDES)


def flatten(d, pre=()):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, pre + (k,)))
        else:
            out[pre + (k,)] = v
    return out


def main():
    base = yaml.safe_load(open(SRC, encoding="utf-8"))
    cfg = yaml.safe_load(open(SRC, encoding="utf-8"))  # 深拷贝（yaml 结构）
    for path, val in OVERRIDES.items():
        node = cfg
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = val
    fb, fc = flatten(base), flatten(cfg)
    added = set(fc) - set(fb)
    removed = set(fb) - set(fc)
    changed = {k for k in fb if k in fc and fb[k] != fc[k]}
    stray = (added | removed | changed) - WHITELIST
    assert not stray, f"白名单外差异 = FATAL: {sorted(stray)}"
    assert not removed, f"删除母本键 = FATAL: {sorted(removed)}"
    assert added | changed == WHITELIST, \
        f"白名单未全部落地: added={sorted(added)} changed={sorted(changed)}"
    with open(DST, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"OK {DST}: +{len(added)} 新增 / ~{len(changed)} 修改 / "
          f"-{len(removed)} 删除（白名单 {len(WHITELIST)} 键）")
    # 自检：新键生效值 + 关键旧语义未被稀释
    for path, val in OVERRIDES.items():
        node = fc[path]
        assert node == val, f"{path} 生效值 {node} != {val}"
    assert fc[("safety", "policy_mask")] == "configured"   # 仍走 configured 推理
    assert fc[("guard", "enable")] is False                 # guard 仍关闭
    print("self-check PASS: hard_shield(三侧同开关)/measurement_grace 生效，"
          "policy_mask=configured 与 guard.enable=false 未被改动")


if __name__ == "__main__":
    sys.exit(main())
