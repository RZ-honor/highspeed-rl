"""H15sm（语义修正小规模重训）配置的机械派生 + 布尔校验。

母本：configs/ns3_stageH13.yaml（生产产物 H13 的训练配置）。
差异白名单（只允许这 4 个键不同，其余逐键相等）：
  training.n_episodes        1200 → 300        （小规模预算）
  training.eval_every_cycles 200  → 150        （监控点 2 次，评估开销核算）
  training.checkpoint_dir    → checkpoints_stageH15sm
  ns3.binary                 → 隔离 HSR4 构建产物（本轮真实验收编译的
                               ns3.43-hsr4-smoke；生产 run_hsr.sh 是**旧
                               HSR3 协议**二进制，不能用来验证新语义）
不修改母本；输出 configs/ns3_stageH15sm.yaml；任何白名单外差异 = FATAL。
"""
import sys

import yaml

SRC = "configs/ns3_stageH13.yaml"
DST = "configs/ns3_stageH15sm.yaml"
BINARY = ("/work/home/<HPC_ACCOUNT>/ns3root/ns-3.43/build/scratch/"
          "hsr4_validation_20261012/ns3.43-hsr4-smoke")

OVERRIDES = {
    ("training", "n_episodes"): 300,
    ("training", "eval_every_cycles"): 150,
    ("training", "checkpoint_dir"): "checkpoints_stageH15sm",
    ("ns3", "binary"): BINARY,
}


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
            node = node[key]
        node[path[-1]] = val
    fb, fc = flatten(base), flatten(cfg)
    assert set(fb) == set(fc), f"键集漂移: {set(fb) ^ set(fc)}"
    diffs = {k for k in fb if fb[k] != fc[k]}
    expect = set(OVERRIDES)
    assert diffs == expect, (
        f"差异越界: 多出 {diffs - expect} 缺失 {expect - diffs}")
    with open(DST, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print("H15SM_CFG_OK", DST, "diffs:", sorted(map("->".join, diffs)) if diffs else diffs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
