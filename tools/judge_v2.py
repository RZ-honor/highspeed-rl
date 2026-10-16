"""H16 v2 判据判读（J1/J5 用 diag_death JSON，J2/J3/J4 用 paired_matrix JSON，
J6 用双跑 A3 逐位比对，J7/J8 读探针 JSON）。每条独立布尔，缺文件记 N/A。

用法（本地）：
    python session_logs/h16_shield_cf/judge_v2.py --dir session_logs/h16_shield_cf/logs
"""
import argparse
import json
import os
from itertools import combinations

import numpy as np

A3P = "A3(3GPP)"
RLP = "RL(hybrid)"


def boot_ci(d, n=4000, seed=0):
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def load(path):
    """支持项目整理后的无损压缩证据（make_cleanup.py 会把 >200 KB 的证据 gzip）。"""
    for cand, opener in ((path, "p"), (path + ".gz", "gz")):
        if not os.path.exists(cand):
            continue
        if opener == "gz":
            import gzip
            with gzip.open(cand, "rt", encoding="utf-8") as f:
                return json.load(f)
        with open(cand, encoding="utf-8") as f:
            return json.load(f)
    return None


def episodes_of(tag, dir_):
    d = load(os.path.join(dir_, f"{tag}.json"))
    return None if d is None else d["episodes"]


def a5_intent(hist):
    """掩码**前**的策略意图：静态合法集 (0,1,2,5) 上 q argmax 是否为 5。

    不能用 a_rl——select_action 返回的是掩码后动作，被盾拦掉的动作永远不会出现在
    a_rl 里，用它统计会把「策略想切但被盾挡下」误判成「策略不想切」（J5b 首次
    FAIL 即此缺陷）。q 向量是掩码前原值，与晚间 H15 分歧点分析同口径。
    """
    return sum(1 for h in hist if h.get("q")
               and int(np.argmax([h["q"][a] for a in (0, 1, 2, 5)])) == 5)


def judge_load16(eps, require_a5_zero):
    """返回 (survival, a5_intent, a5_exec, masked_hard(max,sum), ok)。"""
    surv = np.mean([e["termination"] == "truncated" and e["steps"] == 480 for e in eps])
    sel = sum(a5_intent(e["history"]) for e in eps)
    ex = sum(1 for e in eps for h in e["history"] if h["executed"] == 5)
    mh = [e.get("src_count", {}).get("masked_hard", 0) for e in eps]
    ok = (surv == 1.0 and (sel == 0) == bool(require_a5_zero)
          and max(mh or [0]) <= 2 and sum(mh or [0]) <= 16)
    return surv, sel, ex, (max(mh or [0]), sum(mh or [0])), ok


def per_family(d):
    """paired_matrix json → {fam: dict(a3, rl, d_tput, surv_rl, ho_rl, ho_a3)}"""
    out = {}
    for fam in d["families"]:
        seeds = [str(s) for s in d["seeds"]]
        rl = [d["results"][fam][RLP][s] for s in seeds]
        a3 = [d["results"][fam][A3P][s] for s in seeds]
        out[fam] = {
            "a3": float(np.mean([x["throughput"] for x in a3])),
            "rl": float(np.mean([x["throughput"] for x in rl])),
            "d": [x["throughput"] - y["throughput"] for x, y in zip(rl, a3)],
            "surv": float(np.mean([x["survival"] for x in rl])),
            "ho": float(np.mean([x["ho"] for x in rl])),
            "ho_a3": float(np.mean([x["ho"] for x in a3])),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="session_logs/h16_shield_cf/logs")
    a = p.parse_args()
    res, notes = {}, []

    # J1 主判据：load16 全 16 集，a5 选中必须为 0，盾兜底改写受限
    e = episodes_of("j1_load16", a.dir)
    if e is None:
        res["J1"] = "N/A"
    else:
        surv, sel, ex, mh, ok = judge_load16(e, True)
        res["J1"] = "PASS" if ok and len(e) == 16 else "FAIL"
        notes.append(f"J1: n={len(e)} surv={surv:.3f} a5_sel={sel} a5_exec={ex} "
                     f"masked_hard(max,sum)={mh}")

    # J5a 只留盾（grace=0）也必须全存活 —— 证明存活不靠冻结
    e = episodes_of("j5a_nograce", a.dir)
    if e is None:
        res["J5a"] = "N/A"
    else:
        surv, sel, ex, mh, ok = judge_load16(e, False)
        res["J5a"] = "PASS" if surv == 1.0 else "FAIL"
        notes.append(f"J5a: surv={surv:.3f} a5_sel={sel}")

    # J5b 只留冻结（无盾）：归因诊断，两种结果都可接受但必须可区分
    #   (i)  a5_intent>0            → 盾在替模型挡刀，J1 引用必须附审计值；
    #   (ii) a5_intent=0 且全存活   → 模型自身不再选弱目标（更强结论）。
    e = episodes_of("j5b_noshield", a.dir)
    if e is None:
        res["J5b"] = "N/A"
    else:
        sel = sum(a5_intent(x["history"]) for x in e)
        ex = sum(1 for x in e for h in x["history"] if h["executed"] == 5)
        surv = np.mean([x["termination"] == "truncated" and x["steps"] == 480 for x in e])
        if sel > 0:
            res["J5b"] = "PASS(shield-dependent)"
        elif surv == 1.0 and ex == 0:
            res["J5b"] = "PASS(model-intrinsic)"
        else:
            res["J5b"] = f"FAIL:intent0_but_surv={surv:.3f}_exec={ex}"
        notes.append(f"J5b: a5_intent={sel} a5_exec={ex} surv={surv:.3f} → {res['J5b']}")

    # J2 六族 survival=1.00 + 各族 Δtput bootstrap CI 下界 > 0
    d16 = load(os.path.join(a.dir, "j2_h16.json"))
    if d16 is None:
        res["J2"] = res["J3"] = res["J4"] = "N/A"
    else:
        f16 = per_family(d16)
        sig = {k: boot_ci(v["d"]) for k, v in f16.items()}
        j2 = (all(abs(v["surv"] - 1.0) < 1e-9 for v in f16.values())
              and all(lo > 0 for lo, _ in sig.values()))
        res["J2"] = "PASS" if j2 else "FAIL"
        notes.append("J2: " + "  ".join(
            f"{k} Δ={v['rl']-v['a3']:+.2f}(CI{sig[k][0]:+.2f}) surv={v['surv']:.3f}"
            for k, v in f16.items()))

        d13 = load(os.path.join(a.dir, "j3_h13.json"))
        if d13 is None:
            res["J3"] = res["J4"] = "N/A"
        else:
            f13 = per_family(d13)
            bad3 = []
            bad4 = []
            for fam in f16:
                if fam not in f13:
                    bad3.append((fam, "H13 缺族"))
                    continue
                D13, D16 = f13[fam]["rl"] - f13[fam]["a3"], f16[fam]["rl"] - f16[fam]["a3"]
                tol = 0.02 * abs(D13) + 0.05
                if D16 < D13 - tol:
                    bad3.append((fam, round(D16 - D13, 3)))
                h16, h13 = f16[fam]["ho"], f13[fam]["ho"]
                if not (h16 <= 1.2 * h13 + 0.01 and h16 >= 1.0):
                    bad4.append((fam, h16, h13))
            res["J3"] = "PASS" if not bad3 else f"FAIL:{bad3}"
            res["J4"] = "PASS" if not bad4 else f"FAIL:{bad4}"
            notes.append("J3 Δ16-Δ13: " + "  ".join(
                f"{f} {f16[f]['rl']-f16[f]['a3']-(f13[f]['rl']-f13[f]['a3']):+.2f}"
                for f in f16 if f in f13))

    # J6 A3 同二进制逐位一致性（runA/runB）
    A, B = load(os.path.join(a.dir, "j6_A.json")), load(os.path.join(a.dir, "j6_B.json"))
    if A is None or B is None:
        res["J6"] = "N/A"
    else:
        diffs = []
        for fam, pol, s in [(f, A3P, str(x)) for f in A["families"] for x in A["seeds"]]:
            va = A["results"][fam][pol][s]["throughput"]
            vb = B["results"][fam][pol][s]["throughput"]
            diffs.append(abs(va - vb))
        res["J6"] = "PASS" if max(diffs) == 0.0 else f"FAIL:{max(diffs):.6f}"
        notes.append(f"J6 max_abs_diff={max(diffs)}")

    # J7 M 矩阵：黑洞中位下降 ≥60%；时延只在 N≤64 声明
    blk = []
    for i in range(1, 9):
        d = load(os.path.join(a.dir, f"j7_m{i}.json"))
        if d is None:
            continue
        try:
            k3 = d["a3"]["summary"]["blackout_ms_per_km"]
            k4 = d["rl"]["summary"]["blackout_ms_per_km"]
            p3 = d["a3"]["summary"].get("delay_p95_ms")
            p4 = d["rl"]["summary"].get("delay_p95_ms")
            blk.append((i, 1 - k4 / k3, p3, p4))
        except (KeyError, TypeError, ZeroDivisionError) as exc:
            notes.append(f"J7 m{i} 解析失败: {exc}")
    if blk:
        med = float(np.median([r for _, r, _, _ in blk]))
        res["J7"] = "PASS" if med >= 0.60 else f"FAIL:median={med:.3f}"
        notes.append("J7 黑洞下降: " + "  ".join(f"m{i} {r:+.1%}" for i, r, _, _ in blk)
                     + f"  中位 {med:+.1%}")
        notes.append("J7 p95(A3→RL): " + "  ".join(
            f"m{i} {p3:.0f}→{p4:.0f}" for i, _, p3, p4 in blk
            if p3 is not None and p4 is not None))
    else:
        res["J7"] = "N/A"

    # J8 容量（rl 侧：qos_ok、RLF=0、乒乓=0、存活 1.0）
    d = load(os.path.join(a.dir, "j8_capacity.json"))
    if d is None:
        res["J8"] = "N/A"
    else:
        rows = [r for r in d if r.get("policy") == "rl"]
        bad = [r["n_users"] for r in rows
               if not (r.get("qos_ok") and r.get("rlf_rate", 1) == 0
                       and r.get("pingpong_rate", 1) == 0
                       and r.get("survival", 0) >= 1.0 - 1e-9)]
        res["J8"] = "PASS" if rows and not bad else f"FAIL:{bad or 'no rows'}"
        notes.append("J8 RL 侧: " + "  ".join(
            f"N={r['n_users']} qos={r.get('qos_ok')} rlf={r.get('rlf_rate')} "
            f"pp={r.get('pingpong_rate')} surv={r.get('survival')}" for r in rows))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print("\n".join(notes))


if __name__ == "__main__":
    main()
