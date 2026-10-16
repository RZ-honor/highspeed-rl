import threading
from collections import deque

import numpy as np
import torch

# 动作索引：0 stay; 1,2,5 HO->nbr1,2,3; 3,4 dual(+nbr1,2); 6 triple
HO_ACTIONS = (1, 2, 5)
DUAL_ACTIONS = (3, 4, 6)
NS3_ACTIONS = (0, 1, 2, 5)
# 动作 → 候选数组下标（0 = 服务小区，1/2/3 = nbr1/nbr2/nbr3）
HO_TARGET_IDX = {1: 1, 2: 2, 5: 3}

# ---- hard shield：常开的宽松防护盾（permissive shielding）----
# 根因（session_logs/h15_survival_fix/focused_root_cause_20261014.md）：
# configured 模式 + guard 关闭时整层 SafetyLayer 不生效，弱目标保护随之消失，
# 两条 load16 轨迹以 action5 切向弱 9~11 dB、目标 SINR≈-13 dB 的小区后连续
# 低 SINR 终止。文献结论（Shielding/CACM 2024、BSK-RL、Huang & Ontañón）：
# 安全盾应**常开且宽松**——只挡「切过去即低于终止门限」的动作，不挡
# 「略弱但健康」的负载均衡目标。因此本层与 guard 的性能门禁解耦。
#
# 观测索引表（28 维契约，_build_obs 的 _norm 仿射可逆）：
#   RSRP 维（_norm(x, -140, -44)）：0=服务, 2=nbr1, 3=nbr2, 14=nbr3
#   SINR 维（_norm(x, -30, 30)）： 1=服务, 15=nbr1, 16=nbr2, 17=nbr3
OBS_HARD_RSRP = {0: 0, 1: 2, 2: 3, 5: 14}
OBS_HARD_SINR = {0: 1, 1: 15, 2: 16, 5: 17}


def hard_shield_params(cfg: dict):
    """hard shield 参数；`safety.hard_shield: false` 或缺省 → 关闭（旧行为）。"""
    s = cfg.get("safety", {})
    if not bool(s.get("hard_shield", False)):
        return None
    return {
        "min_rsrp": float(s.get("hard_min_target_rsrp_dbm", -110)),
        "min_sinr": float(s.get("hard_min_target_sinr_db",
                               cfg.get("env", {}).get("rlf_sinr_db", -6))),
        "max_drop": float(s.get("hard_max_drop_vs_serving_db", 10)),
    }


def hard_block_reason(cand_r, cand_s, action, params) -> bool:
    """单动作判据：切向该候选是否低于硬安全线（推理/环境共用一份语义）。

    cand_r/cand_s：候选 RSRP/SINR 数组（下标 0=服务）；cand_s 可为 None
    （旧契约没有 candidate_sinr 时退化为仅 RSRP 判据）。
    """
    i = HO_TARGET_IDX[action]
    if i >= len(cand_r):
        return False
    r = float(cand_r[i])
    if r < params["min_rsrp"] or (float(cand_r[0]) - r) > params["max_drop"]:
        return True
    if cand_s is not None and len(cand_s) > i:
        if float(cand_s[i]) < params["min_sinr"]:
            return True
    return False


def hard_mask(q, info, cfg):
    """推理侧 hard mask：依据 info 的候选 RSRP/SINR 屏蔽危险 HO 动作。"""
    p = hard_shield_params(cfg)
    if p is None:
        return q
    cand_r = info.get("candidate_rsrp")
    if cand_r is None or float(cand_r[0]) <= -190.0:   # 测量未上报 → 不拦（与 legacy 同）
        return q
    cand_s = info.get("candidate_sinr")
    m = q.clone()
    for a in HO_TARGET_IDX:
        if hard_block_reason(cand_r, cand_s, a, p):
            m[..., a] = float("-inf")
    return m


def _denorm_rsrp(v):
    return v * 96.0 - 140.0          # _norm(x, -140, -44) 的仿射逆变换


def _denorm_sinr(v):
    return v * 60.0 - 30.0           # _norm(x, -30, 30) 的仿射逆变换


def hard_admissible_from_obs(obs, cfg, n_actions: int = 7):
    """训练侧：从 next_state **观测本身**重建 hard shield 可行动作集。

    bootstrap 的 target max 必须与部署端同一可行动作集（Huang & Ontañón 2020）。
    hard shield 是观测的确定性函数（候选 RSRP/SINR 都在 28 维观测里），故无需
    额外存储 masked_actions。14 维旧观测缺 nbr3/SINR 维 → 无法重建，返回 None
    （此时 bootstrap 保持旧的静态动作集行为，并在测试里显式声明该边界）。
    """
    p = hard_shield_params(cfg)
    if p is None or not bool(cfg.get("env", {}).get("extended_obs", False)):
        return None
    if int(n_actions) != 7:
        return None
    v = torch.as_tensor(obs, dtype=torch.float32)
    n = v.shape[-1]
    if n <= max(OBS_HARD_RSRP.values()):
        return None
    adm = torch.ones(v.shape[:-1] + (7,), dtype=torch.bool, device=v.device)
    srv_r = _denorm_rsrp(v[..., OBS_HARD_RSRP[0]])
    # 与推理侧同规则：测量未上报（obs[22]=0）时不拦（锁步协议下 obs[22] 就是
    # 环境用来区分「-140 截断值」与「-200 哨兵值」的唯一信息，_norm 已抹掉
    # 二者在 RSRP 维的差异）。
    reported = v[..., 22] > 0.5 if n > 22 else torch.ones_like(srv_r)
    for a, ri in OBS_HARD_RSRP.items():
        if a == 0:
            continue
        tgt_r = _denorm_rsrp(v[..., ri])
        # 无效邻区被 _norm 截到 -140 → 低于 min_rsrp → 屏蔽方向与推理侧(-200)
        # 一致，不产生训练/部署语义分叉。
        bad = (tgt_r < p["min_rsrp"]) | ((srv_r - tgt_r) > p["max_drop"])
        si = OBS_HARD_SINR[a]
        if si < n:
            # 与推理侧同规则（OR）：目标自身 SINR 低于终止门限即危险
            bad = bad | (_denorm_sinr(v[..., si]) < p["min_sinr"])
        adm[..., a] = ~(bad & reported)
    return adm


def legal_actions(cfg, n_actions):
    """Return the actions supported by the selected backend.

    A real NS-3 backend exposes only the four single-link decisions.  The
    Python environment and the subprocess mock intentionally retain the
    historical seven-action space so existing smoke tests/checkpoints remain
    compatible.  An explicit ``env.legal_actions`` override still wins.
    """
    actions = cfg.get("env", {}).get("legal_actions")
    is_mock = bool(cfg.get("mock_backend", False) or cfg.get("backend") == "mock")
    is_ns3 = bool(cfg.get("ns3")) and not is_mock
    if actions is None:
        actions = NS3_ACTIONS if is_ns3 else range(n_actions)
    actions = tuple(int(a) for a in actions)
    if not actions or len(set(actions)) != len(actions) or any(
            a < 0 or a >= n_actions for a in actions):
        raise ValueError("Invalid env.legal_actions")
    return actions


def mask_actions(q, actions):
    """Mask static unsupported actions without changing network/checkpoint shape."""
    valid = torch.zeros(q.shape[-1], dtype=torch.bool, device=q.device)
    valid[list(actions)] = True
    return q.masked_fill(~valid, float("-inf"))


def policy_mask(q, cfg, info, actions=None, guard=None):
    """Shared inference mask; legacy remains default for existing checkpoints.

    configured follows safety.mode / RuleGuard. Static backend legality is
    unconditional; guard.enable never implicitly disables legacy safety.
    The hard shield (safety.hard_shield) is **always** applied, independent of
    mode/guard: it blocks only targets whose measured quality is itself below
    the termination floor, so it cannot re-impose the strict-mode exploration
    choke that motivated the guard (RuleGuard) in the first place.
    """
    actions = legal_actions(cfg, q.shape[-1]) if actions is None else actions
    safety = cfg.get("safety", {})
    mode = safety.get("policy_mask", "legacy")
    if mode not in ("legacy", "configured"):
        raise ValueError("safety.policy_mask must be legacy or configured")
    q = hard_mask(q, info, cfg)          # 常开；未启用时原样返回
    active = (mode == "legacy" or
              (bool(guard.active) if guard is not None else
               str(safety.get("mode", "strict")) == "strict"))
    if active:
        q = SafetyLayer(safety).mask(q, info)
    return mask_actions(q, actions)


def select_action(q, cfg, info, eps=0.0, rng=None, actions=None, guard=None):
    """Greedy and exploratory decisions use the same admissible action set."""
    masked = policy_mask(q, cfg, info, actions, guard)
    rng = np.random if rng is None else rng
    if eps > 0.0 and rng.random() < eps:
        valid = torch.where(torch.isfinite(masked[0]))[0].cpu().numpy()
        if not len(valid):
            raise ValueError("No admissible policy action")
        return int(rng.choice(valid))
    return int(masked.argmax(dim=1).item())


class SafetyLayer:
    """3GPP A3 兜底：动作掩码 + RLF 触发回落。"""

    def __init__(self, cfg: dict):
        self.min_rsrp = float(cfg["min_target_rsrp_dbm"])
        self.max_drop = float(cfg["max_drop_vs_serving_db"])
        self.pp_block = int(cfg["pingpong_block_steps"])
        self.rlf_sinr = float(cfg["rlf_fallback_sinr_db"])
        # 双连通动作掩码：LENA(LTE) 无 EN-DC，动作 3/4/6 在物理层是空操作，
        # 却在 Python env 里能凭空翻倍吞吐 + 持续领取 stability_bonus。
        # 开启后策略与 env 两侧同时屏蔽，保证动作语义诚实。
        self.mask_dual = bool(cfg.get("mask_dual", False))

    def mask(self, q: torch.Tensor, info: dict) -> torch.Tensor:
        """对 [B, n_actions] 的 Q 值掩码非法动作（置 -inf）。"""
        m = q.clone()
        cand = info["candidate_rsrp"]  # [serving, n1, n2, n3]
        serving_rsrp = cand[0]
        nbr_rsrp = cand[1:4] if len(cand) > 3 else cand[1:]
        # 动作 -> 邻区索引（在 nbr_rsrp 中）：1->0,2->1,5->2
        ho_to_nbr = {1: 0, 2: 1, 5: 2}
        for a, ni in ho_to_nbr.items():
            if ni < len(nbr_rsrp):
                r = nbr_rsrp[ni]
                if r < self.min_rsrp or (serving_rsrp - r) > self.max_drop:
                    m[..., a] = float("-inf")
        # 乒乓阻断：近 pp_block 步内切换 → 掩码所有 HO 动作
        if (info["step_idx"] - info["last_ho_step"]) <= self.pp_block:
            for a in HO_ACTIONS:
                m[..., a] = float("-inf")
        # 双连通掩码（与 env.step 内的处理保持一致，避免策略侧仍在选空操作）
        if self.mask_dual:
            for a in DUAL_ACTIONS:
                m[..., a] = float("-inf")
        return m

    def should_fallback(self, serving_sinr: float, action: int) -> bool:
        """服务 SINR 跌破临界且 RL 未发起合法切换 → A3 回落。"""
        rl_switched = action in HO_ACTIONS or action in DUAL_ACTIONS
        return (serving_sinr < self.rlf_sinr) and (not rl_switched)

    def a3_action(self, info: dict) -> int:
        """A3 标准决策：切到最强合法邻区；无可行则 stay。"""
        cand = info["candidate_rsrp"]
        serving_rsrp = cand[0]
        nbr = cand[1:4] if len(cand) > 3 else cand[1:]
        best, best_i = -200.0, -1
        for i, r in enumerate(nbr):
            if r > serving_rsrp + 3.0 and r >= self.min_rsrp:
                if r > best:
                    best, best_i = r, i
        if best_i < 0:
            return 0
        # 邻区 i -> 动作：0->a1,1->a2,2->a5
        return {0: 1, 1: 2, 2: 5}[best_i]


class RuleGuard:
    """规则接管的门禁：**只有 RL 退化到比规则差才让规则接手**。

    背景
    ----
    常规期对每一步动作做「弱目标掩码 / 乒乓阻断」会把 RL 的探索空间压死：
    实测评估里 79.3% 的 RL 决策被 `masked_weak` 改写成 stay（RL 想切次强小区
    做负载均衡，却被 RSRP 门限拦下），结果 ho=0、 episode 早早 RLF。而这些
    「非贪心切换」恰恰是 RL 相对 A3 的潜在增益所在。

    取值相反的两种立场：
      * 每步都管——策略退化为规则的安全克隆，学不到超越规则的决策；
      * 完全不管——训练前期可能长期失服，经验里全是 RLF 轨迹。

    这里取「性能门禁」折中：默认放行给 RL 探索；用**滚动窗口**比较 RL 回合
    收益与同一批场景下 A3 教师回合收益，只有 RL 均值落到规则均值之下才拉起
    规则（恢复弱目标掩码），并在 RL 追平后**带滞回**释放，避免频繁抖动。

    数据源：ParallelCollector 的 mixed 策略天然同时产生两种来源的回合
    （snapshot 分支 = rl，A3 教师分支 = a3），因此无需额外 rollout 开销。
    """

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enable", True))
        self.window = max(2, int(cfg.get("window_episodes", 8)))
        self.margin = float(cfg.get("margin", 0.0))
        self.min_eps = max(1, int(cfg.get("min_episodes", 2)))
        self.release = max(1, int(cfg.get("release_comparisons", 3)))
        self._lock = threading.Lock()
        self._a3 = deque(maxlen=self.window)
        self._rl = deque(maxlen=self.window)
        self._active = False
        self._ok_streak = 0
        self.comparisons = 0

    # ---- 状态 ----

    @property
    def active(self) -> bool:
        """规则是否接管（True = 恢复常规期掩码；False = 放行给 RL 探索）。"""
        return self.enabled and self._active

    def status(self) -> dict:
        with self._lock:
            return {"active": self._active, "comparisons": self.comparisons,
                    "n_a3": len(self._a3), "n_rl": len(self._rl)}

    # ---- 观测 ----

    def observe(self, source: str, episode_return: float) -> None:
        """记录一集的成绩，source ∈ {"rl", "a3"}；凑够窗口即做一次比较。"""
        if not self.enabled:
            return
        with self._lock:
            (self._rl if source == "rl" else self._a3).append(float(episode_return))
            if len(self._a3) >= self.min_eps and len(self._rl) >= self.min_eps:
                self._compare()

    # ---- 内部 ----

    def _compare(self):
        """两侧样本都够时做一次均值比较；比较完清空窗口，进入下一轮。"""
        m_a3 = float(np.mean(self._a3))
        m_rl = float(np.mean(self._rl))
        self.comparisons += 1
        # margin 为相对比例：RL 需低过规则 margin×|R_rule| 才判定退化
        thr = m_a3 - self.margin * abs(m_a3)
        was = self._active
        if m_rl < thr:
            self._active = True
            self._ok_streak = 0
        elif self._active:
            self._ok_streak += 1
            if self._ok_streak >= self.release:
                self._active = False
                self._ok_streak = 0
        self._a3.clear()
        self._rl.clear()
        if self._active != was:
            print(f"[guard] 规则接管={'ON' if self._active else 'OFF'} "
                  f"(RL={m_rl:.1f} vs A3={m_a3:.1f}, cmp#{self.comparisons})",
                  flush=True)
