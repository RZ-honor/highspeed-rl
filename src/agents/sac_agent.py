"""离散 SAC（soft Q-learning）变体：与 DuelingDDQNAgent 只差**学习规则**。

动机（2026-10-12，算法增强评估）
--------------------------------
H4→H13 整个课程反复出现同一病灶：**自强化坍缩**——在线微调约 c100 达峰后
Q(HO) 下漂 → 快照不再切换 → buffer 里 HO 样本更少 → Q(HO) 进一步被压低。
现有对策是手工信任域（bc_anchor）+ best.pt 早停，都是**事后补救**。

最大熵框架从机制上免疫这条正反馈：soft Q 的策略是 softmax(Q/α)，熵项
α·H(π) 使概率质量塌向单一动作时损失上升——**结构性地禁止 ho=0 式坍缩**，
不需要手工锚定（锚定本阶段仍保留以维持单变量对比）。温度 α 自动调节
（对齐目标熵 H̄ = −entropy_coef·log(1/|A|)），替代固定 ε 衰减的探索调度。

实现选择：soft Q-learning（Christodoulou 2019）而非 actor-critic 版 SAC——
动作空间离散（7 个切换决策），无需引入 actor 网络；策略即 softmax(Q/α)，
评估侧仍可 argmax(Q) 贪心，checkpoint 的 "online" 键与全部评估管线
（paired_matrix / path_probe / eval_ns3）**零改动兼容**。

继承面：act / store / n 步回报 / PER / 安全层掩码 / BC 锚定 全部复用父类。
本类只改：双 critic + 双目标网、soft 目标计算、温度自适应、save/load 格式。

配置键（agent 段，均有默认值，缺省行为与直接实例化一致）：
    algorithm: sac          # train_ns3 据此选择本类
    sac_alpha_init: 0.2     # 初始温度（回报量级 ~±300/集、单步 r ~±2，
                            # α=0.2 时 7 动作 softmax 近均匀 → 随训练退火）
    sac_alpha_lr: 1.0e-4    # log α 的 Adam lr（与主 lr 同量级）
    sac_entropy_coef: 0.98  # H̄ = −coef·log(1/A)，标准取值
"""

import copy

import numpy as np
import torch
import torch.optim as optim

from src.agents.agent import DuelingDDQNAgent


class DiscreteSACAgent(DuelingDDQNAgent):
    """Dueling 双 critic 的 soft Q-learning + 自动温度。

    与父类的构造差异：多一个 online_net2/target_net2 与 log_alpha 参数。
    ``online_net`` 仍是 critic1 —— 所有按 ``agent.online_net`` 取网络的
    外围代码（evaluate、collector 快照、锚点）无需感知本类的存在。
    """

    def __init__(self, cfg: dict, device: torch.device, net_factory=None):
        super().__init__(cfg, device, net_factory=net_factory)

        a = cfg["agent"]
        self.alpha_init = float(a.get("sac_alpha_init", 0.2))
        self.alpha_lr = float(a.get("sac_alpha_lr", 1.0e-4))
        # H̄ = −coef·log(1/A) = coef·log(A)：允许策略比均匀分布略尖。
        self.target_entropy = float(a.get("sac_entropy_coef", 0.98)) * float(
            np.log(len(self.legal_actions)))

        # critic2 与 critic1 同构、同起点（标准实现是独立随机初始化；
        # 这里从 H8 best 热启动时两者会被 load 覆盖，冷启动时同起点
        # 也无碍 min-Q 悲观性——两网从同一点出发、TD 噪声独立分化）。
        self.online_net2 = self.net_factory(self.obs_dim, self.n_actions,
                                            self.hidden, **self.net_kwargs).to(device)
        self.online_net2.load_state_dict(self.online_net.state_dict())
        self.target_net2 = copy.deepcopy(self.online_net2)
        self.target_net2.eval()

        # 父类优化器只含 critic1；重建为双 critic 共同更新（漏掉这步 critic2
        # 永不更新，min-Q 退化成单 critic）。
        self.opt = optim.Adam(list(self.online_net.parameters())
                              + list(self.online_net2.parameters()), lr=self.lr)

        self.log_alpha = torch.tensor(float(np.log(self.alpha_init)),
                                      device=device, requires_grad=True)
        self.opt_alpha = optim.Adam([self.log_alpha], lr=self.alpha_lr)

    # ---------------- 学习 ----------------

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.detach().exp().clamp(1e-4, 10.0).item())

    def _q_of(self, net, s_or_h):
        if self.latent:
            return net.q(s_or_h)
        return net(s_or_h)

    def learn(self) -> float:
        if len(self.buffer) < self.batch:
            return 0.0
        batch, idxs, weights = self.buffer.sample(self.batch)
        w = weights.to(self.device)
        alpha = self.log_alpha.exp().clamp(1e-4, 10.0)

        if self.latent:
            h = batch["latent"].to(self.device)
            h2 = batch["next_latent"].to(self.device)
            s_q = h
            s2_q = h2
        else:
            s_q = batch["state"].to(self.device)
            s2_q = batch["next_state"].to(self.device)
        a = batch["action"].to(self.device)
        r = batch["reward"].to(self.device)

        # ---- soft 目标：y = R_n + γ^n·(1-d)·E_{a'~π}[min_i Q'_i − α log π] ----
        with torch.no_grad():
            qt = torch.min(self._q_of(self.target_net, s2_q),
                           self._q_of(self.target_net2, s2_q))
            qt = qt[:, self.legal_actions]
            logp = torch.log_softmax(qt / alpha, dim=1)
            v_next = (logp.exp() * (qt - alpha * logp)).sum(dim=1)
            y = r + self._bootstrap_discount(batch) * v_next

        q1 = self._q_of(self.online_net, s_q).gather(1, a.unsqueeze(1)).squeeze(1)
        q2 = self._q_of(self.online_net2, s_q).gather(1, a.unsqueeze(1)).squeeze(1)
        td = (q1 - y).detach().abs().cpu().numpy()
        critic_loss = (w * (q1 - y) ** 2).mean() + (w * (q2 - y) ** 2).mean()
        self._last_td = float(critic_loss.item()) / 2.0   # 与父类口径对齐：单 critic 均值

        loss = critic_loss
        if self.dyn_enabled and self._learn_step % self.dyn_every == 0:
            dyn_loss = self._dynamics_loss()
            loss = loss + self.dyn_weight * dyn_loss
            self._last_dyn_loss = float(dyn_loss.item())

        # BC 锚定（与 H13 同权保留 ⇒ 单变量 = 学习规则）
        if self.anchor_net is not None and self.bc_anchor_w > 0.0:
            q_all = self._q_of(self.online_net, s_q)
            with torch.no_grad():
                q_ref = self.anchor_net.q(s_q) if self.latent else self.anchor_net(s_q)
            anchor_loss = ((q_all - q_ref) ** 2).mean()
            loss = loss + self.bc_anchor_w * anchor_loss
            self._last_anchor = float(anchor_loss.item())

        # ---- 温度自适应：min logα·(H(π) − H̄)。符号方向：策略过尖（H<H̄）
        # ⇒ 梯度为负 ⇒ logα 上升 ⇒ α 上升 ⇒ softmax 变平（spinningup 同款）。
        with torch.no_grad():
            q_now = self._q_of(self.online_net, s_q)[:, self.legal_actions]
            logp_now = torch.log_softmax(q_now / alpha, dim=1)
            entropy = -(logp_now.exp() * logp_now).sum(dim=1)
        alpha_loss = (self.log_alpha * (entropy - self.target_entropy)).mean()

        self.opt.zero_grad()
        self.opt_alpha.zero_grad()
        (loss + alpha_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(self.online_net2.parameters(), 10.0)
        self.opt.step()
        self.opt_alpha.step()

        self.buffer.update_priorities(idxs, td)
        self._learn_step += 1
        if self._learn_step % self.sync_steps == 0:
            self.sync_target()
        return float(loss.item() + alpha_loss.item())

    # ---------------- 同步 / 持久化 ----------------

    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net2.load_state_dict(self.online_net2.state_dict())

    def save(self, path: str):
        torch.save({"online": self.online_net.state_dict(),
                    "online2": self.online_net2.state_dict(),
                    "log_alpha": float(self.log_alpha.detach().item()),
                    "eps": self.eps}, path)

    def load(self, path: str):
        """载入。兼容三种来源：

        * 旧 DQN checkpoint（只有 "online"）：critic1/critic2 同起点热启动
          ——H14 的 --init/--anchor-init 正是这种（checkpoints_stageH8/best.pt）。
        * 本类 checkpoint（含 "online2"/"log_alpha"）：完整恢复。
        * 裸 state_dict（无包装）：与父类行为一致。
        """
        ck = torch.load(path, map_location=self.device)
        if isinstance(ck, dict) and "online" in ck:
            self.online_net.load_state_dict(ck["online"])
            self.online_net2.load_state_dict(ck["online"])
            if "online2" in ck:
                self.online_net2.load_state_dict(ck["online2"])
            if "log_alpha" in ck:
                with torch.no_grad():
                    self.log_alpha.fill_(float(ck["log_alpha"]))
            self.eps = float(ck.get("eps", self.eps))
        else:
            sd = ck.state_dict() if hasattr(ck, "state_dict") else ck
            self.online_net.load_state_dict(sd)
            self.online_net2.load_state_dict(sd)
        self.sync_target()
