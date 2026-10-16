import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.networks import DuelingDQN
from src.agents.replay import NStepAccumulator, PrioritizedReplayBuffer
from src.agents.safety import (SafetyLayer, hard_admissible_from_obs,
                               legal_actions, mask_actions, select_action, policy_mask)
from src.agents.seq_replay import EpisodeReplay


# 潜变量动力学的默认超参（configs 的 agent.dynamics 可覆盖）
DEFAULT_DYNAMICS = {
    "enabled": True,
    "weight": 0.5,        # 动力学损失相对 TD 损失的权重
    "burn_in": 8,         # burn-in 步数：带观测的 ODE-RNN 更新
    "horizon": 10,        # 外推步数：无观测的纯动力学预测（10 步 = 1s）
    "batch_size": 32,     # 动力学损失的窗口数
    "episode_capacity": 200,
    # 动力学损失每 N 个 learn 步才计算一次。BPTT 展开 (burn_in+horizon)×ode_steps
    # 层 ODE，是训练的主要算力开销；作为辅助损失无需每步更新，摊销后总开销
    # 降到约 1/N，实测对收敛速度无可测影响。
    "every": 4,
}


class DuelingDDQNAgent:
    """Dueling DDQN + PER + 安全层，可选潜变量 ODE-RNN 动力学。

    网络通过 ``net_factory(obs_dim, n_actions, hidden, **net_kwargs)`` 注入，
    以便在同一训练框架下公平对比不同网络。``net_kwargs`` 来自 config 的
    ``net_kwargs`` 段（所有网络工厂都接受并忽略自身不用的键）。

    两类网络，两条代码路径：
      * **无状态网络**（DuelingDQN / NeuralODEDuelingDQN / HybridDuelingDQN）
        —— 与原有行为完全一致，Q = net(obs)。
      * **潜变量网络**（:class:`src.agents.latent_ode.LatentODEQNetwork`，
        具备 ``step`` / ``flow`` / ``unroll`` / ``predict``）
        —— 隐状态跨 env 步传播，Q = net.q(h_t)；并额外用多步外推的
        预测损失监督 ODE 学到真实的信道时间演化。
    """

    def __init__(self, cfg: dict, device: torch.device, net_factory=None):
        self.cfg = cfg
        a = cfg["agent"]
        self.device = device
        self.gamma = float(a["gamma"])
        self.lr = float(a["lr"])
        self.batch = int(a["batch_size"])
        self.eps = float(a["eps_start"])
        self.eps_end = float(a["eps_end"])
        self.eps_decay = float(a["eps_decay"])
        self.sync_steps = int(a["target_sync_steps"])
        self.obs_dim = int(cfg["env"]["obs_dim"])
        self.n_actions = int(cfg["env"]["n_actions"])
        self.legal_actions = legal_actions(cfg, self.n_actions)
        self.hidden = int(a["hidden_dim"])
        self.net_factory = net_factory or DuelingDQN
        # 统一签名的额外构造参数（ODE 步数/隐层/时间步等）
        self.net_kwargs = dict(cfg.get("net_kwargs", {}) or {})

        self.online_net = self.net_factory(self.obs_dim, self.n_actions, self.hidden,
                                           **self.net_kwargs).to(device)
        self.target_net = self.net_factory(self.obs_dim, self.n_actions, self.hidden,
                                           **self.net_kwargs).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.opt = optim.Adam(self.online_net.parameters(), lr=self.lr)

        # ---- BC 锚定（信任域）----
        # 实测在线微调**不单调变好**：约 c100 达峰后 Q(HO) 持续下漂 → 采集快照
        # 不再切换 → buffer 里 HO 样本更少 → Q(HO) 进一步被压低，形成
        # **自强化坍缩**（C4：c50~c200 全 out=0.0，c250 起 ho=0；E4 同构）。
        # 对策：以 BC 预训练（或本阶段起点）的 Q 为**冻结参照**，在 TD loss 上
        # 加 ‖Q(s,·) − Q_ref(s,·)‖²，把策略约束在信任域内 —— 允许吸收干净
        # 奖励带来的改进，但禁止整体偏好漂移（尤其禁止把 HO 的价值压没）。
        # 锚定用**全动作 Q 向量**而非仅所取动作：坍缩本质是动作间**相对**偏好
        # 漂移，只锚定单动作管不住相对关系。
        self.bc_anchor_w = float(a.get("bc_anchor_w", 0.0))
        self.anchor_net = None
        self._last_anchor = None

        # ---- 潜变量路径开关 ----
        # replay_latent=False（Stage2 并行在线采集）时，即使网络具备 step/flow/unroll
        # 也走普通 obs 重放（forward(x)），PER 不存隐状态、collector 无需滚动 GPU 隐状态。
        replay_latent = bool(a.get("replay_latent", True))
        self.latent = (replay_latent
                       and hasattr(self.online_net, "step")
                       and hasattr(self.online_net, "flow")
                       and hasattr(self.online_net, "unroll"))
        mob = cfg.get("mobility") or {}
        self.dt_s = float(mob.get("dt_s", 0.1))

        dyn = dict(DEFAULT_DYNAMICS)
        dyn.update(a.get("dynamics", {}) or {})
        self.dyn = dyn
        self.dyn_enabled = bool(dyn["enabled"]) and self.latent
        self.dyn_weight = float(dyn["weight"])
        self.dyn_every = max(1, int(dyn["every"]))

        self.buffer = PrioritizedReplayBuffer(a, store_latents=self.latent)
        # ---- n 步回报（设计方案 §4.3 P2）----
        # 切换/中断的后果延迟数步才显现，n 步回报把 bootstrap 次数从 1 降到
        # 1/n。n_step=1 即原单步行为（回退开关）。折扣已折进缓冲中的 reward，
        # 每条 replay 样本保存实际 bootstrap 折扣（短窗口为 γ^K）。
        self.n_step = max(1, int(a.get("n_step", 1)))
        self.nstep = NStepAccumulator(self.n_step, self.gamma, self.buffer,
                                      store_latents=self.latent)
        self.ep_replay = EpisodeReplay(int(dyn["episode_capacity"])) if self.latent else None
        self.safety = SafetyLayer(cfg["safety"])
        self._h = None
        self._learn_step = 0
        self._last_dyn_loss = 0.0
        # ---- loss 分量（2026-09-25 新增）----
        # 存在的理由：train_ns3 的 loss_ema 记的是 learn() 返回的**总和**
        # （TD 项 + bc_anchor_w×锚定项 + 动力学项）。H9/H10 都观测到 loss_ema 从
        # ~0.4 涨到 6~9，但**无法判定**是「价值发散」还是「锚定惩罚随 Q 漂移增长」
        # —— 两者对策完全相反（前者要降 lr/步数，后者说明锚定正在起作用）。
        # 故把各分量单独留存，由调用方打日志。
        self._last_td = None        # 纯 TD 项（含 PER 重要性采样权重）
        self._last_anchor = None    # ‖Q(s,·) − Q_ref(s,·)‖²（未乘 bc_anchor_w）

    # ---------------- 决策 ----------------

    def act(self, obs: np.ndarray, info: dict, eps: float) -> int:
        """选动作。潜变量网络会顺带把隐状态推进到当前观测。"""
        if self.latent:
            if self._h is None:
                self._h = self.online_net.init_hidden(1, self.device)
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                h = self.online_net.step(self._h, x, self.dt_s)
                q = self.online_net.q(h)
            self._h = h
        else:
            with torch.no_grad():
                x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                q = self.online_net(x)

        self.last_decision = {
            "q": q[0].detach().cpu().tolist(),
            "masked_actions": torch.where(~torch.isfinite(
                policy_mask(q, self.cfg, info, self.legal_actions)[0]))[0].cpu().tolist(),
        }
        return select_action(q, self.cfg, info, eps, actions=self.legal_actions)

    def reset_latent(self):
        """在 ``env.reset()`` 之后调用：归档上一段 episode 序列并清空隐状态。

        潜状态**必须**在 episode 边界清零，否则会把上一趟列车的信道历史
        带进新 episode，等于给网络一个非法的先验信息。
        同时刷新 n 步窗口（截断 episode 的短窗口不得跨集累积奖励）。
        """
        if self.ep_replay is not None:
            self.ep_replay.end_episode()
        self.flush_nstep()
        self._h = None

    def flush_nstep(self):
        """episode 边界刷新 n 步回报窗口。

        调用时机：``env.reset()`` 之后（或 episode 循环结束时）。截断
        episode（``truncated`` 而 ``done=False``）不会自动清空窗口，遗漏
        调用会让上一集末尾的奖励混入下一集开头的轨迹。
        """
        self.nstep.flush()

    def store(self, s, a, r, s2, done):
        """写入一条转移（经 n 步累积器折叠后入 PER）。

        潜变量网络同时推进并记录 ``(h_t, h_{t+n})``——窗口末端的
        ``next_state``/``next_latent`` 一一对应，n 步目标不会错位。
        """
        if self.latent:
            if self._h is None:
                self._h = self.online_net.init_hidden(1, self.device)
            with torch.no_grad():
                x2 = torch.as_tensor(s2, dtype=torch.float32, device=self.device).unsqueeze(0)
                h2 = self.online_net.step(self._h, x2, self.dt_s)
            self.nstep.push(s, a, r, s2, done, latent=self._h, next_latent=h2)
            self._h = h2
            if self.ep_replay is not None:
                self.ep_replay.push(s)
                if done:
                    self.ep_replay.push(s2)
                    self.ep_replay.end_episode()
        else:
            self.nstep.push(s, a, r, s2, done)

    # ---------------- 学习 ----------------

    def _next_admissible(self, s2):
        """bootstrap 用 next-state 的 admissible 集（修复2：训练/部署同集）。

        部署端 ``policy_mask`` 常开的 hard shield 是观测的确定性函数 → 与
        shield 联合的 MDP 语义要求在 target 的 argmax 上应用同一集合
        （Huang & Ontañón 2020；Shields for Safe RL, CACM 2024）。
        guard 门控的软掩码（乒乓等）依赖步历史、不是纯观测函数，故**不**
        进入 bootstrap —— 它们属于运行时防护而非 MDP 动作集的一部分。
        hard shield 未启用 / 旧 14 维观测 → 返回 None，行为与历史完全一致。
        """
        return hard_admissible_from_obs(s2, self.cfg, self.n_actions)

    def _q_and_target(self, batch):
        """按网络类型计算 Q(s,a) 与 Double-Q 目标。

        ``reward`` 字段已由 :class:`NStepAccumulator` 折成 n 步折扣回报
        ``R_t^(n) = Σ_{k<n} γ^k r_{t+k}``，``next_state``/``next_latent`` 也
        已推进到窗口末端。bootstrap 使用样本保存的实际 **γ^K**，
        完整窗口 K=n，截断尾窗 K<n；真实 terminal 的折扣为零。

            y = R_t^(K) + discount · Q_target(s_{t+K}, a*)

        """
        a = batch["action"].to(self.device)
        r = batch["reward"].to(self.device)
        adm = self._next_admissible(batch["next_state"])
        if self.latent:
            h = batch["latent"].to(self.device)
            h2 = batch["next_latent"].to(self.device)
            q_sa = self.online_net.q(h).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q2 = mask_actions(self.online_net.q(h2), self.legal_actions)
                if adm is not None:
                    q2 = q2.masked_fill(~adm.to(q2.device), float("-inf"))
                a_star = q2.argmax(dim=1)
                q_next = self.target_net.q(h2).gather(1, a_star.unsqueeze(1)).squeeze(1)
        else:
            s = batch["state"].to(self.device)
            s2 = batch["next_state"].to(self.device)
            q_sa = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                q2 = mask_actions(self.online_net(s2), self.legal_actions)
                if adm is not None:
                    q2 = q2.masked_fill(~adm.to(q2.device), float("-inf"))
                a_star = q2.argmax(dim=1)
                q_next = self.target_net(s2).gather(1, a_star.unsqueeze(1)).squeeze(1)
        y = r + self._bootstrap_discount(batch) * q_next
        return q_sa, y

    def _bootstrap_discount(self, batch):
        """Replay discounts include terminal masking; legacy batches use γ^n."""
        if "discount" in batch:
            return batch["discount"].to(self.device)
        return (self.gamma ** self.n_step) * (1.0 - batch["done"].to(self.device))

    def _dynamics_loss(self):
        """多步外推预测损失：‖decoder(ODESolve(h_t, k·Δt)) − x_{t+k}‖²。

        这是潜变量 ODE 唯一能学到「真实时间演化」的监督信号：外推期间
        **不给网络任何观测**，它只能依靠 f_θ 自身积分出未来状态。
        """
        if self.ep_replay is None or len(self.ep_replay) == 0:
            return torch.zeros((), device=self.device)
        burn_in = int(self.dyn["burn_in"])
        horizon = int(self.dyn["horizon"])
        win = self.ep_replay.sample_windows(int(self.dyn["batch_size"]), burn_in + horizon)
        if win is None:
            return torch.zeros((), device=self.device)

        xs = torch.as_tensor(win, dtype=torch.float32, device=self.device)
        xs = xs.transpose(0, 1)                       # [L, B, obs_dim]
        # burn-in：带观测的 ODE-RNN 更新，取末步隐状态
        h = self.online_net.unroll(xs[:burn_in])[-1]  # [B, hidden]
        # 外推：无观测，纯动力学
        pred = self.online_net.predict(h, horizon, self.dt_s)   # [K, B, obs_dim]
        target = xs[burn_in:burn_in + horizon]
        return ((pred - target) ** 2).mean()

    @torch.no_grad()
    def dynamics_mse(self, n_windows: int = 64):
        """诊断用：返回 ``(模型外推 MSE, 持久性基线 MSE)``。

        持久性基线 = 「未来观测恒等于当前观测」(x̂_{t+k} = x_t)。信道在
        350km/h 下 1s 内 RSRP 变化可达 10dB 以上，因此模型必须显著优于
        该基线，否则说明 ODE 没学到动力学、预测式切换无从谈起。
        """
        if self.ep_replay is None or len(self.ep_replay) == 0:
            return float("nan"), float("nan")
        burn_in = int(self.dyn["burn_in"])
        horizon = int(self.dyn["horizon"])
        win = self.ep_replay.sample_windows(int(n_windows), burn_in + horizon)
        if win is None:
            return float("nan"), float("nan")
        xs = torch.as_tensor(win, dtype=torch.float32, device=self.device).transpose(0, 1)
        h = self.online_net.unroll(xs[:burn_in])[-1]
        pred = self.online_net.predict(h, horizon, self.dt_s)
        target = xs[burn_in:burn_in + horizon]
        model_mse = float(((pred - target) ** 2).mean().item())
        persist = xs[burn_in - 1].unsqueeze(0).expand_as(target)
        base_mse = float(((persist - target) ** 2).mean().item())
        return model_mse, base_mse

    def learn(self) -> float:
        if len(self.buffer) < self.batch:
            return 0.0
        batch, idxs, weights = self.buffer.sample(self.batch)
        w = weights.to(self.device)

        q_sa, y = self._q_and_target(batch)
        td = (q_sa - y).detach().abs().cpu().numpy()
        loss = (w * (q_sa - y) ** 2).mean()
        # 纯 TD 项先留档：下面还会叠加锚定/动力学项，否则事后无法分解
        self._last_td = float(loss.item())

        dyn_loss = None
        if self.dyn_enabled and self._learn_step % self.dyn_every == 0:
            dyn_loss = self._dynamics_loss()
            loss = loss + self.dyn_weight * dyn_loss
        if dyn_loss is not None:
            self._last_dyn_loss = float(dyn_loss.item())

        # BC 锚定：约束 Q(s,·) 不漂离参照策略（详见 __init__ 说明）
        if self.anchor_net is not None and self.bc_anchor_w > 0.0:
            if self.latent:
                h = batch["latent"].to(self.device)
                q_all = self.online_net.q(h)
                with torch.no_grad():
                    q_ref = self.anchor_net.q(h)
            else:
                s = batch["state"].to(self.device)
                q_all = self.online_net(s)
                with torch.no_grad():
                    q_ref = self.anchor_net(s)
            anchor_loss = ((q_all - q_ref) ** 2).mean()
            loss = loss + self.bc_anchor_w * anchor_loss
            self._last_anchor = float(anchor_loss.item())

        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.opt.step()

        self.buffer.update_priorities(idxs, td)
        self._learn_step += 1
        if self._learn_step % self.sync_steps == 0:
            self.sync_target()
        return float(loss.item())

    # ---------------- 其它 ----------------

    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def anneal_beta(self, frac: float):
        self.buffer.anneal_beta(frac)

    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: str):
        torch.save({"online": self.online_net.state_dict(), "eps": self.eps}, path)

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ck["online"])
        self.target_net.load_state_dict(ck["online"])
        self.eps = float(ck.get("eps", self.eps))

    def set_bc_anchor(self) -> bool:
        """把**当前**在线网络冻结为信任域参照（BC 预训练权重或本阶段起点）。

        应在载入 BC/上一阶段权重**之后**调用。返回是否真正启用。
        """
        if self.bc_anchor_w <= 0.0:
            return False
        self.anchor_net = copy.deepcopy(self.online_net).to(self.device)
        for p in self.anchor_net.parameters():
            p.requires_grad_(False)
        self.anchor_net.eval()
        return True
