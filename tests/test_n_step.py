"""n 步回报（设计方案 §4.3 P2）的语义与集成测试。

覆盖：
  1. 折扣求和正确（R_t^(n) = Σ γ^k r_{t+k}）
  2. 滑动窗口：每条转移仍被消费一次（样本量不因 n 步而减少）
  3. done 截断：窗口内出现 done 时不得把后续奖励混入
  4. flush：短截断窗口保留 bootstrap，使用实际 γ^K
  5. n=1 退化为单步直通（回退开关）
  6. 与 DuelingDDQNAgent 集成：n 步路径下 learn 正常、隐含状态配对正确
"""

import numpy as np
import torch

from src.utils.config import load_config
from src.agents.agent import DuelingDDQNAgent
from src.agents.replay import NStepAccumulator, PrioritizedReplayBuffer

GAMMA = 0.9


def make_buf(size=100):
    cfg = load_config("configs/default.yaml")
    cfg["agent"]["buffer_size"] = size
    return PrioritizedReplayBuffer(cfg["agent"])


def entries(buf):
    """返回缓冲中已写入的转移（按写入顺序）。"""
    n = buf.tree.n_entries
    return [buf.tree.data[i] for i in range(n)]


def s(v):
    return np.full(4, float(v), dtype=np.float32)


def make_acc(n, buf=None):
    buf = buf or make_buf()
    return NStepAccumulator(n, GAMMA, buf), buf


def test_nstep_discount_sum():
    acc, buf = make_acc(3)
    for r in (1.0, 2.0, 3.0):
        acc.push(s(0), 0, r, s(1), False)

    assert len(buf) == 1
    st, a, ret, s2, done = entries(buf)[0]
    # r0 + γ·r1 + γ²·r2 = 1 + 1.8 + 2.43
    assert np.isclose(ret, 1.0 + GAMMA * 2.0 + GAMMA ** 2 * 3.0)
    assert done is False
    # next_state 必须是窗口末端（t+n）的状态
    assert np.allclose(s2, s(1))


def test_nstep_window_slides_one_sample_per_transition():
    """滑动窗口：push N 条 → 吐出 N 条（每条转移都被消费一次）。"""
    acc, buf = make_acc(3)
    for r in range(5):
        acc.push(s(0), 0, float(r), s(1), False)
    assert len(buf) == 3          # 5 - (3-1) 条已吐出，窗口还剩 2 条
    assert len(acc) == 2
    acc.flush()
    assert len(buf) == 5          # 刷新后总量 == push 总量
    assert len(acc) == 0


def test_nstep_done_truncates_window():
    """窗口内出现 done 时，不得越过 episode 边界累积奖励。"""
    acc, buf = make_acc(5)
    acc.push(s(0), 0, 1.0, s(1), False)
    acc.push(s(1), 0, 2.0, s(2), False)
    acc.push(s(2), 0, 3.0, s(3), True)      # episode 结束

    got = entries(buf)
    assert len(got) == 3
    # 三个窗口都以 done 步结尾 → 全部 done=True，且折扣项数分别为 3/2/1
    assert all(e[4] is True for e in got)
    assert np.isclose(got[0][2], 1.0 + GAMMA * 2.0 + GAMMA ** 2 * 3.0)
    assert np.isclose(got[1][2], 2.0 + GAMMA * 3.0)
    assert np.isclose(got[2][2], 3.0)
    assert len(acc) == 0


def test_flush_bootstraps_partial_window():
    """Time-limit tails retain bootstrap with gamma raised to the actual length."""
    acc, buf = make_acc(3)
    acc.push(s(0), 0, 1.0, s(1), False)
    acc.push(s(1), 0, 2.0, s(2), False)
    acc.flush()

    got = entries(buf)
    assert len(got) == 2
    assert all(e[4] is False for e in got)
    assert np.allclose(buf.discounts[:2], [GAMMA ** 2, GAMMA])
    assert list(buf.lengths[:2]) == [2, 1]
    assert np.isclose(got[0][2], 1.0 + GAMMA * 2.0)
    assert np.isclose(got[1][2], 2.0)


def test_no_cross_episode_leak_without_flush():
    """反证：不调用 flush 时残留窗口会污染下一集 —— 这正是必须刷新的原因。"""
    acc, buf = make_acc(3)
    acc.push(s(0), 0, 100.0, s(1), False)     # 第 1 集末尾（未刷新）
    acc.push(s(9), 0, 1.0, s(9), False)       # 第 2 集开头
    acc.push(s(9), 0, 1.0, s(9), False)
    # 第一条吐出的转移把两集的奖励混在了一起（100 + γ·1 + γ²·1）
    assert np.isclose(entries(buf)[0][2], 100.0 + GAMMA + GAMMA ** 2)


def test_nstep_one_is_passthrough():
    acc, buf = make_acc(1)
    for r in (1.0, 2.0):
        acc.push(s(0), 0, r, s(1), False)
    assert len(buf) == 2
    assert len(acc) == 0
    assert np.isclose(entries(buf)[0][2], 1.0)
    assert np.isclose(entries(buf)[1][2], 2.0)


def test_agent_nstep_end_to_end():
    """集成：n_step>1 时 store/learn 正常，且缓冲样本量 = push 总量。"""
    cfg = load_config("configs/default.yaml")
    cfg["agent"]["n_step"] = 3
    cfg["agent"]["buffer_size"] = 200
    ag = DuelingDDQNAgent(cfg, torch.device("cpu"))
    assert ag.n_step == 3

    obs = np.zeros(14, dtype=np.float32)
    n = 80
    for i in range(n):
        ag.store(obs, i % 7, 1.0, obs, False)
    ag.flush_nstep()

    assert len(ag.buffer) == n
    assert np.isfinite(ag.learn())


def test_agent_target_bootstrap_uses_gamma_power_n():
    """目标式必须是 y = R^(n) + γ^n·Q(s_{t+n})，不能是单个 γ。

    写入侧的 reward 已含 n 步折扣求和，next_state 也已推进到 t+n；若目标
    侧仍用 γ 做 bootstrap，则每次递推前进 n 步却只折扣一步，等效 γ 被抬高
    到 γ^(1/n)，远期价值系统性高估。
    """
    cfg = load_config("configs/default.yaml")
    cfg["agent"]["n_step"] = 3
    cfg["agent"]["gamma"] = 0.9
    cfg["agent"]["buffer_size"] = 50
    ag = DuelingDDQNAgent(cfg, torch.device("cpu"))

    obs = np.zeros(14, dtype=np.float32)
    for _ in range(3):                      # 凑满一个窗口，奖励全 0
        ag.store(obs, 0, 0.0, obs, False)

    batch, _, _ = ag.buffer.sample(1)
    with torch.no_grad():
        # 目标网输出固定为常数 C：y 应等于 γ^n·C
        for m in ag.target_net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.zeros_(m.bias)
        ag.target_net.value[-1].bias.fill_(10.0)
        _, y = ag._q_and_target(batch)
    expect = (0.9 ** ag.n_step) * 10.0       # γ^n·C = 0.729 * 10
    assert np.isclose(float(y.item()), expect, rtol=2e-3), (float(y.item()), expect)


def test_agent_latent_next_latent_aligned_with_next_state():
    """潜变量网络：n 步窗口末端的 next_latent 必须等于 h_{t+n}（与 s_{t+n} 同轨）。"""
    from src.agents.replay import _as_flat_float
    from src.agents.latent_ode import LatentODEQNetwork

    cfg = load_config("configs/default.yaml")
    cfg["agent"]["n_step"] = 3
    cfg["agent"]["buffer_size"] = 100
    ag = DuelingDDQNAgent(cfg, torch.device("cpu"), net_factory=LatentODEQNetwork)
    assert ag.latent and ag.n_step == 3

    n = 8
    obs_seq = [np.full(14, 0.01 * i, dtype=np.float32) for i in range(n + 1)]
    # 记录每次 store 之后 agent 的隐状态：h_after[i] = 观测 x_i 之后的 h
    h_after = [ag.online_net.init_hidden(1, ag.device)]
    for i in range(n):
        ag.store(obs_seq[i], 0, 1.0, obs_seq[i + 1], False)
        h_after.append(ag._h)
    ag.flush_nstep()

    got = entries(ag.buffer)
    assert len(got) == n                    # 每条转移仍被消费一次
    # 满窗吐出的部分：next_latent 必须严格等于 h_{t+n}（而非 h_{t+1}）
    n_full = n - (ag.n_step - 1)
    for t in range(n_full):
        assert torch.allclose(_as_flat_float(got[t][5]),
                              _as_flat_float(h_after[t]), atol=1e-6)
        assert torch.allclose(_as_flat_float(got[t][6]),
                              _as_flat_float(h_after[t + ag.n_step]), atol=1e-6)
    # 尾部短窗口保留 bootstrap：latent 仍与 s_t 对齐
    for t in range(n_full, n):
        assert torch.allclose(_as_flat_float(got[t][5]),
                              _as_flat_float(h_after[t]), atol=1e-6)
        assert got[t][4] is False
        assert np.isclose(ag.buffer.discounts[t], ag.gamma ** (n - t))


def test_interleaved_workers_and_episode_boundaries():
    from src.envs.parallel_collector import CollectorReplayWriter, EPISODE_END
    from types import SimpleNamespace
    buf = make_buf()
    writer = CollectorReplayWriter(SimpleNamespace(
        latent=False, n_step=3, gamma=GAMMA, buffer=buf))
    for t in range(3):
        for wid, reward in ((0, 1.0), (1, 100.0)):
            assert writer.consume((wid, (s(wid * 10 + t), 0, reward,
                                         s(wid * 10 + t + 1), False)))
    assert np.allclose([e[2] for e in entries(buf)],
                       [1 + GAMMA + GAMMA ** 2, 100 * (1 + GAMMA + GAMMA ** 2)])
    writer.consume((0, EPISODE_END))
    assert len(writer.accumulators[1]) == 2
    writer.consume((0, (s(50), 0, 7.0, s(51), True)))
    writer.consume((1, EPISODE_END))
    assert len(buf) == 7
    assert entries(buf)[4][2] == 7.0
    assert entries(buf)[4][4] is True
    assert np.allclose(entries(buf)[5][3], s(13))
    assert np.allclose(buf.discounts[:7],
                       [GAMMA ** 3, GAMMA ** 3, GAMMA ** 2, GAMMA, 0,
                        GAMMA ** 2, GAMMA])


def test_worker_abort_discards_only_pending_windows():
    from src.envs.parallel_collector import CollectorReplayWriter, EPISODE_ABORT, EPISODE_END
    from types import SimpleNamespace
    buf = make_buf()
    writer = CollectorReplayWriter(SimpleNamespace(
        latent=False, n_step=3, gamma=GAMMA, buffer=buf))
    for _ in range(3):
        writer.consume((0, (s(0), 0, 100.0, s(1), False)))
    writer.consume((1, (s(10), 0, 5.0, s(11), False)))
    writer.consume((0, EPISODE_ABORT))
    writer.consume((0, (s(20), 0, 1.0, s(21), False)))
    writer.consume((0, EPISODE_END))
    writer.consume((1, EPISODE_END))
    assert np.allclose([e[2] for e in entries(buf)],
                       [100 * (1 + GAMMA + GAMMA ** 2), 1, 5])
    writer.consume((2, (s(30), 0, 999.0, s(31), False)))
    writer.discard()
    assert not writer.accumulators
    assert len(buf) == 3


def test_real_terminal_zero_discount_and_single_step_compatibility():
    for n in (1, 3):
        acc, buf = make_acc(n)
        acc.push(s(0), 0, 1.0, s(1), False)
        acc.push(s(1), 0, 2.0, s(2), True)
        assert list(buf.lengths[:2]) == ([1, 1] if n == 1 else [2, 1])
        assert np.allclose(buf.discounts[:2], [GAMMA if n == 1 else 0, 0])
        batch, _, _ = buf.sample(2)
        assert "discount" in batch and "n_steps" in batch
    # Direct buffer users retain their original one-step semantics and tuple shape.
    buf = make_buf()
    buf.add(s(0), 0, 0.0, s(1), False)
    assert len(entries(buf)[0]) == 5
    assert np.isclose(buf.discounts[0], buf.gamma)


def test_replay_discount_metadata_tracks_ring_overwrite():
    buf = make_buf(size=2)
    buf.add(s(0), 0, 0, s(1), False, discount=GAMMA ** 3, n_steps=3)
    buf.add(s(1), 0, 1, s(2), True, n_steps=2)
    buf.add(s(2), 0, 2, s(3), False, discount=GAMMA, n_steps=1)
    assert np.allclose(buf.discounts, [GAMMA, 0])
    assert list(buf.lengths) == [1, 2]
    batch, slots, _ = buf.sample(2)
    assert np.allclose(batch["discount"].numpy(), buf.discounts[slots])
    assert np.array_equal(batch["n_steps"].numpy(), buf.lengths[slots])


def test_dqn_short_truncation_target_and_legacy_batch():
    cfg = load_config("configs/default.yaml")
    cfg["agent"].update(n_step=3, gamma=GAMMA)
    ag = DuelingDDQNAgent(cfg, torch.device("cpu"))
    obs = np.zeros(ag.obs_dim, dtype=np.float32)
    ag.store(obs, 0, 1.0, obs, False)
    ag.store(obs, 0, 2.0, obs, False)
    ag.flush_nstep()
    with torch.no_grad():
        for p in ag.target_net.parameters():
            p.zero_()
        ag.target_net.value[-1].bias.fill_(10)
    batch, _, _ = ag.buffer.sample(2)
    _, y = ag._q_and_target(batch)
    assert torch.allclose(y, batch["reward"] + batch["discount"] * 10)
    assert sorted(batch["n_steps"].tolist()) == [1, 2]
    del batch["discount"]
    _, y = ag._q_and_target(batch)
    assert torch.allclose(y, batch["reward"] + GAMMA ** 3 * 10)


def test_collector_emits_abort_before_retry(monkeypatch):
    import queue
    import threading
    from types import SimpleNamespace
    from src.envs import parallel_collector as pc

    collector = pc.ParallelCollector.__new__(pc.ParallelCollector)
    collector.cfg = {"seed": 0}
    collector.stop_event = threading.Event()
    collector.policy = "a3"
    collector._snapshot = None
    collector.family_pool = []
    collector.outbox = queue.Queue()
    collector._stat_lock = threading.Lock()
    collector.collected = collector.episodes = collector.failures = 0
    collector.guard = SimpleNamespace(observe=lambda *args: None)
    collector._act = lambda *args: (0, "a3")
    monkeypatch.setattr(pc.time, "sleep", lambda _: None)

    class Env:
        episode = 0
        def reset(self, **kwargs):
            self.episode += 1
            self.steps = 0
            return s(self.episode * 10), {}
        def step(self, action):
            self.steps += 1
            if self.episode == 1 and self.steps == 2:
                raise RuntimeError("injected step failure")
            if self.episode == 2:
                collector.stop_event.set()
            return s(self.episode * 10 + self.steps), float(self.episode), False, self.episode == 2, {}
        def close(self):
            pass

    collector._run_one(Env(), SimpleNamespace(), 0)
    messages = []
    while not collector.outbox.empty():
        messages.append(collector.outbox.get_nowait())
    assert len(messages) == 4
    assert messages[1] == (0, pc.EPISODE_ABORT)
    assert messages[3] == (0, pc.EPISODE_END)
    assert collector.failures == 1
    buf = make_buf()
    writer = pc.CollectorReplayWriter(SimpleNamespace(
        latent=False, n_step=3, gamma=GAMMA, buffer=buf))
    for message in messages:
        writer.consume(message)
    assert len(buf) == 1
    assert entries(buf)[0][2] == 2
    assert entries(buf)[0][4] is False
    assert np.isclose(buf.discounts[0], GAMMA)


def test_sac_short_truncation_and_terminal_targets():
    from src.agents.sac_agent import DiscreteSACAgent
    cfg = load_config("configs/default.yaml")
    cfg["agent"].update(n_step=3, gamma=GAMMA, batch_size=3, bc_anchor_w=0)
    ag = DiscreteSACAgent(cfg, torch.device("cpu"))
    obs = np.zeros(ag.obs_dim, dtype=np.float32)
    ag.store(obs, 0, 1.0, obs, False)
    ag.store(obs, 0, 2.0, obs, False)
    ag.flush_nstep()
    ag.store(obs, 0, 3.0, obs, True)
    with torch.no_grad():
        for net in (ag.online_net, ag.online_net2, ag.target_net, ag.target_net2):
            for p in net.parameters():
                p.zero_()
        for net in (ag.target_net, ag.target_net2):
            net.value[-1].bias.fill_(10)
    captured = []
    ag.buffer.update_priorities = lambda idxs, td: captured.extend(zip(idxs, td))
    v_next = 10 + ag.alpha * np.log(ag.n_actions)
    assert np.isfinite(ag.learn())
    assert len(captured) == 3
    for slot, td in captured:
        expect = entries(ag.buffer)[slot][2] + ag.buffer.discounts[slot] * v_next
        assert np.isclose(td, expect, rtol=1e-5)
