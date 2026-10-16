# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

High-speed rail RL-based intelligent network handover decision system. A Dueling-DDQN
agent (MLP / Neural-ODE / **Hybrid ODE+MLP** / **Latent ODE-RNN**) learns
handover/dual-connectivity decisions under the 3GPP TR 38.901 RMa channel model.
The 3GPP A3 event-triggered handover is the baseline. Target scenario: 300-350 km/h
trains.

Two backends share one frozen contract (14-dim obs, 7 actions, identical reward):
- **Python** (`src/envs/handover_env.py`) — cheap, for stage-1 behaviour cloning and ablations
- **NS-3 LENA** (`src/envs/ns3_env.py` + `ns3/hsr-rl.cc`) — real packet-level simulation

Training is a **two-stage pipeline**: Python behaviour cloning (A3 teacher) → NS-3 online
fine-tuning with 6 parallel collectors.

Authoritative docs:
- `完整技术方案.md` — the original spec
- **`docs/建模与训练完整方案.md` — the current design of record** (equations, MDP, training
  methods, recommended final architecture). Where this file and that doc disagree, that doc wins.

## Commands

### Local (Windows, dev / tests only)

```bash
conda activate <你的环境>
# or invoke directly:
<PYTHON>   # 例：python（建议 conda 环境） -m pytest tests/ -q
```

### Cloud (Hygon DCU) — the real training environment

- SSH: `ssh -p 65032 -i ~/.ssh/<HPC_SSH_ALIAS>_<HPC_ACCOUNT> <HPC_ACCOUNT>@<HPC_HOST>`
  (config: `.claude/ssh-config.toml`; key expires 2026-09-29)
- Login node `<LOGIN_NODE>` has **no DCU**; /work is shared with the container, so the
  `<LOGIN_NODE>` → bridge mailbox (`.claude/agent_bridge.sh`) is the usual way to run things
  inside the DCU container. Bridge dir: `/work/home/<HPC_ACCOUNT>/.agent_bridge/`.
- Repo on cloud: `/work/home/<HPC_ACCOUNT>/highspeed-rl/highspeed-rl`
- Python: `/opt/conda/bin/python` (self-installed packages are lost on container rebuild →
  run `scripts/cloud/bootstrap_container.sh` first)
- NS-3: `/work/home/<HPC_ACCOUNT>/ns3root/ns-3.43`, wrapper `/work/home/<HPC_ACCOUNT>/ns3root/run_hsr.sh`

```bash
# container self-heal (deps / imports / bridge restart)
bash scripts/cloud/bootstrap_container.sh

# --- Stage 1: behaviour cloning (pure Python, minutes) ---
/opt/conda/bin/python -m scripts.pretrain_bc --config configs/bc_ns3.yaml \
    --net hybrid --n-demo 30 --epochs 15 --out checkpoints_ns3/bc_init.pt

# --- Stage 2: NS-3 online fine-tuning (6 collectors + DCU learning) ---
export PYTHONPATH=. HIP_VISIBLE_DEVICES=0
setsid bash -c '/opt/conda/bin/python -m scripts.train_ns3 --config configs/ns3.yaml \
    --device cuda --net hybrid --init checkpoints_ns3/bc_init.pt \
    ; echo "STAGE2_EXIT=$?" >> logs/stage2.log' >> logs/stage2.log 2>&1 </dev/null &

# --- Evaluation (paired same-seed A3 vs RL; prints action / action_source / alpha) ---
/opt/conda/bin/python -m scripts.eval_ns3 --config configs/eval_tunnel.yaml \
    --baseline a3 --n-eval 4 --seed 900000
/opt/conda/bin/python -m scripts.eval_ns3 --config configs/eval_tunnel.yaml \
    --net hybrid --resume checkpoints_ns3/latest.pt --n-eval 4 --seed 900000
```

**所有训练/仿真必须在 DCU 容器内执行**（2026-09-22 硬规则）。登录节点 `<LOGIN_NODE>`
（128 核）**没有 DCU**，且 GLIBC/GLIBCXX 版本低于 NS-3 构建环境 —— 在那里直接跑
`ns3.43-hsr-rl` 会报 `GLIBCXX_3.4.29 not found / GLIBC_2.32 not found`；直接 ssh
执行的命令都落在登录节点，必须走 `deploy_container.ps1`（kubectl exec）或 bridge。
每个 `cmd_*.sh` 开头应 `source /work/home/<HPC_ACCOUNT>/container_guard.sh`，它会校验
主机名（`<POD_GROUP>-*`）、hy-smi 可见性与 NS-3 运行库，不在容器即退出。

**DCU 利用率必须被观测**：实测训练中 DCU 利用率 ≈0%（瓶颈是 NS-3 仿真/锁步），
`scripts/train_ns3.py --dcu-sample <秒>` 会后台采样 `hy-smi` 并在结束打印
mean/max。长期 <5% 说明卡在等数据 → 对策是**提高采集并行度 / batch /
learn_steps_per_cycle**，而不是加大网络。当前容器**只可见 1 张** DCU（`device_count=1`），
故 `--eval-device cuda:1` 的隔离评估实际不生效、会回退同卡。

**Bridge discipline** (`.claude/session_logs`-style `session_logs/cmd_*.sh` templates):
never `pkill -f train_ns3|eval_ns3|hsr-rl` (it kills other long jobs); a ~320 s silent
eval window is normal; long jobs need `setsid` + a log under `/work` + a `*_EXIT=$?` line.

**torch thread cap (DCU pitfall)**: the container exposes 32 cores and torch opens that many
intra-op threads by default; with this project's tiny nets/batches it causes **~100× slowdown**
(measured on the latent-ODE loop: 32 threads 307 s vs 4 threads 2.9 s) and made the full pytest
suite look "hung" for 26 min. `src/__init__.py` now caps it automatically (≤8 normally, ≤4 when
≥16 cores) via `src/training/device.py::configure_torch_threads`; override with
`export HSR_TORCH_THREADS=4`. After the cap: cloud full suite = **79 passed in ~184 s**.

```bash
# full cloud regression (pytest-timeout is installed by bootstrap; 300 s/test guard)
PYTHONPATH=. /opt/conda/bin/python -m pytest tests/ -q --timeout=300 --durations=10
```

**NS-3 rebuild (container images differ!)**: the DCU container image changed from
gcc-10 to gcc-11 mid-project, which invalidated `cmake-cache/` (`/usr/bin/gcc-10`
no longer exists). Fix without a full reconfigure: back up the working binary,
then `sed -i 's|gcc-10|gcc-11|g; s|g++-10|g++-11|g'` over `cmake-cache/CMakeCache.txt`
(and `cmake-cache/CMakeFiles/**`), then `./ns3 build` — it recompiles and links
incrementally (~6 min, 1013 steps). Belt-and-braces alternative: `rm -rf cmake-cache`
+ `./ns3 configure` + `./ns3 build`. Real binary used by the env is
`ns3root/ns-3.43/build/scratch/ns3.43-hsr-rl` via the `ns3root/run_hsr.sh` wrapper
(that path is what `configs/*.yaml`'s `ns3.binary` points to).

## Reward & observation contract (path A, validated 2026-09-21)

**Reward is coverage-dominated, not throughput-dominated.** Measured on the NS-3
domain: throughput is near-unlearnable (CV≈1.12, 45% of steps ≈0, and it drifts
with unobserved load — same SINR gives 2.5× throughput differences between
families; predictability r≈0.1-0.3). RSRP/SINR is learnable (extrapolation MAE
1.31 dB vs 4.37 dB persistence, cross-family ×1.00-1.46). So:
`w_coverage=0.15` (serving-SINR margin vs RLF threshold, ±0.15/step) and
`w_gap=0.05` (neighbour clearly better but not handed over) lead;
`w_throughput=0.15` (demoted from 0.5); `w_outage=0.3` (0.6).

**G3 gate before any Stage2 run**: `python -m scripts.reward_sanity --config <cfg>
--families normal --n-eps 3` must print `[G3 PASS]` (teacher reward > 0, teacher −
always-stay ≥ 20, teacher > random). Measured: teacher +30.1±4.6 / stay −69.5 /
random −71.0 — the discriminator is what makes the reward learnable at all.

**`best.pt` + early stopping are mandatory.** Online fine-tuning is **not
monotone**: eval holds `out=0.0` for ~100-200 cycles and then collapses to
`ho=0` (self-reinforcing: Q(HO) drifts down → collection snapshot stops handing
over → fewer HO samples → Q(HO) drops further). `train_ns3.py` now saves
`best.pt` by `(outage, −reward)` and stops after `training.early_stop_patience`
(default 4) evaluations without improvement. Never evaluate `latest.pt`.

**BC anchor trust region** (`agent.bc_anchor_w`, default 0.0 → off). Adds
`w·‖Q(s,·) − Q_ref(s,·)‖²` on the **full action vector** (anchoring only the
taken action cannot stop relative-preference drift, which is what collapse is).
The anchor must be a policy that is **already valid on the target distribution**
(teacher clone). Measured: anchor to the previous-stage snapshot (F4, w=0.1) was
*worse* than no anchor (all four evals `out=1.0`); anchor to the teacher clone
(F5, w=0.02) never collapsed (5/5 evals `out=0.0`) and matched E4's 24/24.

**Load observability** (`env.load_obs`, requires the rebuilt NS-3). Reuses the
original 5 reserved bytes in `HsrState` (protocol stays 56 B): `load_serving`,
`load_nbr[3]` (attached background-UE counts per candidate cell — the load proxy),
`in_tunnel`. With it, `obs_dim` goes 23 → 28. Count by `servingCell`, **not** the
`attached` flag (the flag depends on trace callbacks that mostly never fire for
background UEs: 16 UEs produced counts of 0-1). Measured on the clustered
scenario: `load[srv=7 n1=1 …]` with throughput collapsing to 5-14 Mbps — the
"serving cell congested, neighbour idle" signal that a load-blind rule cannot see.

## Headline result (G4, 2026-09-21): RL beats the A3 rule on every family

Train: 4 families × randomized, 28-dim obs (load-observable), BC init
(acc 99.4%, HO recall 100%, Q-scale calibrated), `lr=1e-4`,
`learn_steps_per_cycle=8`, 10% on-policy / 90% teacher, anchor w=0.02.
**12/12 inline evals `out=0.0` (c50→c600) — no collapse at all.**

Paired evaluation, same seeds and scenarios, 4 episodes/family
(`scripts/baselines_ns3.py --families normal,tunnel,load,congested`):

| family | A3 tput | RL tput | gain | A3 reward | RL reward | gain |
|---|---|---|---|---|---|---|
| normal | 28.53 | 30.09 | +5.4% | 37.63 | 42.19 | +12.1% |
| tunnel | 28.75 | 30.55 | +6.3% | 34.50 | 41.16 | +19.3% |
| load | 19.88 | 21.44 | +7.8% | 39.12 | 46.03 | +17.7% |
| congested | 14.36 | 16.05 | +11.8% | 22.97 | 32.48 | +41.4% |

Survival is 1.00 for both arms everywhere and ping-pong is 0, so the gain is not
bought with dropped calls. `Fixed(1s)` is a useful negative control: highest raw
throughput (22.98 on congested) but `outage=1.0` — judge by survival, not
throughput alone.

**Mechanism audit — the gain is NOT load balancing** (`scripts/load_audit.py`,
same seeds, 3 episodes/family). The gain grows with load severity, which *looked*
like the load dimensions being used; the direct audit refutes that:

| family | metric | A3 | RL | Δ |
|---|---|---|---|---|
| load | load_time_mean | 0.408 | 0.408 | 0.000 |
| load | load_after_ho | 0.417 | 0.417 | 0.000 |
| congested | load_time_mean | 1.478 | 1.473 | −0.005 |
| congested | load_after_ho | 1.364 | 1.273 | −0.091 |
| congested | throughput | 14.34 | 15.88 | +10.8% |

So RL's handover *targets* are marginally lighter on `congested` (−6.7%) but the
**time-averaged** load it sits under is unchanged — the error signal simply is
not there. The +5%~+12% throughput is therefore mostly better **timing / target
SINR quality**, not load steering. Two things are missing before real load
balancing can pay off: (1) an explicit load term in the reward (today the reward
has coverage + throughput + best-neighbour-gap only — throughput rewards low load
only very indirectly), and (2) scenarios where an *idle neighbour is adjacent to a
congested cell along the trajectory*, so the choice is actually available.
**Never infer a mechanism from a monotone trend; measure the mechanism.**

**Explicit load reward: it works, but it costs lives — do not ship it as-is.**
Added `w_load` + `load_gap_penalty()` (penalty for sitting on a cell more loaded
than the least-loaded *measured* neighbour, deadband 1 UE, scale 4 UE). Measured
on `congested`, same seeds, 3 episodes (`scripts/load_audit.py`):

| arm | load_time_mean | survival | ho/ep | tput |
|---|---|---|---|---|
| A3 | 1.478 | 1.00 | 7.33 | 14.34 |
| G4 (`w_load=0`) | 1.473 | 1.00 | 7.33 | 15.88 |
| **H5 (`w_load=0.02`)** | **1.022 (−31%)** | **0.333** | 2.67 | 14.79 |

So the term does produce genuine load reduction, but the way it buys it is
**handing over less** (7.33 → 2.67 per episode), and in this topology that means
RLF. `w_load=0.08` (H4) collapses outright (`ho=0`, `out=1.0` at c200). Root
cause is topological, not a weight-tuning problem: with 1000 m macro spacing the
lighter neighbour is also much weaker, so "escape congestion" and "keep coverage"
are directly opposed — congestion relief is not safely purchasable here. To get
load balancing *and* survival, change the scenario first (denser eNBs / an
explicit hot-cell + cold-neighbour pair along the track), then re-tune `w_load`.

### Local smoke (no NS-3 needed)

```bash
python -m scripts.train_ns3 --config configs/default.yaml --mock --net hybrid --n-episodes 3
```

## Architecture

### MDP

**State** `s_t ∈ ℝ¹⁴`, normalized to [0,1]:

| dim | meaning | dim | meaning |
|---|---|---|---|
| 0/1 | serving RSRP / SINR | 8 | neighbor available |
| 2/3 | strongest / 2nd neighbor RSRP | 9 | recent HO frequency (60-step window) |
| 4/5 | speed / heading cosine | **10/11** | **serving / strongest-neighbor RSRP trend (1 s forward) ** |
| 6/7 | dist to serving BS / tx buffer (dynamic) | **12/13** | **RSRP diff (A3 criterion) / dual-connectivity gain potential** |

- dims 14/15 (optional, `env.obs_load_features: true`, `obs_dim: 16`): serving-cell
  congestion / neighbor-minus-serving congestion difference. **Python backend only** —
  NS-3 has no load model, so the NS-3 contract stays 14-dim.
- Trend features come from a **read-only** look-ahead measurement (current position +
  10 steps ahead), protected by `channel.snapshot_state()/restore_state()` so the shadow
  AR(1) and RNG state are not advanced. **The NS-3 backend approximates this with a
  backward difference** (known deviation — see `docs/ns3_next_steps.md`).

**Action** `A = {a0..a6}`: stay / HO→strongest / HO→2nd / dual(primary+strongest) /
dual(primary+2nd) / HO→3rd / triple(primary+2 neighbors). The env tracks the top-3
neighbors internally; obs exposes only top-2 RSRP (a5 references the 3rd).

**Reward** `R = 0.5·R_tp + 0.3·R_ho + 0.1·R_delay + 0.1·R_stab + r_potential + r_outage`:

- throughput: `log1p(Σ over active links of Shannon)`, scaled by the handover
  interruption window `g(τ) = 1 − A_HO·frac·sin(πτ)`
- handover: `−3·𝟙{HO} − 5·𝟙{pingpong}` (pingpong is an **event**, not a state)
- delay: `−interrupt_ms/100`
- stability: `+0.5·𝟙{active_links>1}`
- `r_potential`: linear penalty inside 500 m before a tunnel mouth (tunnel potential well)
- `r_outage`: linear ramp for serving SINR ∈ [−6, −2] dB, doubled below the RLF threshold
- RLF termination is penalised **per remaining step** — otherwise the agent learns
  "never hand over" to dodge the per-HO penalty.

### Channel (3GPP TR 38.901 RMa)

- RMa LOS/NLOS path loss with breakpoint `d_BP = 2π·h_BS·h_UT·f_c/c`; `PL = max(PL_LOS, PL_NLOS)`
- Spatially correlated shadow fading `ρ(Δx)=exp(−Δx/d_corr)`, `d_corr=37 m (LOS)/120 m (NLOS)`,
  `σ_SF=4.1/6.9 dB`, per-BS first-order distance-AR along the track
- LOS probability `exp(−(d_2D−10)/1000)` with 100 m spatial correlation (no per-step flip)
- Fast fading: Jakes, N=34 rays, **independent random phase per BS** (otherwise the
  fading is a common-mode term and cancels in the RSRP difference)
- Penetration loss `28 dB / cos θ`, `cos θ` floored at 0.7 (40 dB cap)
- **Antenna gains** are an explicit link-budget term (`bs_antenna_gain_dbi=15`,
  `ue_antenna_gain_dbi=3`) — without them RSRP is ~18 dB too low
- SINR counts only the **k strongest interferers** (`channel.n_interferers=3`), modelling
  the along-track directional antenna pattern; all-cell interference makes the scenario
  interference-limited and invalidates A3

### Mobility & topology

- Linear track, constant speed (83.3 m/s = 300 km/h for `default.yaml`, 97.2 m/s = 350 km/h
  for `hsr_load/hsr_stress`), step 100 ms
- BSs along the track, 500-2000 m random spacing, 35 m height, 50-100 m perpendicular offset;
  first BS 100-300 m from the origin (protection against a blind start)
- `--earfcn 100` (band1, 2.1 GHz) is the current LENA limit; the original 3.5 GHz n78
  target would need a band42 patch

### Safety layer (three levels, design doc §2.4)

`src/agents/safety.py` + inside `env.step`:

1. **Action masking** (before argmax): target RSRP < −110 dBm, more than 6 dB below
   serving, or a HO within the last 2 steps → Q set to −∞.
2. **Preventive fallback** (in `env.step`): serving RSRP < −105 dBm and a neighbor 4 dB
   better and RL did not act → replaced with a handover action (`safety_preempt`).
3. **RLF fallback** (L3-filtered): serving SINR < −5 dB for `fallback_persist_steps`
   consecutive steps → forced A3 to the strongest legal neighbor (`safety_a3`).

`info["action_source"] ∈ {rl, masked_dual, masked_weak, safety_preempt, safety_a3}` makes
every substitution auditable. **The buffer must store `info["action_executed"]`, not the
policy's raw output** — otherwise Q learns "dual actions cause stay".

### Agents

Networks live in `src/agents/` behind one factory signature
`(obs_dim, n_actions, hidden, **net_kwargs)` so `DuelingDDQNAgent` can swap them freely.
`net_kwargs` comes from the top-level `net_kwargs` config section.

| network | structure | role |
|---|---|---|
| `DuelingDQN` (`networks.py`) | shared 128×2 → V + A, `Q = V + A − mean(A)` | baseline |
| `NeuralODEDuelingDQN` (`neural_ode.py`) | RK4 on pseudo-time τ∈[0,1] | ablation only (equivalent to a fixed-step residual MLP) |
| **`HybridDuelingDQN`** (`hybrid_network.py`) | ODE branch + MLP branch + learned soft routing `h=(1−α)h_ODE+α h_MLP` | **recommended online policy** |
| `LatentODEQNetwork` (`latent_ode.py`) | `h⁻=ODESolve(h_{t−1},Δt)` on the **real** time axis, `h=GRU(e(x),h⁻)` | predictive upgrade path |

Learning (`agent.py`): Double-Q, PER sum-tree (`α=0.6`, `β: 0.4→1`), gradient clipping 10,
target sync every 100 learn steps, ε 1.0→0.01 (×0.995), and optionally the latent
**dynamics auxiliary loss** `‖decoder(ODESolve(h_t, kΔt)) − x_{t+k}‖²` with weight 0.5
evaluated every 4 learn steps (diagnose against a persistence baseline via
`agent.dynamics_mse()`).

**n-step returns** (`agent.n_step`, `NStepAccumulator` in `replay.py`): stored rewards are
n-step discounted returns and `next_state`/`next_latent` advance to `t+n`, so the target
`y = R_n + γ·Q(s_{t+n})` keeps its form. `n_step=1` disables it. **Episode boundaries must
flush the window** — single-process loops call `agent.flush_nstep()`, the parallel
collector emits an `EPISODE_END` sentinel.

### Baselines (for ablation/comparison)

A3Operator (L3 filter + hysteresis + TTT + Doppler adaptation), idealised A3, greedy-strongest,
fixed-interval, tabular Q (3-bin discretisation), vanilla DQN.

### Evaluation metrics

Primary: avg throughput, HO/min, HO failure rate, p5 delay, outage probability.
Auxiliary: dual-connect usage, pingpong rate, avg SINR, `action_source` mix, hybrid routing
α (mean/σ, aligned to HO steps).

## Key Source Files

| File | Role |
|------|------|
| `src/envs/channel_3gpp.py` | RMa path loss, shadow fading, Jakes, penetration, antenna gains |
| `src/envs/mobility.py` | Train trajectory + BS topology + tunnel segments |
| `src/envs/handover_env.py` | Python-backed Gymnasium env (14/16-dim obs) |
| `src/envs/ns3_env.py` | NS-3-backed env, byte-aligned with `ns3/hsr-common.h` (56 B state / 8 B action) |
| `src/envs/parallel_collector.py` | N parallel NS-3 subprocess collectors → shared PER |
| `src/agents/networks.py` | DuelingDQN (MLP) |
| `src/agents/hybrid_network.py` | Hybrid ODE+MLP DuelingDQN with soft routing |
| `src/agents/latent_ode.py` | Latent ODE-RNN (real-time dynamics + multi-step prediction head) |
| `src/agents/neural_ode.py` | `ODEFunc` / `rk4_step` / `integrate` primitives |
| `src/agents/replay.py` | Prioritized replay (sum-tree) + `NStepAccumulator` |
| `src/agents/seq_replay.py` | Episode-level observation buffer for the dynamics loss |
| `src/agents/agent.py` | `DuelingDDQNAgent`: Double-Q, PER, n-step, latent dynamics loss |
| `src/agents/safety.py` | `SafetyLayer`: action masking + A3 fallback |
| `src/agents/baselines.py` | A3 / A3Operator / greedy / fixed / tabular-Q / vanilla-DQN |
| `src/training/device.py` | Device detection (`cuda` on DCU via DTK), seed, AMP (off) |
| `configs/default.yaml` | Python-only baseline scenario (3.5 GHz, no tunnels) |
| `configs/{ns3,bc_ns3,hsr_load,hsr_stress,eval_*}.yaml` | NS-3 training / BC / stress / evaluation scenarios |
| `scripts/train_ns3.py` + `scripts/pretrain_bc.py` | Stage 2 / Stage 1 |
| `scripts/eval_ns3.py` / `baselines_ns3.py` | Paired A3-vs-RL evaluation / baselines over NS-3 |
| `scripts/cloud/bootstrap_container.sh` | Container self-heal after platform rebuild |
| `ns3/hsr-rl.cc` + `ns3/hsr-common.h` | LENA scenario, FIFO lock-step IPC, tunnel model, tunnel micro eNBs |

## Conventions

- **Device priority**: CUDA > CPU. On the cloud DCU the "cuda" backend is provided by DTK,
  exposed via `HIP_VISIBLE_DEVICES` (not `CUDA_VISIBLE_DEVICES`); AMP stays off (tiny nets).
- **State indices**: the 14-dim layout above — do not silently change. Adding load features
  must be gated by `env.obs_load_features` so the NS-3 contract stays 14-dim.
- **Config nesting**: channel params under `cfg["channel"]`, mobility under `cfg["mobility"]`;
  network constructor args under top-level `net_kwargs` and must match between Stage 1
  (`bc_ns3.yaml`) and Stage 2 (`ns3.yaml`), otherwise the BC checkpoint will not load.
- **Action honesty**: NS-3/LENA has no EN-DC, so dual/triple (3/4/6) are masked on both the
  policy side (`safety.mask_dual`) and the env side; keep both in sync.
- **Test data**: `tests/` uses synthetic fixtures only (no real OpenCelliD data).
- **Contract stability**: `handover_env` and `ns3_env` must stay interchangeable — keep the
  observation/action/info contract aligned when touching either.

## 生产产物与已验证边界

> **更新（2026-10-07 定稿）：生产产物 = `checkpoints_stageH13/best.pt`。**
> H13 = H11 + 两个奖励定价键（`w_gap` 0.05→0.3、`weak_coverage_sinr_db`
> -2.0→0.0），修复 H11 的 load 族掉话（16 集 survival 0.75→1.00，死因 no_ho_*
> 清零）并修复 M3/M8 吞吐缺口；对前代产物 H8 严格占优（六族 Δ吞吐 持平或
> 反超：normal +1.55 / tunnel +1.83 / load +1.62 / congested +1.96 /
> hotcold +1.77 Mbps，全族存活 1.00、ho 无反弹；M 矩阵吞吐 ±1% 持平，
> M8 p95 89→38.9 ms）。已知代价：相对 H11 的 p95 尾部优势回归 H8 档位
> （数据表明该优势与 H11 的欠切换掉话同源，详见下方 H13 节的完整验收
> 记录）。对比口径与数据表见「## H13：当前最优候选（2026-10-07 验收）」。
> **此后任何改动都应与 H13 做**同 seed 配对**比较。**

### 前代：H8（2026-09-21 定稿，checkpoints_stageH8/best.pt）

课程收尾：从 G4 出发 + hotcold 低权重入池；28 维负载可观测 / 锚定教师克隆 /
覆盖主奖励。多种子配对（3~4 seed × 2~3 集，bootstrap 95% CI）全族画像：

| 族 | Δ吞吐 | 增益 | 95% CI | 存活 A3/RL |
|---|---|---|---|---|
| normal | +1.55 | +5.0% | [+1.55,+1.55] | 1.00 / 1.00 |
| tunnel | +2.01 | +6.5% | [+1.85,+2.10] | 1.00 / 1.00 |
| load | +1.59 | +7.9% | [+1.55,+1.61] | 1.00 / 1.00 |
| congested | +2.01 | +12.6% | [+1.85,+2.14] | 1.00 / 1.00 |
| hotcold | +1.71 | +13.2% | [+1.68,+1.73] | 1.00 / 1.00 |
| fading（真实快衰落） | +1.76 | +6.5% | [+1.69,+1.90] | 1.00 / 1.00 |

六族全部显著为正、存活零代价、ping-pong=0。相对上一版产物 G4（四族
+5.2%~+11.5%）在每一族都持平或更好，并新增 hotcold（冷热对）与 fading
（Jakes 快衰落）两族覆盖。此后任何改动都应与它做**同 seed 配对**比较。

## 协议版本与容量测评（2026-09-22 起，P0 进行中）

`HsrState` 三版：HSR1=56B → HSR2=72B → **HSR3=84B**（magic 同步升级，握手
失败即显式报错）。HSR3 修正 HSR2 的三处口径缺陷（`ns3/hsr-common.h` 有完整说明）：

1. `tp_agg_mbps` 是**全网**聚合，`n_ue_cell` 是**单小区**附着数 → 用前者算
   人均会高估约 n_cells 倍。人均速率必须用新增的 `tp_agg_cell_mbps`。
2. `n_ue_cell` 只统计背景 UE（计数器只在 BgUe 回调里增减），控车未计入 →
   C++ 侧显式 +1。
3. `tp_min_ue_mbps`（逐间隔最小）在 1000 用户时被"本间隔没被调度"的瞬时 0
   主导 → 改为**累计口径**的 `tp_ue_p05/p50` + `jain` 公平性指数。

`scripts/capacity_probe.py` 同步支持：`--fixed-scene`（只扫 N，消除场景抽样
噪声）、`--attach-stagger`（错峰接入，避免 1000 UE 的 RRC 建立风暴）、
`--skip-lost-pkts`（省 FlowMonitor CPU）、每任务回报 wall_s/rss_mb 与
`load_sat_ratio`（负载观测饱和告警）。

**100~1000 用户的可行性由 wall_s/rss_mb 外推决定**（NS-3 单线程仿真，墙钟
随 UE 数超线性增长；`ns3.timeout_s=600` 与容器内存可能先于容量极限成为拦路
石）——`session_logs/cmd_0151.sh` 就是先标定 32/64/128/256 再外推。

## 下一阶段 H9：继续强化 RL（配方已生成，待 P0 门禁）

主线 = **规模外扩 + 负载可分辨**（`session_logs/make_stage_cfg.py H9`）：训练池
背景 UE 由 ≤32 推到 load[16,48]/congested[48,96]/hotcold[64,96]，`load_norm_denom`
32 → 96（否则 ≥32 UE/小区时负载观测恒饱和、RL 的负载维度等于瞎），`w_load` 保持
0（H4/H5/H6 已证：congested 拓扑上显式负载项是以掉话换减载）。

两条硬门禁：① P0 墙钟标定通过（N=128 单集可接受，否则降到 [16,64]）；
② **必须重跑 Stage1 BC**——归一化口径变了，旧 `bc_init.pt` 是在旧语义下克隆的，
直接拿来当锚点等于重演 F4（锚到对新分布不成立的策略）。

分支 `H9L` = H9 + `w_load=0.02`，**必须**等 H9 验收后才开；判据不是吞吐，而是
`load_audit` 的 `load_time_mean` 显著下降**且**存活/RLF 不劣化，缺一判负回退 H9。


## H13：当前最优候选（2026-10-07 验收并定稿）

**H13 = H11 + 两个奖励定价键**（`w_gap` 0.05→0.3、`weak_coverage_sinr_db`
-2.0→0.0），针对 cmd_0186 定位到的 load 族死因 no_ho_while_a3_would（该切没
切：切换一次性成本 0.9 vs 赖着不动 0.05/步，差近 20 倍）。ckpt =
`checkpoints_stageH13/best.pt`（2026-09-30 22:39 快照，从 H8 best 续训、锚定
H8 best）。

**主判据 PASS**（cmd_0188）：load 族 16 集 survival 0.75(H11) → **1.00**，
死因 no_ho_* 清零，ho 恢复 4.0/集。

**验收矩阵（cmd_0189/0190，同 seed 同口径，A3 列逐位一致）：**

M 矩阵（path_probe × M1–M8，cbr=20）：

| 对照 | 结论 |
|---|---|
| vs H11 | 6/8 场景 p95 退化（约翻倍：25→48.8 等），M1/M2 吞吐 −1~2% —— 按预写门禁字面 **FAIL** |
| vs H8(S2) | 吞吐 8 场景全部 ±1% 持平；p95 仅 M2(+3.9%)/M7(+5.4%) 微超 2% 容差；**H11 遗留的 M3/M8 吞吐缺口被修复**（M8 p95 89→38.9，均速 +1%） |

六族配对（paired_matrix，H13/H11/H8 三方同框，均跑在 H11 配置上）：

| 族 | Δtput H8 | Δtput H11 | Δtput H13 | 存活 H11→H13 | ho |
|---|---|---|---|---|---|
| normal | +1.58 | +1.68 | +1.55 | 1.00→1.00 | 4.00 = |
| tunnel | +1.84 | +2.11 | +1.83 | 1.00→1.00 | 8.17 = |
| load | +1.60 | +1.66 | +1.62 | **0.83→1.00** | 3.83→4.00 |
| congested | +1.90 | +2.07 | **+1.96** | 1.00→1.00 | 8.17 = |
| hotcold | +1.70 | +1.67 | **+1.77** | 1.00→1.00 | 4.00 = |

**判读（记录，供定稿讨论）**：H13 的六族 Δ 回落到 H8 水平（congested/hotcold
反超），p95 回归 H8 档位——证据指向 **H11 的 p95 优势与它的 load 族掉话是同一
个欠切换偏差的两面**（赖着不动既压低 p95 又在负载场景里拖死链路）。切换定价修
正后：掉话清零、M3/M8 修复、无 ho_rebound（w_gap 0.3 的过度切换担忧未出现）。
代价是放弃 H11 的 p95 尾部优势与 ~0.1–0.3 Mbps 的六族 Δ 增量。

**结论：H13 相对最后一个健康产物 H8 严格占优**（增益持平或更好 + 修复 M3/M8 +
无新退化），建议取代 H8 为生产产物；H11 落选（掉话）。若追求 H11 的 p95，可开
H14（w_gap 0.15 折中）但预期是回到同一权衡曲线，非免费午餐。

已知边界（不判死）：M1/M2/M7 的 p95 对 H8 微超 2% 容差；六族 Δ 对 H11 的
−0.1~−0.3 Mbps；H12 已判负勿再走 CBR=100 路线（8/8 质量退化，饱和源掩盖切换
质量差异，符合修正 B 预警）。

**定稿后补测（cmd_0191/0192/0195，2026-10-07）**：① 容量实验随 H13 刷新
（capacity_probe，H13 vs A3 同 seed 配对，固定压力场景 CBR=20）：N=32/64/128
全部 qos_ok=True、RLF=0、pingpong=0、存活 1.0；吞吐与 A3 持平（±0.1%），
delay_p95 逐级更优（−8.6%/−10.5%/−25%）；load_sat_ratio=0.0（观测未饱和）；
N=128 单集 wall≈21 min、rss≤618 MB，可行性无虞。② 全套 pytest 回归
**101/101 通过**（190 s）——期间发现并隔离了一个远端化石
`configs/ns3_smoke.yaml`（2026-08-30 cmd_0012 用 sed 临时生成、从未进本地
版本库，schema 长出必填键后挂掉 all-configs 校验），已移入
`configs/_attic/ns3_smoke.yaml.fossil-20260830`。教训：**远端仓库是 tar
覆盖式同步，本地删除不会传播；远端独有文件会无限存活**，发现未知配置先查
`configs/_attic/`。

## H14：离散 SAC 算法变体（2026-10-13 验收判负，H13 维持生产产物）

**H14 = H13 + 只改学习算法**（`agent.algorithm: dqn→sac`；
`src/agents/sac_agent.py`，离散 soft Q-learning + 双 critic min + 自动温度，
继承全部现有管线：PER/n 步/BC 锚定/安全层；ckpt 的 "online" 键仍是 critic1，
评估管线零改动兼容）。配置由 `session_logs/make_h14_cfg.py` 机械派生，内置
单变量断言（差异恰好 = {algorithm, checkpoint_dir}）。立项假设：H4→H13 反复
出现的「自强化坍缩到 ho=0」是确定性策略的病，熵项 softmax(Q/α) 结构性禁止
概率塌缩，可替代手工信任域。

**训练（cmd_0196）**：1200/1200 cycles 收口（4646 s，0 失败），loss 单调降
（23→0.72），内联评估全程 ho≥2、无 ho=0 坍缩，best.pt 末值 out=0/R=19.3
（评估族 normal 上 R 从 8.4 一路涨到 c1200，仍在进步）。

**验收（cmd_0198，两判据）**：

| 判据 | 结果 |
|---|---|
| ① 主判据：load 族 16 集 | **PASS**（survival=1.00、no_ho_*=0，R 均值 ≈31） |
| ② 不劣：五族同 seed 配对 vs H13 | **FAIL**：tunnel 存活 0.33、congested 存活 0.50（H13 均 1.00）；hotcold Δtput +1.52 vs +1.77 超容差判 dtput_drop；normal/load 持平 OK |

配对可信度：A3 列数值逐位相同（表格两列 tput 完全一致）；脚本打出的
`[WARN] A3 rows differ` 是 grep 'A3' 误吞了进度行 `[9/30] ... A3(3GPP)` 的
**顺序差异**，非数据差异——该一致性检查今后应改为只匹配结果行。

**判读**：熵项抗坍缩的假设**部分成立**——SAC 全程没有出现 ho=0 式坍缩、
load 族主判据干净通过；但它带来了**新的失败模式**：tunnel/congested 上
欠切换致死（ho 4.33/5.33 vs H13 的 8.17，存活崩到 0.33/0.50）。平均吞吐反而
略高（tunnel 11.51 vs 11.37）——又是那个已知的权衡面：**少切 → 均值好看、
尾部要命**。结论：算法替换无净增益，瓶颈不在学习规则而在奖励定价与场景
难度；**H13 维持生产产物**。若再探 SAC，方向是给 tunnel/congested 的
RLF 后果更强的信用分配（而非换学习器）。

**运维教训（同批发现）**：母本 cmd_0187 的守卫 `echo "TRAIN_EXIT=$?"` 只打印
从未赋值，`set +u` 下 `[ "$TRAIN_EXIT" != 0 ]` 永真 ⇒ 内联主判据永远 SKIP
（H13 当年同样中招，其主判据实为单独投的 cmd_0188）。cmd_0198 起守卫改为
纯产物检查。**机械派生会保真继承母本的语义 bug**——派生校验除 token 残留
外，还须对守卫分支干跑、并检查跨段共享变量（CFG/COMMON）是否随片段丢失。
