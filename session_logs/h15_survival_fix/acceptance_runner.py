"""Isolated, episode-level H15 acceptance; no training or legacy evidence edits."""
import argparse
import copy
import hashlib
import json
import pathlib
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from src.utils.config import load_config
from src.envs.ns3_env import NS3HandoverEnv
from src.agents.baselines import A3Baseline
from src.agents.agent import DuelingDDQNAgent
from src.agents.hybrid_network import HybridDuelingDQN
from src.training.device import get_device

FAMILIES = ['normal', 'tunnel', 'load', 'congested', 'hotcold']
METRICS = ['throughput_mbps', 'tp_agg_mbps', 'tp_agg_cell_mbps', 'tp_ue_p05_mbps', 'tp_ue_p50_mbps', 'jain', 'ctrl_delay_ms']


def sha(path):
    p = pathlib.Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def run_episode(job):
    config, checkpoints, family, seed, policy, outdir = job
    start = time.time()
    cfg = load_config(config)
    env = NS3HandoverEnv(copy.deepcopy(cfg))
    agent = None
    if policy != 'A3':
        agent = DuelingDDQNAgent(cfg, get_device('cpu'), net_factory=HybridDuelingDQN)
        agent.load(checkpoints[policy])
        agent.reset_latent()
    a3 = A3Baseline(cfg['safety'])
    a3.reset() if hasattr(a3, 'reset') else None
    history = []
    try:
        obs, info = env.reset(seed=seed, options={'family': family})
        spawn = float(info['ue_x'])
        spawn_command = list(env._spawn_cmd)
        term = trunc = False
        for _ in range(int(cfg['env']['episode_steps'])):
            action = int(agent.act(obs, info, eps=0.0) if agent else a3.act(obs, info))
            before = {k: np.asarray(info[k]).tolist() for k in ['candidate_rsrp', 'candidate_sinr', 'neighbors']}
            obs, reward, term, trunc, info = env.step(action)
            h = {k: info.get(k) for k in ['step_idx', 'ue_x', 'serving_idx', 'serving_sinr', 'action_requested', 'action_executed', 'action_source', 'ho_event', 'ho_request_result', 'ho_in_progress', 'ho_failed', 'rlf', 'rlf_count', 'termination_reason', 'guard_active', 'n_bg_total'] + METRICS}
            h.update({'before': before, 'reward': float(reward), 'terminated': bool(term), 'truncated': bool(trunc), 'decision': agent.last_decision if agent else None})
            history.append(h)
            if term or trunc:
                break
        steps = len(history)
        episode = {'policy': policy, 'family': family, 'seed': seed, 'steps': steps, 'terminated': bool(term), 'truncated': bool(trunc), 'termination_reason': info.get('termination_reason'), 'full_route': steps == int(cfg['env']['episode_steps']), 'survived': steps == int(cfg['env']['episode_steps']) and not term, 'spawn_x': spawn, 'spawn_command': spawn_command, 'final_x': float(info['ue_x']), 'ho_completed': sum(bool(h['ho_event']) for h in history), 'ho_requests': sum(h['action_executed'] in [1, 2, 5] for h in history), 'safety_a3_count': sum(h['action_source'] == 'safety_a3' for h in history), 'rescue_changes': sum(h['action_requested'] != h['action_executed'] and h['action_source'] == 'safety_a3' for h in history), 'wall_s': time.time() - start, 'metrics': {k: float(np.mean([h[k] for h in history if h[k] is not None])) if any(h[k] is not None for h in history) else None for k in METRICS}, 'exit': 0}
        path = pathlib.Path(outdir) / f'{policy}_{family}_{seed}.json'
        path.write_text(json.dumps({'episode': episode, 'history': history}, ensure_ascii=False, indent=1), encoding='utf-8')
        return episode
    finally:
        env.close()


def verdict(episodes):
    expected = {(p, f, 500000 + s * 1000 + e) for p in ['A3', 'H13', 'H15'] for f in FAMILIES for s in range(3) for e in range(2)}
    observed = {(e['policy'], e['family'], e['seed']) for e in episodes}
    checks = [{'name': 'exact_unique_90_episodes', 'pass': observed == expected and len(episodes) == 90}, {'name': 'all_episodes_alive_full_route', 'pass': len(episodes) == 90 and all(e['survived'] and e['full_route'] and not e['terminated'] for e in episodes)}]
    detail = {}
    for family in FAMILIES:
        groups = {p: [e for e in episodes if e['family'] == family and e['policy'] == p] for p in ['A3', 'H13', 'H15']}
        if any(len(g) != 6 for g in groups.values()):
            continue
        means = {p: {'throughput': float(np.mean([e['metrics']['throughput_mbps'] for e in g])), 'ho': float(np.mean([e['ho_completed'] for e in g])), 'survival': float(np.mean([e['survived'] for e in g])), 'rescue_total': sum(e['rescue_changes'] for e in g), 'fairness': {k: float(np.mean([e['metrics'][k] for e in g])) for k in ['tp_ue_p05_mbps', 'tp_ue_p50_mbps', 'jain']}} for p, g in groups.items()}
        delta13 = means['H13']['throughput'] - means['A3']['throughput']
        delta15 = means['H15']['throughput'] - means['A3']['throughput']
        tol = .02 * abs(delta13) + .05
        checks.extend([{'name': family + '_deltput_noninferiority', 'pass': delta15 >= delta13 - tol, 'H15_delta': delta15, 'H13_delta': delta13, 'tolerance': tol}, {'name': family + '_ho_nonrebound', 'pass': means['H15']['ho'] <= 1.2 * means['H13']['ho'] + .01}])
        detail[family] = means
    return {'passed': all(c['pass'] for c in checks), 'checks': checks, 'families': detail, 'note': 'All policies share rescue-enabled env. One shared A3 reference per seed, no unverified legacy A3 reuse. Dead/incomplete trajectories invalidate any superiority claim.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--h15', required=True)
    ap.add_argument('--h13', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    paths = [args.config, args.h15, args.h13, cfg['ns3']['binary'], __file__, 'src/agents/safety.py', 'src/agents/agent.py', 'src/envs/ns3_env.py', 'scripts/train_ns3.py', 'session_logs/diag_death_probe.py', 'ns3/hsr-rl.cc', 'ns3/hsr-common.h']
    manifest = {'config': cfg, 'hashes': {p: sha(p) for p in paths}, 'seed_formula': '500000+1000*seed_index+episode_index', 'families': FAMILIES, 'policies': ['A3', 'H13', 'H15'], 'workers': args.workers, 'started': time.time()}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    checkpoints = {'H15': args.h15, 'H13': args.h13}
    jobs = [(args.config, checkpoints, f, 500000+s*1000+e, p, str(out)) for f in FAMILIES for s in range(3) for e in range(2) for p in ['A3', 'H13', 'H15']]
    episodes, errors = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs = {pool.submit(run_episode, j): j for j in jobs}
        for future in as_completed(fs):
            try:
                ep = future.result()
                episodes.append(ep)
                print(json.dumps(ep), flush=True)
            except Exception as exc:
                errors.append({'job': fs[future][2:5], 'error': repr(exc)})
                print('ERROR ' + repr(exc), flush=True)
            result = verdict(episodes)
            result.update({'episodes': episodes, 'errors': errors, 'complete': len(episodes)+len(errors) == 90})
            (out/'verdict.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return 0 if result['passed'] and not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
