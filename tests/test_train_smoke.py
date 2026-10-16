import os
import subprocess
import sys


def test_inline_eval_covers_families_and_shared_mask(monkeypatch):
    import copy
    import numpy as np
    import torch
    from src.utils.config import load_config
    from src.agents.agent import DuelingDDQNAgent
    from src.envs.ns3_env import NS3HandoverEnv
    from scripts.train_ns3 import evaluate
    cfg = copy.deepcopy(load_config('configs/default.yaml'))
    cfg['ns3'] = {'binary': 'unused'}
    cfg['env']['episode_steps'] = 2
    cfg['eval'].update(families=['normal', 'tunnel'], max_steps=0)
    agent = DuelingDDQNAgent(cfg, torch.device('cpu'))
    seen = []
    def reset(self, seed, options):
        self.fam = options['family']
        seen.append((self.fam, seed))
        return np.zeros(agent.obs_dim, dtype=np.float32), {
            'candidate_rsrp': [-80., -120., -130., -140.],
            'step_idx': 0, 'last_ho_step': -100}
    def step(self, action):
        assert action == 0  # all HO blocked; dual statically unsupported
        return np.zeros(agent.obs_dim), 1., self.fam == 'tunnel', True, {}
    monkeypatch.setattr(NS3HandoverEnv, 'reset', reset)
    monkeypatch.setattr(NS3HandoverEnv, 'step', step)
    monkeypatch.setattr(NS3HandoverEnv, 'close', lambda self: None)
    m = evaluate(agent, cfg, torch.device('cpu'), 1)
    assert seen == [('normal', 900000), ('tunnel', 900000)]
    assert m['outage'] == .5 and m['worst_family_outage'] == 1.
    assert not m['coverage_complete']
    assert not m['full_survival']
    from scripts.train_ns3 import best_checkpoint_eligible
    cfg['eval']['require_full_survival'] = True
    assert not best_checkpoint_eligible(m, cfg)
    assert [e['steps'] for e in m['episodes']] == [1, 1]
    assert m['episodes'][1]['terminated']


def test_eval_actual_route_and_survival_gate(monkeypatch):
    import copy
    import numpy as np
    import torch
    from src.utils.config import load_config
    from src.agents.agent import DuelingDDQNAgent
    from src.envs.ns3_env import NS3HandoverEnv
    from scripts.train_ns3 import evaluate, best_checkpoint_eligible, format_eval_metrics
    cfg = copy.deepcopy(load_config('configs/default.yaml'))
    cfg['env']['episode_steps'] = 3
    cfg['ns3'] = {'binary': 'unused'}
    cfg['eval'].update(families=['normal'], max_steps=0, require_full_survival=True)
    agent = DuelingDDQNAgent(cfg, torch.device('cpu'))
    mode = ['full']
    def reset(self, **kwargs):
        self.t = 0
        return np.zeros(agent.obs_dim, dtype=np.float32), {
            'candidate_rsrp': [-80., -120., -130., -140.],
            'step_idx': 0, 'last_ho_step': -100}
    def step(self, action):
        self.t += 1
        death = mode[0] == 'death' and self.t == 3
        trunc = (mode[0] == 'early' and self.t == 1) or self.t == 3
        return np.zeros(agent.obs_dim), 1., death, trunc, {
            'candidate_rsrp': [-80., -120., -130., -140.],
            'step_idx': self.t, 'last_ho_step': -100}
    monkeypatch.setattr(NS3HandoverEnv, 'reset', reset)
    monkeypatch.setattr(NS3HandoverEnv, 'step', step)
    monkeypatch.setattr(NS3HandoverEnv, 'close', lambda self: None)
    for outcome, complete, eligible in [('death', True, False),
                                         ('early', False, False), ('full', True, True)]:
        mode[0] = outcome
        m = evaluate(agent, cfg, torch.device('cpu'), 1)
        assert m['coverage_complete'] is complete
        assert best_checkpoint_eligible(m, cfg) is eligible
        text = format_eval_metrics(m)
        assert '["normal"]' in text and 'episodes' in text
        assert '1.0000' in text or '3.0000' in text
    cfg['eval']['max_steps'] = 1
    assert not best_checkpoint_eligible(evaluate(agent, cfg, torch.device('cpu'), 1), cfg)


def test_probe_rescue_rewritten_target():
    from session_logs.diag_death_probe import action_targets, classify, last_ho_detail
    fields = action_targets(0, 1, [-77, -73, -90, -100], [-6, 0, -10, -15], [12, 13, 14])
    assert fields['policy_requested_target']['rsrp'] is None
    assert fields['executed_target'] == {'rsrp': -73., 'sinr': 0., 'cell': 12}
    h = dict(fields, step=1, x=100., serving_rsrp=-77., serving_sinr=-6.,
             pre_serving_rsrp=-77., a_a3=0, a3_tgt_rsrp=None,
             executed=1, ho_completed=True)
    assert classify([h], 'low_sinr', -110) == 'unknown'
    assert last_ho_detail([h])['tgt_rsrp'] == -73.
    h.update(action_targets(1, 2, [-77, -73, -90, -100], [-6, 0, -10, -15], [12, 13, 14]))
    h['executed'] = 2
    assert classify([h], 'low_sinr', -110) == 'ho_to_weaker_against_a3'
    assert last_ho_detail([h])['executed_target']['cell'] == 13
    h['ho_completed'] = False
    assert classify([h], 'low_sinr', -110) == 'unknown'


def test_train_short_runs(tmp_path):
    ckpt = str(tmp_path / "ck.pt").replace("\\", "/")
    code = (
        "from src.utils.config import load_config; "
        "cfg=load_config('configs/default.yaml'); "
        "cfg['training']['n_episodes']=2; "
        "cfg['env']['episode_steps']=20; "
        "cfg['agent']['buffer_size']=200; "
        "import scripts.train as t; t.run(cfg, r'" + ckpt + "'); "
        "print('OK')"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=os.getcwd(),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
