from session_logs.h15_survival_fix.acceptance_runner import FAMILIES, verdict


def fixture_episodes():
    return [dict(policy=p, family=f, seed=500000+s*1000+e,
                 survived=True, full_route=True, terminated=False,
                 metrics=dict(throughput_mbps=10 if p == 'A3' else 11,
                              tp_ue_p05_mbps=1, tp_ue_p50_mbps=2, jain=.5),
                 ho_completed=4, rescue_changes=0)
            for p in ['A3', 'H13', 'H15'] for f in FAMILIES
            for s in range(3) for e in range(2)]


def test_acceptance_full_unique_matrix():
    assert verdict(fixture_episodes())['passed']


def test_acceptance_missing_duplicate_death_or_incomplete_rejected():
    episodes = fixture_episodes()
    assert not verdict(episodes[:-1])['passed']
    assert not verdict(episodes[:-1]+[episodes[0]])['passed']
    episodes[-1]['terminated'] = True
    assert not verdict(episodes)['passed']
    episodes[-1].update(terminated=False, full_route=False, survived=False)
    assert not verdict(episodes)['passed']


def test_acceptance_original_noninferiority_and_ho_thresholds():
    episodes = fixture_episodes()
    for ep in episodes:
        if ep['policy'] == 'H15':
            ep['metrics']['throughput_mbps'] = 10.9
    assert not verdict(episodes)['passed']
    for ep in episodes:
        if ep['policy'] == 'H15':
            ep['metrics']['throughput_mbps'] = 11
            ep['ho_completed'] = 5
    assert not verdict(episodes)['passed']
