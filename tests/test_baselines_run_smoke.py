import os
import subprocess
import sys


def test_baselines_run(tmp_path):
    code = (
        "from src.utils.config import load_config; "
        "cfg=load_config('configs/default.yaml'); cfg['eval']['n_episodes']=1; "
        "cfg['env']['episode_steps']=20; "
        "import scripts.baselines_run as b; m=b.run(cfg); "
        "assert 'A3' in m and 'greedy' in m; print('OK')"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=os.getcwd(),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout
