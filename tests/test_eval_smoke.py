import os
import subprocess
import sys


def test_eval_runs(tmp_path):
    ckpt = str(tmp_path / "m.pt").replace("\\", "/")
    # 先训练 2 eps 产出 checkpoint
    tr = (
        "from src.utils.config import load_config; "
        "cfg=load_config('configs/default.yaml'); "
        "cfg['training']['n_episodes']=2; cfg['env']['episode_steps']=20; "
        "cfg['agent']['buffer_size']=200; "
        "import scripts.train as t; t.run(cfg, r'" + ckpt + "'); "
    )
    r = subprocess.run([sys.executable, "-c", tr], cwd=os.getcwd(),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    ev = (
        "from src.utils.config import load_config; "
        "cfg=load_config('configs/default.yaml'); cfg['eval']['n_episodes']=2; "
        "cfg['env']['episode_steps']=20; "
        "import scripts.eval as e; m=e.evaluate(cfg, r'" + ckpt + "'); "
        "assert 'throughput' in m; print('OK')"
    )
    r2 = subprocess.run([sys.executable, "-c", ev], cwd=os.getcwd(),
                        capture_output=True, text=True, timeout=300)
    assert r2.returncode == 0, r2.stderr
    assert "OK" in r2.stdout
