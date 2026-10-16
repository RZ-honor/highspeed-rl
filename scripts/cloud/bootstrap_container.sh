#!/bin/bash
# bootstrap_container.sh —— 容器重建后的一键自愈（经 bridge 或容器内直接跑）
# 幂等：可重复执行。所有输出落 /work（持久盘）。
# 背景见 memory dcu-cloud-env：/opt/conda 自装包随容器重建丢失；
# nohup 后台任务随终端回收退出，必须 setsid；bridge.status 是判活依据。
set -e
REPO=/work/home/<HPC_ACCOUNT>/highspeed-rl/highspeed-rl
cd "$REPO"

echo "===== host/time ====="
hostname; date

echo "===== 1. python 依赖自愈（/opt/conda 重建后丢自装包）====="
# 注意：新 Pod 可能**没有外网/DNS**（2026-09-18 实测 nameserver 8.8.8.8
# 不可达，所有 pip 镜像都解析失败）。此时走持久盘上的离线 wheel 目录。
WHEELS=/work/home/<HPC_ACCOUNT>/wheels
if /opt/conda/bin/python -c "import gymnasium" 2>/dev/null; then
  echo "gymnasium 已装"
elif /opt/conda/bin/pip install gymnasium==0.29.1 \
      -i https://mirrors.aliyun.com/pypi/simple/ 2>/dev/null; then
  echo "联网安装成功"
elif [ -d "$WHEELS" ]; then
  echo "联网失败 → 离线安装（$WHEELS；缺失时请在本地 pip download 后上传）"
  /opt/conda/bin/pip install --no-index --find-links "$WHEELS" --no-input \
      gymnasium==0.29.1 importlib_metadata farama-notifications \
      typing_extensions zipp
else
  echo "FATAL: 无外网且 $WHEELS 不存在。请在有网的机器执行："
  echo "  pip download -d wheels --no-deps gymnasium==0.29.1 \\"
  echo "      importlib_metadata farama-notifications typing_extensions zipp"
  echo "  打包上传到 $WHEELS 后重跑本脚本。"
  exit 1
fi
/opt/conda/bin/python - <<'EOF'
import gymnasium, torch, numpy, yaml
print("gymnasium", gymnasium.__version__, "| torch", torch.__version__,
      "| cuda avail", torch.cuda.is_available())
EOF

echo "===== 2. import 冒烟 ====="
PYTHONPATH=. /opt/conda/bin/python -c \
  "from src.envs.ns3_env import NS3HandoverEnv; from src.agents.hybrid_network import HybridDuelingDQN; print('import OK')"

echo "===== 3. NS-3 二进制存在性（/work 持久，正常无需重编译）====="
ls -l /work/home/<HPC_ACCOUNT>/ns3root/run_hsr.sh
ls -l /work/home/<HPC_ACCOUNT>/ns3root/ns-3.43/build/scratch/ns3.43-hsr-rl

echo "===== 4. bridge 重启（setsid，nohup 会随终端回收退出）====="
BR=/work/home/<HPC_ACCOUNT>/.agent_bridge
cat "$BR/bridge.status" 2>/dev/null || echo "(no status yet)"
# 只有 status 里 hostname != 当前容器 或 进程不存在时才重启
if ! pgrep -f 'agent_bridge/bridge.sh' >/dev/null; then
  cd "$BR"
  setsid bash bridge.sh >/tmp/bridge.log 2>&1 </dev/null &
  sleep 4
fi
cat "$BR/bridge.status"
echo "BOOTSTRAP_DONE rc=$?"
