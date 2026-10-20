# scripts 说明

## 训练与评估主入口

- `pretrain_bc.py`：Stage 1 行为克隆（A3 teacher）
- `train_ns3.py`：Stage 2 NS-3 在线训练（支持 `--mock` 冒烟）
- `eval_ns3.py`：单模型评估（RL 或 `--baseline a3`）
- `baselines_ns3.py`：A3/规则基线与 RL 对照

## 质量门禁与分析工具

- `reward_sanity.py`：Stage 2 前奖励可学性门禁
- `check_config.py`：配置自检
- `path_probe.py` / `paired_matrix.py` / `load_audit.py` / `capacity_probe.py`：评测与诊断

## 本地最小冒烟

```bash
cd /home/runner/work/highspeed-rl/highspeed-rl
python -m scripts.train_ns3 --config configs/default.yaml --mock --net hybrid --n-episodes 3
```

