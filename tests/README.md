# tests 说明

测试框架为 `pytest`，覆盖环境、智能体、配置校验与 NS-3 mock 协议流程。

## 常用命令

```bash
cd /home/runner/work/highspeed-rl/highspeed-rl
python -m pytest tests/ -q
```

按主题快速回归：

```bash
python -m pytest tests/test_config_validation.py -q
python -m pytest tests/test_ns3_env.py -q
python -m pytest tests/test_train_smoke.py -q
```

> 文档改动通常不需要新增测试；但提交前建议至少做路径/命令自检。
