# TSJT-C1 

这是从旧交接项目中整理出的清晰实验仓库：**TSJT-C1** 是冻结基线，
**TSJT-C2** 是同频率双城 6→6 新协议。仓库不再暴露 v2、v3、screening、
ablation 等旧交接项目中的混杂入口。

## 运行

在仓库根目录激活包含 PyTorch、NumPy 和 PyYAML 的环境，然后运行：

```bash
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0
python -m tsjt_c1.run --config configs/c1.yaml
```

或使用：

```bash
bash scripts/run.sh
```

该命令是正式协议：固定训练 50 epoch，并在 epoch 50 checkpoint 持久化后访问一次
PEMS-BAY 完整测试集。它要求 GPU，不会静默退回 CPU。若只想确认迁移正确，不要
启动正式训练，运行：

```bash
PYTHONPATH=src python scripts/verify_model.py --device cpu
PYTHONPATH=src python -m pytest -q
```

## 目录

```text
configs/c1.yaml          冻结 C1 配置
configs/c2/              四个 C2 迁移方向配置
src/tsjt_c1/             冻结 C1 基线代码包
src/tsjt_c2/             C2 双城协议代码包
  data/                  数据读取、切分、归一化、指标
  models/                图骨干、Dense NPM、共享解码器、C1/CPRR
  training/              fixed TSB 更新与运行时支持
  run.py                 50 epoch 可恢复训练状态机
data/                    四城市运行数据
checkpoints/             冻结 epoch-50 checkpoint
scripts/                 启动和非正式验证脚本
tests/                   C1 主链路测试
docs/ARCHITECTURE.md     架构分析、数据流和设计边界
```

完整设计说明见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。