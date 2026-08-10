# TSJT-C1 / TSJT-C2

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

## C2 同频率双城实验

C2 保留 C1 的 Dense NPM、prompt 图、Graph WaveNet、CPRR、last-value residual
和 fixed TSB，只迁移实验协议：每次使用一个源城市和一个同采样频率目标城市，
统一过去 6 步预测未来 6 步。目标城市仅以前两天训练，测试集只在训练结束后访问一次。

四个正式方向分别运行：

```bash
bash scripts/run_c2.sh configs/c2/pemsbay_to_metrla.yaml
bash scripts/run_c2.sh configs/c2/metrla_to_pemsbay.yaml
bash scripts/run_c2.sh configs/c2/chengdu_to_shenzhen.yaml
bash scripts/run_c2.sh configs/c2/shenzhen_to_chengdu.yaml
```

每个方向最多训练 50 epoch；每个 epoch 完成后以固定权重、eval 模式完整遍历目标
训练集，并按所得 MAE 早停（patience=10）。最终报告第
2、4、6 步的 MAE 和 MAPE。详细实现边界见
[docs/C2_ARCHITECTURE.md](docs/C2_ARCHITECTURE.md)。
