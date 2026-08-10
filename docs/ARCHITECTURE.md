# TSJT-C1 架构分析

## 1. 整理结论

旧项目中的 `v2`、`v3`、`ablation` 和 `screening` 是研究演进阶段，并不是四套需要
并存的产品版本。当前正式结果实际只有一条有效主链：Dense Equation-(5) NPM、
prompt graph、Graph WaveNet、共享小时解码器、last-value residual、CPRR，以及
fixed TSB 多源一阶段训练。新目录把这条主链命名为 **TSJT-C1**，其余历史变体不再
迁移。

## 2. 分层结构

```text
配置层      configs/c1.yaml
入口层      tsjt_c1/run.py
训练层      tsjt_c1/training/engine.py + tsb.py
模型层      tsjt_c1/models/c1.py
            ├── base.py       Dense NPM + prompt graph + GWN
            ├── decoder.py    共享 12 位置物理小时解码器
            └── backbone.py   时空编码与动态扩散图卷积
数据层      tsjt_c1/data/loaders.py + pipeline.py
资产层      data/ + checkpoints/
```

依赖方向严格从入口向下：`run → training/models/data`。模型不读取配置、不访问文件，
数据层不知道训练状态机，训练层只通过模型的 `forward` 和参数列表工作。

## 3. 启动与状态机

唯一入口为：

```bash
python -m tsjt_c1.run --config configs/c1.yaml
```

入口依次完成：配置协议校验、随机种子与 CUDA 校验、模型构造、运行目录身份校验、
数据指纹记录、四城市 loader 构造、训练/恢复以及一次性正式评估。

正式运行状态只能单向前进：

```text
training
  → last.pt（每 epoch 可恢复）
  → epoch_50.pt（冻结权重）
  → evaluation_started.json（正式测试访问标记）
  → formal_candidate_metrics.json
  → complete.json
```

如果正式测试在标记写入后中断，程序会拒绝第二次测试访问；如果输出目录属于不同
配置，也会拒绝覆盖。这些约束用于避免重复查看正式测试结果和混用实验身份。

## 4. 数据架构

四个城市承担固定角色：PEMS-BAY 是目标域，METR-LA、Chengdu、Shenzhen 是源域。

| 城市 | 分辨率 | 历史/预测步数 | 训练用途 |
|---|---:|---:|---|
| PEMS-BAY | 5 分钟 | 12/12 | 前 3 天目标训练，其余正式测试 |
| METR-LA | 5 分钟 | 12/12 | 前 70% 源训练 |
| Chengdu | 10 分钟 | 6/6 | 前 70% 源训练 |
| Shenzhen | 10 分钟 | 6/6 | 前 70% 源训练 |

输入张量为 `[B,T,N,3]`，三个通道是训练区间归一化速度、time-of-day 和
day-of-week。归一化统计只从各自训练区间拟合；窗口的 history 和 target 均完全
位于对应 split 内，未来速度从不作为输入。

## 5. 模型数据流

```text
X [B,T,N,3]
  ├─→ 时空 pattern encoder
  │     → Dense Equation-(5) NPM
  │     → node prompts P [B,T,N,32]
  │          ├─→ cosine prompt graph
  │          └─→ CPRR horizon residual
  └─→ concat(X, P)
          → Dynamic Graph WaveNet
          → shared canonical-hour decoder
          → + last observed speed
          → + bounded CPRR residual
          → prediction [B,H,N,1]
```

Dense NPM 保留 prompt bank 的 `K/T` 语义，不做旧实现中的全局 softmax。共享解码器
学习 12 个五分钟位置；10 分钟城市用索引 `[1,3,5,7,9,11]`，因此 H=6 与 H=12
共享同一套输出参数。CPRR 从 `P_last - mean_time(P)` 读取 12 个 horizon residual，
用 `0.5*tanh` 限幅，仅增加 396 个参数。last-value residual 是性能关键的工程修复。

## 6. 训练更新

每个 logical update 先计算一个目标 batch 的梯度 `g_t`，再分别计算三个源城市
42/41/41 个样本组成的逻辑源 batch 梯度 `g_s`。fixed TSB 对每个参数张量分解：

```text
parallel   = <g_s,g_t> / ||g_t||² · g_t
orthogonal = g_s - parallel
```

当内积为负时丢弃冲突的平行分量，否则保留完整源梯度。最终显式更新：

```text
theta ← theta - gamma_target·g_t - gamma_source·filtered(g_s)
```

这里没有 Adam/SGD optimizer；`gamma_target=0.001`，`gamma_source=0.0005`。

## 7. 文件保留与删除原则

保留项只有运行必要代码、唯一配置、四城市数据、冻结 checkpoint、验证脚本和主链路
测试。旧 `outputs/`、临时目录、缓存、版本化包、消融模型、开发筛选配置、PowerShell
包装以及历史结果表均不属于单一运行版本，已从新目录排除。

数据约占 1.31 GiB，是目录体积的主要来源，不是代码冗余。公开发布前需单独确认
四个 `.npy` 数据集的再分发许可。

## 8. 已知边界

- 正式证据只有 PEMS-BAY、seed 2025，不能宣称多目标城市或多 seed 泛化。
- prompt graph 和直接 prompt feature 的独立正贡献证据较弱；当前保留它们是为了
  保持冻结 C1 身份，而不是因为各自已被充分证明。
- frozen checkpoint 可用于迁移一致性验证；正式训练命令会重新训练 50 epoch，并非
  checkpoint 推理命令。
