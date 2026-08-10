"""经过公式审计的 Dense-NPM TSJT 基座与共享物理时间解码器。

正式基座相对早期复现有两项关键修正：

* Equation (5) 保留 prompt bank 的 ``K`` 轴和时间轴，使用未经 softmax 的缩放乘积，
  最后沿 ``K`` 求和；
* 5 分钟与 10 分钟城市共享一个规范的一小时输出投影。

Graph WaveNet 主干、prompt 余弦知识图、模式编码器和一阶段 TSB 接口保持不变。

``last_value_residual`` 是工程适配，不是原论文 Equation (5) 的组成部分。启用后，
共享解码器预测的是相对最后观测归一化速度的残差。

便于审计的字面 Equation (5) 写法为：

    scores = einsum("bntd,ksd->bnkts", encoded, bank) / sqrt(D)
    per_pattern = einsum("bnkts,kse->bnkte", scores, bank)
    prompts = per_pattern.sum(dim=2)

``equation5_prompt`` 使用代数等价的 Gram 形式
``encoded @ sum_k(bank_k.T @ bank_k) / sqrt(D)``，避免物化五维 score 张量，
同时保持输出和梯度一致。实现刻意不展平 ``K*T``，也不使用 softmax。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import SharedCanonicalHourDecoder
from .backbone import (
    DynamicGraphWaveNetTrunk,
    SpatioTemporalPatternEncoder,
    _as_batched_adjacency,
    interpolate_prompt_bank,
    prompt_knowledge_graph,
)


def _validate_equation5_inputs(
    encoded: torch.Tensor, prompt_bank: torch.Tensor
) -> None:
    """检查 Equation-(5) 两个输入的秩、时间/特征维、device 和 dtype 一致性。"""
    if encoded.ndim != 4 or prompt_bank.ndim != 3:
        raise ValueError("expected encoded [B,N,T,D] and bank [K,T,D]")
    if encoded.shape[2:] != prompt_bank.shape[1:]:
        raise ValueError("encoded and prompt bank time/feature dimensions differ")
    if prompt_bank.shape[0] == 0 or encoded.shape[-1] == 0:
        raise ValueError("prompt bank and feature dimension must be nonempty")
    if encoded.device != prompt_bank.device:
        raise ValueError("encoded and prompt bank must be on the same device")
    if encoded.dtype != prompt_bank.dtype:
        raise ValueError("encoded and prompt bank must have the same dtype")


def equation5_prompt_reference(
    encoded: torch.Tensor, prompt_bank: torch.Tensor
) -> torch.Tensor:
    """Equation (5) 的未优化字面参考实现，用于公式和单元测试审计。

    ``encoded`` 为 ``[B,N,T,D]``，``prompt_bank`` 为 ``[K,T,D]``。先显式产生
    ``[B,N,K,T,T]`` score，再与 bank 相乘并沿 K 求和，输出 ``[B,N,T,D]``。
    此实现内存开销大，不用于正式训练。
    """

    _validate_equation5_inputs(encoded, prompt_bank)
    scale = math.sqrt(encoded.shape[-1])
    scores = torch.einsum("bntd,ksd->bnkts", encoded, prompt_bank) / scale
    per_pattern = torch.einsum("bnkts,kse->bnkte", scores, prompt_bank)
    return per_pattern.sum(dim=2)


def equation5_prompt(
    encoded: torch.Tensor, prompt_bank: torch.Tensor
) -> torch.Tensor:
    """字面参考式的显存受控代数等价实现，也是正式 C1 使用的路径。

    临时 Gram 矩阵仅为 ``[D,D]``，而非 ``[B,N,K,T,T]``。K 与 key-time 分别
    收缩，绝不会被展平为相互竞争的 token 轴。
    """

    _validate_equation5_inputs(encoded, prompt_bank)
    # sum_k(B_k^T B_k)：把整个 bank 压缩为共享的 D×D 线性算子。
    bank_gram = torch.einsum("ksd,kse->de", prompt_bank, prompt_bank)
    return torch.einsum("bntd,de->bnte", encoded, bank_gram) / math.sqrt(
        encoded.shape[-1]
    )


class Equation5NodePromptingModule(nn.Module):
    """包含时空编码器、Dense prompt bank 和审计版 Equation (5) 的正式 NPM。"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        prompt_bank_size: int,
        heads: int,
        dropout: float,
        canonical_steps: int = 12,
    ) -> None:
        """创建模式编码器，并以 Xavier 初始化 ``[K,12,D]`` prompt bank。"""
        super().__init__()
        if prompt_bank_size <= 0:
            raise ValueError("prompt_bank_size must be positive")
        self.encoder = SpatioTemporalPatternEncoder(
            input_dim, hidden_dim, heads, dropout, canonical_steps
        )
        self.prompt_bank = nn.Parameter(
            torch.empty(prompt_bank_size, canonical_steps, hidden_dim)
        )
        nn.init.xavier_uniform_(self.prompt_bank)

    def forward(
        self, x: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """由输入生成 ``[B,T,N,D]`` prompt 与 ``[B,N,N]`` 余弦知识图。"""
        # 编码器输出 [B,T,N,D]，Equation (5) 约定节点轴在时间轴之前。
        encoded = self.encoder(x, adjacency).permute(0, 2, 1, 3)
        bank = interpolate_prompt_bank(self.prompt_bank, x.shape[1])
        prompts = equation5_prompt(encoded, bank)
        knowledge_graph = prompt_knowledge_graph(prompts)
        return prompts.permute(0, 2, 1, 3), knowledge_graph


class BaseTSJT(nn.Module):
    """正式 C1 的基础前向：Dense NPM、动态 GWN 和共享一小时解码器。

    ``last_value_residual`` 启用时，将最后一个归一化速度加到每个未来步，使 decoder
    学习速度变化量而非绝对值。``supported_horizons=(6,12)`` 允许 10 分钟和 5 分钟
    城市共享全部参数。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        prompt_bank_size: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
        canonical_steps: int = 12,
        node_chunk_size: int = 64,
        bank_chunk_size: int = 16,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 64,
        end_channels: int = 128,
        blocks: int = 2,
        layers: int = 2,
        kernel_size: int = 2,
        diffusion_order: int = 2,
        supported_horizons: Sequence[int] = (6, 12),
        history_len: int | None = None,
        horizon: int | None = None,
        backbone: str = "graph_wavenet",
        canonical_history: int | None = None,
        prompt_query_chunk_size: int | None = None,
        gwn_residual_channels: int | None = None,
        gwn_dilation_channels: int | None = None,
        gwn_skip_channels: int | None = None,
        gwn_end_channels: int | None = None,
        gwn_blocks: int | None = None,
        gwn_layers: int | None = None,
        gwn_kernel_size: int | None = None,
        use_reverse_support: bool = True,
        use_knowledge_graph: bool = True,
        prompt_bank_rule: str | None = None,
        last_value_residual: bool = False,
    ) -> None:
        """解析冻结架构配置并组装 NPM、Graph WaveNet 与共享 decoder。

        ``gwn_*`` 和 ``canonical_history`` 是配置文件中的显式别名；最终都会归一到
        内部参数。正式结构强制使用 12 位置规范小时网格。
        """
        super().__init__()
        if backbone != "graph_wavenet":
            raise ValueError("TSJT-C1 implements the Graph WaveNet backbone")
        if canonical_history is not None:
            canonical_steps = canonical_history
        residual_channels = gwn_residual_channels or residual_channels
        dilation_channels = gwn_dilation_channels or dilation_channels
        skip_channels = gwn_skip_channels or skip_channels
        end_channels = gwn_end_channels or end_channels
        blocks = gwn_blocks or blocks
        layers = gwn_layers or layers
        kernel_size = gwn_kernel_size or kernel_size
        if canonical_steps != 12:
            raise ValueError("TSJT-C1 uses a canonical 12-position hour grid")
        if history_len is not None and history_len not in (6, 12):
            raise ValueError("history_len compatibility argument must be 6 or 12")

        horizons = tuple(int(value) for value in supported_horizons)
        if horizon is not None and horizon not in horizons:
            raise ValueError("default horizon is not in supported_horizons")
        self.default_horizon = horizon
        self.supported_horizons = horizons
        self.use_knowledge_graph = use_knowledge_graph
        self.prompt_bank_rule = prompt_bank_rule
        if not isinstance(last_value_residual, bool):
            raise TypeError("last_value_residual must be a boolean")
        self.last_value_residual = last_value_residual
        # 保留分块参数以兼容冻结配置；Gram 版 Equation (5) 实际无需节点/token 分块。
        self.node_chunk_size = int(prompt_query_chunk_size or node_chunk_size)
        self.bank_chunk_size = int(bank_chunk_size)
        if self.node_chunk_size <= 0 or self.bank_chunk_size <= 0:
            raise ValueError("compatibility chunk sizes must be positive")

        self.npm = Equation5NodePromptingModule(
            input_dim,
            hidden_dim,
            prompt_bank_size,
            heads,
            dropout,
            canonical_steps,
        )
        self.trunk = DynamicGraphWaveNetTrunk(
            input_dim + hidden_dim,
            residual_channels,
            dilation_channels,
            skip_channels,
            blocks,
            layers,
            kernel_size,
            diffusion_order,
            dropout,
            use_reverse_support,
        )
        self.end_projection = nn.Conv2d(skip_channels, end_channels, (1, 1))
        self.output_decoder = SharedCanonicalHourDecoder(
            input_channels=end_channels,
            canonical_steps=canonical_steps,
            supported_horizons=self.supported_horizons,
        )

    @property
    def output_heads(self) -> SharedCanonicalHourDecoder:
        """共享 decoder 的只读别名，供统一任务接口使用。"""

        return self.output_decoder

    def task_parameter_sets(self) -> dict[int, tuple[nn.Parameter, ...]]:
        """为每个原生 horizon 返回顺序稳定、对象完全相同的共享参数元组。

        TSB 按参数张量逐一比较目标和源梯度，因此 H=6/H=12 必须返回同一批参数对象
        且顺序一致，尤其不能拥有互不相干的输出头。
        """

        shared = tuple(
            parameter for parameter in self.parameters() if parameter.requires_grad
        )
        return {horizon: shared for horizon in self.supported_horizons}

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        horizon: int | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """执行 C1 基座前向。

        输入 ``x=[B,T,N,3]``、邻接矩阵 ``[N,N]`` 或 ``[B,N,N]``；输出为
        ``[B,H,N,1]``。``return_aux=True`` 时额外暴露 prompt、知识图和 GWN 特征，
        供 CPRR 与诊断代码复用。
        """
        if x.ndim != 4:
            raise ValueError("x must have shape [B,T,N,C]")
        output_horizon = horizon
        if output_horizon is None:
            output_horizon = self.default_horizon or x.shape[1]
        if output_horizon not in self.supported_horizons:
            raise ValueError(
                f"horizon must be one of {self.supported_horizons}, got {output_horizon}"
            )

        # 1) NPM 产生节点 prompt 和样本相关的动态知识图。
        prompts, knowledge_graph = self.npm(x, adjacency)
        # 2) 原始三通道与 D 维 prompt 拼接后送入共享 Graph WaveNet。
        augmented = torch.cat([x, prompts], dim=-1)
        graph = (
            knowledge_graph
            if self.use_knowledge_graph
            else _as_batched_adjacency(adjacency, x.shape[0], x.shape[2])
        )
        trunk_features = self.trunk(augmented, graph)
        # 3) 只取主干末端时间位置，再映射为共享规范小时特征。
        features = F.relu(self.end_projection(trunk_features[..., -1:]))
        prediction = self.output_decoder(features, horizon=output_horizon)
        if self.last_value_residual:
            # 两者均处于训练集归一化速度空间；[B,1,N,1] 会广播到 H=6 或 H=12。
            prediction = prediction + x[:, -1:, :, 0:1]
        if return_aux:
            return {
                "prediction": prediction,
                "prompts": prompts,
                "knowledge_graph": knowledge_graph,
                "trunk_features": trunk_features,
            }
        return prediction


__all__ = [
    "Equation5NodePromptingModule",
    "BaseTSJT",
    "equation5_prompt",
    "equation5_prompt_reference",
]
