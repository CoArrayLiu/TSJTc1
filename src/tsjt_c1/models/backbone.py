"""与节点数量无关的 TSJT 时空编码和 Graph WaveNet 基础组件。

本文件提供图邻接处理、时空模式编码器、prompt 工具以及动态 Graph WaveNet 主干。
其中 ``naive_prompt_attention``/``chunked_prompt_attention`` 是早期注意力解释的
参考实现，保留用于数值对照；正式 C1 使用 ``base.py`` 中不含 softmax 的审计版
Equation-(5) Gram 实现。

参考注意力把 ``K*T`` 个 bank 位置视作一个 key/value 序列：

    A_i = softmax(Z_i @ flatten(B).T / sqrt(D), dim=bank_token)
    P_i = A_i @ flatten(B)

分块版本采用两遍稳定 softmax，在不物化巨大 ``[B,N,T,K*T]`` 张量的情况下与朴素
版本数学等价，不使用时间摘要或低秩近似。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_batched_adjacency(
    adjacency: torch.Tensor, batch_size: int, node_count: int
) -> torch.Tensor:
    """把 ``[N,N]`` 或 ``[1,N,N]`` 邻接矩阵扩展为 ``[B,N,N]``。

    若输入已经是 batch 邻接矩阵，则要求 batch 大小与特征一致。这里只使用
    ``expand`` 创建视图，不复制每个 batch 的邻接数据。
    """
    if adjacency.ndim == 2:
        if adjacency.shape != (node_count, node_count):
            raise ValueError("adjacency node dimensions do not match x")
        return adjacency.unsqueeze(0).expand(batch_size, -1, -1)
    if adjacency.ndim != 3 or adjacency.shape[-2:] != (node_count, node_count):
        raise ValueError("adjacency must have shape [N,N] or [B,N,N]")
    if adjacency.shape[0] == 1:
        return adjacency.expand(batch_size, -1, -1)
    if adjacency.shape[0] != batch_size:
        raise ValueError("batched adjacency must have the same batch size as x")
    return adjacency


def _row_normalize(
    adjacency: torch.Tensor, *, add_self_loops: bool, signed: bool = False
) -> torch.Tensor:
    """对邻接矩阵逐行归一化，可选自环与有符号度数。

    prompt cosine graph 可能包含负边，因此 ``signed=True`` 时用绝对边权计算度数，
    但保留原边权符号参与消息传播。
    """
    if add_self_loops:
        identity = torch.eye(
            adjacency.shape[-1], device=adjacency.device, dtype=adjacency.dtype
        )
        adjacency = adjacency + identity
    degree_source = adjacency.abs() if signed else adjacency
    degree = degree_source.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return adjacency / degree


class DynamicGraphLayer(nn.Module):
    """处理 ``[B,T,N,D]`` 特征的节点数无关图层。

    输出是节点自身线性投影与归一化邻居聚合投影之和；参数维度只依赖特征通道，
    所以同一层可以跨不同节点数量的城市共享。
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        """创建分别作用于自身特征与邻居特征的两个线性映射。"""
        super().__init__()
        self.self_projection = nn.Linear(input_dim, output_dim)
        self.neighbor_projection = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """按 ``A_norm @ X`` 聚合邻居，保持 batch 和时间轴不变。"""
        graph = _as_batched_adjacency(adjacency, x.shape[0], x.shape[2])
        graph = _row_normalize(graph, add_self_loops=True)
        neighbors = torch.einsum("bij,btjd->btid", graph, x)
        return self.self_projection(x) + self.neighbor_projection(neighbors)


class SpatioTemporalPatternEncoder(nn.Module):
    """NPM 的时空模式编码器：GNN → 时间注意力 → GNN。

    两个子层均使用残差、dropout 和 LayerNorm。时间注意力对每个节点独立进行，
    即先把 ``[B,T,N,D]`` 重排成 ``[B*N,T,D]``，不会混合不同节点。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        dropout: float,
        canonical_steps: int = 12,
    ) -> None:
        """初始化输入投影、可学习规范时间嵌入、两层图层和多头时间注意力。"""
        super().__init__()
        self.canonical_steps = canonical_steps
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.temporal_embedding = nn.Parameter(
            torch.empty(1, canonical_steps, 1, hidden_dim)
        )
        nn.init.trunc_normal_(self.temporal_embedding, std=0.02)
        self.graph_1 = DynamicGraphLayer(hidden_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(hidden_dim)
        self.graph_2 = DynamicGraphLayer(hidden_dim, hidden_dim)
        self.norm_2 = nn.LayerNorm(hidden_dim)

    def _temporal_embedding(self, time_steps: int) -> torch.Tensor:
        """将规范 12 步时间嵌入线性插值到当前城市的 T（6 或 12）。"""
        embedding = self.temporal_embedding
        if time_steps == self.canonical_steps:
            return embedding
        # 只插值物理小时对应的时间轴，不改变 batch/节点广播轴和特征维。
        values = embedding[:, :, 0].transpose(1, 2)
        values = F.interpolate(
            values, size=time_steps, mode="linear", align_corners=False
        )
        return values.transpose(1, 2).unsqueeze(2)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """编码 ``[B,T,N,C]``，返回 ``[B,T,N,D]`` 时空模式。"""
        # z0 同时包含观测特征投影和物理时间位置编码。
        z0 = self.input_projection(x) + self._temporal_embedding(x.shape[1])
        graph_features = self.graph_1(z0, adjacency)
        b, t, n, d = graph_features.shape
        # 将每个“样本-节点”视作一条独立长度 T 的序列执行 self-attention。
        sequence = graph_features.permute(0, 2, 1, 3).reshape(b * n, t, d)
        attended, _ = self.attention(
            sequence, sequence, sequence, need_weights=False
        )
        attended = attended.reshape(b, n, t, d).permute(0, 2, 1, 3)
        z2 = self.norm_1(z0 + self.dropout(attended))
        return self.norm_2(z2 + self.dropout(self.graph_2(z2, adjacency)))


def interpolate_prompt_bank(
    prompt_bank: torch.Tensor, time_steps: int
) -> torch.Tensor:
    """只沿时间轴把规范 prompt bank ``[K,12,D]`` 插值到 ``[K,T,D]``。"""

    if prompt_bank.ndim != 3:
        raise ValueError("prompt_bank must have shape [K,T,D]")
    if prompt_bank.shape[1] == time_steps:
        return prompt_bank
    return F.interpolate(
        prompt_bank.transpose(1, 2),
        size=time_steps,
        mode="linear",
        align_corners=False,
    ).transpose(1, 2)


def naive_prompt_attention(
    encoded: torch.Tensor, prompt_bank: torch.Tensor
) -> torch.Tensor:
    """未分块的全局 token 注意力参考实现。

    ``encoded`` 为 ``[B,N,T,D]`` 节点模式，``prompt_bank`` 为 ``[K,T,D]``。
    将 bank 展平为 ``K*T`` 个 token 后计算 softmax，返回 ``[B,N,T,D]``。该函数
    主要用于验证分块版本，不是正式 C1 的 Equation-(5) 路径。
    """

    if encoded.ndim != 4 or prompt_bank.ndim != 3:
        raise ValueError("expected encoded [B,N,T,D] and bank [K,T,D]")
    if encoded.shape[2:] != prompt_bank.shape[1:]:
        raise ValueError("encoded and prompt bank time/feature dimensions differ")
    bank_tokens = prompt_bank.flatten(0, 1)
    logits = torch.einsum("bntd,md->bntm", encoded, bank_tokens)
    weights = torch.softmax(logits / math.sqrt(encoded.shape[-1]), dim=-1)
    return torch.einsum("bntm,md->bntd", weights, bank_tokens)


def chunked_prompt_attention(
    encoded: torch.Tensor,
    prompt_bank: torch.Tensor,
    node_chunk_size: int,
    bank_chunk_size: int,
) -> torch.Tensor:
    """与朴素注意力严格等价的显存受控实现。

    外层按节点分块，内层按 prompt pattern 分块。第一遍求全局最大 logit，第二遍
    累计 softmax 分子/分母，从而避免构造完整注意力矩阵且保持数值稳定。
    """

    if node_chunk_size <= 0 or bank_chunk_size <= 0:
        raise ValueError("attention chunk sizes must be positive")
    if encoded.ndim != 4 or prompt_bank.ndim != 3:
        raise ValueError("expected encoded [B,N,T,D] and bank [K,T,D]")
    if encoded.shape[2:] != prompt_bank.shape[1:]:
        raise ValueError("encoded and prompt bank time/feature dimensions differ")
    scale = math.sqrt(encoded.shape[-1])
    bank_tokens = prompt_bank.flatten(0, 1)
    node_outputs = []
    for node_start in range(0, encoded.shape[1], node_chunk_size):
        query = encoded[:, node_start : node_start + node_chunk_size]
        token_chunk_size = bank_chunk_size * prompt_bank.shape[1]
        # 第一遍遍历全部 K*T token 求全局最大值，作为稳定 softmax 的平移量。
        maximum: torch.Tensor | None = None
        for token_start in range(0, bank_tokens.shape[0], token_chunk_size):
            tokens = bank_tokens[token_start : token_start + token_chunk_size]
            logits = torch.einsum("bntd,md->bntm", query, tokens) / scale
            chunk_maximum = logits.amax(dim=-1, keepdim=True)
            maximum = (
                chunk_maximum
                if maximum is None
                else torch.maximum(maximum, chunk_maximum)
            )
        if maximum is None:
            raise ValueError("prompt bank must contain at least one token")
        # softmax 对整体平移不变；detach 最大值可释放第一遍计算图且不改变正确梯度。
        maximum = maximum.detach()
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros_like(query)
        for token_start in range(0, bank_tokens.shape[0], token_chunk_size):
            tokens = bank_tokens[token_start : token_start + token_chunk_size]
            logits = torch.einsum("bntd,md->bntm", query, tokens) / scale
            unnormalized = torch.exp(logits - maximum)
            denominator = denominator + unnormalized.sum(dim=-1, keepdim=True)
            numerator = numerator + torch.einsum(
                "bntm,md->bntd", unnormalized, tokens
            )
        node_outputs.append(numerator / denominator)
    return torch.cat(node_outputs, dim=1)


def prompt_knowledge_graph(prompts: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """按论文 Equation (7) 用 prompt 的 Frobenius 余弦相似度建图。

    输入 ``[B,N,T,D]`` 先把 ``T,D`` 展平，再计算节点两两余弦，输出动态邻接矩阵
    ``[B,N,N]``。每个样本都有自己的图。
    """

    if prompts.ndim != 4:
        raise ValueError("prompts must have shape [B,N,T,D]")
    flat = prompts.flatten(start_dim=2)
    normalized = F.normalize(flat, p=2, dim=-1, eps=eps)
    return normalized @ normalized.transpose(-1, -2)


class NodePromptingModule(nn.Module):
    """把时空模式映射为节点 prompt，并由 prompt 构造动态知识图的参考 NPM。"""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        prompt_bank_size: int,
        heads: int,
        dropout: float,
        canonical_steps: int,
        node_chunk_size: int,
        bank_chunk_size: int,
    ) -> None:
        """创建模式编码器和 ``[K,canonical_steps,D]`` 可学习 prompt bank。"""
        super().__init__()
        self.encoder = SpatioTemporalPatternEncoder(
            input_dim, hidden_dim, heads, dropout, canonical_steps
        )
        self.prompt_bank = nn.Parameter(
            torch.empty(prompt_bank_size, canonical_steps, hidden_dim)
        )
        nn.init.xavier_uniform_(self.prompt_bank)
        self.node_chunk_size = node_chunk_size
        self.bank_chunk_size = bank_chunk_size

    def forward(
        self, x: torch.Tensor, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 ``[B,T,N,D]`` prompt 与 ``[B,N,N]`` prompt 图。"""
        encoded = self.encoder(x, adjacency).permute(0, 2, 1, 3)
        bank = interpolate_prompt_bank(self.prompt_bank, x.shape[1])
        prompts = chunked_prompt_attention(
            encoded, bank, self.node_chunk_size, self.bank_chunk_size
        )
        graph = prompt_knowledge_graph(prompts)
        return prompts.permute(0, 2, 1, 3), graph


class DynamicDiffusionGraphConv(nn.Module):
    """支持 batch 动态邻接矩阵的 Graph WaveNet 扩散卷积。

    对每个 support 累计 1 到 ``order`` 阶图传播结果，与原输入沿通道拼接后通过
    ``1×1`` 卷积融合。
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        support_count: int,
        order: int,
        dropout: float,
    ) -> None:
        """根据 support 数量与扩散阶数计算拼接后的输入通道数。"""
        super().__init__()
        self.order = order
        self.dropout = dropout
        expanded_channels = input_channels * (1 + support_count * order)
        self.projection = nn.Conv2d(expanded_channels, output_channels, (1, 1))

    @staticmethod
    def _nconv(x: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """执行节点维图传播：``[B,C,N,T] × [B,N,N]``。"""
        return torch.einsum("bcnt,bnm->bcmt", x, support)

    def forward(
        self, x: torch.Tensor, supports: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        """生成多阶扩散特征，投影回输出通道并施加 dropout。"""
        features = [x]
        for support in supports:
            propagated = self._nconv(x, support)
            features.append(propagated)
            for _ in range(2, self.order + 1):
                propagated = self._nconv(propagated, support)
                features.append(propagated)
        output = self.projection(torch.cat(features, dim=1))
        return F.dropout(output, self.dropout, training=self.training)


class DynamicGraphWaveNetTrunk(nn.Module):
    """不含固定节点参数的门控空洞 Graph WaveNet 主干。

    每层先通过 tanh/sigmoid 门控时间卷积提取多尺度模式，再进行动态扩散图卷积，
    同时把 skip 分支对齐到当前时间宽度后累加。所有参数只依赖通道数，因此三个源
    城市和目标城市可以共享同一主干。
    """

    def __init__(
        self,
        input_dim: int,
        residual_channels: int,
        dilation_channels: int,
        skip_channels: int,
        blocks: int,
        layers: int,
        kernel_size: int,
        diffusion_order: int,
        dropout: float,
        use_reverse_support: bool = True,
    ) -> None:
        """按 blocks×layers 构造指数增长 dilation，并计算总感受野。"""
        super().__init__()
        if blocks <= 0 or layers <= 0 or kernel_size <= 1:
            raise ValueError("blocks/layers must be positive and kernel_size > 1")
        self.start_conv = nn.Conv2d(input_dim, residual_channels, (1, 1))
        self.use_reverse_support = use_reverse_support
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.graph_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        receptive_field = 1
        for _ in range(blocks):
            dilation = 1
            for _ in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        (1, kernel_size),
                        dilation=(1, dilation),
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        (1, kernel_size),
                        dilation=(1, dilation),
                    )
                )
                self.skip_convs.append(
                    nn.Conv2d(dilation_channels, skip_channels, (1, 1))
                )
                self.graph_convs.append(
                    DynamicDiffusionGraphConv(
                        dilation_channels,
                        residual_channels,
                        support_count=2 if use_reverse_support else 1,
                        order=diffusion_order,
                        dropout=dropout,
                    )
                )
                self.norms.append(nn.BatchNorm2d(residual_channels))
                receptive_field += (kernel_size - 1) * dilation
                dilation *= 2
        self.receptive_field = receptive_field

    def forward(
        self, x: torch.Tensor, knowledge_graph: torch.Tensor
    ) -> torch.Tensor:
        """处理 ``[B,T,N,C]``，输出 Graph WaveNet skip 特征 ``[B,Cs,N,T']``。"""
        # 外部统一使用 [B,T,N,C]；卷积内部改为 PyTorch 的 [B,C,N,T]。
        hidden = x.permute(0, 3, 2, 1)
        if hidden.shape[-1] < self.receptive_field:
            hidden = F.pad(hidden, (self.receptive_field - hidden.shape[-1], 0))
        hidden = self.start_conv(hidden)
        # prompt cosine 图可能有负权，正向图和转置反向图分别归一化。
        graph = _row_normalize(
            knowledge_graph, add_self_loops=True, signed=True
        )
        reverse_graph = _row_normalize(
            knowledge_graph.transpose(-1, -2), add_self_loops=True, signed=True
        )
        supports = (graph, reverse_graph) if self.use_reverse_support else (graph,)
        skip: torch.Tensor | None = None
        for filter_conv, gate_conv, skip_conv, graph_conv, norm in zip(
            self.filter_convs,
            self.gate_convs,
            self.skip_convs,
            self.graph_convs,
            self.norms,
        ):
            # 门控时间卷积相当于 filter⊙gate；它会缩短时间轴，残差需右对齐裁剪。
            residual = hidden
            gated = torch.tanh(filter_conv(residual)) * torch.sigmoid(
                gate_conv(residual)
            )
            layer_skip = skip_conv(gated)
            if skip is None:
                skip = layer_skip
            else:
                skip = skip[..., -layer_skip.shape[-1] :] + layer_skip
            hidden = graph_conv(gated, supports)
            hidden = norm(hidden + residual[..., -hidden.shape[-1] :])
        if skip is None:
            raise RuntimeError("Graph WaveNet trunk has no layers")
        return F.relu(skip)


class GraphBackboneModel(nn.Module):
    """NPM + 动态 Graph WaveNet + 6/12 独立头的基础参考模型。

    正式 C1 复用本文件的编码和图卷积组件，但在 ``BaseTSJT`` 中改用共享物理小时
    解码器。该类保留作为底层结构参考和训练引擎的兼容构造器。
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
    ) -> None:
        """解析兼容配置名并创建 NPM、GWN 主干、末端投影和输出头。"""
        super().__init__()
        if backbone != "graph_wavenet":
            raise ValueError("TSJT-C1 implements the Graph WaveNet backbone")
        if canonical_history is not None:
            canonical_steps = canonical_history
        if prompt_query_chunk_size is not None:
            node_chunk_size = prompt_query_chunk_size
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
        if horizon is not None and horizon not in supported_horizons:
            raise ValueError("default horizon is not in supported_horizons")
        self.default_horizon = horizon
        self.supported_horizons = tuple(int(value) for value in supported_horizons)
        self.use_knowledge_graph = use_knowledge_graph
        self.prompt_bank_rule = prompt_bank_rule
        self.npm = NodePromptingModule(
            input_dim,
            hidden_dim,
            prompt_bank_size,
            heads,
            dropout,
            canonical_steps,
            node_chunk_size,
            bank_chunk_size,
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
        self.output_heads = nn.ModuleDict(
            {
                str(output_horizon): nn.Conv2d(
                    end_channels, output_horizon, (1, 1)
                )
                for output_horizon in self.supported_horizons
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        horizon: int | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """执行参考前向；可选返回 prompt、知识图和主干特征用于诊断。"""
        if x.ndim != 4:
            raise ValueError("x must have shape [B,T,N,C]")
        output_horizon = horizon
        if output_horizon is None:
            output_horizon = self.default_horizon or x.shape[1]
        if output_horizon not in self.supported_horizons:
            raise ValueError(
                f"horizon must be one of {self.supported_horizons}, got {output_horizon}"
            )
        # prompt 既作为节点附加特征，也用于生成当前 batch 的动态图。
        prompts, knowledge_graph = self.npm(x, adjacency)
        augmented = torch.cat([x, prompts], dim=-1)
        graph = knowledge_graph if self.use_knowledge_graph else _as_batched_adjacency(
            adjacency, x.shape[0], x.shape[2]
        )
        trunk_features = self.trunk(augmented, graph)
        features = F.relu(self.end_projection(trunk_features[..., -1:]))
        raw = self.output_heads[str(output_horizon)](features)
        prediction = raw.squeeze(-1).unsqueeze(-1)
        if return_aux:
            return {
                "prediction": prediction,
                "prompts": prompts,
                "knowledge_graph": knowledge_graph,
                "trunk_features": trunk_features,
            }
        return prediction
