"""唯一正式模型 TSJT-C1：在基础 TSJT 上增加 CPRR 直接残差读出。

CPRR 不改变 Dense Equation-(5) NPM、余弦 prompt 图、Graph WaveNet、共享 decoder
或 last-value residual，而是从 dense prompt 直接增加一个目标条件残差：

    U = P[:, -1] - mean_t(P)
    C = layer_norm_D(U, affine=False)
    Z = Linear_D_to_12(C)
    delta_12 = cap * tanh(Z / cap)
    prediction = baseline_prediction + select_native(delta_12)

读出层权重和偏置均为零初始化，所以初始预测与基础模型逐位一致，但读出层第一步
即可获得梯度。一套 12 行规范参数同时服务两个 horizon；H=6 选择零基索引
``[1,3,5,7,9,11]``。所有参数仍纳入同一阶段 TSB 更新。

CPRR 是本项目的工程扩展，不应表述为原 TSJT 论文组件。
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTSJT, Equation5NodePromptingModule


class TSJTC1(BaseTSJT):
    """完整 Dense TSJT-C1，带零初始化的规范 prompt residual readout。

    ``cprr_cap`` 使用训练归一化速度单位，默认 0.5。``cap*tanh(logits/cap)`` 在
    限幅的同时保持零点导数为 1。正式表示固定 D=32、规范输出 12 步，因此 CPRR
    参数量恒为 ``32*12+12=396``，便于检测架构漂移。
    """

    _CONTEXT_EPS = 1e-5

    def __init__(
        self,
        *args: Any,
        cprr_cap: float = 0.5,
        prompt_feature_mode: str = "npm",
        graph_mode: str = "prompt",
        decoder_mode: str = "shared",
        disable_npm_when_unused: bool = True,
        last_value_residual: bool = True,
        use_knowledge_graph: bool = True,
        **kwargs: Any,
    ) -> None:
        """验证 C1 身份约束、构造基础模型，并创建零初始化 CPRR 线性读出。"""
        if isinstance(cprr_cap, bool):
            raise TypeError("cprr_cap must be a finite positive float")
        cap = float(cprr_cap)
        if not math.isfinite(cap) or cap <= 0.0:
            raise ValueError("cprr_cap must be a finite positive float")
        if str(decoder_mode).lower() != "shared":
            raise ValueError("CPRR requires decoder_mode='shared'")
        if not isinstance(last_value_residual, bool) or not last_value_residual:
            raise ValueError("CPRR requires last_value_residual=True")
        if not isinstance(use_knowledge_graph, bool) or not use_knowledge_graph:
            raise ValueError("CPRR requires use_knowledge_graph=True")

        if str(prompt_feature_mode).lower() != "npm":
            raise ValueError("TSJT-C1 requires prompt_feature_mode='npm'")
        if str(graph_mode).lower() != "prompt":
            raise ValueError("TSJT-C1 requires graph_mode='prompt'")
        if not isinstance(disable_npm_when_unused, bool):
            raise TypeError("disable_npm_when_unused must be a boolean")
        super().__init__(
            *args,
            last_value_residual=last_value_residual,
            use_knowledge_graph=use_knowledge_graph,
            **kwargs,
        )
        self.prompt_feature_mode = "npm"
        self.graph_mode = "prompt"
        self.decoder_mode = "shared"
        self.disable_npm_when_unused = disable_npm_when_unused
        self.npm_disabled = False
        npm = getattr(self, "npm", None)
        if not isinstance(npm, Equation5NodePromptingModule):
            raise TypeError("CPRR requires the original dense Equation-(5) NPM")
        if self.supported_horizons != (6, 12):
            raise ValueError("CPRR requires supported_horizons=(6, 12)")
        hidden_dim = int(npm.prompt_bank.shape[-1])
        if hidden_dim != 32:
            raise ValueError("CPRR requires hidden_dim=32")
        if int(self.output_decoder.canonical_steps) != 12:
            raise ValueError("CPRR requires a 12-position canonical decoder")

        # 线性层对每个节点独立地把 32 维 context 映射到 12 个规范未来位置。
        self.cprr_cap = cap
        self.cprr_readout = nn.Linear(32, 12, bias=True)
        nn.init.zeros_(self.cprr_readout.weight)
        nn.init.zeros_(self.cprr_readout.bias)

    @classmethod
    def cprr_context(cls, prompts: torch.Tensor) -> torch.Tensor:
        """构造 ``[B,N,32]`` 的“末时刻减时间均值”prompt context。

        该差分突出最近状态相对整段历史的变化，再用无仿射 LayerNorm 消除每个节点
        context 的尺度差异。
        """

        if prompts.ndim != 4 or prompts.shape[-1] != 32:
            raise ValueError("CPRR prompts must have shape [B,T,N,32]")
        centered = prompts[:, -1] - prompts.mean(dim=1)
        return F.layer_norm(
            centered,
            normalized_shape=(32,),
            weight=None,
            bias=None,
            eps=cls._CONTEXT_EPS,
        )

    def cprr_delta(
        self, prompts: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算 CPRR 残差并映射到城市原生 horizon。

        返回顺序为：原生 ``[B,H,N,1]`` delta、规范 12 步 delta、未限幅 logits、
        归一化 context。H=6 只选择对应 10/20/.../60 分钟的位置。
        """

        context = self.cprr_context(prompts)
        logits = self.cprr_readout(context).permute(0, 2, 1).unsqueeze(-1)
        canonical_delta = self.cprr_cap * torch.tanh(logits / self.cprr_cap)
        indices = self.output_decoder.selection_indices(horizon).to(logits.device)
        native_delta = canonical_delta.index_select(dim=1, index=indices)
        return native_delta, canonical_delta, logits, context

    @property
    def cprr_parameter_count(self) -> int:
        """返回 CPRR 独有读出层参数量；冻结结构中应恒为 396。"""
        return sum(parameter.numel() for parameter in self.cprr_readout.parameters())

    def cprr_report(self) -> dict[str, int | float | bool]:
        """生成 Dense NPM、CPRR 和全模型参数身份报告。"""
        return {
            "dense_equation5_npm_active": isinstance(
                self.npm, Equation5NodePromptingModule
            ),
            "dense_prompt_bank_parameter_count": int(self.npm.prompt_bank.numel()),
            "cprr_parameter_count": self.cprr_parameter_count,
            "cprr_cap": self.cprr_cap,
            "total_parameter_count": sum(
                parameter.numel() for parameter in self.parameters()
            ),
        }

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        horizon: int | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """先执行完整基础前向，再把有界 CPRR delta 加到基础预测。

        ``return_aux=True`` 会返回残差强度、相对 decoder residual 的 RMS 比值等诊断，
        训练常规路径仅返回最终预测以减少额外字典操作。
        """
        # 刻意委托基础前向，保证 CPRR 不重排或重复实现 NPM/GWN/decoder 主链。
        base = super().forward(
            x, adjacency, horizon=horizon, return_aux=True
        )
        if not isinstance(base, dict):
            raise RuntimeError("baseline TSJT path did not return diagnostics")
        output_horizon = int(horizon or self.default_horizon or x.shape[1])
        # CPRR 始终读取审计后的 dense prompt，而不是任何潜在的特征替代表示。
        dense_prompts = base.get("npm_prompts", base["prompts"])
        delta, canonical_delta, logits, context = self.cprr_delta(
            dense_prompts, output_horizon
        )
        baseline_prediction = base["prediction"]
        prediction = baseline_prediction + delta
        if not return_aux:
            return prediction

        # 下列量仅用于诊断 CPRR 相对基础 decoder 残差的幅度，不参与最终计算图分支。
        last_value = x[:, -1:, :, 0:1]
        decoder_residual = baseline_prediction - last_value
        delta_rms = delta.float().square().mean().sqrt()
        decoder_rms = decoder_residual.float().square().mean().sqrt()
        ratio = delta_rms / decoder_rms.clamp_min(1e-12)
        result = dict(base)
        result.update(
            {
                "prediction": prediction,
                "baseline_prediction": baseline_prediction,
                "last_value": last_value,
                "decoder_residual": decoder_residual,
                "cprr_context": context,
                "cprr_canonical_logits": logits,
                "cprr_canonical_delta": canonical_delta,
                "cprr_delta": delta,
                "cprr_delta_rms": delta_rms,
                "cprr_decoder_residual_rms": decoder_rms,
                "cprr_delta_to_decoder_rms_ratio": ratio,
            }
        )
        return result


CanonicalPromptResidualReadout = TSJTC1


__all__ = ["CanonicalPromptResidualReadout", "TSJTC1"]
