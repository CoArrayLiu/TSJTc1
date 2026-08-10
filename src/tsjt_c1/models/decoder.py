"""用于混合时间分辨率训练的共享物理小时解码器。

协议用两种原生分辨率表示同一个一小时预测区间：

* 5 分钟城市预测 12 步：5、10、...、60 分钟；
* 10 分钟城市预测 6 步：10、20、...、60 分钟。

本类学习一个 12 位置规范 decoder。6 步预测确定性选择索引
``[1,3,5,7,9,11]``，使两种分辨率通过完全相同的投影参数反向传播，同时各数据集
保留原生目标时间戳。该索引映射只依赖时间分辨率，不读取未来输入或标签值。

同时支持 ``decoder(features, horizon=6)`` 与 ``output_heads["6"](features)``。
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class _BoundHorizon:
    """把一个共享 decoder 绑定到固定 horizon 的轻量可调用视图。"""

    decoder: "SharedCanonicalHourDecoder"
    horizon: int

    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        """转发给共享 decoder，并自动携带已绑定的 horizon。"""
        return self.decoder(features, horizon=self.horizon)


class SharedCanonicalHourDecoder(nn.Module):
    """解码共享规范小时并选出各城市原生预测位置。

    输入遵循 GWN 布局 ``[B,C,N,1]``，输出为 ``[B,H,N,1]``。唯一可训练部分是
    ``1×1`` 卷积，它一次产生 12 个规范位置；后续 ``index_select`` 无参数。
    ``canonical_steps`` 必须整除 ``hour_minutes``，每个原生 lead 也必须精确落在
    规范网格上。
    """

    def __init__(
        self,
        input_channels: int,
        canonical_steps: int = 12,
        supported_horizons: Sequence[int] = (6, 12),
        hour_minutes: int = 60,
        bias: bool = True,
    ) -> None:
        """验证物理时间网格，创建共享投影并注册每个 horizon 的索引 buffer。"""
        super().__init__()
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        if canonical_steps <= 0 or hour_minutes <= 0:
            raise ValueError("canonical_steps and hour_minutes must be positive")
        if hour_minutes % canonical_steps != 0:
            raise ValueError("canonical grid must divide the physical hour exactly")

        horizons = tuple(int(value) for value in supported_horizons)
        if not horizons or any(value <= 0 for value in horizons):
            raise ValueError("supported_horizons must contain positive values")
        if len(set(horizons)) != len(horizons):
            raise ValueError("supported_horizons must not contain duplicates")

        self.canonical_steps = int(canonical_steps)
        self.hour_minutes = int(hour_minutes)
        self.supported_horizons = horizons
        self.projection = nn.Conv2d(
            input_channels, self.canonical_steps, kernel_size=(1, 1), bias=bias
        )

        for horizon in self.supported_horizons:
            indices = self._build_indices(horizon)
            self.register_buffer(
                f"_selection_{horizon}", indices, persistent=False
            )

    def _build_indices(self, horizon: int) -> torch.Tensor:
        """根据物理分钟间隔计算某原生 horizon 对应的零基规范索引。"""
        if self.hour_minutes % horizon != 0:
            raise ValueError(
                f"horizon {horizon} does not divide {self.hour_minutes} minutes"
            )
        native_minutes = self.hour_minutes // horizon
        canonical_minutes = self.hour_minutes // self.canonical_steps
        if native_minutes % canonical_minutes != 0:
            raise ValueError(
                f"horizon {horizon} does not align with the canonical time grid"
            )
        stride = native_minutes // canonical_minutes
        indices = torch.arange(stride - 1, self.canonical_steps, stride)
        if indices.numel() != horizon:
            raise ValueError(
                f"horizon {horizon} cannot be represented by the canonical grid"
            )
        return indices.to(dtype=torch.long)

    def selection_indices(self, horizon: int) -> torch.Tensor:
        """返回原生 horizon 对应的规范零基位置。"""

        horizon = int(horizon)
        if horizon not in self.supported_horizons:
            raise ValueError(
                f"horizon must be one of {self.supported_horizons}, got {horizon}"
            )
        return getattr(self, f"_selection_{horizon}")

    def lead_minutes(self, horizon: int) -> torch.Tensor:
        """返回每个原生输出位置对应的实际未来分钟数。"""

        canonical_minutes = self.hour_minutes // self.canonical_steps
        return (self.selection_indices(horizon) + 1) * canonical_minutes

    def forward(self, features: torch.Tensor, horizon: int) -> torch.Tensor:
        """先投影完整 12 步规范小时，再选择 H=6 或 H=12 原生输出。"""
        if features.ndim != 4:
            raise ValueError("features must have shape [B,C,N,1]")
        if features.shape[-1] != 1:
            raise ValueError("shared decoder expects the final GWN time width to be 1")
        canonical = self.projection(features)
        indices = self.selection_indices(horizon)
        return canonical.index_select(dim=1, index=indices)

    # 提供类似 ModuleDict 的索引接口，但所有 horizon 实际共享同一个 projection。
    def __getitem__(self, horizon: str | int) -> _BoundHorizon:
        """校验 horizon 后返回绑定视图。"""
        value = int(horizon)
        self.selection_indices(value)  # validate eagerly, like ModuleDict.
        return _BoundHorizon(self, value)

    def __contains__(self, horizon: object) -> bool:
        """判断给定值能否解释为受支持的原生 horizon。"""
        try:
            return int(horizon) in self.supported_horizons  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    def keys(self) -> tuple[str, ...]:
        """以字符串形式返回 horizon 键，匹配 ModuleDict 的常用接口。"""
        return tuple(str(value) for value in self.supported_horizons)

    def __iter__(self) -> Iterator[str]:
        """迭代字符串 horizon 键。"""
        return iter(self.keys())
