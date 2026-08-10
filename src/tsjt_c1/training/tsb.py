"""Task-Specific Block（TSB）源梯度冲突过滤与显式参数更新。

TSJT-C1 不使用常规 optimizer.step。每次 logical update 分别计算目标梯度 ``g_t``
和多源梯度 ``g_s``，将源梯度分成相对目标梯度的平行分量与正交分量。若平行分量
方向与目标相反，则只删除该冲突分量；正交分量始终保留。随后用两个独立步长直接
更新参数。
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


def filter_source_gradient(
    target_gradient: torch.Tensor,
    source_gradient: torch.Tensor,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """对单个参数张量应用 TSJT Equation (9–10)。

    设源梯度 ``g_s = parallel + orthogonal``。当 ``<g_s,g_t> >= 0`` 时返回完整
    ``g_s``；当内积为负时返回 ``orthogonal``。若目标梯度范数接近零，方向没有
    定义，为避免源域独自改变目标模型，本实现返回零源梯度。

    第二个返回值保留分解结果和内积，供训练日志统计冲突率。
    """

    # 一个参数张量整体作为一个 TSB block，展平只用于内积，不改变原形状更新。
    target_flat = target_gradient.reshape(-1)
    source_flat = source_gradient.reshape(-1)
    target_norm_sq = torch.dot(target_flat, target_flat)
    if target_norm_sq <= eps:
        zero = torch.zeros_like(source_gradient)
        return zero, {
            "parallel": zero,
            "orthogonal": zero,
            "dot": torch.zeros((), device=source_gradient.device, dtype=source_gradient.dtype),
        }
    dot = torch.dot(source_flat, target_flat)
    parallel = (dot / target_norm_sq) * target_gradient
    orthogonal = source_gradient - parallel
    filtered = orthogonal + (parallel if dot >= 0 else torch.zeros_like(parallel))
    return filtered, {"parallel": parallel, "orthogonal": orthogonal, "dot": dot}


def _gradients(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    retain_graph: bool,
) -> tuple[torch.Tensor | None, ...]:
    """通过 autograd 计算损失对全部共享参数的梯度。

    ``allow_unused=True`` 允许某个 horizon 暂时不经过个别参数；对应位置返回 None，
    后续更新会把它视作零梯度。
    """
    return torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )


@torch.no_grad()
def _apply(
    parameters: list[torch.nn.Parameter],
    target_gradients: tuple[torch.Tensor | None, ...],
    source_gradients: tuple[torch.Tensor | None, ...],
    gamma_target: float,
    gamma_source: float,
) -> dict[str, float]:
    """在 ``no_grad`` 环境中原地执行目标更新和过滤后的源更新。

    更新公式为 ``theta -= gamma_target*g_t + gamma_source*filtered(g_s)``。
    返回参与比较的参数张量数、冲突张量数和冲突比例。
    """
    conflict_count = 0
    compared_count = 0
    for parameter, target_gradient, source_gradient in zip(
        parameters, target_gradients, source_gradients
    ):
        if target_gradient is None and source_gradient is None:
            continue
        if target_gradient is None:
            target_gradient = torch.zeros_like(parameter)
        if source_gradient is None:
            source_gradient = torch.zeros_like(parameter)
        filtered, parts = filter_source_gradient(target_gradient, source_gradient)
        compared_count += 1
        conflict_count += int(parts["dot"].item() < 0)
        # 不创建 optimizer state；两个来源的梯度使用各自固定步长直接相加。
        parameter.add_(target_gradient, alpha=-gamma_target)
        parameter.add_(filtered, alpha=-gamma_source)
    return {
        "compared_tensors": float(compared_count),
        "conflicting_tensors": float(conflict_count),
        "conflict_ratio": float(conflict_count / max(1, compared_count)),
    }


def apply_tsb_gradients(
    parameters: Iterable[torch.nn.Parameter],
    target_gradients: tuple[torch.Tensor | None, ...],
    source_gradients: tuple[torch.Tensor | None, ...],
    gamma_target: float,
    gamma_source: float,
) -> dict[str, float]:
    """接收已经计算好的目标/源梯度元组并应用一次 TSB 更新。

    正式训练使用此接口，因为三个源城市的梯度需要先按样本数聚合后再统一过滤。
    """
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if len(trainable) != len(target_gradients) or len(trainable) != len(source_gradients):
        raise ValueError("gradient tuple length does not match trainable parameter count")
    return _apply(
        trainable,
        target_gradients,
        source_gradients,
        gamma_target=gamma_target,
        gamma_source=gamma_source,
    )


def apply_tsb_update(
    parameters: Iterable[torch.nn.Parameter],
    target_loss: torch.Tensor,
    source_loss: torch.Tensor,
    gamma_target: float,
    gamma_source: float,
) -> dict[str, float]:
    """从两个标量损失内部计算梯度并立即应用 TSB 更新的便捷接口。

    目标梯度计算时保留计算图，随后再求源梯度；适用于目标/源损失仍在同一图中的
    简单调用。正式多源训练通常使用 ``apply_tsb_gradients``。
    """
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    target_gradients = _gradients(target_loss, trainable, retain_graph=True)
    source_gradients = _gradients(source_loss, trainable, retain_graph=False)
    return _apply(
        trainable,
        target_gradients,
        source_gradients,
        gamma_target=gamma_target,
        gamma_source=gamma_source,
    )
