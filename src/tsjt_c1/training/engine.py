"""TSJT-C1 的底层训练引擎与可复现运行工具。

本模块承担五类职责：运行环境/配置身份记录、四城市数据组装、确定性 epoch loader、
三源样本加权梯度计算与 fixed TSB 更新，以及测试指标累计和 checkpoint 生命周期。
正式入口 ``tsjt_c1.run`` 在此基础上增加更严格的单模型身份和一次测试状态机。

训练中一个目标 batch 对应一个 logical update；每次同时从三个源城市取得总计 124
个样本（42/41/41），先对每个样本的 ``H×N`` MAE 求均值，再对 124 个样本等权
平均。这样节点数或 horizon 较大的城市不会仅因元素更多而支配源梯度。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from tsjt_c1.models.backbone import GraphBackboneModel
from tsjt_c1.training.tsb import apply_tsb_gradients, filter_source_gradient
from tsjt_c1.data.loaders import (
    CITY_PROTOCOLS,
    build_source_train_loader,
    build_target_train_test_loaders,
)
from tsjt_c1.data.pipeline import ForecastMetricAccumulator, SpeedStats


FORMAL_EPOCHS = 50
FORMAL_TARGET_BATCH = 4
FORMAL_SOURCE_BATCH = 124
FORMAL_SOURCE_CITY_BATCH = 42
ModelBuilder = Callable[[Mapping[str, Any]], torch.nn.Module]


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    """把 mapping 规范序列化后计算 SHA256，消除字典键顺序影响。"""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sampled_file_hash(path: Path, sample_bytes: int = 1 << 20) -> str:
    """用文件首尾各一段及文件长度生成低成本数据指纹。

    这用于记录运行所用大体积 ``.npy`` 的身份，不替代发布清单中的完整 SHA256。
    """
    digest = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if size > sample_bytes:
            handle.seek(max(0, size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    digest.update(str(size).encode("ascii"))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """先写同目录临时文件再原子替换目标 JSON，避免中断留下半文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    """向逐 epoch JSONL 追加一条紧凑记录；每行都是独立合法 JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _set_seed(seed: int) -> None:
    """同步设置 Python、NumPy、CPU/CUDA PyTorch 随机种子并启用确定性 cuDNN。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _rng_state() -> dict[str, Any]:
    """捕获所有随机数生成器状态，随 checkpoint 保存以支持精确恢复。"""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    """从 checkpoint 恢复 Python、NumPy、Torch 及可用 CUDA RNG 状态。"""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _resolve_device(requested: str) -> torch.device:
    """解析设备字符串；明确请求 CUDA 而不可用时禁止静默退回 CPU。"""
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} was requested but is unavailable")
    return device


def _git_revision(root: Path) -> str | None:
    """尝试读取当前 Git commit；非 Git 目录或命令失败时返回 None。"""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _forbidden_validation_keys(mapping: Mapping[str, Any]) -> set[str]:
    """找出可能引入验证选择、早停或 gate 调参的配置键。"""
    forbidden_words = ("validation", "early_stop", "patience", "best_val", "gate")
    return {
        str(key)
        for key in mapping
        if any(word in str(key).lower() for word in forbidden_words)
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    """训练引擎级协议校验。

    检查城市集合、原生 H/T、三天目标训练、split-contained 窗口、Graph WaveNet、
    三输入通道、反向 support 和 prompt 图。正式模式还锁定 50 epoch 与 batch 大小。
    顶层正式入口会在此基础上施加更严格的 C1-only 约束。
    """
    required = {"run", "data", "model", "train", "evaluation"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"configuration is missing sections: {sorted(missing)}")
    run = config["run"]
    data = config["data"]
    model = config["model"]
    train = config["train"]
    evaluation = config["evaluation"]
    target = data["target"]
    target_city = str(target["city"]).lower()
    if target_city not in CITY_PROTOCOLS:
        raise ValueError(f"unsupported target city: {target_city}")
    sources = [str(item["city"]).lower() for item in data["sources"]]
    expected_sources = set(CITY_PROTOCOLS) - {target_city}
    if len(sources) != 3 or len(set(sources)) != 3 or set(sources) != expected_sources:
        raise ValueError(
            f"{target_city} sources must contain exactly {sorted(expected_sources)}"
        )
    for item in (*data["sources"], target):
        city = str(item["city"]).lower()
        expected = CITY_PROTOCOLS[city]
        observed = {key: int(item[key]) for key in expected}
        if observed != expected:
            raise ValueError(f"protocol for {city} must be {expected}; got {observed}")
    if int(target.get("train_days", 0)) != 3:
        raise ValueError("target train_days must be exactly 3")
    if int(target.get("train_stride", 0)) != 1 or int(target.get("test_stride", 0)) != 1:
        raise ValueError("formal target train/test stride must be 1")
    if bool(target.get("allow_prior_split_history", True)):
        raise ValueError("target windows must be split-contained")
    forbidden = (
        _forbidden_validation_keys(target)
        | _forbidden_validation_keys(train)
        | _forbidden_validation_keys(evaluation)
    )
    if forbidden:
        raise ValueError(f"validation/selection keys are forbidden: {sorted(forbidden)}")
    if str(model.get("backbone", "")).lower() != "graph_wavenet":
        raise ValueError("TSJT-C1 requires graph_wavenet")
    if int(model["input_dim"]) != 3:
        raise ValueError("input_dim must be speed + TOD + DOW = 3")
    if not bool(model.get("use_reverse_support", False)):
        raise ValueError("formal model requires reverse graph support")
    if not bool(model.get("use_knowledge_graph", False)):
        raise ValueError("formal model requires the prompt knowledge graph")

    formal = bool(run.get("formal", False))
    if not formal and not bool(run.get("test_override", False)):
        raise ValueError("non-formal runs require explicit run.test_override=true")
    epochs = int(train["epochs"])
    evaluate_epoch = int(evaluation["evaluate_once_after_epoch"])
    if formal:
        if epochs != FORMAL_EPOCHS:
            raise ValueError("formal TSJT-C1 training must run exactly 50 epochs")
        if int(train["target_batch_size"]) != FORMAL_TARGET_BATCH:
            raise ValueError("formal target batch size must be 4")
        if int(train["source_batch_size"]) != FORMAL_SOURCE_BATCH:
            raise ValueError("formal logical source batch size must be 124")
        if int(train["source_microbatch_size"]) != FORMAL_SOURCE_CITY_BATCH:
            raise ValueError(
                "formal source loader batch size must be 42 for the 42/41/41 split"
            )
        if int(train.get("save_every", 0)) != 1:
            raise ValueError("formal resumability requires save_every=1")
        if evaluate_epoch != FORMAL_EPOCHS:
            raise ValueError("formal test evaluation must be after epoch 50")
        expected_name = f"tsjt_c1_{target_city.replace('-', '')}_seed{int(run['seed'])}"
        if Path(run["output_dir"]).name.lower() != expected_name:
            raise ValueError(f"formal output namespace must be {expected_name!r}")
    elif epochs <= 0 or evaluate_epoch != epochs:
        raise ValueError("test override must evaluate only after its final epoch")
    if int(train["source_microbatch_size"]) <= 0:
        raise ValueError("source_microbatch_size must be positive")
    if not bool(evaluation.get("full_test", False)):
        raise ValueError("evaluation.full_test must be true")
    if int(evaluation["batch_size"]) <= 0:
        raise ValueError("evaluation batch size must be positive")


def _build_model(config: Mapping[str, Any]) -> GraphBackboneModel:
    """构建训练引擎的基础参考模型；正式 C1 通过外部 model_builder 替换它。"""
    cfg = config["model"]
    return GraphBackboneModel(
        input_dim=int(cfg["input_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        prompt_bank_size=int(cfg["prompt_bank_size"]),
        heads=int(cfg["heads"]),
        dropout=float(cfg["dropout"]),
        canonical_steps=int(cfg["canonical_history"]),
        node_chunk_size=int(cfg["prompt_query_chunk_size"]),
        bank_chunk_size=int(cfg["prompt_query_chunk_size"]),
        residual_channels=int(cfg["gwn_residual_channels"]),
        dilation_channels=int(cfg["gwn_dilation_channels"]),
        skip_channels=int(cfg["gwn_skip_channels"]),
        end_channels=int(cfg["gwn_end_channels"]),
        blocks=int(cfg["gwn_blocks"]),
        layers=int(cfg["gwn_layers"]),
        kernel_size=int(cfg["gwn_kernel_size"]),
        diffusion_order=int(cfg["diffusion_order"]),
        supported_horizons=(6, 12),
    )


def _prepare_output(
    config_path: Path,
    config: dict[str, Any],
    device: torch.device,
    argv: Sequence[str],
) -> tuple[Path, str]:
    """创建/验证输出目录并记录完整启动元数据。

    ``resolved_config.json`` 是目录归属凭证；若已存在，其规范哈希必须与当前配置
    一致。每次启动都会追加 launches.jsonl，并更新 latest_launch.json。
    """
    output_dir = Path(config["run"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_hash = _canonical_hash(config)
    resolved = output_dir / "resolved_config.json"
    if resolved.exists():
        previous = json.loads(resolved.read_text(encoding="utf-8"))
        if _canonical_hash(previous) != config_hash:
            raise RuntimeError(
                f"output directory {output_dir} belongs to an incompatible config"
            )
    else:
        _atomic_json(resolved, config)
        shutil.copy2(config_path, output_dir / "launch_config.yaml")
    # 元数据足以定位 Python/PyTorch/CUDA/GPU 与启动命令，不包含训练数据本体。
    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(argv),
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "config_hash": config_hash,
        "config_path": str(config_path.resolve()),
        "git_revision": _git_revision(Path(__file__).resolve().parents[3]),
    }
    if device.type == "cuda":
        metadata.update(
            {
                "cuda": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(device),
            }
        )
    _append_jsonl(output_dir / "launches.jsonl", metadata)
    _atomic_json(output_dir / "latest_launch.json", metadata)
    return output_dir, config_hash


def _fingerprint_data(
    data_root: Path, cities: Iterable[str], output_dir: Path
) -> dict[str, Any]:
    """记录每个城市数据/邻接文件的绝对路径、形状和采样指纹。"""
    evidence: dict[str, Any] = {}
    for city in cities:
        dataset = data_root / city / "dataset.npy"
        adjacency = data_root / city / "matrix.npy"
        values = np.load(dataset, mmap_mode="r")
        graph = np.load(adjacency, mmap_mode="r")
        evidence[city] = {
            "dataset_path": str(dataset.resolve()),
            "dataset_shape": list(values.shape),
            "dataset_fingerprint": _sampled_file_hash(dataset),
            "adjacency_path": str(adjacency.resolve()),
            "adjacency_shape": list(graph.shape),
            "adjacency_fingerprint": _sampled_file_hash(adjacency),
        }
    _atomic_json(output_dir / "city_fingerprints.json", evidence)
    return evidence


def _epoch_loader(
    dataset: Any,
    *,
    batch_size: int,
    seed: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """按 epoch 专属 seed 创建 loader，使恢复后样本顺序可重现。"""
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _source_allocation(source_cities: Sequence[str], logical_batch: int) -> dict[str, int]:
    """把逻辑源 batch 尽量均匀分给各城市，余数按配置顺序分配。

    正式参数 ``124/3`` 得到第一个城市 42、其余两个城市 41，即 42/41/41。
    """
    if not source_cities or logical_batch < len(source_cities):
        raise ValueError("logical source batch is too small")
    base, remainder = divmod(logical_batch, len(source_cities))
    return {
        city: base + int(index < remainder)
        for index, city in enumerate(source_cities)
    }


class _CyclingCursor:
    """可循环取样且能精确切出指定样本数的 DataLoader 游标。

    当某源城市数据较短时自动从头循环；若一次 loader batch 只消费了一部分，剩余
    样本放入 ``pending``，下次优先使用，避免丢样本或改变逻辑 batch 大小。
    """
    def __init__(self, loader: DataLoader) -> None:
        """绑定 loader、创建首个 iterator，并初始化 pending/cycle 计数。"""
        self.loader = loader
        self.iterator = iter(loader)
        self.pending: tuple[torch.Tensor, torch.Tensor] | None = None
        self.cycles = 0

    def _next(self) -> tuple[torch.Tensor, torch.Tensor]:
        """取得下一 batch；耗尽时重建 iterator 并增加循环计数。"""
        try:
            return next(self.iterator)
        except StopIteration:
            self.cycles += 1
            self.iterator = iter(self.loader)
            return next(self.iterator)

    def take(self, count: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """精确取得 ``count`` 个样本，可能跨 batch 或跨数据集循环边界。"""
        chunks: list[tuple[torch.Tensor, torch.Tensor]] = []
        remaining = count
        while remaining:
            x, y = self.pending if self.pending is not None else self._next()
            self.pending = None
            used = min(remaining, x.shape[0])
            chunks.append((x[:used], y[:used]))
            if used < x.shape[0]:
                self.pending = (x[used:], y[used:])
            remaining -= used
        return chunks


def _add_gradients(
    accumulator: list[torch.Tensor | None],
    gradients: Sequence[torch.Tensor | None],
) -> None:
    """把一个 microbatch 的梯度逐参数累加到 detached accumulator。"""
    for index, gradient in enumerate(gradients):
        if gradient is None:
            continue
        value = gradient.detach()
        accumulator[index] = (
            value.clone() if accumulator[index] is None else accumulator[index] + value
        )


def _sample_weighted_source_gradients(
    model: torch.nn.Module,
    parameters: Sequence[torch.nn.Parameter],
    microbatches: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    logical_sample_count: int,
    device: torch.device,
) -> tuple[tuple[torch.Tensor | None, ...], float, int]:
    """计算跨城市“逐样本等权精确均值”的源梯度。

    每个样本先在本城市原生 ``H×N`` 输出上平均 MAE，再统一除以逻辑样本总数。
    因此 H=12/N=207 与 H=6/N=627 的样本权重相同。各城市分别 forward，以便使用
    自己的邻接矩阵和 horizon；梯度随后按共享参数位置累加。
    """

    accumulated: list[torch.Tensor | None] = [None] * len(parameters)
    loss_sum = 0.0
    observed = 0
    # 源 forward 保持 train 模式；每城一次真实 42/41/41 batch，使 BN/dropout 的
    # 语义符合一阶段联合训练，而不是许多极小 ghost batch。
    for x, y, adjacency in microbatches:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        prediction = model(x, adjacency, horizon=y.shape[1])
        if prediction.shape != y.shape:
            raise RuntimeError(
                f"source prediction shape {prediction.shape} != target {y.shape}"
            )
        per_sample = (prediction - y).abs().flatten(1).mean(dim=1)
        weighted_loss = per_sample.sum() / logical_sample_count
        gradients = torch.autograd.grad(
            weighted_loss,
            parameters,
            allow_unused=True,
            retain_graph=False,
        )
        _add_gradients(accumulated, gradients)
        loss_sum += float(per_sample.detach().sum())
        observed += int(x.shape[0])
    if observed != logical_sample_count:
        raise RuntimeError(
            f"source logical batch expected {logical_sample_count}, got {observed}"
        )
    return tuple(accumulated), loss_sum / observed, observed


def _gradient_diagnostics(
    target: Sequence[torch.Tensor | None],
    source: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
    gamma_target: float,
    gamma_source: float,
) -> dict[str, float]:
    """在不修改参数的情况下计算目标/源/过滤源/最终更新的全局 L2 范数。

    同时按参数张量统计 TSB 冲突数。正式训练会将这里的冲突率与实际更新函数返回值
    交叉核对，防止诊断公式和应用公式发生漂移。
    """
    target_sq = source_sq = filtered_sq = update_sq = 0.0
    conflicts = compared = 0
    for parameter, target_gradient, source_gradient in zip(parameters, target, source):
        if target_gradient is None and source_gradient is None:
            continue
        tg = torch.zeros_like(parameter) if target_gradient is None else target_gradient
        sg = torch.zeros_like(parameter) if source_gradient is None else source_gradient
        filtered, parts = filter_source_gradient(tg, sg)
        compared += 1
        conflicts += int(float(parts["dot"]) < 0)
        target_sq += float(tg.detach().square().sum())
        source_sq += float(sg.detach().square().sum())
        filtered_sq += float(filtered.detach().square().sum())
        update = gamma_target * tg + gamma_source * filtered
        update_sq += float(update.detach().square().sum())
    return {
        "target_grad_norm": math.sqrt(target_sq),
        "source_grad_norm": math.sqrt(source_sq),
        "filtered_source_grad_norm": math.sqrt(filtered_sq),
        "update_norm": math.sqrt(update_sq),
        "compared_tensors": float(compared),
        "conflicting_tensors": float(conflicts),
        "conflict_ratio": conflicts / max(compared, 1),
    }


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """用 ``torch.save`` 写临时文件后原子替换 checkpoint。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config_hash: str,
    *,
    restore_rng: bool,
) -> dict[str, Any]:
    """加载 checkpoint，核对配置哈希，恢复模型并可选恢复 RNG 状态。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("config_hash") != config_hash:
        raise RuntimeError(f"checkpoint {path} is incompatible with active config")
    model.load_state_dict(payload["model"])
    if restore_rng:
        _restore_rng_state(payload["rng_state"])
    return payload


def _ensure_log_matches_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """确保 JSONL 不超前于 checkpoint，并仅修复可证明安全的一条尾记录。"""
    records: list[dict[str, Any]] = []
    if path.exists():
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    checkpoint_epoch = int(payload["epoch"])
    if records and int(records[-1]["epoch"]) > checkpoint_epoch:
        raise RuntimeError("training log is ahead of the durable checkpoint")
    if len(records) < checkpoint_epoch:
        if len(records) != checkpoint_epoch - 1:
            raise RuntimeError("training log/checkpoint gap cannot be repaired safely")
        _append_jsonl(path, payload["epoch_record"])


def _make_data(
    config: Mapping[str, Any], device: torch.device, output_dir: Path
) -> tuple[
    dict[str, DataLoader],
    torch.Tensor,
    SpeedStats,
    dict[str, Any],
    dict[str, DataLoader],
    dict[str, torch.Tensor],
    dict[str, Any],
]:
    """按配置组装目标 loaders、三个源 loaders、邻接矩阵与协议记录。

    邻接矩阵提前移动到训练设备；DataLoader batch 仍在 CPU，并在循环中异步传输。
    正式模式额外检查 prompt bank 大小等于目标节点数的一半 ``floor(N/2)``。
    """
    data = config["data"]
    run = config["run"]
    train = config["train"]
    root = Path(data["root"])
    target = data["target"]
    pin_memory = device.type == "cuda"
    # 目标域只包含 train/test，无 validation；速度统计量只来自前三天。
    target_loaders, target_adjacency, stats, target_protocol = (
        build_target_train_test_loaders(
            root,
            target["city"],
            batch_size=int(train["target_batch_size"]),
            seed=int(run["seed"]),
            train_days=int(target["train_days"]),
            train_stride=int(target["train_stride"]),
            test_stride=int(target["test_stride"]),
            num_workers=int(run.get("num_workers", 0)),
            pin_memory=pin_memory,
        )
    )
    source_loaders: dict[str, DataLoader] = {}
    source_adjacencies: dict[str, torch.Tensor] = {}
    source_protocols: dict[str, Any] = {}
    # 每个源城市有独立 seed、归一化统计和邻接图，但共享一个模型。
    for index, source in enumerate(data["sources"]):
        city = str(source["city"]).lower()
        loader, adjacency, _, protocol = build_source_train_loader(
            root,
            city,
            batch_size=int(train["source_microbatch_size"]),
            seed=int(run["seed"]) + index + 1,
            stride=int(source.get("stride", 4)),
            train_ratio=float(source.get("train_ratio", 0.7)),
            val_ratio=float(source.get("val_ratio", 0.1)),
            num_workers=int(run.get("num_workers", 0)),
            pin_memory=pin_memory,
        )
        source_loaders[city] = loader
        source_adjacencies[city] = adjacency.to(device)
        source_protocols[city] = protocol
    protocol = {"target": target_protocol, "sources": source_protocols}
    _atomic_json(output_dir / "protocol.json", protocol)
    target_nodes = int(target_adjacency.shape[0])
    if bool(run.get("formal", False)):
        expected_bank = target_nodes // 2
        if int(config["model"]["prompt_bank_size"]) != expected_bank:
            raise ValueError(
                f"prompt bank must be floor(target_nodes/2)={expected_bank}"
            )
    return (
        target_loaders,
        target_adjacency.to(device),
        stats,
        target_protocol,
        source_loaders,
        source_adjacencies,
        source_protocols,
    )


def _train_epoch(
    model: torch.nn.Module,
    parameters: Sequence[torch.nn.Parameter],
    epoch: int,
    config: Mapping[str, Any],
    target_dataset: Any,
    target_adjacency: torch.Tensor,
    source_datasets: Mapping[str, Any],
    source_adjacencies: Mapping[str, torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    """执行一个 epoch 的目标驱动 logical updates。

    每遍历一个目标 batch：

    1. 计算目标 L1 与目标梯度；
    2. 从三个循环游标精确取得 42/41/41 源样本；
    3. 计算逐样本等权源梯度；
    4. 诊断并应用 fixed TSB 显式参数更新。

    返回按样本加权的 epoch 损失、源循环次数、更新次数和平均梯度诊断。
    """
    started = time.perf_counter()
    run = config["run"]
    train = config["train"]
    seed = int(run["seed"])
    # epoch 进入 seed，使中断恢复时同一 epoch 的打乱顺序完全一致。
    target_loader = _epoch_loader(
        target_dataset,
        batch_size=int(train["target_batch_size"]),
        seed=seed + epoch * 10007,
        shuffle=True,
        num_workers=int(run.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    source_cities = tuple(source_datasets)
    allocations = _source_allocation(source_cities, int(train["source_batch_size"]))
    cursors: dict[str, _CyclingCursor] = {}
    for index, city in enumerate(source_cities):
        loader = _epoch_loader(
            source_datasets[city],
            batch_size=int(train["source_microbatch_size"]),
            seed=seed + epoch * 10007 + (index + 1) * 1000003,
            shuffle=True,
            num_workers=int(run.get("num_workers", 0)),
            pin_memory=device.type == "cuda",
        )
        cursors[city] = _CyclingCursor(loader)

    gamma_target = float(train["gamma_target"])
    gamma_source = float(train["gamma_source"])
    target_loss_sum = source_loss_sum = 0.0
    target_samples = source_samples = updates = 0
    diagnostic_sums: dict[str, float] = {}
    model.train()
    for x, y in target_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        prediction = model(x, target_adjacency, horizon=y.shape[1])
        target_loss = F.l1_loss(prediction, y)
        # 不调用 backward/optimizer；目标和源梯度必须保持为两组独立张量。
        target_gradients = torch.autograd.grad(
            target_loss, parameters, allow_unused=True, retain_graph=False
        )

        source_chunks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for city in source_cities:
            city_chunks = cursors[city].take(allocations[city])
            # 跨数据集末尾时可能得到两块 CPU 张量；重新拼成每城一个 train-mode
            # batch，保持 BatchNorm 看到完整的 42 或 41 个样本。
            source_x = torch.cat([chunk[0] for chunk in city_chunks], dim=0)
            source_y = torch.cat([chunk[1] for chunk in city_chunks], dim=0)
            source_chunks.append(
                (source_x, source_y, source_adjacencies[city])
            )
        source_gradients, source_loss, observed = _sample_weighted_source_gradients(
            model,
            parameters,
            source_chunks,
            int(train["source_batch_size"]),
            device,
        )
        diagnostics = _gradient_diagnostics(
            target_gradients,
            source_gradients,
            parameters,
            gamma_target,
            gamma_source,
        )
        # 此调用直接原地修改参数，没有 optimizer state。
        applied = apply_tsb_gradients(
            parameters,
            target_gradients,
            source_gradients,
            gamma_target,
            gamma_source,
        )
        if abs(applied["conflict_ratio"] - diagnostics["conflict_ratio"]) > 1e-12:
            raise RuntimeError("TSB conflict diagnostics disagree with applied update")
        for key, value in diagnostics.items():
            diagnostic_sums[key] = diagnostic_sums.get(key, 0.0) + float(value)
        target_loss_sum += float(target_loss.detach()) * x.shape[0]
        target_samples += int(x.shape[0])
        source_loss_sum += source_loss * observed
        source_samples += observed
        updates += 1

    return {
        "epoch": epoch,
        "target_loss": target_loss_sum / max(target_samples, 1),
        "source_loss": source_loss_sum / max(source_samples, 1),
        "target_samples": target_samples,
        "source_samples": source_samples,
        "logical_updates": updates,
        "source_allocation": allocations,
        "source_cycles": {city: cursor.cycles for city, cursor in cursors.items()},
        "source_gradient_semantics": (
            "sample-weighted logical mean; one train-mode 42/41/41 batch per source city"
        ),
        "diagnostics": {
            key: value / max(updates, 1) for key, value in diagnostic_sums.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def _evaluate_full_test(
    model: torch.nn.Module,
    loader: DataLoader,
    adjacency: torch.Tensor,
    stats: SpeedStats,
    horizon: int,
    device: torch.device,
) -> dict[str, Any]:
    """完整遍历正式测试 loader，并在原始速度单位累计逐步指标。

    除 MAE/RMSE 与 persistence 基线外，还记录归一化 L1、预测/标签均值标准差、
    元素总数和运行时间，用于发现预测坍缩或尺度错误。
    """
    started = time.perf_counter()
    accumulator = ForecastMetricAccumulator(horizon, stats)
    prediction_sum = prediction_sq = target_sum = target_sq = 0.0
    normalized_abs = 0.0
    element_count = 0
    model.eval()
    # 正式评估禁用 autograd；loader 的一次性访问约束由上层代理执行。
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            prediction = model(x, adjacency, horizon=horizon)
            if prediction.shape != y.shape:
                raise RuntimeError("test prediction and target shapes differ")
            normalized_abs += float((prediction - y).abs().sum())
            element_count += y.numel()
            accumulator.update(prediction, y, x[:, -1, :, 0])
            prediction_raw = stats.denormalize(prediction.detach().double())
            target_raw = stats.denormalize(y.detach().double())
            prediction_sum += float(prediction_raw.sum())
            prediction_sq += float(prediction_raw.square().sum())
            target_sum += float(target_raw.sum())
            target_sq += float(target_raw.square().sum())
    if element_count == 0:
        raise RuntimeError("full test loader produced no observations")
    prediction_mean = prediction_sum / element_count
    target_mean = target_sum / element_count
    return {
        "normalized_l1": normalized_abs / element_count,
        "metrics": accumulator.compute().as_dict(),
        "diagnostics": {
            "prediction_mean": prediction_mean,
            "prediction_std": max(
                prediction_sq / element_count - prediction_mean**2, 0.0
            )
            ** 0.5,
            "target_mean": target_mean,
            "target_std": max(target_sq / element_count - target_mean**2, 0.0)
            ** 0.5,
            "element_count": element_count,
        },
        "runtime_seconds": time.perf_counter() - started,
    }


def _evaluation_loader(
    dataset: Any, config: Mapping[str, Any], device: torch.device
) -> DataLoader:
    """创建不打乱、不丢尾 batch 的确定性评估 loader。"""
    return _epoch_loader(
        dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        seed=int(config["run"]["seed"]),
        shuffle=False,
        num_workers=int(config["run"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )


def run_probe(
    config: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    data_bundle: tuple[Any, ...],
    model_builder: ModelBuilder | None = None,
) -> Path:
    """对目标和每个源城市各做一个 forward/backward 健康检查。

    检查形状、输出有限性、梯度有限性及 CUDA 峰值显存，不执行参数更新，也不遍历
    完整数据集。该通用诊断入口不被正式 ``tsjt_c1.run`` 调用。
    """
    (
        target_loaders,
        target_adjacency,
        _,
        target_protocol,
        source_loaders,
        source_adjacencies,
        source_protocols,
    ) = data_bundle
    model = (model_builder or _build_model)(config).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    target_x, target_y = next(iter(target_loaders["train"]))
    model.train()
    target_prediction = model(
        target_x.to(device),
        target_adjacency,
        horizon=target_y.shape[1],
    )
    target_loss = F.l1_loss(target_prediction, target_y.to(device))
    model.zero_grad(set_to_none=True)
    target_loss.backward()
    target_gradient_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    sources: dict[str, Any] = {}
    for city, loader in source_loaders.items():
        x, y = next(iter(loader))
        model.train()
        prediction = model(
            x.to(device), source_adjacencies[city], horizon=y.shape[1]
        )
        loss = F.l1_loss(prediction, y.to(device))
        model.zero_grad(set_to_none=True)
        loss.backward()
        gradient_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        sources[city] = {
            "input_shape": list(x.shape),
            "target_shape": list(y.shape),
            "prediction_shape": list(prediction.shape),
            "finite": bool(torch.isfinite(prediction).all()),
            "gradient_finite": bool(gradient_finite),
            "protocol": source_protocols[city],
        }
    payload = {
        "device": str(device),
        "target": {
            "input_shape": list(target_x.shape),
            "target_shape": list(target_y.shape),
            "prediction_shape": list(target_prediction.shape),
            "finite": bool(torch.isfinite(target_prediction).all()),
            "gradient_finite": bool(target_gradient_finite),
            "protocol": target_protocol,
        },
        "sources": sources,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
    }
    path = output_dir / "probe" / "result.json"
    _atomic_json(path, payload)
    return path


def run_all(
    config: dict[str, Any],
    output_dir: Path,
    config_hash: str,
    device: torch.device,
    data_bundle: tuple[Any, ...],
    model_builder: ModelBuilder | None = None,
) -> Path:
    """训练引擎的通用“训练至最终 epoch 后评估一次”状态机。

    它提供 checkpoint 恢复、日志一致性、最终权重先落盘以及评估中断隔离。正式 C1
    使用 ``run.py`` 中约束更严格的专用状态机，本函数保留为底层通用能力。
    """
    complete_path = output_dir / "complete.json"
    final_metrics = output_dir / "final_metrics.json"
    final_checkpoint = output_dir / f"epoch_{int(config['train']['epochs']):02d}.pt"
    if complete_path.exists():
        if bool(config["run"].get("skip_complete", True)):
            if not final_metrics.exists() or not final_checkpoint.exists():
                raise RuntimeError("completion marker exists without final evidence")
            return final_metrics
        raise RuntimeError("completed output is immutable; choose a new output directory")

    (
        target_loaders,
        target_adjacency,
        stats,
        target_protocol,
        source_loaders,
        source_adjacencies,
        _,
    ) = data_bundle
    model = (model_builder or _build_model)(config).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    last_path = output_dir / "last.pt"
    log_path = output_dir / "train.jsonl"
    start_epoch = 0
    if last_path.exists():
        if not bool(config["run"].get("resume", True)):
            raise RuntimeError("checkpoint exists but run.resume=false")
        payload = _load_checkpoint(
            last_path, model, config_hash, restore_rng=True
        )
        _ensure_log_matches_checkpoint(log_path, payload)
        start_epoch = int(payload["epoch"])
    elif log_path.exists():
        raise RuntimeError("training log exists without a resumable checkpoint")

    epochs = int(config["train"]["epochs"])
    for epoch in range(start_epoch + 1, epochs + 1):
        record = _train_epoch(
            model,
            parameters,
            epoch,
            config,
            target_loaders["train"].dataset,
            target_adjacency,
            {city: loader.dataset for city, loader in source_loaders.items()},
            source_adjacencies,
            device,
        )
        payload = {
            "model": model.state_dict(),
            "epoch": epoch,
            "config_hash": config_hash,
            "rng_state": _rng_state(),
            "epoch_record": record,
            "source_iterator_state": {
                "strategy": "epoch-local deterministic loaders",
                "epoch_seed": int(config["run"]["seed"]) + epoch * 10007,
                "source_cycles": record["source_cycles"],
            },
        }
        _atomic_checkpoint(last_path, payload)
        if epoch == epochs:
            # 在创建任何测试 iterator 前，最终权重必须已经完整持久化。
            _atomic_checkpoint(final_checkpoint, payload)
        _append_jsonl(log_path, record)

    if not final_checkpoint.exists():
        raise RuntimeError("final epoch checkpoint is not durable; refusing test access")
    final_payload = _load_checkpoint(
        final_checkpoint, model, config_hash, restore_rng=False
    )
    if int(final_payload["epoch"]) != epochs:
        raise RuntimeError("final checkpoint does not contain final-epoch weights")
    evaluation_started = output_dir / "evaluation_started.json"
    if evaluation_started.exists() and not final_metrics.exists():
        raise RuntimeError(
            "a previous full-test evaluation was interrupted; use a new output namespace"
        )
    if final_metrics.exists():
        raise RuntimeError("final metrics exist without completion marker")
    _atomic_json(
        evaluation_started,
        {
            "epoch": epochs,
            "checkpoint": str(final_checkpoint),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    test_loader = _evaluation_loader(target_loaders["test"].dataset, config, device)
    test = _evaluate_full_test(
        model,
        test_loader,
        target_adjacency,
        stats,
        int(target_protocol["horizon"]),
        device,
    )
    result = {
        "seed": int(config["run"]["seed"]),
        "trained_epochs": epochs,
        "checkpoint": str(final_checkpoint),
        "checkpoint_epoch": int(final_payload["epoch"]),
        "selection": "none; fixed epoch-50 weights" if epochs == 50 else "none",
        "test_evaluations": 1,
        "test": test,
        "protocol": target_protocol,
    }
    _atomic_json(final_metrics, result)
    _atomic_json(
        complete_path,
        {
            "epoch": epochs,
            "checkpoint": str(final_checkpoint),
            "final_metrics": str(final_metrics),
            "test_evaluations": 1,
        },
    )
    return final_metrics


def run(
    config_path: str | Path,
    mode: str = "all",
    argv: Sequence[str] | None = None,
    model_builder: ModelBuilder | None = None,
) -> Path:
    """通用训练引擎入口，支持 ``probe`` 与 ``all`` 两种模式。

    负责配置读取、基础校验、seed/device、输出元数据、数据指纹与数据组装，再分派到
    对应执行函数。正式用户应优先调用 ``python -m tsjt_c1.run``。
    """
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    _validate_config(config)
    _set_seed(int(config["run"]["seed"]))
    device = _resolve_device(str(config["run"].get("device", "auto")))
    output_dir, config_hash = _prepare_output(
        config_path, config, device, argv or sys.argv
    )
    cities = [
        *(str(item["city"]).lower() for item in config["data"]["sources"]),
        str(config["data"]["target"]["city"]).lower(),
    ]
    _fingerprint_data(Path(config["data"]["root"]), cities, output_dir)
    data_bundle = _make_data(config, device, output_dir)
    if mode == "probe":
        return run_probe(
            config,
            output_dir,
            device,
            data_bundle,
            model_builder=model_builder,
        )
    if mode == "all":
        return run_all(
            config,
            output_dir,
            config_hash,
            device,
            data_bundle,
            model_builder=model_builder,
        )
    raise ValueError(f"unsupported mode: {mode}")


def main() -> None:
    """解析通用引擎命令行并输出生成 artifact 的 JSON 路径。"""
    parser = argparse.ArgumentParser(
        description="TSJT-C1 fixed-50-epoch one-stage experiment"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("probe", "all"), default="all")
    args = parser.parse_args()
    result = run(args.config, args.mode)
    print(json.dumps({"artifact": str(result)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
