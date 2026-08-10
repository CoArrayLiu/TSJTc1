"""面向正式训练协议的数据加载器。

本模块把通用的数据流水线封装成 TSJT-C1 的城市级接口。这里固定了四个城市的
采样频率、历史长度和预测长度，并分别构造：

* 目标城市 PEMS-BAY 的少样本训练集与正式测试集；
* 三个源城市各自的源训练集。

最重要的协议边界是：归一化统计量只能由训练时间段拟合，且每个滑动窗口的历史
与标签都不能跨越其所属 split。返回的 ``protocol`` 字典会把这些决定持久化，
用于复现实验和排查数据泄漏。
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from tsjt_c1.data.pipeline import (
    ForecastWindowDataset,
    SpeedStats,
    TimeRange,
    build_forecast_features,
    fit_speed_stats,
    load_city_forecast_data,
    source_splits,
)


# 每个城市都预测未来一个物理小时。5 分钟数据用 12 步，10 分钟数据用 6 步。
CITY_PROTOCOLS: dict[str, dict[str, int]] = {
    "metr-la": {"steps_per_day": 288, "history": 12, "horizon": 12},
    "pems-bay": {"steps_per_day": 288, "history": 12, "horizon": 12},
    "chengdu": {"steps_per_day": 144, "history": 6, "horizon": 6},
    "shenzhen": {"steps_per_day": 144, "history": 6, "horizon": 6},
}

# 四个 dataset.npy 的第 3 通道都使用五分钟刻度编码一周内时间槽。
# 成都/深圳每行是十分钟，因此该通道每行递增 2，但每天仍跨度 288。
TIME_CHANNEL_STEPS_PER_DAY = 288
TIME_FEATURE_REVISION = "weekly-slot-5min-dow-v1"


def _city_protocol(city: str) -> tuple[str, dict[str, int]]:
    """规范化城市名并取得固定协议；未知城市立即报错，避免静默套用错误配置。"""
    city_key = city.lower()
    try:
        protocol = CITY_PROTOCOLS[city_key]
    except KeyError as error:
        raise ValueError(f"unsupported traffic city: {city_key}") from error
    return city_key, protocol


def _loader(
    dataset: ForecastWindowDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """用显式随机种子构造 DataLoader。

    只有训练 loader 开启 shuffle；测试 loader 保持时间顺序。``drop_last=False``
    保证少样本目标域的最后一个不完整 batch 也参与训练或评估。
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
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


def build_target_train_test_loaders(
    data_root: str | Path,
    city: str,
    *,
    batch_size: int,
    seed: int,
    train_days: int = 3,
    train_stride: int = 1,
    test_stride: int = 1,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[dict[str, DataLoader], torch.Tensor, SpeedStats, dict[str, Any]]:
    """构造少样本目标城市的训练/正式测试 loader，不创建验证集。

    前 ``train_days`` 天（正式配置为 3 天）是唯一目标训练区间，后续全部时间点是
    正式测试区间。速度均值和标准差只由训练区间拟合。训练与测试窗口均设置
    ``allow_prior_split_history=False``，所以测试窗口甚至不会借用训练区间末尾的
    历史观测，输入与标签都严格包含在测试 split 内。

    返回值依次为：``{"train", "test"}`` loaders、``[N,N]`` 邻接矩阵、训练区间
    速度统计量，以及可序列化的协议记录。
    """

    city_key, city_protocol = _city_protocol(city)
    city_data = load_city_forecast_data(data_root, city_key)
    steps_per_day = city_protocol["steps_per_day"]
    train_end = train_days * steps_per_day
    if train_days <= 0 or train_end >= city_data.num_steps:
        raise ValueError("target training range must leave a non-empty test range")
    train_range = TimeRange(0, train_end)
    test_range = TimeRange(train_end, city_data.num_steps)
    # 防泄漏关键点：只把 train_range 交给统计量拟合函数。
    stats = fit_speed_stats(city_data.values, train_range)
    features = build_forecast_features(
        city_data.values,
        stats,
        steps_per_day=steps_per_day,
        time_channel_steps_per_day=TIME_CHANNEL_STEPS_PER_DAY,
    )
    dataset_kwargs = {
        "history": city_protocol["history"],
        "horizon": city_protocol["horizon"],
        "allow_prior_split_history": False,
    }
    datasets = {
        "train": ForecastWindowDataset(
            features,
            train_range,
            stride=train_stride,
            **dataset_kwargs,
        ),
        "test": ForecastWindowDataset(
            features,
            test_range,
            stride=test_stride,
            **dataset_kwargs,
        ),
    }
    loaders = {
        "train": _loader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": _loader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    protocol: dict[str, Any] = {
        "city": city_key,
        **city_protocol,
        "time_channel_steps_per_day": TIME_CHANNEL_STEPS_PER_DAY,
        "time_feature_revision": TIME_FEATURE_REVISION,
        "train_days": train_days,
        "split": {
            "train": (train_range.start, train_range.end),
            "test": (test_range.start, test_range.end),
        },
        "windows": {name: len(dataset) for name, dataset in datasets.items()},
        "stride": {"train": train_stride, "test": test_stride},
        "normalization": asdict(stats),
        "features": ["train-normalized speed", "time-of-day", "day-of-week"],
        "no_validation": True,
        "allow_prior_split_history": False,
    }
    # 显式复制为 float32，避免只读 mmap 数组与 PyTorch 共享不可写内存。
    adjacency = torch.from_numpy(
        np.array(city_data.adjacency, dtype=np.float32, copy=True)
    )
    return loaders, adjacency, stats, protocol


def build_source_train_loader(
    data_root: str | Path,
    city: str,
    *,
    batch_size: int,
    seed: int,
    stride: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, torch.Tensor, SpeedStats, dict[str, Any]]:
    """构造单个源城市的训练 loader。

    源城市仍按时间顺序计算 70/10/20 边界，但本函数只物化并返回前 70% 的训练
    窗口；validation/test 边界只记录在协议里，不参与当前正式训练。每个源城市
    使用自己的训练段均值和标准差，避免不同城市的速度量纲互相污染。
    """

    city_key, city_protocol = _city_protocol(city)
    city_data = load_city_forecast_data(data_root, city_key)
    splits = source_splits(
        city_data.num_steps,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    stats = fit_speed_stats(city_data.values, splits.train)
    features = build_forecast_features(
        city_data.values,
        stats,
        steps_per_day=city_protocol["steps_per_day"],
        time_channel_steps_per_day=TIME_CHANNEL_STEPS_PER_DAY,
    )
    # stride=4 用于减少源域窗口密度；目标域仍使用 stride=1。
    dataset = ForecastWindowDataset(
        features,
        splits.train,
        history=city_protocol["history"],
        horizon=city_protocol["horizon"],
        stride=stride,
        allow_prior_split_history=False,
    )
    loader = _loader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    protocol: dict[str, Any] = {
        "city": city_key,
        **city_protocol,
        "time_channel_steps_per_day": TIME_CHANNEL_STEPS_PER_DAY,
        "time_feature_revision": TIME_FEATURE_REVISION,
        "split": splits.as_dict(),
        "train_windows": len(dataset),
        "stride": stride,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "normalization": asdict(stats),
        "features": ["source-train-normalized speed", "time-of-day", "day-of-week"],
        "allow_prior_split_history": False,
    }
    adjacency = torch.from_numpy(
        np.array(city_data.adjacency, dtype=np.float32, copy=True)
    )
    return loader, adjacency, stats, protocol
