"""交通预测的数据预处理、滑窗数据集与指标累计器。

本模块不包含任何 TSJT 模型逻辑，只负责把原始 ``dataset.npy`` 转换成统一的
``[B,T,N,3]`` 输入和 ``[B,H,N,1]`` 标签。三个输入通道依次是训练集统计量
归一化后的速度、日内时刻和星期位置。所有区间都采用 Python 风格的左闭右开
表示 ``[start, end)``，从类型层面减少边界歧义。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class TimeRange:
    """左闭右开的时间索引区间 ``[start, end)``。

    滑动窗口的预测标签必须完整包含在该区间中；当启用严格模式时，历史输入也必须
    完整包含在其中。
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        """在对象创建后验证边界非负且区间非空。"""
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid half-open time range [{self.start}, {self.end})")

    @property
    def length(self) -> int:
        """返回区间包含的离散时间步数量。"""
        return self.end - self.start


@dataclass(frozen=True)
class DataSplits:
    """按时间顺序排列且首尾相接的 train/val/test 三段。"""
    train: TimeRange
    val: TimeRange
    test: TimeRange

    def __post_init__(self) -> None:
        """禁止 split 之间出现重叠或空洞。"""
        if self.train.end != self.val.start or self.val.end != self.test.start:
            raise ValueError("train, validation, and test ranges must be contiguous")

    def as_dict(self) -> dict[str, tuple[int, int]]:
        """转换为可写入 JSON/YAML 协议文件的普通字典。"""
        return {
            "train": (self.train.start, self.train.end),
            "val": (self.val.start, self.val.end),
            "test": (self.test.start, self.test.end),
        }


@dataclass(frozen=True)
class SpeedStats:
    """训练速度通道的标量均值和标准差。"""
    mean: float
    std: float

    def __post_init__(self) -> None:
        """拒绝 NaN、无穷值和非正标准差。"""
        if not np.isfinite(self.mean) or not np.isfinite(self.std) or self.std <= 0:
            raise ValueError("speed statistics must be finite and std must be positive")

    def normalize(self, value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """执行 z-score 标准化；同时支持 NumPy 与 PyTorch 张量。"""
        return (value - self.mean) / self.std

    def denormalize(self, value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """将模型输出还原到原始速度单位，用于最终 MAE/RMSE。"""
        return value * self.std + self.mean


@dataclass(frozen=True)
class CityForecastData:
    """一个城市的原始时序数组及物理邻接矩阵。"""
    name: str
    values: np.ndarray
    adjacency: np.ndarray

    @property
    def num_steps(self) -> int:
        """时间轴长度 T。"""
        return int(self.values.shape[0])

    @property
    def num_nodes(self) -> int:
        """传感器/道路节点数 N。"""
        return int(self.values.shape[1])


def load_city_forecast_data(
    data_root: str | Path,
    city: str,
    *,
    mmap: bool = True,
) -> CityForecastData:
    """从 ``data/<city>/`` 读取时序和邻接矩阵并检查形状。

    默认使用只读 mmap，避免一次性把数百 MB 的四城市数据全部载入内存。
    ``dataset.npy`` 必须是 ``[T,N,C]``，``matrix.npy`` 必须是 ``[N,N]``。
    """
    city_key = city.lower()
    root = Path(data_root) / city_key
    mmap_mode = "r" if mmap else None
    values = np.load(root / "dataset.npy", mmap_mode=mmap_mode)
    adjacency = np.load(root / "matrix.npy", mmap_mode=mmap_mode)
    if values.ndim != 3 or values.shape[-1] < 1:
        raise ValueError(f"{city_key}: dataset must have shape [T,N,C>=1], got {values.shape}")
    if adjacency.shape != (values.shape[1], values.shape[1]):
        raise ValueError(
            f"{city_key}: adjacency shape {adjacency.shape} does not match N={values.shape[1]}"
        )
    return CityForecastData(city_key, values, adjacency)


def source_splits(
    total_steps: int,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> DataSplits:
    """按时间顺序生成源域 split，默认比例为 70%/10%/20%。

    绝不先随机打乱再切分，因为那会把未来交通状态泄漏到训练区间。
    """

    if total_steps < 3:
        raise ValueError("at least three time steps are required")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("source split ratios must be positive and sum to less than one")
    train_end = int(total_steps * train_ratio)
    val_end = int(total_steps * (train_ratio + val_ratio))
    return DataSplits(
        train=TimeRange(0, train_end),
        val=TimeRange(train_end, val_end),
        test=TimeRange(val_end, total_steps),
    )


def target_few_shot_splits(
    total_steps: int,
    *,
    steps_per_day: int,
    train_days: int = 3,
    val_days: int = 2,
) -> DataSplits:
    """按天生成少样本目标域的 train/val/test 区间。

    该通用工具保留显式验证段；正式 C1 入口不调用它，而是直接构造“前三天训练、
    其余测试”的无验证协议。
    """

    if steps_per_day <= 0 or train_days <= 0 or val_days <= 0:
        raise ValueError("steps_per_day, train_days, and val_days must be positive")
    train_end = train_days * steps_per_day
    val_end = train_end + val_days * steps_per_day
    if val_end >= total_steps:
        raise ValueError(
            f"target data has {total_steps} steps, but train+validation require {val_end}"
        )
    return DataSplits(
        train=TimeRange(0, train_end),
        val=TimeRange(train_end, val_end),
        test=TimeRange(val_end, total_steps),
    )


def fit_speed_stats(values: np.ndarray, train_range: TimeRange) -> SpeedStats:
    """仅使用训练区间的第 0 通道速度拟合全局均值和标准差。

    统计范围覆盖该城市训练段的全部时间和节点。标准差下限为 ``1e-6``，防止常数
    序列导致除零。
    """

    if values.ndim != 3 or values.shape[-1] < 1:
        raise ValueError("values must have shape [T,N,C>=1]")
    if train_range.end > values.shape[0]:
        raise ValueError("training range extends past the available data")
    speed = np.asarray(values[train_range.start : train_range.end, :, 0], dtype=np.float64)
    std = float(speed.std())
    return SpeedStats(mean=float(speed.mean()), std=max(std, 1e-6))


def _time_covariates(
    values: np.ndarray,
    *,
    steps_per_day: int,
    time_channel_steps_per_day: int | None = None,
    start_day_of_week: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成已知未来协变量 TOD 与 DOW，二者都不包含未来速度。

    TOD 优先复用数据中合法的 ``[0,1]`` 第 1 通道，否则按时间索引生成。DOW
    优先从第 3 通道的一周内时间槽推导，否则从
    ``start_day_of_week`` 顺推。``steps_per_day`` 是原生数据行数/天；
    ``time_channel_steps_per_day`` 是第 3 通道的编码刻度/天，未给定时
    默认与原生步数相同。返回数组形状均为 ``[T,N]``。
    """
    total_steps, num_nodes = values.shape[:2]
    if steps_per_day <= 0:
        raise ValueError("steps_per_day must be positive")
    channel_steps_per_day = (
        steps_per_day
        if time_channel_steps_per_day is None
        else time_channel_steps_per_day
    )
    if channel_steps_per_day <= 0:
        raise ValueError("time_channel_steps_per_day must be positive")

    # TOD/DOW 在预测时天然已知，因此使用未来时刻的协变量不构成标签泄漏。
    if values.shape[-1] >= 2:
        candidate = np.asarray(values[:, :, 1], dtype=np.float32)
        if np.isfinite(candidate).all() and candidate.min() >= 0 and candidate.max() <= 1:
            tod = candidate
        else:
            tod = np.broadcast_to(
                (np.arange(total_steps, dtype=np.float32) % steps_per_day)[:, None]
                / float(steps_per_day),
                (total_steps, num_nodes),
            )
    else:
        tod = np.broadcast_to(
            (np.arange(total_steps, dtype=np.float32) % steps_per_day)[:, None]
            / float(steps_per_day),
            (total_steps, num_nodes),
        )

    if values.shape[-1] >= 4:
        weekly_slot = np.asarray(values[:, 0, 3], dtype=np.float64)
        if np.isfinite(weekly_slot).all():
            day_index = (
                np.floor_divide(weekly_slot.astype(np.int64), channel_steps_per_day) % 7
            )
        else:
            day_index = (start_day_of_week + np.arange(total_steps) // steps_per_day) % 7
    else:
        day_index = (start_day_of_week + np.arange(total_steps) // steps_per_day) % 7
    dow = np.broadcast_to(
        (day_index.astype(np.float32) / 7.0)[:, None],
        (total_steps, num_nodes),
    )
    return np.asarray(tod, dtype=np.float32), np.asarray(dow, dtype=np.float32)


def build_forecast_features(
    values: np.ndarray,
    stats: SpeedStats,
    *,
    steps_per_day: int,
    time_channel_steps_per_day: int | None = None,
    start_day_of_week: int = 0,
) -> np.ndarray:
    """拼接 float32 特征 ``[归一化速度, TOD, DOW]``，输出 ``[T,N,3]``。"""

    if values.ndim != 3 or values.shape[-1] < 1:
        raise ValueError("values must have shape [T,N,C>=1]")
    speed = np.asarray(stats.normalize(values[:, :, 0]), dtype=np.float32)
    tod, dow = _time_covariates(
        values,
        steps_per_day=steps_per_day,
        time_channel_steps_per_day=time_channel_steps_per_day,
        start_day_of_week=start_day_of_week,
    )
    return np.stack((speed, tod, dow), axis=-1)


class ForecastWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """把连续特征转换为严格受 split 约束的预测滑窗。

    单个样本 ``x`` 为 ``[history,N,3]``，``y`` 为 ``[horizon,N,1]``。标签始终
    不越过 ``time_range``；当 ``allow_prior_split_history=False`` 时，输入历史也
    不得借用 split 之前的数据，这是正式协议采用的模式。
    """

    def __init__(
        self,
        features: np.ndarray,
        time_range: TimeRange,
        *,
        history: int = 288,
        horizon: int = 12,
        stride: int = 1,
        allow_prior_split_history: bool = True,
    ) -> None:
        """验证窗口参数并预计算所有合法 history 起点。

        ``indices`` 只保存整数起点而不复制特征，因此多个 DataLoader 可以共享同一
        份 NumPy 特征数组。
        """
        if features.ndim != 3 or features.shape[-1] != 3:
            raise ValueError("features must have shape [T,N,3]")
        if history <= 0 or horizon <= 0 or stride <= 0:
            raise ValueError("history, horizon, and stride must be positive")
        if time_range.end > features.shape[0]:
            raise ValueError("time range extends past feature data")
        # 严格模式下，第一个预测点至少是 split.start + history。
        first_forecast = max(
            time_range.start if allow_prior_split_history else time_range.start + history,
            history,
        )
        last_forecast = time_range.end - horizon
        if first_forecast > last_forecast:
            raise ValueError(
                f"range [{time_range.start},{time_range.end}) cannot provide "
                f"a target for a {history}->{horizon} window"
            )
        self.features = features
        self.time_range = time_range
        self.history = int(history)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.allow_prior_split_history = bool(allow_prior_split_history)
        first_history = first_forecast - self.history
        last_history = last_forecast - self.history
        self.indices = tuple(range(first_history, last_history + 1, self.stride))

    def __len__(self) -> int:
        """返回可用滑动窗口数量。"""
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """按预计算起点切出一个输入/标签窗口并零拷贝式转为张量。"""
        start = self.indices[index]
        forecast_start = start + self.history
        x = np.asarray(self.features[start:forecast_start], dtype=np.float32)
        y = np.asarray(
            self.features[forecast_start : forecast_start + self.horizon, :, 0:1],
            dtype=np.float32,
        )
        return torch.from_numpy(x), torch.from_numpy(y)


def make_forecast_datasets(
    values: np.ndarray,
    splits: DataSplits,
    stats: SpeedStats,
    *,
    steps_per_day: int,
    time_channel_steps_per_day: int | None = None,
    history: int = 288,
    horizon: int = 12,
    train_stride: int = 1,
    val_stride: int = 1,
    test_stride: int = 1,
    start_day_of_week: int = 0,
    allow_prior_split_history: bool = True,
) -> dict[str, ForecastWindowDataset]:
    """为通用三段 split 一次性构建共享特征上的 train/val/test 数据集。"""
    features = build_forecast_features(
        values,
        stats,
        steps_per_day=steps_per_day,
        time_channel_steps_per_day=time_channel_steps_per_day,
        start_day_of_week=start_day_of_week,
    )
    return {
        "train": ForecastWindowDataset(
            features,
            splits.train,
            history=history,
            horizon=horizon,
            stride=train_stride,
            allow_prior_split_history=allow_prior_split_history,
        ),
        "val": ForecastWindowDataset(
            features,
            splits.val,
            history=history,
            horizon=horizon,
            stride=val_stride,
            allow_prior_split_history=allow_prior_split_history,
        ),
        "test": ForecastWindowDataset(
            features,
            splits.test,
            history=history,
            horizon=horizon,
            stride=test_stride,
            allow_prior_split_history=allow_prior_split_history,
        ),
    }


def make_forecast_loaders(
    datasets: Mapping[str, ForecastWindowDataset],
    *,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """为通用三段数据集构建 loader；仅训练集按给定 seed 打乱。"""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    missing = {"train", "val", "test"} - set(datasets)
    if missing:
        raise ValueError(f"missing datasets: {sorted(missing)}")
    generator = torch.Generator().manual_seed(seed)
    return {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=split == "train",
            generator=generator if split == "train" else None,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )
        for split in ("train", "val", "test")
    }


@dataclass(frozen=True)
class HorizonMetrics:
    """逐预测步的 MAE、RMSE 与累计元素数。"""
    horizon_steps: tuple[int, ...]
    mae: tuple[float, ...]
    rmse: tuple[float, ...]
    count: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        """转换为 JSON 可序列化结构。"""
        return {
            "horizon_steps": list(self.horizon_steps),
            "mae": list(self.mae),
            "rmse": list(self.rmse),
            "count": list(self.count),
        }


@dataclass(frozen=True)
class ForecastMetrics:
    """同时保存模型结果和 last-value persistence 基线结果。"""
    model: HorizonMetrics
    persistence: HorizonMetrics

    def as_dict(self) -> dict[str, object]:
        """递归转换为 JSON 可序列化结构。"""
        return {"model": self.model.as_dict(), "persistence": self.persistence.as_dict()}


class ForecastMetricAccumulator:
    """以原始速度单位、按元素等权累计逐 horizon 指标。

    除模型预测外，还同时计算“把最后观测速度复制到所有未来步”的 persistence
    基线。累计器保存误差和而非 batch 均值，因此不同大小的最后一个 batch 不会被
    赋予过高权重。
    """

    def __init__(self, horizon: int, stats: SpeedStats) -> None:
        """初始化每个预测步的绝对误差、平方误差和计数数组。"""
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        self.horizon = int(horizon)
        self.stats = stats
        self._model_abs = np.zeros(horizon, dtype=np.float64)
        self._model_sq = np.zeros(horizon, dtype=np.float64)
        self._persistence_abs = np.zeros(horizon, dtype=np.float64)
        self._persistence_sq = np.zeros(horizon, dtype=np.float64)
        self._count = np.zeros(horizon, dtype=np.int64)

    @staticmethod
    def _speed_array(value: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
        """统一输入到 CPU float64 ``[B,H,N]``，便于稳定累计。"""
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float64)
        if array.ndim == 4 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim != 3:
            raise ValueError(f"{name} must have shape [B,H,N] or [B,H,N,1]")
        return array

    def update(
        self,
        prediction_normalized: np.ndarray | torch.Tensor,
        target_normalized: np.ndarray | torch.Tensor,
        last_observed_normalized: np.ndarray | torch.Tensor,
    ) -> None:
        """累加一个 batch 的模型误差与 persistence 误差。

        ``last_observed`` 可为 ``[B,N]`` 或带单例通道的等价形状；它会广播到全部
        horizon。预测和标签先反归一化，再在原始速度单位中计算误差。
        """
        prediction = self._speed_array(prediction_normalized, "prediction")
        target = self._speed_array(target_normalized, "target")
        if prediction.shape != target.shape or prediction.shape[1] != self.horizon:
            raise ValueError(
                f"prediction/target must have equal [B,{self.horizon},N] shapes"
            )

        if isinstance(last_observed_normalized, torch.Tensor):
            last_observed_normalized = last_observed_normalized.detach().cpu().numpy()
        persistence = np.asarray(last_observed_normalized, dtype=np.float64)
        if persistence.ndim == 3 and persistence.shape[-1] == 1:
            persistence = persistence[..., 0]
        if persistence.shape != (target.shape[0], target.shape[2]):
            raise ValueError("last observation must have shape [B,N] or [B,N,1]")

        prediction = np.asarray(self.stats.denormalize(prediction), dtype=np.float64)
        target = np.asarray(self.stats.denormalize(target), dtype=np.float64)
        persistence = np.asarray(self.stats.denormalize(persistence), dtype=np.float64)
        model_error = prediction - target
        persistence_error = persistence[:, None, :] - target
        axes = (0, 2)
        self._model_abs += np.abs(model_error).sum(axis=axes)
        self._model_sq += np.square(model_error).sum(axis=axes)
        self._persistence_abs += np.abs(persistence_error).sum(axis=axes)
        self._persistence_sq += np.square(persistence_error).sum(axis=axes)
        self._count += target.shape[0] * target.shape[2]

    def _finish(self, absolute: np.ndarray, squared: np.ndarray) -> HorizonMetrics:
        """把累计误差和转换为逐步 MAE/RMSE，并检查每步均有样本。"""
        if np.any(self._count == 0):
            raise RuntimeError("cannot summarize metrics before every horizon has observations")
        return HorizonMetrics(
            horizon_steps=tuple(range(1, self.horizon + 1)),
            mae=tuple((absolute / self._count).tolist()),
            rmse=tuple(np.sqrt(squared / self._count).tolist()),
            count=tuple(int(value) for value in self._count),
        )

    def compute(self) -> ForecastMetrics:
        """生成模型与 persistence 基线的最终不可变指标对象。"""
        return ForecastMetrics(
            model=self._finish(self._model_abs, self._model_sq),
            persistence=self._finish(self._persistence_abs, self._persistence_sq),
        )
