"""C2 双城协议、六步模型和 MAE/MAPE 口径的回归测试。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import yaml
from torch.utils.data import TensorDataset

from tsjt_c2.data.loaders import CITY_PROTOCOLS
from tsjt_c2.data.pipeline import ForecastMetricAccumulator, SpeedStats
from tsjt_c2.models.backbone import (
    MAX_TEMPORAL_ATTENTION_BATCH,
    SpatioTemporalPatternEncoder,
)
from tsjt_c2.models.c2 import TSJTC2
from tsjt_c2.run import _validate_config, build_model, run_state_machine
from tsjt_c2.training.engine import _evaluate_target_train_mae


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "c2"


def _configs() -> list[tuple[Path, dict]]:
    return [
        (path, yaml.safe_load(path.read_text(encoding="utf-8")))
        for path in sorted(CONFIG_DIR.glob("*.yaml"))
    ]


def test_four_formal_directions_are_valid_and_unique() -> None:
    configs = _configs()
    assert len(configs) == 4
    directions: set[tuple[str, str]] = set()
    outputs: set[str] = set()
    for _, config in configs:
        _validate_config(config)
        source = config["data"]["sources"][0]["city"]
        target = config["data"]["target"]["city"]
        directions.add((source, target))
        outputs.add(config["run"]["output_dir"])
        assert config["data"]["target"]["train_days"] == 2
        assert config["evaluation"]["report_horizons"] == [1, 3, 6]
    assert directions == {
        ("pems-bay", "metr-la"),
        ("metr-la", "pems-bay"),
        ("chengdu", "shenzhen"),
        ("shenzhen", "chengdu"),
    }
    assert len(outputs) == 4


def test_cross_frequency_direction_is_rejected() -> None:
    config = copy.deepcopy(_configs()[0][1])
    source = config["data"]["sources"][0]
    source.update(city="metr-la", **CITY_PROTOCOLS["metr-la"])
    config["run"]["output_dir"] = "outputs/tsjt_c2_metrla_to_shenzhen_seed2025"
    with pytest.raises(ValueError, match="one group"):
        _validate_config(config)


@pytest.mark.parametrize(
    ("config_name", "expected_minutes", "expected_bank"),
    [
        ("pemsbay_to_metrla.yaml", [5, 10, 15, 20, 25, 30], 103),
        ("chengdu_to_shenzhen.yaml", [10, 20, 30, 40, 50, 60], 313),
    ],
)
def test_c2_model_is_native_six_step(
    config_name: str, expected_minutes: list[int], expected_bank: int
) -> None:
    config = yaml.safe_load((CONFIG_DIR / config_name).read_text(encoding="utf-8"))
    model, identity = build_model(config)
    assert isinstance(model, TSJTC2)
    assert identity == "tsjt_c2.models.c2:TSJTC2"
    assert model.supported_horizons == (6,)
    assert model.npm.prompt_bank.shape == (expected_bank, 6, 32)
    assert model.output_decoder.lead_minutes(6).tolist() == expected_minutes
    assert model.cprr_parameter_count == 198

    x = torch.randn(2, 6, 5, 3)
    adjacency = torch.eye(5)
    prediction = model(x, adjacency, horizon=6)
    assert prediction.shape == (2, 6, 5, 1)
    assert torch.isfinite(prediction).all()


def test_metric_accumulator_reports_mae_and_percentage_mape() -> None:
    accumulator = ForecastMetricAccumulator(2, SpeedStats(mean=0.0, std=1.0))
    prediction = torch.tensor([[[[12.0]], [[18.0]]]])
    target = torch.tensor([[[[10.0]], [[20.0]]]])
    last = torch.tensor([[8.0]])
    accumulator.update(prediction, target, last)
    metrics = accumulator.compute().model
    assert metrics.horizon_steps == (1, 2)
    assert metrics.mae == pytest.approx((2.0, 2.0))
    assert metrics.mape == pytest.approx((20.0, 10.0))


def test_temporal_attention_chunks_batch_node_axis_at_cuda_limit() -> None:
    class RecordingAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes: list[int] = []

        def forward(self, query, key, value, *, need_weights):  # type: ignore[no-untyped-def]
            assert query is key and key is value
            assert need_weights is False
            self.batch_sizes.append(int(query.shape[0]))
            return query + 1.0, None

    encoder = SpatioTemporalPatternEncoder(
        input_dim=3,
        hidden_dim=4,
        heads=1,
        dropout=0.0,
        canonical_steps=6,
    )
    recorder = RecordingAttention()
    encoder.attention = recorder
    sequence = torch.zeros(MAX_TEMPORAL_ATTENTION_BATCH + 7, 2, 4)

    attended = encoder._temporal_attention(sequence)

    assert recorder.batch_sizes == [MAX_TEMPORAL_ATTENTION_BATCH, 7]
    assert attended.shape == sequence.shape
    assert torch.equal(attended, torch.ones_like(sequence))


def test_early_stopping_is_bounded_without_test_selection() -> None:
    for _, config in _configs():
        early = config["train"]["early_stopping"]
        assert config["train"]["max_epochs"] == 50
        assert early == {
            "monitor": "target_train_eval_mae",
            "patience": 10,
            "min_delta": 0.0,
        }
        assert config["evaluation"]["evaluate_once_after_training"] is True


def test_state_machine_early_stops_and_reports_requested_steps(tmp_path: Path) -> None:
    """用轻量替身验证 patience、best/final checkpoint 和一次评测摘要。"""

    config = copy.deepcopy(_configs()[0][1])
    config["train"]["early_stopping"]["patience"] = 2
    model = torch.nn.Linear(1, 1)
    dataset_holder = SimpleNamespace(dataset=object())
    data_bundle = (
        {"train": dataset_holder, "test": dataset_holder},
        torch.eye(1),
        SpeedStats(0.0, 1.0),
        {"city": "shenzhen", "history": 6, "horizon": 6},
        {"chengdu": dataset_holder},
        {"chengdu": torch.eye(1)},
        {},
    )
    # 在线训练 loss 持续改善，但固定权重的 eval-mode MAE 持续变差；早停必须只
    # 服从后者，并选择 epoch 1。
    online_losses = {1: 3.0, 2: 2.0, 3: 1.0}
    evaluated_maes = {1: 1.0, 2: 1.1, 3: 1.2}
    evaluated_epochs: list[int] = []

    def train_epoch(*args, **kwargs):  # type: ignore[no-untyped-def]
        epoch = int(args[2])
        model.epoch = epoch  # type: ignore[attr-defined]
        return {"target_online_loss": online_losses[epoch], "source_loss": 0.0}

    def target_train_evaluator(*args, **kwargs):  # type: ignore[no-untyped-def]
        epoch = int(model.epoch)  # type: ignore[attr-defined]
        evaluated_epochs.append(epoch)
        return evaluated_maes[epoch]

    def evaluator(model, loader, adjacency, stats, horizon, device):  # type: ignore[no-untyped-def]
        assert list(loader) == ["official-test"]
        metric = {
            "horizon_steps": [1, 2, 3, 4, 5, 6],
            "mae": [1.0] * 6,
            "mape": [2.0] * 6,
            "count": [3] * 6,
        }
        return {"metrics": {"model": metric, "persistence": metric}}

    artifact = run_state_machine(
        config,
        tmp_path,
        "test-hash",
        torch.device("cpu"),
        data_bundle,
        model,
        train_epoch=train_epoch,
        target_train_evaluator=target_train_evaluator,
        test_loader_factory=lambda dataset: ["official-test"],
        evaluator=evaluator,
    )
    result = json.loads(artifact.read_text(encoding="utf-8"))
    assert result["trained_epochs"] == 3
    assert result["checkpoint_epoch"] == 1
    assert result["early_stopping"]["stopped_early"] is True
    assert result["early_stopping"]["best_target_train_eval_mae"] == 1.0
    assert evaluated_epochs == [1, 2, 3]
    records = [
        json.loads(line)
        for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["target_online_loss"] == 1.0
    assert records[-1]["target_train_eval_mae"] == 1.2
    assert result["reported_metrics"]["model"]["6"] == {
        "mae": 1.0,
        "mape": 2.0,
        "count": 3,
    }
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "final.pt").is_file()


def test_target_train_mae_uses_fixed_eval_mode_and_element_weighting() -> None:
    class ZeroForecast(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.observed_training_modes: list[bool] = []

        def forward(  # type: ignore[no-untyped-def]
            self, x, adjacency, *, horizon
        ) -> torch.Tensor:
            self.observed_training_modes.append(self.training)
            return torch.zeros_like(x[:, :horizon, :, :1])

    model = ZeroForecast()
    model.train()
    x = torch.zeros(3, 1, 1, 1)
    y = torch.tensor([0.0, 0.0, 9.0]).reshape(3, 1, 1, 1)
    dataset = TensorDataset(x, y)
    config = {
        "run": {"seed": 2025, "num_workers": 0},
        "evaluation": {"batch_size": 2},
    }

    mae = _evaluate_target_train_mae(
        model, dataset, torch.eye(1), 1, config, torch.device("cpu")
    )

    assert mae == pytest.approx(3.0)
    assert model.observed_training_modes == [False, False]
    assert model.training is True
