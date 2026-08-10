from pathlib import Path

import torch
import yaml

from tsjt_c1.models.c1 import TSJTC1
from tsjt_c1.run import _validate_config, build_model


def load_config() -> dict:
    return yaml.safe_load(Path("configs/c1.yaml").read_text(encoding="utf-8"))


def test_frozen_config_builds_the_only_model() -> None:
    config = load_config()
    _validate_config(config)
    model, identity = build_model(config)
    assert isinstance(model, TSJTC1)
    assert identity == "tsjt_c1.models.c1:TSJTC1"
    assert model.cprr_parameter_count == 396


def test_c1_forward_shape_and_finiteness() -> None:
    config = load_config()
    model, _ = build_model(config)
    x = torch.randn(1, 12, 5, 3)
    adjacency = torch.eye(5)
    with torch.no_grad():
        prediction = model(x, adjacency, horizon=12)
    assert prediction.shape == (1, 12, 5, 1)
    assert torch.isfinite(prediction).all()
