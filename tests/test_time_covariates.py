from pathlib import Path

import numpy as np
import yaml

from tsjt_c1.data.loaders import TIME_FEATURE_REVISION
from tsjt_c1.data.pipeline import SpeedStats, build_forecast_features
from tsjt_c1.run import _canonical_hash, _stamp_time_feature_revision


def test_ten_minute_rows_decode_five_minute_weekly_slots_by_288() -> None:
    steps = 2 * 144
    values = np.zeros((steps, 1, 4), dtype=np.float64)
    values[:, 0, 1] = np.arange(steps) % 144 / 144.0
    values[:, 0, 3] = np.arange(steps) * 2

    features = build_forecast_features(
        values,
        SpeedStats(mean=0.0, std=1.0),
        steps_per_day=144,
        time_channel_steps_per_day=288,
    )

    np.testing.assert_array_equal(features[:144, 0, 2], 0.0)
    np.testing.assert_array_equal(features[144:, 0, 2], np.float32(1.0 / 7.0))


def test_time_feature_fix_changes_checkpoint_identity() -> None:
    config = yaml.safe_load(Path("configs/c1.yaml").read_text(encoding="utf-8"))
    old_hash = _canonical_hash(config)

    _stamp_time_feature_revision(config)

    assert config["data"]["time_feature_revision"] == TIME_FEATURE_REVISION
    assert _canonical_hash(config) != old_hash
