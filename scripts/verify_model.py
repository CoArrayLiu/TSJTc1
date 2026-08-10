"""Load the delivered C1 checkpoint and run a deterministic local-data forward."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tsjt_c1.data.pipeline import SpeedStats, build_forecast_features  # noqa: E402
from tsjt_c1 import run as formal_candidate  # noqa: E402


EXPECTED_TRAINING_CONFIG_HASH = (
    "f0b75d29433834c3a83411377f4b0cad1c6e9065ed43b8a1739e9dd693ae7aa0"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)

    config = yaml.safe_load((ROOT / "configs" / "c1.yaml").read_text(encoding="utf-8"))
    model, identity = formal_candidate.build_model(config)
    checkpoint_path = ROOT / "checkpoints" / "best_c1_seed2025_epoch50.pt"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(payload.get("epoch", -1)) != 50:
        raise RuntimeError("delivered checkpoint is not epoch 50")
    if payload.get("config_hash") != EXPECTED_TRAINING_CONFIG_HASH:
        raise RuntimeError("checkpoint training-config hash does not match the frozen run")
    model.load_state_dict(payload["model"], strict=True)
    model = model.to(device).eval()

    raw = np.load(ROOT / "data" / "pems-bay" / "dataset.npy", mmap_mode="r")
    adjacency = torch.from_numpy(
        np.asarray(np.load(ROOT / "data" / "pems-bay" / "matrix.npy"), dtype=np.float32)
    ).to(device)
    stats = SpeedStats(mean=64.25403846152764, std=7.967502039396337)
    features = build_forecast_features(
        np.asarray(raw[:12]), stats, steps_per_day=288
    )
    x = torch.from_numpy(features).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(x, adjacency, horizon=12)
    if prediction.shape != (1, 12, raw.shape[1], 1):
        raise RuntimeError(f"unexpected prediction shape: {tuple(prediction.shape)}")
    if not torch.isfinite(prediction).all():
        raise RuntimeError("checkpoint produced non-finite predictions")
    prediction_cpu = prediction.detach().float().cpu().contiguous().numpy()
    report = {
        "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_epoch": 50,
        "training_config_hash": payload["config_hash"],
        "model_factory": identity,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "input": "PEMS-BAY raw steps [0,12), normalized with frozen train statistics",
        "prediction_shape": list(prediction_cpu.shape),
        "prediction_sha256": hashlib.sha256(prediction_cpu.tobytes()).hexdigest(),
        "prediction_mean_normalized": float(prediction_cpu.mean()),
        "prediction_std_normalized": float(prediction_cpu.std()),
        "official_test_access": False,
    }
    if args.write:
        destination = ROOT / "results" / "checkpoint_verification.json"
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
