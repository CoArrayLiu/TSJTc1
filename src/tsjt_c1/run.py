"""唯一正式 TSJT-C1 的严格、可恢复 50-epoch 运行入口。

配置文件提供具体参数，但本模块把方法身份和实验协议编码成不可放宽的校验：模型
必须保留 Dense Equation-(5) NPM、prompt 图、一阶段 fixed TSB、last-value residual
和共享 6/12 decoder；目标域只能使用 PEMS-BAY 前三天训练，且固定 seed 2025、
50 epoch。只有 epoch-50 checkpoint 已经原子落盘后，才允许创建正式测试迭代器。

状态转换刻意设计为单向：

    training -> epoch_50.pt -> evaluation_started.json ->
    formal_candidate_metrics.json -> complete.json

若写入评估开始标记后、指标持久化前发生中断，该输出目录会被隔离，重启也不能静默
访问第二次正式测试。已完成目录和属于其他配置的目录都不可覆盖。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import yaml

from tsjt_c1.models.decoder import SharedCanonicalHourDecoder
from tsjt_c1.models.base import BaseTSJT, Equation5NodePromptingModule
from tsjt_c1.models.c1 import TSJTC1
from tsjt_c1.training import engine
from tsjt_c1.data.loaders import CITY_PROTOCOLS, TIME_FEATURE_REVISION


FORMAL_EPOCHS = 50
ROUTED_TRANSFER_MODES: frozenset[str] = frozenset()
ALLOWED_TRANSFER_MODES = frozenset({"fixed_tsb"})
FACTORY_PATH = "tsjt_c1.models.c1:TSJTC1"
FINAL_CHECKPOINT_NAME = "epoch_50.pt"
FINAL_METRICS_NAME = "formal_candidate_metrics.json"


TrainEpoch = Callable[..., dict[str, Any]]
TestLoaderFactory = Callable[[Any], Any]
Evaluator = Callable[..., dict[str, Any]]


class _OneShotTestLoader:
    """限制底层正式测试 loader 最多创建一次迭代器的代理。

    ``len`` 和其他属性仍透明转发，但第二次调用 ``iter(loader)`` 会立即报错。这是
    进程内防止重复正式评估的最后一道保护。
    """

    def __init__(self, loader: Any) -> None:
        """保存底层 loader，并把迭代状态初始化为未访问。"""
        self._loader = loader
        self._iterated = False

    def __iter__(self) -> Any:
        """首次返回底层迭代器，后续调用拒绝执行。"""
        if self._iterated:
            raise RuntimeError("official test loader cannot be iterated more than once")
        self._iterated = True
        return iter(self._loader)

    def __len__(self) -> int:
        """返回底层 loader 的 batch 数，不触发数据迭代。"""
        return len(self._loader)

    def __getattr__(self, name: str) -> Any:
        """把未知属性转发到底层 loader，例如 ``dataset``。"""
        return getattr(self._loader, name)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    """计算与键顺序无关的规范 JSON SHA256，作为配置身份。"""
    return engine._canonical_hash(payload)


def _stamp_time_feature_revision(config: dict[str, Any]) -> None:
    """把时间特征口径纳入运行与 checkpoint 身份。"""

    configured_revision = config["data"].get("time_feature_revision")
    if configured_revision not in (None, TIME_FEATURE_REVISION):
        raise ValueError(
            "configured time_feature_revision is incompatible with this data pipeline"
        )
    config["data"]["time_feature_revision"] = TIME_FEATURE_REVISION


def _model_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """提取并深拷贝模型构造参数。

    支持冻结配置采用的 ``model.kwargs`` 嵌套格式，同时拒绝把架构键同时散落在
    ``model`` 外层，避免同一参数出现两个来源。
    """
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("model section must be a mapping")
    nested = model.get("kwargs")
    if nested is not None:
        if not isinstance(nested, Mapping):
            raise ValueError("model.kwargs must be a mapping")
        architecture_outside = set(model) - {
            "factory", "kwargs", "candidate_id", "variant"
        }
        if architecture_outside:
            raise ValueError(
                "nested model.kwargs cannot be mixed with flat architecture keys: "
                f"{sorted(architecture_outside)}"
            )
        return copy.deepcopy(dict(nested))
    return {
        str(key): copy.deepcopy(value)
        for key, value in model.items()
        if key not in {"factory", "candidate_id", "variant"}
    }


def _runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """为底层训练引擎生成 model kwargs 已展平的深拷贝，不改变原配置身份。"""

    runtime = copy.deepcopy(dict(config))
    runtime["model"] = _model_kwargs(config)
    return runtime


def _validate_source_protocols(config: Mapping[str, Any]) -> None:
    """锁定三个源城市及其采样频率、H/T、stride 和 70/10 边界。

    城市必须恰好是 METR-LA、Chengdu、Shenzhen；任何缺失、重复或额外源城市都会
    使正式实验身份失效。
    """
    data = config["data"]
    sources = data["sources"]
    expected_cities = {"metr-la", "chengdu", "shenzhen"}
    observed_cities = {str(item["city"]).lower() for item in sources}
    if len(sources) != 3 or observed_cities != expected_cities:
        raise ValueError(
            "formal candidate sources must be exactly METR-LA, Chengdu, Shenzhen"
        )
    for source in sources:
        city = str(source["city"]).lower()
        expected = CITY_PROTOCOLS[city]
        observed = {key: int(source[key]) for key in expected}
        if observed != expected:
            raise ValueError(
                f"formal source protocol for {city} must be {expected}; got {observed}"
            )
        if int(source.get("stride", 0)) != 4:
            raise ValueError("formal source prefix stride must be 4")
        if float(source.get("train_ratio", -1.0)) != 0.7:
            raise ValueError("formal source train_ratio must be 0.7")
        if float(source.get("val_ratio", -1.0)) != 0.1:
            raise ValueError("formal source val_ratio must be 0.1")


def _validate_routing(config: Mapping[str, Any]) -> None:
    """验证迁移模式；当前单一正式版只允许 ``fixed_tsb``。

    下方 routed 分支保留了更严格的结构校验框架，但由于允许集合中只有 fixed_tsb，
    正式 C1 不会进入这些研究模式。
    """
    train = config["train"]
    mode = str(train.get("transfer_mode", ""))
    if mode not in ALLOWED_TRANSFER_MODES:
        raise ValueError(
            "formal candidate transfer_mode must preserve TSB and be one of "
            f"{sorted(ALLOWED_TRANSFER_MODES)}"
        )
    if mode == "fixed_tsb":
        return
    configured = [
        str(item["city"]).lower() for item in config["data"]["sources"]
    ]
    active = [str(city).lower() for city in train.get("active_sources", [])]
    if not active or len(active) != len(set(active)):
        raise ValueError("routed TSB active_sources must be nonempty and unique")
    if not set(active).issubset(configured):
        raise ValueError("routed TSB active_sources must be configured source cities")
    if int(train.get("routing_refresh_interval", 0)) <= 0:
        raise ValueError("routed TSB routing_refresh_interval must be positive")
    beta = float(train.get("routing_ema_beta", -1.0))
    if not 0.0 <= beta < 1.0:
        raise ValueError("routed TSB routing_ema_beta must lie in [0,1)")

    if mode == "sparse_tsb":
        routing = str(train.get("source_routing", ""))
        if routing not in {"static", "adaptive"}:
            raise ValueError("sparse_tsb source_routing must be static or adaptive")
        top_k = int(train.get("source_top_k", -1))
        if not 0 <= top_k <= len(active):
            raise ValueError("sparse_tsb source_top_k is incompatible with active_sources")
        if routing == "static" and top_k != len(active):
            raise ValueError("static sparse_tsb requires source_top_k=len(active_sources)")
        threshold = float(train.get("routing_threshold", 2.0))
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("sparse_tsb routing_threshold must lie in [-1,1]")
    elif mode == "horizon_weighted_tsb":
        groups = train.get("horizon_groups")
        if not isinstance(groups, list) or not groups:
            raise ValueError("horizon_weighted_tsb requires horizon_groups")
        flattened: list[int] = []
        for group in groups:
            if not isinstance(group, list) or not group:
                raise ValueError("each horizon group must be a nonempty list")
            steps = [int(step) for step in group]
            if any(step < 1 or step > 12 for step in steps):
                raise ValueError("horizon group steps must be one-based in [1,12]")
            flattened.extend(steps)
        if len(flattened) != len(set(flattened)):
            raise ValueError("horizon_groups must not overlap")


def _validate_model_config(config: Mapping[str, Any]) -> None:
    """验证模型工厂及所有决定 C1 身份的关键架构参数。

    这不是一般性的超参数检查，而是冻结模型守卫：prompt bank 必须为 162、hidden
    dim 与 decoder 等必须符合 C1 checkpoint，禁止通过配置悄悄弱化 NPM 或图路径。
    """
    model = config["model"]
    factory = str(model.get("factory", ""))
    if factory != FACTORY_PATH:
        raise ValueError(f"model.factory must be {FACTORY_PATH!r}")
    kwargs = _model_kwargs(config)
    exact = {
        "backbone": "graph_wavenet",
        "input_dim": 3,
        "canonical_history": 12,
        "prompt_bank_size": 162,
        "supported_horizons": [6, 12],
        "use_reverse_support": True,
        "use_knowledge_graph": True,
        "last_value_residual": True,
    }
    for key, expected in exact.items():
        observed = kwargs.get(key)
        if key == "supported_horizons" and observed is not None:
            observed = [int(value) for value in observed]
        if observed != expected:
            raise ValueError(
                f"formal candidate model.{key} must be {expected!r}; got {observed!r}"
            )
    if str(kwargs.get("prompt_bank_rule", "")) != "half_target_nodes":
        raise ValueError("formal candidate prompt_bank_rule must be half_target_nodes")
    optional_exact = {
        "prompt_feature_mode": "npm",
        "graph_mode": "prompt",
        "decoder_mode": "shared",
        "disable_npm_when_unused": True,
    }
    for key, expected in optional_exact.items():
        if key in kwargs and kwargs[key] != expected:
            raise ValueError(
                f"formal candidate cannot weaken {key}; expected {expected!r}"
            )
    if "npm_correction_mode" in kwargs and str(
        kwargs["npm_correction_mode"]
    ).lower() not in {"none", "target_gate", "horizon_gate"}:
        raise ValueError("unsupported target NPM correction mode")


def _validate_config(config: Mapping[str, Any]) -> None:
    """执行正式实验的完整静态协议校验。

    校验范围包括目录身份、四城市文件存在性、前三天少样本目标协议、batch 分配、
    固定步长、50 epoch、一次性评估以及禁止 validation/early-stop 选择键。该函数
    只检查配置和文件存在性，不读取正式测试样本。
    """
    required = {"run", "data", "model", "train", "evaluation"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"formal candidate config is missing: {sorted(missing)}")
    if "development" in config or "validation" in config:
        raise ValueError("formal candidate cannot contain development/validation sections")
    run = config["run"]
    data = config["data"]
    target = data["target"]
    train = config["train"]
    evaluation = config["evaluation"]
    if int(run.get("seed", -1)) != 2025:
        raise ValueError("formal candidate seed must be 2025")
    if str(run.get("device", "")) != "cuda:0":
        raise ValueError("formal candidate device must be cuda:0")
    if not bool(run.get("formal", False)):
        raise ValueError("formal candidate requires run.formal=true")
    if not bool(run.get("resume", False)):
        raise ValueError("formal candidate must be resumable")
    if not bool(run.get("skip_complete", False)):
        raise ValueError("formal candidate requires immutable skip_complete=true")
    output = Path(str(run.get("output_dir", "")))
    output_name = output.name.lower()
    if (
        output_name != "tsjt_c1_pemsbay_seed2025"
        or output.parent.name.lower() != "outputs"
    ):
        raise ValueError(
            "formal output must be outputs/tsjt_c1_pemsbay_seed2025"
        )
    data_root = Path(str(data.get("root", ""))).resolve()
    required_city_files = {
        data_root / city / filename
        for city in CITY_PROTOCOLS
        for filename in ("dataset.npy", "matrix.npy")
    }
    missing_city_files = sorted(
        str(path) for path in required_city_files if not path.is_file()
    )
    if missing_city_files:
        raise ValueError(
            "formal candidate data root is incomplete; missing "
            + ", ".join(missing_city_files)
        )
    if str(target.get("city", "")).lower() != "pems-bay":
        raise ValueError("formal candidate target must be PEMS-BAY")
    expected_target = CITY_PROTOCOLS["pems-bay"]
    observed_target = {key: int(target[key]) for key in expected_target}
    if observed_target != expected_target:
        raise ValueError(
            f"formal PEMS-BAY protocol must be {expected_target}; got {observed_target}"
        )
    if int(target.get("train_days", 0)) != 3:
        raise ValueError("formal candidate must train on exactly target days 1-3")
    if int(target.get("train_stride", 0)) != 1 or int(
        target.get("test_stride", 0)
    ) != 1:
        raise ValueError("formal target train/test stride must be 1")
    if bool(target.get("allow_prior_split_history", True)):
        raise ValueError("formal target windows must be split-contained")
    _validate_source_protocols(config)

    if int(train.get("epochs", 0)) != FORMAL_EPOCHS:
        raise ValueError("formal candidate must train exactly 50 epochs")
    if "max_updates_per_epoch" in train:
        raise ValueError("formal candidate cannot cap epoch updates")
    if int(train.get("target_batch_size", 0)) != 4:
        raise ValueError("formal target batch size must be 4")
    if int(train.get("source_batch_size", 0)) != 124:
        raise ValueError("formal logical source batch size must be 124")
    if int(train.get("source_microbatch_size", 0)) != 42:
        raise ValueError("formal source microbatch size must be 42")
    if float(train.get("gamma_target", -1.0)) != 0.001:
        raise ValueError("formal gamma_target must be exactly 0.001")
    if float(train.get("gamma_source", -1.0)) != 0.0005:
        raise ValueError("formal gamma_source must be exactly 0.0005")
    if int(train.get("save_every", 0)) != 1:
        raise ValueError("formal candidate must checkpoint every epoch")
    if int(train.get("max_epochs", FORMAL_EPOCHS)) != FORMAL_EPOCHS:
        raise ValueError("formal max_epochs, when present, must equal 50")
    _validate_routing(config)

    if not bool(evaluation.get("full_test", False)):
        raise ValueError("formal candidate requires one full official test")
    if int(evaluation.get("evaluate_once_after_epoch", 0)) != FORMAL_EPOCHS:
        raise ValueError("official test is allowed only after epoch 50")
    if int(evaluation.get("batch_size", 0)) <= 0:
        raise ValueError("formal evaluation batch size must be positive")
    forbidden_words = ("validation", "early_stop", "patience", "best_val")
    for section_name, section in (("target", target), ("train", train), ("evaluation", evaluation)):
        forbidden = [
            str(key)
            for key in section
            if any(word in str(key).lower() for word in forbidden_words)
        ]
        if forbidden:
            raise ValueError(
                f"{section_name} contains forbidden selection keys: {sorted(forbidden)}"
            )
    _validate_model_config(config)



def _assert_model_invariants(model: torch.nn.Module) -> None:
    """在模型实例化后检查运行时结构，防止工厂返回伪装成 C1 的其他模型。

    除类型和开关外，还逐对象比较 H=6/H=12 参数集合，确保两个任务真正共享同一
    decoder 和全部可训练参数，而不是仅拥有形状相同的副本。
    """
    if not isinstance(model, BaseTSJT):
        raise TypeError("formal candidate must subclass BaseTSJT")
    if not isinstance(getattr(model, "npm", None), Equation5NodePromptingModule):
        raise TypeError("formal candidate must use the original dense Equation-(5) NPM")
    prompt_bank = model.npm.prompt_bank
    if not isinstance(prompt_bank, torch.nn.Parameter) or prompt_bank.ndim != 3:
        raise TypeError("formal candidate requires a registered dense prompt bank")
    if getattr(model, "npm_disabled", False):
        raise ValueError("formal candidate cannot disable NPM")
    if getattr(model, "prompt_feature_mode", "npm") != "npm":
        raise ValueError("formal candidate must feed NPM prompt features")
    if getattr(model, "graph_mode", "prompt") != "prompt":
        raise ValueError("formal candidate must use the prompt-derived graph")
    if getattr(model, "decoder_mode", "shared") != "shared":
        raise ValueError("formal candidate must use the shared decoder")
    if not bool(getattr(model, "last_value_residual", False)):
        raise ValueError("formal candidate must retain the last-value residual")
    if not bool(getattr(model, "use_knowledge_graph", False)):
        raise ValueError("formal candidate must retain the knowledge graph")
    if tuple(getattr(model, "supported_horizons", ())) != (6, 12):
        raise ValueError("formal candidate must support shared horizons (6,12)")
    if not isinstance(getattr(model, "output_decoder", None), SharedCanonicalHourDecoder):
        raise TypeError("formal candidate must retain SharedCanonicalHourDecoder")
    parameter_sets = model.task_parameter_sets()
    if set(parameter_sets) != {6, 12} or len(parameter_sets[6]) != len(
        parameter_sets[12]
    ):
        raise ValueError("formal candidate task parameter sets are incompatible")
    if not all(
        left is right
        for left, right in zip(parameter_sets[6], parameter_sets[12])
    ):
        raise ValueError("formal candidate 6/12 task parameters must be shared")


def build_model(config: Mapping[str, Any]) -> tuple[torch.nn.Module, str]:
    """从冻结配置创建唯一 ``TSJTC1`` 实例，并返回稳定工厂身份字符串。"""
    _validate_model_config(config)
    model = TSJTC1(**_model_kwargs(config))
    identity = FACTORY_PATH
    _assert_model_invariants(model)
    return model, identity


def _guarded_prepare_output(
    config_path: Path,
    config: dict[str, Any],
    device: torch.device,
    argv: Sequence[str],
) -> tuple[Path, str]:
    """检查输出目录归属后写入运行元数据。

    若目录已存在，必须包含与当前配置哈希一致的 ``resolved_config.json``；否则拒绝
    覆盖。实际创建目录、复制 launch config 和记录软件/GPU 信息由 engine 完成。
    """
    output_dir = Path(config["run"]["output_dir"]).resolve()
    config_hash = _canonical_hash(config)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise RuntimeError("formal candidate output path is not a directory")
        resolved = output_dir / "resolved_config.json"
        if not resolved.is_file():
            raise RuntimeError(
                "pre-existing formal candidate output is unrecognized and immutable"
            )
        try:
            previous = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("existing resolved config is invalid; refusing overwrite") from error
        if not isinstance(previous, Mapping) or _canonical_hash(previous) != config_hash:
            raise RuntimeError("formal candidate output belongs to another config")
    return engine._prepare_output(
        config_path, config, device, argv
    )


def _repair_log(path: Path, checkpoint: Mapping[str, Any]) -> None:
    """在仅缺最后一条时用 checkpoint 内的 epoch_record 安全修复 JSONL。

    checkpoint 先于日志原子写入，因此进程可能恰好在两者之间中断。只允许修复
    一条记录；更大缺口或日志超前意味着证据不一致，必须停止。
    """
    records: list[dict[str, Any]] = []
    if path.exists():
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    epoch = int(checkpoint["epoch"])
    if len(records) > epoch:
        raise RuntimeError("formal training log is ahead of its checkpoint")
    if len(records) < epoch:
        if len(records) != epoch - 1:
            raise RuntimeError("formal log/checkpoint gap is not safely repairable")
        engine._append_jsonl(path, checkpoint["epoch_record"])


def _validated_metrics(
    path: Path, config_hash: str, final_checkpoint: Path
) -> dict[str, Any]:
    """读取并验证已落盘正式指标是否匹配配置、epoch 和一次评估约束。"""
    if not path.is_file():
        raise RuntimeError("formal completion marker is missing durable metrics")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("formal candidate metrics are invalid") from error
    if not isinstance(result, dict):
        raise RuntimeError("formal candidate metrics root is invalid")
    expected = {
        "config_hash": config_hash,
        "trained_epochs": FORMAL_EPOCHS,
        "checkpoint_epoch": FORMAL_EPOCHS,
        "test_evaluations": 1,
    }
    observed = {key: result.get(key) for key in expected}
    if observed != expected or not final_checkpoint.is_file():
        raise RuntimeError(
            "formal candidate metrics/checkpoint evidence is incompatible"
        )
    return result


def run_state_machine(
    config: dict[str, Any],
    output_dir: Path,
    config_hash: str,
    device: torch.device,
    data_bundle: tuple[Any, ...],
    model: torch.nn.Module,
    *,
    train_epoch: TrainEpoch = engine._train_epoch,
    test_loader_factory: TestLoaderFactory | None = None,
    evaluator: Evaluator = engine._evaluate_full_test,
) -> Path:
    """执行/恢复训练，并在安全边界内完成一次正式评估。

    恢复时同时加载模型、epoch、配置哈希和随机数状态，从下一 epoch 确定性继续。
    每个 epoch 先写 ``last.pt``，第 50 轮额外写冻结 checkpoint，再追加 JSONL。
    训练完成后先写正式评估标记，再创建测试 loader，最后原子写指标与完成标记。
    返回最终指标文件路径。
    """

    output_dir = output_dir.resolve()
    complete_path = output_dir / "complete.json"
    metrics_path = output_dir / FINAL_METRICS_NAME
    final_checkpoint = output_dir / FINAL_CHECKPOINT_NAME
    evaluation_started = output_dir / "evaluation_started.json"
    # 已完成目录不可变；允许幂等调用仅返回已经验证过的指标路径。
    if complete_path.exists():
        _validated_metrics(metrics_path, config_hash, final_checkpoint)
        if bool(config["run"].get("skip_complete", True)):
            return metrics_path
        raise RuntimeError("completed formal candidate output is immutable")
    if metrics_path.exists():
        # 指标已持久化但 complete 标记缺失时，可安全补写标记，不需要再次迭代测试集。
        result = _validated_metrics(metrics_path, config_hash, final_checkpoint)
        if not evaluation_started.is_file():
            raise RuntimeError("formal metrics exist without evaluation counter")
        marker = json.loads(evaluation_started.read_text(encoding="utf-8"))
        if marker.get("test_evaluations_started") != 1:
            raise RuntimeError("formal evaluation counter is incompatible")
        engine._atomic_json(
            complete_path,
            {
                "config_hash": config_hash,
                "checkpoint": str(final_checkpoint),
                "checkpoint_epoch": FORMAL_EPOCHS,
                "formal_candidate_metrics": str(metrics_path),
                "test_evaluations": int(result["test_evaluations"]),
            },
        )
        return metrics_path
    # 只有开始标记、没有指标意味着上次评估中断；拒绝任何第二次正式测试访问。
    if evaluation_started.exists():
        raise RuntimeError(
            "official evaluation was started but did not finish; refusing a second test access"
        )

    (
        target_loaders,
        target_adjacency,
        stats,
        target_protocol,
        source_loaders,
        source_adjacencies,
        _,
    ) = data_bundle
    model = model.to(device)
    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    last_path = output_dir / "last.pt"
    log_path = output_dir / "train.jsonl"
    start_epoch = 0
    mode = str(config["train"]["transfer_mode"])
    router_state: dict[str, Any] = {}
    # 恢复点包含模型和所有 RNG 状态，保证 epoch 级打乱与 dropout 可复现。
    if last_path.exists():
        if not bool(config["run"].get("resume", True)):
            raise RuntimeError("formal checkpoint exists but resume=false")
        payload = engine._load_checkpoint(
            last_path, model, config_hash, restore_rng=True
        )
        _repair_log(log_path, payload)
        start_epoch = int(payload["epoch"])
        if not 0 <= start_epoch <= FORMAL_EPOCHS:
            raise RuntimeError("formal checkpoint epoch is out of range")
        if mode in ROUTED_TRANSFER_MODES:
            stored_router = payload.get("routing_state")
            if not isinstance(stored_router, dict):
                raise RuntimeError(
                    "routed formal checkpoint lacks deterministic routing_state"
                )
            router_state = copy.deepcopy(stored_router)
    elif log_path.exists() or final_checkpoint.exists():
        raise RuntimeError("formal output contains orphan training evidence")

    # 正常新运行从 1 开始；中断恢复从 checkpoint_epoch + 1 开始。
    for epoch in range(start_epoch + 1, FORMAL_EPOCHS + 1):
        training_kwargs: dict[str, Any] = {}
        if mode in ROUTED_TRANSFER_MODES:
            training_kwargs["router_state"] = router_state
        record = train_epoch(
            model,
            parameters,
            epoch,
            config,
            target_loaders["train"].dataset,
            target_adjacency,
            {city: loader.dataset for city, loader in source_loaders.items()},
            source_adjacencies,
            device,
            **training_kwargs,
        )
        if not isinstance(record, dict):
            raise TypeError("formal train_epoch must return a record mapping")
        record = dict(record)
        record["epoch"] = epoch
        record["transfer_mode"] = mode
        record["official_test_evaluations"] = 0
        payload: dict[str, Any] = {
            "model": model.state_dict(),
            "epoch": epoch,
            "config_hash": config_hash,
            "rng_state": engine._rng_state(),
            "epoch_record": record,
        }
        if mode in ROUTED_TRANSFER_MODES:
            payload["routing_state"] = copy.deepcopy(router_state)
        # checkpoint 必须先于日志写入；若其后中断，_repair_log 可补最后一条日志。
        engine._atomic_checkpoint(last_path, payload)
        if epoch == FORMAL_EPOCHS:
            # epoch-50 权重先完整落盘，随后才允许创建任何评估标记或测试迭代器。
            engine._atomic_checkpoint(final_checkpoint, payload)
        engine._append_jsonl(log_path, record)

    if not final_checkpoint.is_file():
        raise RuntimeError("durable epoch-50 checkpoint missing; test access refused")
    final_payload = engine._load_checkpoint(
        final_checkpoint, model, config_hash, restore_rng=False
    )
    if int(final_payload.get("epoch", -1)) != FORMAL_EPOCHS:
        raise RuntimeError("final checkpoint is not epoch 50")
    engine._atomic_json(
        evaluation_started,
        {
            "config_hash": config_hash,
            "checkpoint": str(final_checkpoint),
            "checkpoint_epoch": FORMAL_EPOCHS,
            "test_evaluations_started": 1,
        },
    )

    if test_loader_factory is None:
        runtime = _runtime_config(config)
        test_loader_factory = lambda dataset: engine._evaluation_loader(
            dataset, runtime, device
        )
    # 读取 .dataset 属性本身不迭代数据；真正 iterator 由 evaluator 且仅由它创建一次。
    test_loader = _OneShotTestLoader(
        test_loader_factory(target_loaders["test"].dataset)
    )
    test = evaluator(
        model,
        test_loader,
        target_adjacency,
        stats,
        int(target_protocol["horizon"]),
        device,
    )
    result = {
        "config_hash": config_hash,
        "seed": int(config["run"]["seed"]),
        "trained_epochs": FORMAL_EPOCHS,
        "transfer_mode": mode,
        "model_factory": str(config["model"]["factory"]),
        "checkpoint": str(final_checkpoint),
        "checkpoint_epoch": int(final_payload["epoch"]),
        "selection": "none; locked candidate and fixed epoch-50 weights",
        "test_evaluations": 1,
        "test": test,
        "protocol": target_protocol,
    }
    if mode in ROUTED_TRANSFER_MODES:
        result["routing_state"] = copy.deepcopy(router_state)
    engine._atomic_json(metrics_path, result)
    engine._atomic_json(
        complete_path,
        {
            "config_hash": config_hash,
            "checkpoint": str(final_checkpoint),
            "checkpoint_epoch": FORMAL_EPOCHS,
            "formal_candidate_metrics": str(metrics_path),
            "test_evaluations": 1,
        },
    )
    return metrics_path


def run(config_path: str | Path, argv: Sequence[str] | None = None) -> Path:
    """正式 CLI 的编排函数：读取配置、校验、建模、记录指纹并进入状态机。

    该函数在 ``_validate_config`` 通过前不会创建输出；数据指纹和协议会在训练前
    落盘，方便之后确认本次运行使用的确切数据资产。
    """
    config_path = Path(config_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("formal candidate configuration root must be a mapping")
    _validate_config(config)
    # 把数据修复版本纳入 config/checkpoint 哈希，防止旧 DOW 权重被继续训练。
    _stamp_time_feature_revision(config)
    engine._set_seed(int(config["run"]["seed"]))
    device = engine._resolve_device(str(config["run"]["device"]))
    model, _ = build_model(config)
    output_dir, config_hash = _guarded_prepare_output(
        config_path, config, device, argv or sys.argv
    )
    cities = [
        *(str(item["city"]).lower() for item in config["data"]["sources"]),
        str(config["data"]["target"]["city"]).lower(),
    ]
    engine._fingerprint_data(
        Path(config["data"]["root"]), cities, output_dir
    )
    runtime = _runtime_config(config)
    data_bundle = engine._make_data(runtime, device, output_dir)
    return run_state_machine(
        config, output_dir, config_hash, device, data_bundle, model
    )


def main(argv: Sequence[str] | None = None) -> int:
    """解析 ``--config``，运行正式流程，并以 JSON 打印最终 artifact 路径。"""
    parser = argparse.ArgumentParser(
        description="Strict fixed-50 formal runner for TSJT-C1"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    artifact = run(args.config, argv or sys.argv)
    print(json.dumps({"artifact": str(artifact)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_TRANSFER_MODES",
    "FINAL_CHECKPOINT_NAME",
    "FINAL_METRICS_NAME",
    "FORMAL_EPOCHS",
    "build_model",
    "main",
    "run",
    "run_state_machine",
]
