"""Typed, shared configuration used by Fig. 2, Fig. 3 and Fig. 4 experiments."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


VIT_BACKBONES = {"vit_b_16", "vit_b_32"}
RESNET_BACKBONES = {"resnet18", "resnet34", "resnet50", "resnet101"}
TEMPORAL_MODES = {"single", "mean", "lstm", "gru"}


@dataclass
class DataConfig:
    manifest: str = "data/manifest.csv"
    sample_id_column: str = "sample_id"
    participant_id_column: str = "participant_id"
    split_column: str = "split"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    frame_dir_column: str = "frame_dir"
    frames_column: str = "frames"
    frame_pattern: str = "{time}s.jpg"
    frame_times: list[int] = field(default_factory=lambda: list(range(0, 121, 10)))
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    strict_frames: bool = True
    validate_participant_splits: bool = True
    augmentation: bool = True
    brightness: float = 0.2
    contrast: float = 0.2
    rotation_degrees: float = 5.0
    blur_probability: float = 0.3
    blur_sigma_max: float = 1.0


@dataclass
class ModelConfig:
    backbone: str = "vit_b_16"
    pretrained: bool = True
    temporal_mode: str = "lstm"
    rnn_hidden_size: int = 256
    rnn_num_layers: int = 1
    rnn_bidirectional: bool = False
    dropout: float = 0.3
    pruning_ratio: float = 0.4
    pruning_method: str = "adaptive"


@dataclass
class TrainingConfig:
    epochs: int = 120
    learning_rate: float = 1.0e-4
    backbone_learning_rate_scale: float = 0.3
    weight_decay: float = 1.0e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    minimum_learning_rate: float = 1.0e-6
    use_amp: bool = True
    seed: int = 42
    device: str = "auto"
    gpu_ids: list[int] = field(default_factory=lambda: [0])
    data_parallel: bool = False
    output_dir: str = "runs"
    early_stopping_patience: int = 20


@dataclass
class EvaluationConfig:
    aggregate_by: list[str] = field(
        default_factory=lambda: ["participant_id", "cohort", "specimen"]
    )
    subgroup_by: list[str] = field(default_factory=list)
    hb_thresholds: list[float] = field(default_factory=lambda: [8.0, 10.0, 12.0])
    hct_thresholds: list[float] = field(default_factory=lambda: [24.0, 30.0, 36.0])
    bootstrap_replicates: int = 2000
    confidence_level: float = 0.95
    bootstrap_seed: int = 42
    save_predictions: bool = True


@dataclass
class ExperimentConfig:
    name: str = "hemowick"
    kind: str = "cohort_validation"
    variants: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    config_path: str = ""

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], config_path: str = "") -> "AppConfig":
        cfg = cls(
            data=DataConfig(**dict(raw.get("data", {}))),
            model=ModelConfig(**dict(raw.get("model", {}))),
            training=TrainingConfig(**dict(raw.get("training", {}))),
            evaluation=EvaluationConfig(**dict(raw.get("evaluation", {}))),
            experiment=ExperimentConfig(**dict(raw.get("experiment", {}))),
            config_path=config_path,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.model.backbone not in VIT_BACKBONES | RESNET_BACKBONES:
            raise ValueError(f"Unsupported backbone: {self.model.backbone}")
        if self.model.temporal_mode not in TEMPORAL_MODES:
            raise ValueError(f"Unsupported temporal_mode: {self.model.temporal_mode}")
        if not 0.0 <= self.model.pruning_ratio < 1.0:
            raise ValueError("model.pruning_ratio must be in [0, 1).")
        if self.model.pruning_method not in {"adaptive", "uniform"}:
            raise ValueError("model.pruning_method must be 'adaptive' or 'uniform'.")
        if self.model.backbone in RESNET_BACKBONES and self.model.pruning_ratio != 0.0:
            raise ValueError("Patch pruning is available only for ViT backbones.")
        if self.model.temporal_mode == "single" and len(self.data.frame_times) != 1:
            raise ValueError("Single-image models require exactly one data.frame_times value.")
        if not self.data.frame_times:
            raise ValueError("data.frame_times cannot be empty.")
        if self.model.backbone in VIT_BACKBONES:
            patch_size = 16 if self.model.backbone == "vit_b_16" else 32
            if self.data.image_size % patch_size:
                raise ValueError("For ViT, data.image_size must be a multiple of the patch size.")
        if self.data.image_size < 32:
            raise ValueError("data.image_size must be at least 32.")
        if self.training.epochs < 1 or self.data.batch_size < 1:
            raise ValueError("training.epochs and data.batch_size must be positive.")
        if self.evaluation.bootstrap_replicates < 0:
            raise ValueError("evaluation.bootstrap_replicates cannot be negative.")
        if not 0.0 < self.evaluation.confidence_level < 1.0:
            raise ValueError("evaluation.confidence_level must be in (0, 1).")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in update.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        raise ValueError(f"Circular config inheritance detected at {path}")
    seen.add(path)

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    parent = raw.pop("extends", None)
    if not parent:
        return raw
    parent_path = (path.parent / str(parent)).resolve()
    return _deep_merge(_read_yaml(parent_path, seen), raw)


def _parse_override_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    output = copy.deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected KEY=VALUE.")
        dotted_key, value = item.split("=", 1)
        cursor: dict[str, Any] = output
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"Cannot override through non-mapping key: {dotted_key}")
            cursor = child
        cursor[parts[-1]] = _parse_override_value(value)
    return output


# Built-in study configurations; no separate YAML files are required.
EXPERIMENT_PRESETS = {'fig2': {'data': {'manifest': 'data/fig2_standard.csv'},
          'evaluation': {'aggregate_by': ['participant_id', 'batch'],
                         'bootstrap_replicates': 0},
          'experiment': {'name': 'fig2_model_optimization',
                         'kind': 'fig2_model_optimization',
                         'variants': [{'name': 'single_image_vit_b16_60s',
                                       'overrides': {'data': {'frame_times': [60]},
                                                     'model': {'backbone': 'vit_b_16',
                                                               'temporal_mode': 'single',
                                                               'pruning_ratio': 0.0}}},
                                      {'name': 'vit_b16_mean_pool',
                                       'overrides': {'model': {'backbone': 'vit_b_16',
                                                               'temporal_mode': 'mean',
                                                               'pruning_ratio': 0.0}}},
                                      {'name': 'vit_b16_lstm',
                                       'overrides': {'model': {'backbone': 'vit_b_16',
                                                               'temporal_mode': 'lstm',
                                                               'pruning_ratio': 0.0}}},
                                      {'name': 'vit_b16_gru',
                                       'overrides': {'model': {'backbone': 'vit_b_16',
                                                               'temporal_mode': 'gru',
                                                               'pruning_ratio': 0.0}}},
                                      {'name': 'resnet18_lstm',
                                       'overrides': {'model': {'backbone': 'resnet18',
                                                               'temporal_mode': 'lstm',
                                                               'pruning_ratio': 0.0}}},
                                      {'name': 'dynamic_vit_lstm_prune_0.1',
                                       'overrides': {'model': {'pruning_ratio': 0.1}}},
                                      {'name': 'dynamic_vit_lstm_prune_0.2',
                                       'overrides': {'model': {'pruning_ratio': 0.2}}},
                                      {'name': 'dynamic_vit_lstm_prune_0.3',
                                       'overrides': {'model': {'pruning_ratio': 0.3}}},
                                      {'name': 'dynamic_vit_lstm_prune_0.4',
                                       'overrides': {'model': {'pruning_ratio': 0.4}}},
                                      {'name': 'dynamic_vit_lstm_prune_0.5',
                                       'overrides': {'model': {'pruning_ratio': 0.5}}},
                                      {'name': 'dynamic_vit_lstm_prune_0.6',
                                       'overrides': {'model': {'pruning_ratio': 0.6}}},
                                      {'name': 'dynamic_vit_lstm_prune_0.7',
                                       'overrides': {'model': {'pruning_ratio': 0.7}}},
                                      {'name': 'dynamic_vit_lstm_prune_0.8',
                                       'overrides': {'model': {'pruning_ratio': 0.8}}}]}},
 'fig3': {'data': {'manifest': 'data/fig3_korea.csv'},
          'model': {'backbone': 'vit_b_16', 'temporal_mode': 'lstm', 'pruning_ratio': 0.4},
          'evaluation': {'aggregate_by': ['participant_id', 'cohort', 'specimen'],
                         'subgroup_by': ['device', 'sex']},
          'experiment': {'name': 'fig3_korea_cohort', 'kind': 'fig3_korea_cohort'}},
 'fig4': {'data': {'manifest': 'data/fig4_senegal.csv', 'batch_size': 64},
          'model': {'backbone': 'vit_b_16', 'temporal_mode': 'lstm', 'pruning_ratio': 0.4},
          'training': {'epochs': 100, 'learning_rate': 3e-05, 'weight_decay': 0.01},
          'evaluation': {'aggregate_by': ['participant_id', 'cohort', 'specimen'],
                         'subgroup_by': ['specimen', 'scd_status']},
          'experiment': {'name': 'fig4_senegal_cohort', 'kind': 'fig4_senegal_cohort'}}}


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
    *,
    experiment: str = "fig3",
) -> AppConfig:
    """Load a built-in figure preset, or an optional complete external YAML config.

    Built-in manifest paths resolve relative to this source directory. With an
    external YAML file, paths resolve relative to that file, as in the full package.
    """
    if path is None:
        if experiment not in EXPERIMENT_PRESETS:
            raise ValueError(f"Unknown experiment: {experiment}")
        raw = copy.deepcopy(EXPERIMENT_PRESETS[experiment])
        base_dir = Path(__file__).resolve().parent
        provenance = f"builtin:{experiment}"
    else:
        config_path = Path(path).resolve()
        raw = _read_yaml(config_path)
        base_dir = config_path.parent
        provenance = str(config_path)
    cfg = AppConfig.from_dict(_apply_overrides(raw, overrides or []), config_path=provenance)
    manifest_path = Path(cfg.data.manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = base_dir / manifest_path
    cfg.data.manifest = str(manifest_path.resolve())
    return cfg
