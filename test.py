"""Model inference and research-reporting outputs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import AppConfig
from data import build_test_loader
from metrics import (
    aggregate_predictions,
    all_metrics,
    bootstrap_confidence_intervals,
    subgroup_metrics,
)
from plots import save_regression_plot, save_roc_plot
from model import build_model


def _device(config: AppConfig) -> torch.device:
    requested = config.training.device.lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _metadata_frame(metadata: dict[str, Any]) -> pd.DataFrame:
    normalized: dict[str, list[Any]] = {}
    for key, value in metadata.items():
        if isinstance(value, (list, tuple)):
            normalized[key] = list(value)
        elif isinstance(value, torch.Tensor):
            normalized[key] = value.detach().cpu().tolist()
        else:
            normalized[key] = list(value)
    return pd.DataFrame(normalized)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str = "evaluation",
) -> tuple[pd.DataFrame, float]:
    model.eval()
    criterion = nn.MSELoss()
    frames: list[pd.DataFrame] = []
    weighted_loss = 0.0
    count = 0

    for batch in tqdm(loader, desc=description, leave=False):
        images = batch["images"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        predicted = model(images)
        loss = criterion(predicted, target)
        weighted_loss += float(loss.item()) * images.shape[0]
        count += int(images.shape[0])

        metadata = _metadata_frame(batch["metadata"])
        target_array = target.detach().cpu().numpy()
        predicted_array = predicted.detach().cpu().numpy()
        metadata["hb_true"] = target_array[:, 0]
        metadata["hct_true"] = target_array[:, 1]
        metadata["hb_pred"] = predicted_array[:, 0]
        metadata["hct_pred"] = predicted_array[:, 1]
        frames.append(metadata)

    if not frames:
        raise ValueError("Evaluation loader produced no batches.")
    return pd.concat(frames, ignore_index=True), weighted_loss / max(1, count)


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_clean_json(value), indent=2), encoding="utf-8")


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    config: AppConfig,
    output_dir: str | Path,
    split_name: str = "test",
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    image_predictions, loss = predict(model, loader, device, description=split_name)

    metric_function = lambda table: all_metrics(  # noqa: E731
        table,
        hb_thresholds=config.evaluation.hb_thresholds,
        hct_thresholds=config.evaluation.hct_thresholds,
    )
    image_metrics = metric_function(image_predictions)
    image_metrics["loss"] = float(loss)

    participant_predictions = aggregate_predictions(
        image_predictions, config.evaluation.aggregate_by
    )
    participant_metrics = metric_function(participant_predictions)
    confidence_intervals = bootstrap_confidence_intervals(
        participant_predictions,
        metric_function=metric_function,
        replicates=config.evaluation.bootstrap_replicates,
        confidence_level=config.evaluation.confidence_level,
        seed=config.evaluation.bootstrap_seed,
        cluster_column="participant_id",
    )
    subgroup_rows: list[dict[str, Any]] = []
    if config.evaluation.subgroup_by:
        subgroup_group_columns = list(
            dict.fromkeys(
                [
                    "participant_id",
                    *[
                        column
                        for column in config.evaluation.aggregate_by
                        if column != "participant_id"
                    ],
                    *config.evaluation.subgroup_by,
                ]
            )
        )
        subgroup_predictions = aggregate_predictions(
            image_predictions, subgroup_group_columns
        )
        subgroup_rows = subgroup_metrics(
            subgroup_predictions,
            columns=config.evaluation.subgroup_by,
            metric_function=metric_function,
        )

    if config.evaluation.save_predictions:
        image_predictions.to_csv(output / f"{split_name}_predictions_image.csv", index=False)
        participant_predictions.to_csv(
            output / f"{split_name}_predictions_participant.csv", index=False
        )
    save_regression_plot(
        participant_predictions,
        output / f"{split_name}_participant_regression.png",
    )
    save_roc_plot(
        participant_predictions,
        hb_thresholds=config.evaluation.hb_thresholds,
        hct_thresholds=config.evaluation.hct_thresholds,
        output_path=output / f"{split_name}_participant_roc.png",
    )
    if subgroup_rows:
        pd.DataFrame(subgroup_rows).to_csv(
            output / f"{split_name}_subgroup_metrics.csv", index=False
        )

    summary = {
        "image_level": image_metrics,
        "participant_level": participant_metrics,
        "participant_level_confidence_intervals": confidence_intervals,
        "bootstrap": {
            "unit": "participant_id",
            "replicates": config.evaluation.bootstrap_replicates,
            "confidence_level": config.evaluation.confidence_level,
            "seed": config.evaluation.bootstrap_seed,
        },
        "subgroups": subgroup_rows,
    }
    _write_json(output / f"{split_name}_metrics.json", summary)
    return summary


def load_checkpoint(model: nn.Module, path: str | Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)


def evaluate_checkpoint(
    config: AppConfig,
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    from copy import deepcopy
    config = deepcopy(config)
    # Checkpoints contain the complete backbone; avoid an unnecessary weight download.
    config.model.pretrained = False
    loader = build_test_loader(config)
    device = _device(config)
    model = build_model(config).to(device)
    load_checkpoint(model, checkpoint_path, device)
    return evaluate_model(model, loader, config, output_dir)


@torch.no_grad()
def benchmark_model(
    model: nn.Module,
    loader: DataLoader,
    warmup_batches: int = 5,
    measured_batches: int = 20,
) -> dict[str, float]:
    device = next(model.parameters()).device
    model.eval()
    iterator = iter(loader)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timings: list[float] = []
    for index in range(warmup_batches + measured_batches):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        images = batch["images"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        model(images)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if index >= warmup_batches:
            timings.append(elapsed)

    peak_memory = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else float("nan")
    )
    return {
        "mean_batch_latency_seconds": float(np.mean(timings)),
        "peak_memory_mb": float(peak_memory),
        "measured_batches": float(len(timings)),
    }


if __name__ == "__main__":
    import argparse
    from config import load_config
    parser = argparse.ArgumentParser(description="Evaluate a HemoWick checkpoint.")
    parser.add_argument("--experiment", choices=("fig2", "fig3", "fig4"), default="fig4")
    parser.add_argument("--config", help="Optional external YAML config; replaces the built-in preset.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="runs/evaluation")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    result = evaluate_checkpoint(load_config(args.config, args.set, experiment=args.experiment), args.checkpoint, args.output)
    print(json.dumps(_clean_json(result), indent=2, allow_nan=False))
