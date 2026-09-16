"""Shared optimization pipeline for all figure-specific experiments."""

from __future__ import annotations

import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import AppConfig
from data import build_dataloaders
from test import benchmark_model, evaluate_model, load_checkpoint
from model import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(config: AppConfig) -> torch.device:
    requested = config.training.device.lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def _optimizer(model: nn.Module, config: AppConfig) -> torch.optim.Optimizer:
    backbone_parameters = list(_unwrap(model).backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in backbone_ids
    ]
    return torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": config.training.learning_rate
                * config.training.backbone_learning_rate_scale,
            },
            {"params": head_parameters, "lr": config.training.learning_rate},
        ],
        weight_decay=config.training.weight_decay,
    )


def _scheduler(
    optimizer: torch.optim.Optimizer, config: AppConfig
) -> LambdaLR | None:
    if config.training.scheduler == "none":
        return None
    if config.training.scheduler != "cosine":
        raise ValueError("training.scheduler must be 'none' or 'cosine'.")

    epochs = config.training.epochs
    warmup = max(0, config.training.warmup_epochs)
    minimum_ratio = config.training.minimum_learning_rate / config.training.learning_rate

    def multiplier(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return max(1.0e-8, (epoch + 1) / warmup)
        progress = (epoch - warmup) / max(1, epochs - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=multiplier)


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> float:
    training = optimizer is not None
    model.train(training)
    criterion = nn.MSELoss()
    total_loss = 0.0
    sample_count = 0

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in tqdm(loader, desc="train" if training else "val", leave=False):
            images = batch["images"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                predicted = model(images)
                loss = criterion(predicted, target)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss; check the input images, labels and settings.")
            if optimizer is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(loss.item()) * images.shape[0]
            sample_count += int(images.shape[0])
    return total_loss / max(1, sample_count)


def _run_directory(config: AppConfig, suffix: str = "") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = config.experiment.name + (f"_{suffix}" if suffix else "")
    path = Path(config.training.output_dir).resolve() / f"{timestamp}_{name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def train_model(config: AppConfig, run_suffix: str = "") -> dict[str, Any]:
    seed_everything(config.training.seed)
    loaders, _ = build_dataloaders(config)
    device = resolve_device(config)
    model = build_model(config).to(device)

    if (
        config.training.data_parallel
        and device.type == "cuda"
        and torch.cuda.device_count() > 1
    ):
        model = nn.DataParallel(model, device_ids=config.training.gpu_ids)

    optimizer = _optimizer(model, config)
    scheduler = _scheduler(optimizer, config)
    amp_enabled = config.training.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    run_dir = _run_directory(config, suffix=run_suffix)
    import platform
    from importlib.metadata import version
    environment = {
        "python": platform.python_version(), "platform": platform.platform(),
        "device": str(device), "cuda_runtime": torch.version.cuda,
        "packages": {name: version(name) for name in (
            "torch", "torchvision", "numpy", "pandas", "Pillow", "PyYAML",
            "scikit-learn", "matplotlib", "tqdm")},
    }
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (run_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )

    checkpoint_path = run_dir / "best.pt"
    history_path = run_dir / "history.csv"
    best_loss = float("inf")
    stale_epochs = 0
    rows: list[dict[str, float]] = []

    for epoch in range(1, config.training.epochs + 1):
        train_loss = _epoch(
            model, loaders["train"], device, optimizer, scaler, amp_enabled
        )
        val_loss = _epoch(
            model, loaders["val"], device, None, scaler, amp_enabled
        )
        if scheduler is not None:
            scheduler.step()
        rows.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "learning_rate": float(optimizer.param_groups[-1]["lr"]),
            }
        )

        if val_loss < best_loss:
            best_loss = val_loss
            stale_epochs = 0
            torch.save(
                {
                    "model_state": _unwrap(model).state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_loss,
                    "config": config.to_dict(),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= config.training.early_stopping_patience:
            break

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    unwrapped = _unwrap(model)
    load_checkpoint(unwrapped, checkpoint_path, device)
    metrics = evaluate_model(
        unwrapped,
        loaders["test"],
        config,
        output_dir=run_dir / "test",
    )
    benchmark = benchmark_model(unwrapped, loaders["test"])
    (run_dir / "benchmark.json").write_text(
        json.dumps(benchmark, indent=2), encoding="utf-8"
    )
    return {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "best_val_loss": float(best_loss),
        "test_metrics": metrics,
        "benchmark": benchmark,
    }


if __name__ == "__main__":
    import argparse
    from config import load_config
    parser = argparse.ArgumentParser(description="Train one HemoWick model.")
    parser.add_argument("--experiment", choices=("fig2", "fig3", "fig4"), default="fig3")
    parser.add_argument("--config", help="Optional external YAML config; replaces the built-in preset.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    result = train_model(load_config(args.config, args.set, experiment=args.experiment))
    print(f"Completed: {result['run_dir']}")
