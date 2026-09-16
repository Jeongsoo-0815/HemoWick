"""HemoWick command-line entry point and experiment comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config import AppConfig, load_config, _deep_merge


def variant_config(base: AppConfig, variant: dict[str, Any]) -> AppConfig:
    raw = base.to_dict()
    raw.pop("config_path", None)
    raw["experiment"]["variants"] = []
    raw["experiment"]["name"] = f"{base.experiment.name}_{variant['name']}"
    return AppConfig.from_dict(_deep_merge(raw, variant.get("overrides", {})), config_path=base.config_path)


def run_fig2(config: AppConfig) -> dict[str, Any]:
    import pandas as pd
    from train import train_model

    if not config.experiment.variants:
        raise ValueError("Fig. 2 requires at least one experiment.variants entry.")

    results: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    for variant in config.experiment.variants:
        name = str(variant["name"])
        variant_cfg = variant_config(config, variant)
        results[name] = train_model(variant_cfg, run_suffix=name)
        participant = results[name]["test_metrics"]["participant_level"]
        benchmark = results[name]["benchmark"]
        comparison_rows.append(
            {
                "variant": name,
                "backbone": variant_cfg.model.backbone,
                "temporal_mode": variant_cfg.model.temporal_mode,
                "pruning_ratio": variant_cfg.model.pruning_ratio,
                "hb_r": participant.get("hb_r"),
                "hb_r2": participant.get("hb_r2"),
                "hb_mae": participant.get("hb_mae"),
                "hb_rmse": participant.get("hb_rmse"),
                "hct_r": participant.get("hct_r"),
                "hct_r2": participant.get("hct_r2"),
                "hct_mae": participant.get("hct_mae"),
                "hct_rmse": participant.get("hct_rmse"),
                "mean_batch_latency_seconds": benchmark.get(
                    "mean_batch_latency_seconds"
                ),
                "peak_memory_mb": benchmark.get("peak_memory_mb"),
            }
        )

    summary_path = Path(config.training.output_dir).resolve() / (
        f"{config.experiment.name}_latest_summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    comparison_path = summary_path.with_suffix(".csv")
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    return {
        "summary": str(summary_path),
        "comparison_csv": str(comparison_path),
        "variants": results,
    }


def create_demo(output: Path) -> AppConfig:
    import numpy as np
    import pandas as pd
    import yaml
    from PIL import Image, ImageDraw

    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    rows = []
    for split, count in (("train", 4), ("val", 2), ("test", 4)):
        for index in range(count):
            sample_id = f"synthetic_{split}_{index}"
            directory = output / "images" / sample_id
            directory.mkdir(parents=True)
            for second in (0, 10):
                pixels = rng.integers(200, 240, (64, 64, 3), dtype=np.uint8)
                image = Image.fromarray(pixels)
                draw = ImageDraw.Draw(image)
                draw.ellipse((14-second//5, 14, 50, 50), fill=(100+index*10, 30, 40))
                image.save(directory / f"{second}s.jpg")
            rows.append({
                "sample_id": sample_id, "participant_id": sample_id,
                "split": split, "hb": 7.0 + index * 3,
                "hct": 23.0 + index * 8, "frame_dir": f"images/{sample_id}",
                "cohort": "synthetic", "specimen": "synthetic",
            })
    manifest = output / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    cfg = AppConfig()
    cfg.data.manifest = str(manifest.resolve())
    cfg.data.frame_times = [0, 10]
    cfg.data.image_size = 64
    cfg.data.batch_size = 2
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.data.augmentation = False
    cfg.model.backbone = "resnet18"
    cfg.model.pretrained = False
    cfg.model.pruning_ratio = 0.0
    cfg.model.rnn_hidden_size = 16
    cfg.training.epochs = 1
    cfg.training.device = "cpu"
    cfg.training.use_amp = False
    cfg.training.warmup_epochs = 0
    cfg.training.output_dir = str((output / "runs").resolve())
    cfg.evaluation.bootstrap_replicates = 20
    cfg.experiment.name = "hemowick_synthetic_demo"
    cfg.validate()
    raw = cfg.to_dict()
    raw.pop("config_path", None)
    raw["data"]["manifest"] = "manifest.csv"
    (output / "demo.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return cfg



def run_demo(output: str) -> None:
    import time
    import torch
    from train import train_model

    torch.set_num_threads(2)
    started = time.perf_counter()
    cfg = create_demo(Path(output).resolve())
    result = train_model(cfg)
    report = {
        "data": "Synthetic images and arbitrary labels; not clinical validation.",
        "model": "Small ResNet18-LSTM demonstration; not the study ViT configuration.",
        "elapsed_seconds": time.perf_counter() - started,
        "run_dir": result["run_dir"], "checkpoint": result["checkpoint"],
    }
    (Path(output) / "demo_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))




def main() -> int:
    parser = argparse.ArgumentParser(description="HemoWick training and evaluation.")
    parser.add_argument("command", choices=("demo", "validate-config", "check-data", "run", "evaluate"))
    parser.add_argument("--experiment", choices=("fig2", "fig3", "fig4"), default="fig3")
    parser.add_argument("--config", help="Optional external YAML config; replaces the built-in preset.")
    parser.add_argument("--checkpoint", help="Required for evaluate.")
    parser.add_argument("--output", help="Output directory for demo or evaluation.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    if args.command == "demo":
        run_demo(args.output or "demo_output")
        return 0
    cfg = load_config(args.config, args.set, experiment=args.experiment)
    if args.command == "validate-config":
        print(json.dumps(cfg.to_dict(), indent=2))
        return 0
    if args.command == "check-data":
        from data import load_manifest, BloodSpotDataset
        table = load_manifest(cfg.data)
        # Decode every sample so missing or corrupt images are reported before training.
        dataset = BloodSpotDataset(table, cfg, training=False)
        for index in range(len(dataset)):
            dataset[index]
        print(f"Validated {len(table)} sequences: {table['split'].value_counts().to_dict()}")
        return 0
    if args.command == "evaluate":
        if not args.checkpoint:
            parser.error("evaluate requires --checkpoint")
        from test import evaluate_checkpoint
        output = args.output or "runs/evaluation"
        evaluate_checkpoint(cfg, args.checkpoint, output)
        print(f"Evaluation saved to {Path(output).resolve()}")
        return 0
    from train import train_model
    if cfg.experiment.kind == "fig2_model_optimization":
        result = run_fig2(cfg)
        print(f"Comparison saved to {result['comparison_csv']}")
    elif cfg.experiment.kind in {"cohort_validation", "fig3_korea_cohort", "fig4_senegal_cohort"}:
        result = train_model(cfg)
        print(f"Training completed: {result['run_dir']}")
    else:
        raise ValueError(f"Unsupported experiment.kind: {cfg.experiment.kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
