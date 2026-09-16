"""Manifest validation, sequence preprocessing and data loaders."""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import random
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter
from typing import Any
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np
from config import AppConfig, DataConfig


REQUIRED_COLUMNS = {"sample_id", "split", "hb", "hct"}


def load_manifest(config: DataConfig) -> pd.DataFrame:
    path = Path(config.manifest).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}. See README.md for the required schema."
        )

    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    renamed = {
        config.sample_id_column: "sample_id",
        config.participant_id_column: "participant_id",
        config.split_column: "split",
        config.frame_dir_column: "frame_dir",
        config.frames_column: "frames",
    }
    table = table.rename(columns={k: v for k, v in renamed.items() if k in table.columns})

    missing = REQUIRED_COLUMNS - set(table.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if "frame_dir" not in table.columns and "frames" not in table.columns:
        raise ValueError("Manifest requires either a 'frame_dir' or 'frames' column.")

    if table.empty:
        raise ValueError("Manifest must contain at least one sample.")
    if config.validate_participant_splits and "participant_id" not in table.columns:
        raise ValueError("participant_id is required for participant-level split validation.")
    for column in ("sample_id", "split", "participant_id"):
        if column in table.columns and table[column].str.strip().eq("").any():
            raise ValueError(f"Manifest contains empty {column} values.")
    allowed_splits = {config.train_split, config.val_split, config.test_split}
    if len(allowed_splits) != 3:
        raise ValueError("Train, validation and test split names must be distinct.")
    unknown = set(table["split"]) - allowed_splits
    if unknown:
        raise ValueError(f"Unknown split values: {sorted(unknown)}")
    table["sample_id"] = table["sample_id"].astype(str)
    if "participant_id" not in table.columns:
        table["participant_id"] = table["sample_id"]
    table["participant_id"] = table["participant_id"].fillna(table["sample_id"]).astype(str)
    table["hb"] = pd.to_numeric(table["hb"], errors="raise")
    table["hct"] = pd.to_numeric(table["hct"], errors="raise")
    if not np.isfinite(table[["hb", "hct"]].to_numpy(dtype=float)).all():
        raise ValueError("Hb and Hct labels must be finite numeric values.")
    table["_manifest_dir"] = str(path.parent)

    if table["sample_id"].duplicated().any():
        duplicate_ids = table.loc[table["sample_id"].duplicated(), "sample_id"].head(5).tolist()
        raise ValueError(f"sample_id values must be unique; examples: {duplicate_ids}")

    if config.validate_participant_splits:
        validate_participant_splits(table)
    return table.reset_index(drop=True)


def validate_participant_splits(table: pd.DataFrame) -> None:
    """Reject leakage when a clinical participant appears in multiple splits."""
    split_counts = table.groupby("participant_id", dropna=False)["split"].nunique()
    leaked = split_counts[split_counts > 1]
    if not leaked.empty:
        examples = leaked.index.astype(str).tolist()[:10]
        raise ValueError(
            "Participant-level split leakage detected. Each participant must belong to one split; "
            f"examples: {examples}"
        )


def subset_for_split(table: pd.DataFrame, split: str) -> pd.DataFrame:
    subset = table.loc[table["split"].astype(str) == str(split)].copy()
    if subset.empty:
        raise ValueError(f"No rows found for split={split!r}.")
    return subset.reset_index(drop=True)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SequenceTransform:
    """Apply one sampled augmentation consistently to every frame in a sequence."""

    def __init__(self, config: DataConfig, training: bool):
        self.config = config
        self.training = training and config.augmentation

    def _sample(self) -> dict[str, float | bool]:
        if not self.training:
            return {
                "brightness": 1.0,
                "contrast": 1.0,
                "angle": 0.0,
                "blur": False,
                "blur_sigma": 0.0,
            }
        return {
            "brightness": random.uniform(
                1.0 - self.config.brightness, 1.0 + self.config.brightness
            ),
            "contrast": random.uniform(
                1.0 - self.config.contrast, 1.0 + self.config.contrast
            ),
            "angle": random.uniform(
                -self.config.rotation_degrees, self.config.rotation_degrees
            ),
            "blur": random.random() < self.config.blur_probability,
            "blur_sigma": random.uniform(0.1, max(0.1, self.config.blur_sigma_max)),
        }

    def __call__(self, frames: list[Image.Image]) -> torch.Tensor:
        params = self._sample()
        output: list[torch.Tensor] = []
        for image in frames:
            image = F.resize(image, [self.config.image_size, self.config.image_size])
            if self.training:
                image = F.adjust_brightness(image, float(params["brightness"]))
                image = F.adjust_contrast(image, float(params["contrast"]))
                image = F.rotate(image, float(params["angle"]))
                if bool(params["blur"]):
                    image = image.filter(
                        ImageFilter.GaussianBlur(radius=float(params["blur_sigma"]))
                    )
            tensor = F.to_tensor(image)
            tensor = F.normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
            output.append(tensor)
        return torch.stack(output, dim=0)


def _resolve_path(value: str, manifest_dir: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(manifest_dir) / path
    return path.resolve()


class BloodSpotDataset(Dataset):
    def __init__(self, table: pd.DataFrame, config: AppConfig, training: bool):
        self.table = table.reset_index(drop=True)
        self.config = config
        self.transform = SequenceTransform(config.data, training=training)
        self.metadata_columns = [
            column
            for column in (
                "sample_id",
                "participant_id",
                "cohort",
                "site",
                "specimen",
                "device",
                "sex",
                "scd_status",
                "batch",
                "split",
            )
            if column in self.table.columns
        ]

    def __len__(self) -> int:
        return len(self.table)

    def _frame_paths(self, row: pd.Series) -> list[Path]:
        manifest_dir = str(row["_manifest_dir"])
        explicit = row.get("frames")
        if pd.notna(explicit) and str(explicit).strip():
            paths = [
                _resolve_path(item.strip(), manifest_dir)
                for item in str(explicit).split(";")
                if item.strip()
            ]
            if len(paths) != len(self.config.data.frame_times):
                raise ValueError(
                    f"{row['sample_id']}: explicit frames count ({len(paths)}) does not match "
                    f"data.frame_times ({len(self.config.data.frame_times)})."
                )
            return paths

        frame_dir = row.get("frame_dir")
        if pd.isna(frame_dir) or not str(frame_dir).strip():
            raise ValueError(f"{row['sample_id']}: neither frames nor frame_dir is populated.")
        root = _resolve_path(str(frame_dir), manifest_dir)
        return [
            root / self.config.data.frame_pattern.format(time=time_point)
            for time_point in self.config.data.frame_times
        ]

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.table.iloc[index]
        paths = self._frame_paths(row)
        if self.config.data.strict_frames:
            missing = [str(path) for path in paths if not path.exists()]
            if missing:
                raise FileNotFoundError(
                    f"{row['sample_id']}: missing {len(missing)} frame(s); first={missing[0]}"
                )

        frames: list[Image.Image] = []
        for path in paths:
            with Image.open(path) as image:
                frames.append(image.convert("RGB"))

        metadata = {column: str(row[column]) for column in self.metadata_columns}
        metadata["frame_paths"] = ";".join(str(path) for path in paths)
        return {
            "images": self.transform(frames),
            "target": torch.tensor([float(row["hb"]), float(row["hct"])], dtype=torch.float32),
            "metadata": metadata,
        }


def _loader(dataset: Dataset, config: AppConfig, shuffle: bool) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.data.batch_size,
        "shuffle": shuffle,
        "num_workers": config.data.num_workers,
        "pin_memory": config.data.pin_memory,
    }
    if config.data.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def build_dataloaders(
    config: AppConfig,
) -> tuple[dict[str, DataLoader], dict[str, BloodSpotDataset]]:
    table = load_manifest(config.data)
    split_names = {
        "train": config.data.train_split,
        "val": config.data.val_split,
        "test": config.data.test_split,
    }
    datasets = {
        name: BloodSpotDataset(
            subset_for_split(table, split_value),
            config=config,
            training=name == "train",
        )
        for name, split_value in split_names.items()
    }
    loaders = {
        name: _loader(dataset, config, shuffle=name == "train")
        for name, dataset in datasets.items()
    }
    return loaders, datasets


def build_test_loader(config: AppConfig) -> DataLoader:
    """Evaluate a frozen checkpoint using a manifest containing only test rows."""
    table = load_manifest(config.data)
    dataset = BloodSpotDataset(subset_for_split(table, config.data.test_split), config, False)
    return _loader(dataset, config, shuffle=False)
