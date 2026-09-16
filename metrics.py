"""Image-level and participant-level metrics with bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


TARGETS = {
    "hb": ("hb_true", "hb_pred"),
    "hct": ("hct_true", "hct_pred"),
}


def _pearson(true: np.ndarray, predicted: np.ndarray) -> float:
    if len(true) < 2 or np.std(true) == 0.0 or np.std(predicted) == 0.0:
        return float("nan")
    return float(np.corrcoef(true, predicted)[0, 1])


def regression_metrics(table: pd.DataFrame) -> dict[str, float]:
    output: dict[str, float] = {"n": int(len(table))}
    for target, (true_column, predicted_column) in TARGETS.items():
        true = table[true_column].to_numpy(dtype=float)
        predicted = table[predicted_column].to_numpy(dtype=float)
        residual = predicted - true
        mse = float(np.mean(residual**2))
        denominator = float(np.sum((true - np.mean(true)) ** 2))
        output[f"{target}_r"] = _pearson(true, predicted)
        output[f"{target}_r2"] = (
            float(1.0 - np.sum(residual**2) / denominator)
            if denominator > 0.0
            else float("nan")
        )
        output[f"{target}_mae"] = float(np.mean(np.abs(residual)))
        output[f"{target}_rmse"] = float(np.sqrt(mse))
        output[f"{target}_bias"] = float(np.mean(residual))
    return output


def auc_metrics(
    table: pd.DataFrame,
    hb_thresholds: list[float],
    hct_thresholds: list[float],
) -> dict[str, float]:
    output: dict[str, float] = {}
    for target, thresholds in (("hb", hb_thresholds), ("hct", hct_thresholds)):
        true = table[f"{target}_true"].to_numpy(dtype=float)
        predicted = table[f"{target}_pred"].to_numpy(dtype=float)
        for threshold in thresholds:
            label = (true < threshold).astype(int)
            key = f"{target}_auc_at_{threshold:g}"
            if np.unique(label).size < 2:
                output[key] = float("nan")
            else:
                output[key] = float(roc_auc_score(label, -predicted))
    return output


def all_metrics(
    table: pd.DataFrame,
    hb_thresholds: list[float],
    hct_thresholds: list[float],
) -> dict[str, float]:
    return {
        **regression_metrics(table),
        **auc_metrics(table, hb_thresholds, hct_thresholds),
    }


def aggregate_predictions(table: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    available = [column for column in group_columns if column in table.columns]
    if "participant_id" not in available:
        available.insert(0, "participant_id")
    if not available:
        raise ValueError("Participant aggregation requires a participant_id column.")

    value_columns = ["hb_true", "hct_true", "hb_pred", "hct_pred"]
    aggregation: dict[str, str] = {column: "mean" for column in value_columns}
    for column in table.columns:
        if column not in available and column not in value_columns:
            aggregation[column] = "first"
    grouped = table.groupby(available, dropna=False, as_index=False).agg(aggregation)
    counts = (
        table.groupby(available, dropna=False)
        .size()
        .rename("image_dataset_count")
        .reset_index()
    )
    return grouped.merge(counts, on=available, how="left")


def _cluster_sample(
    table: pd.DataFrame,
    rng: np.random.Generator,
    cluster_column: str,
) -> pd.DataFrame:
    clusters = table[cluster_column].drop_duplicates().to_numpy()
    selected = rng.choice(clusters, size=len(clusters), replace=True)
    pieces: list[pd.DataFrame] = []
    for bootstrap_index, cluster in enumerate(selected):
        piece = table.loc[table[cluster_column] == cluster].copy()
        piece["_bootstrap_cluster"] = bootstrap_index
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def bootstrap_confidence_intervals(
    table: pd.DataFrame,
    metric_function: Callable[[pd.DataFrame], dict[str, float]],
    replicates: int,
    confidence_level: float,
    seed: int,
    cluster_column: str = "participant_id",
) -> dict[str, dict[str, float]]:
    point = metric_function(table)
    if replicates <= 0:
        return {
            key: {"estimate": float(value), "lower": float("nan"), "upper": float("nan")}
            for key, value in point.items()
        }
    if cluster_column not in table.columns:
        raise ValueError(f"Bootstrap cluster column not found: {cluster_column}")

    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {key: [] for key in point if key != "n"}
    for _ in range(replicates):
        sample = _cluster_sample(table, rng, cluster_column)
        values = metric_function(sample)
        for key in draws:
            value = float(values[key])
            if np.isfinite(value):
                draws[key].append(value)

    alpha = (1.0 - confidence_level) / 2.0
    output: dict[str, dict[str, float]] = {}
    for key, estimate in point.items():
        if key == "n":
            output[key] = {
                "estimate": float(estimate),
                "lower": float(estimate),
                "upper": float(estimate),
            }
            continue
        values = np.asarray(draws[key], dtype=float)
        if values.size == 0:
            lower = upper = float("nan")
        else:
            lower, upper = np.quantile(values, [alpha, 1.0 - alpha]).tolist()
        output[key] = {
            "estimate": float(estimate),
            "lower": float(lower),
            "upper": float(upper),
        }
    return output


def subgroup_metrics(
    table: pd.DataFrame,
    columns: list[str],
    metric_function: Callable[[pd.DataFrame], dict[str, float]],
) -> list[dict[str, Any]]:
    available = [column for column in columns if column in table.columns]
    if not available:
        return []
    rows: list[dict[str, Any]] = []
    for group_key, subset in table.groupby(available, dropna=False):
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = dict(zip(available, values))
        row.update(metric_function(subset))
        rows.append(row)
    return rows
