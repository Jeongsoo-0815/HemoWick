"""Standardized participant-level diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def save_regression_plot(table: pd.DataFrame, output_path: str | Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, target, unit in zip(axes, ("hb", "hct"), ("g/dL", "%")):
        true = table[f"{target}_true"].to_numpy(dtype=float)
        predicted = table[f"{target}_pred"].to_numpy(dtype=float)
        lower = float(min(np.min(true), np.min(predicted)))
        upper = float(max(np.max(true), np.max(predicted)))
        correlation = (
            float(np.corrcoef(true, predicted)[0, 1])
            if len(true) > 1 and np.std(true) > 0 and np.std(predicted) > 0
            else float("nan")
        )
        axis.scatter(true, predicted, s=22, alpha=0.65, edgecolors="none")
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1)
        axis.set_xlabel(f"CBC {target.upper()} ({unit})")
        axis.set_ylabel(f"HemoWick {target.upper()} ({unit})")
        axis.set_title(f"{target.upper()}: R = {correlation:.3f}")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_roc_plot(
    table: pd.DataFrame,
    hb_thresholds: list[float],
    hct_thresholds: list[float],
    output_path: str | Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, target, thresholds in (
        (axes[0], "hb", hb_thresholds),
        (axes[1], "hct", hct_thresholds),
    ):
        true = table[f"{target}_true"].to_numpy(dtype=float)
        predicted = table[f"{target}_pred"].to_numpy(dtype=float)
        for threshold in thresholds:
            label = (true < threshold).astype(int)
            if np.unique(label).size < 2:
                continue
            false_positive, true_positive, _ = roc_curve(label, -predicted)
            auc = roc_auc_score(label, -predicted)
            axis.plot(
                false_positive,
                true_positive,
                label=f"{threshold:g}: AUC {auc:.3f}",
            )
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        axis.set_xlabel("1 - specificity")
        axis.set_ylabel("Sensitivity")
        axis.set_title(f"{target.upper()} anemia classification")
        if axis.lines and len(axis.lines) > 1:
            axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
