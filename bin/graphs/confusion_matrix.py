"""Create tic-detection and tic-group confusion matrices."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
import numpy as np
import pandas as pd


def _boolean_values(values):
    """Convert CSV boolean values to False and True values."""
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1"])


def _load_predictions(input_path, split_name, task):
    """Load every available fold prediction table for one split."""
    csv_paths = sorted(Path(input_path).glob(f"fold*_{split_name}.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No fold*_{split_name}.csv files found in {input_path}"
        )

    if task == "detection":
        predictions = [
            pd.read_csv(path, dtype={"tic_group": str, "group_pred": str})
            for path in csv_paths
        ]
        required = {"tic_real", "tic_pred", "tic_group", "group_pred"}
    elif task == "segmentation":
        predictions = []
        for path in csv_paths:
            prediction = pd.read_csv(path)
            prediction["_fold"] = path.stem.split("_")[0]
            predictions.append(prediction)
        required = {
            "segment_id",
            "tic_real",
            "tic_pred",
            "segment_real",
            "segment_pred",
        }
    else:
        raise ValueError(f"Unknown task: {task}")
    combined = pd.concat(predictions, ignore_index=True)
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    return combined


def _tic_confusion_matrix(predictions):
    """Return a 2-by-2 tic detection confusion matrix."""
    real = _boolean_values(predictions["tic_real"])
    predicted = _boolean_values(predictions["tic_pred"])
    matrix = pd.crosstab(real, predicted).reindex(
        index=[False, True], columns=[False, True], fill_value=0
    )
    matrix.index = ["No tic", "Tic"]
    matrix.columns = ["No tic", "Tic"]
    return matrix


def _segment_confusion_matrix(predictions):
    """Return a 2-by-2 matrix with one entry per sampled segment."""
    segment_keys = ["_fold", "segment_id"]
    segments = predictions[
        segment_keys + ["segment_real", "segment_pred"]
    ].drop_duplicates()
    consistency = segments.groupby(segment_keys)[
        ["segment_real", "segment_pred"]
    ].nunique()
    if (consistency > 1).any().any():
        raise ValueError("A segment has inconsistent segment-level labels")
    segments = segments.drop_duplicates(segment_keys)
    real = _boolean_values(segments["segment_real"])
    predicted = _boolean_values(segments["segment_pred"])
    matrix = pd.crosstab(real, predicted).reindex(
        index=[False, True], columns=[False, True], fill_value=0
    )
    matrix.index = ["No tic", "Tic"]
    matrix.columns = ["No tic", "Tic"]
    return matrix


def _group_confusion_matrix(predictions):
    """Return the matrix without -1 labels or '+' group combinations."""
    real_groups = predictions["tic_group"].fillna("-1").astype(str)
    predicted_groups = predictions["group_pred"].fillna("-1").astype(str)
    rows = predictions.loc[
        (real_groups != "-1")
        & (predicted_groups != "-1")
        & ~real_groups.str.contains("+", regex=False)
        & ~predicted_groups.str.contains("+", regex=False)
    ].copy()
    if rows.empty:
        raise ValueError("No single-group predictions were found")

    real = rows["tic_group"].astype(str)
    predicted = rows["group_pred"].astype(str)
    labels = sorted(set(real) | set(predicted))
    return pd.crosstab(real, predicted).reindex(
        index=labels, columns=labels, fill_value=0
    )


def _draw_matrix(axis, matrix, title, show_percentages=False, color_positive=False):
    """Draw and annotate one confusion matrix."""
    values = matrix.to_numpy()
    if color_positive:
        colors = plt.cm.Blues(np.linspace(0.25, 1, 256))
        colormap = ListedColormap(colors)
        displayed_values = np.ma.masked_equal(values, 0)
        image = axis.imshow(
            displayed_values,
            cmap=colormap,
            norm=LogNorm(vmin=1, vmax=max(1, values.max())),
        )
        threshold = values.max() ** 0.5
    else:
        image = axis.imshow(values, cmap="Blues")
        threshold = values.max() / 2

    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Real")
    axis.set_xticks(range(len(matrix.columns)))
    axis.set_xticklabels(matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(matrix.index)))
    axis.set_yticklabels(matrix.index)

    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            value = matrix.iloc[row, column]
            color = "white" if value > threshold else "black"
            label = str(value)
            if show_percentages:
                row_total = matrix.iloc[row].sum()
                percentage = 100 * value / row_total if row_total else 0
                label = f"{value}\n({percentage:.1f}%)"
            axis.text(column, row, label, ha="center", va="center", color=color)
    return image


def make_confusion_matrix(input_path, output_path, task="detection"):
    """Load fold predictions, create task-specific matrices, and save a PNG."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    test_predictions = _load_predictions(input_path, "test", task)
    validation_predictions = _load_predictions(input_path, "val", task)

    if task == "detection":
        matrices = [
            (
                "Test",
                _tic_confusion_matrix(test_predictions),
                _group_confusion_matrix(test_predictions),
            ),
            (
                "Validation",
                _tic_confusion_matrix(validation_predictions),
                _group_confusion_matrix(validation_predictions),
            ),
        ]
        column_names = ("tic detection", "tic group")
    elif task == "segmentation":
        matrices = [
            (
                "Test",
                _tic_confusion_matrix(test_predictions),
                _segment_confusion_matrix(test_predictions),
            ),
            (
                "Validation",
                _tic_confusion_matrix(validation_predictions),
                _segment_confusion_matrix(validation_predictions),
            ),
        ]
        column_names = ("frame", "segment")
    else:
        raise ValueError(f"Unknown task: {task}")

    figure, axes = plt.subplots(2, 2, figsize=(22, 18))
    for row, (split_name, first_matrix, second_matrix) in enumerate(matrices):
        first_image = _draw_matrix(
            axes[row, 0],
            first_matrix,
            f"{split_name} {column_names[0]} confusion matrix",
            show_percentages=True,
        )
        second_image = _draw_matrix(
            axes[row, 1],
            second_matrix,
            f"{split_name} {column_names[1]} confusion matrix",
            show_percentages=task == "segmentation",
            color_positive=task == "detection",
        )
        figure.colorbar(first_image, ax=axes[row, 0], fraction=0.046, pad=0.04)
        figure.colorbar(second_image, ax=axes[row, 1], fraction=0.046, pad=0.04)

    figure.suptitle(
        f"{task.title()}: {input_path.name} confusion matrices", fontsize=16
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved confusion matrices to {output_path}")
