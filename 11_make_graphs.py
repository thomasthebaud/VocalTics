"""Generate confusion matrices from cross-validation predictions."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
import numpy as np
import pandas as pd


GLOBAL_NAME = "TDNN_MFCC_bysession"
K_FOLDS = 5
OUTPUT_DIR = Path("outputs/detection") / GLOBAL_NAME
GRAPH_PATH = Path("graphs") / GLOBAL_NAME / "confusion_matrices.png"


def boolean_values(values):
    """Convert CSV boolean values to False and True values."""
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1"])


def load_predictions(split_name):
    """Load and concatenate one prediction split from every fold."""
    predictions = []
    for fold in range(1, K_FOLDS + 1):
        csv_path = OUTPUT_DIR / f"fold{fold}_{split_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {csv_path}")
        predictions.append(
            pd.read_csv(
                csv_path,
                dtype={"tic_group": str, "group_pred": str},
            )
        )

    combined = pd.concat(predictions, ignore_index=True)
    required = {"tic_real", "tic_pred", "tic_group", "group_pred"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    return combined


def tic_confusion_matrix(predictions):
    """Return a 2-by-2 tic detection confusion matrix."""
    real = boolean_values(predictions["tic_real"])
    predicted = boolean_values(predictions["tic_pred"])
    matrix = pd.crosstab(real, predicted).reindex(
        index=[False, True], columns=[False, True], fill_value=0
    )
    matrix.index = ["No tic", "Tic"]
    matrix.columns = ["No tic", "Tic"]
    return matrix


def group_confusion_matrix(predictions):
    """Return the matrix without -1 labels or '+' group combinations."""
    real_groups = predictions["tic_group"].fillna("-1").astype(str)
    predicted_groups = predictions["group_pred"].fillna("-1").astype(str)
    tic_rows = predictions.loc[
        (real_groups != "-1")
        & (predicted_groups != "-1")
        & ~real_groups.str.contains("+", regex=False)
        & ~predicted_groups.str.contains("+", regex=False)
    ].copy()
    if tic_rows.empty:
        raise ValueError("No tic-group predictions were found")

    real = tic_rows["tic_group"].astype(str)
    predicted = tic_rows["group_pred"].astype(str)
    labels = sorted(set(real) | set(predicted))
    return pd.crosstab(real, predicted).reindex(
        index=labels, columns=labels, fill_value=0
    )


def draw_confusion_matrix(
    axis, matrix, title, show_percentages=False, color_positive=False
):
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
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)

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


def main():
    test_predictions = load_predictions("test")
    validation_predictions = load_predictions("val")
    matrices = [
        (
            "Test",
            tic_confusion_matrix(test_predictions),
            group_confusion_matrix(test_predictions),
        ),
        (
            "Validation",
            tic_confusion_matrix(validation_predictions),
            group_confusion_matrix(validation_predictions),
        ),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(22, 18))
    for row, (split_name, tic_matrix, group_matrix) in enumerate(matrices):
        tic_image = draw_confusion_matrix(
            axes[row, 0],
            tic_matrix,
            f"{split_name} tic detection confusion matrix",
            show_percentages=True,
        )
        group_image = draw_confusion_matrix(
            axes[row, 1],
            group_matrix,
            f"{split_name} tic group confusion matrix",
            color_positive=True,
        )
        figure.colorbar(
            tic_image, ax=axes[row, 0], fraction=0.046, pad=0.04
        )
        figure.colorbar(
            group_image, ax=axes[row, 1], fraction=0.046, pad=0.04
        )
    figure.suptitle(f"{GLOBAL_NAME} confusion matrices", fontsize=16)
    figure.tight_layout()

    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(GRAPH_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved confusion matrices to {GRAPH_PATH}")


if __name__ == "__main__":
    main()
