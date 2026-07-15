"""Generate confusion matrices from cross-validation predictions."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


GLOBAL_NAME = "TDNN_MFCC_bysession"
K_FOLDS = 5
SPLIT_NAME = "test"
OUTPUT_DIR = Path("outputs/detection") / GLOBAL_NAME
GRAPH_PATH = Path("graphs") / GLOBAL_NAME / "confusion_matrices.png"


def boolean_values(values):
    """Convert CSV boolean values to False and True values."""
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1"])


def load_predictions():
    """Load and concatenate prediction tables from every fold."""
    predictions = []
    for fold in range(1, K_FOLDS + 1):
        csv_path = OUTPUT_DIR / f"fold{fold}_{SPLIT_NAME}.csv"
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
    """Return the confusion matrix for samples containing a real tic."""
    tic_rows = predictions.loc[
        predictions["tic_group"].fillna("-1") != "-1"
    ].copy()
    if tic_rows.empty:
        raise ValueError("No tic-group predictions were found")

    real = tic_rows["tic_group"].astype(str)
    predicted = tic_rows["group_pred"].astype(str)
    labels = sorted(set(real) | set(predicted))
    return pd.crosstab(real, predicted).reindex(
        index=labels, columns=labels, fill_value=0
    )


def draw_confusion_matrix(axis, matrix, title):
    """Draw and annotate one count-based confusion matrix."""
    image = axis.imshow(matrix.to_numpy(), cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Real")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)

    threshold = matrix.to_numpy().max() / 2
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            value = matrix.iloc[row, column]
            color = "white" if value > threshold else "black"
            axis.text(column, row, str(value), ha="center", va="center", color=color)
    return image


def main():
    predictions = load_predictions()
    tic_matrix = tic_confusion_matrix(predictions)
    group_matrix = group_confusion_matrix(predictions)

    figure, axes = plt.subplots(1, 2, figsize=(22, 9))
    tic_image = draw_confusion_matrix(
        axes[0], tic_matrix, "Tic detection confusion matrix"
    )
    group_image = draw_confusion_matrix(
        axes[1], group_matrix, "Tic group confusion matrix"
    )
    figure.colorbar(tic_image, ax=axes[0], fraction=0.046, pad=0.04)
    figure.colorbar(group_image, ax=axes[1], fraction=0.046, pad=0.04)
    figure.suptitle(f"{GLOBAL_NAME} — {SPLIT_NAME} predictions", fontsize=16)
    figure.tight_layout()

    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(GRAPH_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved confusion matrices to {GRAPH_PATH}")


if __name__ == "__main__":
    main()
