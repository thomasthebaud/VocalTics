"""Aggregate validation and test metrics across cross-validation folds."""

from pathlib import Path

import pandas as pd
import torch

from bin.metrics import get_group_metrics, get_tic_metrics


GLOBAL_NAME = "TDNN_MFCC_bysession"
K_FOLDS = 5
OUTPUT_DIR = Path("outputs/detection") / GLOBAL_NAME

METRIC_NAMES = [
    "Tic accuracy",
    "Tic F1",
    "Tic AUROC",
    "Tic precision",
    "Tic recall",
    "Group accuracy",
    "Group macro F1",
]


def boolean_values(values):
    """Convert CSV boolean values to a tensor of zeros and ones."""
    normalized = values.astype(str).str.strip().str.lower()
    return torch.tensor(normalized.isin(["true", "1"]).astype(int).to_numpy())


def group_values(group_real, group_pred):
    """Convert '+'-separated group labels into shared multi-hot vectors."""
    real = group_real.fillna("-1").astype(str)
    predicted = group_pred.fillna("-1").astype(str)
    labels = sorted(
        {
            group
            for value in list(real) + list(predicted)
            for group in value.split("+")
            if group != "-1"
        }
    )
    group_to_index = {label: index for index, label in enumerate(labels)}

    def encode(values):
        targets = torch.zeros((len(values), len(labels)), dtype=torch.long)
        for row, value in enumerate(values):
            for group in value.split("+"):
                if group != "-1":
                    targets[row, group_to_index[group]] = 1
        return targets

    real_tensor = encode(real)
    predicted_tensor = encode(predicted)
    return predicted_tensor, real_tensor


def load_fold_metrics(csv_path):
    """Calculate all metrics from one fold prediction table."""
    predictions = pd.read_csv(
        csv_path,
        dtype={"tic_type": str, "tic_group": str, "group_pred": str},
    )
    required_columns = {
        "tic_real",
        "tic_probability",
        "tic_group",
        "group_pred",
    }
    missing = required_columns - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")

    tic_real = boolean_values(predictions["tic_real"])
    tic_scores = torch.tensor(
        predictions["tic_probability"].to_numpy(), dtype=torch.float32
    )
    tic_metrics = get_tic_metrics(tic_scores, tic_real)
    group_pred, group_real = group_values(
        predictions["tic_group"], predictions["group_pred"]
    )
    group_metrics = get_group_metrics(group_pred, group_real)
    return tic_metrics + group_metrics


def collect_metrics(split_name):
    """Load one prediction table for each fold of a split."""
    rows = []
    for fold in range(1, K_FOLDS + 1):
        csv_path = OUTPUT_DIR / f"fold{fold}_{split_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {csv_path}")
        rows.append(load_fold_metrics(csv_path))
    return pd.DataFrame(rows, columns=METRIC_NAMES)


def main():
    validation = collect_metrics("val")
    test = collect_metrics("test")
    table = pd.DataFrame(
        {
            "Validation mean": validation.mean(),
            "Validation std": validation.std(ddof=1),
            "Test mean": test.mean(),
            "Test std": test.std(ddof=1),
        }
    )

    print(f"\nMetrics across {K_FOLDS} folds: {GLOBAL_NAME}\n")
    print(table.to_string(float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
