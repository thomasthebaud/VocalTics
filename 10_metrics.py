"""Aggregate validation and test metrics across cross-validation folds."""

from pathlib import Path

import pandas as pd
import torch

from bin.make_splits import load_split
from bin.detection_metrics import get_group_metrics, get_tic_metrics


K_FOLDS = 5
OUTPUT_ROOT = Path("outputs/detection")
METADATA_PATH = Path("/projects/vocaltics/data/metadata.csv")
SPLIT_PATH = Path("splits.json")

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


def load_predictions(csv_path):
    """Load and validate one fold prediction table."""
    predictions = pd.read_csv(
        csv_path, dtype={"tic_type": str, "tic_group": str, "group_pred": str}
    )
    required_columns = {
        "tic_real",
        "tic_probability",
        "tic_type",
        "tic_group",
        "group_pred",
    }
    missing = required_columns - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
    return predictions


def load_fold_metrics(predictions):
    """Calculate all metrics from one fold prediction table."""

    tic_real = boolean_values(predictions["tic_real"])
    tic_scores = torch.tensor(
        predictions["tic_probability"].to_numpy(), dtype=torch.float32
    )
    tic_metrics = get_tic_metrics(tic_scores, tic_real)
    real_groups = predictions["tic_group"].fillna("-1").astype(str)
    predicted_groups = predictions["group_pred"].fillna("-1").astype(str)
    single_group = ~real_groups.str.contains(
        "+", regex=False
    ) & ~predicted_groups.str.contains("+", regex=False)
    group_pred, group_real = group_values(
        real_groups[single_group], predicted_groups[single_group]
    )
    if group_real.any():
        group_metrics = get_group_metrics(group_pred, group_real)
    else:
        group_metrics = (float("nan"), float("nan"))
    return tic_metrics + group_metrics


def training_tic_types(metadata, recordings):
    """Return the individual TicIDs available in one fold's training split."""
    recording_set = set(recordings)
    selected = metadata.apply(
        lambda row: (row["ID"], row["Phase"], row["Sess"]) in recording_set,
        axis=1,
    )
    tic_rows = metadata.loc[selected & (metadata["tic/nontic"] == "tic")]
    return {
        tic_type
        for value in tic_rows["Type"].dropna()
        for tic_type in str(value).split("+")
        if tic_type != "-1"
    }


def filter_by_training_presence(predictions, train_types, presence):
    """Keep non-tics and either seen or unseen real tic types."""
    tic_real = predictions["tic_real"].astype(str).str.lower().isin(["true", "1"])

    def was_seen(value):
        tic_types = [item for item in str(value).split("+") if item != "-1"]
        return bool(tic_types) and all(item in train_types for item in tic_types)

    seen = predictions["tic_type"].fillna("-1").apply(was_seen)
    selected_tics = seen if presence == "seen" else ~seen
    selected = ~tic_real | (tic_real & selected_tics)
    if not (tic_real & selected_tics).any():
        return None
    return predictions.loc[selected].reset_index(drop=True)


def get_global_names():
    """Return all experiment directory names under outputs/detection."""
    if not OUTPUT_ROOT.exists():
        raise FileNotFoundError(f"Missing output directory: {OUTPUT_ROOT}")
    names = sorted(path.name for path in OUTPUT_ROOT.iterdir() if path.is_dir())
    if not names:
        raise ValueError(f"No experiment directories found in {OUTPUT_ROOT}")
    return names


def collect_metrics(
    global_name, split_name, folds=None, metadata=None, presence=None
):
    """Load one prediction table for each fold of a split."""
    rows = []
    for fold in range(1, K_FOLDS + 1):
        csv_path = OUTPUT_ROOT / global_name / f"fold{fold}_{split_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing prediction file: {csv_path}")
        predictions = load_predictions(csv_path)
        if presence is not None:
            train_types = training_tic_types(metadata, folds[fold]["train"])
            predictions = filter_by_training_presence(
                predictions, train_types, presence
            )
            if predictions is None:
                continue
        rows.append(load_fold_metrics(predictions))
    return pd.DataFrame(rows, columns=METRIC_NAMES)


def formatted_summary(metrics):
    """Return each metric as a 'mean (±std)' string across folds."""
    if metrics.empty:
        return pd.Series("N/A", index=METRIC_NAMES)

    means = metrics.mean()
    standard_deviations = metrics.std(ddof=1)
    values = {}
    for metric in METRIC_NAMES:
        mean = means[metric]
        deviation = standard_deviations[metric]
        if pd.isna(mean):
            values[metric] = "N/A"
        elif pd.isna(deviation):
            values[metric] = f"{mean:.4f} (±N/A)"
        else:
            values[metric] = f"{mean:.4f} (±{deviation:.4f})"
    return pd.Series(values)


def main():
    folds = load_split(SPLIT_PATH)
    metadata = pd.read_csv(METADATA_PATH, dtype={"Type": str})
    metadata["ID"] = metadata["ID"].astype(str).str.upper()
    metadata["Phase"] = metadata["Phase"].astype(str).str.upper()
    metadata["Sess"] = metadata["Sess"].astype(int)

    column_order = [
        "Validation - all",
        "Validation - seen",
        "Validation - unseen",
        "Test - all",
        "Test - seen",
        "Test - unseen",
    ]
    for global_name in get_global_names():
        results = {
            "Validation - all": collect_metrics(global_name, "val"),
            "Test - all": collect_metrics(global_name, "test"),
        }
        for presence in ("seen", "unseen"):
            results[f"Validation - {presence}"] = collect_metrics(
                global_name,
                "val",
                folds=folds,
                metadata=metadata,
                presence=presence,
            )
            results[f"Test - {presence}"] = collect_metrics(
                global_name,
                "test",
                folds=folds,
                metadata=metadata,
                presence=presence,
            )

        table = pd.DataFrame(
            {column: formatted_summary(results[column]) for column in column_order}
        )
        print(f"\nMetrics across {K_FOLDS} folds: {global_name}\n")
        print(table.to_string())


if __name__ == "__main__":
    main()
