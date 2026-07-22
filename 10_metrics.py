"""Aggregate detection and segmentation metrics across folds."""

from pathlib import Path

import pandas as pd
import torch

from bin.detection_metrics import get_group_metrics, get_tic_metrics
from bin.make_splits import load_split
from bin.segmentation_metrics import get_segmentation_metrics


OUTPUT_ROOTS = {
    "detection": Path("outputs/detection"),
    "segmentation": Path("outputs/segmentation"),
}
METADATA_PATH = Path("/projects/vocaltics/data/metadata.csv")
SPLIT_PATH = Path("splits.json")

DETECTION_METRICS = [
    "Tic accuracy",
    "Tic F1",
    "Tic AUROC",
    "Tic precision",
    "Tic recall",
    "Group accuracy",
    "Group macro F1",
]
SEGMENTATION_METRICS = [
    "Frame accuracy",
    "Frame F1",
    "Frame AUROC",
    "Segment accuracy",
    "Segment F1",
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

    return encode(predicted), encode(real)


def discover_experiments():
    """Return all experiment directories under both output roots."""
    experiments = []
    for task, root in OUTPUT_ROOTS.items():
        if not root.exists():
            continue
        experiments.extend(
            (task, path.name, path)
            for path in sorted(root.iterdir())
            if path.is_dir()
        )
    if not experiments:
        roots = ", ".join(str(root) for root in OUTPUT_ROOTS.values())
        raise FileNotFoundError(f"No experiment directories found under {roots}")
    return experiments


def prediction_paths(experiment_dir, split_name):
    """Return every available fold prediction CSV for one split."""
    paths = sorted(experiment_dir.glob(f"fold*_{split_name}.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No fold*_{split_name}.csv files found in {experiment_dir}"
        )
    return paths


def fold_number(csv_path):
    """Extract the integer fold number from foldN_split.csv."""
    try:
        return int(csv_path.stem[4:].split("_")[0])
    except ValueError as error:
        raise ValueError(f"Cannot read fold number from {csv_path}") from error


def load_detection_predictions(csv_path):
    """Load and validate one detection prediction table."""
    predictions = pd.read_csv(
        csv_path, dtype={"tic_type": str, "tic_group": str, "group_pred": str}
    )
    required = {
        "tic_real",
        "tic_probability",
        "tic_type",
        "tic_group",
        "group_pred",
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
    return predictions


def detection_fold_metrics(predictions):
    """Calculate detection and tic-group metrics for one fold."""
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
    """Return the individual TicIDs available in one training split."""
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


def collect_detection_metrics(
    experiment_dir, split_name, folds=None, metadata=None, presence=None
):
    """Calculate detection metrics for every available fold."""
    rows = []
    for csv_path in prediction_paths(experiment_dir, split_name):
        predictions = load_detection_predictions(csv_path)
        if presence is not None:
            number = fold_number(csv_path)
            if number not in folds:
                raise ValueError(f"Fold {number} is missing from {SPLIT_PATH}")
            train_types = training_tic_types(metadata, folds[number]["train"])
            predictions = filter_by_training_presence(
                predictions, train_types, presence
            )
            if predictions is None:
                continue
        rows.append(detection_fold_metrics(predictions))
    return pd.DataFrame(rows, columns=DETECTION_METRICS)


def load_segmentation_predictions(csv_path):
    """Load and validate one frame-level segmentation prediction table."""
    predictions = pd.read_csv(csv_path)
    required = {"segment_id", "frame_id", "tic_real", "tic_probability"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
    return predictions


def segmentation_fold_metrics(predictions):
    """Calculate frame and segment metrics while preserving segment boundaries."""
    score_rows = []
    target_rows = []
    lengths = set()
    for _, segment in predictions.groupby("segment_id", sort=False):
        segment = segment.sort_values("frame_id")
        scores = torch.tensor(
            segment["tic_probability"].to_numpy(), dtype=torch.float32
        )
        targets = boolean_values(segment["tic_real"])
        score_rows.append(scores)
        target_rows.append(targets)
        lengths.add(len(segment))
    if not score_rows:
        raise ValueError("No segmentation predictions were provided")
    if len(lengths) != 1:
        raise ValueError("Segmentation samples do not have equal frame lengths")
    scores = torch.stack(score_rows)
    targets = torch.stack(target_rows)
    return get_segmentation_metrics(scores, targets, from_logits=False)


def collect_segmentation_metrics(experiment_dir, split_name):
    """Calculate segmentation metrics for every available fold."""
    rows = [
        segmentation_fold_metrics(load_segmentation_predictions(csv_path))
        for csv_path in prediction_paths(experiment_dir, split_name)
    ]
    return pd.DataFrame(rows, columns=SEGMENTATION_METRICS)


def formatted_summary(metrics, metric_names):
    """Return each metric as a mean (plus/minus std) string across folds."""
    if metrics.empty:
        return pd.Series("N/A", index=metric_names)
    means = metrics.mean()
    standard_deviations = metrics.std(ddof=1)
    values = {}
    for metric in metric_names:
        mean = means[metric]
        deviation = standard_deviations[metric]
        if pd.isna(mean):
            values[metric] = "N/A"
        elif pd.isna(deviation):
            values[metric] = f"{mean:.4f} (±N/A)"
        else:
            values[metric] = f"{mean:.4f} (±{deviation:.4f})"
    return pd.Series(values)


def detection_table(experiment_dir, folds, metadata):
    """Return the all/seen/unseen detection metrics table."""
    column_order = [
        "Validation - all",
        "Validation - seen",
        "Validation - unseen",
        "Test - all",
        "Test - seen",
        "Test - unseen",
    ]
    results = {
        "Validation - all": collect_detection_metrics(experiment_dir, "val"),
        "Test - all": collect_detection_metrics(experiment_dir, "test"),
    }
    for presence in ("seen", "unseen"):
        results[f"Validation - {presence}"] = collect_detection_metrics(
            experiment_dir,
            "val",
            folds=folds,
            metadata=metadata,
            presence=presence,
        )
        results[f"Test - {presence}"] = collect_detection_metrics(
            experiment_dir,
            "test",
            folds=folds,
            metadata=metadata,
            presence=presence,
        )
    return pd.DataFrame(
        {
            column: formatted_summary(results[column], DETECTION_METRICS)
            for column in column_order
        }
    )


def segmentation_table(experiment_dir):
    """Return validation and test segmentation metrics across folds."""
    results = {
        "Validation": collect_segmentation_metrics(experiment_dir, "val"),
        "Test": collect_segmentation_metrics(experiment_dir, "test"),
    }
    return pd.DataFrame(
        {
            column: formatted_summary(metrics, SEGMENTATION_METRICS)
            for column, metrics in results.items()
        }
    )


def main():
    experiments = discover_experiments()
    folds = None
    metadata = None

    print(f"Found {len(experiments)} experiments")
    for task, global_name, experiment_dir in experiments:
        try:
            if task == "detection":
                if folds is None:
                    folds = load_split(SPLIT_PATH)
                    metadata = pd.read_csv(METADATA_PATH, dtype={"Type": str})
                    metadata["ID"] = metadata["ID"].astype(str).str.upper()
                    metadata["Phase"] = (
                        metadata["Phase"].astype(str).str.upper()
                    )
                    metadata["Sess"] = metadata["Sess"].astype(int)
                table = detection_table(experiment_dir, folds, metadata)
            else:
                table = segmentation_table(experiment_dir)
        except (FileNotFoundError, ValueError) as error:
            print(f"\nSkipped {task}/{global_name}: {error}")
            continue
        print(f"\n{task.title()} metrics: {global_name}\n")
        print(table.to_string())


if __name__ == "__main__":
    main()
