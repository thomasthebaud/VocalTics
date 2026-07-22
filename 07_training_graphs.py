"""Plot training curves for every detection and segmentation experiment."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MODEL_ROOTS = {
    "detection": Path("models/detection"),
    "segmentation": Path("models/segmentation"),
}
GRAPH_ROOT = Path("graphs/training_curve")
AUROC_COLUMNS = {
    "detection": "tic_auroc",
    "segmentation": "frame_auroc",
}


def discover_experiments():
    """Return every experiment directory under both model roots."""
    experiments = []
    for task, root in MODEL_ROOTS.items():
        if not root.exists():
            continue
        experiments.extend(
            (task, path.name, path)
            for path in sorted(root.iterdir())
            if path.is_dir()
        )
    if not experiments:
        roots = ", ".join(str(root) for root in MODEL_ROOTS.values())
        raise FileNotFoundError(f"No experiment directories found under {roots}")
    return experiments


def load_logs(log_dir, auroc_column):
    """Load and combine every available structured fold log."""
    log_paths = sorted(log_dir.glob("fold*.log"))
    if not log_paths:
        raise FileNotFoundError(f"No fold*.log files found in {log_dir}")

    logs = []
    required = {"fold", "epoch", "split", "loss", auroc_column}
    for log_path in log_paths:
        log = pd.read_csv(log_path)
        missing = required - set(log.columns)
        if missing:
            raise ValueError(f"Missing columns in {log_path}: {sorted(missing)}")
        logs.append(log)
    return pd.concat(logs, ignore_index=True)


def make_training_graph(task, global_name, log_dir):
    """Create loss and AUROC curves for one experiment."""
    auroc_column = AUROC_COLUMNS[task]
    logs = load_logs(log_dir, auroc_column)
    logs = logs.loc[logs["split"].isin(["train", "val"])].copy()
    if logs.empty:
        raise ValueError(f"No train or val log rows found in {log_dir}")
    logs["auroc"] = logs[auroc_column]
    fold_count = logs["fold"].nunique()

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(
        data=logs,
        x="epoch",
        y="loss",
        hue="split",
        style="split",
        markers=True,
        errorbar=("ci", 95),
        ax=axes[0],
    )
    axes[0].set_title("Training and validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    sns.lineplot(
        data=logs,
        x="epoch",
        y="auroc",
        hue="split",
        style="split",
        markers=True,
        errorbar=("ci", 95),
        ax=axes[1],
    )
    axes[1].set_title("Training and validation AUROC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Frame AUROC" if task == "segmentation" else "Tic AUROC")
    axes[1].set_ylim(0, 1)

    figure.suptitle(
        f"{task.title()}: {global_name} across {fold_count} folds",
        fontsize=14,
    )
    figure.tight_layout()
    graph_path = GRAPH_ROOT / task / f"{global_name}.png"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(graph_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved training curves to {graph_path}")


def main():
    sns.set_theme(style="whitegrid")
    experiments = discover_experiments()
    print(f"Found {len(experiments)} experiments")
    for task, global_name, log_dir in experiments:
        try:
            make_training_graph(task, global_name, log_dir)
        except (FileNotFoundError, ValueError) as error:
            print(f"Skipped {task}/{global_name}: {error}")


if __name__ == "__main__":
    main()
