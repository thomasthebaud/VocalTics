"""Plot cross-validation training curves from structured fold logs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


GLOBAL_NAME = "TDNN_MFCC_bysession"
K_FOLDS = 5
LOG_DIR = Path("models/detection") / GLOBAL_NAME
GRAPH_PATH = Path("graphs/training_curve") / f"{GLOBAL_NAME}.png"


def load_logs():
    """Load and combine the structured training logs from every fold."""
    logs = []
    for fold in range(1, K_FOLDS + 1):
        log_path = LOG_DIR / f"fold{fold}.log"
        if not log_path.exists():
            raise FileNotFoundError(f"Missing training log: {log_path}")
        log = pd.read_csv(log_path)
        required = {"fold", "epoch", "split", "loss", "tic_auroc"}
        missing = required - set(log.columns)
        if missing:
            raise ValueError(f"Missing columns in {log_path}: {sorted(missing)}")
        logs.append(log)
    return pd.concat(logs, ignore_index=True)


def main():
    logs = load_logs()
    logs = logs.loc[logs["split"].isin(["train", "val"])]
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.lineplot(
        data=logs,
        x="epoch",
        y="loss",
        hue="split",
        style="split",
        markers=True,
        ci=95,
        ax=axes[0],
    )
    axes[0].set_title("Training and validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    sns.lineplot(
        data=logs,
        x="epoch",
        y="tic_auroc",
        hue="split",
        style="split",
        markers=True,
        ci=95,
        ax=axes[1],
    )
    axes[1].set_title("Training and validation tic AUROC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUROC")
    axes[1].set_ylim(0, 1)

    figure.suptitle(f"{GLOBAL_NAME} across {K_FOLDS} folds", fontsize=14)
    figure.tight_layout()
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(GRAPH_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved training curves to {GRAPH_PATH}")


if __name__ == "__main__":
    main()
