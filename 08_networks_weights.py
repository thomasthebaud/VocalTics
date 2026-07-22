"""Print parameter counts for every saved fold-1 network."""

from pathlib import Path

import torch

from bin.detection_models import CNN as DetectionCNN
from bin.detection_models import ResNet34, TCNN, TDNN
from bin.segmentation_models import BiLSTM, CNN, CNN_BiLSTM


MODEL_ROOTS = {
    "Detection": Path("models/detection"),
    "Segmentation": Path("models/segmentation"),
}
MODEL_CLASSES = {
    "Detection": {
        "TDNN": TDNN,
        "ResNet34": ResNet34,
        "TCNN": TCNN,
        "CNN": DetectionCNN,
    },
    "Segmentation": {
        "BiLSTM": BiLSTM,
        "CNN": CNN,
        "CNN_BiLSTM": CNN_BiLSTM,
    },
}
INPUT_WEIGHT_KEYS = {
    "Detection": {
        "TDNN": "frame_layers.0.block.0.weight",
        "ResNet34": "stem.0.weight",
        "TCNN": "input_layer.block.0.weight",
        "CNN": "conv_layers.0.block.0.weight",
    },
    "Segmentation": {
        "BiLSTM": "recurrent_layers.weight_ih_l0",
        "CNN": "feature_layers.0.block.0.weight",
        "CNN_BiLSTM": "feature_layers.0.block.0.weight",
    },
}


def find_checkpoint(fold_dir):
    """Return best.pt, or the latest numbered checkpoint if it is absent."""
    best_path = fold_dir / "best.pt"
    if best_path.exists():
        return best_path
    epoch_paths = [path for path in fold_dir.glob("*.pt") if path.stem.isdigit()]
    if not epoch_paths:
        raise FileNotFoundError(f"No checkpoint found in {fold_dir}")
    return max(epoch_paths, key=lambda path: int(path.stem))


def load_checkpoint(checkpoint_path):
    """Load a trusted local checkpoint on the CPU."""
    try:
        return torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def build_model(task, checkpoint):
    """Reconstruct a model from its checkpoint metadata and saved weights."""
    model_name = checkpoint["model_name"]
    state_dict = checkpoint["model_state_dict"]
    if model_name not in MODEL_CLASSES[task]:
        raise ValueError(f"Unknown {task.lower()} model: {model_name}")

    input_key = INPUT_WEIGHT_KEYS[task][model_name]
    if input_key not in state_dict:
        raise ValueError(f"Cannot infer input dimension: missing {input_key}")
    input_dim = state_dict[input_key].shape[1]

    model_class = MODEL_CLASSES[task][model_name]
    if task == "Detection":
        num_groups = checkpoint.get("num_groups")
        if num_groups is None:
            num_groups = state_dict["group_classifier.weight"].shape[0]
        model = model_class(input_dim=input_dim, num_groups=num_groups)
    else:
        model = model_class(input_dim=input_dim)

    model.load_state_dict(state_dict)
    return model


def collect_networks():
    """Load every available fold-1 model and collect its parameter count."""
    rows = []
    for task, root in MODEL_ROOTS.items():
        if not root.exists():
            continue
        for experiment_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            fold_dir = experiment_dir / "fold1"
            if not fold_dir.is_dir():
                continue
            try:
                checkpoint_path = find_checkpoint(fold_dir)
                checkpoint = load_checkpoint(checkpoint_path)
                model = build_model(task, checkpoint)
            except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
                print(f"Skipped {task}/{experiment_dir.name}: {error}")
                continue
            rows.append(
                {
                    "Task": task,
                    "Network": experiment_dir.name,
                    "Parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                }
            )
    return rows


def print_table(rows):
    """Print parameter counts as an aligned text table."""
    if not rows:
        roots = ", ".join(str(root) for root in MODEL_ROOTS.values())
        raise FileNotFoundError(f"No fold-1 networks found under {roots}")

    headers = ("Task", "Network", "Parameters")
    formatted_rows = [
        (row["Task"], row["Network"], f'{row["Parameters"]:,}') for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in formatted_rows))
        for index in range(len(headers))
    ]
    print(
        f"{headers[0]:<{widths[0]}}  "
        f"{headers[1]:<{widths[1]}}  "
        f"{headers[2]:>{widths[2]}}"
    )
    print("  ".join("-" * width for width in widths))
    for task, network, parameters in formatted_rows:
        print(
            f"{task:<{widths[0]}}  "
            f"{network:<{widths[1]}}  "
            f"{parameters:>{widths[2]}}"
        )


def main():
    """Collect fold-1 networks and print their parameter counts."""
    print_table(collect_networks())


if __name__ == "__main__":
    main()
