"""Generate confusion matrices for detection and segmentation experiments."""

from pathlib import Path

from bin.graphs.confusion_matrix import make_confusion_matrix


OUTPUT_ROOTS = {
    "detection": Path("outputs/detection"),
    "segmentation": Path("outputs/segmentation"),
}
GRAPH_ROOT = Path("graphs")


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


def main():
    experiments = discover_experiments()
    print(f"Found {len(experiments)} experiments")
    for task, global_name, input_path in experiments:
        output_path = (
            GRAPH_ROOT / task / global_name / "confusion_matrices.png"
        )
        try:
            make_confusion_matrix(input_path, output_path, task=task)
        except (FileNotFoundError, ValueError) as error:
            print(f"Skipped {task}/{global_name}: {error}")


if __name__ == "__main__":
    main()
