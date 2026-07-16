"""Generate confusion matrices for every detection experiment."""

from pathlib import Path

from bin.graphs.confusion_matrix import make_confusion_matrix


OUTPUT_ROOT = Path("outputs/detection")
GRAPH_ROOT = Path("graphs")


def get_global_names():
    """Return all experiment directory names under the configured output root."""
    if not OUTPUT_ROOT.exists():
        raise FileNotFoundError(f"Missing output directory: {OUTPUT_ROOT}")
    names = sorted(path.name for path in OUTPUT_ROOT.iterdir() if path.is_dir())
    if not names:
        raise ValueError(f"No experiment directories found in {OUTPUT_ROOT}")
    return names


def main():
    global_names = get_global_names()
    print(f"Generating confusion matrices for {global_names} experiments")
    for global_name in global_names:
        input_path = OUTPUT_ROOT / global_name
        output_path = GRAPH_ROOT / global_name / "confusion_matrices.png"
        make_confusion_matrix(input_path, output_path)


if __name__ == "__main__":
    main()
