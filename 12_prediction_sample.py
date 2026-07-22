"""Plot representative segmentation predictions from every F1 decile."""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from bin.detection_datasets import _load_audio_window, _load_embedding_window
from bin.training_functions import FEATURE_NAMES, make_transform


OUTPUT_ROOT = Path("outputs/segmentation")
GRAPH_ROOT = Path("graphs/samples")
RANDOM_SEED = 42
REQUIRED_COLUMNS = {
    "segment_id",
    "frame_id",
    "tic_real",
    "tic_pred",
    "tic_probability",
    "AudioPath",
    "WindowStart",
    "WindowDuration",
    "AudioDuration",
}


def boolean_values(values):
    """Convert CSV boolean values to a NumPy boolean array."""
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1"]).to_numpy()


def frame_f1(rows):
    """Calculate one sample's frame-level F1 score."""
    real = boolean_values(rows["tic_real"])
    predicted = boolean_values(rows["tic_pred"])
    true_positive = np.sum(real & predicted)
    false_positive = np.sum(~real & predicted)
    false_negative = np.sum(real & ~predicted)
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def feature_name(global_name):
    """Extract the configured feature name from an experiment directory."""
    matches = [
        name for name in FEATURE_NAMES if f"_{name}_by" in global_name
    ]
    if len(matches) != 1:
        raise ValueError(f"Cannot determine feature type from {global_name}")
    return matches[0]


def load_test_predictions(experiment_dir):
    """Load test predictions across folds and retain fold identity."""
    paths = sorted(experiment_dir.glob("fold*_test.csv"))
    if not paths:
        raise FileNotFoundError(f"No fold*_test.csv files in {experiment_dir}")
    predictions = []
    for path in paths:
        table = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(table.columns)
        if missing:
            raise ValueError(
                f"{path} lacks sample provenance columns {sorted(missing)}; "
                "rerun 06_train_tic_segmentation.py for this fold"
            )
        table["fold"] = path.stem.split("_")[0]
        predictions.append(table)
    return pd.concat(predictions, ignore_index=True)


def ranked_samples(predictions):
    """Return sample keys sorted from highest to lowest frame F1."""
    samples = []
    for key, rows in predictions.groupby(["fold", "segment_id"], sort=False):
        samples.append((key, frame_f1(rows)))
    return sorted(samples, key=lambda item: item[1], reverse=True)


def select_deciles(samples, generator):
    """Randomly choose one sample from each F1-ranked decile."""
    if len(samples) < 10:
        raise ValueError(
            f"At least 10 test samples are required, found {len(samples)}"
        )
    deciles = np.array_split(np.arange(len(samples)), 10)
    return [
        (decile_number, samples[generator.choice(indices.tolist())])
        for decile_number, indices in enumerate(deciles, start=1)
    ]


def load_features(rows, name, waveform):
    """Recreate the exact feature window used for one prediction sample."""
    first = rows.iloc[0]
    if name == "WavLM":
        embedding_path = str(first.get("embedding_path", ""))
        if not embedding_path or embedding_path.lower() == "nan":
            raise ValueError("Missing embedding_path for WavLM sample")
        return _load_embedding_window(
            embedding_path=embedding_path,
            window_start=float(first["WindowStart"]),
            win_len=float(first["WindowDuration"]),
            audio_duration=float(first["AudioDuration"]),
            frames_per_second=50,
        )
    transform, _ = make_transform(name)
    return transform(waveform).squeeze(0)


def tic_spans(labels, duration):
    """Yield start and end times for contiguous real tic frames."""
    padded = np.concatenate(([False], labels, [False])).astype(int)
    changes = np.diff(padded)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]
    for start, end in zip(starts, ends):
        yield start / len(labels) * duration, end / len(labels) * duration


def plot_sample(rows, name, score, title, output_path):
    """Draw waveform, features, and frame probabilities for one sample."""
    rows = rows.sort_values("frame_id")
    first = rows.iloc[0]
    duration = float(first["WindowDuration"])
    waveform = _load_audio_window(
        first["AudioPath"], float(first["WindowStart"]), duration
    )
    features = load_features(rows, name, waveform).float().cpu()
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got {tuple(features.shape)}")
    if name in {"Spectrogram", "MelSpectrogram"}:
        features = torch.log1p(features.abs())

    real = boolean_values(rows["tic_real"])
    probabilities = rows["tic_probability"].astype(float).to_numpy()
    waveform_values = waveform.squeeze(0).cpu().numpy()
    waveform_time = np.linspace(0, duration, len(waveform_values), endpoint=False)
    frame_time = (np.arange(len(probabilities)) + 0.5) / len(probabilities)
    frame_time *= duration

    figure, axes = plt.subplots(
        3, 1, figsize=(14, 9), sharex=True, height_ratios=(1, 2, 1)
    )
    axes[0].plot(waveform_time, waveform_values, color="black", linewidth=0.6)
    for start, end in tic_spans(real, duration):
        axes[0].axvspan(start, end, color="red", alpha=0.25)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Waveform and real tic intervals")

    image = axes[1].imshow(
        features.numpy(),
        origin="lower",
        aspect="auto",
        extent=(0, duration, 0, features.shape[0]),
        cmap="magma",
    )
    axes[1].set_ylabel("Feature dimension")
    axes[1].set_title(name)
    figure.colorbar(image, ax=axes[1], fraction=0.02, pad=0.01)

    axes[2].plot(frame_time, probabilities, color="tab:blue", linewidth=1.2)
    axes[2].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    axes[2].set_xlim(0, duration)
    axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("Time in sampled window (seconds)")
    axes[2].set_ylabel("Tic probability")
    axes[2].set_title("Model prediction")

    figure.suptitle(f"{title} | frame F1={score:.3f}")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output_path}")


def process_experiment(experiment_dir, generator):
    """Generate ten F1-decile sample figures for one experiment."""
    predictions = load_test_predictions(experiment_dir)
    name = feature_name(experiment_dir.name)
    selected = select_deciles(ranked_samples(predictions), generator)
    for decile, ((fold, segment_id), score) in selected:
        rows = predictions.loc[
            (predictions["fold"] == fold)
            & (predictions["segment_id"] == segment_id)
        ]
        output_path = (
            GRAPH_ROOT / experiment_dir.name / f"decile_{decile}.png"
        )
        title = (
            f"{experiment_dir.name} | decile {decile} | "
            f"{fold}, segment {segment_id}"
        )
        plot_sample(rows, name, score, title, output_path)


def main():
    """Process every segmentation experiment under the output root."""
    if not OUTPUT_ROOT.exists():
        raise FileNotFoundError(f"Missing output directory: {OUTPUT_ROOT}")
    experiments = sorted(path for path in OUTPUT_ROOT.iterdir() if path.is_dir())
    if not experiments:
        raise FileNotFoundError(f"No experiments found under {OUTPUT_ROOT}")

    generator = random.Random(RANDOM_SEED)
    print(f"Found {len(experiments)} segmentation experiments")
    for experiment_dir in experiments:
        try:
            process_experiment(experiment_dir, generator)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            print(f"Skipped {experiment_dir.name}: {error}")


if __name__ == "__main__":
    main()
