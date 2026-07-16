"""Functions shared by detection and segmentation training scripts."""

import argparse
import csv
from pathlib import Path
import time

import torchaudio
from torch.utils.data import DataLoader

from bin.make_splits import load_split


FEATURE_NAMES = ["Spectrogram", "MelSpectrogram", "MFCC"]
SPLIT_NAMES = ["participant", "session", "file"]


def parse_training_args(
    description,
    model_classes,
    default_model,
    default_split,
    default_feature,
):
    """Parse the command-line arguments shared by both training scripts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--fold", type=int, default=1, help="Fold number to run (default: 1)"
    )
    parser.add_argument(
        "--model-name",
        "--model_name",
        choices=model_classes,
        default=default_model,
        help=f"Model architecture (default: {default_model})",
    )
    parser.add_argument(
        "--split-by",
        "--split_by",
        choices=SPLIT_NAMES,
        default=default_split,
        help=f"Cross-validation split unit (default: {default_split})",
    )
    parser.add_argument(
        "--feat-name",
        "--feat_name",
        choices=FEATURE_NAMES,
        default=default_feature,
        help=f"Input feature transform (default: {default_feature})",
    )
    return parser.parse_args()


def make_transform(feature_name, sample_rate=16000, n_mels=80, n_mfcc=40):
    """Create the selected audio transform and return its feature dimension."""
    if feature_name == "Spectrogram":
        return torchaudio.transforms.Spectrogram(n_fft=400, hop_length=160), 201
    if feature_name == "MelSpectrogram":
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels,
        )
        return transform, n_mels
    if feature_name == "MFCC":
        transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": n_mels},
        )
        return transform, n_mfcc
    raise ValueError(f"Unknown feature name: {feature_name}")


def load_fold(split_path, fold_number):
    """Load one fold from the saved cross-validation definition."""
    split_path = Path(split_path)
    if not split_path.exists():
        raise FileNotFoundError(
            f"No saved split found at {split_path}; "
            "run 04_generate_new_split.py first"
        )
    splits = load_split(split_path)
    print(f"Loaded splits from {split_path}")
    if fold_number not in splits:
        raise ValueError(
            f"Fold {fold_number} is unavailable; choose from {sorted(splits)}"
        )
    return splits[fold_number]


def make_data_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size,
    num_workers,
):
    """Create the train, validation, and test DataLoaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader


def print_metrics(split_name, metrics):
    """Print one compact line containing every evaluation metric."""
    values = " | ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    print(f"{split_name}: {values}")


def initialize_log(log_path, fields):
    """Create a new structured metrics log for one fold."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=fields).writeheader()


def append_log(log_path, fields, fold, epoch, split_name, metrics):
    """Append one epoch and split to the structured fold log."""
    row = {"fold": fold, "epoch": epoch, "split": split_name, **metrics}
    with log_path.open("a", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=fields).writerow(row)


def start_timer():
    """Return a monotonic timestamp for training duration measurements."""
    return time.perf_counter()


def format_duration(seconds):
    """Format a duration compactly as MM:SS or HH:MM:SS."""
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    return f"{minutes:02}:{seconds:02}"


def print_epoch_timing(epoch, total_epochs, training_start, epoch_start):
    """Print elapsed, current-epoch, and estimated remaining durations."""
    now = time.perf_counter()
    elapsed = now - training_start
    epoch_duration = now - epoch_start
    remaining = elapsed / epoch * (total_epochs - epoch)
    print(
        f"\nEpoch {epoch}/{total_epochs} | "
        f"elapsed {format_duration(elapsed)} | "
        f"epoch {format_duration(epoch_duration)} | "
        f"ETA {format_duration(remaining)}"
    )


def print_fold_time(fold, training_start):
    """Print the total training and final-evaluation time for one fold."""
    total_time = time.perf_counter() - training_start
    print(f"Fold {fold} total time: {format_duration(total_time)}")
