"""Train and evaluate a TDNN for tic detection and group classification."""

import argparse
import csv
import math
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from bin.dataset import SpecDataset
from bin.make_splits import (
    load_split,
    save_split,
    splits_by_file,
    splits_by_participant,
    splits_by_session,
)
from bin.metrics import get_group_metrics, get_tic_metrics
from bin.models import ResNet34, TCNN, TDNN


METADATA_PATH = Path("/projects/vocaltics/data/metadata.csv")
SPLIT_PATH = Path("splits.json")
MODEL_NAME = "ResNet34"
SPLIT_BY = "session"
FEAT_NAME = "MFCC"
GLOBAL_NAME = f"{MODEL_NAME}_{FEAT_NAME}_by{SPLIT_BY}"
MODEL_DIR = Path("models/detection") / GLOBAL_NAME
OUTPUT_DIR = Path("outputs/detection") / GLOBAL_NAME
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
NUM_WORKERS = 0
N_MELS = 80
N_MFCC = 40
SAMPLE_RATE = 16000
K_FOLDS = 5

MODEL_CLASSES = {
    "TDNN": TDNN,
    "ResNet34": ResNet34,
    "TCNN": TCNN,
}
SPLIT_FUNCTIONS = {
    "participant": splits_by_participant,
    "session": splits_by_session,
    "file": splits_by_file,
}
LOG_FIELDS = [
    "fold",
    "epoch",
    "split",
    "loss",
    "tic_accuracy",
    "tic_f1",
    "tic_auroc",
    "tic_precision",
    "tic_recall",
    "group_accuracy",
    "group_macro_f1",
]


def make_transform():
    """Create the selected audio transform and return its feature dimension."""
    if FEAT_NAME == "Spectrogram":
        return torchaudio.transforms.Spectrogram(n_fft=400, hop_length=160), 201
    if FEAT_NAME == "MelSpectrogram":
        transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=400,
            hop_length=160,
            n_mels=N_MELS,
        )
        return transform, N_MELS
    if FEAT_NAME == "MFCC":
        transform = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": N_MELS},
        )
        return transform, N_MFCC
    raise ValueError(f"Unknown FEAT_NAME: {FEAT_NAME}")


def parse_args():
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train tic detection on one cross-validation fold."
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold number to run (default: 1)",
    )
    parser.add_argument(
        "--newsplit",
        action="store_true",
        help="Generate and save new splits instead of loading splits.json",
    )
    return parser.parse_args()


def calculate_loss(
    tic_logits,
    group_logits,
    tic_targets,
    groups,
    tic_loss_function,
    group_loss_function,
):
    """Combine tic-presence loss with group loss on tic samples only."""
    group_real = groups.float().to(tic_logits.device)
    tic_loss = tic_loss_function(tic_logits, tic_targets)
    group_mask = tic_targets == 1
    if group_mask.any():
        group_loss = group_loss_function(
            group_logits[group_mask], group_real[group_mask]
        )
    else:
        group_loss = group_logits.sum() * 0
    return tic_loss + group_loss, group_real


def train_one_epoch(
    model,
    loader,
    optimizer,
    tic_loss_function,
    group_loss_function,
    device,
):
    """Run one optimization epoch."""
    model.train()
    for features, _, groups, tic_presence in loader:
        features = features.to(device)
        tic_targets = tic_presence.long().to(device)
        optimizer.zero_grad()
        tic_logits, group_logits = model(features)
        loss, _ = calculate_loss(
            tic_logits,
            group_logits,
            tic_targets,
            groups,
            tic_loss_function,
            group_loss_function,
        )
        loss.backward()
        optimizer.step()


def evaluate(
    model,
    loader,
    tic_loss_function,
    group_loss_function,
    index_to_group,
    device,
):
    """Evaluate a dataset and return all metrics plus a prediction table."""
    model.eval()
    total_loss = 0.0
    sample_count = 0
    all_tic_logits = []
    all_group_logits = []
    all_tic_targets = []
    all_group_targets = []
    prediction_rows = []

    with torch.inference_mode():
        for features, tic_types, groups, tic_presence in loader:
            features = features.to(device)
            tic_targets = tic_presence.long().to(device)
            tic_logits, group_logits = model(features)
            loss, group_real = calculate_loss(
                tic_logits,
                group_logits,
                tic_targets,
                groups,
                tic_loss_function,
                group_loss_function,
            )

            batch_size = len(tic_targets)
            total_loss += loss.item() * batch_size
            sample_count += batch_size
            all_tic_logits.append(tic_logits.cpu())
            all_group_logits.append(group_logits.cpu())
            all_tic_targets.append(tic_targets.cpu())
            all_group_targets.append(group_real.cpu())

            tic_probabilities = tic_logits.softmax(dim=1)[:, 1].cpu()
            tic_predictions = tic_logits.argmax(dim=1).cpu()
            group_probabilities = group_logits.sigmoid().cpu()
            group_predictions = group_probabilities >= 0.5

            for index in range(batch_size):
                real_groups = [
                    index_to_group[group_index]
                    for group_index in torch.where(group_real[index] > 0.5)[0].tolist()
                ]
                predicted_groups = [
                    index_to_group[group_index]
                    for group_index in torch.where(group_predictions[index])[0].tolist()
                ]
                prediction_rows.append(
                    {
                        "tic_type": tic_types[index],
                        "tic_group": "+".join(real_groups) if real_groups else "-1",
                        "tic_real": bool(tic_targets[index].item()),
                        "tic_pred": bool(tic_predictions[index].item()),
                        "tic_probability": tic_probabilities[index].item(),
                        "group_pred": (
                            "+".join(predicted_groups) if predicted_groups else "-1"
                        ),
                        "group_probability": group_probabilities[index].max().item(),
                    }
                )

    tic_logits = torch.cat(all_tic_logits)
    group_logits = torch.cat(all_group_logits)
    tic_real = torch.cat(all_tic_targets)
    group_real = torch.cat(all_group_targets)
    tic_accuracy, tic_f1, tic_auroc, precision, recall = get_tic_metrics(
        tic_logits, tic_real
    )
    if group_real.any():
        group_accuracy, group_macro_f1 = get_group_metrics(
            group_logits, group_real
        )
    else:
        group_accuracy = float("nan")
        group_macro_f1 = float("nan")

    metrics = {
        "loss": total_loss / sample_count,
        "tic_accuracy": tic_accuracy,
        "tic_f1": tic_f1,
        "tic_auroc": tic_auroc,
        "tic_precision": precision,
        "tic_recall": recall,
        "group_accuracy": group_accuracy,
        "group_macro_f1": group_macro_f1,
    }
    return metrics, pd.DataFrame(prediction_rows)


def print_metrics(split_name, metrics):
    """Print one compact line containing every evaluation metric."""
    values = " | ".join(f"{name}={value:.4f}" for name, value in metrics.items())
    print(f"{split_name}: {values}")


def initialize_log(log_path):
    """Create a new structured metrics log for one fold."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as file:
        csv.DictWriter(file, fieldnames=LOG_FIELDS).writeheader()


def append_log(log_path, fold, epoch, split_name, metrics):
    """Append one epoch and split to the structured fold log."""
    row = {"fold": fold, "epoch": epoch, "split": split_name, **metrics}
    with log_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=LOG_FIELDS)
        writer.writerow(row)


def main():
    args = parse_args()
    torch.manual_seed(42)
    metadata = pd.read_csv(
        METADATA_PATH, dtype={"Type": str, "Group": str}
    )
    if SPLIT_BY not in SPLIT_FUNCTIONS:
        raise ValueError(f"Unknown SPLIT_BY: {SPLIT_BY}")
    if MODEL_NAME not in MODEL_CLASSES:
        raise ValueError(f"Unknown MODEL_NAME: {MODEL_NAME}")
    if args.newsplit:
        splits = SPLIT_FUNCTIONS[SPLIT_BY](metadata, K=K_FOLDS)
        save_split(splits, SPLIT_PATH)
        print(f"Generated and saved new splits to {SPLIT_PATH}")
    else:
        if not SPLIT_PATH.exists():
            raise FileNotFoundError(
                f"No saved split found at {SPLIT_PATH}; run with --newsplit"
            )
        splits = load_split(SPLIT_PATH)
        print(f"Loaded splits from {SPLIT_PATH}")
    if args.fold not in splits:
        raise ValueError(
            f"Fold {args.fold} is unavailable; choose from {sorted(splits)}"
        )
    fold = splits[args.fold]
    fold_model_dir = MODEL_DIR / f"fold{args.fold}"
    log_path = MODEL_DIR / f"fold{args.fold}.log"
    initialize_log(log_path)

    transform, input_dim = make_transform()
    print("Train ", end='')
    train_dataset = SpecDataset(METADATA_PATH, fold["train"], transform, win_len=10, p_tics=0.5)
    print("Val ", end='')
    val_dataset = SpecDataset(METADATA_PATH, fold["val"], transform, win_len=10, p_tics=0.5,include_multigroup=False)
    print("Test ", end='')
    test_dataset = SpecDataset(METADATA_PATH, fold["test"], transform,win_len=10, p_tics=0.5,include_multigroup=False)

    if not (
        train_dataset.group_to_index
        == val_dataset.group_to_index
        == test_dataset.group_to_index
    ):
        raise ValueError("Dataset splits use different group mappings")
    num_groups = train_dataset.num_groups
    group_to_index = train_dataset.group_to_index
    index_to_group = train_dataset.index_to_group

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL_CLASSES[MODEL_NAME](
        input_dim=input_dim, num_groups=num_groups
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    tic_loss_function = nn.CrossEntropyLoss()
    group_loss_function = nn.BCEWithLogitsLoss()
    fold_model_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    best_auroc = float("-inf")
    best_epoch = None
    best_path = fold_model_dir / "best.pt"
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            tic_loss_function,
            group_loss_function,
            device,
        )
        train_metrics, _ = evaluate(
            model,
            train_loader,
            tic_loss_function,
            group_loss_function,
            index_to_group,
            device,
        )
        val_metrics, val_predictions = evaluate(
            model,
            val_loader,
            tic_loss_function,
            group_loss_function,
            index_to_group,
            device,
        )
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print_metrics("train", train_metrics)
        print_metrics("val", val_metrics)
        append_log(log_path, args.fold, epoch, "train", train_metrics)
        append_log(log_path, args.fold, epoch, "val", val_metrics)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "group_to_index": group_to_index,
            "num_groups": num_groups,
            "model_name": MODEL_NAME,
            "feature_name": FEAT_NAME,
            "split_by": SPLIT_BY,
            "fold": args.fold,
            "val_tic_auroc": val_metrics["tic_auroc"],
        }
        torch.save(checkpoint, fold_model_dir / f"{epoch}.pt")

        val_auroc = val_metrics["tic_auroc"]
        if best_epoch is None or (
            math.isfinite(val_auroc) and val_auroc > best_auroc
        ):
            torch.save(checkpoint, best_path)
            best_epoch = epoch
            if math.isfinite(val_auroc):
                best_auroc = val_auroc
            print(f"Saved new best checkpoint to {best_path} (best val AUROC: {best_auroc:.4f})")

    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    val_metrics, val_predictions = evaluate(
        model,
        val_loader,
        tic_loss_function,
        group_loss_function,
        index_to_group,
        device,
    )
    print(f"\nBest epoch: {best_checkpoint['epoch']}")
    print_metrics("best val", val_metrics)

    test_metrics, test_predictions = evaluate(
        model,
        test_loader,
        tic_loss_function,
        group_loss_function,
        index_to_group,
        device,
    )
    print("\nTest results")
    print_metrics("test", test_metrics)
    val_predictions.to_csv(
        OUTPUT_DIR / f"fold{args.fold}_val.csv", index=False
    )
    test_predictions.to_csv(
        OUTPUT_DIR / f"fold{args.fold}_test.csv", index=False
    )
    print(f"Saved predictions to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
