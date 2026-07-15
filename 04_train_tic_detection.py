"""Train and evaluate a TDNN for tic detection and group classification."""

import argparse
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader

from bin.dataset import SpecDataset
from bin.make_splits import (
    save_split,
    splits_by_file,
    splits_by_participant,
    splits_by_session,
)
from bin.metrics import get_group_metrics, get_tic_metrics
from bin.models import ResNet34, TCNN, TDNN


METADATA_PATH = Path("/projects/vocaltics/data/metadata.csv")
SPLIT_PATH = Path("splits.json")
MODEL_NAME = "TDNN"
SPLIT_BY = "session"
FEAT_NAME = "MFCC"
GLOBAL_NAME = f"{MODEL_NAME}_{FEAT_NAME}_by{SPLIT_BY}"
MODEL_DIR = Path("models/detection") / GLOBAL_NAME
OUTPUT_DIR = Path("outputs/detection") / GLOBAL_NAME
EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001
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
    return parser.parse_args()


def group_names(dataset):
    """Return the tic-group names present in a dataset split."""
    return {
        str(group)
        for group in dataset.tics["Group"].dropna()
        if str(group) != "-1"
    }


def group_targets(groups, group_to_index, device):
    """Convert group names to class indices, retaining -1 for non-tics."""
    targets = [
        -1 if str(group) == "-1" else group_to_index[str(group)]
        for group in groups
    ]
    return torch.tensor(targets, dtype=torch.long, device=device)


def calculate_loss(
    tic_logits,
    group_logits,
    tic_targets,
    groups,
    group_to_index,
    tic_loss_function,
    group_loss_function,
):
    """Combine tic-presence loss with group loss on tic samples only."""
    group_real = group_targets(groups, group_to_index, tic_logits.device)
    tic_loss = tic_loss_function(tic_logits, tic_targets)
    group_mask = (tic_targets == 1) & (group_real >= 0)
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
    group_to_index,
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
            group_to_index,
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
    group_to_index,
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
                group_to_index,
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
            group_probabilities = group_logits.softmax(dim=1).cpu()
            group_predictions = group_probabilities.argmax(dim=1)

            for index in range(batch_size):
                predicted_group_index = group_predictions[index].item()
                prediction_rows.append(
                    {
                        "tic_type": tic_types[index],
                        "tic_group": groups[index],
                        "tic_real": bool(tic_targets[index].item()),
                        "tic_pred": bool(tic_predictions[index].item()),
                        "tic_probability": tic_probabilities[index].item(),
                        "group_pred": index_to_group[predicted_group_index],
                        "group_probability": group_probabilities[
                            index, predicted_group_index
                        ].item(),
                    }
                )

    tic_logits = torch.cat(all_tic_logits)
    group_logits = torch.cat(all_group_logits)
    tic_real = torch.cat(all_tic_targets)
    group_real = torch.cat(all_group_targets)
    tic_accuracy, tic_f1, tic_auroc, precision, recall = get_tic_metrics(
        tic_logits, tic_real
    )
    if (group_real != -1).any():
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
    splits = SPLIT_FUNCTIONS[SPLIT_BY](metadata, K=K_FOLDS)
    save_split(splits, SPLIT_PATH)
    if args.fold not in splits:
        raise ValueError(
            f"Fold {args.fold} is unavailable; choose from {sorted(splits)}"
        )
    fold = splits[args.fold]
    fold_model_dir = MODEL_DIR / f"fold{args.fold}"

    transform, input_dim = make_transform()
    train_dataset = SpecDataset(
        METADATA_PATH, fold["train"], transform, win_len=10, p_tics=0.5
    )
    val_dataset = SpecDataset(
        METADATA_PATH, fold["val"], transform, win_len=10, p_tics=0.5
    )
    test_dataset = SpecDataset(
        METADATA_PATH, fold["test"], transform, win_len=10, p_tics=0.5
    )

    num_groups = max(
        train_dataset.num_groups,
        val_dataset.num_groups,
        test_dataset.num_groups,
    )
    available_groups = sorted(
        group_names(train_dataset)
        | group_names(val_dataset)
        | group_names(test_dataset)
    )
    if len(available_groups) > num_groups:
        raise ValueError(
            "The union of split groups is larger than the requested model output"
        )
    group_to_index = {
        group: index for index, group in enumerate(available_groups)
    }
    index_to_group = {
        index: group for group, index in group_to_index.items()
    }

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
    group_loss_function = nn.CrossEntropyLoss()
    fold_model_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    val_predictions = None
    for epoch in range(1, EPOCHS + 1):
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            tic_loss_function,
            group_loss_function,
            group_to_index,
            device,
        )
        train_metrics, _ = evaluate(
            model,
            train_loader,
            tic_loss_function,
            group_loss_function,
            group_to_index,
            index_to_group,
            device,
        )
        val_metrics, val_predictions = evaluate(
            model,
            val_loader,
            tic_loss_function,
            group_loss_function,
            group_to_index,
            index_to_group,
            device,
        )
        print(f"\nEpoch {epoch}/{EPOCHS}")
        print_metrics("train", train_metrics)
        print_metrics("val", val_metrics)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "group_to_index": group_to_index,
                "num_groups": num_groups,
                "model_name": MODEL_NAME,
                "feature_name": FEAT_NAME,
                "split_by": SPLIT_BY,
                "fold": args.fold,
            },
            fold_model_dir / f"{epoch}.pt",
        )

    test_metrics, test_predictions = evaluate(
        model,
        test_loader,
        tic_loss_function,
        group_loss_function,
        group_to_index,
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
