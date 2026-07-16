"""Train and evaluate a TDNN for tic detection and group classification."""

import math
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from bin.detection_datasets import SpecDataset
from bin.detection_metrics import get_group_metrics, get_tic_metrics
from bin.detection_models import ResNet34, TCNN, TDNN
from bin.training_functions import (
    append_log,
    initialize_log,
    load_fold,
    make_data_loaders,
    make_transform,
    parse_training_args,
    print_epoch_timing,
    print_fold_time,
    print_metrics,
    start_timer,
)


METADATA_PATH = Path("/projects/vocaltics/data/metadata.csv")
SPLIT_PATH = Path("splits.json")
MODEL_NAME = "ResNet34"
SPLIT_BY = "session"
FEAT_NAME = "MFCC"
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
NUM_WORKERS = 0

MODEL_CLASSES = {
    "TDNN": TDNN,
    "ResNet34": ResNet34,
    "TCNN": TCNN,
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


def main():
    args = parse_training_args(
        description="Train tic detection on one cross-validation fold.",
        model_classes=MODEL_CLASSES,
        default_model=MODEL_NAME,
        default_split=SPLIT_BY,
        default_feature=FEAT_NAME,
    )
    global_name = f"{args.model_name}_{args.feat_name}_by{args.split_by}"
    model_dir = Path("models/detection") / global_name
    output_dir = Path("outputs/detection") / global_name
    torch.manual_seed(42)
    fold = load_fold(SPLIT_PATH, args.fold)
    fold_model_dir = model_dir / f"fold{args.fold}"
    log_path = model_dir / f"fold{args.fold}.log"
    initialize_log(log_path, LOG_FIELDS)

    transform, input_dim = make_transform(args.feat_name)
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

    train_loader, val_loader, test_loader = make_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL_CLASSES[args.model_name](
        input_dim=input_dim, num_groups=num_groups
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    tic_loss_function = nn.CrossEntropyLoss()
    group_loss_function = nn.BCEWithLogitsLoss()
    fold_model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_auroc = float("-inf")
    best_epoch = None
    best_path = fold_model_dir / "best.pt"
    training_start = start_timer()
    for epoch in range(1, EPOCHS + 1):
        epoch_start = start_timer()
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
        print_epoch_timing(epoch, EPOCHS, training_start, epoch_start)
        print_metrics("train", train_metrics)
        print_metrics("val", val_metrics)
        append_log(
            log_path, LOG_FIELDS, args.fold, epoch, "train", train_metrics
        )
        append_log(log_path, LOG_FIELDS, args.fold, epoch, "val", val_metrics)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "group_to_index": group_to_index,
            "num_groups": num_groups,
            "model_name": args.model_name,
            "feature_name": args.feat_name,
            "split_by": args.split_by,
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
        output_dir / f"fold{args.fold}_val.csv", index=False
    )
    test_predictions.to_csv(
        output_dir / f"fold{args.fold}_test.csv", index=False
    )
    print(f"Saved predictions to {output_dir}")
    print_fold_time(args.fold, training_start)


if __name__ == "__main__":
    main()
