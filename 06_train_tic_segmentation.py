"""Train and evaluate frame-level tic segmentation models."""

import math
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from bin.segmentation_datasets import SpecDataset
from bin.segmentation_metrics import get_segmentation_metrics
from bin.segmentation_models import BiLSTM, CNN, CNN_BiLSTM
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
MODEL_NAME = "BiLSTM"
SPLIT_BY = "session"
FEAT_NAME = "MFCC"
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
NUM_WORKERS = 0
WIN_LEN = 10
P_TICS = 0.2

MODEL_CLASSES = {
    "BiLSTM": BiLSTM,
    "CNN": CNN,
    "CNN_BiLSTM": CNN_BiLSTM,
}
LOG_FIELDS = [
    "fold",
    "epoch",
    "split",
    "loss",
    "frame_accuracy",
    "frame_f1",
    "frame_auroc",
    "segment_accuracy",
    "segment_f1",
]


def train_one_epoch(model, loader, optimizer, loss_function, device):
    """Run one segmentation optimization epoch."""
    model.train()
    for features, targets in loader:
        features = features.to(device)
        targets = targets.float().to(device)
        optimizer.zero_grad()
        logits = model(features)
        loss = loss_function(logits, targets)
        loss.backward()
        optimizer.step()


def evaluate(model, loader, loss_function, device, return_predictions=False):
    """Evaluate one split and return metrics plus frame predictions."""
    model.eval()
    total_loss = 0.0
    segment_count = 0
    all_logits = []
    all_targets = []

    with torch.inference_mode():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.float().to(device)
            logits = model(features)
            loss = loss_function(logits, targets)
            batch_size = targets.shape[0]
            total_loss += loss.item() * batch_size
            segment_count += batch_size
            all_logits.append(logits.cpu())
            all_targets.append(targets.bool().cpu())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    (
        frame_accuracy,
        frame_f1,
        frame_auroc,
        segment_accuracy,
        segment_f1,
    ) = get_segmentation_metrics(logits, targets)
    metrics = {
        "loss": total_loss / segment_count,
        "frame_accuracy": frame_accuracy,
        "frame_f1": frame_f1,
        "frame_auroc": frame_auroc,
        "segment_accuracy": segment_accuracy,
        "segment_f1": segment_f1,
    }
    prediction_table = None
    if return_predictions:
        prediction_table = make_prediction_table(logits, targets)
    return metrics, prediction_table


def make_prediction_table(logits, targets):
    """Create one output row per frame, retaining its segment membership."""
    probabilities = logits.sigmoid()
    predictions = probabilities >= 0.5
    num_segments, num_frames = targets.shape
    segment_real = targets.any(dim=1)
    segment_pred = predictions.any(dim=1)
    return pd.DataFrame(
        {
            "segment_id": torch.arange(num_segments)
            .repeat_interleave(num_frames)
            .numpy(),
            "frame_id": torch.arange(num_frames).repeat(num_segments).numpy(),
            "tic_real": targets.reshape(-1).numpy(),
            "tic_pred": predictions.reshape(-1).numpy(),
            "tic_probability": probabilities.reshape(-1).numpy(),
            "segment_real": segment_real.repeat_interleave(num_frames).numpy(),
            "segment_pred": segment_pred.repeat_interleave(num_frames).numpy(),
        }
    )


def main():
    args = parse_training_args(
        description="Train tic segmentation on one cross-validation fold.",
        model_classes=MODEL_CLASSES,
        default_model=MODEL_NAME,
        default_split=SPLIT_BY,
        default_feature=FEAT_NAME,
    )
    global_name = f"{args.model_name}_{args.feat_name}_by{args.split_by}"
    model_dir = Path("models/segmentation") / global_name
    output_dir = Path("outputs/segmentation") / global_name
    torch.manual_seed(42)

    fold = load_fold(SPLIT_PATH, args.fold)
    fold_model_dir = model_dir / f"fold{args.fold}"
    log_path = model_dir / f"fold{args.fold}.log"
    initialize_log(log_path, LOG_FIELDS)

    transform, input_dim = make_transform(args.feat_name)
    print("Train ", end="")
    train_dataset = SpecDataset(
        METADATA_PATH, fold["train"], transform, win_len=WIN_LEN, p_tics=P_TICS
    )
    print("Val ", end="")
    val_dataset = SpecDataset(
        METADATA_PATH, fold["val"], transform, win_len=WIN_LEN, p_tics=P_TICS
    )
    print("Test ", end="")
    test_dataset = SpecDataset(
        METADATA_PATH, fold["test"], transform, win_len=WIN_LEN, p_tics=P_TICS
    )

    train_loader, val_loader, test_loader = make_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MODEL_CLASSES[args.model_name](input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.BCEWithLogitsLoss()
    fold_model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_epoch = None
    best_path = fold_model_dir / "best.pt"
    training_start = start_timer()
    for epoch in range(1, EPOCHS + 1):
        epoch_start = start_timer()
        train_one_epoch(model, train_loader, optimizer, loss_function, device)
        train_metrics, _ = evaluate(model, train_loader, loss_function, device)
        val_metrics, _ = evaluate(model, val_loader, loss_function, device)
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
            "model_name": args.model_name,
            "feature_name": args.feat_name,
            "split_by": args.split_by,
            "fold": args.fold,
            "val_loss": val_metrics["loss"],
            "val_frame_auroc": val_metrics["frame_auroc"],
        }
        torch.save(checkpoint, fold_model_dir / f"{epoch}.pt")

        val_loss = val_metrics["loss"]
        if best_epoch is None or (
            math.isfinite(val_loss) and val_loss < best_loss
        ):
            torch.save(checkpoint, best_path)
            best_epoch = epoch
            if math.isfinite(val_loss):
                best_loss = val_loss
            print(
                f"Saved new best checkpoint to {best_path} "
                f"(validation loss: {val_loss:.4f})"
            )

    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    val_metrics, val_predictions = evaluate(
        model, val_loader, loss_function, device, return_predictions=True
    )
    print(f"\nBest epoch: {best_checkpoint['epoch']}")
    print_metrics("best val", val_metrics)

    test_metrics, test_predictions = evaluate(
        model, test_loader, loss_function, device, return_predictions=True
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
