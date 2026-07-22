"""Frame-wise and segment-wise metrics for tic segmentation."""

import torch


def _predictions_and_scores(predictions, threshold, from_logits):
    """Convert logits, probabilities, or binary values to labels and scores."""
    predictions = torch.as_tensor(predictions)
    if predictions.dtype == torch.bool or not predictions.dtype.is_floating_point:
        labels = predictions.bool()
        scores = labels.float()
    else:
        scores = predictions.float()
        if from_logits:
            scores = scores.sigmoid()
        labels = scores >= threshold
    return labels, scores


def _validate_targets(targets, expected_shape, device):
    """Return binary targets after validating their shape and values."""
    targets = torch.as_tensor(targets)
    if targets.shape != expected_shape:
        raise ValueError("predictions and targets must have the same shape")
    if targets.numel() == 0:
        raise ValueError("No segmentation targets were provided")
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("targets must contain only boolean, 0, or 1 values")
    return targets.bool().to(device)


def _f1(predictions, targets):
    """Calculate binary F1 with tic presence as the positive class."""
    true_positive = (predictions & targets).sum()
    false_positive = (predictions & ~targets).sum()
    false_negative = (~predictions & targets).sum()
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return (2 * true_positive / denominator).item()


def _binary_auroc(scores, targets):
    """Calculate binary AUROC using average ranks for tied scores."""
    scores = scores.reshape(-1)
    targets = targets.reshape(-1)
    positive_count = targets.sum().item()
    negative_count = (~targets).sum().item()
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    _, inverse, counts = torch.unique(
        scores, sorted=True, return_inverse=True, return_counts=True
    )
    rank_ends = counts.cumsum(dim=0).float()
    rank_starts = rank_ends - counts + 1
    average_ranks = (rank_starts + rank_ends) / 2
    positive_rank_sum = average_ranks[inverse][targets].sum()
    auroc = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)
    return auroc.item()


def get_frame_metrics(predictions, targets, threshold=0.5, from_logits=True):
    """Return frame-wise accuracy, F1, and AUROC."""
    frame_predictions, frame_scores = _predictions_and_scores(
        predictions, threshold, from_logits
    )
    frame_targets = _validate_targets(
        targets, frame_predictions.shape, frame_predictions.device
    )
    accuracy = (frame_predictions == frame_targets).float().mean().item()
    f1 = _f1(frame_predictions, frame_targets)
    auroc = _binary_auroc(frame_scores, frame_targets)
    return accuracy, f1, auroc


def _segment_labels(frame_predictions, frame_targets, N_percent):
    """Reduce frame labels to predicted and real segment labels."""
    positive_fraction = frame_predictions.reshape(
        frame_predictions.shape[0], -1
    ).float().mean(dim=1)
    segment_predictions = positive_fraction > N_percent / 100
    segment_targets = frame_targets.reshape(frame_targets.shape[0], -1).any(dim=1)
    return segment_predictions, segment_targets


def find_best_segment_percent(
    predictions, targets, threshold=0.5, from_logits=True
):
    """Return the integer percentage from 0 to 100 with the best segment F1."""
    frame_predictions, _ = _predictions_and_scores(
        predictions, threshold, from_logits
    )
    frame_targets = _validate_targets(
        targets, frame_predictions.shape, frame_predictions.device
    )
    if frame_predictions.ndim == 1:
        frame_predictions = frame_predictions.unsqueeze(0)
        frame_targets = frame_targets.unsqueeze(0)

    best_percent = 0
    best_f1 = -1.0
    for percent in range(101):
        segment_predictions, segment_targets = _segment_labels(
            frame_predictions, frame_targets, percent
        )
        f1 = _f1(segment_predictions, segment_targets)
        if f1 > best_f1:
            best_percent = percent
            best_f1 = f1
    return best_percent


def get_segment_metrics(
    predictions, targets, threshold=0.5, from_logits=True, N_percent=5
):
    """Return segment accuracy and F1 using a positive-frame percentage."""
    if N_percent == -1:
        N_percent = find_best_segment_percent(
            predictions, targets, threshold=threshold, from_logits=from_logits
        )
    elif not 0 <= N_percent <= 100:
        raise ValueError("N_percent must be -1 or between 0 and 100")
    frame_predictions, _ = _predictions_and_scores(
        predictions, threshold, from_logits
    )
    frame_targets = _validate_targets(
        targets, frame_predictions.shape, frame_predictions.device
    )
    if frame_predictions.ndim == 1:
        frame_predictions = frame_predictions.unsqueeze(0)
        frame_targets = frame_targets.unsqueeze(0)
    segment_predictions, segment_targets = _segment_labels(
        frame_predictions, frame_targets, N_percent
    )
    accuracy = (segment_predictions == segment_targets).float().mean().item()
    f1 = _f1(segment_predictions, segment_targets)
    return accuracy, f1


def get_segmentation_metrics(
    predictions, targets, threshold=0.5, from_logits=True, N_percent=5
):
    """Return frame accuracy/F1/AUROC followed by segment accuracy/F1."""
    frame_accuracy, frame_f1, frame_auroc = get_frame_metrics(
        predictions, targets, threshold=threshold, from_logits=from_logits
    )
    segment_accuracy, segment_f1 = get_segment_metrics(
        predictions,
        targets,
        threshold=threshold,
        from_logits=from_logits,
        N_percent=N_percent,
    )
    return frame_accuracy, frame_f1, frame_auroc, segment_accuracy, segment_f1
