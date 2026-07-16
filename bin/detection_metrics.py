"""Metrics for clip-level tic detection and group classification."""

import torch


def _class_predictions(predictions):
    """Convert class logits or predicted labels to a flat label tensor."""
    predictions = torch.as_tensor(predictions)
    if predictions.ndim > 1:
        predictions = predictions.argmax(dim=-1)
    return predictions.reshape(-1).long()


def _binary_predictions(predictions):
    """Return binary labels and positive-class scores."""
    predictions = torch.as_tensor(predictions)
    if predictions.ndim > 1:
        if predictions.shape[-1] != 2:
            raise ValueError("Binary logits must have two classes")
        scores = predictions.float().softmax(dim=-1)[..., 1].reshape(-1)
        labels = predictions.argmax(dim=-1).reshape(-1).long()
    else:
        scores = predictions.reshape(-1).float()
        if scores.numel() and (scores.min() < 0 or scores.max() > 1):
            scores = scores.sigmoid()
        labels = (scores >= 0.5).long()
    return labels, scores


def _f1(predictions, targets, label):
    """Calculate the F1 score for one class label."""
    true_positive = ((predictions == label) & (targets == label)).sum()
    false_positive = ((predictions == label) & (targets != label)).sum()
    false_negative = ((predictions != label) & (targets == label)).sum()
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return (2 * true_positive / denominator).item()


def _binary_auroc(scores, targets):
    """Calculate binary AUROC using average ranks, including tied scores."""
    positive_count = (targets == 1).sum().item()
    negative_count = (targets == 0).sum().item()
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    _, inverse, counts = torch.unique(
        scores, sorted=True, return_inverse=True, return_counts=True
    )
    rank_ends = counts.cumsum(dim=0).float()
    rank_starts = rank_ends - counts + 1
    average_ranks = (rank_starts + rank_ends) / 2
    ranks = average_ranks[inverse]
    positive_rank_sum = ranks[targets == 1].sum()
    auroc = (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)
    return auroc.item()


def get_group_metrics(group_pred, group_real):
    """Return accuracy and macro F1 for tic-group predictions.

    Multi-hot targets use all-zero rows for non-tic samples. Legacy class-index
    targets equal to ``-1`` are also supported.
    """
    raw_predictions = torch.as_tensor(group_pred)
    raw_targets = torch.as_tensor(group_real).to(raw_predictions.device)
    if raw_targets.ndim > 1:
        if raw_predictions.shape != raw_targets.shape:
            raise ValueError("group_pred and group_real must have the same shape")
        scores = raw_predictions.float()
        if raw_predictions.dtype.is_floating_point:
            scores = scores.sigmoid()
        predictions = scores >= 0.5
        targets = raw_targets > 0.5
        tic_mask = targets.any(dim=1)
        predictions = predictions[tic_mask]
        targets = targets[tic_mask]
        if targets.numel() == 0:
            raise ValueError("No tic-group targets were provided")

        accuracy = (predictions == targets).float().mean().item()
        active_groups = (predictions | targets).any(dim=0)
        f1_scores = [
            _f1(predictions[:, index], targets[:, index], label=1)
            for index in torch.where(active_groups)[0]
        ]
        macro_f1 = sum(f1_scores) / len(f1_scores)
        return accuracy, macro_f1

    predictions = _class_predictions(group_pred)
    targets = torch.as_tensor(group_real).reshape(-1).long().to(predictions.device)
    if len(predictions) != len(targets):
        raise ValueError("group_pred and group_real must have the same length")

    tic_mask = targets != -1
    predictions = predictions[tic_mask]
    targets = targets[tic_mask]
    if targets.numel() == 0:
        raise ValueError("No tic-group targets were provided")

    accuracy = (predictions == targets).float().mean().item()
    labels = torch.unique(torch.cat((predictions, targets)))
    labels = labels[labels != -1]
    macro_f1 = sum(_f1(predictions, targets, label) for label in labels) / len(
        labels
    )
    return accuracy, macro_f1


def get_tic_metrics(tic_pred, tic_real):
    """Return accuracy, F1, AUROC, precision, and recall for tic prediction."""
    predictions, scores = _binary_predictions(tic_pred)
    targets = torch.as_tensor(tic_real).reshape(-1).long().to(predictions.device)
    if len(predictions) != len(targets):
        raise ValueError("tic_pred and tic_real must have the same length")
    if targets.numel() == 0:
        raise ValueError("No tic targets were provided")
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("tic_real must contain only 0 and 1")

    true_positive = ((predictions == 1) & (targets == 1)).sum().item()
    false_positive = ((predictions == 1) & (targets == 0)).sum().item()
    false_negative = ((predictions == 0) & (targets == 1)).sum().item()

    accuracy = (predictions == targets).float().mean().item()
    f1 = _f1(predictions, targets, label=1)
    auroc = _binary_auroc(scores, targets)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return accuracy, f1, auroc, precision, recall
