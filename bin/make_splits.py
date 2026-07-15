"""Create reproducible cross-validation splits for SpecDataset."""

import json
import math
from pathlib import Path
import random


REQUIRED_COLUMNS = {"ID", "Sess", "Phase", "AudioPath"}


def _recordings(dataframe):
    """Return unique recordings as (participant, phase, session, audio path)."""
    missing = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing dataframe columns: {sorted(missing)}")

    recordings = dataframe[["ID", "Phase", "Sess", "AudioPath"]].drop_duplicates()
    recordings = [
        (str(row.ID).upper(), str(row.Phase).upper(), int(row.Sess), row.AudioPath)
        for row in recordings.itertuples(index=False)
    ]
    if not recordings:
        raise ValueError("The dataframe does not contain any recordings")
    recording_keys = [_recording_key(recording) for recording in recordings]
    if len(recording_keys) != len(set(recording_keys)):
        raise ValueError(
            "Each (participant, phase, session) must identify exactly one audio file"
        )
    return recordings


def _split_groups(items, k, seed):
    """Shuffle items and divide them into K balanced groups."""
    if k < 2:
        raise ValueError("K must be at least 2")
    if k > len(items):
        raise ValueError(f"K={k} is larger than the {len(items)} split units")

    items = list(items)
    random.Random(seed).shuffle(items)
    return [items[index::k] for index in range(k)]


def _recording_key(recording):
    """Return the tuple format expected by SpecDataset."""
    participant, phase, session, _ = recording
    return participant, phase, session


def _train_validation_split(recordings, seed):
    """Place approximately 20% of non-test recordings in validation."""
    recordings = list(recordings)
    random.Random(seed).shuffle(recordings)
    if len(recordings) <= 1:
        validation_count = 0
    else:
        validation_count = min(math.ceil(0.2 * len(recordings)), len(recordings) - 1)
    validation = recordings[:validation_count]
    train = recordings[validation_count:]
    return train, validation


def _build_folds(recordings, units, unit_for_recording, k, seed):
    """Build train, validation, and test lists from arbitrary split units."""
    test_groups = _split_groups(units, k, seed)
    folds = {}

    for fold_number, test_group in enumerate(test_groups, start=1):
        test_units = set(test_group)
        test_recordings = [
            recording
            for recording in recordings
            if unit_for_recording(recording) in test_units
        ]
        remaining = [
            recording
            for recording in recordings
            if unit_for_recording(recording) not in test_units
        ]
        train_recordings, validation_recordings = _train_validation_split(
            remaining, seed + fold_number
        )
        folds[fold_number] = {
            "train": sorted(_recording_key(item) for item in train_recordings),
            "val": sorted(_recording_key(item) for item in validation_recordings),
            "test": sorted(_recording_key(item) for item in test_recordings),
        }

    return folds


def splits_by_participant(dataframe, K=5, seed=42):
    """Create K folds that hold out complete participants for testing."""
    recordings = _recordings(dataframe)
    participants = sorted({recording[0] for recording in recordings})
    return _build_folds(
        recordings,
        participants,
        unit_for_recording=lambda recording: recording[0],
        k=K,
        seed=seed,
    )


def splits_by_session(dataframe, K=5, seed=42):
    """Create K folds that hold out participant-session pairs for testing."""
    recordings = _recordings(dataframe)
    sessions = sorted({(recording[0], recording[2]) for recording in recordings})
    return _build_folds(
        recordings,
        sessions,
        unit_for_recording=lambda recording: (recording[0], recording[2]),
        k=K,
        seed=seed,
    )


def splits_by_file(dataframe, K=5, seed=42):
    """Create K folds that hold out individual audio files for testing."""
    recordings = _recordings(dataframe)
    audio_paths = sorted({recording[3] for recording in recordings})
    return _build_folds(
        recordings,
        audio_paths,
        unit_for_recording=lambda recording: recording[3],
        k=K,
        seed=seed,
    )


def save_split(split, json_file):
    """Save a fold dictionary to a JSON file."""
    json_file = Path(json_file)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with json_file.open("w", encoding="utf-8") as file:
        json.dump(split, file, indent=2)


def load_split(json_file):
    """Load a fold dictionary and restore integer keys and recording tuples."""
    with Path(json_file).open("r", encoding="utf-8") as file:
        saved_split = json.load(file)

    return {
        int(fold): {
            name: [
                (str(participant), str(phase), int(session))
                for participant, phase, session in recordings
            ]
            for name, recordings in fold_splits.items()
        }
        for fold, fold_splits in saved_split.items()
    }
