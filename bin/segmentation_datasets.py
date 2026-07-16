"""Datasets for frame-level tic segmentation."""

import math

import pandas as pd
import torch

from bin.detection_datasets import (
    Detection_Dataset,
    SUPPORTED_TRANSFORMS,
    _load_audio_window,
    _load_embedding_window,
)


class Segmentation_Dataset(Detection_Dataset):
    """Common sampling and frame-target logic for segmentation datasets."""

    def __init__(
        self,
        metadata_file,
        participant_phase_sessions,
        win_len=10,
        p_tics=0.2,
        include_multigroup=True,
    ):
        super().__init__(
            metadata_file,
            participant_phase_sessions,
            win_len,
            p_tics,
            include_multigroup,
        )

    def __getitem__(self, index):
        """Return features and boolean tic-presence labels over feature time."""
        self._validate_index(index)
        row, window_start, _, _, _ = self._sample_window()
        features = self._load_features(row, window_start)
        labels = self._frame_labels(
            audio_path=row["AudioPath"],
            window_start=window_start,
            num_frames=features.shape[-1],
        )
        return features, labels

    def _frame_labels(self, audio_path, window_start, num_frames):
        """Mark frames covered by any tic annotation in the sampled window."""
        labels = torch.zeros(num_frames, dtype=torch.bool)
        window_end = window_start + self.win_len
        same_audio = self.tics["AudioPath"].astype(str) == str(audio_path)
        overlapping = self.tics.loc[
            same_audio
            & (self.tics["EndTime"].astype(float) > window_start)
            & (self.tics["StartTime"].astype(float) < window_end)
        ]

        for _, tic in overlapping.iterrows():
            relative_start = max(0.0, float(tic["StartTime"]) - window_start)
            relative_end = min(self.win_len, float(tic["EndTime"]) - window_start)
            first_frame = int(relative_start / self.win_len * num_frames)
            last_frame = math.ceil(relative_end / self.win_len * num_frames)
            first_frame = min(first_frame, num_frames - 1)
            last_frame = max(first_frame + 1, min(last_frame, num_frames))
            labels[first_frame:last_frame] = True
        return labels


class SpecDataset(Segmentation_Dataset):
    """Segmentation dataset computing spectrogram features from PCM audio."""

    def __init__(
        self,
        metadata_file,
        participant_phase_sessions,
        transform,
        win_len=10,
        p_tics=0.2,
        include_multigroup=True,
    ):
        if not isinstance(transform, SUPPORTED_TRANSFORMS):
            raise TypeError(
                "transform must be a torchaudio Spectrogram, "
                "MelSpectrogram, or MFCC"
            )
        self.transform = transform
        super().__init__(
            metadata_file,
            participant_phase_sessions,
            win_len,
            p_tics,
            include_multigroup,
        )

    def _load_features(self, row, window_start):
        waveform = _load_audio_window(
            row["AudioPath"], window_start, self.win_len
        )
        return self.transform(waveform)


class WavLmDataset(Segmentation_Dataset):
    """Segmentation dataset loading pre-computed recording-level WavLM tensors."""

    def __init__(
        self,
        metadata_file,
        participant_phase_sessions,
        win_len=10,
        p_tics=0.2,
        include_multigroup=True,
        frames_per_second=50,
    ):
        if frames_per_second <= 0:
            raise ValueError("frames_per_second must be greater than zero")
        self.frames_per_second = float(frames_per_second)
        super().__init__(
            metadata_file,
            participant_phase_sessions,
            win_len,
            p_tics,
            include_multigroup,
        )
        if "embedding_path" not in self.metadata.columns:
            raise ValueError("WavLM metadata must contain an embedding_path column")

    def _load_features(self, row, window_start):
        embedding_path = row["embedding_path"]
        if pd.isna(embedding_path):
            raise ValueError(f"Missing embedding_path for {row['AudioPath']}")
        return _load_embedding_window(
            embedding_path=embedding_path,
            window_start=window_start,
            win_len=self.win_len,
            audio_duration=self.audio_durations[row["AudioPath"]],
            frames_per_second=self.frames_per_second,
        )
