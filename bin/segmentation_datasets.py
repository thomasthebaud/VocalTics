"""Datasets for frame-level tic segmentation."""

import math

import torch

from bin.detection_datasets import SpecDataset as DetectionSpecDataset


class SpecDataset(DetectionSpecDataset):
    """Return transformed windows and one boolean tic label per feature frame."""

    def __init__(
        self,
        metadata_file,
        participant_phase_sessions,
        transform,
        win_len=10,
        p_tics=0.2,
        include_multigroup=True,
    ):
        super().__init__(
            metadata_file=metadata_file,
            participant_phase_sessions=participant_phase_sessions,
            transform=transform,
            win_len=win_len,
            p_tics=p_tics,
            include_multigroup=include_multigroup,
        )

    def __getitem__(self, index):
        """Return features and boolean tic-presence labels over feature time."""
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("dataset index out of range")

        row, window_start, _, _, _ = self._sample_window()
        waveform = self._load_window(row["AudioPath"], window_start)
        features = self.transform(waveform)
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
