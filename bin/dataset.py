"""PyTorch datasets for VocalTics experiments."""

from pathlib import Path

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


SUPPORTED_TRANSFORMS = (
    torchaudio.transforms.Spectrogram,
    torchaudio.transforms.MelSpectrogram,
    torchaudio.transforms.MFCC,
)


class SpecDataset(Dataset):
    """Randomly sample fixed-length tic or non-tic spectrogram windows.

    ``participant_phase_sessions`` must contain ``(ID, Phase, Sess)`` tuples,
    for example ``[("DET0101", "NO", 1), ("DET0102", "HI", 2)]``.
    """

    def __init__(
        self,
        metadata_file,
        participant_phase_sessions,
        transform,
        win_len=10,
        p_tics=0.5,
    ):
        if not isinstance(transform, SUPPORTED_TRANSFORMS):
            raise TypeError(
                "transform must be a torchaudio Spectrogram, "
                "MelSpectrogram, or MFCC"
            )
        if win_len <= 0:
            raise ValueError("win_len must be greater than zero")
        if not 0 <= p_tics <= 1:
            raise ValueError("p_tics must be between zero and one")

        metadata = pd.read_csv(
            Path(metadata_file), dtype={"Type": str, "Group": str}
        )
        required_columns = {
            "AudioPath",
            "ID",
            "Sess",
            "Phase",
            "tic/nontic",
            "Type",
            "Group",
            "StartTime",
            "EndTime",
            "Duration",
        }
        missing = required_columns - set(metadata.columns)
        if missing:
            raise ValueError(f"Missing metadata columns: {sorted(missing)}")

        selections = {
            (str(participant).upper(), str(phase).upper(), int(session))
            for participant, phase, session in participant_phase_sessions
        }
        metadata["ID"] = metadata["ID"].astype(str).str.upper()
        metadata["Phase"] = metadata["Phase"].astype(str).str.upper()
        metadata["Sess"] = metadata["Sess"].astype(int)
        selected = metadata.apply(
            lambda row: (row["ID"], row["Phase"], row["Sess"]) in selections,
            axis=1,
        )
        self.metadata = metadata.loc[selected].reset_index(drop=True)
        self.tics = self.metadata.loc[
            self.metadata["tic/nontic"] == "tic"
        ].reset_index(drop=True)
        self.nontics = self.metadata.loc[
            (self.metadata["tic/nontic"] == "nontic")
            & (self.metadata["Duration"] >= win_len)
        ].reset_index(drop=True)
        self.transform = transform
        self.win_len = float(win_len)
        self.p_tics = float(p_tics)

        tic_types = {
            tic_type
            for value in self.tics["Type"].dropna()
            for tic_type in str(value).split("+")
            if tic_type != "-1"
        }
        tic_groups = {
            str(group)
            for group in self.tics["Group"].dropna()
            if str(group) != "-1"
        }
        self.num_types = len(tic_types)
        self.num_groups = len(tic_groups)

        if self.metadata.empty:
            raise ValueError("No metadata rows match the requested recordings")
        if self.p_tics > 0 and self.tics.empty:
            raise ValueError("No tic segments match the requested recordings")
        if self.p_tics < 1 and self.nontics.empty:
            raise ValueError(
                f"No non-tic intervals are at least {self.win_len} seconds long"
            )
        print(
            f"Split contains {self.num_groups} tic groups "
            f"and {self.num_types} tic types"
        )

    def __len__(self):
        """Use the selected metadata size as the number of samples per epoch."""
        return len(self.metadata)

    def __getitem__(self, index):
        """Return features, tic type, tic group, and tic presence."""
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("dataset index out of range")

        has_tic = torch.rand(1).item() < self.p_tics
        if has_tic:
            row = self._random_row(self.tics)
            center = (float(row["StartTime"]) + float(row["EndTime"])) / 2
            window_start = center - self.win_len / 2
            tic_type = row["Type"]
            tic_group = row["Group"]
        else:
            row = self._random_row(self.nontics)
            first_start = float(row["StartTime"])
            last_start = float(row["EndTime"]) - self.win_len
            window_start = first_start + torch.rand(1).item() * (
                last_start - first_start
            )
            tic_type = "-1"
            tic_group = "-1"

        waveform = self._load_window(row["AudioPath"], window_start)
        features = self.transform(waveform)
        return features, tic_type, tic_group, has_tic

    @staticmethod
    def _random_row(rows):
        """Select one metadata row using PyTorch's worker-aware RNG."""
        position = torch.randint(len(rows), (1,)).item()
        return rows.iloc[position]

    def _load_window(self, audio_path, window_start):
        """Load a fixed-length mono window and zero-pad at file boundaries."""
        info = torchaudio.info(audio_path)
        sample_rate = info.sample_rate
        window_frames = round(self.win_len * sample_rate)
        requested_start = round(window_start * sample_rate)
        frame_offset = max(0, requested_start)
        left_padding = max(0, -requested_start)
        frames_to_load = max(0, window_frames - left_padding)

        waveform, loaded_sample_rate = torchaudio.load(
            audio_path,
            frame_offset=frame_offset,
            num_frames=frames_to_load,
        )
        if loaded_sample_rate != sample_rate:
            raise ValueError(f"Unexpected sample rate while loading {audio_path}")

        waveform = waveform.mean(dim=0, keepdim=True)
        right_padding = window_frames - left_padding - waveform.shape[-1]
        waveform = torch.nn.functional.pad(
            waveform, (left_padding, max(0, right_padding))
        )
        return waveform[..., :window_frames]
