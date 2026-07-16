"""Datasets for clip-level tic detection and group classification."""

from array import array
from pathlib import Path
import sys
import wave

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


SUPPORTED_TRANSFORMS = (
    torchaudio.transforms.Spectrogram,
    torchaudio.transforms.MelSpectrogram,
    torchaudio.transforms.MFCC,
)
REQUIRED_COLUMNS = {
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


def _pcm_to_tensor(data, sample_width, channels):
    """Convert little-endian PCM WAV bytes to [channels, frames]."""
    if not data:
        return torch.empty((channels, 0), dtype=torch.float32)
    if sample_width == 1:
        values = array("B")
        values.frombytes(data)
        waveform = (torch.tensor(values, dtype=torch.float32) - 128) / 128
    elif sample_width == 2:
        values = array("h")
        values.frombytes(data)
        if sys.byteorder == "big":
            values.byteswap()
        waveform = torch.tensor(values, dtype=torch.float32) / 32768
    elif sample_width == 3:
        values = [
            int.from_bytes(data[index : index + 3], "little", signed=True)
            for index in range(0, len(data), 3)
        ]
        waveform = torch.tensor(values, dtype=torch.float32) / 8388608
    elif sample_width == 4:
        values = array("i")
        values.frombytes(data)
        if sys.byteorder == "big":
            values.byteswap()
        waveform = torch.tensor(values, dtype=torch.float32) / 2147483648
    else:
        raise ValueError(f"Unsupported PCM sample width: {sample_width} bytes")
    return waveform.reshape(-1, channels).transpose(0, 1)


def _load_audio_window(audio_path, window_start, win_len):
    """Load a fixed-length mono PCM window and pad at file boundaries."""
    with wave.open(str(audio_path), "rb") as audio_file:
        if audio_file.getcomptype() != "NONE":
            raise ValueError(f"Compressed WAV is not supported: {audio_path}")
        sample_rate = audio_file.getframerate()
        channels = audio_file.getnchannels()
        sample_width = audio_file.getsampwidth()
        total_frames = audio_file.getnframes()
        window_frames = round(win_len * sample_rate)
        requested_start = round(window_start * sample_rate)
        frame_offset = max(0, min(requested_start, total_frames))
        left_padding = max(0, -requested_start)
        frames_to_load = max(0, window_frames - left_padding)
        audio_file.setpos(frame_offset)
        data = audio_file.readframes(frames_to_load)

    waveform = _pcm_to_tensor(data, sample_width, channels)
    waveform = waveform.mean(dim=0, keepdim=True)
    right_padding = window_frames - left_padding - waveform.shape[-1]
    waveform = torch.nn.functional.pad(
        waveform, (left_padding, max(0, right_padding))
    )
    return waveform[..., :window_frames]


def _load_embedding_window(
    embedding_path,
    window_start,
    win_len,
    audio_duration,
    frames_per_second,
):
    """Load a fixed-length WavLM window as [embedding_dim, time]."""
    embeddings = torch.load(embedding_path, map_location="cpu")
    if embeddings.ndim == 3 and embeddings.shape[0] == 1:
        embeddings = embeddings.squeeze(0)
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D WavLM tensor, got {tuple(embeddings.shape)} "
            f"from {embedding_path}"
        )
    if audio_duration <= 0:
        raise ValueError(f"Invalid audio duration for {embedding_path}")

    total_frames = embeddings.shape[0]
    window_frames = round(win_len * frames_per_second)
    requested_start = round(window_start / audio_duration * total_frames)
    frame_offset = max(0, min(requested_start, total_frames))
    left_padding = max(0, -requested_start)
    frames_to_load = max(0, window_frames - left_padding)
    window = embeddings[frame_offset : frame_offset + frames_to_load]
    right_padding = window_frames - left_padding - window.shape[0]
    window = torch.nn.functional.pad(
        window, (0, 0, left_padding, max(0, right_padding))
    )
    return window[:window_frames].transpose(0, 1).contiguous()


class Detection_Dataset(Dataset):
    """Common sampling and target logic for detection feature datasets."""

    def __init__(
        self,
        metadata_file,
        participant_phase_sessions,
        win_len=10,
        p_tics=0.5,
        include_multigroup=True,
    ):
        if win_len <= 0:
            raise ValueError("win_len must be greater than zero")
        if not 0 <= p_tics <= 1:
            raise ValueError("p_tics must be between zero and one")

        metadata = pd.read_csv(
            Path(metadata_file), dtype={"Type": str, "Group": str}
        )
        missing = REQUIRED_COLUMNS - set(metadata.columns)
        if missing:
            raise ValueError(f"Missing metadata columns: {sorted(missing)}")

        all_groups = sorted(
            {
                group
                for value in metadata["Group"].dropna()
                for group in str(value).split("+")
                if group != "-1"
            }
        )
        self.group_to_index = {
            group: index for index, group in enumerate(all_groups)
        }
        self.index_to_group = {
            index: group for group, index in self.group_to_index.items()
        }
        self.num_groups = len(all_groups)

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
        self.audio_durations = (
            self.metadata.groupby("AudioPath")["EndTime"]
            .max()
            .astype(float)
            .to_dict()
        )
        if not include_multigroup:
            multigroup = (
                (self.metadata["tic/nontic"] == "tic")
                & self.metadata["Group"]
                .fillna("-1")
                .str.contains("+", regex=False)
            )
            self.metadata = self.metadata.loc[~multigroup].reset_index(drop=True)

        self.win_len = float(win_len)
        self.p_tics = float(p_tics)
        self.include_multigroup = bool(include_multigroup)
        self.tics = self.metadata.loc[
            self.metadata["tic/nontic"] == "tic"
        ].reset_index(drop=True)
        self.nontics = self.metadata.loc[
            (self.metadata["tic/nontic"] == "nontic")
            & (self.metadata["Duration"] >= win_len)
        ].reset_index(drop=True)

        tic_types = {
            tic_type
            for value in self.tics["Type"].dropna()
            for tic_type in str(value).split("+")
            if tic_type != "-1"
        }
        tic_groups = {
            group
            for value in self.tics["Group"].dropna()
            for group in str(value).split("+")
            if group != "-1"
        }
        self.num_types = len(tic_types)
        self.num_groups_available = len(tic_groups)

        if self.metadata.empty:
            raise ValueError("No metadata rows match the requested recordings")
        if self.p_tics > 0 and self.tics.empty:
            raise ValueError("No tic segments match the requested recordings")
        if self.p_tics < 1 and self.nontics.empty:
            raise ValueError(
                f"No non-tic intervals are at least {self.win_len} seconds long"
            )
        print(
            f"Split contains {self.num_groups_available} tic groups "
            f"and {self.num_types} tic types"
        )

    def __len__(self):
        """Use the selected metadata size as the number of samples per epoch."""
        return len(self.metadata)

    def __getitem__(self, index):
        """Return features, tic type, group target, and tic presence."""
        self._validate_index(index)
        row, window_start, tic_type, group_target, has_tic = self._sample_window()
        features = self._load_features(row, window_start)
        return features, tic_type, group_target, has_tic

    def _load_features(self, row, window_start):
        raise NotImplementedError

    def _validate_index(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("dataset index out of range")

    def _sample_window(self):
        """Select a tic or non-tic row and return its window information."""
        has_tic = torch.rand(1).item() < self.p_tics
        if has_tic:
            row = self._random_row(self.tics)
            center = (float(row["StartTime"]) + float(row["EndTime"])) / 2
            window_start = center - self.win_len / 2
            tic_type = row["Type"]
            group_target = self._group_target(row["Group"])
        else:
            row = self._random_row(self.nontics)
            first_start = float(row["StartTime"])
            last_start = float(row["EndTime"]) - self.win_len
            window_start = first_start + torch.rand(1).item() * (
                last_start - first_start
            )
            tic_type = "-1"
            group_target = torch.zeros(self.num_groups, dtype=torch.float32)
        return row, window_start, tic_type, group_target, has_tic

    def _group_target(self, value):
        """Convert a '+'-separated group label to a multi-hot vector."""
        target = torch.zeros(self.num_groups, dtype=torch.float32)
        for group in str(value).split("+"):
            if group != "-1":
                target[self.group_to_index[group]] = 1
        return target

    @staticmethod
    def _random_row(rows):
        """Select one metadata row using PyTorch's worker-aware RNG."""
        position = torch.randint(len(rows), (1,)).item()
        return rows.iloc[position]


class SpecDataset(Detection_Dataset):
    """Detection dataset computing spectrogram features from PCM audio."""

    def __init__(
        self,
        metadata_file,
        participant_phase_sessions,
        transform,
        win_len=10,
        p_tics=0.5,
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


class WavLmDataset(Detection_Dataset):
    """Detection dataset loading pre-computed recording-level WavLM tensors."""

    def __init__(
        self,
        metadata_file,
        participant_phase_sessions,
        win_len=10,
        p_tics=0.5,
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
