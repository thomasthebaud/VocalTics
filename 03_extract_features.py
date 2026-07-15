"""Extract WavLM-large embeddings for every audio recording."""

from pathlib import Path

import pandas as pd
import torch
import torchaudio
from transformers import AutoFeatureExtractor, WavLMModel


DATA_ROOT = Path("/projects/vocaltics/data/")
METADATA_PATH = DATA_ROOT / "metadata.csv"
OUTPUT_ROOT = Path("/project/vocaltics/data/wavlm_embeddings/")
OUTPUT_METADATA_PATH = OUTPUT_ROOT / "metadata.csv"
MODEL_NAME = "microsoft/wavlm-large"
WINDOW_SECONDS = 30


def load_audio(path, target_sample_rate):
    """Load a WAV file as mono audio at the model's sampling rate."""
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.mean(dim=0)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, target_sample_rate
        )
    return waveform


def extract_embeddings(waveform, model, feature_extractor, device):
    """Extract and concatenate contextual embeddings from 30-second windows."""
    sample_rate = feature_extractor.sampling_rate
    window_samples = WINDOW_SECONDS * sample_rate
    embeddings = []

    for start in range(0, waveform.numel(), window_samples):
        window = waveform[start : start + window_samples]
        inputs = feature_extractor(
            window.numpy(), sampling_rate=sample_rate, return_tensors="pt"
        )
        input_values = inputs.input_values.to(device)
        with torch.inference_mode():
            output = model(input_values).last_hidden_state
        embeddings.append(output.squeeze(0).cpu())

    if not embeddings:
        raise ValueError("Cannot extract embeddings from an empty audio file")
    return torch.cat(embeddings, dim=0)


def main():
    # Load the segment metadata while preserving tic labels as strings.
    metadata = pd.read_csv(METADATA_PATH, dtype={"Type": str, "Group": str})
    # Select a GPU when one is available, otherwise use the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the input processor associated with WavLM-large.
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
    # Load WavLM-large, switch off training behavior, and move it to the device.
    model = WavLMModel.from_pretrained(MODEL_NAME).eval().to(device)
    # Create the root directory that will contain embeddings and new metadata.
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    # Store the output path associated with each unique source recording.
    embedding_paths = {}

    # Process each full audio file only once, even though metadata has many segments.
    for audio_path, rows in metadata.groupby("AudioPath", sort=False):
        # Read the participant ID associated with this recording.
        participant = str(rows["ID"].iloc[0])
        # Create one output folder per participant.
        participant_dir = OUTPUT_ROOT / participant
        # Ensure that the participant output folder exists.
        participant_dir.mkdir(parents=True, exist_ok=True)
        # Build the output filename from the original WAV filename.
        embedding_path = participant_dir / f"{Path(audio_path).stem}.pt"
        # Load and standardize the complete recording.
        waveform = load_audio(audio_path, feature_extractor.sampling_rate)
        # Extract each 30-second window and concatenate its feature frames.
        embeddings = extract_embeddings(
            waveform, model, feature_extractor, device
        )
        # Save the concatenated feature tensor on CPU.
        torch.save(embeddings, embedding_path)
        # Remember the path so every segment from this recording can reference it.
        embedding_paths[audio_path] = str(embedding_path)
        # Print progress after completing the recording.
        print(f"Saved {embedding_path} with shape {tuple(embeddings.shape)}")

    # Add the recording-level embedding path to every corresponding segment row.
    metadata["embedding_path"] = metadata["AudioPath"].map(embedding_paths)
    # Save a copy of the metadata beside the extracted embeddings.
    metadata.to_csv(OUTPUT_METADATA_PATH, index=False)
    # Report where the completed metadata was saved.
    print(f"Saved metadata to {OUTPUT_METADATA_PATH}")


if __name__ == "__main__":
    main()
