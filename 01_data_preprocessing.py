"""Create tic/non-tic segment metadata for the VocalTics audio files."""

from pathlib import Path
import re
import wave

import pandas as pd


ROOT = Path("/projects/vocaltics/data/")
AUDIO_ROOT = ROOT / "DET_Audio_Data_16kHz_mono"
TIC_LIST_PATH = ROOT / "ticList.csv"
OUTPUT_PATH = ROOT / "metadata.csv"
KEEP_TYPES = [
    "1009", "1010", "1034", "1035", "1036", "1037", "1038", "1039",
    "1040", "1042", "1043", "1044", "1045", "1046", "1075", "1076",
    "1077", "1078", "1079", "1080", "1081", "1082", "1083", "1101",
    "1106", "1116", "1117", "1118", "1119", "1140", "1141", "1142",
    "1143", "1145", "1146", "1186", "1187",
    "1188", "1189", "1191", "1192", "1199", "1200", "1226", "1237",
    "1272", "1274", "1314", "1315", "1316", "1336", "1337", "1338",
    "1339", "1363", "1364", "1365",
    "1367", "1368", "1369", "1370", "1371", "1372", "1377", "1398",
    "1399", "1401", "1403", "1406", "1407", "1416", "1417", "1418",
    "1419", "1420",
]

AUDIO_NAME = re.compile(
    r"^(?P<ID>DET01(?:0[1-9]|1[0-9]|2[0-2]))_V"
    r"(?P<Sess>10|[1-9])_(?P<Phase>HI|LO|NO)\.wav$",
    re.IGNORECASE,
)
OUTPUT_COLUMNS = [
    "AudioPath",
    "ID",
    "Sess",
    "Phase",
    "tic/nontic",
    "Type",
    "StartTime",
    "EndTime",
    "Duration",
]


def is_kept_type(tic_type):
    return str(tic_type) in KEEP_TYPES


def wav_duration(path):
    """Return a WAV file's duration in seconds."""
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def load_audio_files():
    """Return one row for every valid audio file found under AUDIO_ROOT."""
    rows = []
    for path in sorted(AUDIO_ROOT.glob("DET01[0-2][0-9]/*.wav")):
        match = AUDIO_NAME.match(path.name)
        if match is None or path.parent.name.upper() != match["ID"].upper():
            continue

        rows.append(
            {
                "AudioPath": str(path),
                "ID": match["ID"].upper(),
                "Sess": int(match["Sess"]),
                "Phase": match["Phase"].upper(),
                "Duration": wav_duration(path),
            }
        )

    return pd.DataFrame(
        rows, columns=["AudioPath", "ID", "Sess", "Phase", "Duration"]
    )


def merged_tic_intervals(tics, audio_duration):
    """Clip and merge tics, retaining the non-removed types in each group."""
    intervals = []
    for start, end, tic_type in tics[
        ["StartTime", "EndTime", "Type"]
    ].itertuples(index=False):
        start = max(0.0, float(start))
        end = min(audio_duration, float(end))
        if start < end:
            labels = [str(tic_type)] if is_kept_type(tic_type) else []
            intervals.append((start, end, labels))

    intervals.sort(key=lambda interval: (interval[0], interval[1]))
    merged = []
    for start, end, labels in intervals:
        if merged and start <= merged[-1][1]:
            old_start, old_end, old_labels = merged[-1]
            for label in labels:
                if label not in old_labels:
                    old_labels.append(label)
            merged[-1] = (old_start, max(old_end, end), old_labels)
        else:
            merged.append((start, end, labels))
    return merged


def build_segments(audio_files, tic_list):
    """Split each audio file into consecutive tic and non-tic segments."""
    tic_groups = {
        (participant, int(session), phase): group
        for (participant, session, phase), group in tic_list.groupby(
            ["ID", "Sess", "Phase"]
        )
    }
    rows = []

    for audio in audio_files.itertuples(index=False):
        key = (audio.ID, audio.Sess, audio.Phase)
        tics = tic_groups.get(key, tic_list.iloc[0:0])
        intervals = merged_tic_intervals(tics, audio.Duration)
        position = 0.0

        for start, end, labels in intervals:
            if position < start:
                rows.append(segment_row(audio, "nontic", position, start))
            if labels:
                rows.append(segment_row(audio, "tic", start, end, "+".join(labels)))
            position = end

        if position < audio.Duration:
            rows.append(segment_row(audio, "nontic", position, audio.Duration))

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def segment_row(audio, label, start, end, tic_type="-1"):
    """Create one output metadata row."""
    return {
        "AudioPath": audio.AudioPath,
        "ID": audio.ID,
        "Sess": audio.Sess,
        "Phase": audio.Phase,
        "tic/nontic": label,
        "Type": tic_type,
        "StartTime": start,
        "EndTime": end,
        "Duration": end - start,
    }


def print_summary(metadata, audio_files):
    """Print summary statistics for the retained, merged tic segments."""
    tic_segments = metadata.loc[metadata["tic/nontic"] == "tic"]
    tic_count = len(tic_segments)
    participant_count = audio_files["ID"].nunique()
    average_per_participant = tic_count / participant_count if participant_count else 0
    average_duration = tic_segments["Duration"].mean() if tic_count else 0
    tic_minutes = tic_segments["Duration"].sum() / 60

    print("\nTic summary")
    print(f"Average tics per participant: {average_per_participant:.2f}")
    print(f"Average tic duration: {average_duration:.3f} seconds")
    print(f"Total tics: {tic_count}")
    print(f"Total tic duration: {tic_minutes:.2f} minutes")
    type_counts = tic_segments["Type"].str.count(r"\+") + 1
    print(f"Segments with 2 types: {(type_counts == 2).sum()}")
    print(f"Segments with 3 types: {(type_counts == 3).sum()}")
    print("Tics per category:")

    category_counts = (
        tic_segments["Type"]
        .str.split("+", regex=False)
        .explode()
        .value_counts()
        .sort_index()
    )
    if category_counts.empty:
        print("  None")
    else:
        for tic_type, count in category_counts.items():
            print(f"  {tic_type}: {count}")


def main():
    # Load tic annotations while preserving tic IDs as strings.
    tic_list = pd.read_csv(TIC_LIST_PATH, dtype={"Type": str})
    # Standardize participant IDs to uppercase for reliable matching.
    tic_list["ID"] = tic_list["ID"].astype(str).str.upper()
    # Standardize phase labels to uppercase for reliable matching.
    tic_list["Phase"] = tic_list["Phase"].astype(str).str.upper()

    # Find the available WAV files and measure their durations.
    audio_files = load_audio_files()
    # Split each recording into retained tic and non-tic segments.
    metadata = build_segments(audio_files, tic_list)
    # Save the generated segment metadata without a CSV index column.
    metadata.to_csv(OUTPUT_PATH, index=False)
    # Report where the metadata was saved and how many rows it contains.
    print(f"Saved {len(metadata)} segments to {OUTPUT_PATH}")
    # Print summary statistics for the retained tic segments.
    print_summary(metadata, audio_files)


if __name__ == "__main__":
    main()
