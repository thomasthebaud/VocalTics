"""Add YGTSS category groups to the segment metadata."""

from pathlib import Path

import pandas as pd


ROOT = Path("/projects/vocaltics/data/")
METADATA_PATH = ROOT / "metadata.csv"

TIC_GROUPS = {
    "1009": "Grunting",
    "1010": "Throat Clearing",
    "1034": "Coughing",
    "1035": "Throat Clearing",
    "1036": "Atypical Breathing",
    "1037": "Syllables",
    "1038": "Syllables",
    "1039": "Words",
    "1040": "Phrases",
    "1042": "Echolalia",
    "1043": "Coughing",
    "1044": "Blocking",
    "1045": "Blocking",
    "1046": "Sniffing",
    "1075": "Coughing",
    "1076": "Throat Clearing",
    "1077": "Sniffing",
    "1078": "Other Animal Noises",
    "1079": "Mouth Movements",
    "1080": "Mouth Movements",
    "1081": "Mouth Movements",
    "1082": "Atypical Breathing",
    "1083": "Syllables",
    "1101": "Syllables",
    "1106": "Mouth Noises",
    "1116": "Coughing",
    "1117": "Grunting",
    "1118": "Other Animal Noises",
    "1119": "Atypical Breathing",
    "1140": "Sniffing",
    "1141": "Sniffing",
    "1142": "Grunting",
    "1143": "Atypical Breathing",
    "1145": "Mouth Noises",
    "1146": "Blocking",
    "1186": "Snorting",
    "1187": "Mouth Noises",
    "1188": "Atypical Breathing",
    "1189": "Mouth Noises",
    "1191": "Mouth Movements",
    "1192": "Mouth Movements",
    "1199": "Mouth Noises",
    "1200": "Mouth Noises",
    "1226": "Nose Movements",
    "1237": "Mouth Noises",
    "1272": "Nose Movements",
    "1274": "Mouth Noises",
    "1314": "Mouth Noises",
    "1315": "Mouth Noises",
    "1316": "Mouth Noises",
    "1336": "Mouth Noises",
    "1337": "Mouth Noises",
    "1338": "Mouth Noises",
    "1339": "Mouth Noises",
    "1363": "Coughing",
    "1364": "Nose Movements",
    "1365": "Nose Movements",
    "1367": "Atypical Breathing",
    "1368": "Atypical Breathing",
    "1369": "Mouth Movements",
    "1370": "Mouth Movements",
    "1371": "Mouth Movements",
    "1372": "Mouth Movements",
    "1377": "Atypical Breathing",
    "1398": "Mouth Movements",
    "1399": "Mouth Movements",
    "1401": "Mouth Movements",
    "1403": "Mouth Movements",
    "1406": "Mouth Movements",
    "1407": "Mouth Movements",
    "1416": "Mouth Movements",
    "1417": "Mouth Movements",
    "1418": "Atypical Breathing",
    "1419": "Words",
    "1420": "Words",
}


def group_tic_types(value):
    """Return the YGTSS group for one metadata Type value."""
    if pd.isna(value):
        return "-1"
    if str(value) == "-1":
        return "-1"

    groups = []
    for tic_type in str(value).split("+"):
        if tic_type not in TIC_GROUPS:
            raise ValueError(f"No group defined for tic type {tic_type}")
        group = TIC_GROUPS[tic_type]
        if group not in groups:
            groups.append(group)
    return "+".join(sorted(groups))


def print_group_summary(metadata):
    """Print summaries for individual groups and group combinations."""
    tics = metadata.loc[metadata["Group"] != "-1"]

    individual_groups = tics.assign(
        Group=tics["Group"].str.split("+", regex=False)
    ).explode("Group")
    individual_summary = (
        individual_groups.groupby("Group")
        .agg(
            Tics=("Group", "size"),
            Minutes=("Duration", lambda duration: duration.sum() / 60),
            Participants=("ID", "nunique"),
        )
        .sort_index()
    )
    combinations = tics.loc[tics["Group"].str.contains("+", regex=False)]
    combination_summary = (
        combinations.groupby("Group")
        .agg(
            Tics=("Group", "size"),
            Minutes=("Duration", lambda duration: duration.sum() / 60),
            Participants=("ID", "nunique"),
        )
        .sort_index()
    )

    print("\nTics by individual group")
    if individual_summary.empty:
        print("No tics found.")
    else:
        print(
            individual_summary.to_string(
                formatters={"Minutes": "{:.2f}".format}
            )
        )

    print("\nTics by group combination")
    if combination_summary.empty:
        print("No group combinations found.")
    else:
        print(
            combination_summary.to_string(
                formatters={"Minutes": "{:.2f}".format}
            )
        )


def main():
    # Load the segment metadata while preserving tic IDs as strings.
    metadata = pd.read_csv(METADATA_PATH, dtype={"Type": str})
    # Assign the special type -1 to every non-tic segment.
    metadata.loc[metadata["tic/nontic"] == "nontic", "Type"] = "-1"
    # Map each tic type, or combination of types, to its group.
    metadata["Group"] = metadata["Type"].apply(group_tic_types)
    # Overwrite the metadata file with the new Group column included.
    metadata.to_csv(METADATA_PATH, index=False)
    # Report how many metadata rows were updated and where they were saved.
    print(f"Added Group to {len(metadata)} segments in {METADATA_PATH}")
    # Print tic counts, durations, and participant counts for each group.
    print_group_summary(metadata)


if __name__ == "__main__":
    main()
