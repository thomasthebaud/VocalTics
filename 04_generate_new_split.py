"""Generate and summarize cross-validation recording splits."""

import argparse
from pathlib import Path

import pandas as pd

from bin.make_splits import (
    save_split,
    splits_by_file,
    splits_by_participant,
    splits_by_session,
)


METADATA_PATH = Path("/projects/vocaltics/data/metadata.csv")
SPLIT_PATH = Path("splits.json")
K_FOLDS = 5
SEED = 42

SPLIT_FUNCTIONS = {
    "participant": splits_by_participant,
    "session": splits_by_session,
    "file": splits_by_file,
}


def parse_args():
    """Read split-generation arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and save cross-validation splits."
    )
    parser.add_argument(
        "--split-by",
        "--split_by",
        choices=SPLIT_FUNCTIONS,
        default="session",
        help="Cross-validation split unit (default: session)",
    )
    parser.add_argument(
        "--k-folds",
        "--k_folds",
        type=int,
        default=K_FOLDS,
        help=f"Number of folds (default: {K_FOLDS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    return parser.parse_args()


def print_summary(splits):
    """Print recording and participant contents for every fold split."""
    rows = []
    for fold_number, fold in sorted(splits.items()):
        for split_name in ("train", "val", "test"):
            recordings = fold[split_name]
            participants = sorted({participant for participant, _, _ in recordings})
            participant_sessions = {
                (participant, session) for participant, _, session in recordings
            }
            phases = sorted({phase for _, phase, _ in recordings})
            rows.append(
                {
                    "Fold": fold_number,
                    "Split": split_name,
                    "Recordings": len(recordings),
                    "Participants": len(participants),
                    "Participant sessions": len(participant_sessions),
                    "Phases": ", ".join(phases),
                    "Participant IDs": ", ".join(participants),
                }
            )
    print("\nSplit summary")
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    args = parse_args()
    metadata = pd.read_csv(METADATA_PATH)
    splits = SPLIT_FUNCTIONS[args.split_by](
        metadata, K=args.k_folds, seed=args.seed
    )
    save_split(splits, SPLIT_PATH)
    print(
        f"Saved {args.k_folds} {args.split_by}-based folds to {SPLIT_PATH} "
        f"with seed {args.seed}"
    )
    print_summary(splits)


if __name__ == "__main__":
    main()
